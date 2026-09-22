//! RenikudPlus Hebrew grapheme-to-phoneme and diacritization via ONNX.
//!
//! A Rust port of the upstream `renikud-plus` Python package (0.6.0), running
//! the same `model.onnx` weights and the same decode around them:
//!
//! - **Exact-MAP cascade decode** (default). Consonant, vowel and stress are
//!   decoded jointly under `E(c,v,s) = log P(c) + log P(v|c) + log P(s|c,v)`,
//!   with per-letter legality and "stress needs a vowel" as hard constraints,
//!   so the one-stress-per-word choice can flip the vowel and consonant too.
//!   [`Options::exact_map`] `= Some(false)` restores the greedy argmax.
//! - **Pointed input**, either ignored ([`NiqqudMode::Strip`], the upstream
//!   default) or read as evidence ([`NiqqudMode::Use`], what MamboTTS uses):
//!   every vowel sign, dagesh qal and shin/sin dot is pinned into the exact-MAP
//!   energy as a hard constraint.
//! - **Niqqud output** — [`G2P::vocalize`] renders the same predictions as
//!   pointed Hebrew instead of IPA.
//! - **Force lexicon**, **Hebrew number front end**, **long-input windowing**
//!   and the **raw-Hebrew backstop**, as upstream.
//!
//! One documented extension: niqqud has no stress mark, so
//! [`Options::with_stress`] writes the predicted stress with the hatama
//! (U+05AB), and [`NiqqudMode::Use`] reads that mark back as a stress
//! constraint. Both are off in upstream and can be turned off here
//! ([`G2PConfig::read_hatama`]).

pub mod decode;
pub mod lexicon;
pub mod niqqud;
pub mod numbers;
mod pychars;
pub mod text;

use std::collections::{HashMap, HashSet};

use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

use decode::Cascade;
use lexicon::{AlignCell, ForceLexicon, Forced, apply_force_lexicon};
use niqqud::{Constraints, HATAMA, NONE, consonant_point, niqqud_vowel, parse_niqqud};
use text::{
    HebrewLeak, SAFE_CHUNK_CHARS, check_no_raw_hebrew, is_hebrew, normalize_graphemes,
    split_for_decode, strip_niqqud,
};

const STRESS: &str = "ˈ";
const ORTHOGRAPHIC_MARKERS: [char; 2] = ['\'', '"'];

/// What to do with points in the INPUT.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NiqqudMode {
    /// Drop them and predict from the consonantal skeleton (upstream default).
    #[default]
    Strip,
    /// Read them: the points become hard constraints on the decode.
    Use,
}

/// Whether to expand digits into Hebrew number words before decoding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NumberNorm {
    /// Expand when the text contains a digit (upstream default).
    #[default]
    Auto,
    /// Leave digits alone.
    Off,
}

/// Session-level settings, fixed when the model is loaded.
#[derive(Clone, Debug)]
pub struct G2PConfig {
    /// Use the exact-MAP cascade decode when the model carries the metadata.
    pub exact_map: bool,
    /// The long-input decode window; `None` disables windowing.
    pub chunk_chars: Option<usize>,
    /// What to do with points in the input.
    pub niqqud: NiqqudMode,
    /// Read the hatama (U+05AB) as a stress constraint in [`NiqqudMode::Use`].
    ///
    /// A MamboTTS extension: it is what makes `vocalize(with_stress)` output
    /// round-trip through `phonemize` with its stress intact.
    pub read_hatama: bool,
}

impl Default for G2PConfig {
    fn default() -> Self {
        Self {
            exact_map: true,
            chunk_chars: Some(SAFE_CHUNK_CHARS),
            niqqud: NiqqudMode::Strip,
            read_hatama: true,
        }
    }
}

/// Per-call settings.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Speaker conditioning: 0 unknown, 1 male, 2 female. Ignored by models
    /// exported without the gender inputs.
    pub speaker: u8,
    /// Target speaker conditioning, same encoding.
    pub target_speaker: u8,
    /// Override the configured decode for this call.
    pub exact_map: Option<bool>,
    /// What to do if raw Hebrew survives into the IPA (`phonemize` only).
    pub on_hebrew_leak: HebrewLeak,
    /// Whether to expand digits first.
    pub number_norm: NumberNorm,
    /// Override the configured niqqud mode for this call.
    pub niqqud: Option<NiqqudMode>,
    /// Mark the predicted stress with the hatama (`vocalize` only).
    pub with_stress: bool,
    /// Point the three letters whose reading pointing alone cannot pin down, so
    /// the output reads back as the same word (`vocalize` only).
    ///
    /// A rafe on a ו the decode reads as silent, a shva on a י it reads as the
    /// glide /j/, and a mapiq in a word-final ה it reads as /h/. All three are
    /// ordinary Hebrew marks for exactly that, but upstream writes none of
    /// them, so this is off by default. On the parity corpus it takes the
    /// share of texts that phonemize identically after a pointing round trip
    /// from 274/327 to 322/327.
    pub round_trip: bool,
}

impl Options {
    /// Options for one speaker pair, everything else at its default.
    pub fn speakers(speaker: u8, target_speaker: u8) -> Self {
        Self {
            speaker,
            target_speaker,
            ..Self::default()
        }
    }
}

/// What one session run over a window produced: where each token sits in the
/// window, the labels the decode picked, and the consonant logits the
/// per-letter fallback needs.
struct Read {
    offsets: Vec<(usize, usize)>,
    decoded: decode::Decoded,
    consonant_logits: Vec<Vec<f32>>,
}

/// One Hebrew letter as the decode read it.
#[derive(Clone, Debug)]
struct Record {
    letter: char,
    consonant: String,
    vowel: String,
    stressed: bool,
    /// Index of the placeholder this letter owns in the output pieces.
    out_index: usize,
    /// Character index of the letter in the window.
    start: usize,
}

pub struct G2P {
    session: Session,
    vocab: HashMap<char, i64>,
    unk_id: i64,
    consonant_vocab: HashMap<usize, String>,
    vowel_vocab: HashMap<usize, String>,
    consonant_ids: HashMap<String, usize>,
    vowel_ids: HashMap<String, usize>,
    letter_constraints: HashMap<char, Vec<usize>>,
    geresh_map: HashMap<char, String>,
    cls_id: i64,
    sep_id: i64,
    gender_conditioned: bool,
    cascade: Option<Cascade>,
    config: G2PConfig,
    lexicon: Option<ForceLexicon>,
    /// The words the lexicon overrode in the most recent decoded window.
    pub last_forced: Vec<Forced>,
}

impl G2P {
    /// Load a model with the upstream defaults.
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        Self::with_config(model_path, G2PConfig::default())
    }

    pub fn with_config(model_path: &str, config: G2PConfig) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Self::from_session_with_config(session, config)
    }

    pub fn from_session(session: Session) -> anyhow::Result<Self> {
        Self::from_session_with_config(session, G2PConfig::default())
    }

    pub fn from_session_with_config(session: Session, config: G2PConfig) -> anyhow::Result<Self> {
        let gender_conditioned = {
            let names: HashSet<&str> = session.inputs().iter().map(|i| i.name()).collect();
            names.contains("speaker") && names.contains("target_speaker")
        };

        let meta = session.metadata()?;
        let custom = |key: &str| meta.custom(key);
        let required = |key: &str| {
            meta.custom(key)
                .ok_or_else(|| anyhow::anyhow!("model metadata is missing `{key}`"))
        };

        let raw_vocab: HashMap<String, i64> = serde_json::from_str(&required("vocab")?)?;
        let unk_id = raw_vocab.get("[UNK]").copied().unwrap_or(0);
        let vocab: HashMap<char, i64> = raw_vocab
            .iter()
            .filter(|(key, _)| key.chars().count() == 1)
            .map(|(key, &id)| (key.chars().next().expect("one char"), id))
            .collect();

        let consonant_vocab = parse_label_vocab(&required("consonant_vocab")?)?;
        let vowel_vocab = parse_label_vocab(&required("vowel_vocab")?)?;
        let consonant_ids = invert(&consonant_vocab);
        let vowel_ids = invert(&vowel_vocab);

        // RenikudPlus prefers letter_consonant_constraints; fall back to mask.
        let constraints_json = custom("letter_consonant_constraints")
            .or_else(|| custom("letter_consonant_mask"))
            .ok_or_else(|| anyhow::anyhow!("missing letter_consonant_constraints/mask"))?;
        let raw_constraints: HashMap<String, Vec<usize>> = serde_json::from_str(&constraints_json)?;
        let letter_constraints: HashMap<char, Vec<usize>> = raw_constraints
            .into_iter()
            .filter_map(|(key, ids)| key.chars().next().map(|c| (c, ids)))
            .collect();

        let raw_geresh: HashMap<String, String> =
            serde_json::from_str(&custom("geresh_map").unwrap_or_else(|| "{}".to_owned()))?;
        let geresh_map: HashMap<char, String> = raw_geresh
            .into_iter()
            .filter_map(|(key, value)| key.chars().next().map(|c| (c, value)))
            .collect();

        let cls_id: i64 = required("cls_token_id")?.parse()?;
        let sep_id: i64 = required("sep_token_id")?.parse()?;

        // Exact structured-cascade MAP needs the conditioning column blocks of
        // the cascade heads; models exported before that feature lack them and
        // fall back to greedy argmax + margin stress.
        let cascade = match (
            custom("vowel_cond_consonant"),
            custom("stress_cond_consonant"),
            custom("stress_cond_vowel"),
        ) {
            (Some(wv_c), Some(ws_c), Some(ws_v)) => {
                let wv_c: Vec<Vec<f32>> = serde_json::from_str(&wv_c)?;
                let ws_c: Vec<Vec<f32>> = serde_json::from_str(&ws_c)?;
                let ws_v: Vec<Vec<f32>> = serde_json::from_str(&ws_v)?;
                let n_cons = wv_c.len();
                let letters = text::TAF as usize - text::ALEF as usize + 1;
                let mut forbidden = vec![vec![true; n_cons]; letters];
                for (letter, allowed) in &letter_constraints {
                    if is_hebrew(*letter) {
                        let row = &mut forbidden[*letter as usize - text::ALEF as usize];
                        for &id in allowed {
                            if id < n_cons {
                                row[id] = false;
                            }
                        }
                    }
                }
                Some(Cascade {
                    wv_c,
                    ws_c,
                    ws_v,
                    cond_softmax: custom("cascade_cond").as_deref().unwrap_or("softmax")
                        == "softmax",
                    forbidden,
                })
            }
            _ => None,
        };
        drop(meta);

        Ok(Self {
            session,
            vocab,
            unk_id,
            consonant_vocab,
            vowel_vocab,
            consonant_ids,
            vowel_ids,
            letter_constraints,
            geresh_map,
            cls_id,
            sep_id,
            gender_conditioned,
            cascade,
            config,
            lexicon: None,
            last_forced: Vec::new(),
        })
    }

    /// True when the model takes `speaker` / `target_speaker` inputs.
    pub fn is_gender_conditioned(&self) -> bool {
        self.gender_conditioned
    }

    /// True when the model carries the cascade conditioning metadata.
    pub fn supports_exact_map(&self) -> bool {
        self.cascade.is_some()
    }

    pub fn config(&self) -> &G2PConfig {
        &self.config
    }

    pub fn set_config(&mut self, config: G2PConfig) {
        self.config = config;
    }

    /// Install (or clear) the force-only lexicon: where it matches, it wins.
    pub fn set_lexicon(&mut self, lexicon: Option<ForceLexicon>) {
        self.lexicon = lexicon;
    }

    /// Take the installed lexicon back out, leaving the decode unforced.
    pub fn take_lexicon(&mut self) -> Option<ForceLexicon> {
        self.lexicon.take()
    }

    /// The aligner cells a lexicon should be validated with. Kept beside the
    /// lexicon loader for callers that build one from a path.
    pub fn align_cells(spec: &str) -> anyhow::Result<Vec<AlignCell>> {
        lexicon::resolve_align_cells(spec)
    }

    /// Convert Hebrew text to IPA.
    ///
    /// `speaker` / `target_speaker`: 0 unknown, 1 male, 2 female.
    pub fn phonemize(
        &mut self,
        text: &str,
        speaker: u8,
        target_speaker: u8,
    ) -> anyhow::Result<String> {
        self.phonemize_with(text, &Options::speakers(speaker, target_speaker))
    }

    /// [`Self::phonemize`] with every upstream knob exposed.
    ///
    /// Inputs longer than the configured window are split on sentence, then
    /// comma, then word boundaries and decoded window by window; the windows
    /// are contiguous, so concatenation is lossless. A `[א-ת]`-in-output
    /// backstop then reports (or raises) on raw Hebrew that survives into the
    /// IPA string.
    pub fn phonemize_with(&mut self, text: &str, options: &Options) -> anyhow::Result<String> {
        let (normalized, constraints, original) = self.read_input(text, options)?;
        let mut result = String::new();
        for (window, window_constraints) in self.windows(&normalized, constraints.as_ref()) {
            result.push_str(&self.phonemize_window(
                &window,
                options,
                window_constraints.as_ref(),
            )?);
        }
        check_no_raw_hebrew(
            result,
            &original,
            options.on_hebrew_leak,
            "renikud_plus_rs::G2P::phonemize",
        )
    }

    /// Add niqqud to Hebrew `text`; non-Hebrew is unchanged.
    ///
    /// Renders the same per-letter (consonant, vowel) predictions as
    /// [`Self::phonemize`], but as niqqud — vowel signs plus dagesh for hard
    /// b/k/p and the shin/sin dot. Diacritization is phonetically faithful but
    /// not publication-grade (e.g. shva in clusters is omitted). The force
    /// lexicon rewrites IPA strings, so it does not apply here.
    pub fn vocalize(
        &mut self,
        text: &str,
        speaker: u8,
        target_speaker: u8,
    ) -> anyhow::Result<String> {
        self.vocalize_with(text, &Options::speakers(speaker, target_speaker))
    }

    /// [`Self::vocalize`] with every knob exposed, including
    /// [`Options::with_stress`].
    pub fn vocalize_with(&mut self, text: &str, options: &Options) -> anyhow::Result<String> {
        let (normalized, constraints, _) = self.read_input(text, options)?;
        let mut result = String::new();
        for (window, window_constraints) in self.windows(&normalized, constraints.as_ref()) {
            result.push_str(&self.vocalize_window(
                &window,
                options,
                window_constraints.as_ref(),
            )?);
        }
        Ok(result)
    }

    /// [`Self::vocalize`], keeping the name MamboTTS calls it by.
    ///
    /// `with_stress` marks the stressed letter of every word with the hatama
    /// (U+05AB), which [`Self::phonemize`] reads back in [`NiqqudMode::Use`].
    pub fn diacritize(
        &mut self,
        text: &str,
        speaker: u8,
        target_speaker: u8,
        with_stress: bool,
    ) -> anyhow::Result<String> {
        self.vocalize_with(
            text,
            &Options {
                with_stress,
                ..Options::speakers(speaker, target_speaker)
            },
        )
    }

    /// Number expansion, grapheme folding, NFD and the niqqud reading: the
    /// front end both output renderings share. Returns the bare text, the
    /// constraints its points force, and the text the leak backstop reports on.
    fn read_input(
        &self,
        text: &str,
        options: &Options,
    ) -> anyhow::Result<(String, Option<Constraints>, String)> {
        let mut text = text.to_owned();
        if options.number_norm == NumberNorm::Auto && numbers::contains_digit(&text) {
            // Digits never appear in training text, so a digit reaching the
            // model is guaranteed-wrong output (and mis-aligns the word after
            // it). Runs BEFORE windowing — the expander lengthens text.
            text = numbers::normalize_numbers(&text);
        }
        text = normalize_graphemes(&text);
        let normalized: String = text.nfd().collect();
        let mode = options.niqqud.unwrap_or(self.config.niqqud);
        Ok(match mode {
            NiqqudMode::Strip => (strip_niqqud(&normalized), None, text),
            NiqqudMode::Use => {
                let (bare, constraints) = parse_niqqud(&normalized, self.config.read_hatama);
                (bare, Some(constraints), text)
            }
        })
    }

    /// Pair each decode window with the constraints falling inside it.
    fn windows(
        &self,
        normalized: &str,
        constraints: Option<&Constraints>,
    ) -> Vec<(String, Option<Constraints>)> {
        let windows = match self.config.chunk_chars {
            Some(limit) => split_for_decode(normalized, limit, true),
            None => vec![normalized.to_owned()],
        };
        // `split_for_decode` returns contiguous substrings, so a running offset
        // is enough to re-base the whole-text indices onto each window.
        let mut offset = 0;
        windows
            .into_iter()
            .map(|window| {
                let length = window.chars().count();
                let window_constraints = constraints.map(|constraints| {
                    constraints
                        .iter()
                        .filter(|&(&index, _)| offset <= index && index < offset + length)
                        .map(|(&index, constraint)| (index - offset, constraint.clone()))
                        .collect::<Constraints>()
                });
                offset += length;
                (window, window_constraints)
            })
            .collect()
    }

    /// Tokenize character by character; returns ids, mask and offsets in
    /// characters.
    fn tokenize(&self, chars: &[char]) -> (Vec<i64>, Vec<i64>, Vec<(usize, usize)>) {
        let mut ids = vec![self.cls_id];
        let mut offsets = vec![(0usize, 0usize)]; // CLS
        for (i, c) in chars.iter().enumerate() {
            ids.push(self.vocab.get(c).copied().unwrap_or(self.unk_id));
            offsets.push((i, i + 1));
        }
        ids.push(self.sep_id);
        offsets.push((0, 0)); // SEP
        let mask = vec![1i64; ids.len()];
        (ids, mask, offsets)
    }

    /// One session run over an already normalized/NFD window.
    fn forward(
        &mut self,
        chars: &[char],
        options: &Options,
        constraints: Option<&Constraints>,
    ) -> anyhow::Result<Read> {
        if options.speaker > 2 || options.target_speaker > 2 {
            anyhow::bail!("speaker and target_speaker must be 0, 1, or 2");
        }
        let (ids, mask, offsets) = self.tokenize(chars);
        let len = ids.len();

        let input_ids = Tensor::<i64>::from_array(([1, len], ids.into_boxed_slice()))?;
        let attention_mask = Tensor::<i64>::from_array(([1, len], mask.into_boxed_slice()))?;
        let outputs = if self.gender_conditioned {
            let speaker = Tensor::<i64>::from_array((
                [1],
                vec![i64::from(options.speaker)].into_boxed_slice(),
            ))?;
            let target = Tensor::<i64>::from_array((
                [1],
                vec![i64::from(options.target_speaker)].into_boxed_slice(),
            ))?;
            self.session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "speaker" => speaker,
                "target_speaker" => target
            ])?
        } else {
            self.session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask
            ])?
        };

        let consonant_logits = rows(&outputs["consonant_logits"])?;
        let vowel_logits = rows(&outputs["vowel_logits"])?;
        let stress_logits = rows(&outputs["stress_logits"])?;
        drop(outputs);

        let use_exact = options
            .exact_map
            .unwrap_or(self.config.exact_map && self.cascade.is_some());
        let pinned = constraints.is_some_and(|constraints| !constraints.is_empty());
        if pinned && !use_exact {
            anyhow::bail!(
                "niqqud=\"use\" pins the input's points into the exact-MAP cascade energy; it \
                 cannot run with exact_map disabled"
            );
        }
        let heads = decode::Heads {
            consonant: &consonant_logits,
            vowel: &vowel_logits,
            stress: &stress_logits,
        };
        let decoded = match (&self.cascade, use_exact) {
            (Some(cascade), true) => decode::exact_map(
                &offsets,
                chars,
                &heads,
                cascade,
                constraints,
                &decode::Labels {
                    consonant_ids: &self.consonant_ids,
                    vowel_ids: &self.vowel_ids,
                },
            ),
            (None, true) => anyhow::bail!(
                "this model was exported without the cascade conditioning metadata; re-export \
                 it, or disable exact_map for the greedy decode"
            ),
            (_, false) => decode::greedy(&offsets, chars, &heads, &self.vowel_vocab),
        };
        Ok(Read {
            offsets,
            decoded,
            consonant_logits,
        })
    }

    /// The (consonant, vowel) the model chose for one Hebrew letter.
    ///
    /// Under the exact-MAP decode the per-letter legality constraint is already
    /// baked into the energy; the greedy path needs the fallback applied here.
    fn labels_at(
        &self,
        char_: char,
        tok_idx: usize,
        decoded: &decode::Decoded,
        consonant_logits: &[Vec<f32>],
    ) -> (String, String) {
        let mut cid = decoded.consonants[tok_idx];
        if let Some(allowed) = self.letter_constraints.get(&char_)
            && !allowed.contains(&cid)
            && let Some(&best) = allowed.iter().reduce(|best, candidate| {
                if consonant_logits[tok_idx][*candidate] > consonant_logits[tok_idx][*best] {
                    candidate
                } else {
                    best
                }
            })
        {
            cid = best;
        }
        let consonant = self
            .consonant_vocab
            .get(&cid)
            .cloned()
            .unwrap_or_else(|| NONE.to_owned());
        let vowel = self
            .vowel_vocab
            .get(&decoded.vowels[tok_idx])
            .cloned()
            .unwrap_or_else(|| NONE.to_owned());
        (consonant, vowel)
    }

    /// Decode one window to IPA.
    fn phonemize_window(
        &mut self,
        window: &str,
        options: &Options,
        constraints: Option<&Constraints>,
    ) -> anyhow::Result<String> {
        let chars: Vec<char> = window.chars().collect();
        let Read {
            offsets,
            decoded,
            consonant_logits,
        } = self.forward(&chars, options, constraints)?;

        let mut result = String::new();
        let mut prev_end = 0;
        for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
            if end - start != 1 {
                // CLS, SEP — skip
                if end > start {
                    prev_end = end;
                }
                continue;
            }
            // Pass through any characters skipped by the tokenizer.
            if start > prev_end {
                result.extend(&chars[prev_end..start]);
            }
            let char_ = chars[start];
            prev_end = end;

            if !is_hebrew(char_) {
                if !ORTHOGRAPHIC_MARKERS.contains(&char_) {
                    result.push(char_);
                }
                continue;
            }

            let (mut consonant, vowel) =
                self.labels_at(char_, tok_idx, &decoded, &consonant_logits);
            // Geresh rule: an apostrophe after the letter forces the geresh
            // consonant variant.
            if let Some(geresh) = self.geresh_map.get(&char_)
                && chars.get(end) == Some(&'\'')
            {
                consonant = geresh.clone();
            }
            let stress = decoded.stressed.contains(&tok_idx);

            // [consonant][ˈ][vowel], except a word-final ח with vowel a, where
            // the furtive patah flips to [ˈ]aχ.
            let word_final = end >= chars.len() || !pychars::is_alpha(chars[end]);
            if char_ == 'ח' && word_final && vowel == "a" {
                if stress {
                    result.push_str(STRESS);
                }
                result.push_str("aχ");
                continue;
            }
            if consonant != NONE {
                result.push_str(&consonant);
            }
            if vowel != NONE {
                if stress {
                    result.push_str(STRESS);
                }
                result.push_str(&vowel);
            }
        }
        if prev_end < chars.len() {
            result.extend(&chars[prev_end..]);
        }

        // Force-lexicon post-pass (a no-op unless a lexicon was installed). It
        // works on whitespace-delimited words, normalizing runs of inter-word
        // whitespace to single spaces; the window's own leading and trailing
        // whitespace is preserved so concatenating windows stays lossless.
        if let Some(lexicon) = self.lexicon.as_ref() {
            let body = pychars::strip(&result);
            let lead = &result[..result.len() - result.trim_start_matches(pychars::is_space).len()];
            let trail = &result[result.trim_end_matches(pychars::is_space).len()..];
            let (body, forced) = apply_force_lexicon(pychars::strip(window), body, lexicon);
            self.last_forced = forced;
            return Ok(format!("{lead}{body}{trail}"));
        }
        Ok(result)
    }

    /// Decode one window to pointed Hebrew.
    fn vocalize_window(
        &mut self,
        window: &str,
        options: &Options,
        constraints: Option<&Constraints>,
    ) -> anyhow::Result<String> {
        let chars: Vec<char> = window.chars().collect();
        let Read {
            offsets,
            decoded,
            consonant_logits,
        } = self.forward(&chars, options, constraints)?;

        let mut out: Vec<String> = Vec::new();
        let mut records: Vec<Record> = Vec::new();
        let mut prev_end = 0;
        for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
            if end - start != 1 {
                if end > start {
                    prev_end = end;
                }
                continue;
            }
            if start > prev_end {
                out.push(chars[prev_end..start].iter().collect());
            }
            let char_ = chars[start];
            prev_end = end;
            if !is_hebrew(char_) {
                // Keep everything non-Hebrew, including geresh markers (ג׳ -> ג').
                out.push(char_.to_string());
                continue;
            }
            let (consonant, vowel) = self.labels_at(char_, tok_idx, &decoded, &consonant_logits);
            records.push(Record {
                letter: char_,
                consonant,
                vowel,
                stressed: decoded.stressed.contains(&tok_idx),
                out_index: out.len(),
                start,
            });
            out.push(String::new()); // filled after the mater-lectionis fixup
        }
        if prev_end < chars.len() {
            out.push(chars[prev_end..].iter().collect());
        }

        // Mater lectionis fixup — a vowel-letter (ו/י/ה/א) is silent but the
        // vowel it represents belongs on a neighbouring letter. Both cases are
        // guarded to adjacent letters in the same word.
        //  - Silent final ה/א (consonant ∅) took the preceding consonant's
        //    vowel; shift it back so the mark never lands on a silent letter.
        //  - A silent ו should itself carry an adjacent /o/ or /u/, so it
        //    renders as holam male (וֹ) or shuruk (וּ). י as a mater already
        //    renders correctly (hiriq male, ִי), so it needs no fixup.
        for i in 1..records.len() {
            let (previous, record) = records.split_at_mut(i);
            let previous = previous.last_mut().expect("i >= 1");
            let record = &mut record[0];
            let adjacent = previous.start + 1 == record.start;
            if matches!(record.letter, 'ה' | 'א')
                && record.consonant == NONE
                && record.vowel != NONE
            {
                if previous.vowel == NONE && adjacent {
                    previous.vowel = std::mem::replace(&mut record.vowel, NONE.to_owned());
                    previous.stressed |= std::mem::take(&mut record.stressed);
                }
            } else if record.letter == 'ו'
                && record.consonant == NONE
                && record.vowel == NONE
                && adjacent
                && matches!(previous.vowel.as_str(), "o" | "u")
            {
                record.vowel = std::mem::replace(&mut previous.vowel, NONE.to_owned());
                record.stressed |= std::mem::take(&mut previous.stressed);
            }
        }

        for (i, record) in records.iter().enumerate() {
            let stress_mark = if options.with_stress && record.stressed && record.vowel != NONE {
                Some(HATAMA)
            } else {
                None
            };
            // What the points alone cannot pin down, marked only when the caller
            // asks for a pointing that reads back as the same word (a MamboTTS
            // extension; upstream writes none of these).
            //   - A bare ו inside a pointed word is the consonant /v/ by the
            //     rules of pointed text, so a ו the decode reads as silent needs
            //     its rafe to stay silent.
            //   - A word-final ה is silent unless it carries the mapiq, so a ה
            //     the decode reads as /h/ needs one.
            //   - A bare י after hiriq or tsere is that vowel's mater, so a י
            //     the decode reads as the glide /j/ (the second half of ej, aj)
            //     needs the shva that says it closes the syllable instead.
            //
            // Only those three: pointing every vowel-less consonant with its
            // shva, which upstream also leaves out, measures worse rather than
            // better (see plans/renikud-parity), because a shva admits /e/ as
            // well as nothing.
            let round_trip_mark = options.round_trip.then(|| {
                let word_final = records.get(i + 1).is_none_or(|next| {
                    chars[record.start + 1..next.start]
                        .iter()
                        .any(|&c| pychars::is_space(c))
                });
                match (
                    record.letter,
                    record.consonant.as_str(),
                    record.vowel.as_str(),
                ) {
                    ('ו', NONE, NONE) => Some(niqqud::RAFE),
                    ('ה', "h", _) if word_final => Some(niqqud::DAGESH),
                    ('י', consonant, NONE) if consonant != NONE => Some(niqqud::SHEVA),
                    _ => None,
                }
            });
            let mut piece = String::new();
            piece.push(record.letter);
            if record.letter == 'ו'
                && record.consonant == NONE
                && matches!(record.vowel.as_str(), "o" | "u")
            {
                // Vav as a vowel letter: shuruk (וּ) for /u/, holam male (וֹ) for /o/.
                piece.push(if record.vowel == "u" {
                    niqqud::DAGESH
                } else {
                    niqqud::HOLAM
                });
            } else {
                if let Some(point) = consonant_point(record.letter, &record.consonant) {
                    piece.push(point);
                }
                piece.extend(round_trip_mark.flatten());
                if let Some(point) = niqqud_vowel(&record.vowel) {
                    piece.push(point);
                }
            }
            piece.extend(stress_mark);
            out[record.out_index] = piece.nfc().collect();
        }

        Ok(out.concat())
    }
}

fn parse_label_vocab(json: &str) -> anyhow::Result<HashMap<usize, String>> {
    let raw: HashMap<String, String> = serde_json::from_str(json)?;
    Ok(raw
        .into_iter()
        .filter_map(|(key, value)| key.parse::<usize>().ok().map(|id| (id, value)))
        .collect())
}

fn invert(vocab: &HashMap<usize, String>) -> HashMap<String, usize> {
    vocab
        .iter()
        .map(|(&id, label)| (label.clone(), id))
        .collect()
}

/// `[1, seq_len, classes]` logits as one row per token.
fn rows(value: &ort::value::DynValue) -> anyhow::Result<Vec<Vec<f32>>> {
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    let classes = *shape.last().expect("logits have a class axis") as usize;
    Ok(data.chunks(classes).map(<[f32]>::to_vec).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_the_upstream_defaults() {
        let config = G2PConfig::default();
        assert!(config.exact_map);
        assert_eq!(config.niqqud, NiqqudMode::Strip);
        assert_eq!(config.chunk_chars, Some(2046));
        let options = Options::default();
        assert_eq!(options.number_norm, NumberNorm::Auto);
        assert_eq!(options.on_hebrew_leak, HebrewLeak::Warn);
        assert!(!options.with_stress);
    }

    /// Needs `MAMBOTTS_RENIKUD_PATH` pointing at `renikud-plus.onnx`.
    #[test]
    #[ignore = "requires MAMBOTTS_RENIKUD_PATH pointing to renikud-plus.onnx"]
    fn the_readme_examples_decode_as_documented() {
        let model = std::env::var("MAMBOTTS_RENIKUD_PATH").expect("set MAMBOTTS_RENIKUD_PATH");
        let mut g2p = G2P::new(&model).unwrap();
        assert_eq!(
            g2p.phonemize("שלום לכולם", 0, 0).unwrap(),
            "ʃalˈom lekulˈam"
        );
        assert_eq!(g2p.vocalize("שלום לכולם", 0, 0).unwrap(), "שַׁלוֹם לֶכּוּלַם");
        // Pointed input is ignored by default...
        assert_eq!(g2p.phonemize("שָׁלוֹם לְכֻּלָּם", 0, 0).unwrap(), "ʃalˈom lekulˈam");
        // ...and read as evidence in "use" mode: the four ספר pointings.
        let used = Options {
            niqqud: Some(NiqqudMode::Use),
            ..Options::default()
        };
        for (pointed, expected) in [
            ("סֵפֶר", "sˈefeʁ"),
            ("סַפָּר", "sˈapaʁ"),
            ("סָפַר", "sˈafaʁ"),
            ("סִפֵּר", "sˈipeʁ"),
        ] {
            assert_eq!(
                g2p.phonemize_with(pointed, &used).unwrap(),
                expected,
                "{pointed}"
            );
        }
    }

    /// Needs `MAMBOTTS_RENIKUD_PATH` pointing at `renikud-plus.onnx`.
    #[test]
    #[ignore = "requires MAMBOTTS_RENIKUD_PATH pointing to renikud-plus.onnx"]
    fn stressed_niqqud_round_trips_through_phonemize() {
        let model = std::env::var("MAMBOTTS_RENIKUD_PATH").expect("set MAMBOTTS_RENIKUD_PATH");
        let mut g2p = G2P::new(&model).unwrap();
        let used = Options {
            niqqud: Some(NiqqudMode::Use),
            ..Options::default()
        };
        let stressed = Options {
            with_stress: true,
            round_trip: true,
            ..used
        };
        for text in [
            "שלום עולם, מה שלומך היום?",
            "הילדים הלכו לבית הספר בבוקר.",
            "שבת שלום",
            "רוח ותקווה",
            "אני אוהב machine learning וגם ג׳אז.",
        ] {
            let ipa = g2p.phonemize_with(text, &used).unwrap();
            let pointed = g2p.vocalize_with(text, &stressed).unwrap();
            let round_trip = g2p.phonemize_with(&pointed, &used).unwrap();
            // With the round-trip marks the pointing carries the whole reading,
            // stress included, so the IPA comes back unchanged.
            assert_eq!(round_trip, ipa, "{text} -> {pointed}");
        }
        // Without them, the stress still survives (that is what the hatama is
        // for), even where the reading drifts.
        let plain_stress = Options {
            with_stress: true,
            ..used
        };
        for text in ["שלום עולם", "הילדים הלכו לבית הספר בבוקר."] {
            let ipa = g2p.phonemize_with(text, &used).unwrap();
            let pointed = g2p.vocalize_with(text, &plain_stress).unwrap();
            let round_trip = g2p.phonemize_with(&pointed, &used).unwrap();
            assert_eq!(
                round_trip.matches(STRESS).count(),
                ipa.matches(STRESS).count(),
                "{text} -> {pointed}: {ipa} vs {round_trip}"
            );
        }
    }
}
