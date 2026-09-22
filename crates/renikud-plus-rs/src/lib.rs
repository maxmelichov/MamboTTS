//! RenikudPlus Hebrew grapheme-to-phoneme and diacritization via ONNX.
//!
//! Supports legacy models (`input_ids` + `attention_mask` only) and gender-conditioned
//! RenikudPlus models that also require `speaker` / `target_speaker`
//! (0 = unknown, 1 = male, 2 = female).
//!
//! One inference pass predicts a consonant, a vowel and a stress score for every
//! Hebrew letter. [`G2P::phonemize`] renders that reading as IPA and
//! [`G2P::diacritize`] renders the same reading as niqqud on the original letters,
//! so `phonemize(diacritize(text))` reads the same as `phonemize(text)`.

use std::collections::{HashMap, HashSet};

use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

const ALEF: u32 = 0x05D0;
const TAF: u32 = 0x05EA;
const STRESS: &str = "ˈ";
const NONE: &str = "∅";

const SHVA: char = '\u{05B0}';
const TSERE: char = '\u{05B5}';
const SEGOL: char = '\u{05B6}';
const PATAH: char = '\u{05B7}';
const QAMATS: char = '\u{05B8}';
const HIRIQ: char = '\u{05B4}';
const HOLAM: char = '\u{05B9}';
const HOLAM_HASER_FOR_VAV: char = '\u{05BA}';
const QUBUTS: char = '\u{05BB}';
const DAGESH: char = '\u{05BC}';
const RAFE: char = '\u{05BF}';
const SHIN_DOT: char = '\u{05C1}';
const SIN_DOT: char = '\u{05C2}';
const QAMATS_QATAN: char = '\u{05C7}';
/// Hebrew accent "ole", the stress mark Phonikud-style vocalized text uses.
const HATAMA: char = '\u{05AB}';

fn is_hebrew(c: char) -> bool {
    let cp = c as u32;
    (ALEF..=TAF).contains(&cp)
}

fn normalize_grapheme(c: char) -> char {
    match c {
        '\u{05F3}' | '\'' | '`' | '\u{00B4}' => '\'',
        '\u{05F4}' | '\u{201C}' | '\u{201D}' => '"',
        _ => c,
    }
}

fn is_mark(c: char) -> bool {
    matches!(c, '\u{0591}'..='\u{05bd}' | '\u{05bf}' | '\u{05c1}'..='\u{05c2}' | '\u{05c4}'..='\u{05c5}' | '\u{05c7}')
}

/// Plain text as the model reads it, split from the writer's marks.
struct Separated {
    /// NFD text without Hebrew marks, geresh and quotes folded to ASCII.
    plain: String,
    /// Marks keyed by the byte offset of the Hebrew letter they sit on.
    marks: HashMap<usize, String>,
    /// Characters that grapheme folding changed, keyed by plain byte offset,
    /// so diacritized output can give the writer's own characters back.
    restore: HashMap<usize, char>,
}

// Keep marks aligned to plain-text byte offsets: the model sees its usual input,
// while decoding can honor the user's vowels and consonant marks.
fn separate_nikud(text: &str) -> Separated {
    let mut plain = String::new();
    let mut marks = HashMap::<usize, String>::new();
    let mut restore = HashMap::new();
    let mut letter = None;
    for original in text.nfd() {
        if is_mark(original) {
            if let Some(offset) = letter {
                marks.entry(offset).or_default().push(original);
            }
            continue;
        }
        let c = normalize_grapheme(original);
        if c != original {
            restore.insert(plain.len(), original);
        }
        letter = is_hebrew(c).then_some(plain.len());
        plain.push(c);
    }
    Separated {
        plain,
        marks,
        restore,
    }
}

fn marked_vowel(marks: &str) -> Option<&'static str> {
    marks.chars().find_map(|c| match c {
        'ְ' => Some("∅"),
        'ֱ' | 'ֵ' | 'ֶ' => Some("e"),
        'ֲ' | 'ַ' | 'ָ' => Some("a"),
        'ֳ' | 'ֹ' | 'ֺ' | 'ׇ' => Some("o"),
        'ִ' => Some("i"),
        'ֻ' => Some("u"),
        _ => None,
    })
}

fn marked_consonant(c: char, marks: &str) -> Option<&'static str> {
    match c {
        'ב' if marks.contains('ּ') => Some("b"),
        'כ' | 'ך' if marks.contains('ּ') => Some("k"),
        'פ' | 'ף' if marks.contains('ּ') => Some("p"),
        'ב' if marks.contains('ֿ') => Some("v"),
        'כ' | 'ך' if marks.contains('ֿ') => Some("χ"),
        'פ' | 'ף' if marks.contains('ֿ') => Some("f"),
        'ש' if marks.contains('ׁ') => Some("ʃ"),
        'ש' if marks.contains('ׂ') => Some("s"),
        'ו' if marks.contains('ּ') && marked_vowel(marks).is_none() => Some("∅"),
        'ו' if marks.contains('ֹ') => Some("∅"),
        'ו' if marked_vowel(marks).is_some() => Some("v"),
        _ => None,
    }
}

/// True when the letter's vowel is written on the vav that follows it. A vav
/// carrying holam or shuruk is the vowel of the consonant before it rather than
/// a syllable of its own, and the model, which reads the plain letters, has
/// already put that vowel on the consonant. Counting both spells it twice.
fn before_mater_vav(
    normalized: &str,
    nikud: &HashMap<usize, String>,
    c: char,
    marks: &str,
    end: usize,
) -> bool {
    c != 'ו' && marked_vowel(marks).is_none() && normalized[end..].starts_with('ו') && {
        let vav = nikud.get(&end).map(String::as_str).unwrap_or("");
        vav.contains('\u{05b9}') || vav.contains('\u{05bc}')
    }
}

/// True for an unpointed yod that follows an explicit hiriq, tsere or segol: a
/// vowel letter, not an extra /j/ inferred from the stripped word.
fn mater_yod(
    normalized: &str,
    nikud: &HashMap<usize, String>,
    c: char,
    marks: &str,
    start: usize,
) -> bool {
    c == 'י'
        && marks.is_empty()
        && normalized[..start]
            .char_indices()
            .next_back()
            .is_some_and(|(offset, previous)| {
                matches!(
                    explicit_vowel(
                        previous,
                        nikud.get(&offset).map(String::as_str).unwrap_or("")
                    ),
                    Some("i" | "e")
                )
            })
}

/// The vowel a letter actually contributes: the writer's mark wins over the
/// model's guess, and a letter whose vowel is carried by a following mater
/// contributes none. The stress search and the output have to agree on this,
/// or the stress lands on a letter that then spells no vowel and is lost.
fn effective_vowel<'a>(
    normalized: &str,
    nikud: &HashMap<usize, String>,
    start: usize,
    end: usize,
    model: &'a str,
) -> &'a str {
    let Some(c) = normalized[start..end].chars().next() else {
        return model;
    };
    let marks = nikud.get(&start).map(String::as_str).unwrap_or("");
    if mater_yod(normalized, nikud, c, marks, start)
        || before_mater_vav(normalized, nikud, c, marks, end)
    {
        return "\u{2205}";
    }
    explicit_vowel(c, marks).unwrap_or(model)
}

fn explicit_vowel(c: char, marks: &str) -> Option<&'static str> {
    marked_vowel(marks).or_else(|| (c == 'ו' && marks.contains('ּ')).then_some("u"))
}

/// The reading of one Hebrew letter after the writer's marks are applied.
#[derive(Clone, Debug)]
struct Letter {
    /// Byte span of the letter in the plain text.
    start: usize,
    end: usize,
    letter: char,
    /// Index of the whitespace-separated word the letter belongs to.
    word: usize,
    /// IPA consonant, or `∅` for a silent letter.
    consonant: String,
    /// One of `∅ a e i o u`.
    vowel: String,
    stressed: bool,
}

/// One inference pass over a text: the plain letters, the writer's marks and
/// the per-letter reading both renderers share.
struct Reading {
    plain: String,
    marks: HashMap<usize, String>,
    restore: HashMap<usize, char>,
    letters: Vec<Letter>,
}

impl Reading {
    fn marks_of(&self, letter: &Letter) -> &str {
        self.marks
            .get(&letter.start)
            .map(String::as_str)
            .unwrap_or("")
    }

    /// No Hebrew letter follows, looking past a geresh or gershayim inside a
    /// word (ג׳ירפה, צה״ל).
    fn word_final(&self, letter: &Letter) -> bool {
        let rest = &self.plain[letter.end..];
        let rest = rest.strip_prefix(['\'', '"']).unwrap_or(rest);
        !rest.starts_with(is_hebrew)
    }

    fn to_ipa(&self) -> String {
        let plain = &self.plain;
        let mut result = String::new();
        let mut letters = self.letters.iter().peekable();
        for (offset, c) in plain.char_indices() {
            let Some(letter) = letters.next_if(|letter| letter.start == offset) else {
                if c != '\'' && c != '"' {
                    result.push(c);
                }
                continue;
            };
            let consonant = letter.consonant.as_str();
            let vowel = letter.vowel.as_str();
            let end = letter.end;
            let word_final = end >= plain.len()
                || plain[end..].starts_with(|c: char| c.is_whitespace() || !c.is_alphabetic());
            if c == 'ח' && word_final && vowel == "a" {
                if letter.stressed {
                    result.push_str(STRESS);
                }
                result.push_str("aχ");
                continue;
            }
            if consonant != NONE {
                result.push_str(consonant);
            }
            if vowel != NONE {
                if letter.stressed {
                    result.push_str(STRESS);
                }
                result.push_str(vowel);
            }
        }
        result
    }

    /// Render the reading as niqqud on the writer's letters.
    ///
    /// Every mark category the writer typed on a letter (vowel, dagesh/rafe,
    /// shin/sin dot, accent) is kept as typed; only missing categories are
    /// filled in from the prediction.
    fn to_nikud(&self, with_stress: bool) -> String {
        let mut letters = self.letters.clone();
        let count = letters.len();
        // Furtive patah: the model may hang the /a/ of רוּחַ on the silent vav.
        // A pointed vav reads as /v/, so write the /a/ under the final het,
        // which reads it before the consonant.
        for i in 0..count.saturating_sub(1) {
            let (vav, het) = (&letters[i], &letters[i + 1]);
            if vav.letter == 'ו'
                && vav.consonant == NONE
                && vav.vowel == "a"
                && self.marks_of(vav).is_empty()
                && vav.end == het.start
                && het.letter == 'ח'
                && het.vowel == NONE
                && self.word_final(het)
                && !has_vowel_mark(het.letter, self.marks_of(het))
            {
                let stressed = vav.stressed;
                letters[i + 1].vowel = "a".to_owned();
                letters[i + 1].stressed |= stressed;
                letters[i].vowel = NONE.to_owned();
                letters[i].stressed = false;
            }
        }
        let letters = &letters;
        let adjacent_next =
            |i: usize| (i + 1 < count && letters[i + 1].start == letters[i].end).then(|| i + 1);
        let adjacent_previous =
            |i: usize| (i > 0 && letters[i - 1].end == letters[i].start).then(|| i - 1);
        let words_with_typed_stress: HashSet<usize> = letters
            .iter()
            .filter(|letter| self.marks_of(letter).contains(HATAMA))
            .map(|letter| letter.word)
            .collect();

        let mut added = vec![String::new(); count];
        for i in 0..count {
            let letter = &letters[i];
            let marks = self.marks_of(letter);
            let word_final = self.word_final(letter);
            let next = adjacent_next(i);
            // A following letter nobody pointed that reads as silent: a vowel
            // letter (mater) this letter's vowel can be written on.
            let bare_next = next.filter(|&n| {
                self.marks_of(&letters[n]).is_empty()
                    && letters[n].consonant == NONE
                    && letters[n].vowel == NONE
            });

            if !marks
                .chars()
                .any(|c| matches!(c, DAGESH | RAFE | SHIN_DOT | SIN_DOT))
                && let Some(mark) = consonant_mark(letter, word_final)
            {
                added[i].push(mark);
            }

            let mut carrier = i;
            if !has_vowel_mark(letter.letter, marks) {
                let mater_vav =
                    bare_next.filter(|&n| letters[n].letter == 'ו' && letter.letter != 'ו');
                let mater_yod = bare_next.filter(|&n| letters[n].letter == 'י');
                let silent_final = bare_next.filter(|&n| {
                    matches!(letters[n].letter, 'ה' | 'א') && self.word_final(&letters[n])
                });
                let is_vav = letter.letter == 'ו';
                let consonantal = letter.consonant != NONE;
                match letter.vowel.as_str() {
                    // Holam male and shuruk: the vowel sits on the vav.
                    "o" | "u" if mater_vav.is_some() => {
                        let vav = mater_vav.expect("checked");
                        added[vav].push(if letter.vowel == "o" { HOLAM } else { DAGESH });
                        carrier = vav;
                    }
                    "a" if silent_final.is_some() || (letter.letter == 'ך' && word_final) => {
                        added[i].push(QAMATS)
                    }
                    "a" => added[i].push(PATAH),
                    "e" if mater_yod.is_some() => added[i].push(TSERE),
                    "e" => added[i].push(SEGOL),
                    "i" => added[i].push(HIRIQ),
                    // Plain holam on a consonantal vav would read as holam male.
                    "o" if is_vav && consonantal => added[i].push(HOLAM_HASER_FOR_VAV),
                    "o" => added[i].push(HOLAM),
                    "u" if is_vav && !consonantal => added[i].push(DAGESH),
                    "u" => added[i].push(QUBUTS),
                    _ => {
                        let next_carries_vowel = next.is_some_and(|n| {
                            letters[n].letter == 'ו'
                                && letters[n].consonant == NONE
                                && matches!(letters[n].vowel.as_str(), "o" | "u")
                        });
                        // A bare yod after hiriq or tsere reads as a vowel
                        // letter, so a consonantal one needs its shva even at
                        // the end of a word.
                        let consonantal_yod = letter.letter == 'י'
                            && adjacent_previous(i)
                                .is_some_and(|p| matches!(letters[p].vowel.as_str(), "i" | "e"));
                        if consonantal
                            && !next_carries_vowel
                            && (!word_final || letter.letter == 'ך' || consonantal_yod)
                        {
                            added[i].push(SHVA);
                        }
                    }
                }
            }

            if with_stress
                && letter.stressed
                && letter.vowel != NONE
                && !words_with_typed_stress.contains(&letter.word)
                && !marks.contains(HATAMA)
            {
                added[carrier].push(HATAMA);
            }
        }

        let mut output = String::with_capacity(self.plain.len() * 2);
        let mut next_letter = 0;
        for (offset, c) in self.plain.char_indices() {
            output.push(self.restore.get(&offset).copied().unwrap_or(c));
            if next_letter < count && letters[next_letter].start == offset {
                output.push_str(self.marks_of(&letters[next_letter]));
                output.push_str(&added[next_letter]);
                next_letter += 1;
            }
        }
        output.nfc().collect()
    }
}

fn has_vowel_mark(letter: char, marks: &str) -> bool {
    marks
        .chars()
        .any(|c| matches!(c, '\u{05B0}'..='\u{05BB}' | QAMATS_QATAN))
        || (letter == 'ו' && marks.contains(DAGESH))
}

fn consonant_mark(letter: &Letter, word_final: bool) -> Option<char> {
    match (letter.letter, letter.consonant.as_str()) {
        ('ב', "b") | ('כ' | 'ך', "k") | ('פ' | 'ף', "p") => Some(DAGESH),
        ('ש', "ʃ") => Some(SHIN_DOT),
        ('ש', "s") => Some(SIN_DOT),
        // Mappiq: a pronounced final he.
        ('ה', "h") if word_final => Some(DAGESH),
        _ => None,
    }
}

pub struct G2P {
    session: Session,
    vocab: HashMap<char, i64>,
    consonant_vocab: HashMap<i64, String>,
    vowel_vocab: HashMap<i64, String>,
    letter_consonant_mask: HashMap<char, Vec<i64>>,
    geresh_map: HashMap<char, String>,
    cls_id: i64,
    sep_id: i64,
    gender_conditioned: bool,
}

impl G2P {
    pub fn new(model_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        Self::from_session(session)
    }

    pub fn from_session(session: Session) -> anyhow::Result<Self> {
        let gender_conditioned = {
            let names: HashSet<&str> = session.inputs().iter().map(|i| i.name()).collect();
            names.contains("speaker") && names.contains("target_speaker")
        };

        let (
            vocab_json,
            consonant_vocab_json,
            vowel_vocab_json,
            letter_consonant_mask_json,
            geresh_map_json,
            cls_id,
            sep_id,
        ) = {
            let meta = session.metadata()?;
            let vocab_json = meta
                .custom("vocab")
                .ok_or_else(|| anyhow::anyhow!("missing vocab"))?;
            let consonant_vocab_json = meta
                .custom("consonant_vocab")
                .ok_or_else(|| anyhow::anyhow!("missing consonant_vocab"))?;
            let vowel_vocab_json = meta
                .custom("vowel_vocab")
                .ok_or_else(|| anyhow::anyhow!("missing vowel_vocab"))?;
            // RenikudPlus prefers letter_consonant_constraints; fall back to mask.
            let letter_consonant_mask_json = meta
                .custom("letter_consonant_constraints")
                .or_else(|| meta.custom("letter_consonant_mask"))
                .ok_or_else(|| anyhow::anyhow!("missing letter_consonant_constraints/mask"))?;
            let geresh_map_json = meta
                .custom("geresh_map")
                .unwrap_or_else(|| "{}".to_string());
            let cls_id: i64 = meta
                .custom("cls_token_id")
                .ok_or_else(|| anyhow::anyhow!("missing cls_token_id"))?
                .parse()?;
            let sep_id: i64 = meta
                .custom("sep_token_id")
                .ok_or_else(|| anyhow::anyhow!("missing sep_token_id"))?
                .parse()?;
            (
                vocab_json,
                consonant_vocab_json,
                vowel_vocab_json,
                letter_consonant_mask_json,
                geresh_map_json,
                cls_id,
                sep_id,
            )
        };

        let raw_vocab: HashMap<String, i64> = serde_json::from_str(&vocab_json)?;
        let vocab: HashMap<char, i64> = raw_vocab
            .into_iter()
            .filter_map(|(k, v)| k.chars().next().map(|c| (c, v)))
            .collect();

        let raw_consonants: HashMap<String, String> = serde_json::from_str(&consonant_vocab_json)?;
        let consonant_vocab: HashMap<i64, String> = raw_consonants
            .into_iter()
            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v)))
            .collect();

        let raw_vowels: HashMap<String, String> = serde_json::from_str(&vowel_vocab_json)?;
        let vowel_vocab: HashMap<i64, String> = raw_vowels
            .into_iter()
            .filter_map(|(k, v)| k.parse::<i64>().ok().map(|id| (id, v)))
            .collect();

        let raw_mask: HashMap<String, Vec<i64>> =
            serde_json::from_str(&letter_consonant_mask_json)?;
        let letter_consonant_mask: HashMap<char, Vec<i64>> = raw_mask
            .into_iter()
            .filter_map(|(k, v)| k.chars().next().map(|c| (c, v)))
            .collect();

        let raw_geresh: HashMap<String, String> = serde_json::from_str(&geresh_map_json)?;
        let geresh_map: HashMap<char, String> = raw_geresh
            .into_iter()
            .filter_map(|(k, v)| k.chars().next().map(|c| (c, v)))
            .collect();

        Ok(Self {
            session,
            vocab,
            consonant_vocab,
            vowel_vocab,
            letter_consonant_mask,
            geresh_map,
            cls_id,
            sep_id,
            gender_conditioned,
        })
    }

    pub fn is_gender_conditioned(&self) -> bool {
        self.gender_conditioned
    }

    fn tokenize(&self, text: &str) -> (Vec<i64>, Vec<i64>, Vec<(usize, usize)>) {
        let normalized: String = text.nfd().collect();
        let unk_id = 0i64;
        let mut ids = vec![self.cls_id];
        let mut offsets = vec![(0usize, 0usize)];
        for (i, c) in normalized.char_indices() {
            ids.push(*self.vocab.get(&c).unwrap_or(&unk_id));
            offsets.push((i, i + c.len_utf8()));
        }
        ids.push(self.sep_id);
        offsets.push((0, 0));
        let mask = vec![1i64; ids.len()];
        (ids, mask, offsets)
    }

    /// Convert Hebrew text to IPA.
    ///
    /// `speaker` / `target_speaker`: 0 unknown, 1 male, 2 female. Ignored on legacy models.
    /// Niqqud the writer typed wins over the model, including the hatama
    /// (U+05AB) as the stressed syllable.
    pub fn phonemize(
        &mut self,
        text: &str,
        speaker: u8,
        target_speaker: u8,
    ) -> anyhow::Result<String> {
        Ok(self.read(text, speaker, target_speaker)?.to_ipa())
    }

    /// Add niqqud to Hebrew text, returning NFC text.
    ///
    /// Punctuation, whitespace, geresh and non-Hebrew text are returned as
    /// written, and every mark the writer already typed is kept. With
    /// `with_stress`, the stressed letter of each word gets the hatama
    /// (U+05AB), which [`Self::phonemize`] reads back as the stress.
    pub fn diacritize(
        &mut self,
        text: &str,
        speaker: u8,
        target_speaker: u8,
        with_stress: bool,
    ) -> anyhow::Result<String> {
        Ok(self
            .read(text, speaker, target_speaker)?
            .to_nikud(with_stress))
    }

    fn read(&mut self, text: &str, speaker: u8, target_speaker: u8) -> anyhow::Result<Reading> {
        if speaker > 2 || target_speaker > 2 {
            anyhow::bail!("speaker and target_speaker must be 0, 1, or 2");
        }
        if !self.gender_conditioned && (speaker != 0 || target_speaker != 0) {
            anyhow::bail!(
                "this ONNX model is not gender-conditioned; speaker/target_speaker must be 0"
            );
        }

        let Separated {
            plain,
            marks: nikud,
            restore,
        } = separate_nikud(text);
        if !plain.chars().any(is_hebrew) {
            return Ok(Reading {
                plain,
                marks: nikud,
                restore,
                letters: Vec::new(),
            });
        }
        let (ids, mask, offsets) = self.tokenize(&plain);
        let len = ids.len();

        let input_ids = Tensor::<i64>::from_array(([1, len], ids.into_boxed_slice()))?;
        let attention_mask = Tensor::<i64>::from_array(([1, len], mask.into_boxed_slice()))?;

        let outputs = if self.gender_conditioned {
            let speaker_t =
                Tensor::<i64>::from_array(([1], vec![i64::from(speaker)].into_boxed_slice()))?;
            let target_t = Tensor::<i64>::from_array((
                [1],
                vec![i64::from(target_speaker)].into_boxed_slice(),
            ))?;
            self.session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask,
                "speaker" => speaker_t,
                "target_speaker" => target_t
            ])?
        } else {
            self.session.run(ort::inputs![
                "input_ids" => input_ids,
                "attention_mask" => attention_mask
            ])?
        };

        let (cons_shape, cons_data) = outputs["consonant_logits"].try_extract_tensor::<f32>()?;
        let (vowel_shape, vowel_data) = outputs["vowel_logits"].try_extract_tensor::<f32>()?;
        let (_, stress_data) = outputs["stress_logits"].try_extract_tensor::<f32>()?;

        let num_consonants = cons_shape[2] as usize;
        let num_vowels = vowel_shape[2] as usize;

        let argmax = |data: &[f32], offset: usize, size: usize| -> i64 {
            data[offset..offset + size]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as i64)
                .unwrap_or(0)
        };

        let mut letters = Vec::new();
        let mut stress_scores = Vec::new();
        let mut word = 0usize;
        let mut previous_end = 0usize;
        for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
            if end <= start {
                continue;
            }
            if plain[previous_end..start].chars().any(char::is_whitespace) {
                word += 1;
            }
            let c = plain[start..end].chars().next().expect("non-empty span");
            if c.is_whitespace() {
                word += 1;
            }
            previous_end = end;
            if !is_hebrew(c) {
                continue;
            }

            let consonant_id = if let Some(allowed) = self.letter_consonant_mask.get(&c) {
                let base = tok_idx * num_consonants;
                allowed
                    .iter()
                    .copied()
                    .max_by(|&a, &b| {
                        let fa = cons_data
                            .get(base + a as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY);
                        let fb = cons_data
                            .get(base + b as usize)
                            .copied()
                            .unwrap_or(f32::NEG_INFINITY);
                        fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            } else {
                argmax(cons_data, tok_idx * num_consonants, num_consonants)
            };
            let model_consonant = match self.geresh_map.get(&c) {
                Some(geresh) if plain[end..].starts_with('\'') => geresh.as_str(),
                _ => self
                    .consonant_vocab
                    .get(&consonant_id)
                    .map(String::as_str)
                    .unwrap_or(NONE),
            };
            let model_vowel = self
                .vowel_vocab
                .get(&argmax(vowel_data, tok_idx * num_vowels, num_vowels))
                .map(String::as_str)
                .unwrap_or(NONE);

            let marks = nikud.get(&start).map(String::as_str).unwrap_or("");
            let consonant = if mater_yod(&plain, &nikud, c, marks, start) {
                NONE
            } else {
                match marked_consonant(c, marks) {
                    // A pointed vav is consonantal; keep the model's /w/ over /v/.
                    Some("v") if c == 'ו' && model_consonant == "w" => "w",
                    Some(marked) => marked,
                    None => model_consonant,
                }
            };
            let vowel = effective_vowel(&plain, &nikud, start, end, model_vowel);

            letters.push(Letter {
                start,
                end,
                letter: c,
                word,
                consonant: consonant.to_owned(),
                vowel: vowel.to_owned(),
                stressed: false,
            });
            stress_scores.push(stress_data[tok_idx * 2 + 1]);
        }
        drop(outputs);

        // A holam or shuruk vav written by the writer carries the vowel the
        // model scored on the consonant before it, so it carries that
        // consonant's stress score too.
        for i in 1..letters.len() {
            let previous = &letters[i - 1];
            if previous.end == letters[i].start
                && before_mater_vav(
                    &plain,
                    &nikud,
                    previous.letter,
                    nikud.get(&previous.start).map(String::as_str).unwrap_or(""),
                    previous.end,
                )
            {
                stress_scores[i] = stress_scores[i].max(stress_scores[i - 1]);
            }
        }

        // One stress per word: the writer's hatama if typed, otherwise the
        // highest-scoring letter that carries a vowel (RenikudPlus behavior).
        let mut first = 0;
        while first < letters.len() {
            let word = letters[first].word;
            let last = letters[first..]
                .iter()
                .position(|letter| letter.word != word)
                .map_or(letters.len(), |n| first + n);
            let typed = (first..last).rev().find(|&i| {
                nikud
                    .get(&letters[i].start)
                    .is_some_and(|marks| marks.contains(HATAMA))
            });
            let chosen = typed.or_else(|| {
                let voiced: Vec<usize> = (first..last)
                    .filter(|&i| letters[i].vowel != NONE)
                    .collect();
                let pool = if voiced.is_empty() {
                    (first..last).collect()
                } else {
                    voiced
                };
                pool.into_iter().max_by(|&a, &b| {
                    stress_scores[a]
                        .partial_cmp(&stress_scores[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            });
            if let Some(i) = chosen {
                letters[i].stressed = true;
            }
            first = last;
        }

        Ok(Reading {
            plain,
            marks: nikud,
            restore,
            letters,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn letter(start: usize, c: char, word: usize, consonant: &str, vowel: &str) -> Letter {
        Letter {
            start,
            end: start + c.len_utf8(),
            letter: c,
            word,
            consonant: consonant.to_owned(),
            vowel: vowel.to_owned(),
            stressed: false,
        }
    }

    /// Build a reading as the model would for plain `text`, one (consonant,
    /// vowel) per Hebrew letter, with the stressed letter index per word.
    fn reading(text: &str, predictions: &[(&str, &str)], stressed: &[usize]) -> Reading {
        let Separated {
            plain,
            marks,
            restore,
        } = separate_nikud(text);
        let mut letters = Vec::new();
        let mut word = 0;
        let mut predictions = predictions.iter();
        for (offset, c) in plain.char_indices() {
            if c.is_whitespace() {
                word += 1;
            }
            if is_hebrew(c) {
                let (consonant, vowel) = predictions.next().expect("prediction per letter");
                letters.push(letter(offset, c, word, consonant, vowel));
            }
        }
        for &i in stressed {
            letters[i].stressed = true;
        }
        Reading {
            plain,
            marks,
            restore,
            letters,
        }
    }

    #[test]
    fn nikud_stays_aligned_in_mixed_text() {
        let Separated {
            plain,
            marks: nikud,
            ..
        } = separate_nikud("לכן אֶן קֶלְוִין שלום");
        assert_eq!(plain, "לכן אן קלוין שלום");
        assert_eq!(
            explicit_vowel('א', &nikud[&plain.find('א').unwrap()]),
            Some("e")
        );
        assert_eq!(
            explicit_vowel('ק', &nikud[&plain.find('ק').unwrap()]),
            Some("e")
        );
        assert_eq!(
            explicit_vowel('ל', &nikud[&(plain.find('ק').unwrap() + 2)]),
            Some("∅")
        );
        assert_eq!(
            explicit_vowel('ו', &nikud[&plain.find('ו').unwrap()]),
            Some("i")
        );
        assert!(!nikud.contains_key(&plain.find('ש').unwrap()));
    }

    #[test]
    fn explicit_consonants_and_vowels_override_predictions() {
        assert_eq!(marked_consonant('ב', "ֵּ"), Some("b"));
        assert_eq!(marked_consonant('ש', "ׂ"), Some("s"));
        assert_eq!(marked_consonant('ו', "ִ"), Some("v"));
        assert_eq!(marked_consonant('ו', "ּ"), Some("∅"));
        assert_eq!(marked_consonant('ו', "ֺ"), Some("v"));
        assert_eq!(explicit_vowel('ו', "ּ"), Some("u"));
        assert_eq!(explicit_vowel('ק', ""), None);
        assert_eq!(separate_nikud("אֶן־שלום׃").plain, "אן־שלום׃");
    }

    #[test]
    fn a_mater_vav_takes_the_vowel_rather_than_adding_one() {
        let Separated {
            plain,
            marks: nikud,
            ..
        } = separate_nikud("שָׁלוֹם");
        let lamed = plain.find('ל').expect("lamed");
        let end = lamed + 'ל'.len_utf8();
        // The lamed carries no vowel of its own; the holam vav after it does.
        assert!(before_mater_vav(&plain, &nikud, 'ל', "", end));
        // So the model's guess for the lamed is dropped rather than kept
        // alongside the writer's mark, which is what spelled שלום as ʃalˈoom.
        assert_eq!(effective_vowel(&plain, &nikud, lamed, end, "o"), "∅");
    }

    #[test]
    fn a_consonant_with_its_own_vowel_keeps_it_before_a_vav() {
        let Separated {
            plain,
            marks: nikud,
            ..
        } = separate_nikud("שָׁלוֹם");
        let shin = plain.find('ש').expect("shin");
        let end = shin + 'ש'.len_utf8();
        assert_eq!(effective_vowel(&plain, &nikud, shin, end, "∅"), "a");
    }

    #[test]
    fn diacritizes_holam_male_dagesh_and_shin_dot() {
        let reading = reading(
            "שלום, ספר",
            &[
                ("ʃ", "a"),
                ("l", "o"),
                (NONE, NONE),
                ("m", NONE),
                ("s", "e"),
                ("f", "e"),
                ("ʁ", NONE),
            ],
            &[1, 4],
        );
        assert_eq!(reading.to_ipa(), "ʃalˈom, sˈefeʁ");
        let nikud = reading.to_nikud(true);
        let expected: String = "שַׁלוֹ\u{05AB}ם, סֶ\u{05AB}פֶר".nfc().collect();
        assert_eq!(nikud, expected);
        let plain: String = "שַׁלוֹם, סֶפֶר".nfc().collect();
        assert_eq!(reading.to_nikud(false), plain);
    }

    #[test]
    fn diacritizes_dagesh_shuruk_hiriq_male_and_final_kaf() {
        // בּוּקִי לְךָ: b u (shuruk on the vav), k i + mater yod, l shva, χ a.
        let reading = reading(
            "בוקי לך",
            &[
                ("b", "u"),
                (NONE, NONE),
                ("k", "i"),
                (NONE, NONE),
                ("l", NONE),
                ("χ", "a"),
            ],
            &[0, 5],
        );
        let expected: String = "בּוּ\u{05AB}קִי לְךָ\u{05AB}".nfc().collect();
        assert_eq!(reading.to_nikud(true), expected);
    }

    #[test]
    fn furtive_patah_moves_from_a_silent_vav_to_the_final_het() {
        let reading = reading("רוח", &[("ʁ", "u"), (NONE, "a"), ("χ", NONE)], &[0]);
        assert_eq!(reading.to_ipa(), "ʁˈuaχ");
        let expected: String = "רוּ\u{05AB}חַ".nfc().collect();
        assert_eq!(reading.to_nikud(true), expected);
    }

    #[test]
    fn typed_niqqud_and_punctuation_are_preserved() {
        // The writer pointed the qamats and the sin; the rest is predicted.
        let reading = reading(
            "שָׂרה׳ \"abc\"",
            &[("s", "a"), ("ʁ", "a"), (NONE, NONE)],
            &[1],
        );
        let expected: String = "שָׂרָ\u{05AB}ה׳ \"abc\"".nfc().collect();
        assert_eq!(reading.to_nikud(true), expected);
    }

    #[test]
    fn consonantal_vav_keeps_its_consonant_through_the_round_trip() {
        let reading = reading(
            "וורד",
            &[("v", "e"), (NONE, NONE), ("ʁ", "e"), ("d", NONE)],
            &[0],
        );
        let nikud = reading.to_nikud(true);
        // The consonantal vav takes its segol; the bare second vav stays bare.
        assert!(nikud.starts_with("ו\u{05B6}\u{05AB}ו"), "{nikud}");
        let reading = reading_from_nikud("וֺ");
        assert_eq!(reading.consonant, "v");
    }

    fn reading_from_nikud(text: &str) -> Letter {
        let Separated {
            plain,
            marks: nikud,
            ..
        } = separate_nikud(text);
        let c = plain.chars().next().unwrap();
        let marks = nikud.get(&0).map(String::as_str).unwrap_or("");
        let consonant = marked_consonant(c, marks).unwrap_or(NONE);
        let vowel = effective_vowel(&plain, &nikud, 0, c.len_utf8(), NONE);
        letter(0, c, 0, consonant, vowel)
    }

    #[test]
    fn diacritized_letters_read_back_as_the_prediction() {
        // Each generated pointing must decode to the class it came from.
        for (text, consonant, vowel) in [
            ("בַּ", "b", "a"),
            ("כֶ", NONE, "e"),
            ("פִּ", "p", "i"),
            ("שׁ", "ʃ", NONE),
            ("שׂ", "s", NONE),
            ("לְ", NONE, NONE),
            ("קֻ", NONE, "u"),
        ] {
            let letter = reading_from_nikud(text);
            assert_eq!(letter.consonant, consonant, "{text}");
            assert_eq!(letter.vowel, vowel, "{text}");
        }
    }

    /// Needs `MAMBOTTS_RENIKUD_PATH` pointing at `renikud-plus.onnx`.
    #[test]
    #[ignore = "requires MAMBOTTS_RENIKUD_PATH pointing to renikud-plus.onnx"]
    fn diacritize_then_phonemize_matches_phonemize_with_real_model() {
        let model = std::env::var("MAMBOTTS_RENIKUD_PATH").expect("set MAMBOTTS_RENIKUD_PATH");
        let mut g2p = G2P::new(&model).unwrap();
        for text in [
            "שלום עולם, מה שלומך היום?",
            "הילדים הלכו לבית הספר בבוקר.",
            "ספר",
            "שבת שלום",
            "אני אוהב machine learning וגם ג׳אז.",
            "הַמְּנוֹרָה מאירה.",
        ] {
            let ipa = g2p.phonemize(text, 0, 0).unwrap();
            let nikud = g2p.diacritize(text, 0, 0, true).unwrap();
            let round_trip = g2p.phonemize(&nikud, 0, 0).unwrap();
            println!("{text}\n  {nikud}\n  {ipa}\n  {round_trip}");
            assert_eq!(round_trip, ipa, "{text} -> {nikud}");
            let plain: String = nikud.chars().filter(|&c| !is_mark(c)).collect();
            let original: String = text.nfd().filter(|&c| !is_mark(c)).collect();
            assert_eq!(plain, original, "letters and punctuation stay as written");
        }
    }
}
