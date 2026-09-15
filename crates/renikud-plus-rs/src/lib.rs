//! RenikudPlus Hebrew grapheme-to-phoneme via ONNX.
//!
//! Supports legacy models (`input_ids` + `attention_mask` only) and gender-conditioned
//! RenikudPlus models that also require `speaker` / `target_speaker`
//! (0 = unknown, 1 = male, 2 = female).

use std::collections::{HashMap, HashSet};

use ort::session::Session;
use ort::value::Tensor;
use unicode_normalization::UnicodeNormalization;

const ALEF: u32 = 0x05D0;
const TAF: u32 = 0x05EA;
const STRESS: &str = "ˈ";

fn is_hebrew(c: char) -> bool {
    let cp = c as u32;
    (ALEF..=TAF).contains(&cp)
}

fn normalize_graphemes(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{05F3}' | '\'' | '`' | '\u{00B4}' => '\'',
            '\u{05F4}' | '\u{201C}' | '\u{201D}' => '"',
            _ => c,
        })
        .collect()
}

// Keep marks aligned to plain-text byte offsets: the model sees its usual input,
// while decoding can honor the user's vowels and consonant marks.
fn separate_nikud(text: &str) -> (String, HashMap<usize, String>) {
    let mut plain = String::new();
    let mut marks = HashMap::<usize, String>::new();
    let mut letter = None;
    for c in text.nfd() {
        if matches!(c, '\u{0591}'..='\u{05bd}' | '\u{05bf}' | '\u{05c1}'..='\u{05c2}' | '\u{05c4}'..='\u{05c5}' | '\u{05c7}')
        {
            if let Some(offset) = letter {
                marks.entry(offset).or_default().push(c);
            }
        } else {
            letter = is_hebrew(c).then_some(plain.len());
            plain.push(c);
        }
    }
    (plain, marks)
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

fn explicit_vowel(c: char, marks: &str) -> Option<&'static str> {
    marked_vowel(marks).or_else(|| (c == 'ו' && marks.contains('ּ')).then_some("u"))
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
    pub fn phonemize(
        &mut self,
        text: &str,
        speaker: u8,
        target_speaker: u8,
    ) -> anyhow::Result<String> {
        if speaker > 2 || target_speaker > 2 {
            anyhow::bail!("speaker and target_speaker must be 0, 1, or 2");
        }
        if !self.gender_conditioned && (speaker != 0 || target_speaker != 0) {
            anyhow::bail!(
                "this ONNX model is not gender-conditioned; speaker/target_speaker must be 0"
            );
        }

        let text = normalize_graphemes(text);
        let (normalized, nikud) = separate_nikud(&text);
        let (ids, mask, offsets) = self.tokenize(&normalized);
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

        let word_spans: Vec<(usize, usize)> = {
            let mut spans = Vec::new();
            let mut in_word = false;
            let mut word_start = 0;
            for (i, c) in normalized.char_indices() {
                if c.is_whitespace() {
                    if in_word {
                        spans.push((word_start, i));
                        in_word = false;
                    }
                } else if !in_word {
                    word_start = i;
                    in_word = true;
                }
            }
            if in_word {
                spans.push((word_start, normalized.len()));
            }
            spans
        };

        let vowel_ids: Vec<i64> = (0..offsets.len())
            .map(|tok_idx| argmax(vowel_data, tok_idx * num_vowels, num_vowels))
            .collect();

        // Prefer stress over tokens predicted to carry a vowel (RenikudPlus behavior).
        let stressed_positions: HashSet<usize> = {
            let mut stressed = HashSet::new();
            for (ws, we) in &word_spans {
                let candidates: Vec<usize> = offsets
                    .iter()
                    .enumerate()
                    .filter(|&(tok_idx, &(start, end))| {
                        end > start
                            && start >= *ws
                            && start < *we
                            && explicit_vowel(
                                normalized[start..end].chars().next().unwrap(),
                                nikud.get(&start).map(String::as_str).unwrap_or(""),
                            )
                            .unwrap_or_else(|| {
                                self.vowel_vocab
                                    .get(&vowel_ids[tok_idx])
                                    .map(String::as_str)
                                    .unwrap_or("∅")
                            }) != "∅"
                    })
                    .map(|(i, _)| i)
                    .collect();
                let pool = if candidates.is_empty() {
                    offsets
                        .iter()
                        .enumerate()
                        .filter(|&(_, &(start, end))| end > start && start >= *ws && start < *we)
                        .map(|(i, _)| i)
                        .collect::<Vec<_>>()
                } else {
                    candidates
                };
                if let Some(idx) = pool.into_iter().max_by(|&a, &b| {
                    let sa = stress_data[a * 2 + 1];
                    let sb = stress_data[b * 2 + 1];
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    stressed.insert(idx);
                }
            }
            stressed
        };

        let mut result = String::new();
        let mut prev_end = 0usize;
        for (tok_idx, &(start, end)) in offsets.iter().enumerate() {
            let char_len = end.saturating_sub(start);
            if char_len == 0 {
                continue;
            }

            if start > prev_end {
                result.push_str(&normalized[prev_end..start]);
            }

            let c = normalized[start..end].chars().next().unwrap();
            prev_end = end;

            if !is_hebrew(c) {
                if c == '\'' || c == '"' {
                    continue;
                }
                result.push(c);
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
            let vowel_id = vowel_ids[tok_idx];
            let stressed = stressed_positions.contains(&tok_idx);

            let consonant_str: String;
            let consonant = if let Some(geresh) = self.geresh_map.get(&c) {
                if normalized[end..].starts_with('\'') {
                    geresh.as_str()
                } else {
                    self.consonant_vocab
                        .get(&consonant_id)
                        .map(String::as_str)
                        .unwrap_or("∅")
                }
            } else {
                consonant_str = self
                    .consonant_vocab
                    .get(&consonant_id)
                    .cloned()
                    .unwrap_or_else(|| "∅".to_string());
                &consonant_str
            };
            let vowel = self
                .vowel_vocab
                .get(&vowel_id)
                .map(String::as_str)
                .unwrap_or("∅");

            let marks = nikud.get(&start).map(String::as_str).unwrap_or("");
            // An unpointed yod following an explicit hiriq/tsere/segol is a
            // vowel letter, not an extra /j/ inferred from the stripped word.
            let mater_yod = c == 'י'
                && marks.is_empty()
                && normalized[..start].char_indices().next_back().is_some_and(
                    |(offset, previous)| {
                        matches!(
                            explicit_vowel(
                                previous,
                                nikud.get(&offset).map(String::as_str).unwrap_or("")
                            ),
                            Some("i" | "e")
                        )
                    },
                );
            let consonant = if mater_yod {
                "∅"
            } else {
                marked_consonant(c, marks).unwrap_or(consonant)
            };
            let vowel = if mater_yod {
                "∅"
            } else {
                explicit_vowel(c, marks).unwrap_or(vowel)
            };

            let word_final = end >= normalized.len()
                || normalized[end..].starts_with(|c: char| c.is_whitespace() || !c.is_alphabetic());
            if c == 'ח' && word_final && vowel == "a" {
                if stressed {
                    result.push_str(STRESS);
                }
                result.push_str("aχ");
            } else {
                if consonant != "∅" {
                    result.push_str(consonant);
                }
                if stressed && vowel != "∅" {
                    result.push_str(STRESS);
                }
                if vowel != "∅" {
                    result.push_str(vowel);
                }
            }
        }

        if prev_end < normalized.len() {
            result.push_str(&normalized[prev_end..]);
        }

        drop(outputs);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nikud_stays_aligned_in_mixed_text() {
        let (plain, marks) = separate_nikud("לכן אֶן קֶלְוִין שלום");
        assert_eq!(plain, "לכן אן קלוין שלום");
        assert_eq!(
            explicit_vowel('א', &marks[&plain.find('א').unwrap()]),
            Some("e")
        );
        assert_eq!(
            explicit_vowel('ק', &marks[&plain.find('ק').unwrap()]),
            Some("e")
        );
        assert_eq!(
            explicit_vowel('ל', &marks[&(plain.find('ק').unwrap() + 2)]),
            Some("∅")
        );
        assert_eq!(
            explicit_vowel('ו', &marks[&plain.find('ו').unwrap()]),
            Some("i")
        );
        assert!(!marks.contains_key(&plain.find('ש').unwrap()));
    }

    #[test]
    fn explicit_consonants_and_vowels_override_predictions() {
        assert_eq!(marked_consonant('ב', "ֵּ"), Some("b"));
        assert_eq!(marked_consonant('ש', "ׂ"), Some("s"));
        assert_eq!(marked_consonant('ו', "ִ"), Some("v"));
        assert_eq!(marked_consonant('ו', "ּ"), Some("∅"));
        assert_eq!(explicit_vowel('ו', "ּ"), Some("u"));
        assert_eq!(explicit_vowel('ק', ""), None);
        assert_eq!(separate_nikud("אֶן־שלום׃").0, "אן־שלום׃");
    }
}
