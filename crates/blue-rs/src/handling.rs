//! Model-independent handling for difficult Hebrew TTS input.
//!
//! Niqqud-bearing words are kept intact for [Phonikud](https://github.com/phonikud/phonikud)
//! grapheme-to-IPA. Separately, niqqud is stripped so Renikud can run on plain
//! Hebrew, then Phonikud's stress mark (`ˈ`) is copied onto the Renikud IPA.

use anyhow::{Result, bail};
use regex::{Captures, Regex};

/// Primary stress mark used by Phonikud (`U+02C8`).
pub const STRESS_MARK: char = '\u{02c8}';
/// Internal boundaries for segments that need slower, clearer synthesis.
pub const REF_CODE_MARK_OPEN: char = '【';
pub const REF_CODE_MARK_CLOSE: char = '】';

/// Text that is ready for G2P, split by its requested synthesis pacing.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedSegment {
    pub text: String,
    pub is_reference_code: bool,
}

/// Destination representation expected by the selected synthesis model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputMode {
    Text,
    Phonemes,
}

/// A vocalized Hebrew source word and its IPA stages.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhoneticSpan {
    /// Original word, including niqqud.
    pub source: String,
    /// Phonikud IPA (stress included when present).
    pub phonikud_ipa: String,
    /// Renikud IPA from the stripped (plain) form, when available.
    pub renikud_ipa: Option<String>,
    /// Final IPA: Renikud base with Phonikud stress when both exist,
    /// otherwise Phonikud alone.
    pub ipa: String,
}

/// Safe text and phoneme-ready renderings of one original input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedText {
    pub original: String,
    /// Normalized text with niqqud preserved (for text-native models).
    pub text: String,
    /// Same as `text`, but niqqud words replaced by final IPA.
    pub phonetic_text: String,
    pub phonetic_spans: Vec<PhoneticSpan>,
}

impl PreparedText {
    pub fn for_input(&self, mode: InputMode) -> &str {
        match mode {
            InputMode::Text => &self.text,
            InputMode::Phonemes => &self.phonetic_text,
        }
    }
}

/// Adapter for Phonikud (or any niqqud → IPA engine).
///
/// Callers must pass the vocalized word **with niqqud kept**.
pub trait NikudPhonemizer {
    fn phonemize_nikud(&mut self, vocalized: &str) -> Result<String>;
}

impl<F> NikudPhonemizer for F
where
    F: FnMut(&str) -> Result<String>,
{
    fn phonemize_nikud(&mut self, vocalized: &str) -> Result<String> {
        self(vocalized)
    }
}

/// Adapter for Renikud (or any plain-Hebrew → IPA engine).
///
/// Callers must pass Hebrew **without niqqud**.
pub trait PlainHebrewPhonemizer {
    fn phonemize_plain(&mut self, unvocalized: &str) -> Result<String>;
}

impl<F> PlainHebrewPhonemizer for F
where
    F: FnMut(&str) -> Result<String>,
{
    fn phonemize_plain(&mut self, unvocalized: &str) -> Result<String> {
        self(unvocalized)
    }
}

/// Strip Hebrew niqqud / cantillation while keeping letters and geresh.
pub fn strip_nikud(text: &str) -> String {
    text.chars()
        .filter(|character| !is_nikud(*character))
        .collect()
}

/// True when `text` contains any Hebrew niqqud / cantillation mark.
pub fn contains_nikud(text: &str) -> bool {
    text.chars().any(is_nikud)
}

/// Remove internal slow-segment markers before displaying prepared text.
pub fn strip_reference_code_markers(text: &str) -> String {
    text.replace(REF_CODE_MARK_OPEN, "")
        .replace(REF_CODE_MARK_CLOSE, "")
}

/// Split prepared text into ordinary and slow reference-code segments.
///
/// Punctuation-only fragments are retained with the preceding segment, matching
/// the Python pipeline's protection against a stray final phoneme.
pub fn split_prepared_by_reference_codes(text: &str) -> Vec<PreparedSegment> {
    let mut segments = Vec::new();
    let mut ordinary = String::new();
    let mut slow = String::new();
    let mut in_slow = false;

    let push = |text: &mut String, is_reference_code, out: &mut Vec<PreparedSegment>| {
        let value = text.trim();
        if !value.is_empty() {
            out.push(PreparedSegment {
                text: value.to_owned(),
                is_reference_code,
            });
        }
        text.clear();
    };

    for character in text.chars() {
        match character {
            REF_CODE_MARK_OPEN if !in_slow => {
                push(&mut ordinary, false, &mut segments);
                in_slow = true;
            }
            REF_CODE_MARK_CLOSE if in_slow => {
                push(&mut slow, true, &mut segments);
                in_slow = false;
            }
            _ if in_slow => slow.push(character),
            _ => ordinary.push(character),
        }
    }
    if in_slow {
        ordinary.push(REF_CODE_MARK_OPEN);
        ordinary.push_str(&slow);
    }
    push(&mut ordinary, false, &mut segments);

    let mut merged: Vec<PreparedSegment> = Vec::new();
    for segment in segments {
        if segment
            .text
            .chars()
            .all(|c| matches!(c, '.' | '!' | '?' | ',' | ';' | ':' | '…'))
            && !merged.is_empty()
        {
            merged
                .last_mut()
                .expect("not empty")
                .text
                .push_str(&segment.text);
        } else {
            merged.push(segment);
        }
    }
    if merged.is_empty() {
        merged.push(PreparedSegment {
            text: strip_reference_code_markers(text).trim().to_owned(),
            is_reference_code: false,
        });
    }
    merged
}

/// Index of the stressed vowel in IPA (`0` = first vowel), if any.
pub fn vowel_stress_index(ipa: &str) -> Option<usize> {
    let mut vowel_index = 0usize;
    let mut pending_stress = false;
    for character in ipa.chars() {
        if character == STRESS_MARK {
            pending_stress = true;
            continue;
        }
        if is_ipa_vowel(character) {
            if pending_stress {
                return Some(vowel_index);
            }
            vowel_index += 1;
        } else {
            // Stress must sit immediately before its vowel.
            pending_stress = false;
        }
    }
    None
}

/// Insert `ˈ` immediately before the vowel at `vowel_index`, replacing any
/// existing stress marks.
pub fn apply_vowel_stress(ipa: &str, vowel_index: usize) -> String {
    let plain: String = ipa.chars().filter(|&c| c != STRESS_MARK).collect();
    let mut output = String::with_capacity(plain.len() + STRESS_MARK.len_utf8());
    let mut seen = 0usize;
    let mut placed = false;
    for character in plain.chars() {
        if !placed && is_ipa_vowel(character) {
            if seen == vowel_index {
                output.push(STRESS_MARK);
                placed = true;
            }
            seen += 1;
        }
        output.push(character);
    }
    if !placed {
        // Clamp: stress the last vowel when indices diverge.
        return apply_vowel_stress_last(&plain);
    }
    output
}

/// Copy Phonikud stress placement onto Renikud IPA.
///
/// Renikud receives the plain (niqqud-stripped) form; Phonikud keeps stress
/// from hatama / milra rules. When Phonikud has no stress mark, Renikud IPA is
/// returned unchanged (aside from whitespace normalize).
pub fn transfer_stress(phonikud_ipa: &str, renikud_ipa: &str) -> String {
    let renikud = normalize_spaces(renikud_ipa);
    match vowel_stress_index(phonikud_ipa) {
        Some(index) => apply_vowel_stress(&renikud, index),
        None => renikud,
    }
}

/// Phonemize one vocalized Hebrew word:
/// 1. Phonikud on the word **with niqqud**
/// 2. optionally Renikud on the **stripped** form
/// 3. merge Phonikud stress onto Renikud IPA
pub fn phonemize_nikud_word(
    word: &str,
    phonikud: &mut dyn NikudPhonemizer,
    renikud: Option<&mut dyn PlainHebrewPhonemizer>,
) -> Result<PhoneticSpan> {
    if !contains_nikud(word) {
        bail!("phonemize_nikud_word expects a niqqud-bearing word, got `{word}`");
    }

    let phonikud_ipa = phonikud.phonemize_nikud(word)?.trim().to_owned();
    if phonikud_ipa.is_empty() {
        bail!("Phonikud returned empty IPA for `{word}`");
    }

    let plain = strip_nikud(word);
    if plain.chars().any(is_hebrew_letter) {
        if let Some(renikud) = renikud {
            let renikud_ipa = renikud.phonemize_plain(&plain)?.trim().to_owned();
            if renikud_ipa.is_empty() {
                bail!("Renikud returned empty IPA for stripped `{plain}`");
            }
            let ipa = transfer_stress(&phonikud_ipa, &renikud_ipa);
            return Ok(PhoneticSpan {
                source: word.to_owned(),
                phonikud_ipa,
                renikud_ipa: Some(renikud_ipa),
                ipa,
            });
        }
    }

    Ok(PhoneticSpan {
        source: word.to_owned(),
        ipa: phonikud_ipa.clone(),
        phonikud_ipa,
        renikud_ipa: None,
    })
}

/// Normalize structured text while preserving ordinary Hebrew words **and**
/// their niqqud.
pub fn normalize_for_speech(text: &str) -> String {
    let heading = Regex::new(r"(?m)^\s{0,3}#{1,6}\s+").expect("valid heading regex");
    let list = Regex::new(r"(?m)(^|\n)\s*(\d{1,2})[.)]\s+").expect("valid list regex");
    let email =
        Regex::new(r"(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}").expect("valid email regex");
    let phone = Regex::new(r"(?:\*\d{2,}|0\d{1,2}-\d{6,8})").expect("valid phone regex");
    let date = Regex::new(r"([0-3]?\d)[/.]([01]?\d)[/.](\d{2}|\d{4})").expect("valid date regex");
    let time = Regex::new(r"([01]?\d|2[0-3]):([0-5]\d)").expect("valid time regex");
    let identifier =
        Regex::new(r"[A-Za-z]+(?:[-_/]?[A-Za-z0-9]+)+").expect("valid identifier regex");

    let mut text = heading.replace_all(text, "").into_owned();
    text = text
        .replace('–', "-")
        .replace('—', "-")
        .replace('‑', "-")
        .replace('[', " ")
        .replace(']', " ")
        .replace('{', " ")
        .replace('}', " ")
        .replace('(', ",")
        .replace(')', ",");
    text = normalize_hebrew_punctuation(&text);
    text = list
        .replace_all(&text, |caps: &Captures| {
            format!(
                "{}{}. ",
                &caps[1],
                list_number(caps[2].parse().unwrap_or(0))
            )
        })
        .into_owned();
    text = email
        .replace_all(&text, |caps: &Captures| expand_email(&caps[0]))
        .into_owned();
    text = phone
        .replace_all(&text, |caps: &Captures| expand_phone(&caps[0]))
        .into_owned();
    text = date
        .replace_all(&text, |caps: &Captures| expand_date(caps))
        .into_owned();
    text = time
        .replace_all(&text, |caps: &Captures| expand_time(caps))
        .into_owned();
    text = identifier
        .replace_all(&text, |caps: &Captures| expand_identifier(&caps[0]))
        .into_owned();
    normalize_punctuation(&text)
}

/// Apply BlueTTS's text normalization before G2P.
///
/// This intentionally keeps `<en>…</en>` spans and `【…】` pacing spans in the
/// output. The G2P and synthesis layers consume those markers later.
pub fn prepare_text_for_synthesis(text: &str, lang: &str) -> String {
    let lang = canonical_lang(lang);
    let mut text = strip_helper_markup(text);
    text = normalize_common_text(&text);
    if lang == "he" {
        text = normalize_hebrew_punctuation(&text);
        text = expand_geresh_loanwords(&text);
        text = expand_dialogue_quotes(&text);
        text = expand_lamed_before_latin(&text);
    }
    text = expand_alphanumeric_codes(&text, &lang);
    text = expand_list_markers(&text, &lang);
    text = expand_plus_sign(&text, &lang);
    if lang == "he" {
        text = expand_phone_numbers(&text);
        text = expand_times(&text);
        text = expand_dates(&text);
    }
    text = expand_percent_symbols(&text, &lang);
    text = expand_ratios(&text, &lang);
    text = expand_numbers(&text, &lang);
    strip_silent_separator_tokens(&text)
}

fn canonical_lang(lang: &str) -> String {
    match lang.to_ascii_lowercase().as_str() {
        "en-us" => "en".to_owned(),
        "ge" => "de".to_owned(),
        value => value.to_owned(),
    }
}

fn strip_helper_markup(text: &str) -> String {
    let block = Regex::new(r"(?is)<lang_list\b[^>]*>.*?</lang_list>").expect("valid regex");
    let tag = Regex::new(r"(?i)</?lang_list\b[^>]*>").expect("valid regex");
    tag.replace_all(&block.replace_all(text, " "), " ")
        .into_owned()
}

fn normalize_common_text(text: &str) -> String {
    let heading = Regex::new(r"(?m)(^|\s)#{1,6}\s*").expect("valid regex");
    let mut text = heading.replace_all(text, "$1").into_owned();
    text = text
        .replace(['(', '[', '{'], ", ")
        .replace([')', ']', '}'], ", ");
    text = normalize_punctuation(&text);
    Regex::new(r"(?i)\banymore\b")
        .expect("valid regex")
        .replace_all(&text, |caps: &Captures| {
            if caps[0].starts_with('A') {
                "Any more"
            } else {
                "any more"
            }
        })
        .into_owned()
}

fn mark_slow_segment(text: impl AsRef<str>) -> String {
    format!("{REF_CODE_MARK_OPEN}{}{REF_CODE_MARK_CLOSE}", text.as_ref())
}

fn expand_geresh_loanwords(text: &str) -> String {
    text.replace("ג'מיני", "<en>Gemini</en>")
        .replace("ג׳מיני", "<en>Gemini</en>")
        .replace("מנג'ר", "<en>Manager</en>")
        .replace("מנג׳ר", "<en>Manager</en>")
}

fn expand_dialogue_quotes(text: &str) -> String {
    let terminal = Regex::new(r#"(?u)(\S)\s*["“„”]\s*$"#).expect("valid regex");
    let opening = Regex::new(r#"(?u)\s*:?\s*["“„”]\s*"#).expect("valid regex");
    let closing = Regex::new(r#"(?u)(\S)\s*["“„”]"#).expect("valid regex");
    let text = terminal.replace_all(text, "$1.").into_owned();
    let text = opening.replace_all(&text, ", ").into_owned();
    closing.replace_all(&text, "$1, ").into_owned()
}

fn expand_lamed_before_latin(text: &str) -> String {
    Regex::new(r"(?u)(^|[^\u{0590}-\u{05ff}])ל\s*[-–—‑]?\s*([A-Za-z0-9])")
        .expect("valid regex")
        .replace_all(text, "$1אל $2")
        .into_owned()
}

fn expand_alphanumeric_codes(text: &str, lang: &str) -> String {
    let token = Regex::new(r"(?i)[a-z0-9]+(?:[-_/][a-z0-9]+)*").expect("valid regex");
    token
        .replace_all(text, |caps: &Captures| {
            let value = &caps[0];
            let letters = value.chars().filter(|c| c.is_ascii_alphabetic()).count();
            let digits = value.chars().filter(|c| c.is_ascii_digit()).count();
            if letters == 0 || digits == 0 || (letters < 2 && digits < 2) {
                return value.to_owned();
            }
            let letter_block = value
                .chars()
                .filter(|c| c.is_ascii_alphabetic())
                .map(|c| c.to_ascii_uppercase().to_string())
                .collect::<Vec<_>>()
                .join(" ");
            let digits = value
                .split(['-', '_', '/'])
                .filter_map(|part| {
                    let digits: String = part.chars().filter(|c| c.is_ascii_digit()).collect();
                    (!digits.is_empty()).then(|| spell_digits_lang(&digits, lang))
                })
                .collect::<Vec<_>>()
                .join(" , ");
            let letters = if letter_block.is_empty() {
                String::new()
            } else {
                format!("<en>{letter_block}</en>")
            };
            mark_slow_segment(format!("{} {} .", letters, digits).trim())
        })
        .into_owned()
}

fn expand_list_markers(text: &str, lang: &str) -> String {
    let list = Regex::new(r"(?m)(^|[^\d/])(\d{1,2})\.\s+").expect("valid regex");
    list.replace_all(text, |caps: &Captures| {
        let n: u8 = caps[2].parse().unwrap_or_default();
        let word = if lang == "he" {
            list_number(n)
        } else {
            number_to_words(u16::from(n), lang)
        };
        format!("{}{}. ", &caps[1], word)
    })
    .into_owned()
}

fn expand_plus_sign(text: &str, lang: &str) -> String {
    let word = match lang {
        "he" => "פלוס",
        "es" => "más",
        "it" => "più",
        _ => "plus",
    };
    Regex::new(r"\s+\+\s+")
        .expect("valid regex")
        .replace_all(text, format!(" {word} "))
        .into_owned()
}

fn expand_phone_numbers(text: &str) -> String {
    let phone = Regex::new(r"(?x)(?:\*\d{2,}|0\d{0,2}-\d{6,8})").expect("valid regex");
    phone
        .replace_all(text, |caps: &Captures| {
            let raw = &caps[0];
            let prefix = if raw.starts_with('*') {
                "כוכבית "
            } else {
                ""
            };
            mark_slow_segment(format!("{prefix}{}", spell_digits(raw)))
        })
        .into_owned()
}

fn expand_times(text: &str) -> String {
    let time = Regex::new(r"([01]?\d|2[0-3]):([0-5]\d)").expect("valid regex");
    time.replace_all(text, |caps: &Captures| {
        let hour = caps[1].parse::<u16>().unwrap_or_default();
        let minute = caps[2].parse::<u16>().unwrap_or_default();
        let spoken = if minute == 0 {
            number_to_words(hour, "he")
        } else {
            format!(
                "{} ו{}",
                number_to_words(hour, "he"),
                number_to_words(minute, "he")
            )
        };
        mark_slow_segment(spoken)
    })
    .into_owned()
}

fn expand_dates(text: &str) -> String {
    let date = Regex::new(r"([0-3]?\d)[/.]([01]?\d)[/.](\d{2}|\d{4})").expect("valid regex");
    date.replace_all(text, |caps: &Captures| {
        let day = caps[1].parse::<u16>().unwrap_or_default();
        let month = caps[2].parse::<usize>().unwrap_or_default();
        let mut year = caps[3].parse::<u16>().unwrap_or_default();
        if caps[3].len() == 2 {
            year += if year < 70 { 2000 } else { 1900 };
        }
        let ordinals = [
            "",
            "לראשון",
            "לשני",
            "לשלישי",
            "לרביעי",
            "לחמישי",
            "לשישי",
            "לשביעי",
            "לשמיני",
            "לתשיעי",
            "לעשירי",
            "לאחד עשר",
            "לשנים עשר",
        ];
        if !(1..=31).contains(&day) || month == 0 || month >= ordinals.len() {
            caps[0].to_owned()
        } else {
            mark_slow_segment(format!(
                "{} {} {}",
                number_to_words(day, "he"),
                ordinals[month],
                number_to_words(year, "he")
            ))
        }
    })
    .into_owned()
}

fn expand_percent_symbols(text: &str, lang: &str) -> String {
    let word = match lang {
        "he" => "אחוז",
        "es" => "por ciento",
        "de" => "Prozent",
        "it" => "per cento",
        _ => "percent",
    };
    Regex::new(r"(\d+(?:[.,]\d+)?)\s*%")
        .expect("valid regex")
        .replace_all(text, format!("$1 {word}"))
        .into_owned()
}

fn expand_ratios(text: &str, lang: &str) -> String {
    let word = match lang {
        "he" => "ל",
        "de" => "zu",
        _ => "to",
    };
    Regex::new(r"(\d+)\s*:\s*(\d+)")
        .expect("valid regex")
        .replace_all(text, format!("$1 {word} $2"))
        .into_owned()
}

fn expand_numbers(text: &str, lang: &str) -> String {
    let numbers = Regex::new(r"\d+(?:[.,]\d+)?").expect("valid regex");
    numbers
        .replace_all(text, |caps: &Captures| {
            let raw = &caps[0];
            if raw.contains(['.', ',']) {
                let separator = if lang == "he" {
                    " נקודה "
                } else {
                    " point "
                };
                raw.replace(['.', ','], separator)
                    .split_whitespace()
                    .map(|piece| {
                        piece
                            .parse::<u16>()
                            .ok()
                            .map_or_else(|| piece.to_owned(), |n| number_to_words(n, lang))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                raw.parse::<u16>()
                    .map(|n| number_to_words(n, lang))
                    .unwrap_or_else(|_| spell_digits_lang(raw, lang))
            }
        })
        .into_owned()
}

fn strip_silent_separator_tokens(text: &str) -> String {
    let text = Regex::new(r"(?u)([\u{0590}-\u{05ff}])[-–—‑]+([A-Za-z0-9])")
        .expect("valid regex")
        .replace_all(text, "$1 $2")
        .into_owned();
    let text = Regex::new(r"(?u)([A-Za-z0-9])[-–—‑]+([\u{0590}-\u{05ff}])")
        .expect("valid regex")
        .replace_all(&text, "$1 $2")
        .into_owned();
    let text = Regex::new(r"\s*[-–—‑]+\s*")
        .expect("valid regex")
        .replace_all(&text, " ")
        .into_owned();
    let text = Regex::new(r"\s*:+\s*")
        .expect("valid regex")
        .replace_all(&text, " ")
        .into_owned();
    Regex::new(r"\s+")
        .expect("valid regex")
        .replace_all(&text, " ")
        .trim()
        .to_owned()
}

/// Prepare one sentence.
///
/// - `text` keeps niqqud (never stripped for text-native models).
/// - `phonetic_text` replaces only niqqud-bearing words with final IPA via
///   Phonikud (+ optional Renikud stress merge).
pub fn prepare_text(
    text: &str,
    mut phonikud: Option<&mut dyn NikudPhonemizer>,
    renikud: Option<&mut dyn PlainHebrewPhonemizer>,
    phonetic_mode: bool,
) -> Result<PreparedText> {
    let normalized = normalize_for_speech(text);
    if !phonetic_mode || !contains_nikud(&normalized) {
        return Ok(PreparedText {
            original: text.to_owned(),
            text: normalized.clone(),
            phonetic_text: normalized,
            phonetic_spans: Vec::new(),
        });
    }

    let phonikud = phonikud
        .as_deref_mut()
        .ok_or_else(|| anyhow::anyhow!("niqqud input requires a Phonikud-compatible phonemizer"))?;
    let (phonetic_text, phonetic_spans) = replace_nikud_words(&normalized, phonikud, renikud)?;
    Ok(PreparedText {
        original: text.to_owned(),
        text: normalized,
        phonetic_text,
        phonetic_spans,
    })
}

fn replace_nikud_words(
    text: &str,
    phonikud: &mut dyn NikudPhonemizer,
    renikud: Option<&mut dyn PlainHebrewPhonemizer>,
) -> Result<(String, Vec<PhoneticSpan>)> {
    match renikud {
        Some(engine) => replace_nikud_words_inner(text, phonikud, Some(engine)),
        None => replace_nikud_words_inner(text, phonikud, None),
    }
}

fn replace_nikud_words_inner(
    text: &str,
    phonikud: &mut dyn NikudPhonemizer,
    mut renikud: Option<&mut dyn PlainHebrewPhonemizer>,
) -> Result<(String, Vec<PhoneticSpan>)> {
    let mut output = String::with_capacity(text.len());
    let mut spans = Vec::new();
    let mut word = String::new();

    for character in text.chars() {
        if is_hebrew_word_character(character) {
            word.push(character);
            continue;
        }
        if !word.is_empty() {
            if contains_nikud(&word) {
                let span = match renikud.as_mut() {
                    Some(engine) => phonemize_nikud_word(&word, phonikud, Some(&mut **engine))?,
                    None => phonemize_nikud_word(&word, phonikud, None)?,
                };
                output.push_str(&span.ipa);
                spans.push(span);
            } else {
                output.push_str(&word);
            }
            word.clear();
        }
        output.push(character);
    }

    if !word.is_empty() {
        if contains_nikud(&word) {
            let span = match renikud.as_mut() {
                Some(engine) => phonemize_nikud_word(&word, phonikud, Some(&mut **engine))?,
                None => phonemize_nikud_word(&word, phonikud, None)?,
            };
            output.push_str(&span.ipa);
            spans.push(span);
        } else {
            output.push_str(&word);
        }
    }
    Ok((output, spans))
}

fn apply_vowel_stress_last(plain: &str) -> String {
    let vowel_count = plain.chars().filter(|&c| is_ipa_vowel(c)).count();
    if vowel_count == 0 {
        return plain.to_owned();
    }
    apply_vowel_stress(plain, vowel_count - 1)
}

fn is_ipa_vowel(character: char) -> bool {
    matches!(
        character,
        'a' | 'e'
            | 'i'
            | 'o'
            | 'u'
            | 'ə'
            | 'ɑ'
            | 'ɛ'
            | 'ɔ'
            | 'ɪ'
            | 'ʊ'
            | 'ɐ'
            | 'æ'
            | 'ø'
            | 'œ'
            | 'ɨ'
            | 'ʉ'
    )
}

fn is_nikud(character: char) -> bool {
    // Hebrew points + cantillation, including Phonikud's Ole (hatama) / Meteg.
    ('\u{0591}'..='\u{05c7}').contains(&character)
}

fn normalize_hebrew_punctuation(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut output = String::with_capacity(text.len());
    for (index, character) in chars.iter().copied().enumerate() {
        let previous = index.checked_sub(1).and_then(|i| chars.get(i)).copied();
        let next = chars.get(index + 1).copied();
        if matches!(character, '"' | '״')
            && previous.is_some_and(is_hebrew_letter)
            && next.is_some_and(is_hebrew_letter)
        {
            continue;
        }
        if character == '\''
            && matches!(previous, Some('ג' | 'צ' | 'ז'))
            && next.is_some_and(is_hebrew_letter)
        {
            output.push('׳');
            continue;
        }
        if character == '-'
            && previous.is_some_and(is_hebrew_letter)
            && next.is_some_and(is_hebrew_letter)
        {
            continue;
        }
        output.push(character);
    }
    output
}

fn normalize_punctuation(text: &str) -> String {
    let text = text.replace('…', ",");
    let dots = Regex::new(r"\.{2,}").expect("valid dot regex");
    let bangs = Regex::new(r"!{2,}").expect("valid exclamation regex");
    let questions = Regex::new(r"\?{2,}").expect("valid question regex");
    let commas = Regex::new(r",{2,}").expect("valid comma regex");
    let whitespace = Regex::new(r"\s+").expect("valid whitespace regex");
    let text = dots.replace_all(&text, ",");
    let text = bangs.replace_all(&text, "!");
    let text = questions.replace_all(&text, "?");
    let text = commas.replace_all(&text, ",");
    whitespace.replace_all(&text, " ").trim().to_owned()
}

fn expand_email(email: &str) -> String {
    let (local, domain) = email.split_once('@').expect("email regex contains @");
    let local = local.replace(['.', '_'], " dot ").replace('-', " dash ");
    let domain = domain
        .split('.')
        .map(|part| {
            if part.len() <= 2 && part.chars().all(|c| c.is_ascii_alphabetic()) {
                part.chars()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                part.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" dot ");
    format!("{local} at {domain}")
}

fn expand_phone(phone: &str) -> String {
    let prefix = if phone.starts_with('*') {
        "כוכבית "
    } else {
        ""
    };
    format!("{prefix}{}.", spell_digits(phone))
}

fn expand_time(caps: &Captures) -> String {
    format!(
        "שעה {} ו{}",
        spell_digits(&format!("{:0>2}", &caps[1])),
        spell_digits(&caps[2])
    )
}

fn expand_date(caps: &Captures) -> String {
    let day = caps[1].parse::<u8>().unwrap_or(0);
    let month = caps[2].parse::<u8>().unwrap_or(0);
    let Some(month_name) = month_name(month) else {
        return caps[0].to_owned();
    };
    if !(1..=31).contains(&day) {
        return caps[0].to_owned();
    }
    format!(
        "{} ב{month_name} {}",
        spell_digits(&caps[1]),
        spell_digits(&caps[3])
    )
}

fn expand_identifier(token: &str) -> String {
    let letters: Vec<_> = token
        .chars()
        .filter(|character| character.is_ascii_alphabetic())
        .map(|character| character.to_ascii_uppercase())
        .collect();
    if letters.is_empty() || !token.chars().any(|character| character.is_ascii_digit()) {
        return token.to_owned();
    }
    let digit_groups = token
        .split(['-', '_', '/'])
        .filter(|part| part.chars().any(|character| character.is_ascii_digit()))
        .map(spell_digits)
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>();
    let letters = letters
        .into_iter()
        .map(|letter| letter.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    format!("{letters} {}.", digit_groups.join(", "))
}

fn spell_digits(text: &str) -> String {
    spell_digits_lang(text, "he")
}

fn spell_digits_lang(text: &str, lang: &str) -> String {
    let words = match lang {
        "he" => [
            "אפס",
            "אחת",
            "שתיים",
            "שלוש",
            "ארבע",
            "חמש",
            "שש",
            "שבע",
            "שמונה",
            "תשע",
        ],
        _ => [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        ],
    };
    text.chars()
        .filter_map(|digit| digit.to_digit(10).map(|digit| words[digit as usize]))
        .collect::<Vec<_>>()
        .join(" ")
}

fn number_to_words(number: u16, lang: &str) -> String {
    if lang != "he" {
        return match number {
            0 => "zero".to_owned(),
            1 => "one".to_owned(),
            2 => "two".to_owned(),
            3 => "three".to_owned(),
            4 => "four".to_owned(),
            5 => "five".to_owned(),
            6 => "six".to_owned(),
            7 => "seven".to_owned(),
            8 => "eight".to_owned(),
            9 => "nine".to_owned(),
            10 => "ten".to_owned(),
            11 => "eleven".to_owned(),
            12 => "twelve".to_owned(),
            13 => "thirteen".to_owned(),
            14 => "fourteen".to_owned(),
            15 => "fifteen".to_owned(),
            16 => "sixteen".to_owned(),
            17 => "seventeen".to_owned(),
            18 => "eighteen".to_owned(),
            19 => "nineteen".to_owned(),
            20 => "twenty".to_owned(),
            _ => spell_digits_lang(&number.to_string(), lang),
        };
    }
    let units = [
        "אפס",
        "אחת",
        "שתיים",
        "שלוש",
        "ארבע",
        "חמש",
        "שש",
        "שבע",
        "שמונה",
        "תשע",
        "עשר",
        "אחת עשרה",
        "שתים עשרה",
        "שלוש עשרה",
        "ארבע עשרה",
        "חמש עשרה",
        "שש עשרה",
        "שבע עשרה",
        "שמונה עשרה",
        "תשע עשרה",
    ];
    if number < 20 {
        return units[number as usize].to_owned();
    }
    let tens = [
        "",
        "",
        "עשרים",
        "שלושים",
        "ארבעים",
        "חמישים",
        "שישים",
        "שבעים",
        "שמונים",
        "תשעים",
    ];
    if number < 100 {
        let ten = number / 10;
        let remainder = number % 10;
        return if remainder == 0 {
            tens[ten as usize].to_owned()
        } else {
            format!("{} ו{}", tens[ten as usize], units[remainder as usize])
        };
    }
    if number < 1_000 {
        let hundred = number / 100;
        let remainder = number % 100;
        let head = match hundred {
            1 => "מאה".to_owned(),
            2 => "מאתיים".to_owned(),
            n => format!("{} מאות", number_to_words(n, "he")),
        };
        return if remainder == 0 {
            head
        } else {
            format!("{head} {}", number_to_words(remainder, "he"))
        };
    }
    if number < 10_000 {
        let thousand = number / 1_000;
        let remainder = number % 1_000;
        let head = if thousand == 1 {
            "אלף".to_owned()
        } else {
            format!("{} אלף", number_to_words(thousand, "he"))
        };
        return if remainder == 0 {
            head
        } else {
            format!("{head} {}", number_to_words(remainder, "he"))
        };
    }
    spell_digits_lang(&number.to_string(), "he")
}

fn list_number(number: u8) -> String {
    match number {
        1 => "אחד".to_owned(),
        2 => "שתיים".to_owned(),
        3 => "שלוש".to_owned(),
        4 => "ארבע".to_owned(),
        5 => "חמש".to_owned(),
        6 => "שש".to_owned(),
        7 => "שבע".to_owned(),
        8 => "שמונה".to_owned(),
        9 => "תשע".to_owned(),
        10 => "עשר".to_owned(),
        _ => spell_digits(&number.to_string()),
    }
}

fn month_name(month: u8) -> Option<&'static str> {
    Some(match month {
        1 => "ינואר",
        2 => "פברואר",
        3 => "מרץ",
        4 => "אפריל",
        5 => "מאי",
        6 => "יוני",
        7 => "יולי",
        8 => "אוגוסט",
        9 => "ספטמבר",
        10 => "אוקטובר",
        11 => "נובמבר",
        12 => "דצמבר",
        _ => return None,
    })
}

pub fn is_hebrew_letter(character: char) -> bool {
    ('\u{05d0}'..='\u{05ea}').contains(&character)
}

fn is_hebrew_word_character(character: char) -> bool {
    is_hebrew_letter(character) || is_nikud(character) || matches!(character, '׳' | '״' | '|')
}

fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_nikud_in_text_path() {
        let mut phonikud = |word: &str| Ok(format!("ipa:{word}"));
        let prepared = prepare_text("הַמְּנוֹרָה מאירה.", Some(&mut phonikud), None, true).unwrap();
        assert!(contains_nikud(&prepared.text));
        assert_eq!(prepared.for_input(InputMode::Text), "הַמְּנוֹרָה מאירה.");
        assert_eq!(prepared.phonetic_text, "ipa:הַמְּנוֹרָה מאירה.");
    }

    #[test]
    fn transfers_phonikud_stress_onto_renikud_ipa() {
        // Phonikud: stress on 2nd vowel (o). Renikud: different quality, no stress.
        let merged = transfer_stress("ʃalˈom", "ʃalom");
        assert_eq!(merged, "ʃalˈom");

        let merged = transfer_stress("haˈir", "heir");
        assert_eq!(merged, "heˈir");
    }

    #[test]
    fn hybrid_word_uses_renikud_base_and_phonikud_stress() {
        let mut phonikud = |_word: &str| Ok("menˈora".to_owned());
        let mut renikud = |plain: &str| {
            assert!(!contains_nikud(plain));
            Ok("menora".to_owned())
        };
        let span = phonemize_nikud_word("מְנוֹרָה", &mut phonikud, Some(&mut renikud)).unwrap();
        assert_eq!(span.phonikud_ipa, "menˈora");
        assert_eq!(span.renikud_ipa.as_deref(), Some("menora"));
        assert_eq!(span.ipa, "menˈora");
    }

    #[test]
    fn strip_nikud_preserves_letters() {
        assert_eq!(strip_nikud("הַמְּנוֹרָה"), "המנורה");
    }

    #[test]
    fn normalizes_reference_codes_and_phone_numbers() {
        let text = normalize_for_speech("מספר TKT-90254, טלפון 03-5551234.");
        assert!(text.contains("T K T"));
        assert!(text.contains("אפס שלוש חמש חמש חמש"));
    }

    #[test]
    fn prepares_hebrew_reference_segments_and_structured_values() {
        let text = prepare_text_for_synthesis(
            "הג'מיני הגיע ל-GPU ב-08:15, קוד TKT-90254 הוא 50% הצלחה.",
            "he",
        );
        assert!(text.contains("<en>Gemini</en>"));
        assert!(text.contains("אל GPU"));
        assert!(text.contains(REF_CODE_MARK_OPEN));
        assert!(text.contains("אחוז"));
    }

    #[test]
    fn splits_slow_reference_segments() {
        let segments = split_prepared_by_reference_codes("רגיל 【אחת שתיים】.");
        assert_eq!(segments.len(), 2);
        assert!(!segments[0].is_reference_code);
        assert!(segments[1].is_reference_code);
        assert_eq!(segments[1].text, "אחת שתיים.");
    }

    #[test]
    fn expands_dialogue_and_list_markers() {
        let text = prepare_text_for_synthesis("1. אמר \"שלום\"", "he");
        assert!(
            text.starts_with("אחד."),
            "unexpected prepared text: {text:?}"
        );
        assert!(
            text.contains("אמר, שלום."),
            "unexpected prepared text: {text:?}"
        );
    }
}
