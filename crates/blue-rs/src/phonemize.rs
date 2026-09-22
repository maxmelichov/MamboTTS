use std::path::Path;

use anyhow::{Result, anyhow, bail};
use espeak_rs::text_to_phonemes;
use ort::session::Session;
use regex::Regex;
use renikud_plus_rs::lexicon::{ForceLexicon, OnInvalidEntry};
use renikud_plus_rs::{G2P, G2PConfig, NiqqudMode};

use crate::handling::prepare_text_for_synthesis;

/// RenikudPlus as MamboTTS runs it: niqqud in the input is read as evidence
/// rather than dropped, because the app lets people type and edit the points
/// and expects them honored.
fn hebrew_config() -> G2PConfig {
    G2PConfig {
        niqqud: NiqqudMode::Use,
        ..G2PConfig::default()
    }
}

/// Languages supported by the BlueTTS model.
///
/// Codes:
/// - `he` Hebrew, via RenikudPlus when Hebrew characters are present.
/// - `en` English, via eSpeak voice `en-us`.
/// - `es` Spanish, via eSpeak voice `es`.
/// - `de` German, via eSpeak voice `de`.
/// - `it` Italian, via eSpeak voice `it`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Language {
    Hebrew,
    English,
    Spanish,
    German,
    Italian,
}

impl Language {
    pub fn code(self) -> &'static str {
        match self {
            Self::Hebrew => "he",
            Self::English => "en",
            Self::Spanish => "es",
            Self::German => "de",
            Self::Italian => "it",
        }
    }

    pub fn espeak_voice(self) -> Option<&'static str> {
        match self {
            Self::Hebrew => None,
            Self::English => Some("en-us"),
            Self::Spanish => Some("es"),
            Self::German => Some("de"),
            Self::Italian => Some("it"),
        }
    }
}

impl TryFrom<&str> for Language {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "he" => Ok(Self::Hebrew),
            "en" | "en-us" => Ok(Self::English),
            "es" => Ok(Self::Spanish),
            "de" | "ge" => Ok(Self::German),
            "it" => Ok(Self::Italian),
            _ => bail!("unsupported language code `{value}`; expected he, en, es, de, or it"),
        }
    }
}

pub struct Phonemizer {
    hebrew: Option<G2P>,
    language: Language,
    mixed_re: Regex,
    inline_tag_re: Regex,
    /// RenikudPlus speaker conditioning (0=unknown, 1=male, 2=female).
    speaker: u8,
    target_speaker: u8,
}

impl Phonemizer {
    /// Create a phonemizer with Hebrew as the default language.
    ///
    /// Supported language codes are `he`, `en`, `es`, `de`, and `it`. Use
    /// [`Self::with_language`] or [`Self::phonemize_lang`] to select one.
    ///
    /// `renikud_model` is only required when phonemizing Hebrew text (RenikudPlus ONNX).
    pub fn new(renikud_model: Option<impl AsRef<Path>>) -> Result<Self> {
        Self::with_language(renikud_model, Language::Hebrew)
    }

    /// Create a phonemizer with an explicit default language.
    ///
    /// Supported model language codes are `he`, `en`, `es`, `de`, and `it`.
    /// Non-Hebrew languages use eSpeak. Hebrew uses RenikudPlus when Hebrew
    /// characters are present.
    pub fn with_language(
        renikud_model: Option<impl AsRef<Path>>,
        language: Language,
    ) -> Result<Self> {
        let hebrew = match renikud_model {
            Some(path) => Some(G2P::with_config(
                path.as_ref().to_string_lossy().as_ref(),
                hebrew_config(),
            )?),
            None => None,
        };
        Ok(Self {
            hebrew,
            language,
            mixed_re: Regex::new(
                r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\d+[A-Za-z]+|[A-Za-z]+(?:[.'’\-][A-Za-z0-9]+)*",
            )?,
            inline_tag_re: Regex::new(
                r"(?is)<(en|en-us|he|es|de|ge|it)>(.*?)</(?:en|en-us|he|es|de|ge|it)>",
            )?,
            speaker: 0,
            target_speaker: 0,
        })
    }

    /// Create a phonemizer from embedded RenikudPlus ONNX bytes.
    ///
    /// Supported model language codes are `he`, `en`, `es`, `de`, and `it`.
    /// This is useful for self-contained binaries built with `include_bytes!`.
    pub fn from_renikud_bytes(bytes: &[u8], language: Language) -> Result<Self> {
        let builder = Session::builder()?;
        let session = builder.commit_from_memory(bytes)?;
        Ok(Self {
            hebrew: Some(G2P::from_session_with_config(session, hebrew_config())?),
            language,
            mixed_re: Regex::new(
                r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\d+[A-Za-z]+|[A-Za-z]+(?:[.'’\-][A-Za-z0-9]+)*",
            )?,
            inline_tag_re: Regex::new(
                r"(?is)<(en|en-us|he|es|de|ge|it)>(.*?)</(?:en|en-us|he|es|de|ge|it)>",
            )?,
            speaker: 0,
            target_speaker: 0,
        })
    }

    /// Set RenikudPlus speaker conditioning (0=unknown, 1=male, 2=female).
    pub fn set_speakers(&mut self, speaker: u8, target_speaker: u8) {
        self.speaker = speaker;
        self.target_speaker = target_speaker;
    }

    /// Add niqqud to the Hebrew in `text` with RenikudPlus.
    ///
    /// Non-Hebrew text and punctuation are passed through; digits are read as
    /// Hebrew number words first, as they are before synthesis. With
    /// `with_stress`, each stressed syllable also gets the hatama (U+05AB),
    /// which [`Self::g2p`] reads back, so the pointed text phonemizes the same
    /// as the original. The whole text is read in one pass, as [`Self::g2p`]
    /// reads it, because the model's reading of a word depends on its context.
    pub fn diacritize(&mut self, text: &str, with_stress: bool) -> Result<String> {
        if !contains_hebrew(text) {
            return Ok(text.to_owned());
        }
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("Hebrew diacritization needs a RenikudPlus model path");
        };
        g2p.diacritize(text, self.speaker, self.target_speaker, with_stress)
    }

    /// Install a force lexicon over the Hebrew G2P: a `surface<TAB>IPA` file
    /// whose entries always win where they match.
    ///
    /// `align_cells` is `""` for the standard aligner or `"extended"` to widen
    /// the validation the entries have to pass.
    pub fn set_hebrew_lexicon(&mut self, path: impl AsRef<Path>, align_cells: &str) -> Result<()> {
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("a Hebrew force lexicon needs a RenikudPlus model path");
        };
        let cells = G2P::align_cells(align_cells)?;
        let lexicon = ForceLexicon::from_tsv(path, &cells, OnInvalidEntry::Report)?;
        for (surface, ipa, reason) in &lexicon.rejected {
            eprintln!("WARNING: lexicon entry {surface:?} -> {ipa:?} rejected: {reason}");
        }
        g2p.set_lexicon(Some(lexicon));
        Ok(())
    }

    /// Phonemize text using the default language.
    ///
    /// Supported model language codes are `he`, `en`, `es`, `de`, and `it`.
    /// For mixed Hebrew/Latin input, Hebrew spans use RenikudPlus and Latin spans
    /// use the default language's eSpeak voice, falling back to English for
    /// Hebrew default.
    pub fn phonemize(&mut self, text: &str) -> Result<String> {
        self.phonemize_lang(text, self.language)
    }

    /// Prepare raw text and return BlueTTS-ready, language-tagged IPA.
    pub fn g2p(&mut self, text: &str, language: Language) -> Result<String> {
        let prepared = prepare_text_for_synthesis(text, language.code());
        self.phonemize_prepared(&prepared, language)
    }

    /// Phonemize text using an explicit supported model language.
    ///
    /// Supported model language codes are `he`, `en`, `es`, `de`, and `it`.
    pub fn phonemize_lang(&mut self, text: &str, language: Language) -> Result<String> {
        self.g2p(text, language)
    }

    fn phonemize_prepared(&mut self, text: &str, language: Language) -> Result<String> {
        if !self.inline_tag_re.is_match(text) {
            let segments = self.phonemize_segments(text, language)?;
            return Ok(self.wrap_segments(segments));
        }

        let mut segments = Vec::new();
        let mut last = 0;
        let tags: Vec<(usize, usize, Language, String)> = self
            .inline_tag_re
            .captures_iter(text)
            .map(|caps| {
                let all = caps.get(0).expect("full tag match");
                let language = Language::try_from(caps.get(1).expect("tag language").as_str())?;
                Ok((
                    all.start(),
                    all.end(),
                    language,
                    caps.get(2).expect("tag content").as_str().to_owned(),
                ))
            })
            .collect::<Result<_>>()?;
        for (start, end, tagged_language, content) in tags {
            if start > last {
                segments.extend(self.phonemize_segments(&text[last..start], language)?);
            }
            segments.extend(self.phonemize_segments(&content, tagged_language)?);
            last = end;
        }
        if last < text.len() {
            segments.extend(self.phonemize_segments(&text[last..], language)?);
        }
        Ok(self.wrap_segments(segments))
    }

    fn phonemize_segments(
        &mut self,
        text: &str,
        language: Language,
    ) -> Result<Vec<(Language, String)>> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }
        if language != Language::Hebrew && !contains_hebrew(text) {
            let ipa = self.phonemize_espeak(text, language)?;
            return Ok((!ipa.is_empty())
                .then_some((language, ipa))
                .into_iter()
                .collect());
        }
        if language != Language::Hebrew || !contains_latin_or_digit(text) {
            let ipa = self.phonemize_renikud(text)?;
            return Ok((!ipa.is_empty())
                .then_some((language, ipa))
                .into_iter()
                .collect());
        }

        let spans: Vec<(usize, usize)> = self
            .mixed_re
            .find_iter(text)
            .map(|m| (m.start(), m.end()))
            .collect();
        let mut result = Vec::new();
        let mut last = 0;
        for (start, end) in spans {
            if start > last {
                self.push_segment(&mut result, &text[last..start], Language::Hebrew)?;
            }
            let latin = &text[start..end];
            let latin = if is_email(latin) {
                email_to_spoken_english(latin)
            } else {
                latin.to_owned()
            };
            self.push_segment(&mut result, &latin, Language::English)?;
            last = end;
        }
        if last < text.len() {
            self.push_segment(&mut result, &text[last..], Language::Hebrew)?;
        }
        Ok(result)
    }

    fn push_segment(
        &mut self,
        output: &mut Vec<(Language, String)>,
        text: &str,
        language: Language,
    ) -> Result<()> {
        let ipa = if language == Language::Hebrew {
            self.phonemize_renikud(text)?
        } else {
            self.phonemize_espeak(text, language)?
        };
        if !ipa.trim().is_empty() {
            output.push((language, ipa));
        }
        Ok(())
    }

    fn wrap_segments(&self, segments: Vec<(Language, String)>) -> String {
        let mut result = String::new();
        let mut previous = None;
        for (language, ipa) in segments {
            if let Some(previous) = previous {
                if previous != language {
                    result.push_str(" , ");
                } else if !result.is_empty() {
                    result.push(' ');
                }
            }
            result.push_str(&format!(
                "<{}>{}</{}>",
                language.code(),
                ipa.trim(),
                language.code()
            ));
            previous = Some(language);
        }
        normalize_spaces(&result)
    }

    fn phonemize_espeak(&self, text: &str, language: Language) -> Result<String> {
        let voice = language
            .espeak_voice()
            .ok_or_else(|| anyhow!("language `{}` does not use eSpeak", language.code()))?;
        Ok(text_to_phonemes(text, voice, None)
            .map_err(|e| anyhow!("{e}"))?
            .join(" "))
    }

    fn phonemize_renikud(&mut self, text: &str) -> Result<String> {
        if !contains_hebrew(text) {
            return Ok(text.to_owned());
        }
        let Some(g2p) = self.hebrew.as_mut() else {
            bail!("Hebrew phonemization needs a RenikudPlus model path");
        };
        let speaker = self.speaker;
        let target_speaker = self.target_speaker;
        g2p.phonemize(text, speaker, target_speaker)
    }
}

pub fn phonemize(text: &str, renikud_model: Option<impl AsRef<Path>>) -> Result<String> {
    let mut phonemizer = Phonemizer::new(renikud_model)?;
    phonemizer.phonemize(text)
}

fn contains_hebrew(text: &str) -> bool {
    text.chars().any(|c| ('\u{0590}'..='\u{05ff}').contains(&c))
}

fn normalize_spaces(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn contains_latin_or_digit(text: &str) -> bool {
    text.chars().any(|c| c.is_ascii_alphanumeric())
}

fn is_email(text: &str) -> bool {
    Regex::new(r"(?i)^[A-Z0-9._%+\-]+@[A-Z0-9.-]+\.[A-Z]{2,}$")
        .expect("valid email regex")
        .is_match(text)
}

fn email_to_spoken_english(email: &str) -> String {
    let (local, domain) = email.split_once('@').unwrap_or((email, ""));
    let local = local
        .replace(['.', '_'], " dot ")
        .replace('-', " dash ")
        .replace('+', " plus ");
    let domain = domain
        .split('.')
        .filter(|part| !part.is_empty())
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

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    #[ignore = "requires MAMBOTTS_RENIKUD_PATH pointing to renikud-plus.onnx"]
    fn user_nikud_with_real_model() {
        let model = std::env::var("MAMBOTTS_RENIKUD_PATH").expect("set MAMBOTTS_RENIKUD_PATH");
        let mut phonemizer = Phonemizer::new(Some(model)).unwrap();
        for (text, expected) in [("אֶן", "en"), ("קֶלְוִין", "kelvin"), ("לכן תשובה ב׳", "bet")]
        {
            let ipa = phonemizer.g2p(text, Language::Hebrew).unwrap();
            println!("{text}: {ipa}");
            assert!(ipa.replace('ˈ', "").contains(expected), "{text}: {ipa}");
        }
        let plain = phonemizer.g2p("שלום", Language::Hebrew).unwrap();
        let mixed = phonemizer.g2p("אֶן שלום קֶלְוִין", Language::Hebrew).unwrap();
        println!("plain: {plain}; mixed: {mixed}");
        assert!(mixed.replace('ˈ', "").contains("en"));
        assert!(mixed.replace('ˈ', "").contains("kelvin"));
        assert!(!mixed.chars().any(|c| ('א'..='ת').contains(&c)));
        assert!(mixed.contains(plain.trim_start_matches("<he>").trim_end_matches("</he>")));
        let paused = phonemizer
            .g2p("לכן תשובה ב׳: אֶן", Language::Hebrew)
            .unwrap();
        assert!(paused.contains('.'), "{paused}");

        let text = "שבת שלום, hello!\nספר";
        let nikud = phonemizer.diacritize(text, true).unwrap();
        println!("diacritized: {nikud}");
        assert!(nikud.contains(", hello!\n"), "{nikud}");
        for text in [
            text,
            "צה\"ל הודיע שתשובה ב׳ נכונה.",
            "הוא עבר ל-GPU חדש עם 12 ליבות.",
            "ג׳אז וצ׳יפס, רוח ותקווה.",
        ] {
            let nikud = phonemizer.diacritize(text, true).unwrap();
            let original = phonemizer.g2p(text, Language::Hebrew).unwrap();
            let round_trip = phonemizer.g2p(&nikud, Language::Hebrew).unwrap();
            println!("{text}\n  {nikud}\n  {original}\n  {round_trip}");
            // The points pin the reading, so what the pointed text phonemizes
            // to has the same syllable count and the same stressed syllables.
            assert_eq!(
                round_trip.matches('ˈ').count(),
                original.matches('ˈ').count(),
                "{text} -> {nikud}: {original} vs {round_trip}"
            );
        }
    }
}
