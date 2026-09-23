use std::path::Path;

use anyhow::{Result, anyhow, bail};
use espeak_rs::text_to_phonemes;
use ort::session::Session;
use regex::Regex;
use renikud_plus_rs::lexicon::{ForceLexicon, OnInvalidEntry};
use renikud_plus_rs::{G2P, G2PConfig, NiqqudMode, Options};

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
        g2p.vocalize_with(
            text,
            &Options {
                with_stress,
                // The text goes back to the user to edit and then returns
                // through `g2p`, so it has to read back as what it came from.
                round_trip: true,
                ..Options::speakers(self.speaker, self.target_speaker)
            },
        )
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
        wrap_segments(segments)
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

/// Join language segments into one tagged string. A change of language is
/// only a word break: the text's own punctuation is the only pause, so a
/// segment that starts with punctuation, or is nothing but punctuation (the
/// period after a Latin word), attaches to what came before it.
fn wrap_segments(segments: Vec<(Language, String)>) -> String {
    let mut merged: Vec<(Language, String)> = Vec::new();
    for (language, ipa) in segments {
        let ipa = normalize_spaces(&ipa);
        if ipa.is_empty() {
            continue;
        }
        match merged.last_mut() {
            Some(last) if !ipa.chars().any(char::is_alphanumeric) => last.1.push_str(&ipa),
            Some(last) if last.0 == language => {
                if !starts_with_punctuation(&ipa) {
                    last.1.push(' ');
                }
                last.1.push_str(&ipa);
            }
            _ => merged.push((language, ipa)),
        }
    }
    let mut result = String::new();
    for (language, ipa) in merged {
        if !result.is_empty() && !starts_with_punctuation(&ipa) {
            result.push(' ');
        }
        let code = language.code();
        result.push_str(&format!("<{code}>{ipa}</{code}>"));
    }
    result
}

fn starts_with_punctuation(text: &str) -> bool {
    text.starts_with(['.', ',', '!', '?', ';', ':', ')', ']', '}'])
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
mod espeak_tests {
    use super::*;

    /// A change of language is a word break, not a pause: no comma between
    /// segments, and a trailing period attaches to the last word.
    #[test]
    fn joins_language_segments_with_a_space() {
        let he = |ipa: &str| (Language::Hebrew, ipa.to_owned());
        let en = |ipa: &str| (Language::English, ipa.to_owned());
        for (segments, expected) in [
            (
                vec![he("pʁatˈim mˈeha"), en("pˌiːdˌiːˈɛf"), he("hametsuʁˈaf.")],
                "<he>pʁatˈim mˈeha</he> <en>pˌiːdˌiːˈɛf</en> <he>hametsuʁˈaf.</he>",
            ),
            (
                vec![he("kitˈa ʔˈalef, mˈeha"), en("pˌiːdˌiːˈɛf"), he(".")],
                "<he>kitˈa ʔˈalef, mˈeha</he> <en>pˌiːdˌiːˈɛf.</en>",
            ),
            (
                vec![en("pˌiːdˌiːˈɛf"), he(", ʃalˈom")],
                "<en>pˌiːdˌiːˈɛf</en><he>, ʃalˈom</he>",
            ),
            (vec![he("ʃalˈom"), he(" ʔolˈam ")], "<he>ʃalˈom ʔolˈam</he>"),
        ] {
            assert_eq!(wrap_segments(segments), expected);
        }
    }

    #[test]
    fn mixed_latin_and_punctuation_have_no_stray_commas() {
        // Without a Hebrew model the Hebrew segments pass through as-is, which
        // is enough to see how Latin segments and punctuation are joined.
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::English).unwrap();
        let ipa = phonemizer
            .phonemize_prepared("hello PDF, world.", Language::Hebrew)
            .unwrap();
        assert_eq!(ipa, "<en>həlˈoʊ pˌiːdˌiːˈɛf, wˈɜːld.</en>");
    }

    /// Issue #12: eSpeak-backed languages came back as one run-on token, with
    /// no word breaks and no punctuation for the model to pause on.
    #[test]
    fn keeps_word_breaks_and_punctuation() {
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::English).unwrap();
        for (language, text, expected) in [
            (
                Language::English,
                "Hello, world. Goodbye!",
                "<en>həlˈoʊ, wˈɜːld. ɡʊdbˈaɪ!</en>",
            ),
            (
                Language::Spanish,
                "Hola, mundo. ¡Adiós!",
                "<es>ˈola, mˈundo. ¡aðjˈos!</es>",
            ),
            (
                Language::German,
                "Hallo, Welt. Tschüss!",
                "<de>hˈaloː, vˈɛlt. tʃˈʏs!</de>",
            ),
            (
                Language::Italian,
                "Ciao, mondo. Arrivederci!",
                "<it>tʃˈao, mˈondo. arɾivedˈɛrtʃɪ!</it>",
            ),
        ] {
            let ipa = phonemizer.g2p(text, language).unwrap();
            assert_eq!(ipa, expected, "{}", language.code());
        }
    }

    #[test]
    fn keeps_question_marks_and_sentence_breaks() {
        let mut phonemizer = Phonemizer::with_language(None::<&str>, Language::English).unwrap();
        let ipa = phonemizer
            .g2p(
                "How are you today? I am fine, thank you.",
                Language::English,
            )
            .unwrap();
        assert!(ipa.contains("? "), "{ipa}");
        assert!(ipa.contains(", "), "{ipa}");
        assert!(ipa.ends_with(".</en>"), "{ipa}");
        // One word break per word, bar the pairs espeak runs together.
        assert!(
            ipa.split_whitespace().count() >= 7,
            "{ipa} has too few word breaks"
        );
    }
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
            // The points pin the reading, stress included, so phonemizing the
            // pointed text gives back what the plain text gave.
            assert_eq!(
                round_trip, original,
                "{text} -> {nikud}: {original} vs {round_trip}"
            );
        }
    }
}
