use std::{collections::HashMap, path::PathBuf, sync::OnceLock};

use anyhow::{Context, Result, bail};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};

use super::{Language as RuntimeLanguage, Runtime};

use mambotts_registry::{BLUE_POLYG2P_DIR, BLUE_VOICES, DEFAULT_BLUE_VOICE, blue_voice_name};

pub struct BlueRuntime {
    tts: BlueTts,
    phonemizer: Phonemizer,
    styles: HashMap<String, VoiceStyle>,
    languages: Vec<RuntimeLanguage>,
}

/// The `--lexicon` TSV, if the server was started with one.
static LEXICON_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Record the force lexicon every Hebrew G2P in this process should use.
///
/// Called once from the CLI before any model is loaded; a later call is
/// ignored, which keeps the flag a start-up decision.
pub fn set_lexicon_path(path: PathBuf) {
    let _ = LEXICON_PATH.set(path);
}

fn lexicon_path() -> Option<&'static PathBuf> {
    LEXICON_PATH.get()
}

impl BlueRuntime {
    pub fn load(
        model_dir: PathBuf,
        renikud_path: PathBuf,
        speaker: u8,
        target_speaker: u8,
    ) -> Result<Self> {
        if !renikud_path.is_file() {
            bail!(
                "RenikudPlus ONNX model is missing at {}",
                renikud_path.display()
            );
        }
        let tts = BlueTts::from_dir(&model_dir)
            .with_context(|| format!("load Blue ONNX models from {}", model_dir.display()))?;
        // RenikudPlus runs natively in-process (ONNX). Do not shell out to uv/git —
        // that downloaded a second package copy and hung first Hebrew synth.
        // The front end is part of the model: polyg2p's lexicons sit in the
        // bundle, and a missing one fails here rather than at the first request.
        let polyg2p_dir = model_dir.join(BLUE_POLYG2P_DIR);
        let mut phonemizer =
            Phonemizer::open(Some(&renikud_path), Some(&polyg2p_dir), Language::English)
                .with_context(|| {
                    format!(
                        "load the text front end (RenikudPlus at {}, polyg2p data at {})",
                        renikud_path.display(),
                        polyg2p_dir.display()
                    )
                })?;
        phonemizer.set_speakers(speaker, target_speaker);
        // `--lexicon` is a property of the process rather than of one model, so
        // it survives the /v1/load the desktop sends after startup.
        if let Some(path) = lexicon_path() {
            phonemizer
                .set_hebrew_lexicon(path, "")
                .with_context(|| format!("load force lexicon from {}", path.display()))?;
        }

        let voices_dir = model_dir.join("voices");
        let mut styles = HashMap::new();
        for voice in BLUE_VOICES {
            let path = voices_dir.join(voice.file);
            let style = VoiceStyle::from_json(&path)
                .with_context(|| format!("load Blue voice style {}", path.display()))?;
            styles.insert(voice.name.to_owned(), style);
        }
        Ok(Self {
            tts,
            phonemizer,
            styles,
            languages: vec![
                language("en", 0),
                language("he", 1),
                language("es", 2),
                language("de", 3),
                language("it", 4),
                language("ru", 5),
            ],
        })
    }

    fn language_for(text: &str, requested: &str) -> Result<(Language, &'static str)> {
        let requested = requested.trim().to_ascii_lowercase();
        let code = if requested.is_empty() || requested == "auto" {
            let has = |range: std::ops::RangeInclusive<char>| {
                text.chars().any(|character| range.contains(&character))
            };
            if has('\u{0590}'..='\u{05ff}') {
                "he"
            } else if has('\u{0400}'..='\u{04ff}') {
                "ru"
            } else {
                "en"
            }
        } else {
            requested.as_str()
        };
        let language = match code {
            "en" | "en-us" => Language::English,
            "he" => Language::Hebrew,
            "es" => Language::Spanish,
            "de" | "ge" => Language::German,
            "it" => Language::Italian,
            "ru" | "ru-ru" => Language::Russian,
            _ => bail!("unsupported Blue language `{code}`; expected he, en, es, de, it, or ru"),
        };
        Ok((language, language.code()))
    }

    /// The language of a phoneme request.
    ///
    /// IPA has no script to detect, so with `auto` the leading `<xx>` routing
    /// tag the G2P writes (and the Python bindings keep) decides, then the
    /// script check of [`Self::language_for`], then English.
    fn phoneme_language_for(phonemes: &str, requested: &str) -> Result<(Language, &'static str)> {
        let requested = requested.trim();
        if requested.is_empty() || requested.eq_ignore_ascii_case("auto") {
            let tagged = phonemes
                .trim_start()
                .strip_prefix('<')
                .and_then(|rest| rest.split_once('>'))
                .map(|(code, _)| code.trim().to_ascii_lowercase())
                .filter(|code| SUPPORTED_CODES.contains(&code.as_str()));
            if let Some(code) = tagged {
                return Self::language_for(phonemes, &code);
            }
        }
        Self::language_for(phonemes, requested)
    }

    fn normalize_voice(voice: &str) -> &str {
        blue_voice_name(voice)
    }
}

impl Runtime for BlueRuntime {
    fn languages(&self) -> &[RuntimeLanguage] {
        &self.languages
    }

    fn voices(&self) -> Option<Vec<String>> {
        Some(
            BLUE_VOICES
                .iter()
                .map(|voice| voice.name.to_owned())
                .collect(),
        )
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    fn phonemize(&mut self, text: &str, language: &str) -> Result<String> {
        let (language, _) = Self::language_for(text, language)?;
        self.phonemizer.g2p(text, language).map(strip_language_tags)
    }

    fn diacritize(&mut self, text: &str, stress: bool) -> Result<String> {
        self.phonemizer.diacritize(text, stress)
    }

    fn set_speakers(&mut self, speaker: u8, target_speaker: u8) {
        self.phonemizer.set_speakers(speaker, target_speaker);
    }

    fn set_honor_niqqud(&mut self, on: bool) {
        self.phonemizer.set_honor_niqqud(on);
    }

    fn supported_phonemes(&self) -> Vec<char> {
        self.tts.supported_phonemes()
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        language: &str,
        speed: f32,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let (_, language_code) = Self::language_for(text, language)?;
        let voice = Self::normalize_voice(voice.unwrap_or(DEFAULT_BLUE_VOICE));
        let style = self.styles.get(voice).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown Blue voice `{voice}`; this bundle has {}",
                BLUE_VOICES
                    .iter()
                    .map(|voice| voice.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let sample_rate = self.tts.sample_rate();
        self.tts.synthesize_text_streaming(
            &mut self.phonemizer,
            text,
            style,
            SynthesisOptions {
                lang: language_code.to_owned(),
                speed: effective_speed(speed, language_code),
                ..SynthesisOptions::default()
            },
            |chunk| on_chunk(chunk, sample_rate),
        )
    }

    fn synthesize_phonemes_streaming(
        &mut self,
        phonemes: &str,
        voice: Option<&str>,
        language: &str,
        speed: f32,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let (_, language_code) = Self::phoneme_language_for(phonemes, language)?;
        let voice = Self::normalize_voice(voice.unwrap_or(DEFAULT_BLUE_VOICE));
        let style = self.styles.get(voice).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown Blue voice `{voice}`; this bundle has {}",
                BLUE_VOICES
                    .iter()
                    .map(|voice| voice.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let sample_rate = self.tts.sample_rate();
        // Streamed chunk by chunk like the text path, so the client hears the
        // start early, every frame stays small, and a cancel from the callback
        // stops the run at the next chunk instead of after the whole text.
        self.tts.create_streaming(
            phonemes,
            style,
            SynthesisOptions {
                lang: language_code.to_owned(),
                speed: effective_speed(speed, language_code),
                ..SynthesisOptions::default()
            },
            |chunk| on_chunk(chunk, sample_rate),
        )
    }
}

/// The speed the engine runs at, as bluetts.com sets it.
///
/// A request's speed is the engine's speed, exactly as the site's slider
/// sends it. Unset (zero) means the site's default for the language: 0.9 for
/// Hebrew, 0.95 for the others, where the release's own 1.0 reads a little
/// fast. The default is never applied on top of a chosen speed.
fn effective_speed(requested: f32, language_code: &str) -> f32 {
    if requested > 0.0 {
        requested
    } else if language_code == "he" {
        0.9
    } else {
        0.95
    }
}

fn language(name: &str, id: i32) -> RuntimeLanguage {
    RuntimeLanguage {
        name: name.to_owned(),
        id,
    }
}

/// The language codes the release speaks, as its routing tags spell them.
const SUPPORTED_CODES: [&str; 6] = ["en", "es", "de", "it", "ru", "he"];

fn strip_language_tags(phonemes: String) -> String {
    SUPPORTED_CODES
        .into_iter()
        .fold(phonemes, |output, language| {
            output
                .replace(&format!("<{language}>"), "")
                .replace(&format!("</{language}>"), "")
        })
}

#[cfg(test)]
mod site_parity;

#[cfg(test)]
mod tests {
    use super::{BlueRuntime, strip_language_tags};

    #[test]
    fn detects_hebrew_and_rejects_unsupported_languages() {
        assert_eq!(BlueRuntime::language_for("שלום", "auto").unwrap().1, "he");
        assert_eq!(BlueRuntime::language_for("Hello", "auto").unwrap().1, "en");
        assert_eq!(BlueRuntime::language_for("Hola", "es").unwrap().1, "es");
        assert_eq!(BlueRuntime::language_for("Hallo", "de").unwrap().1, "de");
        assert_eq!(BlueRuntime::language_for("Ciao", "it").unwrap().1, "it");
        assert_eq!(BlueRuntime::language_for("Привет", "auto").unwrap().1, "ru");
        assert_eq!(BlueRuntime::language_for("Привет", "ru").unwrap().1, "ru");
        assert!(BlueRuntime::language_for("Bonjour", "fr").is_err());
    }

    #[test]
    fn auto_phoneme_input_takes_its_language_from_the_leading_tag() {
        let code = |phonemes: &str, requested: &str| {
            BlueRuntime::phoneme_language_for(phonemes, requested)
                .unwrap()
                .1
        };
        assert_eq!(code("<he>ʃalˈom</he>", "auto"), "he");
        assert_eq!(code("  <ru>prʲɪvʲˈet</ru>", ""), "ru");
        assert_eq!(code("<it>ʧˈao</it>", "AUTO"), "it");
        // An explicit language wins over the tag.
        assert_eq!(code("<he>ʃalˈom</he>", "en"), "en");
        // No tag, or one the release does not speak: script detection, then English.
        assert_eq!(code("ʃalˈom", "auto"), "en");
        assert_eq!(code("<fr>bɔ̃ʒˈuʁ</fr>", "auto"), "en");
        assert!(BlueRuntime::phoneme_language_for("ʃalˈom", "fr").is_err());
    }

    #[test]
    fn strips_internal_language_tags_from_preview_ipa() {
        assert_eq!(
            strip_language_tags("<he>ʃalˈom</he> , <en>həlˈoʊ</en>".into()),
            "ʃalˈom , həlˈoʊ"
        );
    }
}
