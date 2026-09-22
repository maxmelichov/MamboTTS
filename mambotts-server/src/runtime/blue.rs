use std::{collections::HashMap, path::PathBuf, sync::OnceLock};

use anyhow::{Context, Result, bail};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};

use super::{Language as RuntimeLanguage, Runtime};

use mambotts_registry::{BLUE_VOICES, DEFAULT_BLUE_VOICE, blue_voice_name};

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
        let mut phonemizer = Phonemizer::with_language(Some(&renikud_path), Language::English)
            .with_context(|| format!("load RenikudPlus ONNX from {}", renikud_path.display()))?;
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
            ],
        })
    }

    fn language_for(text: &str, requested: &str) -> Result<(Language, &'static str)> {
        let requested = requested.trim().to_ascii_lowercase();
        let code = if requested.is_empty() || requested == "auto" {
            if text
                .chars()
                .any(|character| ('\u{0590}'..='\u{05ff}').contains(&character))
            {
                "he"
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
            _ => bail!("unsupported Blue language `{code}`; expected he, en, es, de, or it"),
        };
        Ok((language, language.code()))
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
                total_step: 8,
                cfg_scale: 4.0,
                speed,
                chunking: Some(blue_rs::ChunkingOptions {
                    enabled: true,
                    silence_seconds: 0.15,
                    max_chars: Some(200),
                }),
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
        let (_, language_code) = Self::language_for(phonemes, language)?;
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
                total_step: 8,
                cfg_scale: 4.0,
                speed,
                chunking: Some(blue_rs::ChunkingOptions {
                    enabled: true,
                    silence_seconds: 0.15,
                    max_chars: Some(200),
                }),
            },
            |chunk| on_chunk(chunk, sample_rate),
        )
    }
}

fn language(name: &str, id: i32) -> RuntimeLanguage {
    RuntimeLanguage {
        name: name.to_owned(),
        id,
    }
}

fn strip_language_tags(phonemes: String) -> String {
    ["en", "es", "de", "it", "he"]
        .into_iter()
        .fold(phonemes, |output, language| {
            output
                .replace(&format!("<{language}>"), "")
                .replace(&format!("</{language}>"), "")
        })
}

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
        assert!(BlueRuntime::language_for("Bonjour", "fr").is_err());
    }

    #[test]
    fn strips_internal_language_tags_from_preview_ipa() {
        assert_eq!(
            strip_language_tags("<he>ʃalˈom</he> , <en>həlˈoʊ</en>".into()),
            "ʃalˈom , həlˈoʊ"
        );
    }
}
