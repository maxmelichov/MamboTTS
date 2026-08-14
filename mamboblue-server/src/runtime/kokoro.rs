use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use kokoro_rs::{Kokoro, KokoroConfig, SynthesizeRequest, list_voices, write_mono_i16_wav};

use super::{Language, Runtime, language_code_alias};

const KOKORO_LANGUAGES: &[&str] = &["en-us", "en", "es", "fr", "ja", "hi", "it", "pt-br"];

pub struct KokoroRuntime {
    model_path: PathBuf,
    voices_path: PathBuf,
    default_voice: String,
    default_language: String,
    speed: f32,
    voices: Vec<String>,
    languages: Vec<Language>,
    loaded: HashMap<KokoroKey, Kokoro>,
}

impl KokoroRuntime {
    pub fn load(
        model_path: PathBuf,
        voices_path: PathBuf,
        voice: Option<String>,
        language: Option<String>,
        speed: f32,
    ) -> Result<Self> {
        let voices = list_voices(&voices_path)?;
        Ok(Self {
            model_path,
            voices_path,
            default_voice: default_string(voice.as_deref(), "af_heart"),
            default_language: kokoro_language(language.as_deref().unwrap_or("")),
            speed: if speed > 0.0 { speed } else { 1.0 },
            voices,
            languages: KOKORO_LANGUAGES
                .iter()
                .enumerate()
                .map(|(id, name)| Language {
                    name: (*name).into(),
                    id: id as i32,
                })
                .collect(),
            loaded: HashMap::new(),
        })
    }
}

impl Runtime for KokoroRuntime {
    fn languages(&self) -> &[Language] {
        &self.languages
    }

    fn voices(&self) -> Option<Vec<String>> {
        Some(self.voices.clone())
    }

    fn synthesize_to_file(
        &mut self,
        text: &str,
        voice_or_reference: Option<&Path>,
        output_path: &Path,
        language: &str,
    ) -> Result<()> {
        if text.is_empty() {
            bail!("text is required");
        }
        let voice = voice_or_reference
            .and_then(Path::to_str)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| self.default_voice.clone());
        let language = if language.trim().is_empty() {
            self.default_language.clone()
        } else {
            kokoro_language(language)
        };
        let key = KokoroKey { voice, language };
        if !self.loaded.contains_key(&key) {
            let config = KokoroConfig::new(&self.model_path, &self.voices_path)
                .voice(&key.voice)
                .language(&key.language)
                .speed(self.speed);
            self.loaded.insert(key.clone(), Kokoro::load(config)?);
        }
        let kokoro = self
            .loaded
            .get_mut(&key)
            .ok_or_else(|| anyhow::anyhow!("failed to load Kokoro runtime"))?;
        let audio = kokoro.synthesize(SynthesizeRequest::new(text))?;
        write_mono_i16_wav(output_path, &audio)?;
        Ok(())
    }
}

pub fn kokoro_language(language: &str) -> String {
    match language_code_alias(language).as_str() {
        "" | "auto" | "english" | "american" | "en-us" => "en-us".into(),
        "british" | "en" | "en-gb" => "en".into(),
        "spanish" | "es" => "es".into(),
        "french" | "fr" | "fr-fr" => "fr".into(),
        "japanese" | "ja" => "ja".into(),
        "hindi" | "hi" => "hi".into(),
        "italian" | "it" => "it".into(),
        "portuguese" | "pt" | "pt-br" => "pt-br".into(),
        other => other.into(),
    }
}

fn default_string(value: Option<&str>, fallback: &str) -> String {
    match value {
        Some(value) if !value.is_empty() && value != "auto" => value.into(),
        _ => fallback.into(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KokoroKey {
    voice: String,
    language: String,
}
