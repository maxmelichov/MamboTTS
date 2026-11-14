use std::path::PathBuf;

use anyhow::Result;
use serde::Serialize;
use utoipa::ToSchema;

mod blue;

pub use blue::{BlueRuntime, set_lexicon_path};

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct Language {
    pub name: String,
    pub id: i32,
}

pub trait Runtime: Send {
    fn languages(&self) -> &[Language];
    fn voices(&self) -> Option<Vec<String>>;
    fn sample_rate(&self) -> u32;
    fn phonemize(&mut self, text: &str, language: &str) -> Result<String>;
    fn diacritize(&mut self, text: &str, stress: bool) -> Result<String>;
    /// Change the RenikudPlus speaker hints without reloading anything.
    fn set_speakers(&mut self, speaker: u8, target_speaker: u8);
    fn supported_phonemes(&self) -> Vec<char>;
    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        language: &str,
        speed: f32,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>>;
    fn synthesize_phonemes_streaming(
        &mut self,
        phonemes: &str,
        voice: Option<&str>,
        language: &str,
        speed: f32,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>>;
}

pub enum RuntimeParams {
    Blue {
        model_dir: PathBuf,
        renikud_path: PathBuf,
        speaker: u8,
        target_speaker: u8,
    },
}

/// The code a request's `language` names: `English`, `en-US` and `en` are
/// all `en`, and an empty value is `auto`.
pub fn language_code_alias(language: &str) -> String {
    match language.trim().to_lowercase().as_str() {
        "" | "auto" => "auto".into(),
        "english" | "american english" | "american" | "british english" | "british" | "en-us"
        | "en-gb" => "en".into(),
        "hebrew" => "he".into(),
        "spanish" => "es".into(),
        "german" | "ge" => "de".into(),
        "italian" => "it".into(),
        "russian" | "ru-ru" => "ru".into(),
        other => other.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::language_code_alias;

    #[test]
    fn language_names_and_regional_codes_resolve_to_the_bare_code() {
        assert_eq!(language_code_alias("English"), "en");
        assert_eq!(language_code_alias(" en-US "), "en");
        assert_eq!(language_code_alias("Hebrew"), "he");
        assert_eq!(language_code_alias("RU"), "ru");
        assert_eq!(language_code_alias("ru-RU"), "ru");
        assert_eq!(language_code_alias("ge"), "de");
        assert_eq!(language_code_alias(""), "auto");
        assert_eq!(language_code_alias("fr"), "fr");
    }
}
