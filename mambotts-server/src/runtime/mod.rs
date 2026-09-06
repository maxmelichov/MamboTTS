use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use utoipa::ToSchema;

mod blue;

pub use blue::BlueRuntime;

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
    fn diacritize(&mut self, text: &str) -> Result<String>;
    fn supported_phonemes(&self) -> Vec<char>;
    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        language: &str,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>>;
    fn synthesize_phonemes_streaming(
        &mut self,
        phonemes: &str,
        voice: Option<&str>,
        language: &str,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>>;
    fn synthesize_to_file(
        &mut self,
        text: &str,
        voice: Option<&str>,
        output_path: &Path,
        language: &str,
    ) -> Result<()>;
}

pub enum RuntimeParams {
    Blue {
        model_dir: PathBuf,
        renikud_path: PathBuf,
        hebrew_g2p_engine: String,
        phonikud_path: Option<PathBuf>,
        speaker: u8,
        target_speaker: u8,
    },
}

pub fn language_display_name(language: &str) -> String {
    match language.trim().to_lowercase().as_str() {
        "en-us" | "en" | "en-gb" => "English".into(),
        "he" => "Hebrew".into(),
        "es" => "Spanish".into(),
        "de" | "ge" => "German".into(),
        "it" => "Italian".into(),
        other => title_case_language(other),
    }
}

pub fn language_code_alias(language: &str) -> String {
    match language.trim().to_lowercase().as_str() {
        "auto" => "auto".into(),
        "english" | "american english" | "american" | "british english" | "british" | "en-us"
        | "en-gb" => "en".into(),
        "hebrew" => "he".into(),
        "spanish" => "es".into(),
        "german" => "de".into(),
        "italian" => "it".into(),
        other => other.into(),
    }
}

fn title_case_language(language: &str) -> String {
    language
        .split(['-', '_', ' '])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
