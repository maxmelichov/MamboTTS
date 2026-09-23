use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::runtime::Language;

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub loaded: bool,
    pub model: String,
    pub runtime: String,
    /// True while a synthesis is running or waiting for the engine.
    pub busy: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelsResponse {
    pub loaded: bool,
    pub runtime: String,
    pub model: String,
    pub path: String,
    pub codec: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct LanguagesResponse {
    pub languages: Vec<String>,
    pub items: Vec<Language>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct VoicesResponse {
    pub runtime: String,
    pub voices: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct StatusResponse {
    pub status: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct LoadResponse {
    pub status: String,
    pub runtime: String,
    pub model: String,
}

/// Unknown fields are ignored, so older clients that still send the retired
/// `hebrew_g2p_engine` / `phonikud_path` fields keep loading RenikudPlus.
#[derive(Debug, Deserialize, ToSchema)]
pub struct LoadBody {
    #[serde(default = "default_runtime")]
    pub runtime: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub renikud_path: String,
    #[serde(default)]
    pub speaker: u8,
    #[serde(default)]
    pub target_speaker: u8,
}

impl Default for LoadBody {
    fn default() -> Self {
        Self {
            runtime: mambotts_registry::DEFAULT_RUNTIME_ID.into(),
            model_path: String::new(),
            renikud_path: String::new(),
            speaker: 0,
            target_speaker: 0,
        }
    }
}

fn default_runtime() -> String {
    mambotts_registry::DEFAULT_RUNTIME_ID.into()
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct SpeechBody {
    pub input: String,
    #[serde(default)]
    pub voice_reference: String,
    #[serde(default)]
    pub voice: String,
    #[serde(default)]
    pub response_format: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub input_is_phonemes: bool,
    /// Pace multiplier. 1.0 is the pace the app has always used; below one
    /// is slower, above one is faster. Zero or absent means 1.0.
    ///
    /// The API accepts 0.5 to 2.0, which only guards against nonsense. The
    /// app's own slider offers 0.9 to 1.1, because further out the model
    /// spreads the text over a canvas it cannot fill and the take stops
    /// sounding like the same voice at a different pace.
    #[serde(default)]
    pub speed: f32,
}

/// Which generation to stop. Without an id, every running and queued
/// generation is cancelled.
#[derive(Debug, Default, Deserialize, ToSchema)]
pub struct CancelBody {
    /// The `x-mambotts-generation-id` header of the speech response.
    #[serde(default)]
    pub id: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct CancelResponse {
    pub status: String,
    /// How many generations were told to stop.
    pub cancelled: usize,
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct PhonemizeBody {
    pub input: String,
    #[serde(default)]
    pub language: String,
}

/// Diacritization only ever applies to Hebrew, so unlike [`PhonemizeBody`]
/// this one has no language: a `language` field in the request is accepted and
/// ignored, which is what the desktop sends.
#[derive(Debug, Deserialize, ToSchema)]
pub struct DiacritizeBody {
    pub input: String,
    /// Mark the stressed syllable of every word with the hatama (U+05AB).
    ///
    /// Niqqud has no stress mark of its own, so without this the pointed text
    /// cannot say where the stress falls, and phonemizing it back gives the
    /// model a second, independent guess. Defaults to true, which is what the
    /// desktop's Phoneme editor round-trips through.
    #[serde(default = "default_stress")]
    pub stress: bool,
}

fn default_stress() -> bool {
    true
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PhonemizeResponse {
    pub phonemes: String,
}

/// Pointed Hebrew, not phonemes, so it goes out under `text`.
#[derive(Debug, Serialize, ToSchema)]
pub struct DiacritizeResponse {
    pub text: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PhonemeInventoryResponse {
    pub phonemes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diacritize_answers_with_text_not_phonemes() {
        let body = serde_json::to_value(DiacritizeResponse {
            text: "שָׁלוֹם".to_owned(),
        })
        .unwrap();
        assert_eq!(body, serde_json::json!({ "text": "שָׁלוֹם" }));
    }
}
