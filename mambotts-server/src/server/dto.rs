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
    /// One of the fifteen BlueV3 voices from /v1/voices, in any letter case
    /// (`Noa`, `noa`). Empty means `Shira`, the default voice of bluetts.com.
    #[serde(default)]
    pub voice: String,
    /// `wav` (the default) or `mp3` (128 kbps mono). Streamed responses are
    /// always WAV chunks.
    #[serde(default)]
    pub response_format: String,
    /// `auto` (the default), `he`, `en`, `es`, `de`, `it` or `ru`; language
    /// names such as `Hebrew` and regional codes such as `en-US` are accepted.
    /// `auto` picks Hebrew for Hebrew script, Russian for Cyrillic, otherwise
    /// English; for phoneme input it reads the leading `<xx>` tag instead.
    #[serde(default)]
    pub language: String,
    /// Stream MamboTTS binary frames (WAV chunks, then the whole WAV) instead
    /// of returning one file.
    #[serde(default)]
    pub stream: bool,
    /// `input` is IPA (for example from /v1/phonemize) rather than text.
    /// Requires `stream: true`.
    #[serde(default)]
    pub input_is_phonemes: bool,
    /// The input is pointed Hebrew whose points were chosen on purpose (the
    /// app's niqqud editor): RenikudPlus reads them. Off by default, where
    /// points are dropped and the bare letters read, as bluetts.com and the
    /// BlueV3 reference read any pointed text.
    #[serde(default)]
    pub niqqud: bool,
    /// The engine's speed, as bluetts.com sends it: 1.0 is the voice's
    /// natural pace (BlueV3 v3.0.1 multiplies it by the voice's
    /// `speed_scale` for the language), below one is slower. Zero or absent
    /// means the release's default, 1.0, in every language.
    ///
    /// The API accepts 0.5 to 2.0, which only guards against nonsense. The
    /// app's slider offers the site's 0.8 to 1.2.
    #[serde(default)]
    pub speed: f64,
    /// The number of flow-matching steps, bluetts.com's Quality slider: more
    /// is slower and cleaner. Zero or absent means the site's default, 8.
    ///
    /// The API accepts 1 to 32. The app's slider offers the site's 5 to 16
    /// (Fast 5-6, Mid 7-9, High 10-12, Best 13-16).
    #[serde(default)]
    pub steps: u32,
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
    /// Same values as the speech request's `language`.
    #[serde(default)]
    pub language: String,
    /// The input is pointed Hebrew whose points were chosen on purpose (the
    /// app's niqqud editor): RenikudPlus reads them. Off by default, where
    /// points are dropped and the bare letters read, as bluetts.com and the
    /// BlueV3 reference read any pointed text.
    #[serde(default)]
    pub niqqud: bool,
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
