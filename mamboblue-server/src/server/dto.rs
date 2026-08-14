use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::runtime::Language;

#[derive(Debug, Serialize, ToSchema)]
pub struct HealthResponse {
    pub status: String,
    pub loaded: bool,
    pub model: String,
    pub runtime: String,
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

#[derive(Debug, Deserialize, ToSchema)]
pub struct LoadBody {
    #[serde(default = "default_runtime")]
    pub runtime: String,
    #[serde(default)]
    pub model_path: String,
    #[serde(default)]
    pub renikud_path: String,
    #[serde(default)]
    pub hebrew_g2p_engine: String,
    #[serde(default)]
    pub phonikud_path: String,
    #[serde(default)]
    pub speaker: u8,
    #[serde(default)]
    pub target_speaker: u8,
}

impl Default for LoadBody {
    fn default() -> Self {
        Self {
            runtime: mamborambo_registry::DEFAULT_RUNTIME_ID.into(),
            model_path: String::new(),
            renikud_path: String::new(),
            hebrew_g2p_engine: "renikud".into(),
            phonikud_path: String::new(),
            speaker: 0,
            target_speaker: 0,
        }
    }
}

fn default_runtime() -> String {
    mamborambo_registry::DEFAULT_RUNTIME_ID.into()
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
}

#[derive(Debug, Deserialize, ToSchema)]
pub struct PhonemizeBody {
    pub input: String,
    #[serde(default)]
    pub language: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PhonemizeResponse {
    pub phonemes: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct PhonemeInventoryResponse {
    pub phonemes: Vec<String>,
}
