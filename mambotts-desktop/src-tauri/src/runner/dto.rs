use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct ReadySignal {
    pub status: String,
    pub port: u16,
}

#[derive(Debug, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Debug, Deserialize)]
pub struct ErrorBody {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct RunnerInfo {
    pub base_url: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LanguagesResponse {
    pub languages: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct VoicesResponse {
    pub voices: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct LoadModelRequest {
    #[serde(default = "default_runtime")]
    pub runtime: String,
    pub model_path: String,
    pub renikud_path: String,
    pub speaker: Option<u8>,
    pub target_speaker: Option<u8>,
}

fn default_runtime() -> String {
    mambotts_registry::DEFAULT_RUNTIME_ID.into()
}

#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    pub input: String,
    pub voice_reference: Option<String>,
    pub voice: Option<String>,
    pub output_path: Option<String>,
    pub language: Option<String>,
    pub input_is_phonemes: Option<bool>,
    pub speed: Option<f32>,
    /// Chosen by the webview so it can cancel this synthesis by name.
    pub synthesis_id: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SpeechResult {
    /// The finished recording.
    pub path: String,
    /// How many chunks went over the channel, so the webview can wait for
    /// the last of them before it treats playback as complete.
    pub chunks: usize,
}

#[derive(Debug, Deserialize)]
pub struct PhonemizeRequest {
    pub input: String,
    pub language: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PhonemizeResponse {
    pub phonemes: String,
}

/// The pointed text from `/v1/diacritize`. Older servers named the field
/// `phonemes` (it shared the phonemize response); newer ones call it `text`.
/// Either is accepted, so the app works against both.
#[derive(Debug, Deserialize)]
pub struct DiacritizeResponse {
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub phonemes: Option<String>,
}

impl DiacritizeResponse {
    pub fn into_text(self) -> Option<String> {
        self.text.or(self.phonemes)
    }
}

#[derive(Debug, Deserialize)]
pub struct PhonemeInventoryResponse {
    pub phonemes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::DiacritizeResponse;

    fn read(body: serde_json::Value) -> Option<String> {
        serde_json::from_value::<DiacritizeResponse>(body)
            .unwrap()
            .into_text()
    }

    #[test]
    fn diacritize_text_is_read_under_either_name() {
        assert_eq!(
            read(serde_json::json!({"text": "שָׁלוֹם"})).as_deref(),
            Some("שָׁלוֹם")
        );
        assert_eq!(
            read(serde_json::json!({"phonemes": "שָׁלוֹם"})).as_deref(),
            Some("שָׁלוֹם")
        );
        assert_eq!(read(serde_json::json!({})), None);
    }
}
