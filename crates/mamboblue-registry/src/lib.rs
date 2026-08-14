//! The single source of truth for TTS runtime metadata.
//!
//! Adding an engine starts here: define its model manifest and capabilities,
//! then implement the matching server runtime adapter.

use serde::Serialize;

pub const DEFAULT_RUNTIME_ID: &str = "blue";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InstallKind {
    Files,
    Archive,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub hebrew: bool,
    pub streaming: bool,
    pub voice_reference: bool,
    pub fixed_voices: bool,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct RuntimeManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub version: &'static str,
    pub size: &'static str,
    pub description: &'static str,
    pub directory: &'static str,
    pub install_kind: InstallKind,
    pub files: &'static [ModelFile],
    pub required_files: &'static [&'static str],
    pub capabilities: RuntimeCapabilities,
}

const BLUE_MODEL_BASE_URL: &str = "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main";
const RENIKUD_URL: &str = "https://huggingface.co/notmax123/RenikudPlus/resolve/main/model.onnx";

const BLUE_FILES: &[ModelFile] = &[
    ModelFile {
        name: "duration_predictor.onnx",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/duration_predictor.onnx",
    },
    ModelFile {
        name: "text_encoder.onnx",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/text_encoder.onnx",
    },
    ModelFile {
        name: "vector_estimator.onnx",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/vector_estimator.onnx",
    },
    ModelFile {
        name: "vocoder.onnx",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/vocoder.onnx",
    },
    ModelFile {
        name: "vocab.json",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/vocab.json",
    },
    ModelFile {
        name: "tts.json",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/tts.json",
    },
    ModelFile {
        name: "voices/female1.json",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/voices/female1.json",
    },
    ModelFile {
        name: "voices/male1.json",
        url: "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main/voices/male1.json",
    },
    ModelFile {
        name: "renikud-plus.onnx",
        url: RENIKUD_URL,
    },
];

const BLUE_REQUIRED_FILES: &[&str] = &[
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
    "vocab.json",
    "tts.json",
    "voices/female1.json",
    "voices/male1.json",
    "renikud-plus.onnx",
];

const BLUE: RuntimeManifest = RuntimeManifest {
    id: DEFAULT_RUNTIME_ID,
    name: "BlueTTS",
    version: "blue-onnx-v2",
    size: "~560 MB",
    description: "Fast local speech for Hebrew and English.",
    directory: "blue-onnx-v2",
    install_kind: InstallKind::Files,
    files: BLUE_FILES,
    required_files: BLUE_REQUIRED_FILES,
    capabilities: RuntimeCapabilities {
        hebrew: true,
        streaming: true,
        voice_reference: false,
        fixed_voices: true,
    },
};

const RUNTIMES: &[RuntimeManifest] = &[BLUE];

pub fn runtimes() -> &'static [RuntimeManifest] {
    RUNTIMES
}

pub fn runtime(id: &str) -> Option<&'static RuntimeManifest> {
    RUNTIMES.iter().find(|manifest| manifest.id == id)
}

pub fn blue_model_base_url() -> &'static str {
    BLUE_MODEL_BASE_URL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blue_manifest_has_the_complete_hebrew_bundle() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        assert!(blue.capabilities.hebrew);
        assert!(blue.required_files.contains(&"renikud-plus.onnx"));
        assert_eq!(blue.files.len(), blue.required_files.len());
    }
}
