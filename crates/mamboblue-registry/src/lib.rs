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

/// Stable id for the Qwen3-TTS Hebrew runtime.
pub const QWEN_HE_RUNTIME_ID: &str = "qwen_he";

// Two GGUFs drive qwentts.cpp: the talker turns stressed IPA into 12 Hz
// codec frames, the tokenizer decodes those frames to 24 kHz audio. The
// talker carries the merged Hebrew LoRA; the tokenizer is the stock
// upstream codec. RenikudPlus rides along because this model is fed IPA,
// never Hebrew script.
const QWEN_HE_FILES: &[ModelFile] = &[
    ModelFile {
        name: "qwen-talker-1.7b-base-Q4_K_M.gguf",
        url: "https://huggingface.co/notmax123/QwenTTS-he-1.7B-GGUF/resolve/main/qwen-talker-1.7b-base-Q4_K_M.gguf",
    },
    ModelFile {
        name: "qwen-tokenizer-12hz-Q4_K_M.gguf",
        url: "https://huggingface.co/notmax123/QwenTTS-he-1.7B-GGUF/resolve/main/qwen-tokenizer-12hz-Q4_K_M.gguf",
    },
    ModelFile {
        name: "renikud-plus.onnx",
        url: RENIKUD_URL,
    },
];

const QWEN_HE_REQUIRED_FILES: &[&str] = &[
    "qwen-talker-1.7b-base-Q4_K_M.gguf",
    "qwen-tokenizer-12hz-Q4_K_M.gguf",
    "renikud-plus.onnx",
];

const QWEN_HE: RuntimeManifest = RuntimeManifest {
    id: QWEN_HE_RUNTIME_ID,
    name: "QwenTTS Hebrew",
    version: "qwentts-he-1.7b",
    size: "~1.5 GB",
    description: "Hebrew voice cloning from a short reference recording.",
    directory: "qwentts-he-1.7b",
    install_kind: InstallKind::Files,
    files: QWEN_HE_FILES,
    required_files: QWEN_HE_REQUIRED_FILES,
    capabilities: RuntimeCapabilities {
        hebrew: true,
        streaming: true,
        voice_reference: true,
        fixed_voices: false,
    },
};

// Blue stays first and stays the default: it is the small, fast bundle
// most users want. Qwen is the heavyweight opt-in beside it.
const RUNTIMES: &[RuntimeManifest] = &[BLUE, QWEN_HE];

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
    fn qwen_manifest_ships_both_ggufs_and_a_hebrew_g2p() {
        let qwen = runtime(QWEN_HE_RUNTIME_ID).expect("Qwen manifest");
        assert!(qwen.capabilities.hebrew);
        assert!(qwen.capabilities.voice_reference);
        // The model is fed stressed IPA, so the G2P model is not optional.
        assert!(qwen.required_files.contains(&"renikud-plus.onnx"));
        assert_eq!(
            qwen.required_files
                .iter()
                .filter(|file| file.ends_with(".gguf"))
                .count(),
            2,
            "qwentts.cpp needs a talker and a codec GGUF"
        );
        assert_eq!(qwen.files.len(), qwen.required_files.len());
    }

    #[test]
    fn blue_stays_the_default_runtime() {
        assert_eq!(DEFAULT_RUNTIME_ID, "blue");
        assert_eq!(runtimes()[0].id, DEFAULT_RUNTIME_ID);
        assert!(runtimes().len() >= 2, "Qwen is additive, not a replacement");
    }

    #[test]
    fn every_runtime_has_a_unique_id_and_directory() {
        for (index, manifest) in runtimes().iter().enumerate() {
            for other in &runtimes()[index + 1..] {
                assert_ne!(manifest.id, other.id);
                assert_ne!(
                    manifest.directory, other.directory,
                    "bundles would overwrite each other on disk"
                );
            }
        }
    }

    #[test]
    fn blue_manifest_has_the_complete_hebrew_bundle() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        assert!(blue.capabilities.hebrew);
        assert!(blue.required_files.contains(&"renikud-plus.onnx"));
        assert_eq!(blue.files.len(), blue.required_files.len());
    }
}
