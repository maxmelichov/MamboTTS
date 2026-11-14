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

// BlueV3 v3.0.0, pinned to the commit the release's `v3.0.0` tag names, so a
// bundle is the same files on every install. Pin a tag's commit, not a
// branch commit: the repository's main history has been squashed more than
// once (d80497f, then ba91a77), and a squashed-away revision stops resolving. Moving to a new release means a new revision
// here and, when a file's contract changes, a new `directory`.
const BLUE_MODEL_BASE_URL: &str =
    "https://huggingface.co/notmax123/BlueV3-onnx/resolve/fb3dc3862557b8fad913a5c66ec9dfb5c9bbbf86";

macro_rules! blue_file {
    ($name:literal, $path:literal) => {
        ModelFile {
            name: $name,
            url: concat!(
                "https://huggingface.co/notmax123/BlueV3-onnx/resolve/fb3dc3862557b8fad913a5c66ec9dfb5c9bbbf86/",
                $path
            ),
        }
    };
}

// RenikudPlus 0.5.0, the Hebrew G2P of the release, at the revision the
// release names. The int8 export, not the float model the release ships. It
// is a quarter of the size (312 MB against 1.23 GB), holds 368 MB resident
// against 1287 MB, and runs inference twice as fast. Quantization leaves the
// metadata the port reads intact: every key is present and semantically
// identical, so exact-MAP decode still runs rather than falling back to greedy.
//
// On the 327-text parity corpus in plans/renikud-parity it disagrees with the
// float model on 1 to 2% of words (1.56% of phonemize words, 1.20% of vocalize
// words). Reading those disagreements by hand, int8 is right about twice as
// often as it is wrong. The one place it is worse is that a handful of its
// misses delete a segment rather than swap a vowel colour - שעות as ʃʔˈot,
// הדוח as hadˈoχ - and a dropped segment is more audible than a patah for a
// segol. That is the trade for the float model being most of the download.
const RENIKUD_URL: &str = "https://huggingface.co/notmax123/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f/model_int8.onnx";

// The graphs, the voices, and the text front end: polyg2p's lexicons for
// English, German, Italian and Russian (Spanish is rules alone) with their
// licence files, and RenikudPlus for Hebrew. The model reads IPA from exactly
// these, so they are part of the bundle rather than optional extras.
const BLUE_FILES: &[ModelFile] = &[
    blue_file!(
        "duration_predictor_style.onnx",
        "onnx/duration_predictor_style.onnx"
    ),
    blue_file!("text_encoder.onnx", "onnx/text_encoder.onnx"),
    blue_file!("vector_estimator.onnx", "onnx/vector_estimator.onnx"),
    blue_file!("vocoder.onnx", "onnx/vocoder.onnx"),
    blue_file!("vocab.json", "onnx/vocab.json"),
    blue_file!("tts.json", "onnx/tts.json"),
    blue_file!("voices/noa.json", "voices/noa.json"),
    blue_file!("voices/lily.json", "voices/lily.json"),
    blue_file!("voices/maya.json", "voices/maya.json"),
    blue_file!("voices/emma.json", "voices/emma.json"),
    blue_file!("voices/sara.json", "voices/sara.json"),
    blue_file!("voices/shira.json", "voices/shira.json"),
    blue_file!("voices/tamar.json", "voices/tamar.json"),
    blue_file!("voices/yael.json", "voices/yael.json"),
    blue_file!("voices/adam.json", "voices/adam.json"),
    blue_file!("voices/daniel.json", "voices/daniel.json"),
    blue_file!("voices/ariel.json", "voices/ariel.json"),
    blue_file!("voices/david.json", "voices/david.json"),
    blue_file!("voices/ethan.json", "voices/ethan.json"),
    blue_file!("voices/jonathan.json", "voices/jonathan.json"),
    blue_file!("voices/michael.json", "voices/michael.json"),
    blue_file!("polyg2p/en.lex", "frontend/polyg2p_data/en.lex"),
    blue_file!("polyg2p/en-large.g2p", "frontend/polyg2p_data/en-large.g2p"),
    blue_file!("polyg2p/en.g2p", "frontend/polyg2p_data/en.g2p"),
    blue_file!("polyg2p/de.lex", "frontend/polyg2p_data/de.lex"),
    blue_file!("polyg2p/it.lex", "frontend/polyg2p_data/it.lex"),
    blue_file!(
        "polyg2p/ru_accents.lex",
        "frontend/polyg2p_data/ru_accents.lex"
    ),
    blue_file!(
        "polyg2p/ru_homographs.lex",
        "frontend/polyg2p_data/ru_homographs.lex"
    ),
    blue_file!(
        "polyg2p/ru_suffix.lex",
        "frontend/polyg2p_data/ru_suffix.lex"
    ),
    blue_file!("polyg2p/ru_yo.lex", "frontend/polyg2p_data/ru_yo.lex"),
    blue_file!(
        "polyg2p/ru_yo_homographs.lex",
        "frontend/polyg2p_data/ru_yo_homographs.lex"
    ),
    blue_file!("polyg2p/LICENSE", "frontend/polyg2p_data/LICENSE"),
    blue_file!("polyg2p/NOTICE", "frontend/polyg2p_data/NOTICE"),
    ModelFile {
        name: "renikud-plus.onnx",
        url: RENIKUD_URL,
    },
];

const BLUE_REQUIRED_FILES: &[&str] = &[
    "duration_predictor_style.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
    "vocab.json",
    "tts.json",
    "voices/noa.json",
    "voices/lily.json",
    "voices/maya.json",
    "voices/emma.json",
    "voices/sara.json",
    "voices/shira.json",
    "voices/tamar.json",
    "voices/yael.json",
    "voices/adam.json",
    "voices/daniel.json",
    "voices/ariel.json",
    "voices/david.json",
    "voices/ethan.json",
    "voices/jonathan.json",
    "voices/michael.json",
    "polyg2p/en.lex",
    "polyg2p/en-large.g2p",
    "polyg2p/en.g2p",
    "polyg2p/de.lex",
    "polyg2p/it.lex",
    "polyg2p/ru_accents.lex",
    "polyg2p/ru_homographs.lex",
    "polyg2p/ru_suffix.lex",
    "polyg2p/ru_yo.lex",
    "polyg2p/ru_yo_homographs.lex",
    "polyg2p/LICENSE",
    "polyg2p/NOTICE",
    "renikud-plus.onnx",
];

/// Where polyg2p's data sits inside the bundle directory.
pub const BLUE_POLYG2P_DIR: &str = "polyg2p";

const BLUE: RuntimeManifest = RuntimeManifest {
    id: DEFAULT_RUNTIME_ID,
    name: "BlueTTS",
    version: "bluev3-3.0.0",
    size: "~740 MB",
    description: "Fast local speech for Hebrew, English, Spanish, German, Italian, and Russian.",
    // A new directory: BlueV3 has its own graphs, voices and front end, so
    // it must not land on top of a 2.5 install.
    directory: "bluev3-3.0.0",
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

/// A selectable BlueTTS voice: the name users see and the style file it
/// loads from the bundle's `voices/` directory.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct BlueVoice {
    pub name: &'static str,
    pub file: &'static str,
}

/// The BlueV3 voice catalog, in the release's order. Both the server runtime
/// and the Python bindings read this, so there is exactly one list.
pub const BLUE_VOICES: &[BlueVoice] = &[
    BlueVoice {
        name: "Noa",
        file: "noa.json",
    },
    BlueVoice {
        name: "Lily",
        file: "lily.json",
    },
    BlueVoice {
        name: "Maya",
        file: "maya.json",
    },
    BlueVoice {
        name: "Emma",
        file: "emma.json",
    },
    BlueVoice {
        name: "Sara",
        file: "sara.json",
    },
    BlueVoice {
        name: "Shira",
        file: "shira.json",
    },
    BlueVoice {
        name: "Tamar",
        file: "tamar.json",
    },
    BlueVoice {
        name: "Yael",
        file: "yael.json",
    },
    BlueVoice {
        name: "Adam",
        file: "adam.json",
    },
    BlueVoice {
        name: "Daniel",
        file: "daniel.json",
    },
    BlueVoice {
        name: "Ariel",
        file: "ariel.json",
    },
    BlueVoice {
        name: "David",
        file: "david.json",
    },
    BlueVoice {
        name: "Ethan",
        file: "ethan.json",
    },
    BlueVoice {
        name: "Jonathan",
        file: "jonathan.json",
    },
    BlueVoice {
        name: "Michael",
        file: "michael.json",
    },
];

pub const DEFAULT_BLUE_VOICE: &str = "Noa";

/// File stems the 2.5 bundle named its four voices by. A caller that pinned
/// one keeps landing on the voice of the same name.
const LEGACY_VOICE_STEMS: &[(&str, &str)] = &[
    ("libri_female_1088", "Noa"),
    ("libri_female_6147", "Lily"),
    ("libri_male_6209", "Daniel"),
    ("libri_male_8088", "Adam"),
];

/// Resolves a file stem, in any letter case, to its display name, so `noa`
/// and the older `libri_male_6209` both land on a catalog voice.
pub fn blue_voice_name(id: &str) -> &str {
    let id = id.trim();
    if let Some((_, name)) = LEGACY_VOICE_STEMS.iter().find(|(stem, _)| *stem == id) {
        return name;
    }
    BLUE_VOICES
        .iter()
        .find(|voice| voice.name.eq_ignore_ascii_case(id))
        .map(|voice| voice.name)
        .unwrap_or(id)
}

// Blue stays first and stays the default: it is the small, fast bundle
// most users want. A second engine is added by appending its manifest here.
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
    fn blue_stays_the_default_runtime() {
        assert_eq!(DEFAULT_RUNTIME_ID, "blue");
        assert_eq!(runtimes()[0].id, DEFAULT_RUNTIME_ID);
        assert!(runtime(DEFAULT_RUNTIME_ID).is_some());
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
    fn every_catalog_voice_is_in_the_bundle() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        for voice in BLUE_VOICES {
            let path = format!("voices/{}", voice.file);
            assert!(
                blue.required_files.contains(&path.as_str()),
                "{} is offered but never downloaded",
                voice.name
            );
        }
        assert!(BLUE_VOICES.iter().any(|v| v.name == DEFAULT_BLUE_VOICE));
    }

    #[test]
    fn old_voice_stems_still_resolve() {
        assert_eq!(blue_voice_name("libri_male_6209"), "Daniel");
        assert_eq!(blue_voice_name("Noa"), "Noa");
        assert_eq!(blue_voice_name("michael"), "Michael");
        assert_eq!(blue_voice_name("nonsense"), "nonsense");
    }

    #[test]
    fn every_file_comes_from_a_pinned_revision() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        for file in blue.files {
            assert!(
                !file.url.contains("/resolve/main/"),
                "{} would change under an installed app",
                file.name
            );
            if file.name != "renikud-plus.onnx" {
                assert!(file.url.starts_with(blue_model_base_url()), "{}", file.url);
            }
        }
    }

    #[test]
    fn the_bundle_carries_the_polyg2p_front_end() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        for name in [
            "en.lex",
            "en-large.g2p",
            "de.lex",
            "it.lex",
            "ru_accents.lex",
        ] {
            let path = format!("{BLUE_POLYG2P_DIR}/{name}");
            assert!(blue.required_files.contains(&path.as_str()), "{path}");
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
