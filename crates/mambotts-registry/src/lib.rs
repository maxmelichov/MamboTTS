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
    /// The file's exact size in bytes at the pinned revision.
    pub size: u64,
    /// Lowercase hex SHA-256 of the file at the pinned revision. A download
    /// takes the file's real name only once its size and hash match.
    pub sha256: &'static str,
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
    /// The bundle also needs one RenikudPlus file ([`HebrewG2pVariant`]),
    /// which `files` leaves out because the machine decides which.
    pub hebrew_g2p: bool,
    pub capabilities: RuntimeCapabilities,
}

impl RuntimeManifest {
    /// Every file to download, the RenikudPlus variant included.
    pub fn files_for(&self, variant: HebrewG2pVariant) -> Vec<ModelFile> {
        let mut files = self.files.to_vec();
        if self.hebrew_g2p {
            files.push(variant.model_file());
        }
        files
    }

    /// The download size label for a variant.
    pub fn size_for(&self, variant: HebrewG2pVariant) -> &'static str {
        match (self.hebrew_g2p, variant) {
            (true, HebrewG2pVariant::Fp32) => "~1.7 GB",
            _ => self.size,
        }
    }
}

// BlueV3 v3.0.0, pinned to the commit the release's `v3.0.0` tag names
// (fb3dc38), so a bundle is the same files on every install. Pin a tag's
// commit, not a branch commit: the repository's main history has been
// squashed more than once (d80497f, then ba91a77), and a squashed-away
// revision stops resolving. ba91a77 is already half gone: the Hub no longer
// lists it, and `frontend/polyg2p_data/LICENSE` at it answers 404. Every graph
// and lexicon is byte-identical between the two; tts.json and the voice files
// differ only in an added `model` field (the style tensors are equal). Moving
// to a new release means a new revision here, new sizes and hashes, and, when
// a file's contract changes, a new `directory`.
const BLUE_MODEL_BASE_URL: &str =
    "https://huggingface.co/notmax123/BlueV3-onnx/resolve/fb3dc3862557b8fad913a5c66ec9dfb5c9bbbf86";

macro_rules! blue_file {
    ($name:literal, $path:literal, $size:literal, $sha256:literal) => {
        ModelFile {
            name: $name,
            url: concat!(
                "https://huggingface.co/notmax123/BlueV3-onnx/resolve/fb3dc3862557b8fad913a5c66ec9dfb5c9bbbf86/",
                $path
            ),
            size: $size,
            sha256: $sha256,
        }
    };
}

// RenikudPlus, the Hebrew G2P, at the Hub revision bluetts.com pins
// (js/g2p/hebrew.js REVISION): the decode reads cascade-conditioning
// matrices from the model's metadata, which earlier revisions do not carry.
// The revision has two official weight files, and the app picks between them
// as the site does (backend-policy.js hebrewG2PVariant): the float model on
// computers, the int8 export only where memory is short.
/// One of the two RenikudPlus weight files of the pinned revision.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HebrewG2pVariant {
    /// `model.onnx`, 1.23 GB, ~1.3 GB resident: what RenikudPlus itself
    /// loads, what the BlueV3 golden cases were made with, and what the site
    /// runs on computers.
    Fp32,
    /// `model_int8.onnx`, 312 MB, ~0.4 GB resident: dynamic quantization. It
    /// reads 1 to 2% of words differently from fp32 (3 of the 116 BlueV3
    /// golden cases), so it is the low-memory fallback only.
    Int8,
}

impl HebrewG2pVariant {
    /// The file name inside the bundle directory.
    pub fn file_name(self) -> &'static str {
        match self {
            // The int8 file keeps the name installs have always had.
            Self::Fp32 => "renikud-plus-fp32.onnx",
            Self::Int8 => "renikud-plus.onnx",
        }
    }

    pub fn model_file(self) -> ModelFile {
        match self {
            Self::Fp32 => ModelFile {
                name: "renikud-plus-fp32.onnx",
                url: "https://huggingface.co/notmax123/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f/model.onnx",
                size: 1_234_209_530,
                sha256: "c3a2bdc35afbf53f28cdece4672dcca9b530dfa43a0d65b4c5bbc47150b32d0d",
            },
            Self::Int8 => ModelFile {
                name: "renikud-plus.onnx",
                url: "https://huggingface.co/notmax123/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f/model_int8.onnx",
                size: 311_605_741,
                sha256: "0337468fad56eadf5ecc7becfdf21648a4d668de41de5bc27e860141b5afcadd",
            },
        }
    }

    pub fn other(self) -> Self {
        match self {
            Self::Fp32 => Self::Int8,
            Self::Int8 => Self::Fp32,
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "fp32" | "float" | "float32" => Some(Self::Fp32),
            "int8" | "quantized" => Some(Self::Int8),
            _ => None,
        }
    }
}

/// Machines with this much memory or less get the int8 RenikudPlus, as
/// devices reporting 4 GB or less do on the site.
pub const HEBREW_G2P_INT8_MAX_MEMORY: u64 = 4 * 1024 * 1024 * 1024;

/// The site's `hebrewG2PVariant` for a computer: fp32, unless the machine
/// has 4 GB of memory or less. `override_` (the `MAMBOTTS_HEBREW_G2P`
/// setting: `fp32` or `int8`) wins over the measurement; an unknown memory
/// size is a computer's, so fp32.
pub fn hebrew_g2p_variant(total_memory: Option<u64>, override_: Option<&str>) -> HebrewG2pVariant {
    if let Some(variant) = override_.and_then(HebrewG2pVariant::parse) {
        return variant;
    }
    match total_memory {
        Some(bytes) if bytes > 0 && bytes <= HEBREW_G2P_INT8_MAX_MEMORY => HebrewG2pVariant::Int8,
        _ => HebrewG2pVariant::Fp32,
    }
}

/// The RenikudPlus file to load from a bundle directory: the preferred
/// variant when it is there, else the other one, else the preferred path
/// (which fails to load with a clear message).
pub fn hebrew_g2p_path(dir: &std::path::Path, preferred: HebrewG2pVariant) -> std::path::PathBuf {
    let present = |variant: HebrewG2pVariant| {
        let path = dir.join(variant.file_name());
        std::fs::metadata(&path)
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
            .then_some(path)
    };
    present(preferred)
        .or_else(|| present(preferred.other()))
        .unwrap_or_else(|| dir.join(preferred.file_name()))
}

/// What a bundle directory's RenikudPlus files mean on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HebrewG2pPlan {
    /// The variant to load now, if any file can be loaded at all.
    pub load: Option<HebrewG2pVariant>,
    /// The variant to fetch in the background while `load` keeps working: the
    /// fp32 file on a machine whose policy is fp32 and which so far has only
    /// int8 (an install from before the fp32 default).
    pub upgrade: Option<HebrewG2pVariant>,
    /// The variant to delete because the policy's own file is in place.
    pub remove: Option<HebrewG2pVariant>,
}

/// Plans RenikudPlus for a bundle directory under the machine's policy.
///
/// A file is complete when it has the pinned revision's exact size (its hash
/// was checked when it was downloaded); any non-empty file can still be
/// loaded. The policy's complete file is loaded; otherwise the other
/// variant keeps the app working. Only the fp32 direction is upgraded in the
/// background: a low-memory machine left with only fp32 runs it rather than
/// downloading int8. The unused variant goes once the policy's file is
/// complete, so an fp32 machine deletes int8 only after fp32 is verified.
pub fn hebrew_g2p_plan(dir: &std::path::Path, policy: HebrewG2pVariant) -> HebrewG2pPlan {
    let len = |variant: HebrewG2pVariant| {
        std::fs::metadata(dir.join(variant.file_name()))
            .ok()
            .filter(|metadata| metadata.is_file())
            .map(|metadata| metadata.len())
            .unwrap_or(0)
    };
    let complete = |variant: HebrewG2pVariant| len(variant) == variant.model_file().size;
    let usable = |variant: HebrewG2pVariant| len(variant) > 0;
    let other = policy.other();
    let load = if complete(policy) {
        Some(policy)
    } else if usable(other) {
        Some(other)
    } else if usable(policy) {
        Some(policy)
    } else {
        None
    };
    let upgrade = (policy == HebrewG2pVariant::Fp32 && !complete(policy) && load.is_some())
        .then_some(policy);
    let remove = (complete(policy) && usable(other)).then_some(other);
    HebrewG2pPlan {
        load,
        upgrade,
        remove,
    }
}

// The graphs, the voices, and the text front end: polyg2p's lexicons for
// English, German, Italian and Russian (Spanish is rules alone) with their
// licence files, and RenikudPlus for Hebrew. The model reads IPA from exactly
// these, so they are part of the bundle rather than optional extras.
const BLUE_FILES: &[ModelFile] = &[
    blue_file!(
        "duration_predictor_style.onnx",
        "onnx/duration_predictor_style.onnx",
        1_524_267,
        "9aa8951637c7de4c2f6d43296af6394f4527a810fe8ba56df5aeedc2cec7247b"
    ),
    blue_file!(
        "text_encoder.onnx",
        "onnx/text_encoder.onnx",
        27_951_369,
        "288d48053d82f12fb4b7319ea0a5c8c669c4e87ade983c695666fa9e346408c1"
    ),
    blue_file!(
        "vector_estimator.onnx",
        "onnx/vector_estimator.onnx",
        256_556_761,
        "46abc33ac65e24cb5331b6d29ec4520b9a33a4a86a7723b5e44677af10d5b584"
    ),
    blue_file!(
        "vocoder.onnx",
        "onnx/vocoder.onnx",
        101_428_551,
        "054a700e9ef17d43de203212dca335fa2cd000cd9ca7701df49fcc544b49ccbe"
    ),
    blue_file!(
        "vocab.json",
        "onnx/vocab.json",
        3_382,
        "9cf96e595f62ea8a36d3be6724b06fb821614c2938d0724852d1a25feff5bad5"
    ),
    blue_file!(
        "tts.json",
        "onnx/tts.json",
        8_509,
        "35af77b0634c2d4be4eb2faeeabdb364ad6bf80302202f6b5952e83984326eea"
    ),
    blue_file!(
        "voices/noa.json",
        "voices/noa.json",
        287_857,
        "d392adf5542220ca57841308ffd7ac76f013025b624efd8f43a8f53a799e4638"
    ),
    blue_file!(
        "voices/lily.json",
        "voices/lily.json",
        288_281,
        "7d1b9f54d4a6797b31e997f59a55399e3daf73782aa2a477721afd963ca59969"
    ),
    blue_file!(
        "voices/maya.json",
        "voices/maya.json",
        288_203,
        "83a9f7303c954951677fc0db17e58bfeee8064553f67f3991b7c7d0ed4c8f8ce"
    ),
    blue_file!(
        "voices/emma.json",
        "voices/emma.json",
        288_397,
        "46cedaaa14f68dc9017e8c0fac94b21ba615845c1f7981cb5bfe3740cf4a2365"
    ),
    blue_file!(
        "voices/sara.json",
        "voices/sara.json",
        288_277,
        "dc43188552dfede47f9450104cce931ae356e8896f1b9b093317d4dff4fc6863"
    ),
    blue_file!(
        "voices/shira.json",
        "voices/shira.json",
        288_019,
        "19611940a2cdc1b90ce70b13b12682c3c2b1ef6be14a78907c4fb8228a913ca3"
    ),
    blue_file!(
        "voices/tamar.json",
        "voices/tamar.json",
        288_114,
        "09026d10ef83b9e7e490b68f17ef7fb54e80940bf6321990b53a5a7340c79987"
    ),
    blue_file!(
        "voices/yael.json",
        "voices/yael.json",
        288_145,
        "3c82bf087ce6831b75c2755741dc519ebe87a58cc1e02e36f4cfa7ca35650501"
    ),
    blue_file!(
        "voices/adam.json",
        "voices/adam.json",
        288_210,
        "9daa9a234c5a4a36cab1d02c527c307a8a0058a3f806b316af01150a98604d22"
    ),
    blue_file!(
        "voices/daniel.json",
        "voices/daniel.json",
        288_828,
        "c8a6dd2a63259f686230cf3fd55e28906a158c9363cd33cb374a02201b862f5f"
    ),
    blue_file!(
        "voices/ariel.json",
        "voices/ariel.json",
        288_672,
        "67cc2e5c2f5c582d120aee4c5cd65b4ab21c4a4d89a0fe81ef1938a1d9f266cc"
    ),
    blue_file!(
        "voices/david.json",
        "voices/david.json",
        288_598,
        "6e51be7341233ce360659031f0ab4c01e6829de845cd3ccf1e9a1a5cde370109"
    ),
    blue_file!(
        "voices/ethan.json",
        "voices/ethan.json",
        288_591,
        "802e09b837dc8ff8bf2023ea0d9a89b845b5385b5312d00d95734ca40c840b20"
    ),
    blue_file!(
        "voices/jonathan.json",
        "voices/jonathan.json",
        288_406,
        "7af846f12fb30ba29dcf5047e04ce2f9d0f6d4164cd31109220cb03d744a52ff"
    ),
    blue_file!(
        "voices/michael.json",
        "voices/michael.json",
        288_215,
        "2018e228025313923abec2bc6b468db2e4a98ef916fea40d5501c0ec6fb3f335"
    ),
    blue_file!(
        "polyg2p/en.lex",
        "frontend/polyg2p_data/en.lex",
        2_759_393,
        "713aaed9ce1891e111eab2d213df560f043947b9268c6411b26a8c756ea99d32"
    ),
    blue_file!(
        "polyg2p/en-large.g2p",
        "frontend/polyg2p_data/en-large.g2p",
        4_880_559,
        "a119ddaabc98ceb2696f2d9af52cfc0733388286d489b175c43129365ea15c4b"
    ),
    blue_file!(
        "polyg2p/en.g2p",
        "frontend/polyg2p_data/en.g2p",
        1_780_155,
        "2f638c5358a2a90075a575024d747931d1d92160e18ce0f9ed7176894a366808"
    ),
    blue_file!(
        "polyg2p/de.lex",
        "frontend/polyg2p_data/de.lex",
        17_815_772,
        "eeb7df564ef1bc894bf7cc39f0f054f5c516d38c2119922ac69ff8c31681594c"
    ),
    blue_file!(
        "polyg2p/it.lex",
        "frontend/polyg2p_data/it.lex",
        516_990,
        "9f7e3363abe1beaeb6eab1d5393388d9f91b877491ade9841f47a27a871554f7"
    ),
    blue_file!(
        "polyg2p/ru_accents.lex",
        "frontend/polyg2p_data/ru_accents.lex",
        5_529_194,
        "505e3235f41a61fb54af70fb0a426f6ae68bb75e4d422841f4063a08865c8b07"
    ),
    blue_file!(
        "polyg2p/ru_homographs.lex",
        "frontend/polyg2p_data/ru_homographs.lex",
        82_364,
        "27a3bb226121130717c7e44fb21a25b826e1e5b77900f0a151bdf560bbaaa9e0"
    ),
    blue_file!(
        "polyg2p/ru_suffix.lex",
        "frontend/polyg2p_data/ru_suffix.lex",
        399_562,
        "139cf5af5174d9408b27c428be5b341aaeeac144f3dd23bed0ab740a2b871ea6"
    ),
    blue_file!(
        "polyg2p/ru_yo.lex",
        "frontend/polyg2p_data/ru_yo.lex",
        2_163_336,
        "16dbb14ea719a5f8562f64a41873e701c6163737cb03af55efa038c07b112165"
    ),
    blue_file!(
        "polyg2p/ru_yo_homographs.lex",
        "frontend/polyg2p_data/ru_yo_homographs.lex",
        19_043,
        "057cddf02713732706127274c796a6f418ad7096679ac532f7c31cf7c9266064"
    ),
    blue_file!(
        "polyg2p/LICENSE",
        "frontend/polyg2p_data/LICENSE",
        11_357,
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
    blue_file!(
        "polyg2p/NOTICE",
        "frontend/polyg2p_data/NOTICE",
        2_734,
        "1a5544c762f18316b934e03db8ced24931f752f5332aa70c3bb15ca1ba92a114"
    ),
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
    hebrew_g2p: true,
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

/// The BlueV3 voice catalog: the bundle's fifteen `voices/*.json`, in the
/// order its README lists them (female: maya, noa, lily, tamar, shira, yael,
/// emma, sara; male: adam, daniel, david, jonathan, ethan, michael, ariel).
/// All fifteen are real voices with their own style file; there are no
/// aliases. Both the server runtime and the Python bindings read this, so
/// there is exactly one list.
pub const BLUE_VOICES: &[BlueVoice] = &[
    BlueVoice {
        name: "Maya",
        file: "maya.json",
    },
    BlueVoice {
        name: "Noa",
        file: "noa.json",
    },
    BlueVoice {
        name: "Lily",
        file: "lily.json",
    },
    BlueVoice {
        name: "Tamar",
        file: "tamar.json",
    },
    BlueVoice {
        name: "Shira",
        file: "shira.json",
    },
    BlueVoice {
        name: "Yael",
        file: "yael.json",
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
        name: "Adam",
        file: "adam.json",
    },
    BlueVoice {
        name: "Daniel",
        file: "daniel.json",
    },
    BlueVoice {
        name: "David",
        file: "david.json",
    },
    BlueVoice {
        name: "Jonathan",
        file: "jonathan.json",
    },
    BlueVoice {
        name: "Ethan",
        file: "ethan.json",
    },
    BlueVoice {
        name: "Michael",
        file: "michael.json",
    },
    BlueVoice {
        name: "Ariel",
        file: "ariel.json",
    },
];

/// The voice a request without one gets: bluetts.com's default, Shira.
pub const DEFAULT_BLUE_VOICE: &str = "Shira";

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
    fn the_catalog_is_the_bundles_fifteen_voices_in_its_order() {
        let names: Vec<&str> = BLUE_VOICES.iter().map(|voice| voice.name).collect();
        assert_eq!(
            names,
            [
                "Maya", "Noa", "Lily", "Tamar", "Shira", "Yael", "Emma", "Sara", "Adam", "Daniel",
                "David", "Jonathan", "Ethan", "Michael", "Ariel"
            ]
        );
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        let mut files: Vec<&str> = blue
            .files
            .iter()
            .filter_map(|file| file.name.strip_prefix("voices/"))
            .collect();
        let mut catalog: Vec<&str> = BLUE_VOICES.iter().map(|voice| voice.file).collect();
        files.sort_unstable();
        catalog.sort_unstable();
        assert_eq!(files, catalog, "every bundled voice is offered, once");
        assert_eq!(DEFAULT_BLUE_VOICE, "Shira");
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
            assert!(file.url.starts_with(blue_model_base_url()), "{}", file.url);
        }
        for variant in [HebrewG2pVariant::Fp32, HebrewG2pVariant::Int8] {
            let file = variant.model_file();
            assert!(
                file.url.contains("/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f/"),
                "{}",
                file.url
            );
            assert_eq!(file.name, variant.file_name());
        }
    }

    #[test]
    fn hebrew_g2p_follows_the_sites_policy() {
        const GB: u64 = 1024 * 1024 * 1024;
        assert_eq!(hebrew_g2p_variant(Some(16 * GB), None), HebrewG2pVariant::Fp32);
        assert_eq!(hebrew_g2p_variant(Some(8 * GB), None), HebrewG2pVariant::Fp32);
        assert_eq!(hebrew_g2p_variant(Some(4 * GB), None), HebrewG2pVariant::Int8);
        assert_eq!(hebrew_g2p_variant(Some(2 * GB), None), HebrewG2pVariant::Int8);
        assert_eq!(hebrew_g2p_variant(None, None), HebrewG2pVariant::Fp32);
        assert_eq!(hebrew_g2p_variant(Some(16 * GB), Some("int8")), HebrewG2pVariant::Int8);
        assert_eq!(hebrew_g2p_variant(Some(2 * GB), Some("fp32")), HebrewG2pVariant::Fp32);
        assert_eq!(hebrew_g2p_variant(Some(2 * GB), Some("bogus")), HebrewG2pVariant::Int8);
    }

    #[test]
    fn hebrew_g2p_path_falls_back_to_the_installed_variant() {
        let dir = std::env::temp_dir().join(format!("mambotts-g2p-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fp32 = dir.join(HebrewG2pVariant::Fp32.file_name());
        let int8 = dir.join(HebrewG2pVariant::Int8.file_name());
        assert_eq!(hebrew_g2p_path(&dir, HebrewG2pVariant::Fp32), fp32);
        std::fs::write(&int8, b"x").unwrap();
        assert_eq!(hebrew_g2p_path(&dir, HebrewG2pVariant::Fp32), int8);
        std::fs::write(&fp32, b"x").unwrap();
        assert_eq!(hebrew_g2p_path(&dir, HebrewG2pVariant::Fp32), fp32);
        assert_eq!(hebrew_g2p_path(&dir, HebrewG2pVariant::Int8), int8);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    fn plan_dir(name: &str, files: &[(HebrewG2pVariant, u64)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("mambotts-plan-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (variant, len) in files {
            let file = std::fs::File::create(dir.join(variant.file_name())).unwrap();
            // Sparse: the full fp32 size costs no disk.
            file.set_len(*len).unwrap();
        }
        dir
    }

    #[test]
    fn an_int8_install_keeps_working_and_upgrades_to_fp32_in_the_background() {
        use HebrewG2pVariant::{Fp32, Int8};
        let int8 = Int8.model_file().size;
        let fp32 = Fp32.model_file().size;

        // Upgrader: int8 only, fp32 policy. Load int8 now, fetch fp32, keep int8.
        let dir = plan_dir("upgrader", &[(Int8, int8)]);
        let plan = hebrew_g2p_plan(&dir, Fp32);
        assert_eq!(plan, HebrewG2pPlan { load: Some(Int8), upgrade: Some(Fp32), remove: None });

        // A partial fp32 (wrong size) does not count; int8 is still loaded.
        std::fs::File::create(dir.join(Fp32.file_name())).unwrap().set_len(fp32 / 2).unwrap();
        let plan = hebrew_g2p_plan(&dir, Fp32);
        assert_eq!(plan, HebrewG2pPlan { load: Some(Int8), upgrade: Some(Fp32), remove: None });

        // fp32 verified: switch to it, and only now delete int8.
        std::fs::File::create(dir.join(Fp32.file_name())).unwrap().set_len(fp32).unwrap();
        let plan = hebrew_g2p_plan(&dir, Fp32);
        assert_eq!(plan, HebrewG2pPlan { load: Some(Fp32), upgrade: None, remove: Some(Int8) });
        std::fs::remove_dir_all(&dir).unwrap();

        // A low-memory machine never deletes or replaces a working fp32.
        let dir = plan_dir("lowmem", &[(Fp32, fp32)]);
        assert_eq!(
            hebrew_g2p_plan(&dir, Int8),
            HebrewG2pPlan { load: Some(Fp32), upgrade: None, remove: None }
        );
        // ...and with both files, keeps its own int8 and drops fp32.
        std::fs::File::create(dir.join(Int8.file_name())).unwrap().set_len(int8).unwrap();
        assert_eq!(
            hebrew_g2p_plan(&dir, Int8),
            HebrewG2pPlan { load: Some(Int8), upgrade: None, remove: Some(Fp32) }
        );
        std::fs::remove_dir_all(&dir).unwrap();

        // Nothing installed: nothing to load, and the bundle download (not a
        // background upgrade) fetches the policy's file.
        let dir = plan_dir("empty", &[]);
        assert_eq!(
            hebrew_g2p_plan(&dir, Fp32),
            HebrewG2pPlan { load: None, upgrade: None, remove: None }
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn every_file_has_a_size_and_a_sha256() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        for variant in [HebrewG2pVariant::Fp32, HebrewG2pVariant::Int8] {
            for file in blue.files_for(variant) {
                assert!(file.size > 0, "{}", file.name);
                assert_eq!(file.sha256.len(), 64, "{}", file.name);
                assert!(
                    file.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                    "{}",
                    file.name
                );
            }
        }
    }

    /// Every file of an installed bundle against the registry's hash:
    /// `MAMBOTTS_BLUE_MODEL_DIR=<bundle> cargo test -p mambotts-registry -- --ignored`.
    /// RenikudPlus files are checked when present (one variant is enough).
    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR (an installed bundle)"]
    fn an_installed_bundle_matches_the_registry_hashes() {
        use sha2::{Digest, Sha256};
        let dir = std::path::PathBuf::from(
            std::env::var("MAMBOTTS_BLUE_MODEL_DIR").expect("MAMBOTTS_BLUE_MODEL_DIR"),
        );
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        let hash = |path: &std::path::Path| -> String {
            Sha256::digest(std::fs::read(path).unwrap())
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect()
        };
        for file in blue.files {
            let path = dir.join(file.name);
            assert_eq!(std::fs::metadata(&path).unwrap().len(), file.size, "{}", file.name);
            assert_eq!(hash(&path), file.sha256, "{}", file.name);
        }
        let mut variants = 0;
        for variant in [HebrewG2pVariant::Fp32, HebrewG2pVariant::Int8] {
            let file = variant.model_file();
            let path = dir.join(file.name);
            if path.is_file() {
                variants += 1;
                assert_eq!(hash(&path), file.sha256, "{}", file.name);
            }
        }
        assert!(variants > 0, "no RenikudPlus in {}", dir.display());
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
        assert!(blue.hebrew_g2p);
        assert_eq!(blue.files.len(), blue.required_files.len());
        for variant in [HebrewG2pVariant::Fp32, HebrewG2pVariant::Int8] {
            let files = blue.files_for(variant);
            assert_eq!(files.len(), blue.files.len() + 1);
            assert_eq!(files.last().unwrap().name, variant.file_name());
        }
    }
}
