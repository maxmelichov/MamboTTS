//! The single source of truth for the model MamboTTS runs: BlueV3 v3.0.1,
//! its files, its voices and its Hebrew G2P policy. It is the only runtime.

use serde::Serialize;

pub const DEFAULT_RUNTIME_ID: &str = "blue";

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct RuntimeCapabilities {
    pub hebrew: bool,
    pub streaming: bool,
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
    /// The file's bytes, compiled in, for the files the model repository
    /// keeps in git rather than LFS (`vocab.json`, `tts.json`, the voices,
    /// two polyg2p lexicons, the licence): a squash of the Hub history
    /// deletes git-stored files at old revisions (LFS objects survive), so
    /// the app ships them instead of fetching `url`. `None`: download `url`.
    #[serde(skip)]
    pub shipped: Option<&'static [u8]>,
}

impl ModelFile {
    /// Whether the app carries this file itself ([`ModelFile::shipped`]).
    pub fn is_shipped(&self) -> bool {
        self.shipped.is_some()
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct RuntimeManifest {
    pub id: &'static str,
    pub name: &'static str,
    pub version: &'static str,
    pub size: &'static str,
    pub description: &'static str,
    pub directory: &'static str,
    pub files: &'static [ModelFile],
    pub required_files: &'static [&'static str],
    /// Directories of earlier releases of this bundle, newest first. An
    /// install found in one keeps working while the background upgrade
    /// fills `directory`, copying every file whose bytes did not change.
    pub previous_directories: &'static [&'static str],
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

// BlueV3 v3.0.1 (checkpoint 702000), pinned to BlueV3-onnx d0a0270e, the
// revision its MANIFEST.json (sizes and SHA-256 below) and BlueV3's
// `download_models.py` name, so a bundle is the same files on every install.
// The repository's history has been squashed before (d80497f, ba91a77), and a
// squashed-away revision stops resolving, so pin what the release pins.
// Against v3.0.0 (fb3dc38, checkpoint 701500) the text encoder, the duration
// predictor, the vector estimator, tts.json and every voice changed (new
// weights, a new duration model, and each voice carries `speed_scale`);
// vocab.json, the vocoder, polyg2p's data and RenikudPlus are byte-identical,
// so the upgrade from a 3.0.0 install copies them instead of downloading
// them again. Moving to a new release means a new revision here, new sizes
// and hashes, a new `directory`, and the old one in `previous_directories`.
const BLUE_MODEL_BASE_URL: &str =
    "https://huggingface.co/notmax123/BlueV3-onnx/resolve/d0a0270e60b96d5066cfe7ebed1f7c2872361c43";

macro_rules! blue_file {
    ($name:literal, $path:literal, $size:literal, $sha256:literal) => {
        blue_file!(@ $name, $path, $size, $sha256, None)
    };
    // A git-stored file of the model repository: shipped from `shipped/`.
    (shipped $name:literal, $path:literal, $size:literal, $sha256:literal) => {
        blue_file!(@ $name, $path, $size, $sha256, Some(include_bytes!(concat!("../shipped/", $name))))
    };
    (@ $name:literal, $path:literal, $size:literal, $sha256:literal, $shipped:expr) => {
        ModelFile {
            name: $name,
            url: concat!(
                "https://huggingface.co/notmax123/BlueV3-onnx/resolve/d0a0270e60b96d5066cfe7ebed1f7c2872361c43/",
                $path
            ),
            size: $size,
            sha256: $sha256,
            shipped: $shipped,
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
                shipped: None,
            },
            Self::Int8 => ModelFile {
                name: "renikud-plus.onnx",
                url: "https://huggingface.co/notmax123/RenikudPlus/resolve/679c56ca449d41873fb8ff7711ddf7563d28198f/model_int8.onnx",
                size: 311_605_741,
                sha256: "0337468fad56eadf5ecc7becfdf21648a4d668de41de5bc27e860141b5afcadd",
                shipped: None,
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
        "46197a2c10f06e151adc6dd109082889e6aaf903c46ce25d798eebaa7b07e313"
    ),
    blue_file!(
        "text_encoder.onnx",
        "onnx/text_encoder.onnx",
        27_951_369,
        "9dd84651a48048dce980d9d54c11fd3c04170602fb4f9e66957c08eeee2e55ba"
    ),
    blue_file!(
        "vector_estimator.onnx",
        "onnx/vector_estimator.onnx",
        256_556_761,
        "2f0e4af007f166afdc020331b863b1ea6b93c29258ae3b649f81234c1d7ad7c1"
    ),
    blue_file!(
        "vocoder.onnx",
        "onnx/vocoder.onnx",
        101_428_551,
        "054a700e9ef17d43de203212dca335fa2cd000cd9ca7701df49fcc544b49ccbe"
    ),
    blue_file!(
        shipped "vocab.json",
        "onnx/vocab.json",
        3_382,
        "9cf96e595f62ea8a36d3be6724b06fb821614c2938d0724852d1a25feff5bad5"
    ),
    blue_file!(
        shipped "tts.json",
        "onnx/tts.json",
        8_654,
        "38c65e35451577954dd94eaf7fc619b8231cc37b3328ffb95556fa821983ef48"
    ),
    blue_file!(
        shipped "voices/noa.json",
        "voices/noa.json",
        287_992,
        "34fea6e7746dc16399640604c8de87564bd759659ee91c536c1d30a3e9bab49d"
    ),
    blue_file!(
        shipped "voices/lily.json",
        "voices/lily.json",
        288_400,
        "8e3d320955ff8faf9ee5363111f9883ad476dd60d9919137d2851dd8c1ecefc1"
    ),
    blue_file!(
        shipped "voices/maya.json",
        "voices/maya.json",
        288_319,
        "e0e1ab7de876c6097b2c3e378ed4add4aef2744d98a7755c38e684f2d3760c41"
    ),
    blue_file!(
        shipped "voices/emma.json",
        "voices/emma.json",
        288_500,
        "e547c10547a306d3b81448c56e49a6ab03fababce23f15dc121eee19f4a1e9c4"
    ),
    blue_file!(
        shipped "voices/sara.json",
        "voices/sara.json",
        288_412,
        "373ff9b480cc74961c0df98f9c7cfad9feff000b5ab08237d0e22844546c97a4"
    ),
    blue_file!(
        shipped "voices/shira.json",
        "voices/shira.json",
        288_188,
        "b42d26f1cb8d6ae58b0df074eba753442b56291aaf8d6e5ad2ac45ab66efcd53"
    ),
    blue_file!(
        shipped "voices/tamar.json",
        "voices/tamar.json",
        288_221,
        "2ee2bcce78444da3f29417812695ae54d075b95fafaaa05eab40bb1e7ed19766"
    ),
    blue_file!(
        shipped "voices/yael.json",
        "voices/yael.json",
        288_284,
        "d8019210bc57798ae03448ecf71e58538d23a69c0c120f7989fa851fc99d1d5a"
    ),
    blue_file!(
        shipped "voices/adam.json",
        "voices/adam.json",
        288_306,
        "6b296fe8cb515aca19c5e6accdfc755fe9450867cdd2c4a223414b8e7a734d03"
    ),
    blue_file!(
        shipped "voices/daniel.json",
        "voices/daniel.json",
        288_934,
        "4114b4183caa2f6336f575d380b877ea23015adc92f8d5184c9fef1b031c143a"
    ),
    blue_file!(
        shipped "voices/ariel.json",
        "voices/ariel.json",
        288_782,
        "ac793ec51a7c4cf9fe4980fe254cbdb04f42ac2fbf37007b6e714a9348b9d6f2"
    ),
    blue_file!(
        shipped "voices/david.json",
        "voices/david.json",
        288_731,
        "07a2f81ac84142602c1c90dcea254af5c8497c4f116f71b2aaa403764c795945"
    ),
    blue_file!(
        shipped "voices/ethan.json",
        "voices/ethan.json",
        288_722,
        "441e9ed43f163784d435353fe8baca5fb90a978450542eb993d6e656ceef3ab0"
    ),
    blue_file!(
        shipped "voices/jonathan.json",
        "voices/jonathan.json",
        288_527,
        "90f38ae4f18f585419c2705ed5ce0099a4e591258f75d5e872536318ff16bcd9"
    ),
    blue_file!(
        shipped "voices/michael.json",
        "voices/michael.json",
        288_321,
        "6ddf319b702b438cc500bee7b85be47cc42d1259282b91428679636e37b2cd91"
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
        shipped "polyg2p/ru_homographs.lex",
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
        shipped "polyg2p/ru_yo_homographs.lex",
        "frontend/polyg2p_data/ru_yo_homographs.lex",
        19_043,
        "057cddf02713732706127274c796a6f418ad7096679ac532f7c31cf7c9266064"
    ),
    blue_file!(
        shipped "polyg2p/LICENSE",
        "frontend/polyg2p_data/LICENSE",
        11_357,
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
    blue_file!(
        shipped "polyg2p/NOTICE",
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
    version: "bluev3-3.0.1",
    size: "~740 MB",
    description: "Fast local speech for Hebrew, English, Spanish, German, Italian, and Russian.",
    // Each release in its own directory: its graphs and voices go together,
    // so a new release never lands on top of the one the app is running.
    directory: "bluev3-3.0.1",
    files: BLUE_FILES,
    required_files: BLUE_REQUIRED_FILES,
    previous_directories: &["bluev3-3.0.0"],
    hebrew_g2p: true,
    capabilities: RuntimeCapabilities {
        hebrew: true,
        streaming: true,
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

/// Resolves a voice name in any letter case to its catalog spelling, so
/// `noa` lands on `Noa`. Anything else, BlueTTS 2.5's `libri_*` voices
/// included, comes back unchanged and is rejected as an unknown voice.
pub fn blue_voice_name(id: &str) -> &str {
    let id = id.trim();
    BLUE_VOICES
        .iter()
        .find(|voice| voice.name.eq_ignore_ascii_case(id))
        .map(|voice| voice.name)
        .unwrap_or(id)
}

// BlueV3 v3.0.1 is the only model MamboTTS ships.
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
    fn only_the_fifteen_voices_resolve() {
        assert_eq!(blue_voice_name("libri_male_6209"), "libri_male_6209");
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
    fn the_release_is_v3_0_1_and_upgrades_from_v3_0_0() {
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        assert_eq!(blue.version, "bluev3-3.0.1");
        assert_eq!(blue.directory, "bluev3-3.0.1");
        assert_eq!(blue.previous_directories, ["bluev3-3.0.0"]);
        assert!(!blue.previous_directories.contains(&blue.directory));
        assert!(blue_model_base_url().ends_with("/d0a0270e60b96d5066cfe7ebed1f7c2872361c43"));
        // What v3.0.0 (fb3dc38) shipped and v3.0.1 did not change: the
        // upgrade takes these from the old directory instead of downloading.
        let unchanged = [
            ("vocab.json", "9cf96e595f62ea8a36d3be6724b06fb821614c2938d0724852d1a25feff5bad5"),
            ("vocoder.onnx", "054a700e9ef17d43de203212dca335fa2cd000cd9ca7701df49fcc544b49ccbe"),
            ("polyg2p/en.lex", "713aaed9ce1891e111eab2d213df560f043947b9268c6411b26a8c756ea99d32"),
            ("polyg2p/de.lex", "eeb7df564ef1bc894bf7cc39f0f054f5c516d38c2119922ac69ff8c31681594c"),
        ];
        for (name, sha256) in unchanged {
            let file = blue.files.iter().find(|file| file.name == name).unwrap();
            assert_eq!(file.sha256, sha256, "{name}");
        }
        // ...and what it did change: the graphs the voices belong to.
        let changed = [
            ("vector_estimator.onnx", "46abc33ac65e24cb5331b6d29ec4520b9a33a4a86a7723b5e44677af10d5b584"),
            ("voices/shira.json", "19611940a2cdc1b90ce70b13b12682c3c2b1ef6be14a78907c4fb8228a913ca3"),
        ];
        for (name, old) in changed {
            let file = blue.files.iter().find(|file| file.name == name).unwrap();
            assert_ne!(file.sha256, old, "{name}");
        }
    }

    /// The shipped files are the release's bytes, and exactly the files the
    /// model repository keeps in git (its tree API at d0a0270e lists every
    /// other one as LFS).
    #[test]
    fn shipped_files_are_the_git_stored_ones_byte_for_byte() {
        use sha2::{Digest, Sha256};
        let blue = runtime(DEFAULT_RUNTIME_ID).expect("Blue manifest");
        let mut shipped: Vec<&str> = Vec::new();
        for file in blue.files {
            if let Some(bytes) = file.shipped {
                assert_eq!(bytes.len() as u64, file.size, "{}", file.name);
                let hex: String = Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect();
                assert_eq!(hex, file.sha256, "{}", file.name);
                shipped.push(file.name);
            }
        }
        let mut expected = vec![
            "vocab.json",
            "tts.json",
            "polyg2p/ru_homographs.lex",
            "polyg2p/ru_yo_homographs.lex",
            "polyg2p/LICENSE",
            "polyg2p/NOTICE",
        ];
        expected.extend(BLUE_VOICES.iter().map(|voice| &*Box::leak(format!("voices/{}", voice.file).into_boxed_str())));
        shipped.sort_unstable();
        expected.sort_unstable();
        assert_eq!(shipped, expected);
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
