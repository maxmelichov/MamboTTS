use mambotts_registry::{
    HEBREW_G2P_INT8_MAX_MEMORY, HebrewG2pVariant, ModelFile, RuntimeManifest, runtimes,
};
use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSourceFile {
    name: String,
    url: String,
    /// Exact size in bytes.
    size: u64,
    /// Lowercase hex SHA-256.
    sha256: String,
}

impl From<&ModelFile> for ModelSourceFile {
    fn from(file: &ModelFile) -> Self {
        Self {
            name: file.name.into(),
            url: file.url.into(),
            size: file.size,
            sha256: file.sha256.into(),
        }
    }
}

/// One RenikudPlus weight file; a bundle needs exactly one of them, which
/// `files` leaves out because the machine decides which.
#[derive(Debug, Serialize, ToSchema)]
pub struct HebrewG2pSource {
    /// `fp32` or `int8`.
    #[schema(value_type = String)]
    variant: HebrewG2pVariant,
    #[serde(flatten)]
    file: ModelSourceFile,
}

/// bluetts.com's choice between the variants: fp32, unless the machine has
/// `int8_max_memory` bytes of memory or less; `MAMBOTTS_HEBREW_G2P=fp32|int8`
/// overrides it.
#[derive(Debug, Serialize, ToSchema)]
pub struct HebrewG2pPolicy {
    #[schema(value_type = String)]
    default: HebrewG2pVariant,
    #[schema(value_type = String)]
    low_memory: HebrewG2pVariant,
    int8_max_memory: u64,
    override_env: &'static str,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RuntimeCapabilitiesResponse {
    hebrew: bool,
    streaming: bool,
    voice_reference: bool,
    fixed_voices: bool,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSource {
    id: String,
    name: String,
    version: String,
    size: String,
    description: String,
    files: Vec<ModelSourceFile>,
    /// The RenikudPlus variants, empty for a runtime without Hebrew G2P.
    hebrew_g2p: Vec<HebrewG2pSource>,
    hebrew_g2p_policy: Option<HebrewG2pPolicy>,
    directory: String,
    capabilities: RuntimeCapabilitiesResponse,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSourcesResponse {
    runtimes: Vec<ModelSource>,
    voices_url: &'static str,
    default_paths: Vec<&'static str>,
}

pub fn model_sources() -> ModelSourcesResponse {
    ModelSourcesResponse {
        runtimes: runtimes().iter().map(source_from_manifest).collect(),
        voices_url: "",
        default_paths: vec![
            "macOS: ~/Library/Application Support/com.maxmelichov.mambotts/models",
            "Windows: %LOCALAPPDATA%\\com.maxmelichov.mambotts\\models",
            "Linux: ~/.local/share/com.maxmelichov.mambotts/models",
        ],
    }
}

fn source_from_manifest(manifest: &RuntimeManifest) -> ModelSource {
    ModelSource {
        id: manifest.id.into(),
        name: manifest.name.into(),
        version: manifest.version.into(),
        size: manifest.size.into(),
        description: manifest.description.into(),
        files: manifest.files.iter().map(ModelSourceFile::from).collect(),
        hebrew_g2p: if manifest.hebrew_g2p {
            [HebrewG2pVariant::Fp32, HebrewG2pVariant::Int8]
                .into_iter()
                .map(|variant| HebrewG2pSource {
                    variant,
                    file: ModelSourceFile::from(&variant.model_file()),
                })
                .collect()
        } else {
            Vec::new()
        },
        hebrew_g2p_policy: manifest.hebrew_g2p.then_some(HebrewG2pPolicy {
            default: HebrewG2pVariant::Fp32,
            low_memory: HebrewG2pVariant::Int8,
            int8_max_memory: HEBREW_G2P_INT8_MAX_MEMORY,
            override_env: "MAMBOTTS_HEBREW_G2P",
        }),
        directory: manifest.directory.into(),
        capabilities: RuntimeCapabilitiesResponse {
            hebrew: manifest.capabilities.hebrew,
            streaming: manifest.capabilities.streaming,
            voice_reference: manifest.capabilities.voice_reference,
            fixed_voices: manifest.capabilities.fixed_voices,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::model_sources;

    #[test]
    fn exposes_every_registered_runtime() {
        let sources = model_sources();
        assert_eq!(sources.runtimes.len(), mambotts_registry::runtimes().len());
        assert_eq!(sources.runtimes[0].id, "blue");
        assert!(sources.runtimes[0].capabilities.hebrew);
    }

    #[test]
    fn the_hebrew_g2p_variants_are_listed_with_their_file_names() {
        let json = serde_json::to_value(model_sources()).unwrap();
        let blue = &json["runtimes"][0];
        assert_eq!(blue["hebrew_g2p"][0]["variant"], "fp32");
        assert_eq!(blue["hebrew_g2p"][0]["name"], "renikud-plus-fp32.onnx");
        assert_eq!(blue["hebrew_g2p"][1]["variant"], "int8");
        assert_eq!(blue["hebrew_g2p"][1]["name"], "renikud-plus.onnx");
        assert_eq!(blue["hebrew_g2p_policy"]["int8_max_memory"], 4u64 << 30);
        assert_eq!(blue["files"][0]["sha256"].as_str().unwrap().len(), 64);
    }

    #[test]
    fn each_source_is_its_registry_manifest() {
        for (source, manifest) in model_sources()
            .runtimes
            .iter()
            .zip(mambotts_registry::runtimes())
        {
            assert_eq!(source.id, manifest.id);
            assert_eq!(source.version, manifest.version);
            assert_eq!(source.directory, manifest.directory);
            assert_eq!(source.description, manifest.description);
            assert_eq!(source.files.len(), manifest.files.len());
            for (file, expected) in source.files.iter().zip(manifest.files) {
                assert_eq!(
                    (
                        file.name.as_str(),
                        file.url.as_str(),
                        file.size,
                        file.sha256.as_str()
                    ),
                    (expected.name, expected.url, expected.size, expected.sha256)
                );
            }
            assert_eq!(
                source.hebrew_g2p.len(),
                if manifest.hebrew_g2p { 2 } else { 0 }
            );
            assert_eq!(
                source.capabilities.voice_reference,
                manifest.capabilities.voice_reference
            );
        }
    }
}
