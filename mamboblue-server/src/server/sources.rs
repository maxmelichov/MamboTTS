use mamborambo_registry::{RuntimeManifest, runtimes};
use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelSourceFile {
    name: String,
    url: String,
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
            "macOS: ~/Library/Application Support/com.maxmelichov.mamborambo/models",
            "Windows: %LOCALAPPDATA%\\com.maxmelichov.mamborambo\\models",
            "Linux: ~/.local/share/com.maxmelichov.mamborambo/models",
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
        files: manifest
            .files
            .iter()
            .map(|file| ModelSourceFile {
                name: file.name.into(),
                url: file.url.into(),
            })
            .collect(),
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
        assert_eq!(
            sources.runtimes.len(),
            mamborambo_registry::runtimes().len()
        );
        assert_eq!(sources.runtimes[0].id, "blue");
        assert!(sources.runtimes[0].capabilities.hebrew);
    }
}
