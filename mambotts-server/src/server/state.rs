use std::{path::Path, sync::Arc};

use anyhow::Result;
use tokio::sync::Mutex;

use crate::runtime::{BlueRuntime, Runtime, RuntimeParams};

pub type SharedServer = Arc<Server>;

pub struct LoadParams {
    pub runtime: String,
    pub params: RuntimeParams,
}

pub struct Server {
    pub(crate) inner: Mutex<ServerState>,
    pub(crate) version: String,
    pub(crate) commit: String,
}

pub(crate) struct ServerState {
    pub(crate) ctx: Option<Box<dyn Runtime>>,
    pub(crate) runtime: String,
    pub(crate) model_name: String,
    pub(crate) model_path: String,
    pub(crate) codec_path: String,
}

impl ServerState {
    /// Whether `runtime` is loaded from exactly these model files.
    fn has_loaded(&self, runtime: &str, model_dir: &Path, renikud_path: &Path) -> bool {
        self.ctx.is_some()
            && self.runtime == runtime
            && self.model_path == model_dir.display().to_string()
            && self.codec_path == renikud_path.display().to_string()
    }
}

impl Server {
    pub fn new(version: String, commit: String) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(ServerState {
                ctx: None,
                runtime: String::new(),
                model_name: String::new(),
                model_path: String::new(),
                codec_path: String::new(),
            }),
            version,
            commit,
        })
    }

    pub async fn load_model(&self, params: LoadParams) -> Result<()> {
        let mut inner = self.inner.lock().await;
        // The desktop sends a load before every Generate, Add diacritics and
        // Generate phonemes, to be sure a model is there. Rebuilding the
        // ONNX sessions each time cost 10 to 20 seconds per click, most of it
        // re-reading the 1.2 GB RenikudPlus model. When the same files are
        // already loaded, only the speaker hints can differ, and those are
        // plain fields on the phonemizer.
        let RuntimeParams::Blue {
            model_dir,
            renikud_path,
            speaker,
            target_speaker,
        } = &params.params;
        if inner.has_loaded(&params.runtime, model_dir, renikud_path) {
            if let Some(ctx) = inner.ctx.as_mut() {
                ctx.set_speakers(*speaker, *target_speaker);
                return Ok(());
            }
        }
        let (ctx, model_path, renikud_path): (Box<dyn Runtime>, _, _) = match params.params {
            RuntimeParams::Blue {
                model_dir,
                renikud_path,
                speaker,
                target_speaker,
            } => (
                Box::new(BlueRuntime::load(
                    model_dir.clone(),
                    renikud_path.clone(),
                    speaker,
                    target_speaker,
                )?),
                model_dir,
                renikud_path,
            ),
        };
        inner.ctx = Some(ctx);
        inner.runtime = params.runtime;
        inner.model_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .into();
        inner.model_path = model_path.display().to_string();
        inner.codec_path = renikud_path.display().to_string();
        Ok(())
    }

    pub(crate) async fn unload_model(&self) {
        let mut inner = self.inner.lock().await;
        inner.ctx = None;
        inner.runtime.clear();
        inner.model_name.clear();
        inner.model_path.clear();
        inner.codec_path.clear();
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::ServerState;

    fn state(loaded: bool) -> ServerState {
        ServerState {
            ctx: None,
            runtime: if loaded { "blue".into() } else { String::new() },
            model_name: String::new(),
            model_path: "/m/bluetts-2.5".into(),
            codec_path: "/m/bluetts-2.5/renikud-plus.onnx".into(),
        }
    }

    #[test]
    fn nothing_counts_as_loaded_without_a_runtime() {
        let model = Path::new("/m/bluetts-2.5");
        let renikud = Path::new("/m/bluetts-2.5/renikud-plus.onnx");
        // No ctx means nothing is loaded, whatever the recorded paths say.
        assert!(!state(true).has_loaded("blue", model, renikud));
        assert!(!state(false).has_loaded("blue", model, renikud));
    }
}
