use std::sync::Arc;

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
        let (ctx, model_path, renikud_path): (Box<dyn Runtime>, _, _) = match params.params {
            RuntimeParams::Blue {
                model_dir,
                renikud_path,
                hebrew_g2p_engine,
                phonikud_path,
                speaker,
                target_speaker,
            } => (
                Box::new(BlueRuntime::load(
                    model_dir.clone(),
                    renikud_path.clone(),
                    hebrew_g2p_engine,
                    phonikud_path,
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
