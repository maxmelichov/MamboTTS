use std::{
    collections::HashMap,
    path::Path,
    sync::{
        Arc, Mutex as StdMutex, RwLock, RwLockReadGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use anyhow::Result;
use tokio::sync::Mutex;

use crate::runtime::{BlueRuntime, Language, Runtime, RuntimeParams};

pub type SharedServer = Arc<Server>;

pub struct LoadParams {
    pub runtime: String,
    pub params: RuntimeParams,
}

type Engine = Option<Box<dyn Runtime>>;

/// The server keeps two kinds of state apart.
///
/// The engine is `&mut` for the whole of a synthesis, which on a long document
/// is minutes, so it lives behind its own lock that only engine work takes.
/// Everything a status request wants to know about the loaded model is copied
/// into `info` when the model is loaded, so `/health`, `/v1/models`,
/// `/v1/voices`, `/v1/languages` and `/v1/phonemes` answer immediately even
/// while a generation holds the engine.
pub struct Server {
    engine: Arc<Mutex<Engine>>,
    info: RwLock<ModelInfo>,
    generations: Generations,
    pub(crate) version: String,
    pub(crate) commit: String,
}

/// A snapshot of the loaded model, readable without touching the engine.
#[derive(Default)]
pub(crate) struct ModelInfo {
    pub(crate) loaded: bool,
    pub(crate) runtime: String,
    pub(crate) model_name: String,
    pub(crate) model_path: String,
    pub(crate) codec_path: String,
    pub(crate) languages: Vec<Language>,
    pub(crate) voices: Option<Vec<String>>,
    pub(crate) phonemes: Vec<char>,
}

impl ModelInfo {
    /// Whether `runtime` is loaded from exactly these model files.
    fn has_loaded(&self, runtime: &str, model_dir: &Path, renikud_path: &Path) -> bool {
        self.loaded
            && self.runtime == runtime
            && self.model_path == model_dir.display().to_string()
            && self.codec_path == renikud_path.display().to_string()
    }
}

/// Why a piece of engine work did not produce a result.
#[derive(Debug)]
pub(crate) enum EngineError {
    NoModel,
    Cancelled,
    Failed(anyhow::Error),
}

/// Returned from a synthesis callback to stop the run. Kept as its own type so
/// the handler can tell a requested stop apart from a real failure.
#[derive(Debug, thiserror::Error)]
#[error("generation cancelled")]
pub(crate) struct Cancelled;

impl Server {
    pub fn new(version: String, commit: String) -> Arc<Self> {
        Arc::new(Self {
            engine: Arc::new(Mutex::new(None)),
            info: RwLock::new(ModelInfo::default()),
            generations: Generations::default(),
            version,
            commit,
        })
    }

    pub(crate) fn info(&self) -> RwLockReadGuard<'_, ModelInfo> {
        self.info
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn set_info(&self, info: ModelInfo) {
        *self
            .info
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = info;
    }

    pub async fn load_model(&self, params: LoadParams) -> Result<()> {
        // Loading replaces the engine, so it waits for the running generation
        // like any other engine work. Status requests keep answering from the
        // previous snapshot until the new model is in place.
        let mut engine = self.engine.lock().await;
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
        if self
            .info()
            .has_loaded(&params.runtime, model_dir, renikud_path)
        {
            if let Some(ctx) = engine.as_mut() {
                ctx.set_speakers(*speaker, *target_speaker);
                return Ok(());
            }
        }
        let (ctx, model_path, renikud_path) = tokio::task::spawn_blocking(move || {
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
            anyhow::Ok((ctx, model_path, renikud_path))
        })
        .await??;
        self.set_info(ModelInfo {
            loaded: true,
            runtime: params.runtime,
            model_name: model_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default()
                .into(),
            model_path: model_path.display().to_string(),
            codec_path: renikud_path.display().to_string(),
            languages: ctx.languages().to_vec(),
            voices: ctx.voices(),
            phonemes: ctx.supported_phonemes(),
        });
        *engine = Some(ctx);
        Ok(())
    }

    pub(crate) async fn unload_model(&self) {
        let mut engine = self.engine.lock().await;
        *engine = None;
        self.set_info(ModelInfo::default());
    }

    /// Run blocking work against the loaded engine on the blocking pool.
    ///
    /// Waiting for the engine is asynchronous, so queued requests do not tie
    /// up runtime threads, and the work itself never runs on one either.
    pub(crate) async fn with_engine<T, F>(&self, work: F) -> Result<T, EngineError>
    where
        T: Send + 'static,
        F: FnOnce(&mut dyn Runtime) -> Result<T> + Send + 'static,
    {
        let mut engine = self.engine.clone().lock_owned().await;
        tokio::task::spawn_blocking(move || {
            let ctx = engine.as_mut().ok_or(EngineError::NoModel)?;
            work(ctx.as_mut()).map_err(|err| {
                if err.is::<Cancelled>() {
                    EngineError::Cancelled
                } else {
                    EngineError::Failed(err)
                }
            })
        })
        .await
        .map_err(|err| EngineError::Failed(anyhow::anyhow!("engine task failed: {err}")))?
    }

    /// Register a generation so it can be cancelled from another request.
    pub(crate) fn begin_generation(self: &Arc<Self>) -> Generation {
        let id = self.generations.next.fetch_add(1, Ordering::Relaxed) + 1;
        let cancel = Arc::new(AtomicBool::new(false));
        self.generations.active().insert(id, cancel.clone());
        Generation {
            id,
            cancel,
            server: self.clone(),
        }
    }

    /// Cancel one generation by id, or every generation (running and queued)
    /// when no id is given. Returns how many were signalled.
    pub(crate) fn cancel_generations(&self, id: Option<u64>) -> usize {
        let active = self.generations.active();
        let flags: Vec<_> = match id {
            Some(id) => active.get(&id).into_iter().collect(),
            None => active.values().collect(),
        };
        for flag in &flags {
            flag.store(true, Ordering::SeqCst);
        }
        flags.len()
    }

    pub(crate) fn active_generations(&self) -> usize {
        self.generations.active().len()
    }
}

#[derive(Default)]
struct Generations {
    next: AtomicU64,
    active: StdMutex<HashMap<u64, Arc<AtomicBool>>>,
}

impl Generations {
    fn active(&self) -> std::sync::MutexGuard<'_, HashMap<u64, Arc<AtomicBool>>> {
        self.active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// One registered synthesis. It stays cancellable until it is dropped, which
/// happens when its work finishes.
pub(crate) struct Generation {
    id: u64,
    cancel: Arc<AtomicBool>,
    server: SharedServer,
}

impl Generation {
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn cancel_flag(&self) -> CancelFlag {
        CancelFlag(self.cancel.clone())
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        self.server.generations.active().remove(&self.id);
    }
}

/// A handle on a generation's cancellation state.
#[derive(Clone)]
pub(crate) struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    /// Stop the run if it has been cancelled. Called between chunks.
    pub(crate) fn check(&self) -> Result<()> {
        if self.is_cancelled() {
            Err(Cancelled.into())
        } else {
            Ok(())
        }
    }

    /// Cancel when the returned guard is dropped. Tying this to the response
    /// body (or the handler future) means a client that goes away stops the
    /// work instead of leaving it running for nobody.
    pub(crate) fn cancel_on_drop(&self) -> CancelOnDrop {
        CancelOnDrop(self.clone())
    }
}

pub(crate) struct CancelOnDrop(CancelFlag);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelling_by_id_only_touches_that_generation() {
        let server = Server::new("test".into(), "test".into());
        let first = server.begin_generation();
        let second = server.begin_generation();
        assert_eq!(server.active_generations(), 2);

        assert_eq!(server.cancel_generations(Some(first.id())), 1);
        assert!(first.cancel_flag().is_cancelled());
        assert!(!second.cancel_flag().is_cancelled());
        assert!(first.cancel_flag().check().unwrap_err().is::<Cancelled>());

        assert_eq!(server.cancel_generations(Some(9999)), 0);
        assert_eq!(server.cancel_generations(None), 2);
        assert!(second.cancel_flag().is_cancelled());
    }

    #[test]
    fn finished_generations_are_no_longer_cancellable() {
        let server = Server::new("test".into(), "test".into());
        let generation = server.begin_generation();
        let id = generation.id();
        drop(generation);
        assert_eq!(server.active_generations(), 0);
        assert_eq!(server.cancel_generations(Some(id)), 0);
    }

    #[test]
    fn dropping_the_guard_cancels() {
        let server = Server::new("test".into(), "test".into());
        let generation = server.begin_generation();
        let flag = generation.cancel_flag();
        drop(flag.cancel_on_drop());
        assert!(flag.is_cancelled());
    }

    #[tokio::test]
    async fn status_does_not_wait_for_the_engine() {
        let server = Server::new("test".into(), "test".into());
        let _held = server.engine.clone().lock_owned().await;
        // With the engine held, as it is for a whole generation, the model
        // snapshot is still readable.
        assert!(!server.info().loaded);
        let queued = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            server.with_engine(|_| Ok(())),
        )
        .await;
        assert!(queued.is_err(), "engine work should wait for the holder");
    }

    fn info(loaded: bool) -> ModelInfo {
        ModelInfo {
            loaded,
            runtime: "blue".into(),
            model_path: "/m/bluetts-2.5".into(),
            codec_path: "/m/bluetts-2.5/renikud-plus.onnx".into(),
            ..ModelInfo::default()
        }
    }

    #[test]
    fn nothing_counts_as_loaded_until_a_model_is_in_place() {
        let model = Path::new("/m/bluetts-2.5");
        let renikud = Path::new("/m/bluetts-2.5/renikud-plus.onnx");
        // The recorded paths mean nothing while no model is loaded, which is
        // what keeps the reuse shortcut from skipping the very first load.
        assert!(!info(false).has_loaded("blue", model, renikud));
        assert!(info(true).has_loaded("blue", model, renikud));
    }

    #[test]
    fn a_different_model_is_not_the_loaded_one() {
        let renikud = Path::new("/m/bluetts-2.5/renikud-plus.onnx");
        assert!(!info(true).has_loaded("blue", Path::new("/m/other"), renikud));
        assert!(!info(true).has_loaded(
            "blue",
            Path::new("/m/bluetts-2.5"),
            Path::new("/m/other/renikud-plus.onnx"),
        ));
        assert!(!info(true).has_loaded("kokoro", Path::new("/m/bluetts-2.5"), renikud));
    }
}
