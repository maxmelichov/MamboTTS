//! Where the flow runs: the GPU engine in this process, the GPU engine in a
//! worker process (`ve_worker.rs`), or neither (ONNX Runtime, the caller's
//! fallback), and the verdict cache that remembers what this machine's GPU did
//! (`gpu_verdict.rs`).
//!
//! In a worker, whatever the driver does (crash, hang, wrong numbers) costs
//! the GPU and never the process: the probe or the chunk is redone on ONNX
//! Runtime and the verdict is cached as failed, so the next start does not
//! try that adapter again until its driver or the app changes.
//!
//! The worker is the default everywhere a worker executable is at hand: on
//! an Apple M5 the boundary costs well under 1 ms per chunk (about 0.5% of a
//! one-minute text; `examples/gpu_isolation_bench.rs`). In-process
//! (`MAMBOTTS_GPU_ISOLATION=inprocess`, or no worker executable, as in the
//! Python module) the verdict is cached too: a probe that never reported back
//! (the process died mid-probe, perhaps by the driver's hand) is redone in a
//! worker when one is available, and counts as failed when not. A crash
//! in-process after the probe still takes the process down.

use std::{path::Path, time::Duration};

use anyhow::Result;

use crate::{
    gpu_verdict::{self, State, VerdictFile},
    ve_gpu::{self, AdapterId, GpuVectorEstimator, Opened},
    ve_worker::{self, Probe, WorkerCommand, WorkerFlow},
};

/// Where the GPU engine runs when it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isolation {
    InProcess,
    Worker,
}

impl Isolation {
    /// `MAMBOTTS_GPU_ISOLATION=process|inprocess`, else a worker.
    pub fn from_env() -> Self {
        match std::env::var("MAMBOTTS_GPU_ISOLATION").ok().as_deref().map(str::trim) {
            Some(v) if v.eq_ignore_ascii_case("process") || v.eq_ignore_ascii_case("worker") => Self::Worker,
            Some(v) if v.eq_ignore_ascii_case("inprocess") || v.eq_ignore_ascii_case("in-process") => Self::InProcess,
            _ => Self::platform_default(),
        }
    }

    /// A worker on every platform: the boundary costs about 0.5% (see the
    /// module docs), and a driver crash or hang then never costs speech.
    pub fn platform_default() -> Self {
        Self::Worker
    }
}

/// How [`crate::BlueTts`] opens the GPU flow engine.
#[derive(Clone, Debug)]
pub struct GpuOptions {
    /// Try the GPU at all (see `gpu_flow_wanted`, `MAMBOTTS_VE_BACKEND`).
    pub wanted: bool,
    pub isolation: Isolation,
    /// The worker executable. `None`: the engine can only run in-process.
    pub worker: Option<WorkerCommand>,
    pub verdicts: VerdictFile,
    /// Ignore cached failures and probe again (`MAMBOTTS_GPU_REPROBE=1`).
    pub reprobe: bool,
    /// The worker's test fault (`MAMBOTTS_GPU_WORKER_FAULT`, see `ve_worker.rs`).
    pub fault: Option<String>,
    /// The whole worker probe: start, device, kernels, self-check.
    pub open_timeout: Duration,
    /// A flow's deadline is this plus `flow_timeout_per_frame_step` per
    /// latent frame and step.
    pub flow_timeout: Duration,
    pub flow_timeout_per_frame_step: Duration,
}

impl GpuOptions {
    /// The options from the environment: `MAMBOTTS_VE_BACKEND`,
    /// `MAMBOTTS_GPU_ISOLATION`, `MAMBOTTS_GPU_WORKER`,
    /// `MAMBOTTS_GPU_VERDICT_FILE`, `MAMBOTTS_GPU_REPROBE`,
    /// `MAMBOTTS_GPU_WORKER_FAULT`, `MAMBOTTS_GPU_OPEN_TIMEOUT_SECS`,
    /// `MAMBOTTS_GPU_FLOW_TIMEOUT_SECS`.
    pub fn from_env() -> Self {
        let secs = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<f64>().ok())
                .filter(|s| s.is_finite() && *s > 0.0)
                .map(Duration::from_secs_f64)
        };
        Self {
            wanted: crate::gpu_flow_wanted(std::env::var("MAMBOTTS_VE_BACKEND").ok().as_deref()),
            isolation: Isolation::from_env(),
            worker: WorkerCommand::locate(),
            verdicts: VerdictFile::from_env(),
            reprobe: std::env::var("MAMBOTTS_GPU_REPROBE").is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true")),
            fault: std::env::var(ve_worker::FAULT_ENV).ok().filter(|v| !v.is_empty()),
            open_timeout: secs("MAMBOTTS_GPU_OPEN_TIMEOUT_SECS").unwrap_or(Duration::from_secs(180)),
            flow_timeout: secs("MAMBOTTS_GPU_FLOW_TIMEOUT_SECS").unwrap_or(Duration::from_secs(15)),
            flow_timeout_per_frame_step: Duration::from_millis(2),
        }
    }

    fn flow_deadline(&self, frames: usize, steps: usize) -> Duration {
        self.flow_timeout + self.flow_timeout_per_frame_step * (frames * steps.max(1)) as u32
    }
}

/// The GPU flow engine, wherever it runs.
pub struct GpuFlow {
    engine: Engine,
    /// The verdict this engine's failures are written to.
    key: String,
    verdicts: VerdictFile,
    opts: GpuOptions,
}

enum Engine {
    InProcess(GpuVectorEstimator),
    Worker(WorkerFlow),
}

impl GpuFlow {
    /// e.g. "Apple M3 Pro (Metal)".
    pub fn adapter(&self) -> &str {
        match &self.engine {
            Engine::InProcess(e) => e.adapter(),
            Engine::Worker(w) => w.label(),
        }
    }

    pub fn isolation(&self) -> Isolation {
        match self.engine {
            Engine::InProcess(_) => Isolation::InProcess,
            Engine::Worker(_) => Isolation::Worker,
        }
    }

    /// One chunk's flow (see [`GpuVectorEstimator::flow`]). An error has been
    /// recorded as this adapter's verdict; the caller drops the engine and
    /// runs the chunk on ONNX Runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn flow(
        &mut self,
        text_emb: &[f32],
        text_mask: &[f32],
        style_ttl: &[f32],
        latent_mask: &[f32],
        noisy_latent: &[f32],
        steps: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        let result = match &mut self.engine {
            Engine::InProcess(e) => e.flow(text_emb, text_mask, style_ttl, latent_mask, noisy_latent, steps, cfg_scale),
            Engine::Worker(w) => {
                let timeout = self.opts.flow_deadline(latent_mask.len(), steps);
                w.flow(text_emb, text_mask, style_ttl, latent_mask, noisy_latent, steps, cfg_scale, timeout)
            }
        };
        if let Err(err) = &result {
            self.verdicts.set(&self.key, &State::Failed { reason: format!("during synthesis: {err:#}") });
        }
        result
    }
}

/// Opens the GPU flow engine as `opts` says, trying each backend in order
/// and keeping the first that passes the self-check. `None`: ONNX Runtime
/// (not wanted, no adapter, every adapter failed now or before). Logs one
/// line per decision.
pub fn open(vector_estimator: &Path, opts: &GpuOptions) -> Option<GpuFlow> {
    if !opts.wanted {
        eprintln!("BlueV3 flow on ONNX Runtime (MAMBOTTS_VE_BACKEND=gpu tries the GPU engine)");
        return None;
    }
    let backends = ve_gpu::backend_candidates();
    if backends.is_empty() {
        eprintln!("BlueV3 flow on ONNX Runtime: no GPU backend here");
        return None;
    }
    let isolation = match (opts.isolation, &opts.worker) {
        (Isolation::Worker, None) => {
            eprintln!("BlueV3 GPU engine: no worker executable here, so it runs in-process");
            Isolation::InProcess
        }
        (i, _) => i,
    };
    for backend in backends {
        let flow = match isolation {
            Isolation::Worker => open_in_worker(vector_estimator, backend, opts),
            Isolation::InProcess => open_in_process(vector_estimator, backend, opts),
        };
        if let Some(flow) = flow {
            eprintln!(
                "BlueV3 flow on the GPU engine ({}, {})",
                flow.adapter(),
                match flow.isolation() {
                    Isolation::InProcess => "in-process",
                    Isolation::Worker => "worker process",
                }
            );
            return Some(flow);
        }
    }
    eprintln!("BlueV3 flow on ONNX Runtime: no GPU backend passed");
    None
}

/// What the cache says about trying an adapter (or a backend before its
/// adapter is known).
enum Decision {
    Try,
    Skip(String),
    /// An in-process probe died before: probe in a worker instead.
    Isolate,
}

fn decide(opts: &GpuOptions, key: &str, in_process: bool) -> Decision {
    if opts.reprobe {
        return Decision::Try;
    }
    match opts.verdicts.get(key) {
        Some(State::Failed { reason }) => Decision::Skip(reason),
        // The process died while probing in-process: maybe the driver, maybe
        // the user quitting during the compile.
        Some(State::Probing { isolated: false }) if in_process => match opts.worker {
            Some(_) => Decision::Isolate,
            None => Decision::Skip("the app stopped during the last GPU probe".into()),
        },
        // A worker probe the app did not see to its end (the app was quit):
        // nothing is known, so try again.
        _ => Decision::Try,
    }
}

fn open_in_worker(model: &Path, backend: &str, opts: &GpuOptions) -> Option<GpuFlow> {
    let worker = opts.worker.as_ref()?;
    let build = host_build();
    let bkey = gpu_verdict::backend_key(backend, &build);
    if let Decision::Skip(reason) = decide(opts, &bkey, false) {
        eprintln!("BlueV3 GPU engine: skipping {backend}, it failed before ({reason}); MAMBOTTS_GPU_REPROBE=1 retries");
        return None;
    }
    opts.verdicts.set(&bkey, &State::Probing { isolated: true });
    let mut key = None;
    let probe = ve_worker::probe(worker, backend, model, opts.fault.as_deref(), opts.open_timeout, &mut |id| {
        opts.verdicts.remove(&bkey);
        let k = gpu_verdict::adapter_key(id, &build);
        let go = match decide(opts, &k, false) {
            Decision::Skip(reason) => {
                eprintln!("BlueV3 GPU engine: skipping {}, it failed before ({reason}); MAMBOTTS_GPU_REPROBE=1 retries", id.label());
                false
            }
            _ => {
                opts.verdicts.set(&k, &State::Probing { isolated: true });
                true
            }
        };
        key = Some(k);
        go
    });
    match probe {
        Probe::Ready(w) => {
            let key = key.expect("a ready worker reported its adapter");
            opts.verdicts.set(&key, &State::Ok);
            Some(GpuFlow { engine: Engine::Worker(w), key, verdicts: opts.verdicts.clone(), opts: opts.clone() })
        }
        Probe::NoAdapter => {
            opts.verdicts.remove(&bkey);
            None
        }
        Probe::Declined => None,
        Probe::Failed { adapter, reason } => {
            record_failure(opts, adapter.as_ref(), key.as_deref(), &bkey, &reason);
            None
        }
    }
}

fn open_in_process(model: &Path, backend: &str, opts: &GpuOptions) -> Option<GpuFlow> {
    let build = host_build();
    let bkey = gpu_verdict::backend_key(backend, &build);
    match decide(opts, &bkey, true) {
        Decision::Skip(reason) => {
            eprintln!("BlueV3 GPU engine: skipping {backend}, it failed before ({reason}); MAMBOTTS_GPU_REPROBE=1 retries");
            return None;
        }
        Decision::Isolate => {
            eprintln!("BlueV3 GPU engine: the last in-process probe of {backend} never finished; probing it in a worker");
            return open_in_worker(model, backend, opts);
        }
        Decision::Try => {}
    }
    opts.verdicts.set(&bkey, &State::Probing { isolated: false });
    let mut key = None;
    let mut isolate = false;
    let opened = GpuVectorEstimator::open_backend(
        model,
        backend,
        &mut |id: &AdapterId| {
            opts.verdicts.remove(&bkey);
            let k = gpu_verdict::adapter_key(id, &build);
            let go = match decide(opts, &k, true) {
                Decision::Skip(reason) => {
                    eprintln!("BlueV3 GPU engine: skipping {}, it failed before ({reason}); MAMBOTTS_GPU_REPROBE=1 retries", id.label());
                    false
                }
                Decision::Isolate => {
                    isolate = true;
                    false
                }
                Decision::Try => {
                    opts.verdicts.set(&k, &State::Probing { isolated: false });
                    true
                }
            };
            key = Some(k);
            go
        },
        false,
    );
    match opened {
        Ok(Opened::Ready(engine)) => {
            let key = key.expect("an open engine reported its adapter");
            opts.verdicts.set(&key, &State::Ok);
            Some(GpuFlow { engine: Engine::InProcess(engine), key, verdicts: opts.verdicts.clone(), opts: opts.clone() })
        }
        Ok(Opened::NoAdapter) => {
            opts.verdicts.remove(&bkey);
            None
        }
        Ok(Opened::Declined) if isolate => {
            eprintln!("BlueV3 GPU engine: the last in-process probe of {backend} never finished; probing it in a worker");
            open_in_worker(model, backend, opts)
        }
        Ok(Opened::Declined) => None,
        Err(err) => {
            record_failure(opts, None, key.as_deref(), &bkey, &format!("{err:#}"));
            None
        }
    }
}

/// This app build, by the executable hosting the engine: its worker ships
/// and updates with it, so in-process and worker verdicts share keys.
fn host_build() -> String {
    gpu_verdict::build_id(std::env::current_exe().ok().as_deref())
}

fn record_failure(opts: &GpuOptions, adapter: Option<&AdapterId>, key: Option<&str>, bkey: &str, reason: &str) {
    let who = adapter.map(AdapterId::label).unwrap_or_else(|| bkey.split('|').next().unwrap_or("").to_owned());
    eprintln!("BlueV3 GPU engine failed on {who}, not trying it again until the driver or the app changes: {reason}");
    let state = State::Failed { reason: reason.to_owned() };
    match key {
        Some(k) => opts.verdicts.set(k, &state),
        None => opts.verdicts.set(bkey, &state),
    }
}
