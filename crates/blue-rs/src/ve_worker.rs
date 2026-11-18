//! The GPU flow engine in a worker process.
//!
//! A graphics driver can crash or hang inside wgpu, and Rust cannot catch
//! either: in-process, the speech server would go down with it. So the engine
//! can run in a child process instead (the host executable re-invoked with
//! [`WORKER_ARG`], or the `bluev3-gpu-worker` binary), which owns wgpu and
//! serves whole flows over its stdin and stdout. The parent treats the worker
//! exiting, a broken pipe, a malformed or non-finite answer, or a call past
//! its deadline as "the GPU failed": it kills the worker and the caller runs
//! that chunk on ONNX Runtime.
//!
//! A flow moves one chunk's inputs in and its final latent out (about 1 MB
//! for the longest chunk), once per chunk rather than once per step, so pipes
//! are enough.
//!
//! Wire format, both directions: a one-byte kind, a little-endian `u64`
//! length, then that many bytes.
//!
//! ```text
//! worker -> parent  ADAPTER (JSON AdapterId)   after the device opens, before anything compiles
//! parent -> worker  GO | DECLINE
//! worker -> parent  READY (label) | NO_ADAPTER | ERROR (message), then for READY:
//! parent -> worker  FLOW (5 x u32/f32 header, then the five input tensors as f32)
//! worker -> parent  LATENT (f32) | ERROR (message)
//! ```
//!
//! The worker exits when its stdin closes, so it never outlives the parent.
//!
//! Test hook: `MAMBOTTS_GPU_WORKER_FAULT=<what>@<when>` in the worker's
//! environment, `<what>` one of `abort` (a native crash), `hang`, `exit`
//! (a clean exit with status 3) or `wrong` (the self-check reads wrong numbers;
//! in a flow, the latent comes back with NaNs), `<when>` one of `adapter`
//! (before the device is reported), `open` (after it, while the kernels
//! compile) or `flow:N` (the N-th flow, from 1).

use std::{
    ffi::OsString,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, RecvTimeoutError, Sender},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};

use crate::ve_gpu::{AdapterId, GpuVectorEstimator, Opened};

/// The hidden first argument that turns a host executable into the worker.
pub const WORKER_ARG: &str = "--bluev3-gpu-worker";

/// The environment variable of the test hook (see the module docs).
pub const FAULT_ENV: &str = "MAMBOTTS_GPU_WORKER_FAULT";

const ADAPTER: u8 = 1;
const READY: u8 = 2;
const NO_ADAPTER: u8 = 3;
const ERROR: u8 = 4;
const LATENT: u8 = 5;
const GO: u8 = 10;
const DECLINE: u8 = 11;
const FLOW: u8 = 12;

/// No message is anywhere near this; a length past it is a broken stream.
const MAX_FRAME: u64 = 1 << 30;

static HOST_SERVES_WORKER: AtomicBool = AtomicBool::new(false);

/// Call first thing in `main` of an executable that may host the worker
/// (mambotts-server does). Invoked as `<exe> --bluev3-gpu-worker <backend>
/// <vector_estimator.onnx>`, it serves as the worker and exits; otherwise it
/// returns at once, and the GPU engine knows it can re-invoke this executable
/// as its worker (see [`WorkerCommand::locate`]).
pub fn serve_if_requested() {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(WORKER_ARG.as_ref()) {
        HOST_SERVES_WORKER.store(true, Ordering::SeqCst);
        return;
    }
    let backend = args.next().and_then(|a| a.into_string().ok()).unwrap_or_default();
    let model = args.next().map(PathBuf::from).unwrap_or_default();
    let code = match serve(&backend, &model) {
        Ok(()) => 0,
        Err(err) => {
            eprintln!("BlueV3 GPU worker: {err:#}");
            1
        }
    };
    std::process::exit(code);
}

/// How to start the worker: a program and the arguments before
/// [`WORKER_ARG`]'s.
#[derive(Clone, Debug)]
pub struct WorkerCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

impl WorkerCommand {
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self { program: program.into(), args: Vec::new() }
    }

    /// `MAMBOTTS_GPU_WORKER` (an executable that serves [`WORKER_ARG`]), else
    /// this executable if its `main` called [`serve_if_requested`], else
    /// `bluev3-gpu-worker` beside it. `None`: no worker here (the Python
    /// module, a test binary), so the engine can only run in-process.
    pub fn locate() -> Option<Self> {
        if let Some(p) = std::env::var_os("MAMBOTTS_GPU_WORKER").filter(|p| !p.is_empty()) {
            return Some(Self::new(p));
        }
        let exe = std::env::current_exe().ok()?;
        if HOST_SERVES_WORKER.load(Ordering::SeqCst) {
            return Some(Self::new(exe));
        }
        let beside = exe.parent()?.join(format!("bluev3-gpu-worker{}", std::env::consts::EXE_SUFFIX));
        beside.is_file().then(|| Self::new(beside))
    }
}

// ---------------------------------------------------------------- the worker

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultKind {
    Abort,
    Hang,
    Exit,
    Wrong,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultStage {
    Adapter,
    Open,
    Flow(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Fault {
    kind: FaultKind,
    stage: FaultStage,
}

impl Fault {
    fn parse(spec: &str) -> Option<Self> {
        let (kind, stage) = spec.trim().split_once('@')?;
        let kind = match kind {
            "abort" => FaultKind::Abort,
            "hang" => FaultKind::Hang,
            "exit" => FaultKind::Exit,
            "wrong" => FaultKind::Wrong,
            _ => return None,
        };
        let stage = match stage {
            "adapter" => FaultStage::Adapter,
            "open" => FaultStage::Open,
            s => FaultStage::Flow(s.strip_prefix("flow:")?.parse().ok().filter(|n| *n > 0)?),
        };
        Some(Self { kind, stage })
    }

    fn from_env() -> Option<Self> {
        let spec = std::env::var(FAULT_ENV).ok()?;
        let fault = Self::parse(&spec);
        if fault.is_none() {
            eprintln!("BlueV3 GPU worker: ignoring {FAULT_ENV}={spec:?}");
        }
        fault
    }

    /// Crashes, hangs or exits here if that is the fault; `true` for `wrong`.
    fn strike(fault: Option<Self>, stage: FaultStage) -> bool {
        let Some(f) = fault.filter(|f| f.stage == stage) else { return false };
        eprintln!("BlueV3 GPU worker: test fault {:?} at {:?}", f.kind, f.stage);
        match f.kind {
            FaultKind::Abort => std::process::abort(),
            FaultKind::Hang => loop {
                thread::sleep(Duration::from_secs(3600));
            },
            FaultKind::Exit => std::process::exit(3),
            FaultKind::Wrong => true,
        }
    }
}

/// The worker's side: open `backend`, then serve flows until stdin closes.
fn serve(backend: &str, model: &Path) -> Result<()> {
    let fault = Fault::from_env();
    // Stdin is read on its own thread, so a closed pipe (the parent gone or
    // done with us) ends the process even while the GPU is busy or stuck.
    let (tx, rx) = mpsc::channel::<(u8, Vec<u8>)>();
    thread::Builder::new().name("bluev3-gpu-worker-stdin".into()).spawn(move || {
        let mut stdin = io::stdin().lock();
        while let Ok(frame) = read_frame(&mut stdin) {
            if tx.send(frame).is_err() {
                break;
            }
        }
        std::process::exit(0);
    })?;
    let mut out = io::stdout().lock();

    Fault::strike(fault, FaultStage::Adapter);
    let mut send_err: Option<io::Error> = None;
    let mut on_adapter = |id: &AdapterId| -> bool {
        let msg = json!({
            "backend": id.backend, "name": id.name, "vendor": id.vendor, "device": id.device,
            "driver": id.driver, "driver_info": id.driver_info,
        });
        if let Err(e) = write_frame(&mut out, ADAPTER, msg.to_string().as_bytes()) {
            send_err = Some(e);
            return false;
        }
        match rx.recv() {
            Ok((GO, _)) => true,
            _ => false,
        }
    };
    // A `wrong@open` fault corrupts the self-check; the others strike once the
    // parent has the adapter, while the kernels compile, which is where a bad
    // driver tends to fall over.
    let wrong_open = fault.is_some_and(|f| f.kind == FaultKind::Wrong && f.stage == FaultStage::Open);
    let opened = GpuVectorEstimator::open_backend(
        model,
        backend,
        &mut |id| {
            let go = on_adapter(id);
            if go && !wrong_open {
                Fault::strike(fault, FaultStage::Open);
            }
            go
        },
        wrong_open,
    );
    if let Some(e) = send_err {
        return Err(e).context("report the adapter");
    }
    let mut engine = match opened {
        Ok(Opened::Ready(engine)) => engine,
        Ok(Opened::NoAdapter) => return Ok(write_frame(&mut out, NO_ADAPTER, &[])?),
        Ok(Opened::Declined) => return Ok(()),
        Err(err) => {
            write_frame(&mut out, ERROR, format!("{err:#}").as_bytes())?;
            return Ok(());
        }
    };
    write_frame(&mut out, READY, engine.adapter().as_bytes())?;

    let mut flows = 0u64;
    while let Ok((kind, payload)) = rx.recv() {
        if kind != FLOW {
            bail!("unexpected message {kind}");
        }
        flows += 1;
        let wrong = Fault::strike(fault, FaultStage::Flow(flows));
        let result = FlowRequest::decode(&payload).and_then(|r| {
            engine.flow(&r.text_emb, &r.text_mask, &r.style_ttl, &r.latent_mask, &r.noisy, r.steps as usize, r.cfg_scale)
        });
        match result {
            Ok(mut latent) => {
                if wrong {
                    latent.iter_mut().step_by(5).for_each(|v| *v = f32::NAN);
                }
                write_frame(&mut out, LATENT, bytemuck::cast_slice(&latent))?;
            }
            Err(err) => write_frame(&mut out, ERROR, format!("{err:#}").as_bytes())?,
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- the parent

/// How a probe ended.
pub enum Probe {
    Ready(WorkerFlow),
    NoAdapter,
    /// `on_adapter` declined the adapter.
    Declined,
    /// The worker failed: crashed, hung past the deadline, failed the
    /// self-check or could not open the engine. `adapter` is what it had
    /// reported by then, if anything.
    Failed { adapter: Option<AdapterId>, reason: String },
}

/// Starts a worker on `backend` and waits, up to `timeout` in all, for it to
/// open the engine and pass the self-check. `on_adapter` sees the adapter
/// before anything is compiled on it and may decline it.
pub fn probe(
    cmd: &WorkerCommand,
    backend: &str,
    model: &Path,
    fault: Option<&str>,
    timeout: Duration,
    on_adapter: &mut dyn FnMut(&AdapterId) -> bool,
) -> Probe {
    let deadline = Instant::now() + timeout;
    let mut command = Command::new(&cmd.program);
    command
        .args(&cmd.args)
        .arg(WORKER_ARG)
        .arg(backend)
        .arg(model)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    match fault {
        Some(f) => command.env(FAULT_ENV, f),
        None => command.env_remove(FAULT_ENV),
    };
    #[cfg(windows)]
    {
        // The server runs without a console; without this a console
        // executable it starts gets a console window of its own.
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let mut link = match Link::spawn(command) {
        Ok(link) => link,
        Err(err) => return Probe::Failed { adapter: None, reason: format!("could not start the GPU worker: {err:#}") },
    };
    let adapter = match link.recv(deadline) {
        Ok((ADAPTER, payload)) => match parse_adapter(&payload) {
            Ok(id) => id,
            Err(err) => return link.fail(None, format!("{err:#}")),
        },
        Ok((NO_ADAPTER, _)) => {
            link.shut_down();
            return Probe::NoAdapter;
        }
        Ok((ERROR, msg)) => return link.fail(None, String::from_utf8_lossy(&msg).into_owned()),
        Ok((kind, _)) => return link.fail(None, format!("unexpected message {kind} from the GPU worker")),
        Err(err) => return link.fail(None, err),
    };
    if !on_adapter(&adapter) {
        let _ = link.send(DECLINE, Vec::new());
        link.shut_down();
        return Probe::Declined;
    }
    if let Err(err) = link.send(GO, Vec::new()) {
        return link.fail(Some(adapter), err);
    }
    match link.recv(deadline) {
        Ok((READY, label)) => Probe::Ready(WorkerFlow {
            label: String::from_utf8_lossy(&label).into_owned(),
            adapter,
            link,
        }),
        Ok((NO_ADAPTER, _)) => {
            link.shut_down();
            Probe::NoAdapter
        }
        Ok((ERROR, msg)) => link.fail(Some(adapter), String::from_utf8_lossy(&msg).into_owned()),
        Ok((kind, _)) => link.fail(Some(adapter), format!("unexpected message {kind} from the GPU worker")),
        Err(err) => link.fail(Some(adapter), err),
    }
}

fn parse_adapter(payload: &[u8]) -> Result<AdapterId> {
    let v: Value = serde_json::from_slice(payload).context("the GPU worker's adapter report")?;
    let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_owned();
    let n = |k: &str| v.get(k).and_then(Value::as_u64).unwrap_or(0) as u32;
    Ok(AdapterId {
        backend: s("backend"),
        name: s("name"),
        vendor: n("vendor"),
        device: n("device"),
        driver: s("driver"),
        driver_info: s("driver_info"),
    })
}

/// A worker that opened the engine and passed the self-check.
pub struct WorkerFlow {
    label: String,
    adapter: AdapterId,
    link: Link,
}

impl WorkerFlow {
    /// e.g. "Apple M3 Pro (Metal)".
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn adapter(&self) -> &AdapterId {
        &self.adapter
    }

    /// As [`GpuVectorEstimator::flow`], within `timeout`. Any error leaves the
    /// worker killed; drop this then.
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
        timeout: Duration,
    ) -> Result<Vec<f32>> {
        let deadline = Instant::now() + timeout;
        let payload = FlowRequest::encode(text_emb, text_mask, style_ttl, latent_mask, noisy_latent, steps, cfg_scale);
        if let Err(err) = self.link.send(FLOW, payload) {
            self.link.kill();
            bail!("GPU worker: {err}");
        }
        let reply = match self.link.recv(deadline) {
            Ok(r) => r,
            Err(err) => {
                self.link.kill();
                bail!("GPU worker: {err}");
            }
        };
        let latent = match reply {
            (LATENT, bytes) if bytes.len() % 4 == 0 => bytemuck::pod_collect_to_vec::<u8, f32>(&bytes),
            (ERROR, msg) => {
                self.link.kill();
                bail!("GPU worker: {}", String::from_utf8_lossy(&msg));
            }
            (kind, _) => {
                self.link.kill();
                bail!("GPU worker: unexpected message {kind}");
            }
        };
        if latent.len() != noisy_latent.len() {
            self.link.kill();
            bail!("GPU worker: the latent has {} values, expected {}", latent.len(), noisy_latent.len());
        }
        if latent.iter().any(|v| !v.is_finite()) {
            self.link.kill();
            bail!("GPU worker: the latent is not finite");
        }
        Ok(latent)
    }
}

/// The pipes to one worker: frames out through a writer thread (so a stuck
/// worker cannot block the caller on a full pipe), frames in through a reader
/// thread, so every wait has a deadline.
struct Link {
    child: Option<Child>,
    tx: Option<Sender<(u8, Vec<u8>)>>,
    rx: Receiver<io::Result<(u8, Vec<u8>)>>,
}

impl Link {
    fn spawn(mut command: Command) -> Result<Self> {
        let mut child = command.spawn().with_context(|| format!("spawn {:?}", command.get_program()))?;
        let stdin: ChildStdin = child.stdin.take().context("worker stdin")?;
        let mut stdout = child.stdout.take().context("worker stdout")?;
        let (out_tx, out_rx) = mpsc::channel::<(u8, Vec<u8>)>();
        thread::Builder::new().name("bluev3-gpu-link-out".into()).spawn(move || {
            let mut stdin = io::BufWriter::new(stdin);
            for (kind, payload) in out_rx {
                if write_frame(&mut stdin, kind, &payload).is_err() {
                    break;
                }
            }
        })?;
        let (in_tx, in_rx) = mpsc::channel();
        thread::Builder::new().name("bluev3-gpu-link-in".into()).spawn(move || {
            loop {
                let frame = read_frame(&mut stdout);
                let end = frame.is_err();
                if in_tx.send(frame).is_err() || end {
                    break;
                }
            }
        })?;
        Ok(Self { child: Some(child), tx: Some(out_tx), rx: in_rx })
    }

    fn send(&self, kind: u8, payload: Vec<u8>) -> Result<(), String> {
        self.tx
            .as_ref()
            .and_then(|tx| tx.send((kind, payload)).ok())
            .ok_or_else(|| "the GPU worker is gone".to_owned())
    }

    fn recv(&mut self, deadline: Instant) -> Result<(u8, Vec<u8>), String> {
        let wait = deadline.saturating_duration_since(Instant::now());
        match self.rx.recv_timeout(wait) {
            Ok(Ok(frame)) => Ok(frame),
            Ok(Err(_)) | Err(RecvTimeoutError::Disconnected) => {
                // Give the exit status a moment to be there for the message.
                let status = self.wait_status(Duration::from_millis(500));
                Err(format!("the GPU worker stopped answering ({status})"))
            }
            Err(RecvTimeoutError::Timeout) => Err("the GPU worker did not answer in time (hung?)".into()),
        }
    }

    fn wait_status(&mut self, within: Duration) -> String {
        let Some(child) = self.child.as_mut() else { return "gone".into() };
        let until = Instant::now() + within;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return format!("exited: {status}"),
                Ok(None) if Instant::now() < until => thread::sleep(Duration::from_millis(10)),
                _ => return "pipe closed".into(),
            }
        }
    }

    fn fail(mut self, adapter: Option<AdapterId>, reason: String) -> Probe {
        self.kill();
        Probe::Failed { adapter, reason }
    }

    /// Kills the worker now; it is reaped on a background thread.
    fn kill(&mut self) {
        self.tx = None;
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            thread::spawn(move || {
                let _ = child.wait();
            });
        }
    }

    /// Closes the worker's stdin (it exits on its own), and kills it if it is
    /// still there after a moment.
    fn shut_down(&mut self) {
        self.tx = None;
        if let Some(mut child) = self.child.take() {
            thread::spawn(move || {
                let until = Instant::now() + Duration::from_secs(2);
                while Instant::now() < until {
                    if let Ok(Some(_)) = child.try_wait() {
                        return;
                    }
                    thread::sleep(Duration::from_millis(20));
                }
                let _ = child.kill();
                let _ = child.wait();
            });
        }
    }
}

impl Drop for Link {
    fn drop(&mut self) {
        self.shut_down();
    }
}

struct FlowRequest {
    text_emb: Vec<f32>,
    text_mask: Vec<f32>,
    style_ttl: Vec<f32>,
    latent_mask: Vec<f32>,
    noisy: Vec<f32>,
    steps: u32,
    cfg_scale: f32,
}

impl FlowRequest {
    fn encode(text_emb: &[f32], text_mask: &[f32], style_ttl: &[f32], latent_mask: &[f32], noisy: &[f32], steps: usize, cfg_scale: f32) -> Vec<u8> {
        let parts = [text_emb, text_mask, style_ttl, latent_mask, noisy];
        let mut out = Vec::with_capacity(28 + 4 * parts.iter().map(|p| p.len()).sum::<usize>());
        out.extend_from_slice(&(steps as u32).to_le_bytes());
        out.extend_from_slice(&cfg_scale.to_le_bytes());
        for p in parts {
            out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        }
        for p in parts {
            out.extend_from_slice(bytemuck::cast_slice(p));
        }
        out
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let word = |i: usize| -> Result<[u8; 4]> {
            bytes.get(4 * i..4 * i + 4).and_then(|b| b.try_into().ok()).ok_or_else(|| anyhow!("short flow request"))
        };
        let steps = u32::from_le_bytes(word(0)?);
        let cfg_scale = f32::from_le_bytes(word(1)?);
        let mut lens = [0usize; 5];
        for (i, len) in lens.iter_mut().enumerate() {
            *len = u32::from_le_bytes(word(2 + i)?) as usize;
        }
        let mut at = 28usize;
        let mut take = |n: usize| -> Result<Vec<f32>> {
            let b = bytes.get(at..at + 4 * n).ok_or_else(|| anyhow!("short flow request"))?;
            at += 4 * n;
            Ok(bytemuck::pod_collect_to_vec::<u8, f32>(b))
        };
        Ok(Self {
            text_emb: take(lens[0])?,
            text_mask: take(lens[1])?,
            style_ttl: take(lens[2])?,
            latent_mask: take(lens[3])?,
            noisy: take(lens[4])?,
            steps,
            cfg_scale,
        })
    }
}

fn write_frame(w: &mut impl Write, kind: u8, payload: &[u8]) -> io::Result<()> {
    w.write_all(&[kind])?;
    w.write_all(&(payload.len() as u64).to_le_bytes())?;
    w.write_all(payload)?;
    w.flush()
}

fn read_frame(r: &mut impl Read) -> io::Result<(u8, Vec<u8>)> {
    let mut head = [0u8; 9];
    r.read_exact(&mut head)?;
    let len = u64::from_le_bytes(head[1..9].try_into().expect("8 bytes"));
    if len > MAX_FRAME {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("frame of {len} bytes")));
    }
    let mut payload = vec![0u8; len as usize];
    r.read_exact(&mut payload)?;
    Ok((head[0], payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faults_parse() {
        assert_eq!(Fault::parse("abort@open"), Some(Fault { kind: FaultKind::Abort, stage: FaultStage::Open }));
        assert_eq!(Fault::parse("hang@flow:3"), Some(Fault { kind: FaultKind::Hang, stage: FaultStage::Flow(3) }));
        assert_eq!(Fault::parse("wrong@adapter"), Some(Fault { kind: FaultKind::Wrong, stage: FaultStage::Adapter }));
        assert_eq!(Fault::parse("hang@flow:0"), None);
        assert_eq!(Fault::parse("explode@open"), None);
        assert_eq!(Fault::parse("abort"), None);
    }

    #[test]
    fn flow_requests_round_trip() {
        let a: Vec<f32> = (0..256 * 3).map(|i| i as f32 * 0.5).collect();
        let req = FlowRequest::encode(&a, &[1.0; 3], &[0.25; 512], &[1.0; 7], &[-2.0; 144 * 7], 8, 4.0);
        let back = FlowRequest::decode(&req).unwrap();
        assert_eq!(back.text_emb, a);
        assert_eq!(back.style_ttl.len(), 512);
        assert_eq!(back.noisy, vec![-2.0; 144 * 7]);
        assert_eq!((back.steps, back.cfg_scale), (8, 4.0));
        assert!(FlowRequest::decode(&req[..req.len() - 1]).is_err());
    }

    #[test]
    fn frames_round_trip() {
        let mut buf = Vec::new();
        write_frame(&mut buf, LATENT, &[1, 2, 3]).unwrap();
        write_frame(&mut buf, GO, &[]).unwrap();
        let mut r = &buf[..];
        assert_eq!(read_frame(&mut r).unwrap(), (LATENT, vec![1, 2, 3]));
        assert_eq!(read_frame(&mut r).unwrap(), (GO, vec![]));
        assert!(read_frame(&mut r).is_err());
    }
}
