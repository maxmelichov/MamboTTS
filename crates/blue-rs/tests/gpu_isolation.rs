//! The GPU flow engine in a worker process: whatever the driver does (crash,
//! hang, wrong numbers, at open or at the N-th chunk), synthesis completes
//! with ONNX Runtime's audio and the verdict is cached so the next start
//! skips that adapter. The faults are the worker's test hook
//! (`MAMBOTTS_GPU_WORKER_FAULT`, `blue_rs::ve_worker`); they stand in for the
//! Windows drivers that did this for real.
//!
//!   MAMBOTTS_BLUE_MODEL_DIR=<bundle> cargo test -p blue-rs --release \
//!       --test gpu_isolation -- --ignored --nocapture --test-threads=2
//!
//! Each test that needs a GPU skips (passing) when the machine has none.

use std::{
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Duration,
};

use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    gpu_flow::{GpuOptions, Isolation},
    gpu_verdict::{State, VerdictFile},
    phonemize::{Language, Phonemizer},
    ve_worker::WorkerCommand,
};
use serde_json::Value;

/// About 40 s of speech: four or five chunks, so a fault at the second flow
/// leaves one GPU chunk before it and several ONNX Runtime chunks after.
const TEXT: &str = "The lighthouse keeper climbed the narrow stairs every evening, counting each step as his father had taught him. \
At the top he trimmed the wick, polished the great lens, and watched the ships slide past the rocks in the fading light. \
Some nights the fog rolled in so thick that he could not see his own hand, and then he rang the bell until morning. \
Sailors in the harbor town said they owed him their lives, though few of them had ever met him in person. \
He did not mind the solitude; he had his books, his radio, and a cat named Captain who slept beside the stove. \
When the coast guard finally replaced the lamp with an automatic beacon, he packed his things and walked down the stairs for the last time. \
He still visits on clear evenings, and the new light blinks at him as if it remembers.";

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("MAMBOTTS_BLUE_MODEL_DIR").expect("set MAMBOTTS_BLUE_MODEL_DIR to an installed BlueV3 bundle"))
}

fn worker() -> WorkerCommand {
    WorkerCommand::new(env!("CARGO_BIN_EXE_bluev3-gpu-worker"))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("blue-rs-gpu-isolation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.json"));
    let _ = std::fs::remove_file(&path);
    path
}

fn options(verdicts: &Path, fault: Option<&str>) -> GpuOptions {
    GpuOptions {
        wanted: true,
        isolation: Isolation::Worker,
        worker: Some(worker()),
        verdicts: VerdictFile::at(verdicts),
        reprobe: false,
        fault: fault.map(str::to_owned),
        open_timeout: Duration::from_secs(20),
        flow_timeout: Duration::from_secs(4),
        flow_timeout_per_frame_step: Duration::ZERO,
    }
}

fn english() -> SynthesisOptions {
    SynthesisOptions { lang: "en".to_owned(), ..SynthesisOptions::default() }
}

fn voice(dir: &Path) -> VoiceStyle {
    VoiceStyle::from_json(dir.join("voices/adam.json")).unwrap()
}

fn phonemizer(dir: &Path) -> Phonemizer {
    Phonemizer::open(None::<&Path>, Some(dir.join("polyg2p")), Language::English).unwrap()
}

/// Synthesizes [`TEXT`]; returns the audio and each chunk's sample range.
fn speak(tts: &mut BlueTts) -> (Vec<f32>, Vec<std::ops::Range<usize>>) {
    let dir = model_dir();
    let audio = tts.synthesize_text(&mut phonemizer(&dir), TEXT, &voice(&dir), english()).unwrap();
    let gap = (tts.sample_rate() as f64 * 0.1).round() as usize;
    let mut at = 0;
    let ranges = tts
        .last_chunks()
        .iter()
        .map(|c| {
            let r = at..at + c.samples;
            at += c.samples + gap;
            r
        })
        .collect();
    (audio, ranges)
}

/// [`TEXT`] on ONNX Runtime alone, once per test binary.
fn reference() -> &'static (Vec<f32>, Vec<std::ops::Range<usize>>) {
    static REF: OnceLock<(Vec<f32>, Vec<std::ops::Range<usize>>)> = OnceLock::new();
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|p| p.into_inner());
    REF.get_or_init(|| {
        let mut opts = options(&scratch("reference"), None);
        opts.wanted = false;
        let mut tts = BlueTts::from_dir_with(model_dir(), opts).unwrap();
        let r = speak(&mut tts);
        assert!(r.1.len() >= 3, "the test text should make at least three chunks, got {}", r.1.len());
        r
    })
}

fn open(verdicts: &Path, fault: Option<&str>) -> BlueTts {
    open_with(options(verdicts, fault))
}

fn open_with(opts: GpuOptions) -> BlueTts {
    let mut tts = BlueTts::from_dir_with(model_dir(), opts).unwrap();
    tts.wait_for_gpu();
    tts
}

/// Whether this machine has a GPU the engine passes on (in a fault-free
/// worker); the fault tests skip without one.
fn have_gpu() -> bool {
    static HAVE: OnceLock<bool> = OnceLock::new();
    *HAVE.get_or_init(|| {
        let tts = open(&scratch("have-gpu"), None);
        let have = tts.gpu_adapter().is_some();
        if !have {
            eprintln!("no GPU the engine passes on here: the isolation tests skip");
        }
        have
    })
}

fn verdicts(path: &Path) -> Vec<(String, String)> {
    let raw = std::fs::read(path).unwrap_or_default();
    let v: Value = serde_json::from_slice(&raw).unwrap_or(Value::Null);
    v.as_object()
        .map(|m| m.iter().map(|(k, e)| (k.clone(), e["state"].as_str().unwrap_or("").to_owned())).collect())
        .unwrap_or_default()
}

fn assert_failed_verdict(path: &Path) {
    let all = verdicts(path);
    assert!(all.iter().any(|(_, s)| s == "failed"), "expected a failed verdict in {path:?}: {all:?}");
    assert!(!all.iter().any(|(_, s)| s == "probing"), "a probe was left unfinished in {path:?}: {all:?}");
}

/// SNR of `got` against `want` in dB.
fn snr(got: &[f32], want: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        sig += (*w as f64).powi(2);
        err += (*g as f64 - *w as f64).powi(2);
    }
    10.0 * (sig / err.max(1e-30)).log10()
}

/// A fault while the engine opens: the flow never leaves ONNX Runtime, the
/// audio is ONNX Runtime's bit for bit, the failure is cached, and the next
/// start skips the adapter without starting a probe that could crash again.
fn fault_at_open(name: &str, fault: &str) {
    if !have_gpu() {
        return;
    }
    let file = scratch(name);
    let mut tts = open(&file, Some(fault));
    assert_eq!(tts.gpu_adapter(), None, "{fault}: the GPU engine should have failed to open");
    let (audio, _) = speak(&mut tts);
    assert!(audio == reference().0, "{fault}: the audio is not ONNX Runtime's");
    assert_failed_verdict(&file);

    // The next start, without the fault: the cached failure keeps it off...
    let again = open(&file, None);
    assert_eq!(again.gpu_adapter(), None, "{fault}: a cached failure should skip the adapter");
    // ...until a re-probe is asked for.
    let mut opts = options(&file, None);
    opts.reprobe = true;
    let reprobed = open_with(opts);
    assert!(reprobed.gpu_adapter().is_some(), "{fault}: a re-probe should open the GPU engine");
    assert!(verdicts(&file).iter().any(|(_, s)| s == "ok"));
}

/// A fault at the N-th flow: the chunks before it are the GPU's (ONNX
/// Runtime's to within the self-check's tolerance), that chunk and every
/// later one are ONNX Runtime's bit for bit, and the failure is cached.
fn fault_at_flow(name: &str, fault: &str, n: usize) {
    if !have_gpu() {
        return;
    }
    let (want, ranges) = reference();
    let file = scratch(name);
    let mut tts = open(&file, Some(fault));
    assert!(tts.gpu_adapter().is_some(), "{fault}: the GPU engine should open");
    assert_eq!(tts.gpu_isolation(), Some(Isolation::Worker));
    let (audio, got_ranges) = speak(&mut tts);
    assert_eq!(&got_ranges, ranges, "{fault}: the chunks differ");
    assert_eq!(audio.len(), want.len());
    assert_eq!(tts.gpu_adapter(), None, "{fault}: the failed engine should be dropped");
    for (i, r) in ranges.iter().enumerate() {
        let (g, w) = (&audio[r.clone()], &want[r.clone()]);
        if i + 1 < n {
            let s = snr(g, w);
            assert!(s > 30.0, "{fault}: GPU chunk {i} is too far from ONNX Runtime's ({s:.1} dB)");
        } else {
            assert!(g == w, "{fault}: chunk {i} (after the fault) is not ONNX Runtime's");
        }
    }
    // Everything from the fault on, gaps included, is ONNX Runtime's.
    let from = ranges[n - 1].start;
    assert!(audio[from..] == want[from..]);
    assert_failed_verdict(&file);
    assert_eq!(open(&file, None).gpu_adapter(), None, "{fault}: a cached failure should skip the adapter");
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn crash_at_adapter_falls_back_and_is_remembered() {
    fault_at_open("abort-adapter", "abort@adapter");
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn crash_at_open_falls_back_and_is_remembered() {
    fault_at_open("abort-open", "abort@open");
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn hang_at_open_times_out_falls_back_and_is_remembered() {
    fault_at_open("hang-open", "hang@open");
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn wrong_numbers_at_open_fail_the_self_check_and_are_remembered() {
    fault_at_open("wrong-open", "wrong@open");
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn crash_at_the_first_flow_gives_onnx_runtime_audio() {
    fault_at_flow("abort-flow1", "abort@flow:1", 1);
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn exit_at_the_second_flow_redoes_that_chunk_on_onnx_runtime() {
    fault_at_flow("exit-flow2", "exit@flow:2", 2);
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn hang_at_the_second_flow_times_out_and_redoes_that_chunk() {
    fault_at_flow("hang-flow2", "hang@flow:2", 2);
}

#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn wrong_numbers_at_the_third_flow_are_caught_and_redone() {
    fault_at_flow("wrong-flow3", "wrong@flow:3", 3);
}

/// No fault: the worker's audio is the in-process engine's bit for bit (the
/// same kernels, a deterministic flow) and ONNX Runtime's to within the
/// self-check's tolerance; the verdict is `ok`.
#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn the_worker_speaks_as_the_in_process_engine_does() {
    if !have_gpu() {
        return;
    }
    let file = scratch("happy");
    let mut in_worker = open(&file, None);
    assert_eq!(in_worker.gpu_isolation(), Some(Isolation::Worker));
    let (a, _) = speak(&mut in_worker);
    assert!(in_worker.gpu_adapter().is_some(), "the worker should still be in use");
    let mut opts = options(&scratch("happy-inproc"), None);
    opts.isolation = Isolation::InProcess;
    let mut in_process = open_with(opts);
    assert_eq!(in_process.gpu_isolation(), Some(Isolation::InProcess));
    let (b, _) = speak(&mut in_process);
    assert!(a == b, "the worker's audio differs from the in-process engine's");
    let s = snr(&a, &reference().0);
    eprintln!("GPU vs ONNX Runtime audio: {s:.1} dB");
    assert!(s > 30.0);
    assert!(verdicts(&file).iter().all(|(_, s)| s == "ok"), "{:?}", verdicts(&file));
}

/// An in-process probe that never reported back (the process died during
/// it) is redone in a worker rather than in-process again; once that passes,
/// the next start is in-process as usual.
#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle) and a GPU"]
fn an_unfinished_in_process_probe_is_redone_in_a_worker() {
    if !have_gpu() {
        return;
    }
    let file = scratch("stale-probe");
    let mut opts = options(&file, None);
    opts.isolation = Isolation::InProcess;
    drop(open_with(opts.clone()));
    let all = verdicts(&file);
    let (key, _) = all.iter().find(|(_, s)| s == "ok").expect("an ok verdict").clone();
    VerdictFile::at(&file).set(&key, &State::Probing { isolated: false });

    let escalated = open_with(opts.clone());
    assert_eq!(escalated.gpu_isolation(), Some(Isolation::Worker));
    assert_eq!(VerdictFile::at(&file).get(&key), Some(State::Ok));
    drop(escalated);
    assert_eq!(open_with(opts).gpu_isolation(), Some(Isolation::InProcess));
}

/// A worker executable that is not there: ONNX Runtime, no panic, and a
/// failure recorded for the backend.
#[test]
#[ignore = "needs MAMBOTTS_BLUE_MODEL_DIR (a BlueV3 bundle)"]
fn a_missing_worker_means_onnx_runtime() {
    let file = scratch("missing-worker");
    let mut opts = options(&file, None);
    opts.worker = Some(WorkerCommand::new("/nonexistent/bluev3-gpu-worker"));
    let mut tts = open_with(opts);
    assert_eq!(tts.gpu_adapter(), None);
    let (audio, _) = speak(&mut tts);
    assert!(audio == reference().0);
}
