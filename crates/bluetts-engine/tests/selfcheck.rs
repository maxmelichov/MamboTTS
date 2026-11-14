//! The engine's start-up self-check on this machine's GPU (examples/selfcheck.rs as a test).
//!
//!   cargo test -p bluetts-engine --release -- --ignored --nocapture
//!
//! VE_ONNX overrides the model (default: the desktop app's BlueV3 download).

use bluetts_engine::engine::{map_read, selfcheck_compare, Engine};

#[test]
fn every_kernel_has_an_entry_point() {
    let sources = Engine::kernel_sources();
    assert!(!sources.is_empty());
    for s in &sources {
        assert!(s.contains("fn main"), "kernel without `main`:\n{s}");
    }
}

#[test]
#[ignore = "needs vector_estimator.onnx and a GPU"]
fn selfcheck_matches_golden() {
    let path = std::env::var_os("VE_ONNX").map(std::path::PathBuf::from).unwrap_or_else(|| {
        let home = std::env::var_os("HOME").expect("HOME");
        std::path::Path::new(&home).join("Library/Application Support/com.maxmelichov.mambotts/models/bluev3-3.0.0/vector_estimator.onnx")
    });
    let model = std::fs::read(&path).expect("model");
    pollster::block_on(async {
        let (device, queue) = match Engine::new_device(false).await {
            Ok(dq) => dq,
            Err(e) => {
                eprintln!("skipped: {e}");
                return;
            }
        };
        let mut e = Engine::load(device, queue, &model[..]).unwrap();
        let (a, b) = e.selfcheck_start().unwrap();
        let lat = map_read(e.device.clone(), a).await.unwrap();
        let out = map_read(e.device.clone(), b).await.unwrap();
        let r = selfcheck_compare(&lat, &out);
        eprintln!("selfcheck: {} values, max|d| {:.3e}, SNR {:.1} dB", r.values, r.max_abs, r.snr_db);
        assert!(r.finite && r.max_abs <= 1e-3, "max|d| {:.3e}", r.max_abs);
    });
}
