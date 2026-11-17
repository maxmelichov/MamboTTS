//! The engine's start-up self-check on this machine's GPU (examples/selfcheck.rs as a test).
//!
//!   cargo test -p bluetts-engine --release -- --ignored --nocapture
//!
//! VE_ONNX overrides the model (default: the desktop app's BlueV3 download).

use bluetts_engine::engine::{map_read, selfcheck_compare, Engine};

/// The desktop app's BlueV3 download on this OS (Tauri's app data directory).
fn app_bundle_dir() -> std::path::PathBuf {
    let base = if cfg!(windows) {
        std::path::PathBuf::from(std::env::var_os("LOCALAPPDATA").expect("LOCALAPPDATA"))
    } else {
        let home = std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME"));
        if cfg!(target_os = "macos") { home.join("Library/Application Support") } else { home.join(".local/share") }
    };
    base.join("com.maxmelichov.mambotts/models/bluev3-3.0.1")
}

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
    let path = std::env::var_os("VE_ONNX")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| app_bundle_dir().join("vector_estimator.onnx"));
    let model = std::fs::read(&path).expect("model");
    // As the app does (blue-rs ve_gpu): each backend on its own, and the GPU
    // counts as usable when one of them computes correctly. A driver can be
    // wrong on one backend and right on another (an AMD Radeon iGPU on
    // Vulkan vs DX12), so one combined device would test whichever wgpu picks.
    let mut opened = 0;
    let mut passed = Vec::new();
    for backends in Engine::native_backend_candidates() {
        pollster::block_on(async {
            let (device, queue) = match Engine::new_device_on(backends, false).await {
                Ok(dq) => dq,
                Err(e) => {
                    eprintln!("{backends:?}: skipped: {e}");
                    return;
                }
            };
            opened += 1;
            let adapter = device.adapter_info().name;
            let mut e = Engine::load(device, queue, &model[..]).unwrap();
            let (a, b) = e.selfcheck_start().unwrap();
            let lat = map_read(e.device.clone(), a).await.unwrap();
            let out = map_read(e.device.clone(), b).await.unwrap();
            let r = selfcheck_compare(&lat, &out);
            eprintln!("{backends:?} on {adapter}: {} values, max|d| {:.3e}, SNR {:.1} dB", r.values, r.max_abs, r.snr_db);
            if r.finite && r.max_abs <= 1e-3 {
                passed.push(backends);
            }
        });
    }
    if opened == 0 {
        eprintln!("skipped: no GPU adapter");
        return;
    }
    assert!(!passed.is_empty(), "no backend passed the self-check (see stderr)");
}
