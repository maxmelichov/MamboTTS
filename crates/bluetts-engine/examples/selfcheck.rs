//! The start-up self-check natively (Metal through wgpu): the same inputs and
//! golden output the browser engine checks itself against (selfcheck/).
//!
//!   cargo run --release --example selfcheck -- <vector_estimator.onnx>
//!
//! Also writes the inputs as case files for examples/parity.rs when given a
//! directory as a second argument.

use bluetts_engine::engine::{map_read, selfcheck_compare, Engine};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = std::fs::read(&args[1]).expect("model");
    pollster::block_on(async {
        let (device, queue) = Engine::new_device(false).await.unwrap();
        let mut e = Engine::load(device, queue, &model[..]).unwrap();
        let (a, b) = e.selfcheck_start().unwrap();
        let lat = map_read(e.device.clone(), a).await.unwrap();
        let out = map_read(e.device.clone(), b).await.unwrap();
        let r = selfcheck_compare(&lat, &out);
        println!(
            "selfcheck: {} values, max|d| {:.3e} (reference max {:.3}), SNR {:.1} dB, finite {}",
            r.values, r.max_abs, r.ref_max, r.snr_db, r.finite
        );
    });
}
