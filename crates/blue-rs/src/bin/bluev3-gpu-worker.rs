//! The BlueV3 GPU flow engine as a worker process (see `blue_rs::ve_worker`),
//! for hosts that cannot re-invoke themselves as one (the Python module, the
//! tests). mambotts-server serves the same role itself.

fn main() {
    blue_rs::ve_worker::serve_if_requested();
    eprintln!(
        "bluev3-gpu-worker is started by the BlueV3 engine: bluev3-gpu-worker {} <backend> <vector_estimator.onnx>",
        blue_rs::ve_worker::WORKER_ARG
    );
    std::process::exit(2);
}
