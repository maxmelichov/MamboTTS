mod cli;
mod parent;
mod runtime;
mod server;

fn main() {
    // `mambotts-server --bluev3-gpu-worker ...` is the BlueV3 GPU flow engine's
    // worker process (blue_rs::ve_worker): it serves and exits here, before
    // any CLI parsing or runtime. Otherwise this marks the executable as one
    // that can be re-invoked as the worker.
    blue_rs::ve_worker::serve_if_requested();
    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(runtime) => runtime,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };
    if let Err(err) = runtime.block_on(cli::run()) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
