mod cli;
mod parent;
mod runtime;
mod server;

#[tokio::main]
async fn main() {
    if let Err(err) = cli::run().await {
        eprintln!("{err}");
        std::process::exit(1);
    }
}
