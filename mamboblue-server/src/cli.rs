use std::path::PathBuf;

use anyhow::Result;
use clap::{ArgAction, Parser, Subcommand};

use crate::{
    parent::watch_parent,
    runtime::RuntimeParams,
    server::{LoadParams, Server, listen_and_serve},
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const COMMIT: &str = match option_env!("MAMBORAMBO_COMMIT") {
    Some(commit) => commit,
    None => "dev",
};

#[derive(Debug, Parser)]
#[command(name = "mamborambo-server", version = VERSION, about = "MamboRambo local TTS server")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, short, default_value_t = 0)]
        port: u16,
        #[arg(long)]
        model_dir: Option<PathBuf>,
        #[arg(long)]
        renikud: Option<PathBuf>,
        #[arg(long, default_value_t = true, action = ArgAction::Set)]
        exit_with_parent: bool,
    },
}

pub async fn run() -> Result<()> {
    match Args::parse().command {
        Command::Serve {
            host,
            port,
            model_dir,
            renikud,
            exit_with_parent,
        } => {
            if exit_with_parent {
                tokio::spawn(watch_parent());
            }
            let server = Server::new(VERSION.into(), COMMIT.into());
            if model_dir.is_some() || renikud.is_some() {
                let (Some(model_dir), Some(renikud_path)) = (model_dir, renikud) else {
                    anyhow::bail!("--model-dir and --renikud must be provided together");
                };
                server
                    .load_model(LoadParams {
                        runtime: mamborambo_registry::DEFAULT_RUNTIME_ID.into(),
                        params: RuntimeParams::Blue {
                            model_dir,
                            renikud_path,
                            hebrew_g2p_engine: "renikud".into(),
                            phonikud_path: None,
                            speaker: 0,
                            target_speaker: 0,
                        },
                    })
                    .await?;
            }
            listen_and_serve(&host, port, server).await
        }
    }
}
