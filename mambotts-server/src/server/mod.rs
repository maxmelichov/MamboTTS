use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use tokio::net::TcpListener;

mod docs;
mod dto;
mod errors;
mod handlers;
mod skill;
mod sources;
mod state;
mod util;

pub use sources::model_sources;
pub use state::{LoadParams, Server, SharedServer};

pub async fn listen_and_serve(host: &str, port: u16, server: SharedServer) -> Result<()> {
    let listener = TcpListener::bind((host, port)).await?;
    let actual_port = listener.local_addr()?.port();
    println!(
        "{}",
        serde_json::json!({
            "status": "ready",
            "port": actual_port,
            "version": server.version,
            "commit": server.commit,
        })
    );
    axum::serve(listener, router(server))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn router(server: SharedServer) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/skill", get(handlers::skill_handler))
        .route("/v1/languages", get(handlers::languages))
        .route("/v1/voices", get(handlers::voices))
        .route(
            "/v1/models",
            get(handlers::models).delete(handlers::model_unload),
        )
        .route("/v1/models/sources", get(handlers::model_sources_handler))
        .route("/v1/models/load", post(handlers::model_load))
        .route("/v1/phonemize", post(handlers::phonemize))
        .route("/v1/diacritize", post(handlers::diacritize))
        .route("/v1/phonemes", get(handlers::phoneme_inventory))
        .route("/v1/audio/speech", post(handlers::speech))
        .merge(docs::swagger_ui())
        .with_state(server)
}
