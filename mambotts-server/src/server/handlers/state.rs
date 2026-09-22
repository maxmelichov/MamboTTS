use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use super::super::{
    dto::{HealthResponse, LanguagesResponse, ModelsResponse, StatusResponse, VoicesResponse},
    errors::write_error,
    state::SharedServer,
};

#[utoipa::path(get, path = "/health", responses((status = 200, body = HealthResponse)))]
pub async fn health(State(server): State<SharedServer>) -> impl IntoResponse {
    let busy = server.active_generations() > 0;
    let info = server.info();
    Json(HealthResponse {
        status: if info.loaded { "ready" } else { "ok" }.into(),
        loaded: info.loaded,
        model: info.model_name.clone(),
        runtime: info.runtime.clone(),
        busy,
    })
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, body = ModelsResponse)))]
pub async fn models(State(server): State<SharedServer>) -> impl IntoResponse {
    let info = server.info();
    Json(ModelsResponse {
        loaded: info.loaded,
        runtime: info.runtime.clone(),
        model: info.model_name.clone(),
        path: info.model_path.clone(),
        codec: info.codec_path.clone(),
    })
}

#[utoipa::path(
    get,
    path = "/v1/languages",
    responses((status = 200, body = LanguagesResponse), (status = 503))
)]
pub async fn languages(State(server): State<SharedServer>) -> Response {
    let info = server.info();
    if !info.loaded {
        return write_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_model",
            "no model loaded",
        );
    }
    let items = info.languages.clone();
    let languages = std::iter::once("auto".to_string())
        .chain(items.iter().map(|language| language.name.clone()))
        .collect::<Vec<_>>();
    Json(LanguagesResponse { languages, items }).into_response()
}

#[utoipa::path(
    get,
    path = "/v1/voices",
    responses((status = 200, body = VoicesResponse), (status = 503))
)]
pub async fn voices(State(server): State<SharedServer>) -> Response {
    let info = server.info();
    let Some(voices) = info.voices.clone().filter(|_| info.loaded) else {
        return voices_unavailable();
    };
    Json(VoicesResponse {
        runtime: "blue".into(),
        voices,
    })
    .into_response()
}

#[utoipa::path(
    delete,
    path = "/v1/models",
    responses((status = 200, body = StatusResponse))
)]
pub async fn model_unload(State(server): State<SharedServer>) -> impl IntoResponse {
    server.unload_model().await;
    Json(StatusResponse {
        status: "unloaded".into(),
    })
}

fn voices_unavailable() -> Response {
    write_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "no_model",
        "no model loaded or voices unavailable",
    )
}
