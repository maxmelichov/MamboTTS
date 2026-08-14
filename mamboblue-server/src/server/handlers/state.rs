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
    let inner = server.inner.lock().await;
    let loaded = inner.ctx.is_some();
    Json(HealthResponse {
        status: if loaded { "ready" } else { "ok" }.into(),
        loaded,
        model: inner.model_name.clone(),
        runtime: inner.runtime.clone(),
    })
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, body = ModelsResponse)))]
pub async fn models(State(server): State<SharedServer>) -> impl IntoResponse {
    let inner = server.inner.lock().await;
    Json(ModelsResponse {
        loaded: inner.ctx.is_some(),
        runtime: inner.runtime.clone(),
        model: inner.model_name.clone(),
        path: inner.model_path.clone(),
        codec: inner.codec_path.clone(),
    })
}

#[utoipa::path(
    get,
    path = "/v1/languages",
    responses((status = 200, body = LanguagesResponse), (status = 503))
)]
pub async fn languages(State(server): State<SharedServer>) -> Response {
    let inner = server.inner.lock().await;
    let Some(ctx) = inner.ctx.as_ref() else {
        return write_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_model",
            "no model loaded",
        );
    };
    let items = ctx.languages().to_vec();
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
    let inner = server.inner.lock().await;
    let Some(ctx) = inner.ctx.as_ref() else {
        return voices_unavailable();
    };
    let Some(voices) = ctx.voices() else {
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
