use axum::{
    Json,
    http::{HeaderMap, header},
    response::IntoResponse,
};

use super::super::{model_sources, skill};

#[utoipa::path(
    get,
    path = "/v1/models/sources",
    responses((status = 200, body = super::super::sources::ModelSourcesResponse))
)]
pub async fn model_sources_handler() -> impl IntoResponse {
    Json(model_sources())
}

#[utoipa::path(
    get,
    path = "/skill",
    responses((status = 200, content_type = "text/markdown"))
)]
pub async fn skill_handler(headers: HeaderMap) -> impl IntoResponse {
    let host = headers
        .get(header::HOST)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("127.0.0.1");
    (
        [("content-type", "text/markdown; charset=utf-8")],
        skill::render_skill(host),
    )
}
