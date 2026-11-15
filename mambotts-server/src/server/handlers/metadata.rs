use axum::{
    Json,
    extract::Path,
    http::{HeaderMap, StatusCode, header},
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

/// A model file the server carries ([`mambotts_registry::ModelFile::shipped`]),
/// by its registry name (`voices/noa.json`): the exact bytes the registry's
/// size and SHA-256 describe.
#[utoipa::path(
    get,
    path = "/v1/models/files/{name}",
    params(("name" = String, Path, description = "the registry name, e.g. voices/noa.json")),
    responses((status = 200, content_type = "application/octet-stream"), (status = 404))
)]
pub async fn model_file_handler(Path(name): Path<String>) -> impl IntoResponse {
    let shipped = mambotts_registry::runtimes()
        .iter()
        .flat_map(|runtime| runtime.files)
        .find(|file| file.name == name)
        .and_then(|file| file.shipped);
    match shipped {
        Some(bytes) => (StatusCode::OK, [(header::CONTENT_TYPE, "application/octet-stream")], bytes).into_response(),
        None => (StatusCode::NOT_FOUND, format!("{name} is not a file this server carries")).into_response(),
    }
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

#[cfg(test)]
mod tests {
    use axum::{body::to_bytes, extract::Path, response::IntoResponse};

    #[tokio::test]
    async fn shipped_files_are_served_by_their_registry_name() {
        let response = super::model_file_handler(Path("voices/shira.json".to_owned())).await.into_response();
        assert_eq!(response.status(), 200);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let blue = mambotts_registry::runtime("blue").unwrap();
        let file = blue.files.iter().find(|file| file.name == "voices/shira.json").unwrap();
        assert_eq!(body.len() as u64, file.size);
        let response = super::model_file_handler(Path("vocoder.onnx".to_owned())).await.into_response();
        assert_eq!(response.status(), 404, "an LFS file is downloaded, not served");
    }
}
