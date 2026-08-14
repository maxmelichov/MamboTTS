use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::runtime::RuntimeParams;

use super::super::{
    dto::{LoadBody, LoadResponse},
    errors::write_error,
    state::{LoadParams, SharedServer},
    util::first_non_empty,
};

#[utoipa::path(
    post,
    path = "/v1/models/load",
    request_body = LoadBody,
    responses((status = 200, body = LoadResponse), (status = 400), (status = 500))
)]
pub async fn model_load(
    State(server): State<SharedServer>,
    body: Option<Json<LoadBody>>,
) -> Response {
    let body = body.map(|Json(body)| body).unwrap_or_default();
    let params = match blue_load_params(body) {
        Ok(params) => params,
        Err(message) => return write_error(StatusCode::BAD_REQUEST, "invalid_request", message),
    };

    if let Err(err) = server.load_model(params).await {
        return write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            format!("failed to load model: {err}"),
        );
    }
    let inner = server.inner.lock().await;
    Json(LoadResponse {
        status: "loaded".into(),
        runtime: inner.runtime.clone(),
        model: inner.model_name.clone(),
    })
    .into_response()
}

fn blue_load_params(body: LoadBody) -> Result<LoadParams, &'static str> {
    if body.runtime != mamborambo_registry::DEFAULT_RUNTIME_ID {
        return Err(
            "unsupported runtime; install a server build that includes the selected runtime",
        );
    }
    let model_path = first_non_empty([
        body.model_path,
        std::env::var("MAMBORAMBO_BLUE_MODEL_DIR").unwrap_or_default(),
    ]);
    let renikud_path = first_non_empty([
        body.renikud_path,
        std::env::var("MAMBORAMBO_RENIKUD_PATH").unwrap_or_default(),
    ]);
    let hebrew_g2p_engine = if body.hebrew_g2p_engine.is_empty() { "renikud".into() } else { body.hebrew_g2p_engine };
    if !matches!(hebrew_g2p_engine.as_str(), "renikud" | "phonikud") {
        return Err("hebrew_g2p_engine must be renikud or phonikud");
    }
    if model_path.is_empty() || renikud_path.is_empty() {
        return Err("Blue runtime requires model_path and renikud_path");
    }
    Ok(LoadParams {
        runtime: mamborambo_registry::DEFAULT_RUNTIME_ID.into(),
        params: RuntimeParams::Blue {
            model_dir: model_path.into(),
            renikud_path: renikud_path.into(),
            hebrew_g2p_engine,
            phonikud_path: (!body.phonikud_path.is_empty()).then(|| body.phonikud_path.into()),
            speaker: body.speaker,
            target_speaker: body.target_speaker,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::blue_load_params;
    use crate::server::dto::LoadBody;

    #[test]
    fn blue_load_requires_model_and_renikud_paths() {
        assert!(blue_load_params(LoadBody::default()).is_err());
        assert!(
            blue_load_params(LoadBody {
                runtime: "blue".into(),
                model_path: "/models/blue".into(),
                renikud_path: "/models/renikud-plus.onnx".into(),
                hebrew_g2p_engine: "renikud".into(),
                phonikud_path: String::new(),
                speaker: 0,
                target_speaker: 0,
            })
            .is_ok()
        );
    }
}
