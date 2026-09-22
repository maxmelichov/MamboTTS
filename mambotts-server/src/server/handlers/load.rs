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
    let params = match load_params(body) {
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

/// Routes a load request to the parameter builder for its runtime id.
fn load_params(body: LoadBody) -> Result<LoadParams, &'static str> {
    match body.runtime.as_str() {
        mambotts_registry::DEFAULT_RUNTIME_ID => blue_load_params(body),
        _ => Err("unsupported runtime"),
    }
}

fn blue_load_params(body: LoadBody) -> Result<LoadParams, &'static str> {
    let model_path = first_non_empty([
        body.model_path,
        std::env::var("MAMBOTTS_BLUE_MODEL_DIR").unwrap_or_default(),
    ]);
    let renikud_path = first_non_empty([
        body.renikud_path,
        std::env::var("MAMBOTTS_RENIKUD_PATH").unwrap_or_default(),
    ]);
    if model_path.is_empty() || renikud_path.is_empty() {
        return Err("Blue runtime requires model_path and renikud_path");
    }
    Ok(LoadParams {
        runtime: mambotts_registry::DEFAULT_RUNTIME_ID.into(),
        params: RuntimeParams::Blue {
            model_dir: model_path.into(),
            renikud_path: renikud_path.into(),
            speaker: body.speaker,
            target_speaker: body.target_speaker,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::load_params;
    use crate::server::dto::LoadBody;

    fn blue(model_path: &str, renikud_path: &str) -> LoadBody {
        LoadBody {
            runtime: "blue".into(),
            model_path: model_path.into(),
            renikud_path: renikud_path.into(),
            ..LoadBody::default()
        }
    }

    #[test]
    fn blue_load_requires_model_and_renikud_paths() {
        assert!(load_params(LoadBody::default()).is_err());
        assert!(load_params(blue("/models/blue", "/models/renikud-plus.onnx")).is_ok());
        assert!(load_params(blue("", "/models/renikud-plus.onnx")).is_err());
    }

    #[test]
    fn retired_phonikud_fields_from_older_clients_are_ignored() {
        let body: LoadBody = serde_json::from_str(
            r#"{"runtime":"blue","model_path":"/models/blue","renikud_path":"/models/renikud-plus.onnx","hebrew_g2p_engine":"phonikud","phonikud_path":"/models/phonikud.onnx","speaker":1}"#,
        )
        .expect("old load body still parses");
        assert_eq!(body.speaker, 1);
        assert!(load_params(body).is_ok());
    }

    #[test]
    fn unknown_runtimes_are_rejected() {
        assert!(
            load_params(LoadBody {
                runtime: "kokoro".into(),
                ..LoadBody::default()
            })
            .is_err()
        );
    }
}
