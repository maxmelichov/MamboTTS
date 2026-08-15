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
        mambotts_registry::QWEN_HE_RUNTIME_ID => qwen_load_params(body),
        _ => Err("unsupported runtime; install a server build that includes the selected runtime"),
    }
}

/// QwenTTS needs both GGUFs plus RenikudPlus, since the checkpoint reads
/// stressed IPA rather than Hebrew script.
fn qwen_load_params(body: LoadBody) -> Result<LoadParams, &'static str> {
    let talker_path = first_non_empty([
        body.talker_path,
        std::env::var("MAMBOTTS_QWEN_TALKER_PATH").unwrap_or_default(),
    ]);
    let codec_path = first_non_empty([
        body.codec_path,
        std::env::var("MAMBOTTS_QWEN_CODEC_PATH").unwrap_or_default(),
    ]);
    let renikud_path = first_non_empty([
        body.renikud_path,
        std::env::var("MAMBOTTS_RENIKUD_PATH").unwrap_or_default(),
    ]);
    if talker_path.is_empty() || codec_path.is_empty() {
        return Err("QwenTTS runtime requires talker_path and codec_path");
    }
    if renikud_path.is_empty() {
        return Err("QwenTTS runtime requires renikud_path for Hebrew phonemes");
    }
    Ok(LoadParams {
        runtime: mambotts_registry::QWEN_HE_RUNTIME_ID.into(),
        params: RuntimeParams::QwenHe {
            talker_path: talker_path.into(),
            codec_path: codec_path.into(),
            renikud_path: renikud_path.into(),
        },
    })
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
    let hebrew_g2p_engine = if body.hebrew_g2p_engine.is_empty() { "renikud".into() } else { body.hebrew_g2p_engine };
    if !matches!(hebrew_g2p_engine.as_str(), "renikud" | "phonikud") {
        return Err("hebrew_g2p_engine must be renikud or phonikud");
    }
    if model_path.is_empty() || renikud_path.is_empty() {
        return Err("Blue runtime requires model_path and renikud_path");
    }
    Ok(LoadParams {
        runtime: mambotts_registry::DEFAULT_RUNTIME_ID.into(),
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
    use super::load_params;
    use crate::runtime::RuntimeParams;
    use crate::server::dto::LoadBody;

    fn blue(model_path: &str, renikud_path: &str) -> LoadBody {
        LoadBody {
            runtime: "blue".into(),
            model_path: model_path.into(),
            renikud_path: renikud_path.into(),
            hebrew_g2p_engine: "renikud".into(),
            ..LoadBody::default()
        }
    }

    fn qwen(talker_path: &str, codec_path: &str, renikud_path: &str) -> LoadBody {
        LoadBody {
            runtime: "qwen_he".into(),
            talker_path: talker_path.into(),
            codec_path: codec_path.into(),
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
    fn qwen_load_requires_both_ggufs_and_renikud() {
        let params =
            load_params(qwen("/m/talker.gguf", "/m/codec.gguf", "/m/renikud-plus.onnx")).unwrap();
        assert_eq!(params.runtime, "qwen_he");
        assert!(matches!(params.params, RuntimeParams::QwenHe { .. }));

        assert!(load_params(qwen("", "/m/codec.gguf", "/m/renikud-plus.onnx")).is_err());
        assert!(load_params(qwen("/m/talker.gguf", "", "/m/renikud-plus.onnx")).is_err());
        // The model reads IPA, so a missing G2P is a load-time failure
        // rather than a surprise at the first Hebrew request.
        assert!(load_params(qwen("/m/talker.gguf", "/m/codec.gguf", "")).is_err());
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
