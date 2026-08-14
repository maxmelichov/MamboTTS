use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use super::{
    dto::{
        HealthResponse, LanguagesResponse, LoadBody, LoadResponse, ModelsResponse, SpeechBody,
        StatusResponse, VoicesResponse,
    },
    errors::{ErrorBody, ErrorDetail},
    handlers,
    sources::{ModelSource, ModelSourceFile, ModelSourcesResponse},
};
use crate::runtime::Language;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "MamboRambo API",
        description = "Local TTS HTTP API served by mamborambo-server."
    ),
    paths(
        handlers::state::health,
        handlers::metadata::skill_handler,
        handlers::state::models,
        handlers::metadata::model_sources_handler,
        handlers::load::model_load,
        handlers::state::model_unload,
        handlers::state::languages,
        handlers::state::voices,
        handlers::speech::speech,
    ),
    components(schemas(
        ErrorBody,
        ErrorDetail,
        HealthResponse,
        ModelsResponse,
        LanguagesResponse,
        VoicesResponse,
        StatusResponse,
        LoadResponse,
        LoadBody,
        SpeechBody,
        Language,
        ModelSourceFile,
        ModelSource,
        ModelSourcesResponse,
    ))
)]
struct ApiDoc;

pub fn swagger_ui() -> SwaggerUi {
    SwaggerUi::new("/docs").url("/openapi.json", ApiDoc::openapi())
}
