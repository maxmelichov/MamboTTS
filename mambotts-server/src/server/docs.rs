use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

use super::{
    dto::{
        CancelBody, CancelResponse, DiacritizeBody, DiacritizeResponse, HealthResponse,
        LanguagesResponse, LoadBody, LoadResponse, ModelsResponse, PhonemeInventoryResponse,
        PhonemizeBody, PhonemizeResponse, SpeechBody, StatusResponse, VoicesResponse,
    },
    errors::{ErrorBody, ErrorDetail},
    handlers,
    sources::{ModelSource, ModelSourceFile, ModelSourcesResponse},
};
use crate::runtime::Language;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "MamboTTS API",
        description = "Local BlueV3 text-to-speech HTTP API served by mambotts-server. \
                       Speaks Hebrew, English, Spanish, German, Italian and Russian \
                       (he, en, es, de, it, ru) in fixed voices; no voice cloning."
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
        handlers::speech::cancel_speech,
        handlers::speech::phonemize,
        handlers::speech::diacritize,
        handlers::speech::phoneme_inventory,
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
        CancelBody,
        CancelResponse,
        PhonemizeBody,
        PhonemizeResponse,
        DiacritizeBody,
        DiacritizeResponse,
        PhonemeInventoryResponse,
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
