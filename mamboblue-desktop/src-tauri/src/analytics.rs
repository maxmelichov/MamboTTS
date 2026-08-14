use tauri::AppHandle;
use tauri_plugin_aptabase::EventTracker;
use tauri_plugin_store::StoreExt;

const STORE_FILENAME: &str = "settings.json";

pub const APTABASE_APP_KEY: &str = match option_env!("APTABASE_APP_KEY") {
    Some(v) => v,
    None => "",
};

pub const APTABASE_BASE_URL: &str = match option_env!("APTABASE_BASE_URL") {
    Some(v) => v,
    None => "",
};

pub mod events {
    pub const APP_STARTED: &str = "app_started";
    pub const ERROR_FILE_OPERATION_FAILED: &str = "error_file_operation_failed";
    pub const ERROR_MODEL_DOWNLOAD_FAILED: &str = "error_model_download_failed";
    pub const ERROR_MODEL_LOAD_FAILED: &str = "error_model_load_failed";
    pub const ERROR_RUNNER_REQUEST_FAILED: &str = "error_runner_request_failed";
    pub const ERROR_RUNNER_START_FAILED: &str = "error_runner_start_failed";
    pub const ERROR_SYNTHESIS_FAILED: &str = "error_synthesis_failed";
    pub const ERROR_VOICE_DOWNLOAD_FAILED: &str = "error_voice_download_failed";
    pub const SYNTHESIS_COMPLETED: &str = "synthesis_completed";
}

fn is_analytics_enabled(app_handle: &AppHandle) -> bool {
    let Ok(store) = app_handle.store(STORE_FILENAME) else {
        return true;
    };
    store
        .get("analytics_enabled")
        .and_then(|v: serde_json::Value| v.as_bool())
        .unwrap_or(true)
}

pub fn is_aptabase_configured() -> bool {
    !APTABASE_APP_KEY.is_empty() && !APTABASE_BASE_URL.is_empty()
}

pub fn track_event_handle(app_handle: &AppHandle, event_name: &str) {
    track_event_handle_with_props(app_handle, event_name, None);
}

pub fn track_event_handle_with_props(
    app_handle: &AppHandle,
    event_name: &str,
    props: Option<serde_json::Value>,
) {
    if !is_aptabase_configured() {
        tracing::debug!(
            "analytics track_event failed for '{}': APTABASE_APP_KEY or APTABASE_BASE_URL is not set",
            event_name
        );
        return;
    }
    if !is_analytics_enabled(app_handle) {
        return;
    }
    let mut merged = match props {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    merged
        .entry("mamborambo_commit")
        .or_insert_with(|| env!("COMMIT_HASH").into());
    tracing::trace!("analytics track_event '{}' sent", event_name);
    if let Err(error) = app_handle.track_event(event_name, Some(serde_json::Value::Object(merged)))
    {
        tracing::debug!(
            "analytics track_event failed for '{}': {}",
            event_name,
            error
        );
    }
}

pub fn track_error_handle(
    app_handle: &AppHandle,
    event_name: &str,
    error: impl std::fmt::Display,
    props: Option<serde_json::Value>,
) {
    let error_message = sanitize_error_message(&error.to_string());
    let mut merged = match props {
        Some(serde_json::Value::Object(m)) => m,
        _ => serde_json::Map::new(),
    };
    merged.insert(
        "error_message".to_string(),
        serde_json::Value::String(error_message),
    );
    track_event_handle_with_props(
        app_handle,
        event_name,
        Some(serde_json::Value::Object(merged)),
    );
}

pub fn track_error(
    app_handle: &AppHandle,
    event_name: &str,
    error: String,
    props: serde_json::Value,
) -> String {
    track_error_handle(app_handle, event_name, &error, Some(props));
    error
}

fn sanitize_error_message(message: &str) -> String {
    const MAX_LEN: usize = 500;
    message.chars().take(MAX_LEN).collect()
}
