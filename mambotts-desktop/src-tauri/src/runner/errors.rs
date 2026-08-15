use crate::analytics;
use serde::de::DeserializeOwned;

use super::dto::ErrorResponse;

pub fn track_err(app: &tauri::AppHandle, event: &str, error: String, operation: &str) -> String {
    analytics::track_error(
        app,
        event,
        error,
        serde_json::json!({"operation": operation}),
    )
}

pub fn track_runner_err(
    app: &tauri::AppHandle,
    event: &str,
    error: String,
    operation: &str,
    runtime: &str,
) -> String {
    analytics::track_error(
        app,
        event,
        error,
        serde_json::json!({"operation": operation, "runtime": runtime}),
    )
}

pub async fn json_response(response: reqwest::Response) -> Result<serde_json::Value, String> {
    if !response.status().is_success() {
        return Err(response_error(response).await);
    }
    response
        .json::<serde_json::Value>()
        .await
        .map_err(|err| format!("failed to parse JSON response: {err}"))
}

pub async fn response_error(response: reqwest::Response) -> String {
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if let Ok(parsed) = serde_json::from_str::<ErrorResponse>(&text) {
        format!(
            "runner request failed ({status}, {}): {}",
            parsed.error.code, parsed.error.message
        )
    } else if text.trim().is_empty() {
        format!("runner request failed ({status})")
    } else {
        format!("runner request failed ({status}): {text}")
    }
}

pub async fn get_json<T: DeserializeOwned>(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
    url: &str,
    operation: &str,
    label: &str,
) -> Result<T, String> {
    let response = client.get(url).send().await.map_err(|err| {
        track_err(
            app,
            analytics::events::ERROR_RUNNER_REQUEST_FAILED,
            format!("failed to send {label} request: {err}"),
            operation,
        )
    })?;
    if !response.status().is_success() {
        let err = response_error(response).await;
        return Err(track_err(
            app,
            analytics::events::ERROR_RUNNER_REQUEST_FAILED,
            err,
            operation,
        ));
    }
    response.json::<T>().await.map_err(|err| {
        track_err(
            app,
            analytics::events::ERROR_RUNNER_REQUEST_FAILED,
            format!("failed to parse {label} response: {err}"),
            operation,
        )
    })
}
