use std::{panic, path::PathBuf};

#[tauri::command]
pub async fn reveal_path(app: tauri::AppHandle, path: PathBuf) -> Result<(), String> {
    reveal_path_inner(path).await.map_err(|err| {
        crate::analytics::track_error(
            &app,
            crate::analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            serde_json::json!({"operation": "reveal_path"}),
        )
    })
}

async fn reveal_path_inner(path: PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(format!("path does not exist: {}", path.display()));
    }

    tauri::async_runtime::spawn_blocking(move || {
        panic::catch_unwind(|| {
            showfile::show_path_in_file_manager(path);
        })
        .map_err(|_| "failed to reveal path in file manager".to_string())
    })
    .await
    .map_err(|err| format!("failed to join reveal task: {err}"))?
}
