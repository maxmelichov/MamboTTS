use std::path::PathBuf;

use crate::analytics;

use super::errors::track_err;

pub async fn copy_audio_file_request(
    app: tauri::AppHandle,
    source_path: String,
    destination_path: String,
) -> Result<(), String> {
    if source_path.trim().is_empty() {
        let err = "source audio path is empty".to_string();
        return Err(track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            "copy_audio_file",
        ));
    }
    if destination_path.trim().is_empty() {
        let err = "destination audio path is empty".to_string();
        return Err(track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            "copy_audio_file",
        ));
    }

    let source = PathBuf::from(source_path);
    if !source.is_file() {
        let err = format!("source audio file does not exist: {}", source.display());
        return Err(track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            err,
            "copy_audio_file",
        ));
    }

    let destination = PathBuf::from(destination_path);
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|err| {
            let message = format!(
                "failed to create destination folder {}: {err}",
                parent.display()
            );
            track_err(
                &app,
                analytics::events::ERROR_FILE_OPERATION_FAILED,
                message,
                "copy_audio_file",
            )
        })?;
    }

    tokio::fs::copy(&source, &destination)
        .await
        .map_err(|err| {
            let message = format!(
                "failed to copy audio from {} to {}: {err}",
                source.display(),
                destination.display()
            );
            track_err(
                &app,
                analytics::events::ERROR_FILE_OPERATION_FAILED,
                message,
                "copy_audio_file",
            )
        })?;
    Ok(())
}
