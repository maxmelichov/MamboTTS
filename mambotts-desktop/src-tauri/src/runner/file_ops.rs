use std::path::PathBuf;

use mambotts_audio::ExportQuality;

use crate::analytics;

use super::errors::track_err;

const OPERATION: &str = "export_audio_file";

/// Save the finished take at `source_path` (always a WAV) to
/// `destination_path` at the chosen size. The largest size is a plain copy;
/// the others are re-encoded as MP3.
pub async fn export_audio_file_request(
    app: tauri::AppHandle,
    source_path: String,
    destination_path: String,
    quality: Option<ExportQuality>,
) -> Result<(), String> {
    let fail = |message: String| {
        track_err(
            &app,
            analytics::events::ERROR_FILE_OPERATION_FAILED,
            message,
            OPERATION,
        )
    };
    if source_path.trim().is_empty() {
        return Err(fail("source audio path is empty".to_string()));
    }
    if destination_path.trim().is_empty() {
        return Err(fail("destination audio path is empty".to_string()));
    }

    let source = PathBuf::from(source_path);
    if !source.is_file() {
        return Err(fail(format!(
            "source audio file does not exist: {}",
            source.display()
        )));
    }

    let destination = PathBuf::from(destination_path);
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|err| {
            fail(format!(
                "failed to create destination folder {}: {err}",
                parent.display()
            ))
        })?;
    }

    let format = quality.unwrap_or_default().format();
    // Encoding a long take is seconds of CPU work; keep it off the async runtime.
    let result = tauri::async_runtime::spawn_blocking({
        let source = source.clone();
        let destination = destination.clone();
        move || mambotts_audio::export_wav_file(&source, &destination, format)
    })
    .await
    .map_err(|err| fail(format!("audio export task failed: {err}")))?;
    result.map_err(|err| {
        fail(format!(
            "failed to save audio from {} to {}: {err:#}",
            source.display(),
            destination.display()
        ))
    })
}
