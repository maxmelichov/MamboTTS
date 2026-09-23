mod binary;
mod cancel;
mod client;
mod dto;
mod errors;
mod file_ops;
mod process;
mod speech_stream;

use std::{
    env,
    time::{SystemTime, UNIX_EPOCH},
};

use tauri::{Manager, State};

use crate::analytics;

pub use dto::{LoadModelRequest, PhonemizeRequest, RunnerInfo, SpeechRequest, SpeechResult};
pub use process::RunnerState;

use binary::resolve_runner_binary;
use client::{
    cancel_synthesis_request, diacritize_request, get_languages_request,
    get_phoneme_inventory_request, get_voices_request, load_model_request, phonemize_request,
    synthesize_request,
};
use errors::track_err;
use file_ops::export_audio_file_request;
use process::RunnerProcess;

#[tauri::command]
pub async fn start_runner(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<RunnerInfo, String> {
    ensure_runner(&app, &state)
        .map(|base_url| RunnerInfo { base_url })
        .map_err(|err| {
            track_err(
                &app,
                analytics::events::ERROR_RUNNER_START_FAILED,
                err,
                "start_runner",
            )
        })
}

#[tauri::command]
pub async fn stop_runner(state: State<'_, RunnerState>) -> Result<(), String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(mut process) = guard.take() {
        process.kill();
    }
    Ok(())
}

pub fn stop_managed_runner(app: &tauri::AppHandle) {
    let state = app.state::<RunnerState>();
    match state.process.lock() {
        Ok(mut guard) => {
            if let Some(mut process) = guard.take() {
                process.kill();
            }
        }
        Err(_) => eprintln!("runner state lock poisoned during app shutdown"),
    };
}

#[tauri::command]
pub async fn get_runner_url(state: State<'_, RunnerState>) -> Result<Option<String>, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(process) = guard.as_mut() {
        if process.is_alive() {
            return Ok(Some(process.base_url()));
        }
    }
    *guard = None;
    Ok(None)
}

#[tauri::command]
pub async fn load_model(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: LoadModelRequest,
) -> Result<serde_json::Value, String> {
    load_model_request(app, state, request).await
}

#[tauri::command]
pub async fn get_languages(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    get_languages_request(app, state).await
}

#[tauri::command]
pub async fn get_voices(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    get_voices_request(app, state).await
}

#[tauri::command]
pub async fn phonemize(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: PhonemizeRequest,
) -> Result<String, String> {
    phonemize_request(app, state, request).await
}

#[tauri::command]
pub async fn diacritize(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: PhonemizeRequest,
) -> Result<String, String> {
    diacritize_request(app, state, request).await
}

#[tauri::command]
pub async fn get_phoneme_inventory(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    get_phoneme_inventory_request(app, state).await
}

/// Synthesize speech. Playable chunks arrive on `on_chunk` as raw WAV bytes
/// while inference runs; the result names the finished recording.
#[tauri::command]
pub async fn synthesize(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: SpeechRequest,
    on_chunk: tauri::ipc::Channel<tauri::ipc::InvokeResponseBody>,
) -> Result<SpeechResult, String> {
    synthesize_request(app, state, request, on_chunk).await
}

/// Stop the synthesis started with this id. It rejects with
/// "synthesis cancelled" and leaves no files behind.
#[tauri::command]
pub async fn cancel_synthesis(
    state: State<'_, RunnerState>,
    synthesis_id: String,
) -> Result<(), String> {
    cancel_synthesis_request(state, synthesis_id);
    Ok(())
}

/// Remove chunk files that earlier versions left in the temp folder.
pub fn sweep_legacy_chunk_files() {
    std::thread::spawn(|| {
        let removed = speech_stream::sweep_legacy_chunk_files(&env::temp_dir());
        if removed > 0 {
            tracing::info!("removed {removed} leftover streamed chunk files");
        }
    });
}

/// Save the finished take at the chosen size (see [`mambotts_audio::ExportQuality`]).
/// Without a quality it is saved at the default, a 128 kbps MP3.
#[tauri::command]
pub async fn export_audio_file(
    app: tauri::AppHandle,
    source_path: String,
    destination_path: String,
    quality: Option<mambotts_audio::ExportQuality>,
) -> Result<(), String> {
    export_audio_file_request(app, source_path, destination_path, quality).await
}

fn ensure_runner(app: &tauri::AppHandle, state: &State<'_, RunnerState>) -> Result<String, String> {
    let mut guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    if let Some(process) = guard.as_mut() {
        if process.is_alive() {
            return Ok(process.base_url());
        }
        let stderr = process.recent_stderr();
        if !stderr.is_empty() {
            analytics::track_error(
                app,
                analytics::events::ERROR_RUNNER_REQUEST_FAILED,
                format!("MamboTTS server exited; recent stderr: {}", stderr.trim()),
                serde_json::json!({"operation": "ensure_runner"}),
            );
        }
        *guard = None;
    }

    let binary_path = resolve_runner_binary(app)?;
    let process = RunnerProcess::spawn(app, &binary_path)?;
    let base_url = process.base_url();
    *guard = Some(process);
    Ok(base_url)
}

pub(crate) fn runner_client(
    app: &tauri::AppHandle,
    state: &State<'_, RunnerState>,
) -> Result<(reqwest::Client, String), String> {
    let base_url = ensure_runner(app, state)?;
    let guard = state
        .process
        .lock()
        .map_err(|_| "runner state lock poisoned".to_string())?;
    let process = guard
        .as_ref()
        .ok_or_else(|| "runner process missing after start".to_string())?;
    Ok((process.client(), base_url))
}

pub(crate) fn default_output_path() -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default();
    env::temp_dir()
        .join(format!("mambotts-speech-{millis}.wav"))
        .as_os_str()
        .to_string_lossy()
        .into_owned()
}
