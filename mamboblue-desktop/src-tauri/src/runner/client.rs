use futures_util::StreamExt;
use tauri::{Emitter, State};

use crate::{analytics, runner::errors::track_runner_err};

use super::{
    dto::{
        LanguagesResponse, LoadModelRequest, PhonemeInventoryResponse, PhonemizeRequest,
        PhonemizeResponse, SpeechRequest, VoicesResponse,
    },
    errors::{get_json, json_response, response_error},
    process::RunnerState,
    runner_client,
};

pub async fn load_model_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: LoadModelRequest,
) -> Result<serde_json::Value, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let runtime = request.runtime.clone();
    let body = serde_json::json!({
        "runtime": runtime.clone(),
        "model_path": request.model_path,
        "renikud_path": request.renikud_path,
        "hebrew_g2p_engine": request.hebrew_g2p_engine.unwrap_or_else(|| "renikud".into()),
        "phonikud_path": request.phonikud_path.unwrap_or_default(),
        "speaker": request.speaker.unwrap_or(0),
        "target_speaker": request.target_speaker.unwrap_or(0),
    });

    let response = client
        .post(format!("{base_url}/v1/models/load"))
        .json(&body)
        .send()
        .await
        .map_err(|err| {
            track_runner_err(
                &app,
                analytics::events::ERROR_MODEL_LOAD_FAILED,
                format!("failed to send model load request: {err}"),
                "load_model",
                &runtime,
            )
        })?;
    json_response(response).await.map_err(|err| {
        track_runner_err(
            &app,
            analytics::events::ERROR_MODEL_LOAD_FAILED,
            err,
            "load_model",
            &runtime,
        )
    })
}

pub async fn get_languages_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let body = get_json::<LanguagesResponse>(
        &app,
        &client,
        &format!("{base_url}/v1/languages"),
        "get_languages",
        "languages",
    )
    .await?;
    Ok(body.languages)
}

pub async fn get_voices_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let body = get_json::<VoicesResponse>(
        &app,
        &client,
        &format!("{base_url}/v1/voices"),
        "get_voices",
        "voices",
    )
    .await?;
    Ok(body.voices)
}

pub async fn phonemize_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: PhonemizeRequest,
) -> Result<String, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let response = client
        .post(format!("{base_url}/v1/phonemize"))
        .json(&serde_json::json!({
            "input": request.input,
            "language": request.language.unwrap_or_else(|| "auto".to_string()),
        }))
        .send()
        .await
        .map_err(|err| format!("failed to send phonemize request: {err}"))?;
    let body = json_response(response).await?;
    serde_json::from_value::<PhonemizeResponse>(body)
        .map(|response| response.phonemes)
        .map_err(|err| format!("invalid phonemize response: {err}"))
}

pub async fn diacritize_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: PhonemizeRequest,
) -> Result<String, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let response = client
        .post(format!("{base_url}/v1/diacritize"))
        .json(&serde_json::json!({ "input": request.input, "language": "he" }))
        .send()
        .await
        .map_err(|err| format!("failed to send diacritize request: {err}"))?;
    let body = json_response(response).await?;
    serde_json::from_value::<PhonemizeResponse>(body)
        .map(|response| response.phonemes)
        .map_err(|err| format!("invalid diacritize response: {err}"))
}

pub async fn get_phoneme_inventory_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
) -> Result<Vec<String>, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let body = get_json::<PhonemeInventoryResponse>(
        &app,
        &client,
        &format!("{base_url}/v1/phonemes"),
        "get_phoneme_inventory",
        "phonemes",
    )
    .await?;
    Ok(body.phonemes)
}

pub async fn synthesize_request(
    app: tauri::AppHandle,
    state: State<'_, RunnerState>,
    request: SpeechRequest,
) -> Result<String, String> {
    let (client, base_url) = runner_client(&app, &state)?;
    let output_path = request
        .output_path
        .unwrap_or_else(super::default_output_path);
    let language = request.language.unwrap_or_else(|| "auto".to_string());
    let voice = request.voice.unwrap_or_default();
    let has_voice_reference = request
        .voice_reference
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    let body = serde_json::json!({
        "input": request.input,
        "voice_reference": request.voice_reference.unwrap_or_default(),
        "voice": voice,
        "response_format": "wav",
        "language": language,
        "stream": true,
        "input_is_phonemes": request.input_is_phonemes.unwrap_or(false),
    });
    let props = || {
        serde_json::json!({
            "operation": "synthesize",
            "voice": body["voice"].as_str().unwrap_or_default(),
            "language": body["language"].as_str().unwrap_or("auto"),
            "has_voice_reference": has_voice_reference,
        })
    };

    let response = client
        .post(format!("{base_url}/v1/audio/speech"))
        .json(&body)
        .send()
        .await
        .map_err(|err| {
            analytics::track_error(
                &app,
                analytics::events::ERROR_SYNTHESIS_FAILED,
                format!("failed to send speech request: {err}"),
                props(),
            )
        })?;
    if !response.status().is_success() {
        let err = response_error(response).await;
        return Err(analytics::track_error(
            &app,
            analytics::events::ERROR_SYNTHESIS_FAILED,
            err,
            props(),
        ));
    }

    stream_speech_response(&app, response, &output_path, props()).await?;
    analytics::track_event_handle_with_props(
        &app,
        analytics::events::SYNTHESIS_COMPLETED,
        Some(props()),
    );
    Ok(output_path)
}

async fn stream_speech_response(
    app: &tauri::AppHandle,
    response: reqwest::Response,
    output_path: &str,
    props: serde_json::Value,
) -> Result<(), String> {
    let is_wav = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|content_type| content_type.starts_with("audio/wav"));
    if is_wav {
        let wav = response.bytes().await.map_err(|err| {
            analytics::track_error(
                app,
                analytics::events::ERROR_SYNTHESIS_FAILED,
                format!("failed to read synthesized audio: {err}"),
                props,
            )
        })?;
        tokio::fs::write(output_path, wav)
            .await
            .map_err(|err| format!("failed to write generated audio {output_path}: {err}"))?;
        return Ok(());
    }

    const MAX_FRAME_BYTES: usize = 128 * 1024 * 1024;
    let mut stream = response.bytes_stream();
    let mut pending = Vec::<u8>::new();
    let mut chunk_index = 0usize;
    let mut complete = false;

    while let Some(next) = stream.next().await {
        let bytes = next.map_err(|err| {
            analytics::track_error(
                app,
                analytics::events::ERROR_SYNTHESIS_FAILED,
                format!("failed to read speech stream: {err}"),
                props.clone(),
            )
        })?;
        pending.extend_from_slice(&bytes);

        while pending.len() >= 5 {
            let kind = pending[0];
            let length =
                u32::from_be_bytes([pending[1], pending[2], pending[3], pending[4]]) as usize;
            if length > MAX_FRAME_BYTES {
                return Err(analytics::track_error(
                    app,
                    analytics::events::ERROR_SYNTHESIS_FAILED,
                    "received an invalidly large speech frame".to_string(),
                    props,
                ));
            }
            if pending.len() < 5 + length {
                break;
            }
            let payload = pending[5..5 + length].to_vec();
            pending.drain(..5 + length);

            match kind {
                1 => {
                    let path = chunk_output_path(output_path, chunk_index);
                    chunk_index += 1;
                    tokio::fs::write(&path, payload).await.map_err(|err| {
                        format!("failed to write streamed audio chunk {path}: {err}")
                    })?;
                    app.emit("synthesis-chunk", &path)
                        .map_err(|err| format!("failed to emit streamed audio chunk: {err}"))?;
                }
                2 => {
                    tokio::fs::write(output_path, payload)
                        .await
                        .map_err(|err| {
                            format!("failed to write final audio {output_path}: {err}")
                        })?;
                    complete = true;
                }
                3 => {
                    return Err(String::from_utf8_lossy(&payload).into_owned());
                }
                _ => return Err("received an unknown speech stream frame".to_string()),
            }
        }
    }

    if !pending.is_empty() {
        return Err("speech stream ended with an incomplete frame".to_string());
    }
    if !complete {
        return Err("speech stream ended before the final WAV was received".to_string());
    }
    Ok(())
}

fn chunk_output_path(output_path: &str, index: usize) -> String {
    let path = std::path::Path::new(output_path);
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("speech");
    path.with_file_name(format!("{stem}-chunk-{index:04}.wav"))
        .as_os_str()
        .to_string_lossy()
        .into_owned()
}
