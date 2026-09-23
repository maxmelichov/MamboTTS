use std::{path::Path, sync::OnceLock};

use tauri::{
    State,
    ipc::{Channel, InvokeResponseBody},
};

use crate::{analytics, runner::errors::track_runner_err};

use super::{
    cancel::{SYNTHESIS_CANCELLED, SynthesisRegistry},
    dto::{
        DiacritizeResponse, LanguagesResponse, LoadModelRequest, PhonemeInventoryResponse,
        PhonemizeRequest, PhonemizeResponse, SpeechRequest, SpeechResult, VoicesResponse,
    },
    errors::{get_json, json_response, response_error},
    process::RunnerState,
    runner_client,
    speech_stream::receive_speech_stream,
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
    serde_json::from_value::<DiacritizeResponse>(body)
        .map_err(|err| format!("invalid diacritize response: {err}"))?
        .into_text()
        .ok_or_else(|| "invalid diacritize response: no text".to_string())
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
    on_chunk: Channel<InvokeResponseBody>,
) -> Result<SpeechResult, String> {
    let synthesis_id = request.synthesis_id.clone().unwrap_or_default();
    let cancelled = state.synthesis.register(&synthesis_id);
    let _registration = Registration {
        registry: &state.synthesis,
        id: &synthesis_id,
    };
    if *cancelled.borrow() {
        return Err(SYNTHESIS_CANCELLED.to_string());
    }

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
        // A multiplier, not an engine value: 1.0 is the pace every previous
        // version produced. The server clamps it and applies the baseline.
        "speed": request.speed.unwrap_or(1.0),
    });
    let props = || {
        serde_json::json!({
            "operation": "synthesize",
            "voice": body["voice"].as_str().unwrap_or_default(),
            "language": body["language"].as_str().unwrap_or("auto"),
            "has_voice_reference": has_voice_reference,
            "speed": body["speed"].as_f64().unwrap_or(1.0),
        })
    };

    let generation_id = OnceLock::<String>::new();
    let run = async {
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
        if let Some(id) = response
            .headers()
            .get(GENERATION_ID_HEADER)
            .and_then(|value| value.to_str().ok())
        {
            let _ = generation_id.set(id.to_owned());
        }
        stream_speech_response(&app, response, &output_path, &on_chunk, props()).await
    };

    let chunks = tokio::select! {
        result = run => result?,
        _ = wait_for_cancel(cancelled) => {
            // The explicit cancel is what stops the engine promptly. Dropping
            // `run` on the way out then closes the connection, which the server
            // also treats as a cancel, and deletes the partial output.
            cancel_server_generation(&client, &base_url, generation_id.get()).await;
            return Err(SYNTHESIS_CANCELLED.to_string());
        }
    };
    analytics::track_event_handle_with_props(
        &app,
        analytics::events::SYNTHESIS_COMPLETED,
        Some(props()),
    );
    Ok(SpeechResult {
        path: output_path,
        chunks,
    })
}

/// Resolve once this synthesis has been cancelled, and never otherwise.
async fn wait_for_cancel(mut cancelled: tokio::sync::watch::Receiver<bool>) {
    loop {
        if *cancelled.borrow_and_update() {
            return;
        }
        if cancelled.changed().await.is_err() {
            // The registration is gone, so nothing can cancel this any more.
            std::future::pending::<()>().await;
        }
    }
}

/// Stop a synthesis started with the same `synthesis_id`. A cancel that
/// arrives before the synthesis registers is kept, so a quick Stop still wins.
pub fn cancel_synthesis_request(state: State<'_, RunnerState>, synthesis_id: String) {
    state.synthesis.cancel(&synthesis_id);
}

const GENERATION_ID_HEADER: &str = "x-mambotts-generation-id";

/// Unregisters a synthesis however `synthesize_request` returns.
struct Registration<'a> {
    registry: &'a SynthesisRegistry,
    id: &'a str,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        self.registry.finish(self.id);
    }
}

/// Ask the server to stop a generation. Without an id (the response headers
/// had not arrived yet) this cancels everything the server is running, which
/// in the app is only ever this one generation.
async fn cancel_server_generation(client: &reqwest::Client, base_url: &str, id: Option<&String>) {
    let body = match id {
        Some(id) => serde_json::json!({ "id": id }),
        None => serde_json::json!({}),
    };
    let request = client
        .post(format!("{base_url}/v1/audio/speech/cancel"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(3))
        .send();
    if let Err(err) = request.await {
        tracing::warn!("failed to cancel the server generation: {err}");
    }
}

async fn stream_speech_response(
    app: &tauri::AppHandle,
    response: reqwest::Response,
    output_path: &str,
    on_chunk: &Channel<InvokeResponseBody>,
    props: serde_json::Value,
) -> Result<usize, String> {
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
        return Ok(0);
    }

    // Chunks go to the webview as raw bytes over the channel rather than as
    // files beside the output, so nothing is left behind in the temp folder.
    receive_speech_stream(response.bytes_stream(), Path::new(output_path), |chunk| {
        on_chunk
            .send(InvokeResponseBody::Raw(chunk))
            .map_err(|err| format!("failed to deliver streamed audio chunk: {err}"))
    })
    .await
    .map_err(|err| {
        analytics::track_error(app, analytics::events::ERROR_SYNTHESIS_FAILED, err, props)
    })
}
