use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::stream;

use super::super::{
    dto::{PhonemeInventoryResponse, PhonemizeBody, PhonemizeResponse, SpeechBody},
    errors::write_error,
    state::SharedServer,
    util::{first_non_empty, first_non_zero_float},
};

pub async fn phonemize(
    State(server): State<SharedServer>,
    Json(body): Json<PhonemizeBody>,
) -> Response {
    if body.input.trim().is_empty() {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        );
    }
    let mut inner = server.inner.lock().await;
    let Some(ctx) = inner.ctx.as_mut() else {
        return write_error(StatusCode::SERVICE_UNAVAILABLE, "no_model", "no model loaded");
    };
    match ctx.phonemize(&body.input, &body.language) {
        Ok(phonemes) => Json(PhonemizeResponse { phonemes }).into_response(),
        Err(err) => write_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", err.to_string()),
    }
}

pub async fn phoneme_inventory(State(server): State<SharedServer>) -> Response {
    let inner = server.inner.lock().await;
    let Some(ctx) = inner.ctx.as_ref() else {
        return write_error(StatusCode::SERVICE_UNAVAILABLE, "no_model", "no model loaded");
    };
    Json(PhonemeInventoryResponse {
        phonemes: ctx
            .supported_phonemes()
            .into_iter()
            .filter(|character| character.is_alphabetic() || !character.is_ascii())
            .map(|character| character.to_string())
            .collect(),
    })
    .into_response()
}

pub async fn diacritize(
    State(server): State<SharedServer>,
    Json(body): Json<PhonemizeBody>,
) -> Response {
    if body.input.trim().is_empty() {
        return write_error(StatusCode::BAD_REQUEST, "invalid_request", "request body must contain input");
    }
    let mut inner = server.inner.lock().await;
    let Some(ctx) = inner.ctx.as_mut() else {
        return write_error(StatusCode::SERVICE_UNAVAILABLE, "no_model", "no model loaded");
    };
    match ctx.diacritize(&body.input) {
        Ok(text) => Json(PhonemizeResponse { phonemes: text }).into_response(),
        Err(err) => write_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", err.to_string()),
    }
}

/// Whether the loaded runtime can clone a voice from a reference
/// recording, read from the registry so the answer tracks the manifest
/// rather than a hardcoded runtime name.
async fn supports_voice_reference(server: &SharedServer) -> bool {
    let runtime = server.inner.lock().await.runtime.clone();
    mambotts_registry::runtime(&runtime)
        .is_some_and(|manifest| manifest.capabilities.voice_reference)
}

#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechBody,
    responses((status = 200, content_type = "audio/wav"), (status = 400), (status = 503), (status = 500))
)]
pub async fn speech(State(server): State<SharedServer>, Json(body): Json<SpeechBody>) -> Response {
    if body.input.is_empty() {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        );
    }
    if !body.response_format.is_empty() && body.response_format != "wav" {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "only wav response_format is supported",
        );
    }
    if !body.voice_reference.is_empty() && !supports_voice_reference(&server).await {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "the loaded runtime supports only its saved voices; \
             load a runtime with voice cloning to use voice_reference",
        );
    }
    if body.stream {
        return streaming_wav_response(server, body).await;
    }
    if body.input_is_phonemes {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "phoneme input requires stream=true",
        );
    }

    let Ok(tmp) = tempfile::Builder::new()
        .prefix("mambotts-speech-")
        .suffix(".wav")
        .tempfile()
    else {
        return write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "failed to create temp output",
        );
    };

    let out_path = tmp.path().to_path_buf();
    let voice = first_non_empty([body.voice_reference.clone(), body.voice.clone()]);
    let speed = resolve_speed(body.speed);
    {
        let mut inner = server.inner.lock().await;
        let Some(ctx) = inner.ctx.as_mut() else {
            return write_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_model",
                "no model loaded",
            );
        };
        if let Err(err) = ctx.synthesize_to_file(
            &body.input,
            (!voice.is_empty()).then_some(voice.as_str()),
            &out_path,
            &body.language,
            speed,
        ) {
            return write_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                err.to_string(),
            );
        }
    }

    let Ok(data) = std::fs::read(&out_path) else {
        return write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            "failed to read output WAV",
        );
    };
    wav_response(data)
}

/// Stream self-contained WAV chunks using a small binary frame protocol:
/// `[kind: u8][payload length: u32 big endian][payload]`, where kind `1` is a
/// playable chunk, `2` is the complete normalized WAV, and `3` is UTF-8 error
/// text. The desktop client consumes this protocol and emits each chunk to the
/// webview immediately.
async fn streaming_wav_response(server: SharedServer, body: SpeechBody) -> Response {
    {
        let inner = server.inner.lock().await;
        if inner.ctx.is_none() {
            return write_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_model",
                "no model loaded",
            );
        }
    }

    let voice = first_non_empty([body.voice_reference.clone(), body.voice.clone()]);
    let speed = resolve_speed(body.speed);
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(2);
    tokio::task::spawn_blocking(move || {
        let mut inner = server.inner.blocking_lock();
        let Some(ctx) = inner.ctx.as_mut() else {
            let _ = tx.blocking_send(Ok(frame(3, b"no model loaded".to_vec())));
            return;
        };
        let sample_rate = ctx.sample_rate();
        let mut send_chunk = |samples: &[f32], sample_rate: u32| -> anyhow::Result<()> {
            // Text chunking can end with a separator-only segment. Do not send
            // an empty WAV frame to clients, because it can interrupt queued
            // playback without contributing any audio.
            if samples.is_empty() {
                return Ok(());
            }
            let wav = wav_bytes(samples, sample_rate)?;
            tx.blocking_send(Ok(frame(1, wav)))
                .map_err(|_| anyhow::anyhow!("streaming client disconnected"))
        };
        let result = if body.input_is_phonemes {
            ctx.synthesize_phonemes_streaming(
                &body.input,
                (!voice.is_empty()).then_some(voice.as_str()),
                &body.language,
                speed,
                &mut send_chunk,
            )
        } else {
            ctx.synthesize_streaming(
                &body.input,
                (!voice.is_empty()).then_some(voice.as_str()),
                &body.language,
                speed,
                &mut send_chunk,
            )
        };
        match result {
            Ok(audio) => {
                // The final frame is retained for download/save. It does not
                // delay playback because every chunk was already sent above.
                match wav_bytes(&audio, sample_rate) {
                    Ok(wav) => {
                        send_final_wav(&tx, wav);
                    }
                    Err(err) => {
                        let _ = tx.blocking_send(Ok(frame(3, err.to_string().into_bytes())));
                    }
                }
            }
            Err(err) => {
                let _ = tx.blocking_send(Ok(frame(3, err.to_string().into_bytes())));
            }
        }
    });

    let body = stream::unfold(rx, |mut receiver| async move {
        receiver.recv().await.map(|item| (item, receiver))
    });
    let mut response = Body::from_stream(body).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-mambotts-audio-chunks"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

/// How much of the finished recording travels in one frame.
///
/// The final frame carries the whole normalised recording, so on a long text
/// it runs to hundreds of megabytes. Sent whole it forces both ends to hold the
/// entire recording in memory at once and overruns any sane frame ceiling the
/// client sets, which loses the result after all the synthesis work is already
/// paid for. Slicing it lets a recording of any length through on a bounded
/// buffer at each end.
const FINAL_SLICE_BYTES: usize = 8 * 1024 * 1024;

/// Send the finished recording as bounded slices.
///
/// Every slice but the last is a continuation for the client to append. The
/// last one is what marks the audio complete, so a stream that stops early is
/// never mistaken for a finished file.
fn send_final_wav(tx: &tokio::sync::mpsc::Sender<Result<Bytes, std::io::Error>>, wav: Vec<u8>) {
    let mut offset = 0usize;
    loop {
        let end = (offset + FINAL_SLICE_BYTES).min(wav.len());
        let last = end >= wav.len();
        let kind = if last { 2 } else { 4 };
        if tx
            .blocking_send(Ok(frame(kind, wav[offset..end].to_vec())))
            .is_err()
        {
            return;
        }
        offset = end;
        if last {
            return;
        }
    }
}

/// The pace MamboTTS has always shipped.
///
/// The request carries a multiplier rather than an engine value, so a speed of
/// 1.0 reproduces exactly what every previous version produced and the control
/// changes nothing until somebody moves it.
const BASELINE_SPEED: f32 = 0.95;

/// The outer bounds the API will accept. 0.75 to 2.0 measured clean; below
/// 0.75 the audio loses level and articulation rather than simply slowing,
/// which is why the app's own slider stops there. The wider clamp is kept for
/// API callers who want to experiment, and only guards against nonsense.
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 2.0;

fn resolve_speed(requested: f32) -> f32 {
    // Absent, zero and negative all mean "unspecified" here, because serde
    // defaults a missing float to zero and no caller means to ask for silence.
    let multiplier = if requested > 0.0 { requested } else { 1.0 };
    BASELINE_SPEED * multiplier.clamp(MIN_SPEED, MAX_SPEED)
}

fn frame(kind: u8, payload: Vec<u8>) -> Bytes {
    let mut out = Vec::with_capacity(5 + payload.len());
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(&payload);
    Bytes::from(out)
}

fn wav_bytes(samples: &[f32], sample_rate: u32) -> anyhow::Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}

fn wav_response(data: Vec<u8>) -> Response {
    let mut response = Bytes::from(data).into_response();
    response
        .headers_mut()
        .insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/wav"));
    response.headers_mut().insert(
        header::CONTENT_DISPOSITION,
        HeaderValue::from_static("attachment; filename=\"speech.wav\""),
    );
    response
}
