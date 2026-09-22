use axum::{
    Json,
    body::{Body, Bytes},
    extract::State,
    http::{HeaderName, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use futures_util::stream;

use super::super::{
    dto::{
        CancelBody, CancelResponse, DiacritizeBody, PhonemeInventoryResponse, PhonemizeBody,
        PhonemizeResponse, SpeechBody,
    },
    errors::write_error,
    state::{Cancelled, EngineError, SharedServer},
    util::first_non_empty,
};

/// Identifies a streamed generation so a client can cancel it by id through
/// `POST /v1/audio/speech/cancel`.
pub const GENERATION_ID_HEADER: HeaderName = HeaderName::from_static("x-mambotts-generation-id");

fn engine_error(err: EngineError) -> Response {
    match err {
        EngineError::NoModel => write_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_model",
            "no model loaded",
        ),
        EngineError::Cancelled => {
            write_error(StatusCode::CONFLICT, "cancelled", Cancelled.to_string())
        }
        EngineError::Failed(err) => write_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal_error",
            err.to_string(),
        ),
    }
}

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
    let PhonemizeBody { input, language } = body;
    match server
        .with_engine(move |ctx| ctx.phonemize(&input, &language))
        .await
    {
        Ok(phonemes) => Json(PhonemizeResponse { phonemes }).into_response(),
        Err(err) => engine_error(err),
    }
}

pub async fn phoneme_inventory(State(server): State<SharedServer>) -> Response {
    let info = server.info();
    if !info.loaded {
        return write_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_model",
            "no model loaded",
        );
    }
    Json(PhonemeInventoryResponse {
        phonemes: info
            .phonemes
            .iter()
            .filter(|character| character.is_alphabetic() || !character.is_ascii())
            .map(|character| character.to_string())
            .collect(),
    })
    .into_response()
}

pub async fn diacritize(
    State(server): State<SharedServer>,
    Json(body): Json<DiacritizeBody>,
) -> Response {
    if body.input.trim().is_empty() {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "request body must contain input",
        );
    }
    let DiacritizeBody { input, stress } = body;
    match server
        .with_engine(move |ctx| ctx.diacritize(&input, stress))
        .await
    {
        Ok(text) => Json(PhonemizeResponse { phonemes: text }).into_response(),
        Err(err) => engine_error(err),
    }
}

/// Whether the loaded runtime can clone a voice from a reference
/// recording, read from the registry so the answer tracks the manifest
/// rather than a hardcoded runtime name.
fn supports_voice_reference(server: &SharedServer) -> bool {
    let runtime = server.info().runtime.clone();
    mambotts_registry::runtime(&runtime)
        .is_some_and(|manifest| manifest.capabilities.voice_reference)
}

#[utoipa::path(
    post,
    path = "/v1/audio/speech",
    request_body = SpeechBody,
    responses(
        (status = 200, content_type = "audio/wav",
         headers(("x-mambotts-generation-id" = String, description = "Streamed responses only: the id to pass to /v1/audio/speech/cancel"))),
        (status = 400),
        (status = 409, description = "The generation was cancelled"),
        (status = 503),
        (status = 500)
    )
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
    if !body.voice_reference.is_empty() && !supports_voice_reference(&server) {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "the loaded runtime supports only its saved voices; \
             load a runtime with voice cloning to use voice_reference",
        );
    }
    if body.stream {
        return streaming_wav_response(server, body);
    }
    if body.input_is_phonemes {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "phoneme input requires stream=true",
        );
    }

    let voice = first_non_empty([body.voice_reference.clone(), body.voice.clone()]);
    let speed = resolve_speed(body.speed);
    let generation = server.begin_generation();
    let cancel = generation.cancel_flag();
    // If this handler is dropped (the connection went away), stop the work.
    let _cancel_on_drop = cancel.cancel_on_drop();
    let SpeechBody {
        input, language, ..
    } = body;
    let result = server
        .with_engine(move |ctx| {
            let _generation = generation;
            cancel.check()?;
            let sample_rate = ctx.sample_rate();
            let audio = ctx.synthesize_streaming(
                &input,
                (!voice.is_empty()).then_some(voice.as_str()),
                &language,
                speed,
                &mut |_, _| cancel.check(),
            )?;
            wav_bytes(&audio, sample_rate)
        })
        .await;
    match result {
        Ok(data) => wav_response(data),
        Err(err) => engine_error(err),
    }
}

#[utoipa::path(
    post,
    path = "/v1/audio/speech/cancel",
    request_body(content = Option<CancelBody>, description = "Omit the body or the id to cancel every generation"),
    responses(
        (status = 200, body = CancelResponse),
        (status = 404, description = "No running generation has that id")
    )
)]
pub async fn cancel_speech(
    State(server): State<SharedServer>,
    body: Option<Json<CancelBody>>,
) -> Response {
    let body = body.map(|Json(body)| body).unwrap_or_default();
    let id = match body
        .id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty())
    {
        None => None,
        Some(raw) => match raw.parse::<u64>() {
            Ok(id) => Some(id),
            Err(_) => return generation_not_found(),
        },
    };
    let cancelled = server.cancel_generations(id);
    if id.is_some() && cancelled == 0 {
        return generation_not_found();
    }
    Json(CancelResponse {
        status: "cancelled".into(),
        cancelled,
    })
    .into_response()
}

fn generation_not_found() -> Response {
    write_error(
        StatusCode::NOT_FOUND,
        "not_found",
        "no running generation has that id",
    )
}

/// Stream self-contained WAV chunks using a small binary frame protocol:
/// `[kind: u8][payload length: u32 big endian][payload]`, where kind `1` is a
/// playable chunk, `2` is the complete normalized WAV, and `3` is UTF-8 error
/// text. The desktop client consumes this protocol and emits each chunk to the
/// webview immediately.
///
/// The generation stops at the next chunk boundary when it is cancelled
/// through `/v1/audio/speech/cancel` or when the client drops the response.
fn streaming_wav_response(server: SharedServer, body: SpeechBody) -> Response {
    if !server.info().loaded {
        return write_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_model",
            "no model loaded",
        );
    }

    let voice = first_non_empty([body.voice_reference.clone(), body.voice.clone()]);
    let speed = resolve_speed(body.speed);
    let generation = server.begin_generation();
    let generation_id = generation.id();
    let cancel = generation.cancel_flag();
    // Owned by the response body: when hyper drops the body because the client
    // disconnected, the run is cancelled rather than finishing for nobody.
    let body_guard = cancel.cancel_on_drop();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(2);
    let error_tx = tx.clone();
    let worker = server.clone();
    tokio::spawn(async move {
        let result = worker
            .with_engine(move |ctx| {
                let _generation = generation;
                // It may have been cancelled while it waited for the engine.
                cancel.check()?;
                let sample_rate = ctx.sample_rate();
                let mut send_chunk = |samples: &[f32], sample_rate: u32| -> anyhow::Result<()> {
                    cancel.check()?;
                    // Text chunking can end with a separator-only segment. Do not send
                    // an empty WAV frame to clients, because it can interrupt queued
                    // playback without contributing any audio.
                    if samples.is_empty() {
                        return Ok(());
                    }
                    let wav = wav_bytes(samples, sample_rate)?;
                    // A closed channel means the client is gone.
                    tx.blocking_send(Ok(frame(1, wav)))
                        .map_err(|_| anyhow::Error::from(Cancelled))
                };
                let voice = (!voice.is_empty()).then_some(voice.as_str());
                let audio = if body.input_is_phonemes {
                    ctx.synthesize_phonemes_streaming(
                        &body.input,
                        voice,
                        &body.language,
                        speed,
                        &mut send_chunk,
                    )
                } else {
                    ctx.synthesize_streaming(
                        &body.input,
                        voice,
                        &body.language,
                        speed,
                        &mut send_chunk,
                    )
                }?;
                cancel.check()?;
                // The final frame is retained for download/save. It does not
                // delay playback because every chunk was already sent above.
                send_final_wav(&tx, wav_bytes(&audio, sample_rate)?);
                Ok(())
            })
            .await;
        let message = match result {
            Ok(()) => return,
            Err(EngineError::NoModel) => "no model loaded".to_string(),
            Err(EngineError::Cancelled) => Cancelled.to_string(),
            Err(EngineError::Failed(err)) => err.to_string(),
        };
        let _ = error_tx.send(Ok(frame(3, message.into_bytes()))).await;
    });

    let body = stream::unfold((rx, body_guard), |(mut receiver, guard)| async move {
        receiver.recv().await.map(|item| (item, (receiver, guard)))
    });
    let mut response = Body::from_stream(body).into_response();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-mambotts-audio-chunks"),
    );
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(GENERATION_ID_HEADER, HeaderValue::from(generation_id));
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

/// The outer bounds the API will accept. These only guard against nonsense.
/// The app's own slider is much tighter, 0.9 to 1.1, because outside that the
/// audio loses level and articulation rather than simply changing pace. The
/// wider clamp is kept for API callers who want to experiment.
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
