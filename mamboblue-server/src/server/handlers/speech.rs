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
    util::first_non_empty,
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
    if !body.voice_reference.is_empty() {
        return write_error(
            StatusCode::BAD_REQUEST,
            "invalid_request",
            "BlueTTS supports only the saved Rotem and Roi voices; voice cloning is unavailable",
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
        .prefix("mamborambo-speech-")
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
    let voice = first_non_empty([body.voice]);
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

    let voice = first_non_empty([body.voice]);
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
                &mut send_chunk,
            )
        } else {
            ctx.synthesize_streaming(
                &body.input,
                (!voice.is_empty()).then_some(voice.as_str()),
                &body.language,
                &mut send_chunk,
            )
        };
        match result {
            Ok(audio) => {
                // The final frame is retained for download/save. It does not
                // delay playback because every chunk was already sent above.
                match wav_bytes(&audio, sample_rate) {
                    Ok(wav) => {
                        let _ = tx.blocking_send(Ok(frame(2, wav)));
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
        HeaderValue::from_static("application/x-mamborambo-audio-chunks"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
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
