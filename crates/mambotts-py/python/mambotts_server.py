"""HTTP server for MamboTTS, backed by the Rust engine through `mambotts`.

Serves the same shape as the desktop sidecar so existing clients work
unchanged:

    POST /v1/audio/speech   -> audio/wav
    POST /v1/audio/phonemize
    GET  /v1/voices
    GET  /v1/languages
    GET  /health

Run it with the model directory in the environment:

    MAMBOTTS_MODEL_DIR=~/…/models/bluetts-2.5 \\
      uvicorn mambotts_server:app --port 8080

The engine serializes calls internally, so there is no lock around
synthesis here; the only lock guards first-use construction.
"""

from __future__ import annotations

import io
import os
import struct
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import mambotts

MODEL_DIR = os.environ.get("MAMBOTTS_MODEL_DIR")
RENIKUD_PATH = os.environ.get("MAMBOTTS_RENIKUD_PATH")

app = FastAPI(title="MamboTTS", version="0.1.0")

_engine: mambotts.BlueTtsEngine | None = None
# Guards construction only. Holding it across a request would deadlock,
# because every handler calls engine() to get the instance.
_load_lock = threading.Lock()


def engine() -> mambotts.BlueTtsEngine:
    """Loads the engine on first use so import stays cheap."""
    global _engine
    with _load_lock:
        if _engine is None:
            if not MODEL_DIR:
                raise HTTPException(503, "MAMBOTTS_MODEL_DIR is not set")
            model_dir = Path(MODEL_DIR).expanduser()
            if not model_dir.is_dir():
                raise HTTPException(503, f"model directory not found: {model_dir}")
            _engine = mambotts.BlueTtsEngine(
                str(model_dir),
                str(Path(RENIKUD_PATH).expanduser()) if RENIKUD_PATH else None,
            )
        return _engine


class SpeechBody(BaseModel):
    input: str
    voice: str = ""
    language: str = "auto"
    response_format: str = "wav"
    input_is_phonemes: bool = False
    speed: float = Field(0.95, gt=0)


class PhonemizeBody(BaseModel):
    input: str
    language: str = "auto"


def wav_bytes(samples: list[float], sample_rate: int) -> bytes:
    """Packs mono float samples into a 16-bit PCM WAV."""
    frames = b"".join(
        struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples
    )
    out = io.BytesIO()
    out.write(b"RIFF")
    out.write(struct.pack("<I", 36 + len(frames)))
    out.write(b"WAVEfmt ")
    out.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    out.write(b"data")
    out.write(struct.pack("<I", len(frames)))
    out.write(frames)
    return out.getvalue()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "loaded": _engine is not None})


@app.get("/v1/voices")
def voices() -> JSONResponse:
    return JSONResponse({"runtime": "blue", "voices": engine().voices()})


@app.get("/v1/languages")
def languages() -> JSONResponse:
    return JSONResponse({"languages": engine().languages()})


@app.post("/v1/audio/phonemize")
def phonemize(body: PhonemizeBody) -> JSONResponse:
    if not body.input.strip():
        raise HTTPException(400, "request body must contain input")
    tts = engine()
    try:
        return JSONResponse({"phonemes": tts.phonemize(body.input, body.language)})
    except ValueError as err:
        raise HTTPException(400, str(err)) from err


@app.post("/v1/audio/speech")
def speech(body: SpeechBody) -> Response:
    if not body.input.strip():
        raise HTTPException(400, "request body must contain input")
    if body.response_format and body.response_format != "wav":
        raise HTTPException(400, "only wav response_format is supported")

    tts = engine()
    # No lock here: the engine serializes calls behind its own mutex and
    # releases the GIL while synthesizing.
    try:
        synth = tts.synthesize_phonemes if body.input_is_phonemes else tts.synthesize
        samples = synth(
            body.input,
            voice=body.voice or None,
            language=body.language,
            speed=body.speed,
        )
    except KeyError as err:
        raise HTTPException(400, str(err)) from err
    except ValueError as err:
        raise HTTPException(400, str(err)) from err
    rate = tts.sample_rate

    return Response(content=wav_bytes(samples, rate), media_type="audio/wav")
