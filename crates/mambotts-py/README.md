# mambotts — Python bindings

The BlueTTS engine the desktop sidecar runs, exposed to Python. Same Rust
inference path, so there is no second copy of the pipeline to keep in
sync.

## Build

Needs ONNX Runtime at build time, the same as the rest of the workspace:

```console
export ORT_STRATEGY=system ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_LOCATION="$PWD/../blue-rs/.ort/onnxruntime-osx-arm64-1.23.2/lib"
uv run --with maturin maturin build --release --auditwheel repair -o dist
```

`--auditwheel repair` copies `libonnxruntime` into the wheel, which is
what makes it installable on a machine that does not already have it. The
result is an abi3 wheel, so one build covers Python 3.9 and up.

## Use

```python
import mambotts

tts = mambotts.BlueTtsEngine("~/…/models/bluetts-2.5", "…/renikud-plus.onnx")
tts.sample_rate          # 44100
tts.voices()             # ['Noa', 'Lily', 'Daniel', 'Adam']
tts.phonemize("שלום", "he")

samples = tts.synthesize("שלום, מה שלומך?", voice="Lily", language="he")
```

`synthesize` returns mono `f32` samples in `[-1, 1]`; writing a WAV is the
caller's job. `synthesize_phonemes` takes IPA directly for text that was
phonemized elsewhere.

The engine holds mutable ONNX state, so it serializes calls behind its own
mutex, and releases the GIL while synthesizing. Sharing one instance
across threads is safe; concurrent calls queue rather than run in
parallel.

## Server

`python/mambotts_server.py` is a FastAPI app over these bindings, serving
the same routes as the desktop sidecar:

```console
pip install dist/mambotts-*.whl fastapi uvicorn
MAMBOTTS_MODEL_DIR=~/…/models/bluetts-2.5 \
MAMBOTTS_RENIKUD_PATH=~/…/renikud-plus.onnx \
  uvicorn mambotts_server:app --port 8080
```

| Route | |
| --- | --- |
| `POST /v1/audio/speech` | WAV audio; `input_is_phonemes` accepts IPA |
| `POST /v1/audio/phonemize` | text → IPA |
| `GET /v1/voices` | the voice catalog |
| `GET /v1/languages` | he, en, es, de, it |
| `GET /health` | liveness, and whether the model has loaded |

The model loads on first request rather than at import, so startup is
cheap and a missing model directory surfaces as a 503 instead of a crash
at boot.
