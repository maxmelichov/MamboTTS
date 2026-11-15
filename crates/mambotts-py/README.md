# mambotts: Python bindings

The BlueV3 engine the desktop sidecar runs, exposed to Python. Same Rust
inference path, so there is no second copy of the pipeline to keep in
sync.

BlueV3 speaks six languages (Hebrew, English, Spanish, German, Italian,
Russian) with 15 voices. The front end needs no system libraries: Hebrew
goes through heb-tts-normalizer and RenikudPlus, the other languages
through polyg2p, whose lexicons ship in the bundle's `polyg2p/` folder.

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

For a local editable install while working on the bindings, with the same
environment:

```console
uv venv && uv pip install maturin fastapi uvicorn
uv run --no-project maturin develop --release
```

## Use

```python
import mambotts

# RenikudPlus defaults to <model_dir>/renikud-plus-fp32.onnx (as bluetts.com runs it),
# else <model_dir>/renikud-plus.onnx (the int8 export); MAMBOTTS_HEBREW_G2P=int8 prefers int8
tts = mambotts.BlueTtsEngine("~/…/models/bluev3-3.0.0")
tts.sample_rate          # 44100
tts.voices()             # ['Noa', 'Lily', 'Maya', 'Emma', 'Sara', 'Shira', 'Tamar', 'Yael',
                         #  'Adam', 'Daniel', 'Ariel', 'David', 'Ethan', 'Jonathan', 'Michael']
tts.languages()          # ['he', 'en', 'es', 'de', 'it', 'ru']
tts.phonemize("שלום", "he")   # '<he>ʃalˈom</he>'

samples = tts.synthesize("שלום, מה שלומך?", voice="Lily", language="he")
samples = tts.synthesize("Привет, как дела?", voice="Adam")  # auto: Cyrillic is Russian
```

`synthesize` returns mono `f32` samples in `[-1, 1]`; writing a WAV is the
caller's job. `language="auto"` picks Hebrew for Hebrew letters, Russian
for Cyrillic, and English otherwise. `speed` is the engine's speed, as on
bluetts.com; left out, it is the site's default for the language (0.9 for
Hebrew, 0.95 for the others).

`synthesize_phonemes` takes IPA directly for text that was phonemized
elsewhere. With `language="auto"` it reads the language from a leading tag
such as `<he>`, the form `phonemize` returns, and falls back to English.

The engine holds mutable ONNX state, so it serializes calls behind its own
mutex, and releases the GIL while synthesizing. Sharing one instance
across threads is safe; concurrent calls queue rather than run in
parallel.

## Server

`python/mambotts_server.py` is a FastAPI app over these bindings, serving
the same routes as the desktop sidecar:

```console
pip install dist/mambotts-*.whl fastapi uvicorn
MAMBOTTS_MODEL_DIR=~/…/models/bluev3-3.0.0 \
MAMBOTTS_RENIKUD_PATH=~/…/renikud-plus.onnx \
  uvicorn mambotts_server:app --port 8080
```

| Route | |
| --- | --- |
| `POST /v1/audio/speech` | WAV audio; `input_is_phonemes` accepts IPA |
| `POST /v1/audio/phonemize` | text → IPA |
| `GET /v1/voices` | the voice catalog |
| `GET /v1/languages` | `auto` plus he, en, es, de, it, ru, with the sidecar's ids |
| `GET /health` | liveness, and whether the model has loaded |

The model loads on first request rather than at import, so startup is
cheap and a missing model directory surfaces as a 503 instead of a crash
at boot.
