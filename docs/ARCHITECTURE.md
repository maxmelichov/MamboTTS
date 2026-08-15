# Architecture

MamboTTS is a local-first TTS application for macOS on Apple Silicon. It has three layers:

```text
React + Tauri desktop app
        ↓ local HTTP
mambotts-server sidecar
        ↓ Runtime trait
TTS runtime crates and downloaded model bundles
```

The desktop never loads a native TTS runtime itself. It starts `mambotts-server` as a sidecar, waits for its ready signal, and calls its local HTTP API.

## Runtime registry

`crates/mambotts-registry` is the source of truth for all model metadata. A runtime manifest declares:

- its stable ID, display name, version, size, and install directory;
- model files and download URLs;
- files that must exist before it is considered installed;
- UI and API capabilities: Hebrew support, streaming, reference-voice support, and fixed voices.

Both the desktop downloader and server `GET /v1/models/sources` build their catalog from this registry. This prevents the app and sidecar from advertising different models.

To add a model, add its manifest to the registry, implement a server `Runtime` adapter, add a `RuntimeParams` variant and loading validation, then package its native dependencies with the sidecar. The model picker reads the registry data; it does not need a hardcoded card per runtime.

Step-by-step contribution guide (licenses, hosting, desktop wiring, and PR checklist): [ADDING_MODELS.md](./ADDING_MODELS.md).

## Current runtimes

BlueTTS is the default runtime and the one every release ships enabled.
QwenTTS Hebrew sits beside it as an opt-in second runtime; selecting one
never disables the other.

### BlueTTS (default)

- Hebrew, English, Spanish, Italian, and German local synthesis;
- streaming WAV output;
- fixed `Rotem` and `Roi` voice styles;
- no reference-voice cloning;
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) ONNX phonemization for Hebrew, including optional source/target speaker conditioning;
- optional [Phonikud](https://github.com/phonikud/phonikud) vocalization and diacritics controls.

Its bundle is installed in the application data directory under `models/blue-onnx-v2/` and requires the [BlueTTS](https://github.com/maxmelichov/BlueTTS) ONNX pipeline, voice embeddings, and `renikud-plus.onnx`.

### QwenTTS Hebrew (`qwen_he`, opt-in)

[QwenTTS-he-1.7B](https://huggingface.co/notmax123/QwenTTS-he-1.7B) is a
LoRA adapter over `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, merged into the base
talker and converted to GGUF. It runs on GGML through
[qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), wrapped by
the `qwentts-rs` crate:

- Hebrew only — the adapter retrained the output heads for Hebrew
  phonotactics, so the base model's other nine languages are not claimed;
- streaming output at 24 kHz;
- voice cloning from a reference recording, and **no** fixed voices: the
  `voice` field carries a path to a WAV file rather than a name;
- RenikudPlus is mandatory, not optional. The checkpoint reads *stressed
  IPA* (`ʃalˈom, mˈa ʃlomχˈa`), never Hebrew script, so the same G2P front
  end Blue uses supplies its input.

The bundle installs under `models/qwentts-he-1.7b/` and is roughly 1.5 GB:
a Q4_K_M talker GGUF, the 12 Hz codec GGUF, and `renikud-plus.onnx`.

The engine compiles qwentts.cpp and GGML from source, so it lives behind
a Cargo feature and is **off by default**:

```console
cargo build -p mambotts-server --release --features qwen,metal
```

Without that feature the sidecar still serves the registry entry, and
`POST /v1/models/load` answers with a build-specific error rather than
pretending the runtime is present.

Qwen and Kokoro are not currently shipped runtimes. Historical code and documentation must not be interpreted as available functionality.

## Server API

The sidecar owns runtime loading and inference:

- `POST /v1/models/load` accepts `runtime`, `model_path`, and runtime-specific fields. Blue requires `renikud_path`.
- `GET /v1/models/sources` returns registered downloadable model manifests and capabilities.
- `GET /v1/languages` and `GET /v1/voices` report metadata for the loaded runtime.
- `POST /v1/audio/speech` returns framed streaming WAV data.

The streaming frame format is `[kind: u8][length: u32 big-endian][payload]`: kind `1` is a playable WAV chunk, `2` the final WAV, and `3` an error message.

## Packaging

`scripts/pre_build.py` builds the sidecar for Tauri's target triple and places it in `mambotts-desktop/src-tauri/binaries/`. It also stages ONNX Runtime libraries beside the sidecar and configures their platform loader paths. Tauri then bundles that sidecar and its native libraries into the desktop installer.

Dynamic native plugins are intentionally not used. They make code-signing, ABI compatibility, and bundled dependency resolution unsafe across three operating systems. Runtimes are compiled into a versioned sidecar; model assets remain independently downloadable.
