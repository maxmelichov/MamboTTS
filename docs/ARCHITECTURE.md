# Architecture

MamboBlue is a local-first TTS application for macOS, Windows, and Linux. It has three layers:

```text
React + Tauri desktop app
        ↓ local HTTP
mamboblue-server sidecar
        ↓ Runtime trait
TTS runtime crates and downloaded model bundles
```

The desktop never loads a native TTS runtime itself. It starts `mamboblue-server` as a sidecar, waits for its ready signal, and calls its local HTTP API.

## Runtime registry

`crates/mamboblue-registry` is the source of truth for all model metadata. A runtime manifest declares:

- its stable ID, display name, version, size, and install directory;
- model files and download URLs;
- files that must exist before it is considered installed;
- UI and API capabilities: Hebrew support, streaming, reference-voice support, and fixed voices.

Both the desktop downloader and server `GET /v1/models/sources` build their catalog from this registry. This prevents the app and sidecar from advertising different models.

To add a model, add its manifest to the registry, implement a server `Runtime` adapter, add a `RuntimeParams` variant and loading validation, then package its native dependencies with the sidecar. The model picker reads the registry data; it does not need a hardcoded card per runtime.

Step-by-step contribution guide (licenses, hosting, desktop wiring, and PR checklist): [ADDING_MODELS.md](./ADDING_MODELS.md).

## Current runtime

BlueTTS is the currently shipped runtime:

- Hebrew, English, Spanish, Italian, and German local synthesis;
- streaming WAV output;
- fixed `Rotem` and `Roi` voice styles;
- no reference-voice cloning;
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) ONNX phonemization for Hebrew, including optional source/target speaker conditioning;
- optional [Phonikud](https://github.com/phonikud/phonikud) vocalization and diacritics controls.

Its bundle is installed in the application data directory under `models/blue-onnx-v2/` and requires the [BlueTTS](https://github.com/maxmelichov/BlueTTS) ONNX pipeline, voice embeddings, and `renikud-plus.onnx`.

Qwen and Kokoro are not currently shipped runtimes. Historical code and documentation must not be interpreted as available functionality.

## Server API

The sidecar owns runtime loading and inference:

- `POST /v1/models/load` accepts `runtime`, `model_path`, and runtime-specific fields. Blue requires `renikud_path`.
- `GET /v1/models/sources` returns registered downloadable model manifests and capabilities.
- `GET /v1/languages` and `GET /v1/voices` report metadata for the loaded runtime.
- `POST /v1/audio/speech` returns framed streaming WAV data.

The streaming frame format is `[kind: u8][length: u32 big-endian][payload]`: kind `1` is a playable WAV chunk, `2` the final WAV, and `3` an error message.

## Packaging

`scripts/pre_build.py` builds the sidecar for Tauri's target triple and places it in `mamboblue-desktop/src-tauri/binaries/`. It also stages ONNX Runtime libraries beside the sidecar and configures their platform loader paths. Tauri then bundles that sidecar and its native libraries into the desktop installer.

Dynamic native plugins are intentionally not used. They make code-signing, ABI compatibility, and bundled dependency resolution unsafe across three operating systems. Runtimes are compiled into a versioned sidecar; model assets remain independently downloadable.
