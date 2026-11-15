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

## Current runtime

BlueTTS is the only runtime, and it is the one every release ships. The
registry, the `Runtime` trait, and the `RuntimeParams` enum stay
plural-shaped so a second engine can be added beside it without reworking
the loading path. See [ADDING_MODELS.md](./ADDING_MODELS.md).

### BlueTTS (default)

- [BlueV3](https://huggingface.co/notmax123/BlueV3-onnx) v3.0.1 (checkpoint `bluev3-702000`, BlueV3-onnx `d0a0270e`): Hebrew, English, Spanish, Italian, German, and Russian local synthesis at 44.1 kHz;
- streaming WAV output;
- the release's fifteen fixed voice styles (`Noa`, `Lily`, `Daniel`, `Adam` and eleven more);
- no reference-voice cloning;
- the release's text front end, compiled in: [polyg2p](https://github.com/maxmelichov/polyg2p) reads English, Spanish, German, Italian, and Russian as written (numbers, dates and money included), and [heb-tts-normalizer](https://github.com/thewh1teagle/heb-tts-normalizer) turns Hebrew numbers, dates, currency and units into words before the G2P. eSpeak is no longer used;
- the release's operating point: ids wrapped in `<lang>` tags, 8 Euler steps at guidance 4.0 inside the vector estimator, the full decoded length of every chunk, 0.1 s between chunks and a 0.95 peak limit per chunk, one sentence a chunk as on bluetts.com (packed sentences run together with almost no pause), and the speed at the release's 1.0 in every language, multiplied by the voice's `speed_scale` for the language (each v3.0.1 voice JSON carries one; a voice must be labelled with the graphs' checkpoint, `bluev3-702000`);
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) ONNX phonemization for Hebrew, including optional source/target speaker conditioning. `crates/renikud-plus-rs` is a port of the upstream Python package: the exact-MAP cascade decode, niqqud in the input read as hard constraints on that decode, the Hebrew number front end, the long-input windowing and the force lexicon;
- Hebrew diacritization (`POST /v1/diacritize`) from the same RenikudPlus pass: its per-letter consonant and vowel predictions are written back as niqqud. Pointed text is read as bluetts.com and the BlueV3 reference read it, points dropped, unless the request sets `"niqqud": true` (the app does for its niqqud editor's text): then phonemization reads the points back as hard constraints, so the app's editable niqqud is what gets spoken; the marks pointing alone cannot carry (stress, a silent ו, a consonantal י, a pronounced final ה) are written as the hatama, rafe, shva and mapiq. Send `"stress": false` for niqqud with no stress mark.

Its bundle is installed in the application data directory under `models/bluev3-3.0.1/`: the four ONNX graphs, `vocab.json`, `tts.json`, the voice styles under `voices/`, polyg2p's lexicons under `polyg2p/`, and one RenikudPlus file chosen as bluetts.com chooses it: `renikud-plus-fp32.onnx` (the float `model.onnx`) on machines with more than 4 GB of memory, `renikud-plus.onnx` (the int8 export) on smaller ones; `MAMBOTTS_HEBREW_G2P=fp32|int8` overrides the choice. Every file is fetched from a pinned revision, except the ones the model repository keeps in git rather than LFS (`vocab.json`, `tts.json`, the fifteen voices, `ru_homographs.lex`, `ru_yo_homographs.lex` and polyg2p's licence files): a squash of the Hub's history deletes git-stored files at old revisions while LFS objects survive, so the registry compiles those in (`crates/mambotts-registry/shipped/`, checked against their SHA-256) and the app writes them itself; the server serves them at `GET /v1/models/files/{name}` for `install.ps1`. A download answered 401, 403, 404 or 410 fails at once with a message saying the file is gone, rather than being retried. An install of BlueV3 v3.0.0 (`models/bluev3-3.0.0/`) keeps running while the app moves it to v3.0.1 in the background: the vocabulary, the vocoder, polyg2p's data and RenikudPlus are byte-identical between the releases and are taken from the old directory (hard links, verified against the registry's SHA-256); only the changed graphs, `tts.json` and the voices are downloaded. The app switches to the new directory once every file is verified, then removes the old one. BlueTTS 2.5 bundles are not loaded any more; an existing `models/bluetts-2.5/` directory is simply unused.

Kokoro is not a shipped runtime. Historical code and documentation must not be interpreted as available functionality.

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
