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

`crates/mambotts-registry` is the source of truth for the model metadata. Its one manifest, BlueV3 v3.0.1, declares:

- its stable ID, display name, version, size, and install directory;
- model files and download URLs;
- files that must exist before it is considered installed;
- its capabilities: Hebrew support, streaming, and fixed voices.

Both the desktop downloader and server `GET /v1/models/sources` build their catalog from this registry, so the app and the sidecar describe the same bundle.

## Current runtime

BlueV3 v3.0.1 is the only model the app, the server and the installers know.
Requests for any other runtime are rejected. Earlier models (BlueTTS 2.5,
BlueV3 v3.0.0, and the older `blue-onnx-v2`, `qwentts-he-1.7b` and
`phonikud-v1` directories) are removed from the models directory once v3.0.1
is installed.

### BlueTTS (default)

- [BlueV3](https://huggingface.co/notmax123/BlueV3-onnx) v3.0.1 (checkpoint `bluev3-702000`, BlueV3-onnx `d0a0270e`): Hebrew, English, Spanish, Italian, German, and Russian local synthesis at 44.1 kHz;
- streaming WAV output;
- the release's fifteen fixed voice styles (Maya, Noa, Lily, Tamar, Shira, Yael, Emma, Sara, Adam, Daniel, David, Jonathan, Ethan, Michael, Ariel; Shira by default);
- no reference-voice cloning;
- the release's text front end, compiled in: [polyg2p](https://github.com/maxmelichov/polyg2p) reads English, Spanish, German, Italian, and Russian as written (numbers, dates and money included), and [heb-tts-normalizer](https://github.com/thewh1teagle/heb-tts-normalizer) turns Hebrew numbers, dates, currency and units into words before the G2P. eSpeak is no longer used;
- the release's operating point: ids wrapped in `<lang>` tags, 8 Euler steps at guidance 4.0 inside the vector estimator, the full decoded length of every chunk, 0.1 s between chunks and a 0.95 peak limit per chunk, one sentence a chunk as on bluetts.com (packed sentences run together with almost no pause), and the speed at the release's 1.0 in every language, multiplied by the voice's `speed_scale` for the language (each v3.0.1 voice JSON carries one; a voice must be labelled with the graphs' checkpoint, `bluev3-702000`);
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) ONNX phonemization for Hebrew, including optional source/target speaker conditioning. `crates/renikud-plus-rs` is a port of the upstream Python package: the exact-MAP cascade decode, niqqud in the input read as hard constraints on that decode, the Hebrew number front end, the long-input windowing and the force lexicon;
- Hebrew diacritization (`POST /v1/diacritize`) from the same RenikudPlus pass: its per-letter consonant and vowel predictions are written back as niqqud. Pointed text is read as the BlueV3 reference reads it (`site_flow.parse_points`): the points are hard constraints on the decode, phonikud's marks included (hatama, meteg, nikud haser, the prefix bar), so the app's editable niqqud, or any pointed text, is what gets spoken; text without points reads the bare letters as before. The marks pointing alone cannot carry (stress, a silent ו, a consonantal י, a pronounced final ה) are written as the hatama, rafe, a shva (a dagesh before a vowel vav, as in הַיּוֹם) and mapiq. Send `"stress": false` for niqqud with no stress mark.

Its bundle is installed in the application data directory under `models/bluev3-3.0.1/`: the four ONNX graphs, `vocab.json`, `tts.json`, the voice styles under `voices/`, polyg2p's lexicons under `polyg2p/`, and one RenikudPlus file chosen as bluetts.com chooses it: `renikud-plus-fp32.onnx` (the float `model.onnx`) on machines with more than 4 GB of memory, `renikud-plus.onnx` (the int8 export) on smaller ones; `MAMBOTTS_HEBREW_G2P=fp32|int8` overrides the choice. Every file is fetched from a pinned revision, except the ones the model repository keeps in git rather than LFS (`vocab.json`, `tts.json`, the fifteen voices, `ru_homographs.lex`, `ru_yo_homographs.lex` and polyg2p's licence files): a squash of the Hub's history deletes git-stored files at old revisions while LFS objects survive, so the registry compiles those in (`crates/mambotts-registry/shipped/`, checked against their SHA-256) and the app writes them itself; the server serves them at `GET /v1/models/files/{name}` for `install.ps1`. A download answered 401, 403, 404 or 410 fails at once with a message saying the file is gone, rather than being retried. An install of BlueV3 v3.0.0 (`models/bluev3-3.0.0/`) keeps running while the app moves it to v3.0.1 in the background: the vocabulary, the vocoder, polyg2p's data and RenikudPlus are byte-identical between the releases and are taken from the old directory (hard links, verified against the registry's SHA-256); only the changed graphs, `tts.json` and the voices are downloaded. The app switches to the new directory once every file is verified, then removes the old one. Coming from 1.3.x, the RenikudPlus file in `models/bluetts-2.5/` is reused when its SHA-256 matches, and that directory is removed with the other retired ones once v3.0.1 is installed.

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
