<p align="center">
  <a target="_blank" href="https://maxmelichov.github.io/MamboRambo-site/">
    <img width="240" alt="MamboTTS logo" src="./mambotts-desktop/src/assets/mambotts-logo.png" />
  </a>
</p>

<h1 align="center">MamboTTS</h1>

<p align="center">
  <strong>Native offline BlueTTS for desktop</strong>
</p>

<p align="center">
  <a target="_blank" href="https://maxmelichov.github.io/MamboRambo-site/">
    🔗 Download MamboTTS
  </a>
  &nbsp; | &nbsp; Give it a Star ⭐ | &nbsp;
  <a target="_blank" href="https://github.com/sponsors/maxmelichov">Support the project 🤝</a>
</p>

<hr />

<p align="center">
  <a target="_blank" href="https://maxmelichov.github.io/MamboRambo-site/">
    <img width="800" alt="MamboTTS desktop screenshot" src="./docs/images/mambotts-studio.png" />
  </a>
</p>

## Features

- Local text-to-speech with BlueTTS
- Fully offline generation after the model is downloaded
- Saved voices: Noa, Lily, Daniel, and Adam
- Supported languages: Hebrew, English, Spanish, German, and Italian
- Audio preview after creation
- 💻 Desktop support for `macOS` on Apple Silicon
- Local HTTP API with Swagger docs for tools and automation
- Agent-ready `/skill` instructions for AI workflows

## Models and phonemizers

MamboTTS builds on these open-source projects:

- [BlueTTS](https://github.com/maxmelichov/BlueTTS) — local ONNX text-to-speech runtime, shipped by default
- [QwenTTS-he-1.7B](https://huggingface.co/notmax123/QwenTTS-he-1.7B) — Hebrew LoRA over [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base), an opt-in runtime with reference-audio voice cloning
- [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) — GGML inference for Qwen3-TTS, wrapped by the `qwentts-rs` crate
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) — Hebrew grapheme-to-IPA conversion with speaker conditioning
- [Phonikud](https://github.com/phonikud/phonikud) — Hebrew vocalization and diacritics-aware IPA tools

## Build

See [BUILDING.md](docs/BUILDING.md).

## Adding models

Want to ship another open-source TTS model or voice bundle with MamboTTS? See [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) for licensing, registry wiring, server/desktop steps, and the PR checklist.

---

The is code taken from [Chirp](https://github.com/thewh1teagle/chirp).
