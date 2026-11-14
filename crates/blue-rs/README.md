# blue-rs

Rust ONNX inference for BlueTTS, built for MamboTTS with [Chirp](https://github.com/thewh1teagle/chirp).

It runs a [BlueV3](https://huggingface.co/notmax123/BlueV3-onnx) v3.0.0 bundle
with the release's own front end: [polyg2p](https://github.com/maxmelichov/polyg2p)
for English, Spanish, German, Italian and Russian,
[heb-tts-normalizer](https://github.com/thewh1teagle/heb-tts-normalizer) and
RenikudPlus for Hebrew.

## Run

```bash
cargo run -p blue-rs --example speak -- <model_dir> noa he "שלום עולם" out.wav
```

`model_dir` is an installed MamboTTS bundle (`models/bluev3-3.0.0`): the four
graphs, `vocab.json`, `tts.json`, `voices/`, `polyg2p/` and `renikud-plus.onnx`.
The `smoke` example takes IPA instead of text and needs only the graphs.
