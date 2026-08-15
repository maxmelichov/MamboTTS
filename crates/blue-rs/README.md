# blue-rs

Rust ONNX inference for BlueTTS, built for MamboTTS with [Chirp](https://github.com/thewh1teagle/chirp).

## Run

```bash
cd crates/blue-rs
cargo run --example hebrew -- "שלום עולם"
```

Models belong in `onnx_models/`; Hebrew also needs `renikud.onnx`.
