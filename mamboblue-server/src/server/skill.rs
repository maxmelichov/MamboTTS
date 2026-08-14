const TEMPLATE: &str = r#"# MamboRambo Local TTS API

You are using MamboRambo, a local BlueTTS HTTP API. The shipped runtime supports Hebrew and English, fixed voices, and streaming WAV output. It does not support voice cloning.

Hebrew grapheme-to-IPA uses RenikudPlus (`renikud-plus.onnx`) with optional `speaker` / `target_speaker` conditioning (0=unknown, 1=male, 2=female). Phonikud is optional for diacritics when selected.

Base URL: {{base_url}}
OpenAPI schema: {{base_url}}/openapi.json
Swagger docs: {{base_url}}/docs
Model sources: {{base_url}}/v1/models/sources

Before calling the API, fetch the OpenAPI schema from /openapi.json and use it as the source of truth for request and response shapes.

Recommended flow:

1. Call GET /health.
2. If loaded=false, call GET /v1/models/sources to discover runtimes, model download URLs, and default MamboRambo Desktop model locations.
3. Check whether the model files already exist in MamboRambo Desktop's default model directory.
4. Call POST /v1/models/load with `runtime`, `model_path`, and `renikud_path` pointing at `renikud-plus.onnx`. Optional: `hebrew_g2p_engine` (`renikud` or `phonikud`), `speaker`, `target_speaker`.
5. Optional IPA preview: POST /v1/phonemize, then edit phonemes client-side.
6. Call POST /v1/audio/speech to synthesize speech. For edited IPA, set `input_is_phonemes: true` and `stream: true` (phoneme input requires streaming).
7. Non-streaming responses return a standalone WAV (`stream: false`). Streaming responses use MamboRambo binary frames.

Example:

~~~sh
curl {{base_url}}/health

curl {{base_url}}/v1/models/sources

curl -X POST {{base_url}}/v1/models/load \
  -H 'Content-Type: application/json' \
  -d '{"runtime":"blue","model_path":"/path/to/blue-onnx-v2","renikud_path":"/path/to/blue-onnx-v2/renikud-plus.onnx","hebrew_g2p_engine":"renikud","speaker":0,"target_speaker":0}'

curl -X POST {{base_url}}/v1/phonemize \
  -H 'Content-Type: application/json' \
  -d '{"input":"שלום מממבו רמבו","language":"he"}'

curl -X POST {{base_url}}/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -o speech.wav \
  -d '{"input":"שלום מממבו רמבו","language":"auto","voice":"Rotem","response_format":"wav","stream":false}'
~~~

Useful endpoints for the desktop Phoneme editor:

- POST /v1/phonemize → `{"phonemes":"..."}`
- GET /v1/phonemes → BlueTTS phoneme inventory
- POST /v1/diacritize → Hebrew diacritics (Phonikud only)
- POST /v1/audio/speech with `input_is_phonemes: true` and `stream: true` to speak edited IPA

If the API returns no_model, ask the user to install the MamboRambo model in the desktop app first.
"#;

pub fn render_skill(host: &str) -> String {
    TEMPLATE.replace("{{base_url}}", &format!("http://{host}"))
}
