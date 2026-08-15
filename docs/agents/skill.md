# MamboTTS Agent Skill

MamboTTS is a local, offline TTS server. The current shipped engine is [BlueTTS](https://github.com/maxmelichov/BlueTTS), with Hebrew, English, Spanish, Italian, and German support, streamed WAV output, and fixed voice styles. Hebrew IPA uses [RenikudPlus](https://github.com/maxmelichov/RenikudPlus), with optional [Phonikud](https://github.com/phonikud/phonikud) diacritics controls. It does not support voice cloning or Qwen model files.

## Start the server

The server can start without a model and load one through HTTP:

```console
mambotts-server serve --host 127.0.0.1 --port 8080 --exit-with-parent false
```

Or load the Blue bundle immediately:

```console
mambotts-server serve \
  --host 127.0.0.1 \
  --port 8080 \
  --model-dir /path/to/blue-onnx-v2 \
  --renikud /path/to/blue-onnx-v2/renikud-plus.onnx \
  --exit-with-parent false
```

The model directory must include `duration_predictor.onnx`, `text_encoder.onnx`, `vector_estimator.onnx`, `vocoder.onnx`, `vocab.json`, `tts.json`, the `voices/` directory, and `renikud-plus.onnx`.

## Load a model over HTTP

```console
curl -sS http://127.0.0.1:8080/v1/models/load \
  -H 'content-type: application/json' \
  -d '{
    "runtime": "blue",
    "model_path": "/path/to/blue-onnx-v2",
    "renikud_path": "/path/to/blue-onnx-v2/renikud-plus.onnx",
    "hebrew_g2p_engine": "renikud",
    "speaker": 0,
    "target_speaker": 0
  }'
```

`hebrew_g2p_engine` is `renikud` (RenikudPlus ONNX, default) or `phonikud`. Speaker IDs are `0` unknown, `1` male, `2` female.

Discover installed runtime metadata before presenting controls:

```console
curl -sS http://127.0.0.1:8080/v1/models/sources
curl -sS http://127.0.0.1:8080/v1/languages
curl -sS http://127.0.0.1:8080/v1/voices
```

## Preview and edit IPA

```console
curl -sS http://127.0.0.1:8080/v1/phonemize \
  -H 'content-type: application/json' \
  -d '{"input":"שלום, זהו ממו רמבו.","language":"he"}'

curl -sS http://127.0.0.1:8080/v1/phonemes
```

To synthesize edited IPA, set `input_is_phonemes: true` and `stream: true` (phoneme input requires streaming).

## Create speech

```console
curl -sS http://127.0.0.1:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{
    "input": "שלום, זהו ממו רמבו.",
    "language": "he",
    "voice": "Noa",
    "response_format": "wav",
    "stream": false
  }' \
  --output output.wav
```

Use `language: "auto"` to detect Hebrew or English. Query `/v1/voices` rather than hardcoding a voice list. For streamed requests, the response body uses MamboTTS binary frames instead of a standalone WAV; desktop clients should decode and save the final frame.

## Available endpoints

```text
GET    /health
GET    /skill
GET    /v1/models
GET    /v1/models/sources
POST   /v1/models/load
DELETE /v1/models
GET    /v1/languages
GET    /v1/voices
GET    /v1/phonemes
POST   /v1/phonemize
POST   /v1/diacritize
POST   /v1/audio/speech
```

## Troubleshooting

- If no model is loaded, call `POST /v1/models/load` with Blue's directory and RenikudPlus path (`renikud-plus.onnx`).
- If a language or voice is rejected, query the loaded runtime metadata first.
- Do not send `voice_reference` to BlueTTS; it is unsupported.
- The sidecar requires ONNX Runtime shared libraries distributed with the desktop build.
- Diacritics (`/v1/diacritize`) require Phonikud selected and downloaded.
