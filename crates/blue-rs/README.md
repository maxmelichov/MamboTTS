# blue-rs

Rust ONNX inference for BlueTTS, built for MamboTTS with [Chirp](https://github.com/thewh1teagle/chirp).

It runs a [BlueV3](https://huggingface.co/notmax123/BlueV3-onnx) v3.0.1 bundle
with the release's own front end: [polyg2p](https://github.com/maxmelichov/polyg2p)
for English, Spanish, German, Italian and Russian,
[heb-tts-normalizer](https://github.com/thewh1teagle/heb-tts-normalizer) and
RenikudPlus for Hebrew.

## Run

```bash
cargo run -p blue-rs --example speak -- <model_dir> noa he "שלום עולם" out.wav
```

`model_dir` is an installed MamboTTS bundle (`models/bluev3-3.0.1`): the four
graphs, `vocab.json`, `tts.json`, `voices/`, `polyg2p/` and `renikud-plus.onnx`.
The `smoke` example takes IPA instead of text and needs only the graphs.

## Text handling contract

The front end reproduces BlueV3's reference `site_flow.py` (v3.0.1), including
its bot rules (`bot_prepare`: pictographs and invisible marks, flattened tags,
one-line Hebrew lists, then `src/bot.rs`'s round-2 rules on every piece in its
own language: phone numbers in every language, cards, fractions, scores,
versions, minus signs, arithmetic, ranges, ISO dates, degrees, units, Russian
ordinals, dates, years, currency and spaced thousands, leading zeros, URLs,
handles, Markdown and bullets; e-mail addresses read in English, typographic
quotes and Cyrillic inside Hebrew; English words over 20 letters read in
pieces, letterless units not spoken, the early clause cut).

Durations follow v3.0.1's per-voice calibration: `raw / max(speed *
speed_scale[lang], 1e-6)` (the divisor in float64, the division in float32),
`speed_scale` from the voice JSON (`en-us` and `ge` resolve to `en` and `de`, a
language it does not list is 1.0), the default speed 1.0 in every language.
A voice must carry the graphs' checkpoint label (`"model": "bluev3-702000"`).

bluetts.com applies the same rules (LightBlue-site `golden-parity`);
`Phonemizer::set_bot_rules(false)` reads text as the site did before them.

Pointed Hebrew is read with its points, as the reference reads it
(`site_flow.parse_points`, tests/golden/README.md "Pointed Hebrew"): vowel
signs, dagesh, shin/sin dots and phonikud's marks (hatama, meteg, nikud haser,
the prefix bar) are hard constraints on RenikudPlus's decode, so בַּר and בֶּר
differ. Hebrew without points reads exactly as before.

`tests/golden.rs` checks every case of `tests/golden/cases.jsonl`, stage by
stage (637 cases: `plan`, `compact` plans of 2k-50k characters compared by
digest, `pack`, `geometry`, and `duration`, which also runs the bundle's
duration predictor for each voice when a v3.0.1 bundle is at hand). It is
vendored, with `vocab.json`, from BlueV3 `tests/golden/`: after the
reference changes, regenerate it there (`make_golden.py`) and copy both files
here. `BLUEV3_GOLDEN=<cases.jsonl>` reads another copy.

```bash
BLUEV3_RENIKUD=<BlueV3>/renikud.onnx cargo test -p blue-rs --test golden -- --nocapture
```

The G2Ps come from `MAMBOTTS_BLUE_MODEL_DIR` or the installed `bluev3-3.0.1`
bundle (`BLUEV3_RENIKUD`, `BLUEV3_POLYG2P_DATA` override them); cases whose G2P
is missing are skipped. The golden file was made with RenikudPlus's float
model, the bundle's `renikud-plus-fp32.onnx`; with the int8 export
(`renikud-plus.onnx`, what machines with 4 GB of memory or less get) a Hebrew
case that differs only in its phonemes is reported as drift, not failed.

## Parity with bluetts.com, end to end

`scripts/site-parity/run.mjs` runs the cases of `scripts/site-parity/cases.json`
through mambotts-server's synthesis path and through the site's own
`js/onnx-web-worker.js` in Node, and compares the prepared text, the chunks,
their token ids, durations and latent frames, and the audio sample by sample.
Its header lists the inputs it needs.
