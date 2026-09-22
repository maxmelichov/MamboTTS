# RenikudPlus Rust port — upstream parity

`renikud_parity_001.py` runs one corpus through the upstream Python package
(`renikud_onnx` 0.6.0, imported from a local clone) and through
`crates/renikud-plus-rs` (via the `parity` example, which speaks JSONL), then
diffs the two outputs character by character. The reference implementation is
used for comparison only — nothing in it ships.

## Running it

```console
export RENIKUD_UPSTREAM=C:/Users/maxme/code/renikud-plus-upstream
export MAMBOTTS_RENIKUD_PATH=.../models/bluetts-2.5/renikud-plus.onnx
export CARGO_TARGET_DIR=...           # optional
export ORT_STRATEGY=system ORT_LIB_LOCATION=.../.ort/onnxruntime-win-x64-1.23.2/lib
export ORT_PREFER_DYNAMIC_LINK=1
uv run plans/renikud-parity/renikud_parity_001.py
```

`--only numbers,align,lexicon` skips the model runs (seconds instead of an
hour), `--limit N` cuts the corpus down, and `--skip-rust` / `--skip-upstream`
reuse `out/rust.jsonl` / `out/upstream.jsonl` from an earlier run. Differences
land in `out/diffs.json`; the exit status is non-zero when there are any.

Both sides must run the same ONNX Runtime, because the logits themselves differ
slightly between releases: the harness pins `onnxruntime==1.23.2` in its script
metadata, which is the version `crates/blue-rs/.ort` carries for `ort`.

## The corpus

327 distinct texts: 100 ordinary sentences, 45 pointed words and phrases (the
four ספר pointings and the pointed possessive-suffix words from the README
among them), 20 acronyms with gershayim and geresh letters, 15 mixed
Hebrew/English texts, 50 numeric texts (clock, date, year, percent, decimal,
identifier, phone, score), 10 gendered readings, single words, question forms
and joined sentence pairs — plus four long inputs (>5,000 characters, a
word-boundary case, an unbroken 2,100-character run and a clause-boundary
case).

Those texts are expanded into ~1,970 cases: `phonemize` in both niqqud modes,
the greedy decode, the four speaker/target-speaker combinations, `number_norm`
off, `vocalize` in both modes, the number front end on its own, `align_word`
with and without the extended cells, and force-lexicon loading plus forced
decodes.

## Results

**1,969/1,969 identical** (phonemize 471 strip + 399 use, vocalize 331 strip +
327 use, numbers 377, aligner 40, lexicon 24). No diffs to explain. The run is
deterministic: the same corpus, the same model file and the same ONNX Runtime
give the same outputs on both sides.

## `renikud_parity_002.py`: what a pointing round trip costs

`/v1/diacritize` hands the desktop pointed Hebrew, the user edits it, and
`/v1/phonemize` reads it back, so the pointing has to say everything the
reading did. Upstream `vocalize` does not: it writes no stress mark, leaves a
silent ו bare (pointed text reads a bare ו as the consonant /v/), leaves the
glide י bare (pointed text reads it as a mater) and writes no mapiq. The second
script measures that, Rust only, over the same corpus:

| pointing | phonemizes back identically |
|---|---:|
| upstream `vocalize` | 274/327 |
| + hatama on the stressed syllable | 274/327 |
| + rafe / shva / mapiq (`Options::round_trip`) | **322/327** |

Pointing every vowel-less consonant with its shva as well — the other thing
upstream leaves out — measures worse, not better (309/327, and 257/327 in the
variant that also skips the letter before a mater ו), because a shva admits
/e/ as well as nothing. The five that still drift are shva-in-cluster cases
(`קלוין` comes back as `kelovin`) and a word-initial glide before a holam male.
