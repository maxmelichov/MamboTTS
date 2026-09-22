//! Batch driver for the upstream-parity harness.
//!
//! Reads one JSON request per line and writes one JSON result per line, so the
//! Python parity script can run the same corpus through this crate and through
//! the upstream `renikud_onnx` package and diff the two.
//!
//! ```console
//! MAMBOTTS_RENIKUD_PATH=/path/to/renikud-plus.onnx \
//!   cargo run -p renikud-plus-rs --example parity -- requests.jsonl results.jsonl
//! ```
//!
//! Request fields (all optional but `op`): `op` (`phonemize`, `vocalize`,
//! `numbers`, `align`, `lexicon`), `text`, `niqqud` (`strip`/`use`),
//! `exact_map`, `speaker`, `target_speaker`, `number_norm`, `with_stress`,
//! `read_hatama`, `use_lexicon`, `surface`, `ipa`, `entries`.

use std::io::{BufRead, BufWriter, Write};

use anyhow::Context;
use renikud_plus_rs::lexicon::{ForceLexicon, OnInvalidEntry, align_word};
use renikud_plus_rs::text::HebrewLeak;
use renikud_plus_rs::{G2P, G2PConfig, NiqqudMode, NumberNorm, Options, numbers};
use serde_json::{Value, json};

fn options(request: &Value) -> Options {
    Options {
        speaker: request["speaker"].as_u64().unwrap_or(0) as u8,
        target_speaker: request["target_speaker"].as_u64().unwrap_or(0) as u8,
        exact_map: request["exact_map"].as_bool(),
        on_hebrew_leak: HebrewLeak::Ignore,
        number_norm: match request["number_norm"].as_str() {
            Some("off") => NumberNorm::Off,
            _ => NumberNorm::Auto,
        },
        niqqud: match request["niqqud"].as_str() {
            Some("use") => Some(NiqqudMode::Use),
            Some("strip") => Some(NiqqudMode::Strip),
            _ => None,
        },
        with_stress: request["with_stress"].as_bool().unwrap_or(false),
    }
}

fn entries(request: &Value) -> Vec<(String, String)> {
    request["entries"]
        .as_object()
        .map(|map| {
            map.iter()
                .map(|(surface, ipa)| {
                    (surface.clone(), ipa.as_str().unwrap_or_default().to_owned())
                })
                .collect()
        })
        .unwrap_or_default()
}

fn run(g2p: &mut G2P, request: &Value) -> anyhow::Result<Value> {
    let text = request["text"].as_str().unwrap_or_default();
    match request["op"].as_str().unwrap_or("phonemize") {
        "numbers" => Ok(json!(numbers::normalize_numbers(text))),
        "align" => {
            let surface = request["surface"].as_str().unwrap_or_default();
            let ipa = request["ipa"].as_str().unwrap_or_default();
            let cells = G2P::align_cells(request["cells"].as_str().unwrap_or(""))?;
            Ok(json!(align_word(surface, ipa, &cells).is_some()))
        }
        "lexicon" => {
            // Which entries a lexicon accepts, and what it does to a decode.
            let cells = G2P::align_cells(request["cells"].as_str().unwrap_or(""))?;
            let lexicon = ForceLexicon::new(entries(request), &cells, OnInvalidEntry::Report)?;
            let mut accepted: Vec<String> = entries(request)
                .into_iter()
                .filter(|(surface, _)| lexicon.lookup(surface).is_some())
                .map(|(surface, _)| surface)
                .collect();
            accepted.sort();
            let out = if text.is_empty() {
                Value::Null
            } else {
                g2p.set_lexicon(Some(lexicon));
                let ipa = g2p.phonemize_with(text, &options(request))?;
                g2p.take_lexicon();
                json!(ipa)
            };
            Ok(json!({ "accepted": accepted, "ipa": out }))
        }
        "vocalize" => Ok(json!(g2p.vocalize_with(text, &options(request))?)),
        "phonemize" => Ok(json!(g2p.phonemize_with(text, &options(request))?)),
        other => anyhow::bail!("unknown op `{other}`"),
    }
}

fn main() -> anyhow::Result<()> {
    let model = std::env::var("MAMBOTTS_RENIKUD_PATH")
        .context("set MAMBOTTS_RENIKUD_PATH to renikud-plus.onnx")?;
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .context("usage: parity <requests.jsonl> <results.jsonl>")?;
    let output = args
        .next()
        .context("usage: parity <requests.jsonl> <results.jsonl>")?;

    let mut g2p = G2P::with_config(
        &model,
        G2PConfig {
            // Every knob the harness varies is a per-call option, so the
            // session itself stays on the upstream defaults.
            ..G2PConfig::default()
        },
    )?;

    let requests = std::fs::File::open(&input).with_context(|| format!("open {input}"))?;
    let mut results =
        BufWriter::new(std::fs::File::create(&output).with_context(|| format!("create {output}"))?);
    for line in std::io::BufReader::new(requests).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request: Value = serde_json::from_str(&line)?;
        let mut config = g2p.config().clone();
        config.read_hatama = request["read_hatama"].as_bool().unwrap_or(true);
        g2p.set_config(config);
        let result = match run(&mut g2p, &request) {
            Ok(out) => json!({ "i": request["i"], "out": out }),
            Err(error) => json!({ "i": request["i"], "error": error.to_string() }),
        };
        writeln!(results, "{result}")?;
    }
    results.flush()?;
    Ok(())
}
