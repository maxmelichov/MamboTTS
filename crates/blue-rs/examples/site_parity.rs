//! Dumps the text front end's work for a set of cases, for comparison with
//! the BlueTTS site's own `js/bluetts-text.js` run under node.
//!
//! Usage: `cargo run -p blue-rs --example site_parity -- <model_dir> <cases.json> <out.json>`
//!
//! `cases.json` is `[{"lang": "he", "text": "…"}, …]`. For each case the
//! output has the prepared text, the phonemized G2P units, the chunks, each
//! chunk's ids, and every Hebrew/Russian G2P call (the site
//! runs its own copies of those models, so the comparison feeds it these).

use std::{fs, path::PathBuf};

use anyhow::{Context, Result};
use blue_rs::{
    BlueTts, PrecisionSetting,
    phonemize::{Language, Phonemizer},
};
use serde_json::{Value, json};

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let [model_dir, cases, out] = match args.get(1..4) {
        Some([a, b, c]) => [a.clone(), b.clone(), c.clone()],
        _ => anyhow::bail!("usage: site_parity <model_dir> <cases.json> <out.json>"),
    };
    let model_dir = PathBuf::from(model_dir);
    // The site runs the float32 graphs: bit parity needs them here too.
    let tts = BlueTts::from_dir_with_precision(&model_dir, PrecisionSetting::Fp32).context("load model")?;
    let renikud = ["renikud-plus-fp32.onnx", "renikud-plus.onnx"]
        .map(|name| model_dir.join(name))
        .into_iter()
        .find(|path| path.is_file())
        .context("no RenikudPlus in the bundle")?;
    let mut phonemizer = Phonemizer::open(
        Some(renikud),
        Some(model_dir.join("polyg2p")),
        Language::English,
    )?;
    phonemizer.set_native_trace(true);
    let cases: Vec<Value> = serde_json::from_str(&fs::read_to_string(&cases)?)?;
    let mut results = Vec::new();
    for case in cases {
        let lang = case["lang"].as_str().context("lang")?;
        let text = case["text"].as_str().context("text")?;
        let plan = tts.plan_text(&mut phonemizer, text, lang)?;
        let ids = plan
            .chunks
            .iter()
            .map(|chunk| tts.token_ids(chunk, lang))
            .collect::<Result<Vec<_>>>()?;
        let native: Vec<Value> = phonemizer
            .take_native_trace()
            .into_iter()
            .map(|(lang, input, output)| json!({"lang": lang, "input": input, "output": output}))
            .collect();
        results.push(json!({
            "lang": lang,
            "text": text,
            "prepared": plan.prepared,
            "phonemes": plan.phonemes,
            "chunks": plan.chunks,
            "ids": ids,
            "native": native,
        }));
    }
    fs::write(&out, serde_json::to_string_pretty(&results)?)?;
    println!("{} cases -> {out}", results.len());
    Ok(())
}
