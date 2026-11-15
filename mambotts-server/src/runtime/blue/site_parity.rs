//! The server's synthesis path over `scripts/site-parity/cases.json`, dumped
//! for comparison with bluetts.com's worker run under Node
//! (`scripts/site-parity/run.mjs`, which runs this test first).
//!
//! Each case goes through [`BlueRuntime::synthesize_streaming`] exactly as a
//! `/v1/audio/speech` request with no voice and no speed reaches it (the
//! desktop's default request), then the dump records what the run did: the
//! prepared text, every chunk with its ids, durations and latent frames, and
//! the audio as little-endian f32.
//!
//! ```text
//! MAMBOTTS_BLUE_MODEL_DIR=<bundle> MAMBOTTS_PARITY_OUT=<dir> \
//!   cargo test -p mambotts-server --release site_parity -- --ignored
//! ```
//!
//! `MAMBOTTS_PARITY_RENIKUD` names the RenikudPlus file (default: the
//! bundle's fp32 one, else its int8 one); `MAMBOTTS_PARITY_CASES` another
//! cases file. The flow runs on ONNX Runtime's CPU provider
//! (`MAMBOTTS_VE_BACKEND=ort`), as the site's worker does in Node.

use std::{fs, path::PathBuf};

use mambotts_registry::{HebrewG2pVariant, hebrew_g2p_path};
use serde_json::{Value, json};

use super::BlueRuntime;
use crate::runtime::Runtime;

#[test]
#[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR and MAMBOTTS_PARITY_OUT; run by scripts/site-parity/run.mjs"]
fn site_parity_dump() {
    let model_dir = PathBuf::from(std::env::var("MAMBOTTS_BLUE_MODEL_DIR").expect("MAMBOTTS_BLUE_MODEL_DIR"));
    let out = PathBuf::from(std::env::var("MAMBOTTS_PARITY_OUT").expect("MAMBOTTS_PARITY_OUT"));
    let cases_path = std::env::var("MAMBOTTS_PARITY_CASES").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../scripts/site-parity/cases.json")
    });
    let renikud = std::env::var("MAMBOTTS_PARITY_RENIKUD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hebrew_g2p_path(&model_dir, HebrewG2pVariant::Fp32));
    assert_eq!(
        std::env::var("MAMBOTTS_VE_BACKEND").as_deref(),
        Ok("ort"),
        "set MAMBOTTS_VE_BACKEND=ort: the comparison runs the flow on the CPU"
    );
    fs::create_dir_all(&out).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&fs::read_to_string(&cases_path).unwrap()).unwrap();
    let filter = std::env::var("MAMBOTTS_PARITY_FILTER").unwrap_or_default();

    // What the desktop's load_model sends: speakers 0 and 0.
    let mut runtime = BlueRuntime::load(model_dir, renikud.clone(), 0, 0).unwrap();
    let mut results = Vec::new();
    for case in &cases {
        let id = case["id"].as_str().unwrap();
        if !id.contains(&filter) {
            continue;
        }
        let lang = case["lang"].as_str().unwrap();
        let text = case["text"].as_str().unwrap();
        let plan = runtime.tts.plan_text(&mut runtime.phonemizer, text, lang).unwrap();
        // The request: no voice, no speed, no niqqud flag (the defaults).
        let mut streamed = Vec::new();
        let audio = runtime
            .synthesize_streaming(text, None, lang, 0.0, &mut |chunk, _| {
                streamed.push(chunk.len());
                Ok(())
            })
            .unwrap();
        let chunks: Vec<Value> = runtime
            .tts
            .last_chunks()
            .iter()
            .map(|chunk| {
                json!({
                    "phonemes": chunk.phonemes,
                    "ids": runtime.tts.token_ids(&chunk.phonemes, lang).unwrap(),
                    "raw_duration": chunk.raw_duration,
                    "duration": chunk.duration,
                    "frames": chunk.frames,
                    "mask_frames": chunk.mask_frames,
                    "samples": chunk.samples,
                })
            })
            .collect();
        assert_eq!(
            chunks.iter().map(|c| c["phonemes"].as_str().unwrap().to_owned()).collect::<Vec<_>>(),
            plan.chunks,
            "{id}: the synthesis spoke the plan's chunks"
        );
        let file = format!("{id}.f32");
        let bytes: Vec<u8> = audio.iter().flat_map(|sample| sample.to_le_bytes()).collect();
        fs::write(out.join(&file), bytes).unwrap();
        println!("{id}: {} chunks, {} samples", chunks.len(), audio.len());
        results.push(json!({
            "id": id,
            "lang": lang,
            "text": text,
            "voice": mambotts_registry::DEFAULT_BLUE_VOICE,
            "speed": super::effective_speed(0.0, lang),
            "prepared": plan.prepared,
            "units": plan.phonemes,
            "chunks": chunks,
            "streamed": streamed,
            "samples": audio.len(),
            "audio": file,
        }));
    }
    let dump = json!({
        "renikud": renikud.file_name().and_then(|name| name.to_str()),
        "cases": results,
    });
    fs::write(out.join("mambo.json"), serde_json::to_string_pretty(&dump).unwrap()).unwrap();
}
