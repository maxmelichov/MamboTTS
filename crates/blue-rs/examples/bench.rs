//! Times the whole pipeline on one text, after a warm-up run.
//!
//! Usage: `cargo run -p blue-rs --release --example bench -- <model_dir> <voice> <lang> <text_file> [runs]`
//!
//! `MAMBOTTS_VE_BACKEND=ort|gpu` picks the flow backend and
//! `MAMBOTTS_ORT_THREADS=<n>` the ONNX Runtime thread count.

use std::{env, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let [model_dir, voice, lang, text_file] = match args.get(1..5) {
        Some([a, b, c, d]) => [a.clone(), b.clone(), c.clone(), d.clone()],
        _ => anyhow::bail!("usage: bench <model_dir> <voice> <lang> <text_file> [runs]"),
    };
    let runs: usize = args.get(5).and_then(|n| n.parse().ok()).unwrap_or(3);
    let model_dir = PathBuf::from(model_dir);
    let language = Language::try_from(lang.as_str())?;
    let text = std::fs::read_to_string(&text_file).context("read text")?;

    let mut tts = BlueTts::from_dir(&model_dir).context("load model")?;
    tts.wait_for_gpu();
    let renikud = ["renikud-plus-fp32.onnx", "renikud-plus.onnx"]
        .map(|name| model_dir.join(name))
        .into_iter()
        .find(|path| path.is_file());
    let mut phonemizer =
        Phonemizer::open(renikud.as_ref(), Some(model_dir.join("polyg2p")), language).context("load front end")?;
    let style = VoiceStyle::from_json(model_dir.join("voices").join(format!("{voice}.json"))).context("load voice")?;
    let options = || SynthesisOptions { lang: language.code().to_owned(), ..SynthesisOptions::default() };

    // Warm-up: the first run pays for graph and kernel set-up.
    tts.synthesize_text(&mut phonemizer, text.split('.').next().unwrap_or(&text), &style, options())?;

    let mut best = f32::MAX;
    let mut seconds = 0.0;
    let mut first = 0.0;
    for _ in 0..runs {
        let started = Instant::now();
        let mut first_chunk = None;
        let audio = tts.synthesize_text_streaming(&mut phonemizer, &text, &style, options(), |_| {
            first_chunk.get_or_insert_with(|| started.elapsed().as_secs_f32());
            Ok(())
        })?;
        let elapsed = started.elapsed().as_secs_f32();
        seconds = audio.len() as f32 / tts.sample_rate() as f32;
        if elapsed < best {
            best = elapsed;
            first = first_chunk.unwrap_or(elapsed);
        }
    }
    println!(
        "backend {} | {seconds:.1} s of audio in {best:.2} s = {:.1}x real time | first chunk after {first:.2} s",
        tts.gpu_adapter().unwrap_or("onnxruntime"),
        seconds / best
    );
    Ok(())
}
