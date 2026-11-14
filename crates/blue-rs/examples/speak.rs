//! Speaks text through the whole BlueV3 pipeline: front end, model, WAV.
//!
//! Usage: `cargo run -p blue-rs --release --example speak -- <model_dir> <voice> <lang> <text> <out.wav>`
//!
//! `model_dir` is an installed bundle: the four graphs, `vocab.json`,
//! `tts.json`, `voices/<voice>.json`, `polyg2p/` and `renikud-plus.onnx`.

use std::{env, fs::File, io::Write, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let [model_dir, voice, lang, text, out] = match args.get(1..6) {
        Some([a, b, c, d, e]) => [a.clone(), b.clone(), c.clone(), d.clone(), e.clone()],
        _ => anyhow::bail!("usage: speak <model_dir> <voice> <lang> <text> <out.wav>"),
    };
    let model_dir = PathBuf::from(model_dir);
    let language = Language::try_from(lang.as_str())?;

    let started = Instant::now();
    let mut tts = BlueTts::from_dir(&model_dir).context("load model")?;
    let renikud = model_dir.join("renikud-plus.onnx");
    let mut phonemizer = Phonemizer::open(
        renikud.is_file().then_some(&renikud),
        Some(model_dir.join("polyg2p")),
        language,
    )
    .context("load front end")?;
    let style = VoiceStyle::from_json(model_dir.join("voices").join(format!("{voice}.json")))
        .context("load voice")?;
    println!("loaded in {:.2} s", started.elapsed().as_secs_f32());

    println!("phonemes: {}", phonemizer.g2p(&text, language)?);
    let started = Instant::now();
    let mut chunks = 0;
    let audio = tts.synthesize_text_streaming(
        &mut phonemizer,
        &text,
        &style,
        SynthesisOptions {
            lang: language.code().to_owned(),
            ..SynthesisOptions::default()
        },
        |_| {
            chunks += 1;
            Ok(())
        },
    )?;
    let rate = tts.sample_rate();
    let seconds = audio.len() as f32 / rate as f32;
    let elapsed = started.elapsed().as_secs_f32();
    let peak = audio.iter().fold(0f32, |acc, s| acc.max(s.abs()));
    println!(
        "{seconds:.2} s of audio in {chunks} chunks, peak {peak:.3}, synthesized in {elapsed:.2} s (RTF {:.3})",
        elapsed / seconds.max(1e-6)
    );
    anyhow::ensure!(peak > 0.01, "audio is silent");

    let mut file = File::create(&out)?;
    let data_len = (audio.len() * 2) as u32;
    file.write_all(b"RIFF")?;
    file.write_all(&(36 + data_len).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&rate.to_le_bytes())?;
    file.write_all(&(rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_len.to_le_bytes())?;
    for sample in &audio {
        file.write_all(&((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())?;
    }
    println!("wrote {out}");
    Ok(())
}
