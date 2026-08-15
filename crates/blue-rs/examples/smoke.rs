//! Loads a model directory and synthesizes one phrase from IPA.
//!
//! Usage: `cargo run -p blue-rs --example smoke -- <model_dir> <voice.json> <ipa> <out.wav>`

use std::{env, fs::File, io::Write, path::PathBuf};

use anyhow::{Context, Result};
use blue_rs::{BlueTts, SynthesisOptions, VoiceStyle};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let [model_dir, voice, ipa, out] = match args.get(1..5) {
        Some([a, b, c, d]) => [a.clone(), b.clone(), c.clone(), d.clone()],
        _ => anyhow::bail!("usage: smoke <model_dir> <voice.json> <ipa> <out.wav>"),
    };

    let mut tts = BlueTts::from_dir(PathBuf::from(&model_dir)).context("load model")?;
    let style = VoiceStyle::from_json(PathBuf::from(&voice)).context("load voice")?;
    let rate = tts.sample_rate();
    println!("sample_rate = {rate}");

    let audio = tts.create(&ipa, &style, SynthesisOptions::default())?;
    let peak = audio.iter().fold(0f32, |acc, s| acc.max(s.abs()));
    let rms = (audio.iter().map(|s| s * s).sum::<f32>() / audio.len().max(1) as f32).sqrt();
    println!(
        "samples = {} ({:.2}s), peak = {peak:.3}, rms = {rms:.4}",
        audio.len(),
        audio.len() as f32 / rate as f32
    );
    anyhow::ensure!(audio.len() > 1000, "audio is implausibly short");
    anyhow::ensure!(peak > 0.01, "audio is silent");

    write_wav(&out, &audio, rate)?;
    println!("wrote {out}");
    Ok(())
}

fn write_wav(path: &str, audio: &[f32], rate: u32) -> Result<()> {
    let mut file = File::create(path)?;
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
    for sample in audio {
        file.write_all(&((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16).to_le_bytes())?;
    }
    Ok(())
}
