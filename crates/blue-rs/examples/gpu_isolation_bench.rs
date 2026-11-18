//! The cost of running the GPU flow engine in a worker process rather than
//! in-process: the same about-a-minute English text through both, rounds
//! interleaved so machine load hits both alike.
//!
//!   cargo run -p blue-rs --release --example gpu_isolation_bench -- <model_dir> [rounds]
//!
//! Prints each run's wall time and audio length, then min and median per mode.

use std::{path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    gpu_flow::{GpuOptions, Isolation},
    gpu_verdict::VerdictFile,
    phonemize::{Language, Phonemizer},
    ve_worker::WorkerCommand,
};

const TEXT: &str = "The lighthouse keeper climbed the narrow stairs every evening, counting each step as his father had taught him. \
At the top he trimmed the wick, polished the great lens, and watched the ships slide past the rocks in the fading light. \
Some nights the fog rolled in so thick that he could not see his own hand, and then he rang the bell until morning. \
Sailors in the harbor town said they owed him their lives, though few of them had ever met him in person. \
He did not mind the solitude; he had his books, his radio, and a cat named Captain who slept beside the stove. \
In the winter the storms came one after another, and the waves broke over the rocks so high that spray reached the gallery windows. \
He kept a logbook of every ship that passed, the wind, the weather and the state of the sea, in a neat hand that never changed over forty years. \
When the coast guard finally replaced the lamp with an automatic beacon, he packed his things and walked down the stairs for the last time. \
He still visits on clear evenings, and the new light blinks at him as if it remembers.";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().context("usage: gpu_isolation_bench <model_dir> [rounds]")?);
    let rounds: usize = args.next().map(|r| r.parse()).transpose()?.unwrap_or(5);
    let worker = std::env::var_os("MAMBOTTS_GPU_WORKER")
        .map(WorkerCommand::new)
        .or_else(|| {
            let exe = std::env::current_exe().ok()?;
            // target/release/examples/<this> -> target/release/bluev3-gpu-worker
            let w = exe.parent()?.parent()?.join(format!("bluev3-gpu-worker{}", std::env::consts::EXE_SUFFIX));
            w.is_file().then(|| WorkerCommand::new(w))
        })
        .context("build the worker first: cargo build -p blue-rs --release --bins (or set MAMBOTTS_GPU_WORKER)")?;
    let open = |isolation| -> Result<BlueTts> {
        let mut opts = GpuOptions::from_env();
        opts.wanted = true;
        opts.isolation = isolation;
        opts.worker = Some(worker.clone());
        opts.verdicts = VerdictFile::disabled();
        let mut tts = BlueTts::from_dir_with(&dir, opts)?;
        tts.wait_for_gpu();
        anyhow::ensure!(tts.gpu_isolation() == Some(isolation), "the GPU engine did not open {isolation:?}");
        Ok(tts)
    };
    let mut engines = [(Isolation::InProcess, open(Isolation::InProcess)?), (Isolation::Worker, open(Isolation::Worker)?)];
    let style = VoiceStyle::from_json(dir.join("voices/adam.json"))?;
    let mut phonemizer = Phonemizer::open(None::<&std::path::Path>, Some(dir.join("polyg2p")), Language::English)?;
    let opts = || SynthesisOptions { lang: "en".into(), ..SynthesisOptions::default() };

    // Warm both (first-run allocations, page cache).
    let mut outputs = Vec::new();
    for (_, tts) in engines.iter_mut() {
        outputs.push(tts.synthesize_text(&mut phonemizer, TEXT, &style, opts())?);
    }
    let seconds = outputs[0].len() as f64 / engines[0].1.sample_rate() as f64;
    println!(
        "{:.1} s of audio in {} chunks; worker audio identical to in-process: {}",
        seconds,
        engines[0].1.last_chunks().len(),
        outputs[0] == outputs[1]
    );

    let mut times: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for round in 0..rounds {
        // Alternate which goes first.
        let order: [usize; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
        for i in order {
            let t0 = Instant::now();
            let audio = engines[i].1.synthesize_text(&mut phonemizer, TEXT, &style, opts())?;
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            anyhow::ensure!(engines[i].1.gpu_adapter().is_some(), "the GPU engine failed during the bench");
            anyhow::ensure!(audio == outputs[0], "the audio changed between runs");
            times[i].push(ms);
            println!("round {round} {:?}: {ms:.0} ms", engines[i].0);
        }
    }
    for (i, (isolation, _)) in engines.iter().enumerate() {
        let mut t = times[i].clone();
        t.sort_by(f64::total_cmp);
        println!("{isolation:?}: min {:.0} ms, median {:.0} ms ({:.1}x real time at the median)", t[0], t[t.len() / 2], seconds * 1e3 / t[t.len() / 2]);
    }
    let (a, b) = (min(&times[0]), min(&times[1]));
    let (ma, mb) = (median(&times[0]), median(&times[1]));
    println!("worker overhead: {:+.1}% at the min, {:+.1}% at the median", 100.0 * (b / a - 1.0), 100.0 * (mb / ma - 1.0));
    drop(engines);

    // The flow call alone, one long chunk (t=290 frames, 300 tokens, 8 steps),
    // so the boundary's cost per chunk is seen without the CPU graphs around it.
    let (t, n_text, steps) = (290usize, 300usize, 8usize);
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
            })
            .collect()
    };
    let text_emb = rnd(256 * n_text, 0.5);
    let style_ttl = rnd(50 * 256, 0.5);
    let noisy = rnd(144 * t, 2.0);
    let (text_mask, latent_mask) = (vec![1f32; n_text], vec![1f32; t]);
    let flow_opts = |isolation| {
        let mut o = GpuOptions::from_env();
        o.wanted = true;
        o.isolation = isolation;
        o.worker = Some(worker.clone());
        o.verdicts = VerdictFile::disabled();
        o
    };
    let ve = dir.join("vector_estimator.onnx");
    let mut flows = [
        blue_rs::gpu_flow::open(&ve, &flow_opts(Isolation::InProcess)).context("in-process GPU flow")?,
        blue_rs::gpu_flow::open(&ve, &flow_opts(Isolation::Worker)).context("worker GPU flow")?,
    ];
    let mut flow_ms: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut outs: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
    for rep in 0..(4 * rounds).max(10) {
        for i in if rep % 2 == 0 { [0, 1] } else { [1, 0] } {
            let t0 = Instant::now();
            let out = flows[i].flow(&text_emb, &text_mask, &style_ttl, &latent_mask, &noisy, steps, 4.0)?;
            if rep > 0 {
                flow_ms[i].push(t0.elapsed().as_secs_f64() * 1e3);
            }
            outs[i] = out;
        }
    }
    println!(
        "one flow (t={t}, {n_text} tokens, {steps} steps): in-process min {:.1} / median {:.1} ms, worker min {:.1} / median {:.1} ms; identical: {}",
        min(&flow_ms[0]),
        median(&flow_ms[0]),
        min(&flow_ms[1]),
        median(&flow_ms[1]),
        outs[0] == outs[1]
    );
    Ok(())
}

fn min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

fn median(v: &[f64]) -> f64 {
    let mut t = v.to_vec();
    t.sort_by(f64::total_cmp);
    t[t.len() / 2]
}
