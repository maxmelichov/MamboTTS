//! Several chunks in one batch against each chunk alone (Metal through wgpu).
//!
//!   cargo run --release --example batch -- <vector_estimator.onnx> <case.bin>... [--flow <ref.bin>...]
//!
//! Case files as in examples/parity.rs (selfcheck/cases.py). Every
//! case runs alone, then all of them in one batch (rows padded to the longest
//! latent and text, `prepare_rows`): one run at step 3 of 8, and the whole
//! 8-step CFG-4 flow. Prints, per case, how far the batched rows are from the
//! rows run alone (max |d|, SNR, identical values) and, with `--flow` (the
//! 8-step flow on ONNX Runtime CPU, one file per case in order; by default the
//! flow_T.bin next to each case), the SNR of both against it.

use bluetts_engine::engine::{map_read, read_buffer, Engine};

struct Case {
    t: usize,
    tt: usize,
    tr: usize,
    noisy: Vec<f32>,
    emb: Vec<f32>,
    style: Vec<f32>,
    lm: Vec<f32>,
    tm: Vec<f32>,
}

fn read_case(path: &str) -> Case {
    let raw = std::fs::read(path).unwrap();
    let h: Vec<u32> = raw[..24].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let f: Vec<f32> = raw[24..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (t, tt, tr) = (h[0] as usize, h[1] as usize, h[2] as usize);
    let mut o = 0;
    let mut take = |n: usize| {
        let s = f[o..o + n].to_vec();
        o += n;
        s
    };
    let noisy = take(144 * t);
    let emb = take(256 * tt);
    let style = take(tr * 256);
    let lm = take(t);
    let tm = take(tt);
    Case { t, tt, tr, noisy, emb, style, lm, tm }
}

fn cmp(got: &[f32], want: &[f32]) -> (f64, f64, usize) {
    let (mut max, mut sig, mut err, mut same) = (0f64, 0f64, 0f64, 0usize);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        max = max.max(d);
        sig += (*w as f64).powi(2);
        err += d * d;
        if g.to_bits() == w.to_bits() {
            same += 1;
        }
    }
    (max, if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() }, same)
}

const STEPS: u32 = 8;
const CFG: f32 = 4.0;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = std::fs::read(&args[1]).expect("model");
    let split = args.iter().position(|a| a == "--flow").unwrap_or(args.len());
    let cases: Vec<Case> = args[2..split].iter().map(|p| read_case(p)).collect();
    // ORT's flows: given after --flow, or flow_T.bin next to each case_T.bin.
    let mut ref_paths: Vec<String> = args.get(split + 1..).unwrap_or(&[]).to_vec();
    if ref_paths.is_empty() {
        ref_paths = args[2..split].iter().map(|p| p.replace("case_", "flow_")).filter(|p| std::path::Path::new(p).exists()).collect();
    }
    let refs: Vec<Vec<f32>> = ref_paths.iter().map(|p| std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()).collect();
    pollster::block_on(async {
        // VE_TILE=i: one matmul tile for every shape (then batched rows should match bit for bit).
        if let Some(i) = std::env::var("VE_TILE").ok().and_then(|v| v.parse::<usize>().ok()) {
            bluetts_engine::engine::TILE_OVERRIDE.store(i, std::sync::atomic::Ordering::Relaxed);
        }
        let (device, queue) = Engine::new_device(false).await.unwrap();
        let mut e = Engine::load(device, queue, &model[..]).unwrap();
        let read_out = |e: &Engine| {
            let (d, q, buf, n) = e.output().unwrap();
            read_buffer(d, q, buf, n)
        };
        // Alone: one run at step 3 (both rows' velocities), then the 8-step flow.
        let mut alone_run = Vec::new();
        let mut alone_flow = Vec::new();
        for c in &cases {
            e.prepare(c.t as u32, c.tt as u32, c.tr as u32, &c.emb, &c.tm, &c.style, &c.lm).unwrap();
            e.set_latent(&c.noisy).unwrap();
            e.run(3, STEPS).unwrap();
            alone_run.push(read_out(&e).await.unwrap());
            e.set_latent(&c.noisy).unwrap();
            for s in 0..STEPS {
                e.run(s, STEPS).unwrap();
                e.blend(CFG).unwrap();
            }
            let (d, st) = e.stage_latent_pairs().unwrap();
            alone_flow.push(map_read(d, st).await.unwrap());
        }
        // Batched: chunks padded to the longest latent (T) and text (TT).
        let k = cases.len();
        let b = 2 * k;
        let t = cases.iter().map(|c| c.t).max().unwrap();
        let tt = cases.iter().map(|c| c.tt).max().unwrap();
        let tr = cases[0].tr;
        let (mut noisy, mut emb, mut style, mut lm, mut tm) = (vec![0f32; k * 144 * t], vec![0f32; k * 256 * tt], vec![0f32; k * tr * 256], vec![0f32; k * t], vec![0f32; k * tt]);
        for (i, c) in cases.iter().enumerate() {
            for ch in 0..144 {
                noisy[(i * 144 + ch) * t..][..c.t].copy_from_slice(&c.noisy[ch * c.t..][..c.t]);
            }
            for ch in 0..256 {
                emb[(i * 256 + ch) * tt..][..c.tt].copy_from_slice(&c.emb[ch * c.tt..][..c.tt]);
            }
            style[i * tr * 256..][..tr * 256].copy_from_slice(&c.style);
            lm[i * t..][..c.t].copy_from_slice(&c.lm);
            tm[i * tt..][..c.tt].copy_from_slice(&c.tm);
        }
        e.prepare_rows(k as u32, t as u32, tt as u32, tr as u32, &emb, &tm, &style, &lm).unwrap();
        e.set_latent(&noisy).unwrap();
        e.run(3, STEPS).unwrap();
        let batch_run = read_out(&e).await.unwrap();
        e.set_latent(&noisy).unwrap();
        for s in 0..STEPS {
            e.run(s, STEPS).unwrap();
            e.blend(CFG).unwrap();
        }
        let (d, st) = e.stage_latent_pairs().unwrap();
        let batch_flow = map_read(d, st).await.unwrap();
        println!("batch: {k} chunks, {b} rows x {t} frames, {tt} text columns");
        for (i, c) in cases.iter().enumerate() {
            // one run: both rows of the pair, valid frames only
            let (mut got, mut want) = (Vec::new(), Vec::new());
            for r in 0..2 {
                for ch in 0..144 {
                    got.extend_from_slice(&batch_run[((2 * i + r) * 144 + ch) * t..][..c.t]);
                    want.extend_from_slice(&alone_run[i][(r * 144 + ch) * c.t..][..c.t]);
                }
            }
            let (m1, s1, same1) = cmp(&got, &want);
            let mut gf = Vec::new();
            for ch in 0..144 {
                gf.extend_from_slice(&batch_flow[(i * 144 + ch) * t..][..c.t]);
            }
            let (m2, s2, same2) = cmp(&gf, &alone_flow[i]);
            print!(
                "  t={} tt={}: run max|d| {m1:.3e} {s1:.1} dB identical {}/{} | flow max|d| {m2:.3e} {s2:.1} dB identical {}/{}",
                c.t,
                c.tt,
                same1,
                got.len(),
                same2,
                gf.len()
            );
            if let Some(r) = refs.get(i) {
                let (_, sa, _) = cmp(&alone_flow[i], r);
                let (_, sb, _) = cmp(&gf, r);
                print!(" | vs ORT CPU flow: alone {sa:.1} dB, batched {sb:.1} dB");
            }
            println!();
        }
    });
}
