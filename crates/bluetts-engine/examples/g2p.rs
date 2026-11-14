//! RenikudPlus natively (Metal through wgpu) against ONNX Runtime CPU's logits.
//!
//!   cargo run --release --example g2p -- <model.onnx> <case.bin>...
//!
//! A case file: u32 n, s, t; u32 ids[n]; f32 ORT logits [n, 33] (consonant
//! 25, vowel 6, stress 2). G2P_TILE=i forces a matmul tile, PROFILE=1 prints
//! a per-kernel GPU profile of each case, G2P_OUT=<dir> keeps the logits.

use bluetts_engine::engine::{gpu_done, now_ms, Engine};
use bluetts_engine::g2p::{G2p, OUT};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t0 = now_ms();
    let model = std::fs::read(&args[1]).expect("model");
    println!("read {:.0} ms", now_ms() - t0);
    pollster::block_on(async {
        if let Some(i) = std::env::var("G2P_TILE").ok().and_then(|v| v.parse::<usize>().ok()) {
            bluetts_engine::g2p::TILE_OVERRIDE.store(i, std::sync::atomic::Ordering::Relaxed);
        }
        // G2P_KZ=a[,b]: K slices for the 1024-wide matmuls, and for the wider ones.
        if let Ok(v) = std::env::var("G2P_KZ") {
            let p: Vec<u32> = v.split(',').map(|x| x.parse().unwrap()).collect();
            bluetts_engine::g2p::KZ_OVERRIDE[0].store(p[0], std::sync::atomic::Ordering::Relaxed);
            bluetts_engine::g2p::KZ_OVERRIDE[1].store(*p.get(1).unwrap_or(&1), std::sync::atomic::Ordering::Relaxed);
        }
        let (device, queue) = Engine::new_device(std::env::var("PROFILE").is_ok()).await.unwrap();
        let mut e = G2p::open(device, queue);
        let t0 = now_ms();
        e.compile();
        let tc = now_ms() - t0;
        e.load_weights(&model[..]).unwrap();
        gpu_done(e.device.clone(), e.queue.clone()).await;
        println!("compile {tc:.0} ms, load {:.0} ms (upload calls {:.0}), {:?}, gpu weights {} B", now_ms() - t0 - tc, e.startup.upload_ms, e.startup, e.weight_bytes());
        for path in &args[2..] {
            let raw = std::fs::read(path).unwrap();
            let h: Vec<u32> = raw[..12].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let n = h[0] as usize;
            let ids: Vec<u32> = raw[12..12 + 4 * n].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let want: Vec<f32> = raw[12 + 4 * n..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            assert_eq!(want.len(), n * OUT);
            let got = e.logits(&ids, h[1], h[2]).await.unwrap();
            if let Ok(dir) = std::env::var("G2P_OUT") {
                let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
                std::fs::write(format!("{dir}/{name}.out"), bytemuck::cast_slice::<f32, u8>(&got)).unwrap();
            }
            let (mut max, mut sig, mut err, mut argmax_diff) = (0f64, 0f64, 0f64, 0usize);
            for (g, w) in got.iter().zip(&want) {
                let d = (*g as f64 - *w as f64).abs();
                max = max.max(if d.is_finite() { d } else { f64::INFINITY });
                sig += (*w as f64).powi(2);
                err += d * d;
            }
            let am = |v: &[f32]| v.iter().enumerate().fold(0, |b, (i, x)| if *x > v[b] { i } else { b });
            for t in 0..n {
                let (g, w) = (&got[t * OUT..(t + 1) * OUT], &want[t * OUT..(t + 1) * OUT]);
                if am(&g[..25]) != am(&w[..25]) || am(&g[25..31]) != am(&w[25..31]) || am(&g[31..]) != am(&w[31..]) {
                    argmax_diff += 1;
                }
            }
            let mut times = Vec::new();
            for _ in 0..20 {
                let t0 = now_ms();
                let st = e.run(&ids, h[1], h[2]).unwrap();
                gpu_done(e.device.clone(), e.queue.clone()).await;
                times.push(now_ms() - t0);
                st.destroy();
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{path}: n={n} max|d| {max:.3e} SNR {:.1} dB, argmax differs at {argmax_diff} tokens; native run median {:.2} ms min {:.2} ms; dispatches {}",
                10.0 * (sig / err).log10(),
                times[times.len() / 2],
                times[0],
                e.dispatches_per_run()
            );
            if std::env::var("PROFILE").is_ok() {
                if let Some(job) = e.profile(&ids) {
                    let mut tot = 0.0;
                    for (k, ms, c) in job.collect().await {
                        tot += ms;
                        println!("  {ms:8.3} ms  x{c:3}  {k}");
                    }
                    println!("  total {tot:.3} ms");
                }
            }
        }
    });
}
