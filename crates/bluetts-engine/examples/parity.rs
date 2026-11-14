//! Native check against ONNX Runtime's CPU output (Metal through wgpu).
//!
//!   cargo run --release --example parity -- <vector_estimator.onnx> <case_T.bin>...
//!
//! Case files come from selfcheck/cases.py: u32 [t, tt, tr, step, total, cfg
//! bits], then f32 noisy [144,t], text_emb [256,tt], style [tr,256],
//! latent_mask [t], text_mask [tt], and ORT's output (the next latent, one
//! step with CFG) [144,t]. A `flow_T.bin` next to a case (cases.py writes it)
//! is ORT's 8-step flow from the same noise: compared too.

use bluetts_engine::engine::{gpu_done, now_ms, read_buffer, Engine};

fn cmp(got: &[f32], want: &[f32]) -> (f64, f64) {
    let (mut max, mut sig, mut err) = (0f64, 0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        max = max.max(d);
        sig += (*w as f64).powi(2);
        err += d * d;
    }
    (max, if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model = std::fs::read(&args[1]).expect("model");
    pollster::block_on(async {
        if let Some(i) = std::env::var("VE_TILE").ok().and_then(|v| v.parse::<usize>().ok()) {
            bluetts_engine::engine::TILE_OVERRIDE.store(i, std::sync::atomic::Ordering::Relaxed);
        }
        let (device, queue) = Engine::new_device(std::env::var("PROFILE").is_ok()).await.unwrap();
        let mut e = Engine::load(device, queue, &model[..]).unwrap();
        println!("startup {:?}", e.startup);
        for path in &args[2..] {
            let raw = std::fs::read(path).unwrap();
            let h: Vec<u32> = raw[..24].chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let f: Vec<f32> = raw[24..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let (t, tt, tr, step, total, cfg) = (h[0] as usize, h[1] as usize, h[2] as usize, h[3], h[4], f32::from_bits(h[5]));
            let mut o = 0;
            let mut take = |n: usize| {
                let s = &f[o..o + n];
                o += n;
                s.to_vec()
            };
            let noisy = take(144 * t);
            let emb = take(256 * tt);
            let style = take(tr * 256);
            let lm = take(t);
            let tm = take(tt);
            let want = take(144 * t);
            e.prepare(t as u32, tt as u32, tr as u32, &emb, &tm, &style, &lm).unwrap();
            e.set_latent(&noisy).unwrap();
            e.run(step, total).unwrap();
            e.blend(cfg).unwrap();
            let (d, q, buf, _) = e.latent().unwrap();
            let got = read_buffer(d, q, buf, 144 * t).await.unwrap();
            // VE_OUT=<dir>: keep the output (f32 LE) to compare two builds bit for bit.
            if let Ok(dir) = std::env::var("VE_OUT") {
                let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
                std::fs::write(format!("{dir}/{name}.out"), bytemuck::cast_slice::<f32, u8>(&got)).unwrap();
            }
            let (max, snr) = cmp(&got, &want);
            // The 8-step flow, when ORT's is next to the case.
            let flow_path = path.replace("case_", "flow_");
            let flow = if flow_path != *path && std::path::Path::new(&flow_path).exists() {
                let fref: Vec<f32> = std::fs::read(&flow_path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                e.set_latent(&noisy).unwrap();
                for s in 0..8 {
                    e.run(s, 8).unwrap();
                    e.blend(cfg).unwrap();
                }
                let (d, q, buf, _) = e.latent().unwrap();
                let fl = read_buffer(d, q, buf, 144 * t).await.unwrap();
                let (fm, fs) = cmp(&fl, &fref);
                format!("  flow max|d|={fm:.3e} snr={fs:.1} dB")
            } else {
                String::new()
            };
            // Native timing (Metal, not the browser): one run + blend, waiting for the GPU each time.
            let mut times = Vec::new();
            for _ in 0..20 {
                let t0 = now_ms();
                e.run(step, total).unwrap();
                e.blend(cfg).unwrap();
                gpu_done(e.device.clone(), e.queue.clone()).await;
                times.push(now_ms() - t0);
            }
            let t0 = now_ms();
            for i in 0..48 {
                e.run(i % 8, 8).unwrap();
                e.blend(cfg).unwrap();
            }
            gpu_done(e.device.clone(), e.queue.clone()).await;
            let b2b = (now_ms() - t0) / 48.0;
            if std::env::var("PROFILE").is_ok() {
                let mut prof: Vec<(String, f64, usize)> = Vec::new();
                for _ in 0..3 {
                    for (k, ms, n) in e.profile(step, total).unwrap().collect().await {
                        match prof.iter_mut().find(|x| x.0 == k) {
                            Some(x) => x.1 += ms / 3.0,
                            None => prof.push((k, ms / 3.0, n)),
                        }
                    }
                }
                for (k, ms, n) in &prof {
                    println!("  {ms:8.3} ms  x{n:3}  {k}");
                }
                println!("  {:8.3} ms  GPU total", prof.iter().map(|x| x.1).sum::<f64>());
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "{path}: t={t} tt={tt} run max|d|={max:.3e} snr={snr:.1} dB{flow}  dispatches={}  native run median {:.2} ms min {:.2}, back-to-back {b2b:.2}",
                e.dispatches_per_run(),
                times[times.len() / 2],
                times[0]
            );
        }
    });
}
