//! The vector estimator on the GPU: the BlueTTS site's Rust engine
//! (`bluetts-engine`, hand-written WGSL on wgpu) driven synchronously.
//!
//! One [`GpuVectorEstimator::flow`] call is what running the ONNX Runtime
//! `vector_estimator.onnx` graph `steps` times does: per step the conditional
//! and unconditional rows (CFG is inside, as in the graph), then
//! `x + (v_u + g (v_c - v_u)) / steps`, masked. Everything stays on the GPU
//! between steps and the final latent is read back once.
//!
//! [`GpuVectorEstimator::open`] returns `Ok(None)` when there is no usable GPU
//! adapter (none at all, or a software one), so callers keep ONNX Runtime's
//! vector estimator. Any error after that (weights, the start-up self-check, a
//! device error or loss during a flow) is an `Err`, and the caller should fall
//! back the same way.

use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, anyhow, bail};
use bluetts_engine::{
    engine::{self, Engine},
    model::{LAT, MAX_LEN, TXT},
    wgpu,
};

/// The site's self-check limit (`RUST_SELFCHECK_MAX_ABS`): every value of the
/// golden run within this of ONNX Runtime CPU's output, and finite.
const SELFCHECK_MAX_ABS: f64 = 1e-3;

/// Style-attention channels (`style_ttl` is [1, rows, 256]).
const STYLE_DIM: usize = 256;

pub struct GpuVectorEstimator {
    engine: Engine,
    adapter: String,
    /// The first uncaptured device error or loss since the last flow.
    error: Arc<Mutex<Option<String>>>,
}

impl GpuVectorEstimator {
    /// Opens a GPU device (Metal on macOS, DX12 or else Vulkan on Windows,
    /// Vulkan on Linux), uploads the weights of `vector_estimator.onnx`,
    /// compiles the kernels and runs the self-check. Each backend is tried on
    /// its own and the first that passes the self-check is kept, so a driver
    /// that computes wrongly on one backend does not cost the GPU when another
    /// works. `Ok(None)`: no hardware adapter on any of them.
    pub fn open(vector_estimator_onnx: impl AsRef<Path>) -> Result<Option<Self>> {
        let path = vector_estimator_onnx.as_ref();
        let mut failures = Vec::new();
        for backends in Engine::native_backend_candidates() {
            match Self::open_on(path, backends) {
                Ok(Some(me)) => return Ok(Some(me)),
                Ok(None) => {}
                Err(err) => failures.push(format!("{err:#}")),
            }
        }
        if failures.is_empty() {
            Ok(None)
        } else {
            bail!("{}", failures.join("; "))
        }
    }

    fn open_on(path: &Path, backends: wgpu::Backends) -> Result<Option<Self>> {
        let (device, queue) = match pollster::block_on(Engine::new_device_on(backends, false)) {
            Ok(dq) => dq,
            Err(e) if e.starts_with("no WebGPU adapter") => return Ok(None),
            Err(e) => bail!("GPU vector estimator: {e}"),
        };
        let info = device.adapter_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            // A software rasteriser (llvmpipe, WARP, SwiftShader) is slower than ONNX Runtime's CPU path.
            device.destroy();
            return Ok(None);
        }
        let adapter = format!("{} ({:?})", info.name, info.backend);

        let error = Arc::new(Mutex::new(None));
        let sink = error.clone();
        // Without a handler wgpu panics on a validation error; keep the first instead.
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            let mut s = sink.lock().unwrap_or_else(|p| p.into_inner());
            s.get_or_insert_with(|| e.to_string());
        }));
        let sink = error.clone();
        device.set_device_lost_callback(move |reason, msg| {
            let mut s = sink.lock().unwrap_or_else(|p| p.into_inner());
            s.get_or_insert_with(|| format!("GPU device lost ({reason:?}): {msg}"));
        });

        let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let engine = Engine::load(device, queue, &bytes[..])
            .map_err(|e| anyhow!("GPU vector estimator: load {}: {e}", path.display()))?;
        drop(bytes);
        let mut me = Self { engine, adapter, error };
        me.self_check()?;
        Ok(Some(me))
    }

    /// The adapter in use, e.g. "Apple M3 Pro (Metal)".
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// One chunk's whole flow. Inputs as the ONNX graph takes them (batch 1,
    /// row-major): `text_emb` [1, 256, n_text], `text_mask` [1, 1, n_text],
    /// `style_ttl` [1, rows, 256], `latent_mask` [1, 1, t], `noisy_latent`
    /// [1, 144, t]. Returns the final normalised latent, [144, t] row-major
    /// (the vocoder's input as it is).
    #[allow(clippy::too_many_arguments)]
    pub fn flow(
        &mut self,
        text_emb: &[f32],
        text_mask: &[f32],
        style_ttl: &[f32],
        latent_mask: &[f32],
        noisy_latent: &[f32],
        steps: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        let shape = ChunkShape::of(text_emb, text_mask, style_ttl, latent_mask, noisy_latent)?;
        if steps == 0 {
            return Ok(noisy_latent.to_vec());
        }
        self.take_error()?;
        let e = &mut self.engine;
        let gpu = |r: Result<(), String>| r.map_err(|e| anyhow!("GPU vector estimator: {e}"));
        gpu(e.prepare(shape.t, shape.n_text, shape.style_rows, text_emb, text_mask, style_ttl, latent_mask))?;
        gpu(e.set_latent(noisy_latent))?;
        for step in 0..steps as u32 {
            gpu(e.run(step, steps as u32))?;
            gpu(e.blend(cfg_scale))?;
        }
        let (device, queue, buf, offset, n) = e.latent_row(0).context("GPU vector estimator: no latent")?;
        let staging = engine::stage(&device, &queue, &buf, offset, n);
        let out = pollster::block_on(engine::map_read(device, staging)).map_err(|e| anyhow!("GPU vector estimator: {e}"))?;
        self.take_error()?;
        if out.iter().any(|v| !v.is_finite()) {
            bail!("GPU vector estimator: the latent is not finite");
        }
        Ok(out)
    }

    /// The golden run the site checks at start-up (two fixed inputs against
    /// ONNX Runtime CPU's output, compiled into the engine).
    fn self_check(&mut self) -> Result<()> {
        let (a, b) = self.engine.selfcheck_start().map_err(|e| anyhow!("GPU self-check: {e}"))?;
        let d = self.engine.device.clone();
        let lat0 = pollster::block_on(engine::map_read(d.clone(), a)).map_err(|e| anyhow!("GPU self-check: {e}"))?;
        let lat1 = pollster::block_on(engine::map_read(d, b)).map_err(|e| anyhow!("GPU self-check: {e}"))?;
        self.take_error()?;
        let r = engine::selfcheck_compare(&lat0, &lat1);
        if !r.finite || r.max_abs.is_nan() || r.max_abs > SELFCHECK_MAX_ABS {
            bail!(
                "GPU self-check failed on {}: max |d| {:.3e}, finite {}",
                self.adapter,
                r.max_abs,
                r.finite
            );
        }
        Ok(())
    }

    fn take_error(&self) -> Result<()> {
        let mut s = self.error.lock().unwrap_or_else(|p| p.into_inner());
        match s.take() {
            Some(e) => bail!("GPU vector estimator: {e}"),
            None => Ok(()),
        }
    }
}

impl Drop for GpuVectorEstimator {
    fn drop(&mut self) {
        self.engine.destroy();
    }
}

/// A chunk's sizes, read off its inputs and checked against each other.
#[derive(Debug, PartialEq, Eq)]
struct ChunkShape {
    t: u32,
    n_text: u32,
    style_rows: u32,
}

impl ChunkShape {
    fn of(text_emb: &[f32], text_mask: &[f32], style_ttl: &[f32], latent_mask: &[f32], noisy: &[f32]) -> Result<Self> {
        let (t, n_text) = (latent_mask.len(), text_mask.len());
        if t == 0 || n_text == 0 {
            bail!("GPU vector estimator: empty chunk ({t} frames, {n_text} tokens)");
        }
        if t > MAX_LEN as usize || n_text > MAX_LEN as usize {
            bail!("GPU vector estimator: {t} frames / {n_text} tokens is past the model's {MAX_LEN}");
        }
        if text_emb.len() != TXT * n_text {
            bail!("GPU vector estimator: text_emb has {} values, expected {TXT} x {n_text}", text_emb.len());
        }
        if noisy.len() != LAT * t {
            bail!("GPU vector estimator: noisy_latent has {} values, expected {LAT} x {t}", noisy.len());
        }
        if style_ttl.is_empty() || !style_ttl.len().is_multiple_of(STYLE_DIM) {
            bail!("GPU vector estimator: style_ttl has {} values, not rows of {STYLE_DIM}", style_ttl.len());
        }
        Ok(Self {
            t: t as u32,
            n_text: n_text as u32,
            style_rows: (style_ttl.len() / STYLE_DIM) as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_is_send() {
        fn send<T: Send>() {}
        send::<GpuVectorEstimator>();
    }

    #[test]
    fn chunk_shape_reads_and_checks_sizes() {
        let (t, tt) = (7, 3);
        let ok = ChunkShape::of(&vec![0.0; 256 * tt], &vec![1.0; tt], &vec![0.0; 50 * 256], &vec![1.0; t], &vec![0.0; 144 * t]);
        assert_eq!(ok.unwrap(), ChunkShape { t: 7, n_text: 3, style_rows: 50 });
        assert!(ChunkShape::of(&vec![0.0; 256 * tt], &vec![1.0; tt], &vec![0.0; 50 * 256], &vec![1.0; t], &vec![0.0; 144 * t - 1]).is_err());
        assert!(ChunkShape::of(&vec![0.0; 255 * tt], &vec![1.0; tt], &vec![0.0; 50 * 256], &vec![1.0; t], &vec![0.0; 144 * t]).is_err());
        assert!(ChunkShape::of(&vec![0.0; 256 * tt], &vec![1.0; tt], &vec![0.0; 100], &vec![1.0; t], &vec![0.0; 144 * t]).is_err());
        assert!(ChunkShape::of(&vec![0.0; 256 * 1001], &vec![1.0; 1001], &vec![0.0; 256], &vec![1.0; t], &vec![0.0; 144 * t]).is_err());
    }

    /// The real bundle through ONNX Runtime's vector estimator and the GPU
    /// engine, from the same noise: the final latents must agree.
    ///
    ///   cargo test -p blue-rs --release ve_gpu -- --ignored --nocapture
    ///
    /// BLUEV3_DIR overrides the bundle (default: the desktop app's download).
    #[test]
    #[ignore = "needs the BlueV3 bundle and a GPU"]
    fn gpu_flow_matches_onnx_runtime() -> Result<()> {
        use crate::text::Tokenizer;
        use ndarray::{Array, Array1, Array3};
        use ort::{session::Session, value::Tensor};
        use rand::SeedableRng;
        use rand_distr::{Distribution, StandardNormal};
        use std::time::Instant;

        let dir = std::env::var_os("BLUEV3_DIR").map(std::path::PathBuf::from).unwrap_or_else(|| {
            let home = std::env::var_os("HOME").expect("HOME");
            Path::new(&home).join("Library/Application Support/com.maxmelichov.mambotts/models/bluev3-3.0.1")
        });
        let ve_path = dir.join("vector_estimator.onnx");
        let t0 = Instant::now();
        let Some(mut gpu) = GpuVectorEstimator::open(&ve_path)? else {
            eprintln!("no GPU adapter: skipped");
            return Ok(());
        };
        eprintln!("GPU open (upload, pipelines, self-check) {:.0} ms on {}", ms(t0), gpu.adapter());

        let session = |name: &str| -> Result<Session> {
            Session::builder()
                .map_err(|e| anyhow!("{e}"))?
                .with_intra_threads(8)
                .map_err(|e| anyhow!("{e}"))?
                .commit_from_file(dir.join(name))
                .map_err(|e| anyhow!("{e}"))
        };
        let mut text_encoder = session("text_encoder.onnx")?;
        let mut ve = session("vector_estimator.onnx")?;

        // Style: voices/<name>.json, style_ttl {dims, data}.
        let voice: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("voices/adam.json"))?)?;
        let dims: Vec<usize> = voice["style_ttl"]["dims"].as_array().context("dims")?.iter().map(|d| d.as_u64().unwrap() as usize).collect();
        fn flat(v: &serde_json::Value, out: &mut Vec<f32>) {
            match v {
                serde_json::Value::Array(a) => a.iter().for_each(|x| flat(x, out)),
                serde_json::Value::Number(n) => out.push(n.as_f64().unwrap() as f32),
                _ => {}
            }
        }
        let mut ttl = Vec::new();
        flat(&voice["style_ttl"]["data"], &mut ttl);
        let style_ttl = Array3::from_shape_vec((dims[0], dims[1], dims[2]), ttl)?;

        let tokenizer = Tokenizer::from_json(dir.join("vocab.json"))?;
        let ipa = "ðə kwˈɪk bɹˈaʊn fˈɑːks ʤˈʌmps ˌoʊvɚ ðə lˈeɪzi dˈɑːɡ, ænd ðɛn ɹˈʌnz hˈoʊm tə ɪts fˈæmɪli.";
        let (text_ids, text_mask) = tokenizer.encode_batch(&[ipa], &["en"])?;
        let text_emb = {
            let out = text_encoder.run(ort::inputs! {
                "text_ids" => Tensor::from_array(text_ids)?,
                "style_ttl" => Tensor::from_array(style_ttl.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
            })?;
            let (shape, data) = out[0].try_extract_tensor::<f32>()?;
            let d: Vec<usize> = shape.iter().map(|x| *x as usize).collect();
            Array3::from_shape_vec((d[0], d[1], d[2]), data.to_vec())?
        };

        // About 6.5 s of audio, noise as the release samples it.
        let (t, steps, cfg) = (93usize, 8usize, 4.0f32);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        let noise: Array3<f32> = Array::from_shape_fn((1, LAT, t), |_| StandardNormal.sample(&mut rng));
        let latent_mask = Array3::<f32>::ones((1, 1, t));

        let t0 = Instant::now();
        let mut xt = noise.clone();
        for step in 0..steps {
            let out = ve.run(ort::inputs! {
                "noisy_latent" => Tensor::from_array(xt)?,
                "text_emb" => Tensor::from_array(text_emb.clone())?,
                "style_ttl" => Tensor::from_array(style_ttl.clone())?,
                "latent_mask" => Tensor::from_array(latent_mask.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
                "current_step" => Tensor::from_array(Array1::from_vec(vec![step as f32]))?,
                "total_step" => Tensor::from_array(Array1::from_vec(vec![steps as f32]))?,
                "cfg_scale" => Tensor::from_array(Array1::from_vec(vec![cfg]))?,
            })?;
            let (_, data) = out[0].try_extract_tensor::<f32>()?;
            xt = Array3::from_shape_vec((1, LAT, t), data.to_vec())?;
        }
        let ort_ms = ms(t0);

        let args = (
            text_emb.as_slice().unwrap(),
            text_mask.as_slice().unwrap(),
            style_ttl.as_slice().unwrap(),
            latent_mask.as_slice().unwrap(),
            noise.as_slice().unwrap(),
        );
        let t0 = Instant::now();
        let got = gpu.flow(args.0, args.1, args.2, args.3, args.4, steps, cfg)?;
        let gpu_first_ms = ms(t0);
        let mut gpu_ms = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            let again = gpu.flow(args.0, args.1, args.2, args.3, args.4, steps, cfg)?;
            gpu_ms = gpu_ms.min(ms(t0));
            assert_eq!(again, got, "the GPU flow is deterministic");
        }

        let want = xt.as_slice().unwrap();
        let (mut max, mut sig, mut err) = (0f64, 0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            let d = (*g as f64 - *w as f64).abs();
            max = max.max(d);
            sig += (*w as f64).powi(2);
            err += d * d;
        }
        let snr = 10.0 * (sig / err.max(1e-30)).log10();
        eprintln!(
            "t={t} tokens={} steps={steps} cfg={cfg}: max |d| {max:.3e}, SNR {snr:.1} dB; ORT CPU {ort_ms:.0} ms, GPU {gpu_ms:.1} ms (first call {gpu_first_ms:.1} ms)",
            text_mask.len()
        );
        assert_eq!(got.len(), want.len());
        assert!(max < 5e-3, "GPU and ORT latents differ: max |d| {max:.3e}");
        assert!(snr > 70.0, "GPU and ORT latents differ: SNR {snr:.1} dB");
        Ok(())
    }

    fn ms(t0: std::time::Instant) -> f64 {
        t0.elapsed().as_secs_f64() * 1e3
    }
}
