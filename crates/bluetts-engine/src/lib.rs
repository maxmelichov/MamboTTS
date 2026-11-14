//! BlueTTS vector estimator, hand-written for WebGPU.
//!
//! Ported from the BlueTTS site (LightBlue-site, branch bluev3, engine-rs/),
//! so the desktop app and the site run the same kernels. `engine` is
//! platform-neutral (wgpu); natively it runs on Metal (macOS), Vulkan or DX12
//! (Windows) and Vulkan (Linux), driven by blue-rs (`ve_gpu.rs`) and by
//! `examples/parity.rs` and `examples/selfcheck.rs`. In the browser the
//! `VeEngine` wasm-bindgen class below is what the site's worker loads; it is
//! kept behind `cfg(target_arch = "wasm32")` so the two copies stay diffable.
//!
//! `g2p` is the second model, RenikudPlus (the Hebrew G2P), as the
//! `G2pEngine` class of the same wasm (natively: `examples/g2p.rs`).

// The sources mirror the site's engine-rs/ line for line; lints from newer
// clippy releases are allowed rather than fixed so the two stay diffable.
#![allow(clippy::manual_is_multiple_of, clippy::type_complexity, clippy::len_without_is_empty)]

pub mod engine;
pub mod g2p;
pub mod g2p_shaders;
pub mod model;
pub mod onnx;
pub mod shaders;

/// The wgpu the engine was built against, for callers that handle its device.
pub use wgpu;

#[cfg(target_arch = "wasm32")]
mod web {
    use crate::engine::{self, Engine};
    use crate::g2p::G2p;
    use crate::onnx::Source;
    use wasm_bindgen::JsCast;
    use std::sync::{Arc, Mutex};
    use wasm_bindgen::prelude::*;

    /// The model bytes stay in JS; the parser copies small windows, the
    /// upload one tensor at a time (wasm memory never holds the 257 MB of BlueV3).
    struct JsBytes(js_sys::Uint8Array);
    impl Source for JsBytes {
        fn len(&self) -> usize {
            self.0.length() as usize
        }
        fn read_into(&self, offset: usize, dst: &mut [u8]) {
            self.0.subarray(offset as u32, (offset + dst.len()) as u32).copy_to(dst);
        }
    }


    impl crate::g2p::Upload for JsBytes {
        /// Straight from the JS bytes into the GPU buffer (queue.writeBuffer
        /// with a view of the model): the 1.2 GB never enters wasm memory.
        fn upload(&self, queue: &wgpu::Queue, dst: &wgpu::Buffer, range: (usize, usize)) {
            let (Some(q), Some(b)) = (queue.as_webgpu(), dst.as_webgpu()) else {
                let mut v = vec![0u8; range.1 - range.0];
                self.read_into(range.0, &mut v);
                queue.write_buffer(dst, 0, &v);
                return;
            };
            let q = JsValue::from(q.clone());
            let f: js_sys::Function = js_sys::Reflect::get(&q, &"writeBuffer".into()).unwrap().unchecked_into();
            let args = js_sys::Array::of5(&JsValue::from(b.clone()), &0.into(), &self.0, &(range.0 as f64).into(), &((range.1 - range.0) as f64).into());
            let _ = f.apply(&q, &args);
        }
    }

    /// RenikudPlus (the Hebrew G2P) on its own WebGPU device; see g2p.rs.
    #[wasm_bindgen]
    pub struct G2pEngine {
        e: G2p,
        device_ms: f64,
        error: Arc<Mutex<Option<String>>>,
    }

    #[wasm_bindgen]
    impl G2pEngine {
        /// A device (default limits, no features); no pipelines or weights yet.
        pub async fn open(timestamps: bool) -> Result<G2pEngine, JsError> {
            let t0 = engine::now_ms();
            let (device, queue) = Engine::new_device(timestamps).await.map_err(err)?;
            let device_ms = engine::now_ms() - t0;
            let error = Arc::new(Mutex::new(None));
            let sink = error.clone();
            device.on_uncaptured_error(Arc::new(move |e| {
                let mut s = sink.lock().unwrap();
                if s.is_none() {
                    *s = Some(e.to_string());
                }
            }));
            Ok(G2pEngine { e: G2p::open(device, queue), device_ms, error })
        }

        #[wasm_bindgen(js_name = gpuDevice)]
        pub fn gpu_device(&self) -> JsValue {
            self.e.device.as_webgpu().map(|d| JsValue::from(d.clone())).unwrap_or(JsValue::NULL)
        }

        /// The WGSL `compile` creates pipelines from (entry point `main`, layout auto).
        #[wasm_bindgen(js_name = kernelSources)]
        pub fn kernel_sources() -> js_sys::Array {
            G2p::kernel_sources().into_iter().map(JsValue::from).collect()
        }

        pub fn compile(&mut self) {
            self.e.compile();
        }

        /// Reads model.onnx (the bytes the site caches) and queues its weights'
        /// upload; `done()` resolves when the GPU has them.
        pub fn load(&mut self, model: js_sys::Uint8Array) -> Result<(), JsError> {
            self.e.load_weights(&JsBytes(model)).map_err(err)
        }

        /// One run: token ids (CLS, characters, SEP) and speaker ids, resolving
        /// to the logits as a Float32Array [S, 33] (consonant 25, vowel 6, stress 2).
        pub fn run(&mut self, ids: &[u32], speaker: u32, target: u32) -> js_sys::Promise {
            let staged = self.e.run(ids, speaker, target);
            let device = self.e.device.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let st = staged.map_err(|e| JsValue::from(JsError::new(&e)))?;
                let v = engine::map_read(device, st).await.map_err(|e| JsValue::from(JsError::new(&e)))?;
                Ok(js_sys::Float32Array::from(&v[..]).into())
            })
        }

        pub fn done(&self) -> js_sys::Promise {
            let (d, q) = (self.e.device.clone(), self.e.queue.clone());
            wasm_bindgen_futures::future_to_promise(async move {
                engine::gpu_done(d, q).await;
                Ok(JsValue::UNDEFINED)
            })
        }

        #[wasm_bindgen(js_name = takeError)]
        pub fn take_error(&self) -> Option<String> {
            self.error.lock().unwrap().take()
        }

        pub fn destroy(&mut self) {
            self.e.destroy();
        }

        /// { deviceMs, parseMs, uploadMs, pipelinesMs, weightBytes }
        pub fn startup(&self) -> js_sys::Object {
            let o = js_sys::Object::new();
            let s = &self.e.startup;
            for (k, v) in [
                ("deviceMs", self.device_ms),
                ("parseMs", s.parse_ms),
                ("uploadMs", s.upload_ms),
                ("pipelinesMs", s.pipelines_ms),
                ("weightBytes", s.weight_bytes as f64),
            ] {
                set(&o, k, v.into());
            }
            o
        }

        #[wasm_bindgen(js_name = shapeBytes)]
        pub fn shape_bytes(&self) -> f64 {
            self.e.shape_bytes() as f64
        }

        #[wasm_bindgen(js_name = dispatchesPerRun)]
        pub fn dispatches_per_run(&self) -> u32 {
            self.e.dispatches_per_run() as u32
        }

        /// Tuning: force matmul tile `tile` of shaders::TILES (-1: automatic) and
        /// K slices (0: automatic) for the 1024-wide and the wider matmuls; the
        /// kept shapes are dropped so the next run uses them.
        #[wasm_bindgen(js_name = setTuning)]
        pub fn set_tuning(&mut self, tile: i32, kz_narrow: u32, kz_wide: u32) {
            use std::sync::atomic::Ordering::Relaxed;
            crate::g2p::TILE_OVERRIDE.store(if tile < 0 { usize::MAX } else { tile as usize }, Relaxed);
            crate::g2p::KZ_OVERRIDE[0].store(kz_narrow, Relaxed);
            crate::g2p::KZ_OVERRIDE[1].store(kz_wide, Relaxed);
            self.e.drop_shapes();
        }

        /// Tuning: per-kernel GPU time of one run, [[kernel, ms, dispatches], ...], or null.
        pub fn profile(&mut self, ids: &[u32]) -> js_sys::Promise {
            let job = self.e.profile(ids);
            wasm_bindgen_futures::future_to_promise(async move {
                let Some(job) = job else { return Ok(JsValue::NULL) };
                let arr = js_sys::Array::new();
                for (k, ms, n) in job.collect().await {
                    arr.push(&js_sys::Array::of3(&k.into(), &ms.into(), &(n as u32).into()));
                }
                Ok(arr.into())
            })
        }
    }

    fn err(e: String) -> JsError {
        JsError::new(&e)
    }

    fn set(o: &js_sys::Object, k: &str, v: JsValue) {
        let _ = js_sys::Reflect::set(o, &k.into(), &v);
    }

    #[wasm_bindgen]
    pub struct VeEngine {
        e: Engine,
        device_ms: f64,
        /// The first error the device reported outside any error scope.
        error: Arc<Mutex<Option<String>>>,
    }

    #[wasm_bindgen]
    impl VeEngine {
        /// A WebGPU device (default limits, no features) and an engine with no
        /// pipelines or weights yet: `compile`, then `load`. `timestamps`: ask for
        /// timestamp queries too (`profile` only).
        pub async fn open(timestamps: bool) -> Result<VeEngine, JsError> {
            let t0 = engine::now_ms();
            let (device, queue) = Engine::new_device(timestamps).await.map_err(err)?;
            let device_ms = engine::now_ms() - t0;
            let error = Arc::new(Mutex::new(None));
            let sink = error.clone();
            // Without a handler the browser only logs these; keep the first, so a
            // chunk that hit one fails instead of returning whatever was in the buffers.
            device.on_uncaptured_error(Arc::new(move |e| {
                let mut s = sink.lock().unwrap();
                if s.is_none() {
                    *s = Some(e.to_string());
                }
            }));
            Ok(VeEngine { e: Engine::open(device, queue), device_ms, error })
        }

        /// open + compile + load in one call (the A/B page).
        pub async fn create(model: js_sys::Uint8Array) -> Result<VeEngine, JsError> {
            let mut v = Self::open(false).await?;
            v.compile();
            v.load(model)?;
            Ok(v)
        }

        /// The GPUDevice wgpu runs on, to watch `lost` and to compile the
        /// kernels asynchronously ahead of `compile`.
        #[wasm_bindgen(js_name = gpuDevice)]
        pub fn gpu_device(&self) -> JsValue {
            self.e.device.as_webgpu().map(|d| JsValue::from(d.clone())).unwrap_or(JsValue::NULL)
        }

        /// The WGSL `compile` creates pipelines from, in its order (entry point `main`, layout auto).
        #[wasm_bindgen(js_name = kernelSources)]
        pub fn kernel_sources() -> js_sys::Array {
            Engine::kernel_sources().into_iter().map(JsValue::from).collect()
        }

        /// Creates every pipeline (synchronously; the browser compiles them in its
        /// GPU process, so the cost shows up at the first use unless they were
        /// compiled asynchronously from `kernelSources` before).
        pub fn compile(&mut self) {
            self.e.compile();
        }

        /// Reads vector_estimator.onnx (the bytes the site caches) and uploads its weights.
        pub fn load(&mut self, model: js_sys::Uint8Array) -> Result<(), JsError> {
            self.e.load_weights(&JsBytes(model)).map_err(err)
        }

        /// The first uncaptured GPU error since the last call, or undefined.
        #[wasm_bindgen(js_name = takeError)]
        pub fn take_error(&self) -> Option<String> {
            self.error.lock().unwrap().take()
        }

        /// Destroys the device: every buffer and pipeline goes with it.
        pub fn destroy(&mut self) {
            self.e.destroy();
        }

        /// { deviceMs, parseMs, uploadMs, pipelinesMs, weightBytes }
        pub fn startup(&self) -> js_sys::Object {
            let o = js_sys::Object::new();
            let s = &self.e.startup;
            for (k, v) in [
                ("deviceMs", self.device_ms),
                ("parseMs", s.parse_ms),
                ("uploadMs", s.upload_ms),
                ("pipelinesMs", s.pipelines_ms),
                ("weightBytes", s.weight_bytes as f64),
            ] {
                set(&o, k, v.into());
            }
            o
        }

        /// One run of fixed inputs per tile family, compared with ONNX Runtime
        /// CPU's output for them (engine-rs/selfcheck/): resolves to
        /// { maxAbs, refMax, snrDb, finite, values }.
        #[wasm_bindgen(js_name = selfCheck)]
        pub fn self_check(&mut self) -> js_sys::Promise {
            let started = self.e.selfcheck_start();
            let device = self.e.device.clone();
            wasm_bindgen_futures::future_to_promise(async move {
                let (a, b) = started.map_err(|e| JsValue::from(JsError::new(&e)))?;
                let lat = engine::map_read(device.clone(), a).await.map_err(|e| JsValue::from(JsError::new(&e)))?;
                let out = engine::map_read(device, b).await.map_err(|e| JsValue::from(JsError::new(&e)))?;
                let r = engine::selfcheck_compare(&lat, &out);
                let o = js_sys::Object::new();
                set(&o, "maxAbs", r.max_abs.into());
                set(&o, "refMax", r.ref_max.into());
                set(&o, "snrDb", r.snr_db.into());
                set(&o, "finite", r.finite.into());
                set(&o, "values", (r.values as u32).into());
                Ok(o.into())
            })
        }

        /// One chunk, from its conditional inputs (the engine adds the
        /// unconditional row as the graph does; see Engine::prepare):
        /// text_emb [256, tt], text_mask [tt], style [tr, 256], latent_mask [t].
        #[allow(clippy::too_many_arguments)]
        pub fn prepare(&mut self, t: u32, tt: u32, tr: u32, text_emb: &[f32], text_mask: &[f32], style: &[f32], latent_mask: &[f32]) -> Result<(), JsError> {
            self.e.prepare(t, tt, tr, text_emb, text_mask, style, latent_mask).map_err(err)
        }

        /// `prepare` for `k` chunks in one batch (Engine::prepare_rows), each
        /// padded to t frames and tt text columns: text_emb [k, 256, tt],
        /// text_mask [k, tt], style [k, tr, 256], latent_mask [k, t].
        #[allow(clippy::too_many_arguments)]
        #[wasm_bindgen(js_name = prepareRows)]
        pub fn prepare_rows(&mut self, k: u32, t: u32, tt: u32, tr: u32, text_emb: &[f32], text_mask: &[f32], style: &[f32], latent_mask: &[f32]) -> Result<(), JsError> {
            self.e.prepare_rows(k, t, tt, tr, text_emb, text_mask, style, latent_mask).map_err(err)
        }

        /// Each chunk's final latent (row 0 of every pair after the last
        /// `blend`), [k, 144, t] with t the padded length: the one readback.
        #[wasm_bindgen(js_name = readLatentPairs)]
        pub fn read_latent_pairs(&self) -> js_sys::Promise {
            let staged = self.e.stage_latent_pairs();
            wasm_bindgen_futures::future_to_promise(async move {
                let (d, s) = staged.ok_or_else(|| JsValue::from(JsError::new("prepare first")))?;
                let v = engine::map_read(d, s).await.map_err(|e| JsValue::from(JsError::new(&e)))?;
                Ok(js_sys::Float32Array::from(&v[..]).into())
            })
        }

        #[wasm_bindgen(js_name = setLatent)]
        pub fn set_latent(&self, x: &[f32]) -> Result<(), JsError> {
            self.e.set_latent(x).map_err(err)
        }

        /// Encodes and submits one run; returns as soon as it is submitted.
        pub fn run(&mut self, step: u32, total: u32) -> Result<(), JsError> {
            self.e.run(step, total).map_err(err)
        }

        /// Throwaway work on the last chunk's buffers (Engine::warm), so the GPU
        /// is busy, and clocked up, while a request's text front end runs on
        /// the CPU. False when there is no shape yet.
        pub fn warm(&mut self) -> bool {
            self.e.warm()
        }

        /// CFG blend and Euler step of the last run into the next step's latent (guidance g).
        pub fn blend(&self, g: f32) -> Result<(), JsError> {
            self.e.blend(g).map_err(err)
        }

        /// Resolves when the GPU has finished everything submitted.
        pub fn done(&self) -> js_sys::Promise {
            let (d, q) = (self.e.device.clone(), self.e.queue.clone());
            wasm_bindgen_futures::future_to_promise(async move {
                engine::gpu_done(d, q).await;
                Ok(JsValue::UNDEFINED)
            })
        }

        /// The last run's velocities [2k, 144, t] (both rows of every pair) as a Float32Array.
        #[wasm_bindgen(js_name = readOutput)]
        pub fn read_output(&self) -> js_sys::Promise {
            Self::read(self.e.output().map(|(d, q, b, n)| (d, q, b, 0, n)))
        }

        /// The latent input buffer (after `blend`: the next step's input), [b, 144, t].
        #[wasm_bindgen(js_name = readLatent)]
        pub fn read_latent(&self) -> js_sys::Promise {
            Self::read(self.e.latent().map(|(d, q, b, n)| (d, q, b, 0, n)))
        }

        /// One row of the latent input, [144, t]: after the last `blend`, the
        /// chunk's final latent (both rows hold it). The one readback of a chunk.
        #[wasm_bindgen(js_name = readLatentRow)]
        pub fn read_latent_row(&self, row: u32) -> js_sys::Promise {
            Self::read(self.e.latent_row(row))
        }

        fn read(src: Option<(wgpu::Device, wgpu::Queue, wgpu::Buffer, usize, usize)>) -> js_sys::Promise {
            // The copy is submitted now, before any later run can change the buffer.
            let staged = src.map(|(d, q, b, off, n)| {
                let s = engine::stage(&d, &q, &b, off, n);
                (d, s)
            });
            wasm_bindgen_futures::future_to_promise(async move {
                let (d, s) = staged.ok_or_else(|| JsValue::from(JsError::new("prepare first")))?;
                let v = engine::map_read(d, s).await.map_err(|e| JsValue::from(JsError::new(&e)))?;
                Ok(js_sys::Float32Array::from(&v[..]).into())
            })
        }

        /// Whether a run's matmuls over `rows` rows split K in two (Engine::tile).
        /// Rows run in a batch of several chunks give bit-identical results to
        /// the chunk run alone when neither splits.
        #[wasm_bindgen(js_name = splitsK)]
        pub fn splits_k(rows: u32) -> bool {
            Engine::splits_k(rows)
        }

        /// Measuring: a comment prepended to every shader, so none is found in a cache.
        #[wasm_bindgen(js_name = setShaderSalt)]
        pub fn set_shader_salt(salt: &str) {
            *engine::SHADER_SALT.lock().unwrap() = format!("// {salt}\n");
        }

        /// Tuning: force matmul tile `i` of shaders::TILES for later `prepare`s (-1: automatic).
        #[wasm_bindgen(js_name = setTileOverride)]
        pub fn set_tile_override(i: i32) {
            engine::TILE_OVERRIDE.store(if i < 0 { usize::MAX } else { i as usize }, std::sync::atomic::Ordering::Relaxed);
        }

        /// Tuning: per-kernel GPU time of one run, [[kernel, ms, dispatches], ...], or
        /// null without timestamp queries.
        pub fn profile(&mut self, step: u32, total: u32) -> js_sys::Promise {
            let job = self.e.profile(step, total);
            wasm_bindgen_futures::future_to_promise(async move {
                let Some(job) = job else { return Ok(JsValue::NULL) };
                let arr = js_sys::Array::new();
                for (k, ms, n) in job.collect().await {
                    arr.push(&js_sys::Array::of3(&k.into(), &ms.into(), &(n as u32).into()));
                }
                Ok(arr.into())
            })
        }

        /// Bytes of GPU memory the current chunk (or batch) shape holds, on top of the weights.
        #[wasm_bindgen(js_name = shapeBytes)]
        pub fn shape_bytes(&self) -> f64 {
            self.e.shape_bytes() as f64
        }

        /// Compute dispatches in one run (and one submit per run).
        #[wasm_bindgen(js_name = dispatchesPerRun)]
        pub fn dispatches_per_run(&self) -> u32 {
            self.e.dispatches_per_run() as u32
        }
    }
}
