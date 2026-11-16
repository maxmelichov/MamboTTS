//! The vector estimator on wgpu: weights uploaded once, pipelines compiled
//! once, and per input shape every buffer and bind group made once, so a
//! run is one buffer write (the step's time projections) and one compute
//! pass of 134 dispatches in one submit, with nothing read back.
//!
//!   load       weights, pipelines, the style keys of both CFG row kinds
//!              (they depend on weights only)
//!   prepare    per chunk (conditional inputs only; the engine builds the
//!              unconditional row from the graph's u_text / u_ref / u_key, as
//!              the graph does): masks, rotary tables, text keys/values, style values
//!   run        per step: everything that depends on the latent, both rows
//!   blend      the graph's CFG and Euler step into the next step's latent

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::task::{Poll, Waker};

use wgpu::util::DeviceExt;

use crate::model::{self, Layer, AS, AT, C, C2, LAT, MAX_LEN, TEMB, TXT};
use crate::onnx::{self, Graph, Source};
use crate::shaders::{self, ALoad, Epi, Tile, TILES};

/// Milliseconds since some origin (performance.now() in the browser).
pub fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        use wasm_bindgen::JsCast;
        let perf = js_sys::Reflect::get(&js_sys::global(), &"performance".into()).unwrap();
        let now: js_sys::Function = js_sys::Reflect::get(&perf, &"now".into()).unwrap().unchecked_into();
        now.call0(&perf).unwrap().as_f64().unwrap()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static T0: OnceLock<std::time::Instant> = OnceLock::new();
        T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

/// A one-shot future for wgpu's callbacks (map_async, on_submitted_work_done).
#[derive(Default)]
struct Shot<T> {
    val: Option<T>,
    waker: Option<Waker>,
}
#[derive(Clone)]
struct OneShot<T>(Arc<Mutex<Shot<T>>>);
impl<T> OneShot<T> {
    fn new() -> Self {
        OneShot(Arc::new(Mutex::new(Shot { val: None, waker: None })))
    }
    fn set(&self, v: T) {
        let mut s = self.0.lock().unwrap();
        s.val = Some(v);
        if let Some(w) = s.waker.take() {
            w.wake();
        }
    }
}
impl<T> std::future::Future for OneShot<T> {
    type Output = T;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<T> {
        let mut s = self.0.lock().unwrap();
        match s.val.take() {
            Some(v) => Poll::Ready(v),
            None => {
                s.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Resolves when the GPU has finished everything submitted so far.
pub async fn gpu_done(device: wgpu::Device, queue: wgpu::Queue) {
    let shot = OneShot::new();
    let s = shot.clone();
    queue.on_submitted_work_done(move || s.set(()));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    shot.await
}

/// Reads `len` floats of a buffer back to the CPU (copy to a staging buffer, map).
pub async fn read_buffer(device: wgpu::Device, queue: wgpu::Queue, src: wgpu::Buffer, len: usize) -> Result<Vec<f32>, String> {
    let staging = stage(&device, &queue, &src, 0, len);
    map_read(device, staging).await
}

/// Copies `len` floats of `src` (from float `offset`) into a new staging
/// buffer and submits the copy now, so later work on `src` cannot race it.
pub fn stage(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, offset: usize, len: usize) -> wgpu::Buffer {
    let size = (len * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, (offset * 4) as u64, &staging, 0, size);
    queue.submit([enc.finish()]);
    staging
}

/// Maps a staging buffer from `stage` and returns its floats. A failed map
/// (a lost device) is an error, never a panic: in wasm a panic aborts, and
/// inside a future that would leave the JS promise pending for good.
pub async fn map_read(device: wgpu::Device, staging: wgpu::Buffer) -> Result<Vec<f32>, String> {
    let shot = OneShot::new();
    let s = shot.clone();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| s.set(r));
    #[cfg(not(target_arch = "wasm32"))]
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    if let Err(e) = shot.await {
        staging.destroy();
        return Err(format!("reading back from the GPU failed: {e}"));
    }
    let out = match staging.slice(..).get_mapped_range() {
        Ok(r) => bytemuck::cast_slice::<u8, f32>(&r).to_vec(),
        Err(e) => {
            staging.destroy();
            return Err(format!("reading back from the GPU failed: {e:?}"));
        }
    };
    staging.unmap();
    staging.destroy();
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kernel {
    Matmul(Tile, ALoad, Epi),
    RowNorm { dw: bool },
    AttnScores { text: bool },
    AttnSoftmax,
    AttnPv { text: bool },
    Rope,
    Blend,
}

impl Kernel {
    pub fn source(&self) -> String {
        let salt = SHADER_SALT.lock().unwrap().clone();
        // The salt also goes into the matmuls' code (a term that is always 0 but
        // that no compiler can prove so): comments alone are stripped before the
        // driver's shader cache sees the source.
        let n = salt.bytes().fold(7u32, |h, c| h.wrapping_mul(31).wrapping_add(c as u32)) % 1_000_000;
        salt + &match *self {
            Kernel::Matmul(t, a, e) => shaders::matmul(t, a, e).replace("let m0 = wid.y * BM;", &format!("let m0 = wid.y * BM + (p.M >> 31u) * {n}u;")),
            Kernel::RowNorm { dw } => shaders::rownorm(dw),
            Kernel::AttnScores { text } => shaders::attn_scores(text),
            Kernel::AttnSoftmax => shaders::ATTN_SOFTMAX.to_string(),
            Kernel::AttnPv { text } => shaders::attn_pv(text),
            Kernel::Rope => shaders::ROPE.to_string(),
            Kernel::Blend => shaders::BLEND.to_string(),
        }
    }
}

/// One dispatch: pipeline index, bind group, workgroup counts.
struct Op {
    pipe: usize,
    bind: wgpu::BindGroup,
    wg: (u32, u32),
}

struct CnW {
    dw_w: wgpu::Buffer,
    dw_b: wgpu::Buffer,
    ln_g: wgpu::Buffer,
    ln_b: wgpu::Buffer,
    w1: wgpu::Buffer,
    b1: wgpu::Buffer,
    w2: wgpu::Buffer,
    b2: wgpu::Buffer,
    gamma: wgpu::Buffer,
    dil: u32,
    time: Option<usize>,
}

struct AttnW {
    text: bool,
    wq: wgpu::Buffer,
    bq: wgpu::Buffer,
    wk: wgpu::Buffer,
    bk: wgpu::Buffer,
    wv: wgpu::Buffer,
    bv: wgpu::Buffer,
    wo: wgpu::Buffer,
    bo: wgpu::Buffer,
    ln_g: wgpu::Buffer,
    ln_b: wgpu::Buffer,
    /// style attention: [tanh(tile Wk + bk); tanh(u_key Wk + bk)], [2, 50, 256],
    /// computed at load (conditional rows use the first half, unconditional the second)
    style_k: Option<wgpu::Buffer>,
}

enum LayerW {
    Cn(CnW),
    Attn(AttnW),
}

/// The time encoder and the four per-super-block projections, on the CPU
/// (a few thousand multiply-adds per step; the site's step graph does the same once per step count).
struct TimeW {
    freqs: Vec<f32>,
    w0: Vec<f32>,
    b0: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    proj: Vec<(Vec<f32>, Vec<f32>)>,
}

/// Buffers, bind groups and the dispatch list for one input shape.
struct ShapeState {
    b: u32,
    t: u32,
    tt: u32,
    tr: u32,
    noisy: wgpu::Buffer,
    out: wgpu::Buffer,
    sb: wgpu::Buffer,
    /// the last run's total_step (the blend divides by it)
    total: u32,
    ops: Vec<Op>,
    blend: Option<(Op, wgpu::Buffer)>,
    /// kept alive for the bind groups
    _keep: Vec<wgpu::Buffer>,
}

/// A submitted `profile` run.
pub struct ProfileJob {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pub(crate) res: wgpu::Buffer,
    pub(crate) names: Vec<String>,
    pub(crate) period: f64,
}

impl ProfileJob {
    /// (kernel, total ms, dispatches), in order of first use.
    pub async fn collect(self) -> Vec<(String, f64, usize)> {
        let n = self.names.len();
        let raw = read_buffer(self.device, self.queue, self.res, 4 * n).await.unwrap_or_else(|_| vec![0.0; 4 * n]);
        let ts: &[u64] = bytemuck::cast_slice(&raw);
        let mut by: Vec<(String, f64, usize)> = Vec::new();
        for (i, name) in self.names.into_iter().enumerate() {
            let dt = ts[2 * i + 1].saturating_sub(ts[2 * i]) as f64 * self.period / 1e6;
            match by.iter_mut().find(|x| x.0 == name) {
                Some(x) => {
                    x.1 += dt;
                    x.2 += 1;
                }
                None => by.push((name, dt, 1)),
            }
        }
        by
    }
}

/// Prepended to every shader (a comment): a new salt defeats the browser's and
/// the driver's compiled-shader caches, to measure a cold start.
pub static SHADER_SALT: Mutex<String> = Mutex::new(String::new());

/// Forces one matmul tile (index into shaders::TILES) for tuning; usize::MAX = automatic.
pub static TILE_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

#[derive(Clone, Debug, Default)]
pub struct Startup {
    pub parse_ms: f64,
    pub upload_ms: f64,
    pub pipelines_ms: f64,
    pub weight_bytes: u64,
}

/// Everything read from vector_estimator.onnx.
struct Weights {
    layers: Vec<LayerW>,
    proj_in: wgpu::Buffer,
    proj_out: wgpu::Buffer,
    /// the style attention's key tables, [tile; u_key], [100, 256]
    tile: wgpu::Buffer,
    time: TimeW,
    theta: Vec<f32>,
    style_k_pending: bool,
    /// the unconditional row's text vector (every text column) and style, from the graph
    u_text: Vec<f32>,
    u_ref: Vec<f32>,
}

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    kernels: Vec<Kernel>,
    pipes: Vec<wgpu::ComputePipeline>,
    pipe_index: HashMap<Kernel, usize>,
    w: Option<Weights>,
    shape: Option<ShapeState>,
    step_cache: HashMap<(u32, u32, u32), Vec<f32>>,
    pub startup: Startup,
}

/// Inputs of the start-up self-check (selfcheck_inputs): one chunk each (the
/// conditional inputs; the engine adds the unconditional row as the graph
/// does), (latent frames, text tokens). 12 frames run the short-chunk matmul
/// tiles (K split), 88 frames the long-chunk ones (Engine::tile).
pub const SELFCHECK_CASES: [(u32, u32); 2] = [(12, 5), (88, 9)];
/// Frames of case 1's latent kept in the golden file (every channel).
pub const SELFCHECK_FRAMES: [usize; 4] = [0, 1, 44, 85];
pub const SELFCHECK_STEP: (u32, u32) = (3, 8);
pub const SELFCHECK_CFG: f32 = 4.0;

/// ONNX Runtime CPU's output for the self-check inputs (engine-rs/selfcheck/golden.py):
/// the graph's denoised_latent (one step, CFG 4) of case 0, [144, 12]; then of
/// case 1 at SELFCHECK_FRAMES, [144][4]. f32 little-endian.
static GOLDEN: &[u8] = include_bytes!("../selfcheck/golden.bin");

/// A deterministic value in [-1, 1) from (seed, index): an integer hash, so
/// Python (golden.py) produces exactly the same f32.
pub fn selfcheck_value(seed: u32, i: u32) -> f32 {
    let mut h = i.wrapping_mul(0x9E37_79B1) ^ seed.wrapping_mul(0x85EB_CA77);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    (h as f64 / 4_294_967_296.0 * 2.0 - 1.0) as f32
}

/// One chunk's conditional inputs: (noisy [144,t], text_emb [256,tt],
/// text_mask [tt], style [50,256], latent_mask [t]). The last two frames are
/// padding (masked), and the noise is zero there, as the site sends it.
#[allow(clippy::type_complexity)]
pub fn selfcheck_inputs(t: usize, tt: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let masked = 2;
    let lm: Vec<f32> = (0..t).map(|i| if i + masked < t { 1.0 } else { 0.0 }).collect();
    let mut noisy = vec![0f32; LAT * t];
    for c in 0..LAT {
        for i in 0..t {
            noisy[c * t + i] = selfcheck_value(1, (c * t + i) as u32) * lm[i];
        }
    }
    // Scaled near the real inputs' size (text_emb std ~0.03, style ~0.08; powers
    // of two, so Python gets the same f32): at full scale [-1, 1] the graph is
    // ill-conditioned and ONNX Runtime's own kernels would differ by ~1e-4.
    let emb: Vec<f32> = (0..TXT * tt).map(|i| selfcheck_value(2, i as u32) * 0.0625).collect();
    let tm = vec![1f32; tt];
    let style: Vec<f32> = (0..50 * AS).map(|i| selfcheck_value(4, i as u32) * 0.125).collect();
    (noisy, emb, tm, style, lm)
}

/// The self-check's result: worst absolute difference to ONNX Runtime's output,
/// the largest reference value, SNR, and whether every value was finite.
#[derive(Clone, Debug, Default)]
pub struct SelfCheck {
    pub max_abs: f64,
    pub ref_max: f64,
    pub snr_db: f64,
    pub finite: bool,
    pub values: usize,
}

/// Compares the self-check's readbacks (each case's next latent, row 0) with GOLDEN.
pub fn selfcheck_compare(lat0: &[f32], lat1: &[f32]) -> SelfCheck {
    let (t0, _) = SELFCHECK_CASES[0];
    let (t1, _) = SELFCHECK_CASES[1];
    let (t0, t1) = (t0 as usize, t1 as usize);
    let mut got: Vec<f32> = lat0[..LAT * t0].to_vec();
    for c in 0..LAT {
        for f in SELFCHECK_FRAMES {
            got.push(lat1[c * t1 + f]);
        }
    }
    let want: Vec<f32> = GOLDEN.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let mut r = SelfCheck { finite: true, values: got.len(), ..Default::default() };
    if want.len() != got.len() {
        r.finite = false;
        r.max_abs = f64::INFINITY;
        return r;
    }
    let (mut sig, mut err) = (0f64, 0f64);
    for (g, w) in got.iter().zip(&want) {
        if !g.is_finite() {
            r.finite = false;
        }
        let d = (*g as f64 - *w as f64).abs();
        r.max_abs = if d.is_nan() { f64::INFINITY } else { r.max_abs.max(d) };
        r.ref_max = r.ref_max.max((*w as f64).abs());
        sig += (*w as f64).powi(2);
        err += if d.is_finite() { d * d } else { f64::INFINITY };
    }
    r.snr_db = if err == 0.0 { f64::INFINITY } else { 10.0 * (sig / err).log10() };
    r
}

/// Most rows a batch can have (16 chunks, both CFG rows each); rownorm's uniform holds their lengths.
pub const MAX_ROWS: u32 = shaders::MAX_ROWS;

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

/// A matmul weight as the kernels read it: W[k][n] in 16-column blocks,
/// [N / 16][K][16] (each workgroup's strip of W contiguous in memory: at
/// short lengths the matmuls were limited by scattered weight reads).
/// `out_in`: the source is [N][K] (a 1x1 conv's [out, in, 1]), else [K][N].
fn wblock(w: &[f32], k: usize, n: usize, out_in: bool) -> Vec<f32> {
    assert!(n % 16 == 0 && w.len() == k * n);
    let mut out = vec![0f32; w.len()];
    for ki in 0..k {
        for ni in 0..n {
            let v = if out_in { w[ni * k + ki] } else { w[ki * n + ni] };
            out[((ni / 16) * k + ki) * 16 + ni % 16] = v;
        }
    }
    out
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

/// y[i] = W[i, :] . x + b[i] (Gemm with transB), accumulated in f64, rounded once.
fn gemv(w: &[f32], b: &[f32], x: &[f32]) -> Vec<f32> {
    let k = x.len();
    b.iter().enumerate().map(|(i, bi)| (dot64(&w[i * k..(i + 1) * k], x) + *bi as f64) as f32).collect()
}

/// The DXC shipped beside the executable (scripts/pre_build.py stages
/// `dxcompiler.dll` and `dxil.dll` there), as wgpu's `DynamicDxc` path.
#[cfg(windows)]
fn bundled_dxc() -> Option<String> {
    let dll = std::env::current_exe().ok()?.parent()?.join("dxcompiler.dll");
    dll.is_file().then(|| dll.to_string_lossy().into_owned())
}

impl Engine {
    /// A device of our own (wgpu cannot adopt ONNX Runtime's), with WebGPU's
    /// default limits and no optional features: every kernel fits the spec
    /// minimums (8 storage buffers per stage, 16 KiB of workgroup memory, 256
    /// invocations, buffers far below 128 MiB at any chunk length the site makes).
    /// `timestamps`: also ask for timestamp queries (tuning only, `profile`).
    pub async fn new_device(timestamps: bool) -> Result<(wgpu::Device, wgpu::Queue), String> {
        // Natively: Metal on macOS, Vulkan or DX12 on Windows, Vulkan on Linux
        // (whichever the build enabled; the high-performance adapter wins).
        let backends = if cfg!(target_arch = "wasm32") {
            wgpu::Backends::BROWSER_WEBGPU
        } else {
            wgpu::Backends::METAL | wgpu::Backends::VULKAN | wgpu::Backends::DX12
        };
        Self::new_device_on(backends, timestamps).await
    }

    /// The native backends to try one at a time, in order, for a caller that
    /// checks each device's output and moves on when one is wrong: on Windows
    /// the same GPU can compute correctly through DX12 and wrongly through its
    /// Vulkan driver (an AMD Radeon iGPU fails the self-check on Vulkan), and
    /// asking for both at once lets wgpu settle on either. `WGPU_BACKEND`
    /// (e.g. `vulkan`) narrows the list to what it names.
    pub fn native_backend_candidates() -> Vec<wgpu::Backends> {
        let order: &[wgpu::Backends] = if cfg!(windows) {
            &[wgpu::Backends::DX12, wgpu::Backends::VULKAN]
        } else if cfg!(target_os = "macos") {
            &[wgpu::Backends::METAL]
        } else {
            &[wgpu::Backends::VULKAN]
        };
        let allowed = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::all());
        order.iter().copied().filter(|b| allowed.contains(*b)).collect()
    }

    /// [`Engine::new_device`] on the given backends only.
    pub async fn new_device_on(backends: wgpu::Backends, timestamps: bool) -> Result<(wgpu::Device, wgpu::Queue), String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = backends;
        #[cfg(windows)]
        if backends.contains(wgpu::Backends::DX12) {
            // wgpu's default looks for dxcompiler.dll on the search path and
            // otherwise falls back to FXC, which spends many minutes of CPU
            // compiling these kernels: model load looks hung. So DX12 runs
            // only on the DXC the app ships beside the executable.
            match bundled_dxc() {
                Some(dxc_path) => desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc { dxc_path },
                None => {
                    desc.backends.remove(wgpu::Backends::DX12);
                    if desc.backends.is_empty() {
                        return Err("no WebGPU adapter: DX12 needs the dxcompiler.dll shipped beside the executable".into());
                    }
                }
            }
        }
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no WebGPU adapter: {e}"))?;
        let required_features = if timestamps { adapter.features() & wgpu::Features::TIMESTAMP_QUERY } else { wgpu::Features::empty() };
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("bluetts-ve"),
                required_features,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no device: {e}"))
    }

    /// An engine with no weights and no pipelines yet (see `compile`, `load_weights`).
    pub fn open(device: wgpu::Device, queue: wgpu::Queue) -> Engine {
        Engine {
            device,
            queue,
            kernels: Vec::new(),
            pipes: Vec::new(),
            pipe_index: HashMap::new(),
            w: None,
            shape: None,
            step_cache: HashMap::new(),
            startup: Startup::default(),
        }
    }

    /// Every pipeline a run can use with the automatic tiles, plus the style
    /// keys' matmul: compiled once, not on the first run of a new length (a
    /// forced tile compiles in prepare). The model is not needed.
    pub fn kernels_to_compile() -> Vec<Kernel> {
        // Every matmul a chunk (or batch) up to MAX_ROWS rows of 1024 frames can
        // pick (Engine::tile depends on the row count only up to a few hundred rows).
        let mut ks: Vec<Kernel> = Vec::new();
        for m in 1..=1024u32 {
            for (mm, n, k, a, e) in Self::matmul_shapes(m, m, m) {
                let kern = Kernel::Matmul(Self::tile(mm, n, k, a), a, e);
                if !ks.contains(&kern) {
                    ks.push(kern);
                }
            }
        }
        ks.extend([
            Kernel::RowNorm { dw: true },
            Kernel::RowNorm { dw: false },
            Kernel::AttnScores { text: true },
            Kernel::AttnScores { text: false },
            Kernel::AttnSoftmax,
            Kernel::AttnPv { text: true },
            Kernel::AttnPv { text: false },
            Kernel::Rope,
            Kernel::Blend,
        ]);
        let style_keys = Self::style_keys_kernel();
        if !ks.contains(&style_keys) {
            ks.push(style_keys);
        }
        ks
    }

    /// The style keys' matmul: [tile; u_key] (100 rows) x W_key, tanh.
    fn style_keys_kernel() -> Kernel {
        Kernel::Matmul(Self::tile(100, AS as u32, AS as u32, ALoad::Rows), ALoad::Rows, Epi::BiasTanh)
    }

    /// The WGSL of `kernels_to_compile`, in order, exactly as `compile` hands
    /// it to the device (the page compiles these asynchronously first, so the
    /// browser has them cached when `compile` asks).
    pub fn kernel_sources() -> Vec<String> {
        Self::kernels_to_compile().iter().map(|k| k.source()).collect()
    }

    /// Creates every pipeline of `kernels_to_compile`; with weights loaded,
    /// also computes the style keys.
    pub fn compile(&mut self) {
        let t0 = now_ms();
        for k in Self::kernels_to_compile() {
            self.pipe(k);
        }
        self.startup.pipelines_ms += now_ms() - t0;
        self.style_keys();
    }

    /// Reads the weights out of the ONNX bytes and uploads them; compiles every pipeline.
    pub fn load<S: Source + ?Sized>(device: wgpu::Device, queue: wgpu::Queue, src: &S) -> Result<Engine, String> {
        let mut e = Self::open(device, queue);
        e.load_weights(src)?;
        e.compile();
        Ok(e)
    }

    /// The style attention's keys depend on weights only: tanh([tile; u_key] Wk + bk),
    /// once, after both the weights and the pipelines exist.
    fn style_keys(&mut self) {
        let Some(w) = self.w.as_ref() else { return };
        if !w.style_k_pending {
            return;
        }
        self.pipe(Self::style_keys_kernel());
        let w = self.w.as_ref().unwrap();
        let mut keep = Vec::new();
        let mut ops = Vec::new();
        let mut keys = Vec::new();
        for l in &w.layers {
            if let LayerW::Attn(a) = l {
                if !a.text {
                    let k = self.buffer(100 * AS);
                    ops.push(self.matmul(ALoad::Rows, Epi::BiasTanh, 100, AS as u32, AS as u32, 1, 0, &w.tile, &a.wk, &k, &[&a.bk], &mut keep));
                    keys.push(k);
                }
            }
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, &ops);
        self.queue.submit([enc.finish()]);
        let w = self.w.as_mut().unwrap();
        let mut keys = keys.into_iter();
        for l in w.layers.iter_mut() {
            if let LayerW::Attn(a) = l {
                if !a.text {
                    a.style_k = keys.next();
                }
            }
        }
        w.style_k_pending = false;
    }

    /// Reads the weights out of the ONNX bytes and uploads them (no pipelines
    /// needed; the style keys follow at `compile`).
    pub fn load_weights<S: Source + ?Sized>(&mut self, src: &S) -> Result<(), String> {
        let device = self.device.clone();
        let t0 = now_ms();
        let g = onnx::parse(src)?;
        let seq = model::layers(&g)?;
        let freqs = model::time_freqs(&g)?;
        let parse_ms = now_ms() - t0;

        let t0 = now_ms();
        let mut bytes = 0u64;
        // mat: Some((K, N, out_in)) for a matmul weight (see wblock), None as stored.
        let mut w = |name: &str, shape: &[usize], mat: Option<(usize, usize, bool)>| -> Result<wgpu::Buffer, String> {
            let v = Self::weight(&g, src, name, shape)?;
            let v = match mat {
                Some((k, n, out_in)) => wblock(&v, k, n, out_in),
                None => v,
            };
            bytes += (v.len() * 4) as u64;
            Ok(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(&v),
                usage: STORAGE,
            }))
        };
        // A 1x1 conv: `<name>.weight_mmT` [in, out] in onnx/ (after the release's
        // Conv -> MatMul pass), `<name>.weight` [out, in, 1] in onnx_plain/.
        let conv = |w: &mut dyn FnMut(&str, &[usize], Option<(usize, usize, bool)>) -> Result<wgpu::Buffer, String>, name: &str, cin: usize, cout: usize| {
            let mm = format!("{name}.weight_mmT");
            if g.inits.contains_key(&mm) {
                w(&mm, &[cin, cout], Some((cin, cout, false)))
            } else {
                w(&format!("{name}.weight"), &[cout, cin, 1], Some((cin, cout, true)))
            }
        };
        let mut layers = Vec::new();
        for l in &seq {
            match l {
                Layer::ConvNext { prefix, dil, time } => {
                    let p = |s: &str| format!("vf.{prefix}.{s}");
                    layers.push(LayerW::Cn(CnW {
                        dw_w: w(&p("dwconv.weight"), &[C, 1, 5], None)?,
                        dw_b: w(&p("dwconv.bias"), &[C], None)?,
                        ln_g: w(&p("norm.norm.weight"), &[C], None)?,
                        ln_b: w(&p("norm.norm.bias"), &[C], None)?,
                        w1: conv(&mut w, &p("pwconv1"), C, C2)?,
                        b1: w(&p("pwconv1.bias"), &[C2], None)?,
                        w2: conv(&mut w, &p("pwconv2"), C2, C)?,
                        b2: w(&p("pwconv2.bias"), &[C], None)?,
                        gamma: w(&p("gamma"), &[1, C, 1], None)?,
                        dil: *dil,
                        time: *time,
                    }));
                }
                Layer::TextAttn { block } | Layer::StyleAttn { block } => {
                    let text = matches!(l, Layer::TextAttn { .. });
                    let attn = if text { "attn" } else { "attention" };
                    // text: q 512 -> 512, k/v 256 -> 512, out 512 -> 512; style: q 512 -> 256, k/v 256 -> 256, out 256 -> 512
                    let (kin, aw) = if text { (TXT, AT) } else { (AS, AS) };
                    let lw = |which: &str| model::linear_weight(&g, *block, attn, which);
                    let bias = |which: &str| format!("vf.main_blocks.{block}.{attn}.{which}.linear.bias");
                    layers.push(LayerW::Attn(AttnW {
                        text,
                        wq: w(&lw("W_query")?, &[C, aw], Some((C, aw, false)))?,
                        bq: w(&bias("W_query"), &[aw], None)?,
                        wk: w(&lw("W_key")?, &[kin, aw], Some((kin, aw, false)))?,
                        bk: w(&bias("W_key"), &[aw], None)?,
                        wv: w(&lw("W_value")?, &[kin, aw], Some((kin, aw, false)))?,
                        bv: w(&bias("W_value"), &[aw], None)?,
                        wo: w(&lw("out_fc")?, &[aw, C], Some((aw, C, false)))?,
                        bo: w(&bias("out_fc"), &[C], None)?,
                        ln_g: w(&format!("vf.main_blocks.{block}.norm.norm.weight"), &[C], None)?,
                        ln_b: w(&format!("vf.main_blocks.{block}.norm.norm.bias"), &[C], None)?,
                        style_k: None,
                    }));
                }
            }
        }
        let proj_in = conv(&mut w, "vf.proj_in.net", LAT, C)?;
        let proj_out = conv(&mut w, "vf.proj_out.net", C, LAT)?;
        let f = |name: &str, shape: &[usize]| Self::weight(&g, src, name, shape);
        // The style keys' two tables, one after the other: [tile; u_key].
        let mut keys_in = f("tile", &[1, 50, AS])?;
        keys_in.extend(f("u_key", &[1, 50, AS])?);
        bytes += (keys_in.len() * 4) as u64;
        let tile = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("tile|u_key"), contents: bytemuck::cast_slice(&keys_in), usage: STORAGE });
        let u_text = f("u_text", &[1, TXT, 1])?;
        let u_ref = f("u_ref", &[1, 50, AS])?;
        let time = TimeW {
            freqs,
            w0: f("vf.time_encoder.mlp.0.linear.weight", &[256, TEMB])?,
            b0: f("vf.time_encoder.mlp.0.linear.bias", &[256])?,
            w2: f("vf.time_encoder.mlp.2.linear.weight", &[TEMB, 256])?,
            b2: f("vf.time_encoder.mlp.2.linear.bias", &[TEMB])?,
            proj: [1, 7, 13, 19]
                .iter()
                .map(|b| {
                    Ok((
                        f(&format!("vf.main_blocks.{b}.linear.linear.weight"), &[C, TEMB])?,
                        f(&format!("vf.main_blocks.{b}.linear.linear.bias"), &[C])?,
                    ))
                })
                .collect::<Result<_, String>>()?,
        };
        let theta = f("vf.main_blocks.3.attn.theta", &[1, 1, 32])?;
        self.startup.parse_ms = parse_ms;
        self.startup.upload_ms = now_ms() - t0;
        self.startup.weight_bytes = bytes;
        self.w = Some(Weights { layers, proj_in, proj_out, tile, time, theta, style_k_pending: true, u_text, u_ref });
        self.drop_shape();
        self.step_cache.clear();
        if !self.pipes.is_empty() {
            self.style_keys();
        }
        Ok(())
    }

    fn weight<S: Source + ?Sized>(g: &Graph, src: &S, name: &str, shape: &[usize]) -> Result<Vec<f32>, String> {
        let t = g.inits.get(name).ok_or_else(|| format!("no initializer {name}"))?;
        if t.dims != shape {
            return Err(format!("{name}: dims {:?}, expected {:?}", t.dims, shape));
        }
        onnx::read_f32(src, t)
    }

    fn pipe(&mut self, k: Kernel) -> usize {
        if let Some(&i) = self.pipe_index.get(&k) {
            return i;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("VE_DUMP").is_ok() {
            eprintln!("=== {k:?}\n{}", k.source());
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(k.source().into()),
        });
        let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipes.push(p);
        self.kernels.push(k);
        self.pipe_index.insert(k, self.pipes.len() - 1);
        self.pipes.len() - 1
    }

    fn buffer(&self, floats: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (floats.max(4) * 4) as u64,
            usage: STORAGE,
            mapped_at_creation: false,
        })
    }

    fn uniform(&self, words: [u32; 8], keep: &mut Vec<wgpu::Buffer>) -> wgpu::Buffer {
        let b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        keep.push(b.clone());
        b
    }

    /// A uniform of 4 words, then each row's length (rownorm's depthwise conv).
    fn rows_uniform(&self, head: [u32; 4], lens: &[u32], keep: &mut Vec<wgpu::Buffer>) -> wgpu::Buffer {
        let mut words = vec![0u32; 4 + MAX_ROWS as usize];
        words[..4].copy_from_slice(&head);
        words[4..4 + lens.len()].copy_from_slice(lens);
        let b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&words),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        keep.push(b.clone());
        b
    }

    fn op(&self, pipe: usize, bufs: &[&wgpu::Buffer], wg: (u32, u32)) -> Op {
        let entries: Vec<_> = bufs
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes[pipe].get_bind_group_layout(0),
            entries: &entries,
        });
        Op { pipe, bind, wg }
    }

    /// Whether the automatic tile for a run's matmuls over `rows` rows splits K.
    pub fn splits_k(rows: u32) -> bool {
        Self::tile(rows, C2 as u32, C as u32, ALoad::Rows).ks > 1
    }

    /// Every matmul of a run and of `prepare`: (M, N, K, A layout, epilogue),
    /// for m latent rows, text rows and style rows.
    fn matmul_shapes(m: u32, text_rows: u32, style_rows: u32) -> [(u32, u32, u32, ALoad, Epi); 11] {
        let (c, c2, at, as_, txt, lat) = (C as u32, C2 as u32, AT as u32, AS as u32, TXT as u32, LAT as u32);
        [
            (m, c, lat, ALoad::ChanFirst, Epi::ProjIn),
            (m, c2, c, ALoad::Rows, Epi::BiasGelu),
            (m, c, c2, ALoad::Rows, Epi::CnOut { time: false }),
            (m, c, c2, ALoad::Rows, Epi::CnOut { time: true }),
            (m, at, c, ALoad::Rows, Epi::Bias),
            (m, as_, c, ALoad::Rows, Epi::Bias),
            (m, c, at, ALoad::Rows, Epi::AttnOut),
            (m, c, as_, ALoad::Rows, Epi::AttnOut),
            (m, lat, c, ALoad::Rows, Epi::ProjOut),
            (text_rows, at, txt, ALoad::ChanFirst, Epi::Bias),
            (style_rows, as_, as_, ALoad::Rows, Epi::Bias),
        ]
    }

    /// The tile for an M x N x K matmul, from sweeps in Chromium on Apple M5
    /// (tools/perf/ve-ab.mjs --flow --tiles ...). At the row counts a chunk
    /// has, a run is a few waves of workgroups and its time steps up with the
    /// number of row tiles, so the tile follows M: 64-row tiles when M fills
    /// them, 32-row tiles when M is just past a multiple of 64 (an odd number
    /// of 32-row tiles), from 161 to 256 rows always. That gained up to 23%
    /// over one tile per range (88-96 frames), nothing at 136-160 frames.
    /// K is split in 2 up to 160 rows and never past them, as before, so every
    /// run adds its products in the same order as with the earlier tiles: the
    /// output is bit-identical. Swept again for BlueV3 (every tile, 40-380
    /// frames, batches of 1 and 3): no tile beat this rule by more than the
    /// run-to-run noise (~5%), so it stays.
    fn tile(m: u32, n: u32, k: u32, a: ALoad) -> Tile {
        let fits = |t: &Tile| k % (t.bk * t.ks) == 0 && (t.ks == 1 || (a == ALoad::Rows && t.tm % 4 == 0));
        let i = TILE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if i < TILES.len() && fits(&TILES[i]) {
            return TILES[i];
        }
        let odd32 = m.div_ceil(32) % 2 == 1;
        let t = if n < 256 {
            TILES[0] // 64 x 64, 4 x 4 per thread
        } else if a == ALoad::Rows && n == AS as u32 && k == AS as u32 {
            // The style values and keys (50 rows per batch row): K split as it
            // was at 2 rows (100), whatever the batch, so a chunk's rows in a
            // batch of several get exactly its values.
            TILES[12]
        } else if m <= 160 {
            if odd32 {
                TILES[11] // 32 x 64, 4 x 4, K split in 2
            } else {
                TILES[12] // 64 x 32, 4 x 4, K split in 2
            }
        } else if m <= 256 {
            TILES[1] // 32 x 64, 4 x 4
        } else if m <= 320 || !odd32 {
            TILES[3] // 64 x 128, 4 x 8
        } else {
            TILES[13] // 32 x 64, 4 x 4, 32 deep
        };
        if fits(&t) { t } else { TILES[0] }
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul(
        &self,
        a_load: ALoad,
        epi: Epi,
        m: u32,
        n: u32,
        k: u32,
        t: u32,
        tb: u32,
        a: &wgpu::Buffer,
        w: &wgpu::Buffer,
        c: &wgpu::Buffer,
        extra: &[&wgpu::Buffer],
        keep: &mut Vec<wgpu::Buffer>,
    ) -> Op {
        let tile = Self::tile(m, n, k, a_load);
        let pipe = self.pipe_index[&Kernel::Matmul(tile, a_load, epi)];
        let u = self.uniform([m, n, k, t, tb, 0, 0, 0], keep);
        let mut bufs = vec![a, w, c, &u];
        bufs.extend_from_slice(extra);
        self.op(pipe, &bufs, (n.div_ceil(tile.bn), m.div_ceil(tile.bm)))
    }

    fn encode(&self, enc: &mut wgpu::CommandEncoder, ops: &[Op]) {
        let mut pass = enc.begin_compute_pass(&Default::default());
        let mut last = usize::MAX;
        for op in ops {
            if op.pipe != last {
                pass.set_pipeline(&self.pipes[op.pipe]);
                last = op.pipe;
            }
            pass.set_bind_group(0, &op.bind, &[]);
            pass.dispatch_workgroups(op.wg.0, op.wg.1, 1);
        }
    }

    pub fn dispatches_per_run(&self) -> usize {
        self.shape.as_ref().map_or(0, |s| s.ops.len())
    }

    /// Per chunk: shapes, masks, rotary tables, text keys/values and style
    /// values, from the chunk's conditional inputs as the graph takes them:
    /// t = latent frames, tt = text tokens, tr = style rows (50); text_emb
    /// [256, tt], text_mask [tt], style [tr, 256], latent_mask [t]. The engine
    /// runs it as a CFG pair (conditional, unconditional), as the graph does.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(&mut self, t: u32, tt: u32, tr: u32, text_emb: &[f32], text_mask: &[f32], style: &[f32], latent_mask: &[f32]) -> Result<(), String> {
        self.prepare_rows(1, t, tt, tr, text_emb, text_mask, style, latent_mask)
    }

    /// `prepare` for `k` chunks in one batch (2k rows: each chunk's conditional
    /// then unconditional row), padded to `t` frames and `tt` text columns:
    /// text_emb [k, 256, tt], text_mask [k, tt], style [k, tr, 256], latent_mask [k, t].
    /// A row's own length is its latent mask's sum: its depthwise convs fill and
    /// pad from its own last valid frame, its padding frames and text columns
    /// are masked, so each row computes what it would alone (up to the matmul
    /// tile, which follows the batch's row count).
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_rows(
        &mut self,
        k: u32,
        t: u32,
        tt: u32,
        tr: u32,
        text_emb: &[f32],
        text_mask: &[f32],
        style: &[f32],
        latent_mask: &[f32],
    ) -> Result<(), String> {
        if self.w.is_none() {
            return Err("prepare: no weights loaded".into());
        }
        self.style_keys();
        let b = 2 * k;
        let (ku, bu, tu, ttu, tru) = (k as usize, b as usize, t as usize, tt as usize, tr as usize);
        if k == 0 || b > MAX_ROWS {
            return Err(format!("prepare: {k} chunks; the engine runs at most {} at once", MAX_ROWS / 2));
        }
        if t == 0 || tt == 0 {
            return Err("prepare: empty input".into());
        }
        if t > MAX_LEN || tt > MAX_LEN {
            return Err(format!("prepare: {t} frames, {tt} tokens; the graph's rotary tables cover {MAX_LEN}"));
        }
        if text_emb.len() != ku * TXT * ttu || text_mask.len() != ku * ttu || style.len() != ku * tru * AS || latent_mask.len() != ku * tu {
            return Err("prepare: input sizes do not match the shape".into());
        }
        if tr != 50 {
            return Err(format!("prepare: style has {tr} rows; the style keys have 50"));
        }
        // The CFG pairs, as the graph builds them: the unconditional row's text
        // is u_text in every column (same text mask), its style u_ref.
        let (text_emb, text_mask, style, latent_mask, lens) = {
            let w = self.w.as_ref().unwrap();
            let mut emb = vec![0f32; bu * TXT * ttu];
            let mut tm = vec![0f32; bu * ttu];
            let mut sty = vec![0f32; bu * tru * AS];
            let mut lm = vec![0f32; bu * tu];
            let mut lens = vec![0u32; bu];
            for c in 0..ku {
                let (r0, r1) = (2 * c, 2 * c + 1);
                emb[r0 * TXT * ttu..(r0 + 1) * TXT * ttu].copy_from_slice(&text_emb[c * TXT * ttu..(c + 1) * TXT * ttu]);
                for ch in 0..TXT {
                    emb[(r1 * TXT + ch) * ttu..(r1 * TXT + ch + 1) * ttu].fill(w.u_text[ch]);
                }
                for r in [r0, r1] {
                    tm[r * ttu..(r + 1) * ttu].copy_from_slice(&text_mask[c * ttu..(c + 1) * ttu]);
                    lm[r * tu..(r + 1) * tu].copy_from_slice(&latent_mask[c * tu..(c + 1) * tu]);
                    // The graph's edge fill: frames from the mask's sum on take the last valid one.
                    lens[r] = (latent_mask[c * tu..(c + 1) * tu].iter().sum::<f32>().round() as u32).clamp(1, t);
                }
                sty[r0 * tru * AS..(r0 + 1) * tru * AS].copy_from_slice(&style[c * tru * AS..(c + 1) * tru * AS]);
                sty[r1 * tru * AS..(r1 + 1) * tru * AS].copy_from_slice(&w.u_ref);
            }
            (emb, tm, sty, lm, lens)
        };
        let (text_emb, text_mask, style, latent_mask, lens) = (&text_emb[..], &text_mask[..], &style[..], &latent_mask[..], &lens[..]);
        let m = b * t;
        let mu = m as usize;
        // This shape's matmul pipelines (already compiled unless a tile is forced).
        let (c, c2, at, as_, lat) = (C as u32, C2 as u32, AT as u32, AS as u32, LAT as u32);
        for (mm, n, k, a, e) in Self::matmul_shapes(m, b * tt, b * tr) {
            self.pipe(Kernel::Matmul(Self::tile(mm, n, k, a), a, e));
        }
        let w = self.w.as_ref().unwrap();
        let mut keep: Vec<wgpu::Buffer> = Vec::new();
        let dev = &self.device;
        let init = |data: &[f32], keep: &mut Vec<wgpu::Buffer>| {
            let buf = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: STORAGE,
            });
            keep.push(buf.clone());
            buf
        };

        // Rotary tables: angle = (position / row length) * theta, as the graph
        // computes them in f32 (Div, then Mul), cos/sin rounded once from f64.
        let rope = |len_of: &dyn Fn(usize) -> f32, rows: usize, per: usize| {
            let mut r = vec![0f32; rows * 64];
            for row in 0..rows {
                let pos = (row % per) as f32 / len_of(row / per);
                for i in 0..32 {
                    let ang = (pos * w.theta[i]) as f64;
                    r[row * 64 + i] = ang.cos() as f32;
                    r[row * 64 + 32 + i] = ang.sin() as f32;
                }
            }
            r
        };
        let lat_len = |bi: usize| latent_mask[bi * tu..(bi + 1) * tu].iter().sum::<f32>();
        let txt_len = |bi: usize| text_mask[bi * ttu..(bi + 1) * ttu].iter().sum::<f32>();
        let rope_q = init(&rope(&lat_len, mu, tu), &mut keep);
        let rope_k = init(&rope(&txt_len, bu * ttu, ttu), &mut keep);
        let mask = init(latent_mask, &mut keep);
        let tmask = init(text_mask, &mut keep);
        let emb = init(text_emb, &mut keep);
        let sty = init(style, &mut keep);

        let mk = |n: usize, keep: &mut Vec<wgpu::Buffer>| {
            let buf = self.buffer(n);
            keep.push(buf.clone());
            buf
        };
        let noisy = mk(bu * LAT * tu, &mut keep);
        let out = mk(bu * LAT * tu, &mut keep);
        let x = mk(mu * C, &mut keep);
        let y = mk(mu * C, &mut keep);
        let h = mk(mu * C2, &mut keep);
        let q = mk(mu * AT, &mut keep);
        let o = mk(mu * AT, &mut keep);
        let sb = mk(4 + 4 * bu * C, &mut keep);
        // attention scores scratch, [B, H, T, L], the larger of text (8 heads) and style (2)
        let sc = mk((bu * tu * (8 * ttu).max(2 * tru)).max(4), &mut keep);

        // Per-chunk keys and values.
        let mut prep = Vec::new();
        let mut kv = Vec::new();
        let txt = TXT as u32;
        for l in &w.layers {
            if let LayerW::Attn(a) = l {
                if a.text {
                    let k = mk(bu * ttu * AT, &mut keep);
                    let v = mk(bu * ttu * AT, &mut keep);
                    let r = b * tt;
                    prep.push(self.matmul(ALoad::ChanFirst, Epi::Bias, r, at, txt, tt, 0, &emb, &a.wk, &k, &[&a.bk], &mut keep));
                    prep.push(self.matmul(ALoad::ChanFirst, Epi::Bias, r, at, txt, tt, 0, &emb, &a.wv, &v, &[&a.bv], &mut keep));
                    let u = self.uniform([r, 0, 0, 0, 0, 0, 0, 0], &mut keep);
                    prep.push(self.op(self.pipe_index[&Kernel::Rope], &[&k, &rope_k, &u], (r, 1)));
                    kv.push((k, v));
                } else {
                    let v = mk(bu * tru * AS, &mut keep);
                    prep.push(self.matmul(ALoad::Rows, Epi::Bias, b * tr, as_, as_, tr, 0, &sty, &a.wv, &v, &[&a.bv], &mut keep));
                    kv.push((a.style_k.clone().unwrap(), v));
                }
            }
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, &prep);
        self.queue.submit([enc.finish()]);

        // The run.
        let mut ops = vec![self.matmul(ALoad::ChanFirst, Epi::ProjIn, m, c, lat, t, 0, &noisy, &w.proj_in, &x, &[&mask], &mut keep)];
        let rn_dw = self.pipe_index[&Kernel::RowNorm { dw: true }];
        let rn_ln = self.pipe_index[&Kernel::RowNorm { dw: false }];
        let mut kvi = kv.iter();
        for l in &w.layers {
            match l {
                LayerW::Cn(cw) => {
                    let u = self.rows_uniform([m, t, cw.dil, 0], lens, &mut keep);
                    ops.push(self.op(rn_dw, &[&x, &y, &u, &mask, &cw.ln_g, &cw.ln_b, &cw.dw_w, &cw.dw_b], (m, 1)));
                    ops.push(self.matmul(ALoad::Rows, Epi::BiasGelu, m, c2, c, t, 0, &y, &cw.w1, &h, &[&cw.b1], &mut keep));
                    match cw.time {
                        Some(k) => {
                            let tb = 4 + (k * bu * C) as u32;
                            ops.push(self.matmul(ALoad::Rows, Epi::CnOut { time: true }, m, c, c2, t, tb, &h, &cw.w2, &x, &[&cw.b2, &cw.gamma, &mask, &sb], &mut keep));
                        }
                        None => ops.push(self.matmul(ALoad::Rows, Epi::CnOut { time: false }, m, c, c2, t, 0, &h, &cw.w2, &x, &[&cw.b2, &cw.gamma, &mask], &mut keep)),
                    }
                }
                LayerW::Attn(a) => {
                    let (k, v) = kvi.next().unwrap();
                    let aw = if a.text { at } else { as_ };
                    ops.push(self.matmul(ALoad::Rows, Epi::Bias, m, aw, c, t, 0, &x, &a.wq, &q, &[&a.bq], &mut keep));
                    // scores -> softmax -> weighted values, through the scratch `sc` [B, H, T, L]
                    let (heads, keys) = if a.text { (8u32, tt) } else { (2u32, tr) };
                    let u = self.uniform([t, keys, heads, 0, 0, 0, 0, 0], &mut keep);
                    let qblocks = t.div_ceil(shaders::QB);
                    let scores = self.pipe_index[&Kernel::AttnScores { text: a.text }];
                    if a.text {
                        ops.push(self.op(scores, &[&q, k, &sc, &u, &rope_q, &tmask], (qblocks, b * heads)));
                    } else {
                        ops.push(self.op(scores, &[&q, k, &sc, &u], (qblocks, b * heads)));
                    }
                    let rows = b * heads * t;
                    ops.push(self.op(self.pipe_index[&Kernel::AttnSoftmax], &[&sc, &u, &mask], (rows.min(65535), rows.div_ceil(65535))));
                    ops.push(self.op(self.pipe_index[&Kernel::AttnPv { text: a.text }], &[&sc, v, &o, &u], (qblocks, b * heads)));
                    ops.push(self.matmul(ALoad::Rows, Epi::AttnOut, m, c, aw, t, 0, &o, &a.wo, &y, &[&a.bo, &mask, &x], &mut keep));
                    let u = self.uniform([m, t, 0, 0, 0, 0, 0, 0], &mut keep);
                    ops.push(self.op(rn_ln, &[&y, &x, &u, &mask, &a.ln_g, &a.ln_b], (m, 1)));
                }
            }
        }
        ops.push(self.matmul(ALoad::Rows, Epi::ProjOut, m, lat, c, t, 0, &x, &w.proj_out, &out, &[&mask], &mut keep));

        // The CFG blend and Euler step: out (both velocities of every pair) -> the next step's noisy (both rows).
        let n = (LAT * tu) as u32;
        let pairs = b / 2;
        let bu_ = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[n, 0, pairs, 0, t, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let blend = Some((self.op(self.pipe_index[&Kernel::Blend], &[&out, &noisy, &bu_, &mask], ((n * pairs).div_ceil(256), 1)), bu_.clone()));
        self.drop_shape();
        self.shape = Some(ShapeState { b, t, tt, tr, noisy, out, sb, total: 0, ops, blend, _keep: keep });
        Ok(())
    }

    /// Destroys the per-shape buffers. On WebGPU a dropped wgpu::Buffer is only
    /// freed when JS collects its GPUBuffer, so without this every chunk's
    /// activations (tens of MB at long chunks) would wait for the garbage
    /// collector. Work already submitted still completes: destroy() takes
    /// effect after it.
    fn drop_shape(&mut self) {
        if let Some(s) = self.shape.take() {
            for b in &s._keep {
                b.destroy();
            }
            if let Some((_, u)) = &s.blend {
                u.destroy();
            }
        }
    }

    /// Each chunk's latent, [k, 144, t]: written to both rows of its pair.
    pub fn set_latent(&self, x: &[f32]) -> Result<(), String> {
        let s = self.shape.as_ref().ok_or("prepare first")?;
        let n = s.t as usize * LAT;
        if x.len() != (s.b / 2) as usize * n {
            return Err("set_latent: wrong size".into());
        }
        for (c, chunk) in x.chunks_exact(n).enumerate() {
            for r in 0..2 {
                self.queue.write_buffer(&s.noisy, ((2 * c + r) * n * 4) as u64, bytemuck::cast_slice(chunk));
            }
        }
        Ok(())
    }

    /// sb: [1/total, 0, 0, 0, then the four time projections, each [b, 512]].
    fn step_buffer(&mut self, step: u32, total: u32, b: u32) -> &Vec<f32> {
        let tw = &self.w.as_ref().expect("weights").time;
        self.step_cache.entry((step, total, b)).or_insert_with(|| {
            // The time encoder exactly as the graph orders it, in f32 where it rounds:
            // t = step / total; x = t * 1000; arg_j = x * freq_j; [sin, cos] -> MLP (Mish) -> 64.
            let tt = (step as f32 / total as f32) * 1000.0;
            let mut e = vec![0f32; TEMB];
            for j in 0..TEMB / 2 {
                let a = (tt * tw.freqs[j]) as f64;
                e[j] = a.sin() as f32;
                e[TEMB / 2 + j] = a.cos() as f32;
            }
            let h: Vec<f32> = gemv(&tw.w0, &tw.b0, &e)
                .into_iter()
                .map(|v| {
                    let v = v as f64;
                    let sp = v.exp().ln_1p();
                    (v * sp.tanh()) as f32
                })
                .collect();
            let temb = gemv(&tw.w2, &tw.b2, &h);
            let mut sb = vec![0f32; 4 + 4 * b as usize * C];
            sb[0] = (1.0 / total as f32) * 1.0;
            for (k, (w, bias)) in tw.proj.iter().enumerate() {
                let p = gemv(w, bias, &temb);
                for bi in 0..b as usize {
                    let o = 4 + (k * b as usize + bi) * C;
                    sb[o..o + C].copy_from_slice(&p);
                }
            }
            sb
        })
    }

    /// One vector-estimator run: one buffer write and one submit.
    pub fn run(&mut self, step: u32, total: u32) -> Result<(), String> {
        let b = self.shape.as_ref().ok_or("prepare first")?.b;
        if total == 0 || step >= total {
            return Err(format!("run: step {step} of {total}"));
        }
        let sbv = self.step_buffer(step, total, b).clone();
        self.shape.as_mut().unwrap().total = total;
        let s = self.shape.as_ref().unwrap();
        self.queue.write_buffer(&s.sb, 0, bytemuck::cast_slice(&sbv));
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, &s.ops);
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// About one 96-frame run's worth of the current shape's dispatches (all of
    /// them for a chunk that short): work for the GPU to clock up on while the
    /// next request's text is prepared. Its results are never read, and the
    /// next `prepare` replaces every buffer. False when there is no shape.
    pub fn warm(&mut self) -> bool {
        let Some(b) = self.shape.as_ref().map(|s| s.b) else { return false };
        let sbv = self.step_buffer(0, 8, b).clone();
        let s = self.shape.as_ref().unwrap();
        let n = (s.ops.len() * 96).div_ceil((s.t as usize * s.b as usize / 2).max(96)).min(s.ops.len());
        self.queue.write_buffer(&s.sb, 0, bytemuck::cast_slice(&sbv));
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, &s.ops[..n]);
        self.queue.submit([enc.finish()]);
        true
    }

    /// The CFG blend and Euler step of the last run into the next step's
    /// latent (both rows), guidance `g`: (x + (v_u + g (v_c - v_u)) / total) x mask.
    pub fn blend(&self, g: f32) -> Result<(), String> {
        let s = self.shape.as_ref().ok_or("prepare first")?;
        let (op, u) = s.blend.as_ref().ok_or("no blend")?;
        if s.total == 0 {
            return Err("blend: run first".into());
        }
        let n = (LAT as u32) * s.t;
        let mut w = [0u32; 8];
        w[0] = n;
        w[1] = g.to_bits();
        w[2] = s.b / 2;
        w[3] = (s.total as f32).to_bits();
        w[4] = s.t;
        self.queue.write_buffer(u, 0, bytemuck::cast_slice(&w));
        let mut enc = self.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, std::slice::from_ref(op));
        self.queue.submit([enc.finish()]);
        Ok(())
    }

    /// (device, queue, buffer, floats) of the run's output, for read_buffer.
    pub fn output(&self) -> Option<(wgpu::Device, wgpu::Queue, wgpu::Buffer, usize)> {
        let s = self.shape.as_ref()?;
        Some((self.device.clone(), self.queue.clone(), s.out.clone(), (s.b * s.t) as usize * LAT))
    }

    /// Same for the latent input (after `blend`, the next step's input).
    pub fn latent(&self) -> Option<(wgpu::Device, wgpu::Queue, wgpu::Buffer, usize)> {
        let s = self.shape.as_ref()?;
        Some((self.device.clone(), self.queue.clone(), s.noisy.clone(), (s.b * s.t) as usize * LAT))
    }

    /// GPU time of each dispatch of one run (one pass per dispatch with
    /// timestamps; needs the timestamp-query feature, in Chromium
    /// --enable-webgpu-developer-features for full precision). Submits the
    /// run; `ProfileJob::collect` reads the result.
    pub fn profile(&mut self, step: u32, total: u32) -> Option<ProfileJob> {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        self.run(step, total).ok()?;
        let s = self.shape.as_ref()?;
        let n = s.ops.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * n });
        let res = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16 * n as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, op) in s.ops.iter().enumerate() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some(2 * i as u32),
                    end_of_pass_write_index: Some(2 * i as u32 + 1),
                }),
            });
            pass.set_pipeline(&self.pipes[op.pipe]);
            pass.set_bind_group(0, &op.bind, &[]);
            pass.dispatch_workgroups(op.wg.0, op.wg.1, 1);
        }
        enc.resolve_query_set(&qs, 0..2 * n, &res, 0);
        self.queue.submit([enc.finish()]);
        Some(ProfileJob {
            device: self.device.clone(),
            queue: self.queue.clone(),
            res,
            names: s.ops.iter().map(|op| format!("{:?}", self.kernels[op.pipe])).collect(),
            period: self.queue.get_timestamp_period() as f64,
        })
    }

    /// Starts the self-check: both SELFCHECK_CASES through prepare, one run
    /// and the CFG blend, each next latent (row 0) copied to a staging buffer
    /// now. `selfcheck_compare` checks them against GOLDEN.
    pub fn selfcheck_start(&mut self) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
        let (step, total) = SELFCHECK_STEP;
        let mut staged = Vec::new();
        for (t, tt) in SELFCHECK_CASES.iter() {
            let (noisy, emb, tm, style, lm) = selfcheck_inputs(*t as usize, *tt as usize);
            self.prepare(*t, *tt, 50, &emb, &tm, &style, &lm)?;
            self.set_latent(&noisy)?;
            self.run(step, total)?;
            self.blend(SELFCHECK_CFG)?;
            let s = self.shape.as_ref().unwrap();
            staged.push(stage(&self.device, &self.queue, &s.noisy, 0, LAT * *t as usize));
        }
        self.drop_shape();
        let b = staged.pop().unwrap();
        let a = staged.pop().unwrap();
        Ok((a, b))
    }

    /// (device, queue, buffer, offset, floats) of row `row` of the latent input.
    pub fn latent_row(&self, row: u32) -> Option<(wgpu::Device, wgpu::Queue, wgpu::Buffer, usize, usize)> {
        let s = self.shape.as_ref()?;
        if row >= s.b {
            return None;
        }
        let n = s.t as usize * LAT;
        Some((self.device.clone(), self.queue.clone(), s.noisy.clone(), row as usize * n, n))
    }

    /// Rows 0, 2, 4, ... of the latent input (each chunk's conditional row, [144, t]
    /// padded), copied into one staging buffer now: [b / 2, 144, t].
    pub fn stage_latent_pairs(&self) -> Option<(wgpu::Device, wgpu::Buffer)> {
        let s = self.shape.as_ref()?;
        let n = (s.t as usize * LAT * 4) as u64;
        let pairs = (s.b as u64).div_ceil(2);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: n * pairs,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        let step = if s.b >= 2 { 2 } else { 1 };
        for k in 0..pairs {
            enc.copy_buffer_to_buffer(&s.noisy, k * step * n, &staging, k * n, n);
        }
        self.queue.submit([enc.finish()]);
        Some((self.device.clone(), staging))
    }

    /// GPU memory the current shape's buffers take (activations, scratch, per-chunk keys and values).
    pub fn shape_bytes(&self) -> u64 {
        self.shape.as_ref().map_or(0, |s| s._keep.iter().map(|b| b.size()).sum::<u64>() + s.blend.as_ref().map_or(0, |(_, u)| u.size()))
    }

    /// Drops the per-shape buffers and destroys the device (and so every buffer on it).
    pub fn destroy(&mut self) {
        self.drop_shape();
        self.w = None;
        self.device.destroy();
    }

    pub fn shape(&self) -> Option<(u32, u32, u32, u32)> {
        self.shape.as_ref().map(|s| (s.b, s.t, s.tt, s.tr))
    }
}
