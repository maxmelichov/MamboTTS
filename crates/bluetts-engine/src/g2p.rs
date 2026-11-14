//! RenikudPlus (notmax123/RenikudPlus @679c56c, model.onnx, fp32) on wgpu:
//! the Hebrew G2P's encoder and heads as hand-written WGSL, read from the
//! same ONNX bytes the site caches (docs/MODELS.md, "Rust engine (Hebrew G2P)").
//!
//! The graph is a BERT-large encoder over characters, one forward pass (no
//! loop, nothing autoregressive):
//!
//!   x = LN(word[id] + type[0] + pos[t])                        eps 1e-12
//!   24 x [ q, k, v = x Wq + bq, x Wk + bk, x Wv + bv          (16 heads of 64)
//!          a = softmax((q / 64^.25) . (k / 64^.25)) v          (all keys: batch 1, full mask)
//!          x = LN(a Wo + bo + x)
//!          x = LN(gelu(x W1 + b1) W2 + b2 + x) ]                (1024 -> 4096 -> 1024)
//!   film = [spk[s], tspk[t]] Wf^T  (2048 = scale | shift)
//!   y = x * (1 + scale) + shift;  z = LN(y)                    eps 1e-5
//!   r = y + gelu(z M0 + b0) M3 + b3
//!   consonant = r Wc + bc; vowel = [r, softmax c] Wv + bv;
//!   stress = [r, softmax c, softmax v] Ws + bs                  (25, 6, 2)
//!
//! A run: ids and the FiLM vector written, 223 dispatches (271 up to 32
//! rows, where every matmul splits K across workgroups and adds a reduce
//! pass) in one compute pass and one submit, the [S, 33] logits copied out
//! and read back once.
//! Per padded length (rows rounded up to 32) the buffers, bind groups and
//! dispatch list are built once and kept (a few lengths at a time).

use std::collections::HashMap;

use wgpu::util::DeviceExt;

use crate::engine::{map_read, now_ms};
use crate::onnx::{self, Graph, Source};
use crate::shaders::{self, ALoad, Epi, Tile, TILES};
use crate::g2p_shaders::{self, LnMode};

pub const H: usize = 1024; // hidden width
pub const FF: usize = 4096; // feed-forward width
pub const LAYERS: usize = 24;
pub const MAX_POS: usize = 2048;
pub const OUT: usize = 33; // 25 consonant + 6 vowel + 2 stress logits
/// Rows are padded to a multiple of this.
pub const ROW_STEP: u32 = 32;
/// Padded lengths whose buffers are kept at once.
const SHAPES_KEPT: usize = 3;

/// Model bytes that can also be written straight into a GPU buffer (natively
/// a slice; in the browser the JS Uint8Array, uploaded without passing
/// through wasm memory).
pub trait Upload: Source {
    fn upload(&self, queue: &wgpu::Queue, dst: &wgpu::Buffer, range: (usize, usize));
}

impl Upload for [u8] {
    fn upload(&self, queue: &wgpu::Queue, dst: &wgpu::Buffer, range: (usize, usize)) {
        queue.write_buffer(dst, 0, &self[range.0..range.1]);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GKernel {
    /// tile, epilogue, K slices across workgroups (1: none; else Epi::Partial)
    Matmul(Tile, Epi, u32),
    Reduce(u32, Epi),
    Ln(LnMode),
    Attn,
    Heads,
    Reblock,
}

impl GKernel {
    pub fn source(&self) -> String {
        let salt = crate::engine::SHADER_SALT.lock().unwrap().clone();
        let n = salt.bytes().fold(7u32, |h, c| h.wrapping_mul(31).wrapping_add(c as u32)) % 1_000_000;
        salt + &match *self {
            GKernel::Matmul(t, e, kz) => shaders::matmul_ex(t, ALoad::Rows, e, true)
                .replace("let m0 = wid.x * BM;", &format!("let m0 = wid.x * BM + (p.M >> 31u) * {n}u;"))
                .replace("let nit = p.K / (BK * KS);\n    let kbeg = lid.z * nit * BK;", &format!("let nit = p.K / (BK * KS * {kz}u);\n    let kbeg = (wid.z * KS + lid.z) * nit * BK;")),
            GKernel::Reduce(kz, e) => g2p_shaders::reduce(kz, e),
            GKernel::Ln(m) => g2p_shaders::layernorm(m),
            GKernel::Attn => g2p_shaders::attention(),
            GKernel::Heads => g2p_shaders::HEADS.to_string(),
            GKernel::Reblock => g2p_shaders::REBLOCK.to_string(),
        }
    }
}

struct Op {
    pipe: usize,
    bind: wgpu::BindGroup,
    wg: (u32, u32, u32),
}

struct LayerW {
    wqkv: wgpu::Buffer,
    bqkv: wgpu::Buffer,
    wo: wgpu::Buffer,
    bo: wgpu::Buffer,
    ln1_g: wgpu::Buffer,
    ln1_b: wgpu::Buffer,
    w1: wgpu::Buffer,
    b1: wgpu::Buffer,
    w2: wgpu::Buffer,
    b2: wgpu::Buffer,
    ln2_g: wgpu::Buffer,
    ln2_b: wgpu::Buffer,
}

struct Weights {
    word: wgpu::Buffer,
    tok: wgpu::Buffer,
    pos: wgpu::Buffer,
    emb_g: wgpu::Buffer,
    emb_b: wgpu::Buffer,
    layers: Vec<LayerW>,
    lnf_g: wgpu::Buffer,
    lnf_b: wgpu::Buffer,
    m0: wgpu::Buffer,
    m0b: wgpu::Buffer,
    m3: wgpu::Buffer,
    m3b: wgpu::Buffer,
    wc: wgpu::Buffer,
    wv: wgpu::Buffer,
    ws: wgpu::Buffer,
    hb: wgpu::Buffer,
    /// on the CPU: speaker tables [3, 32] x 2 and the FiLM projection [2048, 64]
    spk: Vec<f32>,
    tspk: Vec<f32>,
    film: Vec<f32>,
    vocab: usize,
    speakers: usize,
    all: Vec<wgpu::Buffer>,
}

struct Shape {
    m: u32,
    ids: wgpu::Buffer,
    len: wgpu::Buffer,
    film: wgpu::Buffer,
    out: wgpu::Buffer,
    ops: Vec<Op>,
    keep: Vec<wgpu::Buffer>,
    used: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Startup {
    pub parse_ms: f64,
    pub upload_ms: f64,
    pub pipelines_ms: f64,
    pub weight_bytes: u64,
}

/// Forces one matmul tile (index into shaders::TILES) for tuning; usize::MAX = automatic.
pub static TILE_OVERRIDE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Forces K slices across workgroups (tuning; 0 = automatic): for the
/// 1024-wide matmuls, and for the wider ones.
pub static KZ_OVERRIDE: [std::sync::atomic::AtomicU32; 2] = [std::sync::atomic::AtomicU32::new(0), std::sync::atomic::AtomicU32::new(0)];

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

pub struct G2p {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    kernels: Vec<GKernel>,
    pipes: Vec<wgpu::ComputePipeline>,
    pipe_index: HashMap<GKernel, usize>,
    w: Option<Weights>,
    shapes: Vec<Shape>,
    film_cache: HashMap<(u32, u32), Vec<f32>>,
    clock: u64,
    pub startup: Startup,
}

/// The weight of the linear layer whose bias is `bias`: Add(MatMul(x, W), bias).
fn linear_weight(g: &Graph, bias: &str) -> Result<String, String> {
    let add = g
        .nodes
        .iter()
        .find(|n| n.op == "Add" && n.inputs.get(1).map(|s| s.as_str()) == Some(bias))
        .ok_or_else(|| format!("graph: no Add of {bias}"))?;
    let mm = g
        .nodes
        .iter()
        .find(|n| n.op == "MatMul" && n.outputs.first() == add.inputs.first())
        .ok_or_else(|| format!("graph: {bias} does not follow a MatMul"))?;
    mm.inputs.get(1).cloned().ok_or_else(|| format!("{} has no weight", mm.name))
}

fn dot64(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

impl G2p {
    pub fn open(device: wgpu::Device, queue: wgpu::Queue) -> G2p {
        G2p {
            device,
            queue,
            kernels: Vec::new(),
            pipes: Vec::new(),
            pipe_index: HashMap::new(),
            w: None,
            shapes: Vec::new(),
            film_cache: HashMap::new(),
            clock: 0,
            startup: Startup::default(),
        }
    }

    /// Every matmul (M, N, K, epilogue) of a run over m padded rows.
    fn matmul_shapes(m: u32) -> [(u32, u32, u32, Epi); 6] {
        let (h, ff) = (H as u32, FF as u32);
        [
            (m, 3 * h, h, Epi::Bias),
            (m, h, h, Epi::BiasRes),
            (m, ff, h, Epi::BiasGeluX),
            (m, h, ff, Epi::BiasRes),
            (m, h, h, Epi::BiasGeluX),
            (m, h, h, Epi::BiasRes),
        ]
    }

    /// K slices across workgroups for an M x N x K matmul.
    pub fn kz(m: u32, n: u32, k: u32) -> u32 {
        let o = KZ_OVERRIDE[(n > 1024) as usize].load(std::sync::atomic::Ordering::Relaxed);
        if o > 0 {
            return o;
        }
        let _ = k;
        // Short inputs stream the weights: more workgroups per weight strip.
        // From 64 rows the 1024-wide matmuls (out projection, FFN down, the
        // tail) still have too few column tiles to fill the GPU alone.
        match (m <= 32, n <= 1024, Self::small_tile(m)) {
            (true, true, _) => 8,
            (true, false, _) => 2,
            (false, true, _) => 4,
            (false, false, true) => 2,
            (false, false, false) => 1,
        }
    }

    /// 32-row tiles up to 64 rows and at 96 (three of them instead of two
    /// half-empty 64-row tiles); 64-row tiles otherwise.
    fn small_tile(m: u32) -> bool {
        m <= 64 || m == 96
    }

    /// The kernels of one matmul.
    fn matmul_kernels(m: u32, n: u32, k: u32, e: Epi) -> Vec<GKernel> {
        let (t, kz) = (Self::tile(m, n, k), Self::kz(m, n, k));
        if kz > 1 && k % (t.bk * t.ks * kz) == 0 {
            vec![GKernel::Matmul(t, Epi::Partial, kz), GKernel::Reduce(kz, e)]
        } else {
            vec![GKernel::Matmul(t, e, 1)]
        }
    }

    /// The tile for an M x N x K matmul. Swept in Chromium on an Apple M5
    /// (engine-rs/harness/g2p.html ?sweep=, every tile x K slices at 32-224
    /// rows); see docs/MODELS.md, "Rust engine (Hebrew G2P)".
    pub fn tile(m: u32, n: u32, k: u32) -> Tile {
        let fits = |t: &Tile| k % (t.bk * t.ks) == 0 && (t.ks == 1 || t.tm % 4 == 0) && t.tm % 4 == 0;
        let i = TILE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if i < TILES.len() && fits(&TILES[i]) {
            return TILES[i];
        }
        let _ = n;
        let t = if Self::small_tile(m) {
            TILES[7] // 32 x 64, 4 x 8
        } else {
            TILES[5] // 64 x 64, 8 x 8
        };
        if fits(&t) { t } else { TILES[0] }
    }

    pub fn kernels_to_compile() -> Vec<GKernel> {
        let mut ks: Vec<GKernel> = Vec::new();
        for m in (ROW_STEP..=MAX_POS as u32).step_by(ROW_STEP as usize) {
            for (mm, n, k, e) in Self::matmul_shapes(m) {
                for kern in Self::matmul_kernels(mm, n, k, e) {
                    if !ks.contains(&kern) {
                        ks.push(kern);
                    }
                }
            }
        }
        ks.extend([GKernel::Ln(LnMode::Embed), GKernel::Ln(LnMode::Plain), GKernel::Ln(LnMode::Film), GKernel::Attn, GKernel::Heads, GKernel::Reblock]);
        ks
    }

    pub fn kernel_sources() -> Vec<String> {
        Self::kernels_to_compile().iter().map(|k| k.source()).collect()
    }

    pub fn compile(&mut self) {
        let t0 = now_ms();
        for k in Self::kernels_to_compile() {
            self.pipe(k);
        }
        self.startup.pipelines_ms += now_ms() - t0;
    }

    fn pipe(&mut self, k: GKernel) -> usize {
        if let Some(&i) = self.pipe_index.get(&k) {
            return i;
        }
        #[cfg(not(target_arch = "wasm32"))]
        if std::env::var("G2P_DUMP").is_ok() {
            eprintln!("=== {k:?}\n{}", k.source());
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(k.source().into()) });
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
        self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (floats.max(4) * 4) as u64, usage: STORAGE, mapped_at_creation: false })
    }

    fn uniform(&self, words: &[u32], keep: &mut Vec<wgpu::Buffer>) -> wgpu::Buffer {
        let b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(words),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        keep.push(b.clone());
        b
    }

    fn op(&self, pipe: usize, bufs: &[&wgpu::Buffer], wg: (u32, u32)) -> Op {
        self.op3(pipe, bufs, (wg.0, wg.1, 1))
    }

    fn op3(&self, pipe: usize, bufs: &[&wgpu::Buffer], wg: (u32, u32, u32)) -> Op {
        let entries: Vec<_> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &self.pipes[pipe].get_bind_group_layout(0), entries: &entries });
        Op { pipe, bind, wg }
    }

    /// Reads model.onnx's weights and uploads them: matmul weights through a
    /// scratch buffer and the reblock kernel (one submit per weight), the rest
    /// straight into their buffers. Needs the pipelines (`compile`) first.
    pub fn load_weights<S: Upload + ?Sized>(&mut self, src: &S) -> Result<(), String> {
        if self.pipes.is_empty() {
            self.compile();
        }
        let t0 = now_ms();
        let g = onnx::parse(src)?;
        self.startup.parse_ms = now_ms() - t0;
        let t0 = now_ms();
        let reblock = self.pipe(GKernel::Reblock);
        let mut all: Vec<wgpu::Buffer> = Vec::new();
        let scratch = self.buffer(H * FF);
        let mut tmp: Vec<wgpu::Buffer> = Vec::new();

        let init = |name: &str, shape: &[usize]| -> Result<&onnx::Tensor, String> {
            let t = g.inits.get(name).ok_or_else(|| format!("no initializer {name}"))?;
            if t.dims != shape {
                return Err(format!("{name}: dims {:?}, expected {:?}", t.dims, shape));
            }
            if t.data_type != 1 {
                return Err(format!("{name}: data_type {}, expected float32", t.data_type));
            }
            Ok(t)
        };
        // A tensor as stored, straight into a new buffer.
        let raw = |this: &Self, all: &mut Vec<wgpu::Buffer>, name: &str, shape: &[usize]| -> Result<wgpu::Buffer, String> {
            let t = init(name, shape)?;
            let b = this.buffer(t.numel());
            match t.raw {
                Some(r) => src.upload(&this.queue, &b, r),
                None => this.queue.write_buffer(&b, 0, bytemuck::cast_slice(&t.inline)),
            }
            all.push(b.clone());
            Ok(b)
        };
        // Matmul weights [K][N] -> blocked, into `dst` at column n_off.
        let blocked = |this: &Self, tmp: &mut Vec<wgpu::Buffer>, dst: &wgpu::Buffer, name: &str, k: usize, n: usize, n_off: usize| -> Result<(), String> {
            let t = init(name, &[k, n])?;
            let r = t.raw.ok_or_else(|| format!("{name}: no raw data"))?;
            src.upload(&this.queue, &scratch, r);
            let u = this.uniform(&[k as u32, n as u32, n_off as u32, 0], tmp);
            let op = this.op(reblock, &[&scratch, dst, &u], (((k * n / 4) as u32).div_ceil(256), 1));
            let mut enc = this.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&this.pipes[op.pipe]);
                pass.set_bind_group(0, &op.bind, &[]);
                pass.dispatch_workgroups(op.wg.0, op.wg.1, op.wg.2);
            }
            this.queue.submit([enc.finish()]);
            Ok(())
        };
        let read = |name: &str, shape: &[usize]| -> Result<Vec<f32>, String> { onnx::read_f32(src, init(name, shape)?) };

        let e = "model.encoder.embeddings";
        let word_t = g.inits.get(&format!("{e}.word_embeddings.weight")).ok_or("no word embeddings")?;
        let vocab = word_t.dims[0];
        let word = raw(self, &mut all, &format!("{e}.word_embeddings.weight"), &[vocab, H])?;
        let tok = {
            let t = read(&format!("{e}.token_type_embeddings.weight"), &[2, H])?;
            let b = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&t[..H]), usage: STORAGE });
            all.push(b.clone());
            b
        };
        let pos = raw(self, &mut all, &format!("{e}.position_embeddings.weight"), &[MAX_POS, H])?;
        let emb_g = raw(self, &mut all, &format!("{e}.LayerNorm.weight"), &[H])?;
        let emb_b = raw(self, &mut all, &format!("{e}.LayerNorm.bias"), &[H])?;
        let mut layers = Vec::new();
        for i in 0..LAYERS {
            let p = |s: &str| format!("model.encoder.encoder.layer.{i}.{s}");
            let wqkv = self.buffer(H * 3 * H);
            all.push(wqkv.clone());
            let mut bq = Vec::with_capacity(3 * H);
            for (j, which) in ["query", "key", "value"].iter().enumerate() {
                let bias = p(&format!("attention.self.{which}.bias"));
                blocked(self, &mut tmp, &wqkv, &linear_weight(&g, &bias)?, H, H, j * H)?;
                bq.extend(read(&bias, &[H])?);
            }
            let bqkv = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&bq), usage: STORAGE });
            all.push(bqkv.clone());
            let mat = |this: &Self, all: &mut Vec<wgpu::Buffer>, tmp: &mut Vec<wgpu::Buffer>, bias: &str, k: usize, n: usize| -> Result<wgpu::Buffer, String> {
                let b = this.buffer(k * n);
                blocked(this, tmp, &b, &linear_weight(&g, bias)?, k, n, 0)?;
                all.push(b.clone());
                Ok(b)
            };
            let wo = mat(self, &mut all, &mut tmp, &p("attention.output.dense.bias"), H, H)?;
            let w1 = mat(self, &mut all, &mut tmp, &p("intermediate.dense.bias"), H, FF)?;
            let w2 = mat(self, &mut all, &mut tmp, &p("output.dense.bias"), FF, H)?;
            layers.push(LayerW {
                wqkv,
                bqkv,
                wo,
                bo: raw(self, &mut all, &p("attention.output.dense.bias"), &[H])?,
                ln1_g: raw(self, &mut all, &p("attention.output.LayerNorm.weight"), &[H])?,
                ln1_b: raw(self, &mut all, &p("attention.output.LayerNorm.bias"), &[H])?,
                w1,
                b1: raw(self, &mut all, &p("intermediate.dense.bias"), &[FF])?,
                w2,
                b2: raw(self, &mut all, &p("output.dense.bias"), &[H])?,
                ln2_g: raw(self, &mut all, &p("output.LayerNorm.weight"), &[H])?,
                ln2_b: raw(self, &mut all, &p("output.LayerNorm.bias"), &[H])?,
            });
            // Let the GPU take each layer before the next is queued (bounded staging memory).
            #[cfg(not(target_arch = "wasm32"))]
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        }
        let mat = |this: &Self, all: &mut Vec<wgpu::Buffer>, tmp: &mut Vec<wgpu::Buffer>, bias: &str, k: usize, n: usize| -> Result<wgpu::Buffer, String> {
            let b = this.buffer(k * n);
            blocked(this, tmp, &b, &linear_weight(&g, bias)?, k, n, 0)?;
            all.push(b.clone());
            Ok(b)
        };
        let m0 = mat(self, &mut all, &mut tmp, "model.mlp.0.bias", H, H)?;
        let m3 = mat(self, &mut all, &mut tmp, "model.mlp.3.bias", H, H)?;
        let lnf_g = raw(self, &mut all, "model.layer_norm.weight", &[H])?;
        let lnf_b = raw(self, &mut all, "model.layer_norm.bias", &[H])?;
        let m0b = raw(self, &mut all, "model.mlp.0.bias", &[H])?;
        let m3b = raw(self, &mut all, "model.mlp.3.bias", &[H])?;
        let wc = raw(self, &mut all, &linear_weight(&g, "model.consonant_head.bias")?, &[H, 25])?;
        let wv = raw(self, &mut all, &linear_weight(&g, "model.vowel_head.bias")?, &[H + 25, 6])?;
        let ws = raw(self, &mut all, &linear_weight(&g, "model.stress_head.bias")?, &[H + 31, 2])?;
        let mut hbv = read("model.consonant_head.bias", &[25])?;
        hbv.extend(read("model.vowel_head.bias", &[6])?);
        hbv.extend(read("model.stress_head.bias", &[2])?);
        let hb = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&hbv), usage: STORAGE });
        all.push(hb.clone());
        let st = g.inits.get("model.speaker_emb.weight").ok_or("no speaker embedding")?;
        let speakers = st.dims[0];
        let spk = read("model.speaker_emb.weight", &[speakers, 32])?;
        let tspk = read("model.target_speaker_emb.weight", &[speakers, 32])?;
        let film = read("model.film_projection.weight", &[2 * H, 64])?;
        // The graph's attention scale, which the kernel has as a constant.
        let sc = g.inits.get("val_137").map(|t| onnx::read_f32(src, t)).transpose()?.and_then(|v| v.first().copied());
        if let Some(sc) = sc {
            if sc != 0.35355338f32 {
                return Err(format!("graph: attention scale {sc}, expected 64^-0.25"));
            }
        }
        self.startup.upload_ms = now_ms() - t0;
        self.startup.weight_bytes = all.iter().map(|b| b.size()).sum();
        self.drop_shapes();
        self.film_cache.clear();
        self.w = Some(Weights {
            word,
            tok,
            pos,
            emb_g,
            emb_b,
            layers,
            lnf_g,
            lnf_b,
            m0,
            m0b,
            m3,
            m3b,
            wc,
            wv,
            ws,
            hb,
            spk,
            tspk,
            film,
            vocab,
            speakers,
            all,
        });
        // destroy() takes effect after the submitted reblocks complete.
        scratch.destroy();
        for b in tmp {
            b.destroy();
        }
        Ok(())
    }

    /// Bytes of weights on the GPU.
    pub fn weight_bytes(&self) -> u64 {
        self.w.as_ref().map_or(0, |w| w.all.iter().map(|b| b.size()).sum())
    }

    pub fn vocab(&self) -> usize {
        self.w.as_ref().map_or(0, |w| w.vocab)
    }

    fn film(&mut self, s: u32, t: u32) -> Result<Vec<f32>, String> {
        let w = self.w.as_ref().ok_or("load first")?;
        if s as usize >= w.speakers || t as usize >= w.speakers {
            return Err(format!("speaker ids {s}, {t}: the model has {}", w.speakers));
        }
        if let Some(v) = self.film_cache.get(&(s, t)) {
            return Ok(v.clone());
        }
        let mut cat = Vec::with_capacity(64);
        cat.extend_from_slice(&w.spk[s as usize * 32..(s as usize + 1) * 32]);
        cat.extend_from_slice(&w.tspk[t as usize * 32..(t as usize + 1) * 32]);
        let v: Vec<f32> = (0..2 * H).map(|j| dot64(&w.film[j * 64..(j + 1) * 64], &cat) as f32).collect();
        self.film_cache.insert((s, t), v.clone());
        Ok(v)
    }

    /// Drops every kept shape (after a tuning override, so the next run rebuilds).
    pub fn drop_shapes(&mut self) {
        for s in self.shapes.drain(..) {
            for b in &s.keep {
                b.destroy();
            }
        }
    }

    /// Buffers, bind groups and dispatches for m padded rows.
    fn build_shape(&mut self, m: u32) -> Result<Shape, String> {
        for (mm, n, k, e) in Self::matmul_shapes(m) {
            for kern in Self::matmul_kernels(mm, n, k, e) {
                self.pipe(kern);
            }
        }
        let w = self.w.as_ref().ok_or("load first")?;
        let mu = m as usize;
        let mut keep = Vec::new();
        let mk = |this: &Self, keep: &mut Vec<wgpu::Buffer>, floats: usize| {
            let b = this.buffer(floats);
            keep.push(b.clone());
            b
        };
        let ids = mk(self, &mut keep, mu);
        let film = mk(self, &mut keep, 2 * H);
        let x = mk(self, &mut keep, mu * H);
        let y = mk(self, &mut keep, mu * H);
        let att = mk(self, &mut keep, mu * H);
        let qkv = mk(self, &mut keep, mu * 3 * H);
        let hid = mk(self, &mut keep, mu * FF);
        let out = mk(self, &mut keep, mu * OUT);
        let len = self.uniform(&[m, m, 0, 0], &mut keep);
        let kzmax = Self::matmul_shapes(m).iter().map(|&(mm, n, k, e)| if Self::matmul_kernels(mm, n, k, e).len() > 1 { Self::kz(mm, n, k) as usize * n as usize } else { 0 }).max().unwrap_or(0);
        let part = mk(self, &mut keep, mu * kzmax);
        let mut ops = Vec::new();
        let pi = |k: GKernel| self.pipe_index[&k];
        let mm = |this: &Self, keep: &mut Vec<wgpu::Buffer>, ops: &mut Vec<Op>, n: u32, k: u32, e: Epi, a: &wgpu::Buffer, wt: &wgpu::Buffer, c: &wgpu::Buffer, extra: &[&wgpu::Buffer]| {
            let tile = Self::tile(m, n, k);
            let u = this.uniform(&[m, n, k, 0, 0, 0, 0, 0], keep);
            let kerns = Self::matmul_kernels(m, n, k, e);
            let wg = (m.div_ceil(tile.bm), n.div_ceil(tile.bn));
            if kerns.len() == 1 {
                let mut bufs = vec![a, wt, c, &u];
                bufs.extend_from_slice(extra);
                ops.push(this.op(this.pipe_index[&kerns[0]], &bufs, wg));
            } else {
                let kz = Self::kz(m, n, k);
                ops.push(this.op3(this.pipe_index[&kerns[0]], &[a, wt, &part, &u], (wg.0, wg.1, kz)));
                let mut bufs = vec![&part, c, &u];
                bufs.extend_from_slice(extra);
                ops.push(this.op(this.pipe_index[&kerns[1]], &bufs, ((m * n / 4).div_ceil(256), 1)));
            }
        };
        let (h, ff) = (H as u32, FF as u32);
        ops.push(self.op(pi(GKernel::Ln(LnMode::Embed)), &[&ids, &x, &w.emb_g, &w.emb_b, &w.word, &w.tok, &w.pos], (m, 1)));
        for l in &w.layers {
            mm(self, &mut keep, &mut ops, 3 * h, h, Epi::Bias, &x, &l.wqkv, &qkv, &[&l.bqkv]);
            ops.push(self.op(pi(GKernel::Attn), &[&qkv, &att, &len], (m.div_ceil(g2p_shaders::QB), 16)));
            mm(self, &mut keep, &mut ops, h, h, Epi::BiasRes, &att, &l.wo, &y, &[&l.bo, &x]);
            ops.push(self.op(pi(GKernel::Ln(LnMode::Plain)), &[&y, &x, &l.ln1_g, &l.ln1_b], (m, 1)));
            mm(self, &mut keep, &mut ops, ff, h, Epi::BiasGeluX, &x, &l.w1, &hid, &[&l.b1]);
            mm(self, &mut keep, &mut ops, h, ff, Epi::BiasRes, &hid, &l.w2, &y, &[&l.b2, &x]);
            ops.push(self.op(pi(GKernel::Ln(LnMode::Plain)), &[&y, &x, &l.ln2_g, &l.ln2_b], (m, 1)));
        }
        // y = FiLM(x) (kept for the MLP's residual), att = LN(y)
        ops.push(self.op(pi(GKernel::Ln(LnMode::Film)), &[&x, &att, &w.lnf_g, &w.lnf_b, &film, &y], (m, 1)));
        mm(self, &mut keep, &mut ops, h, h, Epi::BiasGeluX, &att, &w.m0, &qkv, &[&w.m0b]);
        mm(self, &mut keep, &mut ops, h, h, Epi::BiasRes, &qkv, &w.m3, &x, &[&w.m3b, &y]);
        ops.push(self.op(pi(GKernel::Heads), &[&x, &w.wc, &w.wv, &w.ws, &w.hb, &out], (m, 1)));
        Ok(Shape { m, ids, len, film, out, ops, keep, used: 0 })
    }

    fn shape_for(&mut self, rows: u32) -> Result<usize, String> {
        let m = rows.div_ceil(ROW_STEP) * ROW_STEP;
        self.clock += 1;
        if let Some(i) = self.shapes.iter().position(|s| s.m == m) {
            self.shapes[i].used = self.clock;
            return Ok(i);
        }
        if self.shapes.len() >= SHAPES_KEPT {
            let (i, _) = self.shapes.iter().enumerate().min_by_key(|(_, s)| s.used).unwrap();
            let s = self.shapes.swap_remove(i);
            for b in &s.keep {
                b.destroy();
            }
        }
        let mut s = self.build_shape(m)?;
        s.used = self.clock;
        self.shapes.push(s);
        Ok(self.shapes.len() - 1)
    }

    /// Queues one run over token ids (CLS, characters, SEP) for speaker ids
    /// (s, t) and the copy of its logits into a staging buffer: [S, 33] once
    /// mapped (`map_read`). One submit.
    pub fn run(&mut self, ids: &[u32], s: u32, t: u32) -> Result<wgpu::Buffer, String> {
        let n = ids.len();
        if n == 0 || n > MAX_POS {
            return Err(format!("run: {n} tokens (1 to {MAX_POS})"));
        }
        let vocab = self.vocab() as u32;
        if let Some(bad) = ids.iter().find(|&&i| i >= vocab) {
            return Err(format!("run: token id {bad} out of the vocabulary ({vocab})"));
        }
        let film = self.film(s, t)?;
        let si = self.shape_for(n as u32)?;
        let sh = &self.shapes[si];
        let mut padded = vec![0u32; sh.m as usize];
        padded[..n].copy_from_slice(ids);
        self.queue.write_buffer(&sh.ids, 0, bytemuck::cast_slice(&padded));
        self.queue.write_buffer(&sh.len, 0, bytemuck::cast_slice(&[n as u32, sh.m, 0, 0]));
        self.queue.write_buffer(&sh.film, 0, bytemuck::cast_slice(&film));
        let size = (n * OUT * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("g2p-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            let mut last = usize::MAX;
            for op in &sh.ops {
                if op.pipe != last {
                    pass.set_pipeline(&self.pipes[op.pipe]);
                    last = op.pipe;
                }
                pass.set_bind_group(0, &op.bind, &[]);
                pass.dispatch_workgroups(op.wg.0, op.wg.1, op.wg.2);
            }
        }
        enc.copy_buffer_to_buffer(&sh.out, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        Ok(staging)
    }

    /// One run and its logits, [S, 33] (consonant 25, vowel 6, stress 2 per token).
    pub async fn logits(&mut self, ids: &[u32], s: u32, t: u32) -> Result<Vec<f32>, String> {
        let st = self.run(ids, s, t)?;
        map_read(self.device.clone(), st).await
    }

    pub fn dispatches_per_run(&self) -> usize {
        self.shapes.first().map_or(0, |s| s.ops.len())
    }

    /// GPU memory of the kept shapes (activations, ids, uniforms).
    pub fn shape_bytes(&self) -> u64 {
        self.shapes.iter().map(|s| s.keep.iter().map(|b| b.size()).sum::<u64>()).sum()
    }

    pub fn destroy(&mut self) {
        self.drop_shapes();
        if let Some(w) = self.w.take() {
            for b in &w.all {
                b.destroy();
            }
        }
        self.device.destroy();
    }

    /// Per-kernel GPU time of one run over `ids` (timestamp queries, one pass
    /// per dispatch): (kernel, ms, dispatches).
    pub fn profile(&mut self, ids: &[u32]) -> Option<ProfileJob> {
        if !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let st = self.run(ids, 0, 0).ok()?;
        st.destroy();
        let si = self.shape_for(ids.len() as u32).ok()?;
        let sh = &self.shapes[si];
        let n = sh.ops.len() as u32;
        let qs = self.device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * n });
        let res = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16 * n as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        for (i, op) in sh.ops.iter().enumerate() {
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
            pass.dispatch_workgroups(op.wg.0, op.wg.1, op.wg.2);
        }
        enc.resolve_query_set(&qs, 0..2 * n, &res, 0);
        self.queue.submit([enc.finish()]);
        Some(ProfileJob {
            device: self.device.clone(),
            queue: self.queue.clone(),
            res,
            names: sh.ops.iter().map(|op| format!("{:?}", self.kernels[op.pipe])).collect(),
            period: self.queue.get_timestamp_period() as f64,
        })
    }
}

pub use crate::engine::ProfileJob;
