//! WGSL kernels, specialized in Rust by string substitution (WGSL has no
//! templates). Every elementwise op of the graph is folded into one of four
//! kernels, in the graph's own order of operations:
//!
//!   matmul    C = A x W, tiled, f32 accumulation, with a per-use epilogue
//!             (bias, erf GELU, ConvNeXt gamma + residual + mask (+ time),
//!             attention output + residual, the Euler step of proj_out)
//!   rownorm   depthwise conv (edge pad) + mask + LayerNorm, or residual
//!             LayerNorm + mask, one workgroup per (batch, frame) row
//!   attention q (rotary) . k / 16, softmax over the keys, x mask, . v;
//!             one workgroup per (head, row)
//!   rope, blend   per chunk / per step helpers (not part of a run)
//!
//! Activations are [B*T, C] row-major (channels last), so LayerNorm reads
//! contiguous rows and every 1x1 conv is a plain matmul.

/// Matmul tile: BM x BN outputs per workgroup, TM rows x TN (4 or 8) columns per thread, BK deep.
/// Every tile with the same `ks` adds the K products in the same order, so
/// tiles of one `ks` give bit-identical results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub bm: u32,
    pub bn: u32,
    pub tm: u32,
    pub tn: u32,
    pub bk: u32,
    /// K split across `ks` slices of the workgroup (lid.z), summed in shared
    /// memory before the epilogue: more threads when there are few rows.
    pub ks: u32,
}

pub const TILES: [Tile; 14] = [
    Tile { bm: 64, bn: 64, tm: 4, tn: 4, bk: 16, ks: 1 },
    Tile { bm: 32, bn: 64, tm: 4, tn: 4, bk: 16, ks: 1 },
    Tile { bm: 128, bn: 64, tm: 8, tn: 4, bk: 16, ks: 1 },
    Tile { bm: 64, bn: 128, tm: 4, tn: 8, bk: 16, ks: 1 },
    Tile { bm: 32, bn: 32, tm: 4, tn: 4, bk: 16, ks: 1 },
    Tile { bm: 64, bn: 64, tm: 8, tn: 8, bk: 16, ks: 1 },
    Tile { bm: 16, bn: 64, tm: 4, tn: 4, bk: 16, ks: 1 },
    Tile { bm: 32, bn: 64, tm: 4, tn: 8, bk: 16, ks: 1 },
    Tile { bm: 64, bn: 64, tm: 8, tn: 8, bk: 32, ks: 1 },
    Tile { bm: 32, bn: 32, tm: 4, tn: 4, bk: 32, ks: 1 },
    Tile { bm: 32, bn: 32, tm: 4, tn: 4, bk: 16, ks: 4 },
    Tile { bm: 32, bn: 64, tm: 4, tn: 4, bk: 16, ks: 2 },
    Tile { bm: 64, bn: 32, tm: 4, tn: 4, bk: 16, ks: 2 },
    Tile { bm: 32, bn: 64, tm: 4, tn: 4, bk: 32, ks: 1 },
];

/// How A is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ALoad {
    /// A[m, k] = A[m * K + k]
    Rows,
    /// channels first per batch, [B, K, T]: A[m = b*T + t, k] = A[(b*K + k)*T + t]
    /// (the latent and text_emb as the graph takes them)
    ChanFirst,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Epi {
    /// C = acc + bias
    Bias,
    /// C = gelu_erf(acc + bias), the graph's Div/Erf/Add/Mul/Mul
    BiasGelu,
    /// C = tanh(acc + bias) (the style keys, at load)
    BiasTanh,
    /// C (in place, holds the block input r) = (gamma * (acc + bias) + r) * mask,
    /// then with `time`: (that + tb[b]) * mask
    CnOut { time: bool },
    /// C = R + (acc + bias) * mask (attention output + residual, before LayerNorm)
    AttnOut,
    /// C = acc * mask (proj_in, no bias)
    ProjIn,
    /// out[b, n, t] = acc * mask (proj_out: the row's velocity, channels first;
    /// the CFG blend then takes the Euler step)
    ProjOut,
    /// C = z * (0.5 * (erf(z / sqrt 2) + 1)), z = acc + bias: erf GELU in the
    /// order RenikudPlus's graph has it (Div, Erf, Add, Mul 0.5, Mul z)
    BiasGeluX,
    /// C = (acc + bias) + R (a dense layer and its residual, before LayerNorm)
    BiasRes,
    /// C[wid.z, m, n] = acc: one of several K slices across workgroups,
    /// summed (and given the real epilogue) by a reduce pass
    Partial,
}

const ERF: &str = r#"
// erf for f32: Numerical Recipes' erfc Chebyshev fit (fractional error < 1.2e-7).
fn erf_(x: f32) -> f32 {
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))));
    return select(r - 1.0, 1.0 - r, x >= 0.0);
}
"#;

pub fn matmul(tile: Tile, a: ALoad, epi: Epi) -> String {
    matmul_ex(tile, a, epi, false)
}

/// `matmul` with `m_fast`: workgroup x walks the row tiles and y the column
/// tiles, so the workgroups that share a strip of W run next to each other
/// and read it from cache (RenikudPlus: few rows, 1.2 GB of weights).
pub fn matmul_ex(tile: Tile, a: ALoad, epi: Epi, m_fast: bool) -> String {
    let (extra, body) = epilogue(epi);
    // Every per-thread value is a named scalar or vec4, never a dynamically
    // indexed private array (which drivers keep in slow thread memory): the
    // accumulators are unrolled here in Rust. Each thread owns TM consecutive
    // rows x TN columns (one vec4, or two vec4 half a tile apart); W is read
    // as vec4 along n (every N here is a multiple of 4), row-major A as vec4
    // along k. With TM a multiple of 4 the A tile is read as vec4 too.
    let (tm, tn) = (tile.tm, tile.tn);
    let tx = tile.bn / tn;
    let ty = tile.bm / tm;
    let nv = tn / 4; // vec4 columns per thread
    let av = tm % 4 == 0;
    let xyzw = ["x", "y", "z", "w"];
    let mut decl = String::new();
    let mut inner = String::new();
    let mut epi_code = String::new();
    for j in 0..nv {
        inner += &format!("let bv{j} = Bs[kk * (BN / 4u) + lid.x + {}u];\n            ", j * tx);
    }
    if av {
        for q in 0..tm / 4 {
            inner += &format!("let a{q} = As[kk * (BM / 4u) + lid.y * {}u + {q}u];\n            ", tm / 4);
        }
    }
    for r in 0..tm {
        let ar = if av { format!("a{}.{}", r / 4, xyzw[(r % 4) as usize]) } else { format!("As[kk * BM + lid.y * {tm}u + {r}u]") };
        for j in 0..nv {
            decl += &format!("var acc{r}_{j} = vec4<f32>();\n    ");
            inner += &format!("acc{r}_{j} = fma(vec4<f32>({ar}), bv{j}, acc{r}_{j});\n            ");
            for (c, comp) in xyzw.iter().enumerate() {
                epi_code += &format!(
                    "{{ let m = mr + {r}u; let n = n0 + (lid.x + {}u) * 4u + {c}u; if (m < p.M && n < p.N) {{ let v = acc{r}_{j}.{comp}; {body} }} }}\n    ",
                    j * tx
                );
            }
        }
    }
    // The A tile in shared memory is [k][m]. With TM a multiple of 4 it is
    // stored as vec4 of 4 consecutive rows, each vec4 written whole by one
    // thread (component writes from different threads would race).
    let (a_decl, load_a) = match (a, av) {
        (ALoad::Rows, true) => (
            "@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;",
            "for (var i = tid; i < (BM / 4u) * (BK / 4u); i += NT) {
            let m4 = i / (BK / 4u); let k4 = i % (BK / 4u); let m = m0 + m4 * 4u;
            let r0 = select(vec4<f32>(), A[(min(m, p.M - 1u) * p.K + k0) / 4u + k4], m < p.M);
            let r1 = select(vec4<f32>(), A[(min(m + 1u, p.M - 1u) * p.K + k0) / 4u + k4], m + 1u < p.M);
            let r2 = select(vec4<f32>(), A[(min(m + 2u, p.M - 1u) * p.K + k0) / 4u + k4], m + 2u < p.M);
            let r3 = select(vec4<f32>(), A[(min(m + 3u, p.M - 1u) * p.K + k0) / 4u + k4], m + 3u < p.M);
            As[(k4 * 4u) * (BM / 4u) + m4] = vec4<f32>(r0.x, r1.x, r2.x, r3.x);
            As[(k4 * 4u + 1u) * (BM / 4u) + m4] = vec4<f32>(r0.y, r1.y, r2.y, r3.y);
            As[(k4 * 4u + 2u) * (BM / 4u) + m4] = vec4<f32>(r0.z, r1.z, r2.z, r3.z);
            As[(k4 * 4u + 3u) * (BM / 4u) + m4] = vec4<f32>(r0.w, r1.w, r2.w, r3.w);
        }",
        ),
        (ALoad::Rows, false) => (
            "@group(0) @binding(0) var<storage, read> A: array<vec4<f32>>;",
            "for (var i = tid; i < BM * BK / 4u; i += NT) {
            let mm = i / (BK / 4u); let k4 = i % (BK / 4u); let m = m0 + mm;
            let v = select(vec4<f32>(), A[(min(m, p.M - 1u) * p.K + k0) / 4u + k4], m < p.M);
            As[(k4 * 4u) * BM + mm] = v.x;
            As[(k4 * 4u + 1u) * BM + mm] = v.y;
            As[(k4 * 4u + 2u) * BM + mm] = v.z;
            As[(k4 * 4u + 3u) * BM + mm] = v.w;
        }",
        ),
        // Channels first: consecutive threads walk m (contiguous along t).
        (ALoad::ChanFirst, true) => (
            "@group(0) @binding(0) var<storage, read> A: array<f32>;
fn acf(m: u32, k: u32) -> f32 {
    let mc = min(m, p.M - 1u);
    return select(0.0, A[((mc / p.T) * p.K + k) * p.T + (mc % p.T)], m < p.M);
}",
            "for (var i = tid; i < BK * (BM / 4u); i += NT) {
            let kk = i / (BM / 4u); let m4 = i % (BM / 4u); let m = m0 + m4 * 4u;
            As[i] = vec4<f32>(acf(m, k0 + kk), acf(m + 1u, k0 + kk), acf(m + 2u, k0 + kk), acf(m + 3u, k0 + kk));
        }",
        ),
        (ALoad::ChanFirst, false) => (
            "@group(0) @binding(0) var<storage, read> A: array<f32>;",
            "for (var i = tid; i < BM * BK; i += NT) {
            let kk = i / BM; let mm = i % BM; let m = min(m0 + mm, p.M - 1u);
            As[kk * BM + mm] = select(0.0, A[((m / p.T) * p.K + k0 + kk) * p.T + (m % p.T)], m0 + mm < p.M);
        }",
        ),
    };
    // Row-major A with vec4 tiles (every large matmul of a run): software
    // pipelined. The next tile's global loads are issued into registers before
    // this tile's multiply-adds, so their latency is hidden even when only a
    // few workgroups fit on each core (short inputs).
    let nt = tx * ty;
    let pipelined = a == ALoad::Rows && av;
    assert!(tile.ks == 1 || pipelined, "split-K needs the pipelined path");
    let ks = tile.ks;
    let nacc = tm * nv; // vec4 accumulators per thread
    let asize = tile.bm * tile.bk / 4 * ks; // vec4 in As
    let mut reduce = String::new();
    if ks > 1 {
        assert!((ks - 1) * nt * nacc <= asize + tile.bk * tile.bn / 4 * ks, "split-K partials do not fit");
        let mut put = String::new();
        let mut add = String::new();
        let mut i = 0;
        for r in 0..tm {
            for j in 0..nv {
                put += &format!("{{ let i = ((lid.z - 1u) * NT + tid) * {nacc}u + {i}u; if (i < {asize}u) {{ As[i] = acc{r}_{j}; }} else {{ Bs[i - {asize}u] = acc{r}_{j}; }} }}\n        ");
                add += &format!("{{ let i = ((s - 1u) * NT + tid) * {nacc}u + {i}u; if (i < {asize}u) {{ acc{r}_{j} += As[i]; }} else {{ acc{r}_{j} += Bs[i - {asize}u]; }} }}\n            ");
                i += 1;
            }
        }
        reduce = format!(
            "
    if (lid.z > 0u) {{
        {put}}}
    workgroupBarrier();
    if (lid.z > 0u) {{ return; }}
    for (var s = 1u; s < KS; s++) {{
            {add}}}"
        );
    }
    let main_loop = if pipelined {
        let ia_tot = (tile.bm / 4) * (tile.bk / 4);
        let ib_tot = tile.bk * tile.bn / 4;
        let (ia, ib) = (ia_tot.div_ceil(nt), ib_tot.div_ceil(nt));
        let (mut regs, mut fetch, mut store) = (String::new(), String::new(), String::new());
        for i in 0..ia {
            regs += &format!("var pa{i}_0 = vec4<f32>(); var pa{i}_1 = vec4<f32>(); var pa{i}_2 = vec4<f32>(); var pa{i}_3 = vec4<f32>();\n    ");
            let mut f = format!("{{ let it = tid + {}u; if (it < {ia_tot}u) {{ let m4 = it / (BK / 4u); let k4 = it % (BK / 4u); let m = m0 + m4 * 4u;\n", i * nt);
            for r in 0..4 {
                f += &format!("                pa{i}_{r} = select(vec4<f32>(), A[(min(m + {r}u, p.M - 1u) * p.K + kn) / 4u + k4], m + {r}u < p.M);\n");
            }
            f += "            } }\n            ";
            fetch += &f;
            let mut st = format!("{{ let it = tid + {}u; if (it < {ia_tot}u) {{ let m4 = it / (BK / 4u); let k4 = it % (BK / 4u);\n", i * nt);
            for (c, comp) in xyzw.iter().enumerate() {
                st += &format!("            As[(k4 * 4u + {c}u) * (BM / 4u) + m4] = vec4<f32>(pa{i}_0.{comp}, pa{i}_1.{comp}, pa{i}_2.{comp}, pa{i}_3.{comp});\n");
            }
            st += "        } }\n        ";
            store += &st;
        }
        for j in 0..ib {
            regs += &format!("var pb{j} = vec4<f32>();\n    ");
            // Consecutive threads walk one 16-column block of W down k (contiguous; see wblock).
            fetch += &format!(
                "{{ let it = tid + {}u; if (it < {ib_tot}u) {{ let kk = (it / 4u) % BK; let n = n0 + ((it / (4u * BK)) * 4u + it % 4u) * 4u;
                pb{j} = select(vec4<f32>(), W[wblock(min(n, p.N - 4u), kn + kk)], n < p.N); }} }}\n            ",
                j * nt
            );
            store += &format!(
                "{{ let it = tid + {}u; if (it < {ib_tot}u) {{ Bs[((it / 4u) % BK) * (BN / 4u) + (it / (4u * BK)) * 4u + it % 4u] = pb{j}; }} }}\n        ",
                j * nt
            );
        }
        // With ks > 1 each slice (lid.z) has its own tiles and its own part of
        // K; the loop counts iterations, not k, so its bound stays uniform.
        let store = store.replace("As[", "As[soa + ").replace("Bs[", "Bs[sob + ");
        let inner = inner.replace("As[", "As[soa + ").replace("Bs[", "Bs[sob + ");
        format!(
            "{regs}
    let soa = lid.z * (BM * BK / 4u);
    let sob = lid.z * (BK * BN / 4u);
    let nit = p.K / (BK * KS);
    let kbeg = lid.z * nit * BK;
    {{ let kn = kbeg;
            {fetch}}}
    for (var it = 0u; it < nit; it++) {{
        let k0 = kbeg + it * BK;
        {store}
        workgroupBarrier();
        if (it + 1u < nit) {{
            let kn = k0 + BK;
            {fetch}}}
        for (var kk = 0u; kk < BK; kk++) {{
            {inner}
        }}
        workgroupBarrier();
    }}{reduce}"
        )
    } else {
        format!(
            "for (var k0 = 0u; k0 < p.K; k0 += BK) {{
        {load_a}
        for (var i = tid; i < BK * BN / 4u; i += NT) {{
            let kk = i / (BN / 4u); let n4 = i % (BN / 4u); let n = n0 + n4 * 4u;
            Bs[i] = select(vec4<f32>(), W[wblock(min(n, p.N - 4u), k0 + kk)], n < p.N);
        }}
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {{
            {inner}
        }}
        workgroupBarrier();
    }}"
        )
    };
    let as_decl = if av { format!("array<vec4<f32>, {}>", tile.bm * tile.bk / 4 * ks) } else { format!("array<f32, {}>", tile.bm * tile.bk) };
    format!(
        r#"
struct P {{ M: u32, N: u32, K: u32, T: u32, tb: u32, p0: u32, p1: u32, p2: u32 }}
{a_decl}
@group(0) @binding(1) var<storage, read> W: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
{extra}
{ERF}
// W is stored in 16-column blocks, [N / 16][K][16], so the strip a workgroup
// reads is contiguous in memory: the vec4 index of W[k][n .. n + 4].
fn wblock(n: u32, k: u32) -> u32 {{ return ((n / 16u) * p.K + k) * 4u + (n % 16u) / 4u; }}
const BM = {bm}u;
const BN = {bn}u;
const BK = {bk}u;
const NT = {nt}u;
const KS = {ks}u;
var<workgroup> As: {as_decl};
var<workgroup> Bs: array<vec4<f32>, {bsz}>;

@compute @workgroup_size({tx}, {ty}, {ks})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tid = lid.y * {tx}u + lid.x;
    {grid}
    {decl}
    {main_loop}
    let mr = m0 + lid.y * {tm}u;
    {epi_code}
}}
"#,
        bm = tile.bm,
        bn = tile.bn,
        bsz = tile.bn * tile.bk / 4 * ks,
        bk = tile.bk,
        grid = if m_fast { "let m0 = wid.x * BM;\n    let n0 = wid.y * BN;" } else { "let m0 = wid.y * BM;\n    let n0 = wid.x * BN;" },
    )
}

fn epilogue(epi: Epi) -> (&'static str, &'static str) {
    match epi {
        Epi::Bias => ("@group(0) @binding(4) var<storage, read> bias: array<f32>;", "C[m * p.N + n] = v + bias[n];"),
        Epi::BiasGelu => (
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;",
            "let z = v + bias[n]; C[m * p.N + n] = (z * (erf_(z / 1.4142135381698608) + 1.0)) * 0.5;",
        ),
        Epi::BiasTanh => ("@group(0) @binding(4) var<storage, read> bias: array<f32>;", "C[m * p.N + n] = tanh(v + bias[n]);"),
        Epi::CnOut { time: false } => (
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> gamma: array<f32>;
@group(0) @binding(6) var<storage, read> mask: array<f32>;",
            "let i = m * p.N + n; C[i] = (gamma[n] * (v + bias[n]) + C[i]) * mask[m];",
        ),
        Epi::CnOut { time: true } => (
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> gamma: array<f32>;
@group(0) @binding(6) var<storage, read> mask: array<f32>;
@group(0) @binding(7) var<storage, read> sb: array<f32>;",
            "let i = m * p.N + n; let o = (gamma[n] * (v + bias[n]) + C[i]) * mask[m];
             C[i] = (o + sb[p.tb + (m / p.T) * p.N + n]) * mask[m];",
        ),
        Epi::AttnOut => (
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;
@group(0) @binding(6) var<storage, read> R: array<f32>;",
            "let i = m * p.N + n; C[i] = R[i] + (v + bias[n]) * mask[m];",
        ),
        Epi::ProjIn => ("@group(0) @binding(4) var<storage, read> mask: array<f32>;", "C[m * p.N + n] = v * mask[m];"),
        Epi::ProjOut => (
            "@group(0) @binding(4) var<storage, read> mask: array<f32>;",
            "let i = ((m / p.T) * p.N + n) * p.T + (m % p.T); C[i] = v * mask[m];",
        ),
        Epi::BiasGeluX => (
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;",
            "let z = v + bias[n]; C[m * p.N + n] = z * (0.5 * (erf_(z / 1.4142135381698608) + 1.0));",
        ),
        Epi::Partial => ("", "C[(wid.z * p.M + m) * p.N + n] = v;"),
        Epi::BiasRes => (
            "@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> R: array<f32>;",
            "let i = m * p.N + n; C[i] = (v + bias[n]) + R[i];",
        ),
    }
}

const REDUCE: &str = r#"
var<workgroup> red: array<f32, 128>;
fn sum128(v: f32, l: u32) -> f32 {
    red[l] = v;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s >>= 1u) {
        if (l < s) { red[l] += red[l + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}
"#;

/// One row of 512 channels per workgroup (128 threads x 4 channels).
/// `dw`: Y = LayerNorm((dwconv_edge(X) + bias) * mask) — a ConvNeXt's first half;
/// otherwise Y = LayerNorm(X) * mask — the residual LayerNorm after attention.
pub fn rownorm(dw: bool) -> String {
    let (extra, post) = if dw {
        (
            "@group(0) @binding(6) var<storage, read> dww: array<f32>;
@group(0) @binding(7) var<storage, read> dwb: array<f32>;
fn load(b: u32, t: u32, m: u32, c: u32) -> f32 {
    // Edge padding at the row's own last frame (rows of a batch of chunks
    // are padded to the longest; a padding frame's value is masked off).
    let last = i32(p.lens[b / 4u][b % 4u]) - 1;
    var a = 0.0;
    for (var k = 0; k < 5; k++) {
        let tt = u32(clamp(i32(t) + (k - 2) * i32(p.dil), 0, last));
        a = fma(dww[c * 5u + u32(k)], X[(b * p.T + tt) * 512u + c], a);
    }
    return (a + dwb[c]) * mask[m];
}",
            "",
        )
    } else {
        ("fn load(b: u32, t: u32, m: u32, c: u32) -> f32 { return X[m * 512u + c]; }", "* mask[m]")
    };
    let lens = if dw { format!(", lens: array<vec4<u32>, {}>", MAX_ROWS / 4) } else { String::new() };
    format!(
        r#"
struct P {{ M: u32, T: u32, dil: u32, p0: u32{lens} }}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<storage, read> g: array<f32>;
@group(0) @binding(5) var<storage, read> beta: array<f32>;
{extra}
{REDUCE}
@compute @workgroup_size(128)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let l = lid.x;
    let m = wid.x;
    let b = m / p.T;
    let t = m % p.T;
    var v0 = load(b, t, m, l);
    var v1 = load(b, t, m, l + 128u);
    var v2 = load(b, t, m, l + 256u);
    var v3 = load(b, t, m, l + 384u);
    let mean = sum128(v0 + v1 + v2 + v3, l) / 512.0;
    v0 -= mean; v1 -= mean; v2 -= mean; v3 -= mean;
    let inv = 1.0 / sqrt(sum128(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3, l) / 512.0 + 9.999999974752427e-07);
    Y[m * 512u + l] = (v0 * inv * g[l] + beta[l]) {post};
    Y[m * 512u + l + 128u] = (v1 * inv * g[l + 128u] + beta[l + 128u]) {post};
    Y[m * 512u + l + 256u] = (v2 * inv * g[l + 256u] + beta[l + 256u]) {post};
    Y[m * 512u + l + 384u] = (v3 * inv * g[l + 384u] + beta[l + 384u]) {post};
}}
"#
    )
}

/// Most rows in one batch: rownorm's uniform holds each row's length.
pub const MAX_ROWS: u32 = 32;

/// Attention in three kernels over a scores scratch buffer S [B, H, T, L]:
/// scores, softmax, and the weighted sum of values. Scores and values each
/// take a block of QB query rows per workgroup, so every key and value row
/// is read once per QB rows instead of once per row (at 380 frames the
/// one-kernel version spent 1.1 ms per text attention re-reading K and V).
/// Text: H = 8, D = 64 (width 512), rotary on q (K comes rotated at prepare),
/// masked keys get no weight. Style: H = 2, D = 128 (width 256), K per CFG row
/// kind ([conditional keys; unconditional keys], row b uses b % 2), no key mask.
pub const QB: u32 = 16;

/// (head dim, heads); the width is their product.
fn attn_dims(text: bool) -> (u32, u32) {
    if text { (64, 8) } else { (128, 2) }
}

/// S[b, h, t, j] = q_t . k_j / 16, or -3e38 for a masked key.
pub fn attn_scores(text: bool) -> String {
    let (d, h) = attn_dims(text);
    let d4 = d / 4;
    let w = d * h;
    let (extra, qload, kbase, keep) = if text {
        (
            "@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read> kmask: array<f32>;",
            // The graph's rotate-half: [x1 c - x2 s, x1 s + x2 c].
            "fn qv(m: u32, h: u32, e: u32) -> f32 {
    let i = e % 32u;
    let c = rope[m * 64u + i];
    let sn = rope[m * 64u + 32u + i];
    let base = m * W + h * D;
    if (e < 32u) { return Q[base + e] * c - Q[base + e + 32u] * sn; }
    return Q[base + e - 32u] * sn + Q[base + e] * c;
}",
            "(b * p.L + j)",
            "kmask[b * p.L + j] != 0.0",
        )
    } else {
        ("", "fn qv(m: u32, h: u32, e: u32) -> f32 { return Q[m * W + h * D + e]; }", "((b % 2u) * p.L + j)", "true")
    };
    let mut acc = String::new();
    let mut body = String::new();
    let mut store = String::new();
    for q in 0..QB {
        acc += &format!("var a{q} = 0.0; ");
        body += &format!("a{q} += dot(qs[{q}u * D4 + e4], kv); ");
        store += &format!("if (t0 + {q}u < p.T) {{ S[(bh * p.T + t0 + {q}u) * p.L + j] = select(-3.0e38, a{q} / 16.0, keep); }}\n        ");
    }
    format!(
        r#"
struct P {{ T: u32, L: u32, p0: u32, p1: u32 }}
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> S: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
{extra}
const D = {d}u;
const D4 = {d4}u;
const H = {h}u;
const W = {w}u;
const QB = {QB}u;
var<workgroup> qs: array<vec4<f32>, {qsz}>;
{qload}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let bh = wid.y;
    let b = bh / H;
    let h = bh % H;
    let t0 = wid.x * QB;
    for (var i = lid.x; i < QB * D4; i += 64u) {{
        let q = i / D4;
        let t = min(t0 + q, p.T - 1u);
        let m = b * p.T + t;
        let e = (i % D4) * 4u;
        qs[i] = vec4<f32>(qv(m, h, e), qv(m, h, e + 1u), qv(m, h, e + 2u), qv(m, h, e + 3u));
    }}
    workgroupBarrier();
    for (var j = lid.x; j < p.L; j += 64u) {{
        let kb = {kbase} * (W / 4u) + h * D4;
        {acc}
        for (var e4 = 0u; e4 < D4; e4++) {{
            let kv = K[kb + e4];
            {body}
        }}
        let keep = {keep};
        {store}
    }}
}}
"#,
        qsz = QB * d4,
    )
}

/// One row of S per workgroup: softmax over the keys (masked keys weigh 0),
/// then zero for a masked latent frame (the graph's Where after Softmax).
pub const ATTN_SOFTMAX: &str = r#"
struct P { T: u32, L: u32, H: u32, p0: u32 }
@group(0) @binding(0) var<storage, read_write> S: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
var<workgroup> red: array<f32, 64>;
fn reduce(v: f32, l: u32, is_max: bool) -> f32 {
    red[l] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (l < s) { red[l] = select(red[l] + red[l + s], max(red[l], red[l + s]), is_max); }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x + wid.y * 65535u;
    let l = lid.x;
    let t = r % p.T;
    let b = (r / p.T) / p.H;
    let base = r * p.L;
    var mx = -3.0e38;
    for (var j = l; j < p.L; j += 64u) { mx = max(mx, S[base + j]); }
    mx = reduce(mx, l, true);
    var sum = 0.0;
    for (var j = l; j < p.L; j += 64u) {
        let s = S[base + j];
        if (s > -1.0e38) { sum += exp(s - mx); }
    }
    sum = reduce(sum, l, false);
    let keep = mask[b * p.T + t] != 0.0;
    for (var j = l; j < p.L; j += 64u) {
        let s = S[base + j];
        var e = 0.0;
        if (s > -1.0e38) { e = exp(s - mx); }
        S[base + j] = select(0.0, e / sum, keep);
    }
}
"#;

/// O[b, t, h*D + d] = sum_j S[b, h, t, j] V[b, j, h*D + d], QB rows per workgroup,
/// one thread per d, the block's weights staged in shared memory 64 keys at a time.
pub fn attn_pv(text: bool) -> String {
    let (d, h) = attn_dims(text);
    let w = d * h;
    let mut acc = String::new();
    let mut body = String::new();
    let mut store = String::new();
    for q in 0..QB {
        acc += &format!("var a{q} = 0.0; ");
        body += &format!("a{q} = fma(ps[{q}u * 64u + jj], v, a{q}); ");
        store += &format!("if (t0 + {q}u < p.T) {{ O[(b * p.T + t0 + {q}u) * W + h * D + d] = a{q}; }}\n    ");
    }
    format!(
        r#"
struct P {{ T: u32, L: u32, p0: u32, p1: u32 }}
@group(0) @binding(0) var<storage, read> S: array<f32>;
@group(0) @binding(1) var<storage, read> V: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
const D = {d}u;
const H = {h}u;
const W = {w}u;
const QB = {QB}u;
var<workgroup> ps: array<f32, {psz}>;

@compute @workgroup_size({d})
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let d = lid.x;
    let bh = wid.y;
    let b = bh / H;
    let h = bh % H;
    let t0 = wid.x * QB;
    {acc}
    for (var j0 = 0u; j0 < p.L; j0 += 64u) {{
        for (var i = d; i < QB * 64u; i += D) {{
            let q = i / 64u;
            let j = j0 + i % 64u;
            ps[i] = select(0.0, S[(bh * p.T + min(t0 + q, p.T - 1u)) * p.L + min(j, p.L - 1u)], j < p.L && t0 + q < p.T);
        }}
        workgroupBarrier();
        let n = min(64u, p.L - j0);
        for (var jj = 0u; jj < n; jj++) {{
            let v = V[(b * p.L + j0 + jj) * W + h * D + d];
            {body}
        }}
        workgroupBarrier();
    }}
    {store}
}}
"#,
        psz = QB * 64,
    )
}

/// Rotary on the text keys, in place, once per chunk: [R rows, 8 heads, 32 pairs].
pub const ROPE: &str = r#"
struct P { R: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read_write> K: array<f32>;
@group(0) @binding(1) var<storage, read> rope: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    if (x >= p.R * 256u) { return; }
    let r = x / 256u;
    let h = (x % 256u) / 32u;
    let i = x % 32u;
    let c = rope[r * 64u + i];
    let s = rope[r * 64u + 32u + i];
    let a = r * 512u + h * 64u + i;
    let x1 = K[a];
    let x2 = K[a + 32u];
    K[a] = x1 * c - x2 * s;
    K[a + 32u] = x1 * s + x2 * c;
}
"#;

/// The graph's CFG and Euler step, for each of `pairs` chunks: rows [0, n)
/// of the pair in `o` are the conditional velocity, [n, 2n) the unconditional
/// one; v = v_u + g (v_c - v_u), then (x + v / total) x mask, in the graph's
/// order of operations, goes to both rows of the next step's input.
pub const BLEND: &str = r#"
struct P { n: u32, g: f32, pairs: u32, total: f32, T: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<storage, read> o: array<f32>;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let k = id.x / p.n;
    if (k >= p.pairs) { return; }
    let j = id.x % p.n;
    let i = 2u * k * p.n + j;
    let u = o[i + p.n];
    let d = o[i] - u;
    let v = u + p.g * d;
    let y = (x[i] + v / p.total) * mask[2u * k * p.T + j % p.T];
    x[i] = y;
    x[i + p.n] = y;
}
"#;
