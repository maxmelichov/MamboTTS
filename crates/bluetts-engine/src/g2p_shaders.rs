//! WGSL kernels of the RenikudPlus G2P (g2p.rs) besides the matmuls, which
//! are shaders::matmul_ex. Activations are [M, 1024] row-major, M = the
//! token count padded to a multiple of 32 (padding rows are computed and
//! never read; attention ignores their keys).

/// LayerNorm over 1024 channels, one workgroup (256 threads x 4 channels) per row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LnMode {
    /// Y = LN(X), eps 1e-12 (the encoder's residual LayerNorms)
    Plain,
    /// Y = LN(word[id] + type[0] + pos[t]), eps 1e-12 (the embeddings)
    Embed,
    /// R = X * (1 + a) + b (FiLM by the speakers), Y = LN(R), eps 1e-5
    Film,
}

pub fn layernorm(mode: LnMode) -> String {
    let (extra, load, eps, store_r) = match mode {
        LnMode::Plain => (
            "@group(0) @binding(0) var<storage, read> X: array<f32>;",
            "X[m * 1024u + c]",
            "9.999999960041972e-13",
            "",
        ),
        LnMode::Embed => (
            "@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<storage, read> word: array<f32>;
@group(0) @binding(5) var<storage, read> tok: array<f32>;
@group(0) @binding(6) var<storage, read> pos: array<f32>;",
            "(word[ids[m] * 1024u + c] + tok[c]) + pos[m * 1024u + c]",
            "9.999999960041972e-13",
            "",
        ),
        LnMode::Film => (
            "@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(4) var<storage, read> film: array<f32>;
@group(0) @binding(5) var<storage, read_write> R: array<f32>;",
            "X[m * 1024u + c] * (film[c] + 1.0) + film[1024u + c]",
            "9.999999747378752e-06",
            "R[m * 1024u + l] = v0; R[m * 1024u + l + 256u] = v1; R[m * 1024u + l + 512u] = v2; R[m * 1024u + l + 768u] = v3;",
        ),
    };
    format!(
        r#"
{extra}
@group(0) @binding(1) var<storage, read_write> Y: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
var<workgroup> red: array<f32, 256>;
fn sum256(v: f32, l: u32) -> f32 {{
    red[l] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {{
        if (l < s) {{ red[l] += red[l + s]; }}
        workgroupBarrier();
    }}
    let r = red[0];
    workgroupBarrier();
    return r;
}}
fn load(m: u32, c: u32) -> f32 {{ return {load}; }}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let l = lid.x;
    let m = wid.x;
    var v0 = load(m, l);
    var v1 = load(m, l + 256u);
    var v2 = load(m, l + 512u);
    var v3 = load(m, l + 768u);
    {store_r}
    let mean = sum256((v0 + v1) + (v2 + v3), l) / 1024.0;
    v0 -= mean; v1 -= mean; v2 -= mean; v3 -= mean;
    let inv = 1.0 / sqrt(sum256((v0 * v0 + v1 * v1) + (v2 * v2 + v3 * v3), l) / 1024.0 + {eps});
    Y[m * 1024u + l] = v0 * inv * g[l] + beta[l];
    Y[m * 1024u + l + 256u] = v1 * inv * g[l + 256u] + beta[l + 256u];
    Y[m * 1024u + l + 512u] = v2 * inv * g[l + 512u] + beta[l + 512u];
    Y[m * 1024u + l + 768u] = v3 * inv * g[l + 768u] + beta[l + 768u];
}}
"#
    )
}

/// Query rows per attention workgroup.
pub const QB: u32 = 16;

/// Self-attention, 16 heads of 64, in one kernel: QB query rows of one head
/// per workgroup, keys in tiles of 64 with a running max and sum (the
/// softmax over all keys, without a scores buffer: S can be 2,048). q and k
/// are each scaled by 64^-0.25 before the dot product, as the graph does.
/// QKV is [M, 3072] (q | k | v, head-major inside each), O is [M, 1024].
pub fn attention() -> String {
    let mut acc = String::new();
    let mut score = String::new();
    let mut put = String::new();
    let mut rescale = String::new();
    let mut pv = String::new();
    let mut store = String::new();
    for q in 0..QB {
        acc += &format!("var a{q} = 0.0; ");
        score += &format!("var s{q} = 0.0; ");
        put += &format!("ps[{q}u * 64u + l] = s{q};\n            ");
        rescale += &format!("a{q} *= al[{q}u]; ");
        pv += &format!("a{q} = fma(ps[{q}u * 64u + jj], v, a{q}); ");
        store += &format!("if (t0 + {q}u < p.M) {{ O[(t0 + {q}u) * 1024u + h * 64u + l] = a{q} / ls[{q}u]; }}\n    ");
    }
    let mut dots = String::new();
    for q in 0..QB {
        dots += &format!("s{q} += dot(qs[{q}u * 16u + e4], kv); ");
    }
    format!(
        r#"
struct P {{ S: u32, M: u32, p0: u32, p1: u32 }}
@group(0) @binding(0) var<storage, read> QKV: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
const SC = 0.3535533845424652;
var<workgroup> qs: array<vec4<f32>, {qsz}>;
var<workgroup> ps: array<f32, {psz}>;
var<workgroup> al: array<f32, {QB}>;
var<workgroup> ls: array<f32, {QB}>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let l = lid.x;
    let h = wid.y;
    let t0 = wid.x * {QB}u;
    for (var i = l; i < {QB}u * 16u; i += 64u) {{
        let t = min(t0 + i / 16u, p.M - 1u);
        qs[i] = QKV[t * 768u + h * 16u + i % 16u] * SC;
    }}
    var mrow = -3.4028234663852886e38;
    var lrow = 0.0;
    {acc}
    workgroupBarrier();
    for (var j0 = 0u; j0 < p.S; j0 += 64u) {{
        let j = j0 + l;
        {{
            {score}
            if (j < p.S) {{
                let kb = j * 768u + 256u + h * 16u;
                for (var e4 = 0u; e4 < 16u; e4++) {{
                    let kv = QKV[kb + e4] * SC;
                    {dots}
                }}
            }}
            {put}
        }}
        workgroupBarrier();
        let n = min(64u, p.S - j0);
        if (l < {QB}u) {{
            var mx = mrow;
            for (var jj = 0u; jj < n; jj++) {{ mx = max(mx, ps[l * 64u + jj]); }}
            let alpha = exp(mrow - mx);
            var sum = 0.0;
            for (var jj = 0u; jj < n; jj++) {{
                let e = exp(ps[l * 64u + jj] - mx);
                ps[l * 64u + jj] = e;
                sum += e;
            }}
            lrow = lrow * alpha + sum;
            mrow = mx;
            al[l] = alpha;
        }}
        workgroupBarrier();
        {rescale}
        for (var jj = 0u; jj < n; jj++) {{
            let v4 = QKV[(j0 + jj) * 768u + 512u + h * 16u + l / 4u];
            let v = v4[l % 4u];
            {pv}
        }}
        workgroupBarrier();
    }}
    if (l < {QB}u) {{ ls[l] = lrow; }}
    workgroupBarrier();
    {store}
}}
"#,
        qsz = QB * 16,
        psz = QB * 64,
    )
}

/// The three cascaded heads, one workgroup (64 threads) per row:
///   c = x Wc + bc (25), v = [x, softmax c] Wv + bv (6),
///   s = [x, softmax c, softmax v] Ws + bs (2); OUT[m] = [c, v, s] (33).
/// W* are [in, out] as the graph stores them; B = [bc, bv, bs].
pub const HEADS: &str = r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> WC: array<f32>;
@group(0) @binding(2) var<storage, read> WV: array<f32>;
@group(0) @binding(3) var<storage, read> WS: array<f32>;
@group(0) @binding(4) var<storage, read> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> OUT: array<f32>;
var<workgroup> red: array<f32, 64>;
var<workgroup> lg: array<f32, 33>;
fn sum64(v: f32, l: u32) -> f32 {
    red[l] = v;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s >>= 1u) {
        if (l < s) { red[l] += red[l + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let l = lid.x;
    let m = wid.x;
    var x: array<f32, 16>;
    for (var i = 0u; i < 16u; i++) { x[i] = X[m * 1024u + l + 64u * i]; }
    for (var o = 0u; o < 25u; o++) {
        var a = 0.0;
        for (var i = 0u; i < 16u; i++) { a = fma(x[i], WC[(l + 64u * i) * 25u + o], a); }
        let c = sum64(a, l);
        if (l == 0u) { lg[o] = c + B[o]; }
    }
    workgroupBarrier();
    var mx = -3.4028234663852886e38;
    for (var o = 0u; o < 25u; o++) { mx = max(mx, lg[o]); }
    var sm = 0.0;
    for (var o = 0u; o < 25u; o++) { sm += exp(lg[o] - mx); }
    let pc = select(0.0, exp(lg[min(l, 24u)] - mx) / sm, l < 25u);
    for (var o = 0u; o < 6u; o++) {
        var a = 0.0;
        for (var i = 0u; i < 16u; i++) { a = fma(x[i], WV[(l + 64u * i) * 6u + o], a); }
        if (l < 25u) { a = fma(pc, WV[(1024u + l) * 6u + o], a); }
        let v = sum64(a, l);
        if (l == 0u) { lg[25u + o] = v + B[25u + o]; }
    }
    workgroupBarrier();
    var mv = -3.4028234663852886e38;
    for (var o = 0u; o < 6u; o++) { mv = max(mv, lg[25u + o]); }
    var sv = 0.0;
    for (var o = 0u; o < 6u; o++) { sv += exp(lg[25u + o] - mv); }
    let pv = select(0.0, exp(lg[25u + min(l, 5u)] - mv) / sv, l < 6u);
    for (var o = 0u; o < 2u; o++) {
        var a = 0.0;
        for (var i = 0u; i < 16u; i++) { a = fma(x[i], WS[(l + 64u * i) * 2u + o], a); }
        if (l < 25u) { a = fma(pc, WS[(1024u + l) * 2u + o], a); }
        if (l < 6u) { a = fma(pv, WS[(1049u + l) * 2u + o], a); }
        let s = sum64(a, l);
        if (l == 0u) { lg[31u + o] = s + B[31u + o]; }
    }
    workgroupBarrier();
    if (l < 33u) { OUT[m * 33u + l] = lg[l]; }
}
"#;

/// A matmul weight from its stored [K][N] (in scratch) to the 16-column
/// blocks the matmuls read, [N_total / 16][K][16], at column `n_off` of a
/// fused weight (q | k | v): load time only.
pub const REBLOCK: &str = r#"
struct P { K: u32, N: u32, n_off: u32, p0: u32 }
@group(0) @binding(0) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x + id.y * 65535u * 256u;
    if (i >= p.K * p.N / 4u) { return; }
    let k = i / (p.N / 4u);
    let n = (i % (p.N / 4u)) * 4u + p.n_off;
    dst[((n / 16u) * p.K + k) * 4u + (n % 16u) / 4u] = src[i];
}
"#;


/// The sum of a split-K matmul's `kz` partials [kz, M, N], then its epilogue.
pub fn reduce(kz: u32, epi: crate::shaders::Epi) -> String {
    use crate::shaders::Epi;
    let (extra, body) = match epi {
        Epi::Bias => ("@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;", "C[i] = v + bias[n4];"),
        Epi::BiasRes => (
            "@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> R: array<vec4<f32>>;",
            "C[i] = (v + bias[n4]) + R[i];",
        ),
        Epi::BiasGeluX => (
            "@group(0) @binding(3) var<storage, read> bias: array<vec4<f32>>;",
            "let z = v + bias[n4]; C[i] = z * (0.5 * (erf4(z / 1.4142135381698608) + 1.0));",
        ),
        e => panic!("reduce: no epilogue {e:?}"),
    };
    let mut sum = String::from("var v = P[i];");
    for z in 1..kz {
        sum += &format!(" v += P[i + {z}u * mn4];");
    }
    format!(
        r#"
struct Pm {{ M: u32, N: u32, p0: u32, p1: u32 }}
@group(0) @binding(0) var<storage, read> P: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> C: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: Pm;
{extra}
fn erf_(x: f32) -> f32 {{
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))));
    return select(r - 1.0, 1.0 - r, x >= 0.0);
}}
fn erf4(x: vec4<f32>) -> vec4<f32> {{ return vec4<f32>(erf_(x.x), erf_(x.y), erf_(x.z), erf_(x.w)); }}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    let mn4 = p.M * p.N / 4u;
    if (i >= mn4) {{ return; }}
    let n4 = i % (p.N / 4u);
    {sum}
    {body}
}}
"#
    )
}
