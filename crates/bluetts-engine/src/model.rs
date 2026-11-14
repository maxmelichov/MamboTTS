//! The vector estimator's architecture, as read from vector_estimator.onnx
//! (notmax123/BlueV3-onnx @978d7dd, `onnx/`; docs/MODELS.md, "Rust engine").
//!
//! Hidden width 512, latent 144 channels. The graph takes one conditional
//! row per chunk and builds the CFG pair itself (classifier-free guidance is
//! baked in, input `cfg_scale`); the engine does the same: per chunk two rows,
//! conditional then unconditional. The unconditional row's text is `u_text`
//! in every text column (with the real text mask), its style `u_ref`, its
//! style keys come from `u_key` instead of `tile`.
//!
//!   proj_in    1x1 conv 144 -> 512 (no bias), x latent_mask
//!   4 x [ 4 ConvNeXt (dilation 1, 2, 4, 8)
//!         + time:  x += Linear(time_emb)[b], x latent_mask      (main_blocks.6k+1)
//!         ConvNeXt (dilation 1)
//!         text attention + residual + LayerNorm               (main_blocks.6k+3)
//!         ConvNeXt (dilation 1)
//!         style attention + residual + LayerNorm              (main_blocks.6k+5) ]
//!   last_convnext  4 ConvNeXt (dilation 1)
//!   proj_out   1x1 conv 512 -> 144 (no bias), x mask = v (per row)
//!   step       out = (x + (v_u + g (v_c - v_u)) / total_step) x mask (the blend)
//!
//! ConvNeXt: padding frames take the row's last valid frame (edge fill), edge
//! pad, depthwise conv k=5 -> x mask -> LayerNorm(512) -> 1x1 512->2048 ->
//! erf GELU -> 1x1 2048->512 -> x gamma -> + input -> x mask.
//!
//! Text attention: 8 heads of 64 (width 512); Q from x, K/V from text_emb
//! (256 -> 512); rotary on Q and K with positions t / len (len = the row's
//! mask sum); scores / 16; masked text columns get no weight; rows of masked
//! latent frames are zero. Style attention: 2 heads of 128 (width 256);
//! K = tanh(Linear(tile)) for the conditional row, tanh(Linear(u_key)) for
//! the unconditional one, V from style_ttl (u_ref); scores / 16; no key mask.

use crate::onnx::{Graph, Node};

pub const C: usize = 512; // hidden width
pub const C2: usize = 2048; // ConvNeXt expansion
pub const LAT: usize = 144; // latent channels (24 x 6)
pub const TXT: usize = 256; // text_emb channels
pub const AT: usize = 512; // text attention width (8 heads of 64)
pub const AS: usize = 256; // style attention width (2 heads of 128), style_ttl channels
pub const TEMB: usize = 64; // time embedding width
/// Positions the graph's rotary tables cover (its `increments`, 0..999): the
/// longest latent and text it can take.
pub const MAX_LEN: u32 = 1000;

#[derive(Clone, Debug)]
pub enum Layer {
    /// prefix like "main_blocks.0.convnext.2"; `time` = which of the 4 time
    /// projections is added after this block (fused into its epilogue)
    ConvNext { prefix: String, dil: u32, time: Option<usize> },
    TextAttn { block: usize },
    StyleAttn { block: usize },
}

fn node<'a>(g: &'a Graph, name: &str) -> Result<&'a Node, String> {
    g.nodes.iter().find(|n| n.name == name).ok_or_else(|| format!("graph: no node {name}"))
}

/// The ConvNeXt's dilation, from its depthwise Conv node.
fn dilation(g: &Graph, prefix: &str) -> Result<u32, String> {
    // "main_blocks.0.convnext.2" -> "/vf/main_blocks.0/convnext.2/dwconv/Conv"
    let (block, cn) = prefix.rsplit_once(".convnext.").ok_or("bad prefix")?;
    let n = node(g, &format!("/vf/{block}/convnext.{cn}/dwconv/Conv"))?;
    let k = n.attrs.get("kernel_shape").map(|a| a.ints.clone()).unwrap_or_default();
    if k != [5] {
        return Err(format!("{prefix}: expected kernel 5, got {k:?}"));
    }
    Ok(n.attrs.get("dilations").and_then(|a| a.ints.first().copied()).unwrap_or(1) as u32)
}

/// The run's layer sequence, checked against the graph.
pub fn layers(g: &Graph) -> Result<Vec<Layer>, String> {
    // CFG inside the graph (the engine builds the pair the same way).
    for n in ["u_text", "u_ref", "u_key", "tile"] {
        if !g.inits.contains_key(n) {
            return Err(format!("graph: no {n}; this engine runs the BlueV3 vector estimator (CFG in the graph)"));
        }
    }
    let mut out = Vec::new();
    let cn = |out: &mut Vec<Layer>, prefix: String, time: Option<usize>| -> Result<(), String> {
        let dil = dilation(g, &prefix)?;
        out.push(Layer::ConvNext { prefix, dil, time });
        Ok(())
    };
    for sb in 0..4 {
        let base = sb * 6;
        for i in 0..4 {
            cn(&mut out, format!("main_blocks.{base}.convnext.{i}"), (i == 3).then_some(sb))?;
        }
        cn(&mut out, format!("main_blocks.{}.convnext.0", base + 2), None)?;
        node(g, &format!("/vf/main_blocks.{}/attn/W_query/linear/MatMul", base + 3))?;
        out.push(Layer::TextAttn { block: base + 3 });
        cn(&mut out, format!("main_blocks.{}.convnext.0", base + 4), None)?;
        node(g, &format!("/vf/main_blocks.{}/attention/W_query/linear/MatMul", base + 5))?;
        out.push(Layer::StyleAttn { block: base + 5 });
    }
    for i in 0..4 {
        cn(&mut out, format!("last_convnext.convnext.{i}"), None)?;
    }
    Ok(out)
}

/// The weight initializer of a linear layer (`MatMul(x, W)`), W already [in, out].
pub fn linear_weight(g: &Graph, block: usize, attn: &str, which: &str) -> Result<String, String> {
    let n = node(g, &format!("/vf/main_blocks.{block}/{attn}/{which}/linear/MatMul"))?;
    n.inputs.get(1).cloned().ok_or_else(|| format!("{} has no weight input", n.name))
}

/// The time encoder's sinusoid frequencies (a Constant node, 32 values).
pub fn time_freqs(g: &Graph) -> Result<Vec<f32>, String> {
    let n = g
        .nodes
        .iter()
        .find(|n| n.op == "Mul" && n.name == "/vf/time_encoder/sinusoidal/Mul_1")
        .ok_or("graph: no time encoder")?;
    let c = g
        .nodes
        .iter()
        .find(|c| c.op == "Constant" && c.outputs.first() == n.inputs.get(1))
        .ok_or("graph: time frequencies are not a Constant")?;
    let t = c.attrs.get("value").and_then(|a| a.t.as_ref()).ok_or("graph: bad Constant")?;
    if t.inline.len() != TEMB / 2 {
        return Err(format!("graph: {} time frequencies", t.inline.len()));
    }
    Ok(t.inline.clone())
}
