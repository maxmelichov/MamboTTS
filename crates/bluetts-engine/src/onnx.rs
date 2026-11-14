//! Just enough of ONNX's protobuf to find the vector estimator's weights.
//!
//! The model bytes are never copied whole into wasm memory (BlueV3's 257 MB, which
//! the wasm heap would then keep for good): the walker reads through a `Source`,
//! a small window at a time, and records where each initializer's
//! `raw_data` lies. Only nodes (small) are copied, to read their names,
//! inputs and attributes; the weights are read one tensor at a time when
//! they are uploaded.
//!
//! Field numbers (onnx.proto):
//!   ModelProto      graph = 7
//!   GraphProto      node = 1, initializer = 5
//!   NodeProto       input = 1, output = 2, name = 3, op_type = 4, attribute = 5
//!   AttributeProto  name = 1, f = 2, i = 3, t = 5, ints = 8
//!   TensorProto     dims = 1, data_type = 2, float_data = 4, int64_data = 7,
//!                   name = 8, raw_data = 9, data_location = 14

use std::collections::HashMap;

/// Random access to the model bytes (a slice natively, a JS Uint8Array in the browser).
pub trait Source {
    fn len(&self) -> usize;
    fn read_into(&self, offset: usize, dst: &mut [u8]);
}

impl Source for [u8] {
    fn len(&self) -> usize {
        <[u8]>::len(self)
    }
    fn read_into(&self, offset: usize, dst: &mut [u8]) {
        dst.copy_from_slice(&self[offset..offset + dst.len()]);
    }
}

/// A byte cursor over a `Source` with a 64 KB read-ahead window.
struct Window<'a, S: Source + ?Sized> {
    src: &'a S,
    buf: Vec<u8>,
    start: usize,
}

const WINDOW: usize = 64 * 1024;

impl<'a, S: Source + ?Sized> Window<'a, S> {
    fn new(src: &'a S) -> Self {
        Window { src, buf: Vec::new(), start: 0 }
    }
    fn byte(&mut self, p: usize) -> u8 {
        if p < self.start || p >= self.start + self.buf.len() {
            let n = WINDOW.min(self.src.len() - p);
            self.buf.resize(n, 0);
            self.src.read_into(p, &mut self.buf);
            self.start = p;
        }
        self.buf[p - self.start]
    }
    fn varint(&mut self, p: &mut usize) -> u64 {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = self.byte(*p);
            *p += 1;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return v;
            }
            shift += 7;
        }
    }
    /// The fields of the message in [start, end): (field, wire, value start, value end).
    fn fields(&mut self, start: usize, end: usize) -> Result<Vec<(u32, u8, usize, usize)>, String> {
        let mut out = Vec::new();
        let mut p = start;
        while p < end {
            let key = self.varint(&mut p);
            let (field, wire) = ((key >> 3) as u32, (key & 7) as u8);
            let vs = p;
            match wire {
                0 => {
                    self.varint(&mut p);
                }
                1 => p += 8,
                5 => p += 4,
                2 => {
                    let len = self.varint(&mut p) as usize;
                    out.push((field, wire, p, p + len));
                    p += len;
                    continue;
                }
                w => return Err(format!("protobuf: wire type {w}")),
            }
            out.push((field, wire, vs, p));
        }
        if p != end {
            return Err("protobuf: truncated message".into());
        }
        Ok(out)
    }
}

/// Where a tensor's data lies in the model bytes, or its (small) inline data.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub dims: Vec<usize>,
    pub data_type: u32,
    /// raw_data's byte range in the model
    pub raw: Option<(usize, usize)>,
    /// float_data / int64_data, as f32 (only small tensors use these)
    pub inline: Vec<f32>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Attr {
    pub f: Option<f32>,
    pub i: Option<i64>,
    pub ints: Vec<i64>,
    pub t: Option<Tensor>,
}

#[derive(Clone, Debug, Default)]
pub struct Node {
    pub name: String,
    pub op: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: HashMap<String, Attr>,
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub inits: HashMap<String, Tensor>,
}

/// Varint read from an in-memory slice.
fn varint_at(b: &[u8], p: &mut usize) -> u64 {
    let mut v = 0u64;
    let mut shift = 0;
    loop {
        let x = b[*p];
        *p += 1;
        v |= ((x & 0x7f) as u64) << shift;
        if x & 0x80 == 0 {
            return v;
        }
        shift += 7;
    }
}

fn parse_tensor<S: Source + ?Sized>(w: &mut Window<S>, start: usize, end: usize) -> Result<(String, Tensor), String> {
    let mut t = Tensor { dims: vec![], data_type: 0, raw: None, inline: vec![] };
    let mut name = String::new();
    for (field, wire, s, e) in w.fields(start, end)? {
        match (field, wire) {
            (1, 0) => {
                let mut p = s;
                t.dims.push(w.varint(&mut p) as usize);
            }
            (1, 2) => {
                let mut p = s;
                while p < e {
                    t.dims.push(w.varint(&mut p) as usize);
                }
            }
            (2, 0) => {
                let mut p = s;
                t.data_type = w.varint(&mut p) as u32;
            }
            (4, 2) => {
                let mut b = vec![0u8; e - s];
                w.src.read_into(s, &mut b);
                t.inline = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            }
            (7, 2) => {
                let mut p = s;
                while p < e {
                    t.inline.push(w.varint(&mut p) as i64 as f32);
                }
            }
            (8, 2) => {
                let mut b = vec![0u8; e - s];
                w.src.read_into(s, &mut b);
                name = String::from_utf8_lossy(&b).into_owned();
            }
            (9, 2) => t.raw = Some((s, e)),
            // data_location: 0 DEFAULT (set explicitly by some exporters), 1 EXTERNAL.
            (14, 0) => {
                let mut p = s;
                if w.varint(&mut p) != 0 {
                    return Err(format!("tensor {name}: external data is not supported"));
                }
            }
            _ => {}
        }
    }
    Ok((name, t))
}

fn parse_node(b: &[u8]) -> Result<Node, String> {
    let src: &[u8] = b;
    let mut w = Window::new(src);
    let mut n = Node::default();
    let s = |b: &[u8], s: usize, e: usize| String::from_utf8_lossy(&b[s..e]).into_owned();
    for (field, wire, vs, ve) in w.fields(0, b.len())? {
        match (field, wire) {
            (1, 2) => n.inputs.push(s(b, vs, ve)),
            (2, 2) => n.outputs.push(s(b, vs, ve)),
            (3, 2) => n.name = s(b, vs, ve),
            (4, 2) => n.op = s(b, vs, ve),
            (5, 2) => {
                let mut a = Attr::default();
                let mut aname = String::new();
                for (af, aw, as_, ae) in w.fields(vs, ve)? {
                    match (af, aw) {
                        (1, 2) => aname = s(b, as_, ae),
                        (2, 5) => a.f = Some(f32::from_le_bytes([b[as_], b[as_ + 1], b[as_ + 2], b[as_ + 3]])),
                        (3, 0) => {
                            let mut p = as_;
                            a.i = Some(varint_at(b, &mut p) as i64);
                        }
                        (5, 2) => {
                            let (_, mut t) = parse_tensor(&mut Window::new(src), as_, ae)?;
                            // A Constant's data is inline in the node: keep it as floats.
                            if let Some((rs, re)) = t.raw.take() {
                                let raw = &b[rs..re];
                                t.inline = match t.data_type {
                                    1 => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
                                    7 => raw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
                                    _ => vec![],
                                };
                            }
                            a.t = Some(t);
                        }
                        (8, 0) => {
                            let mut p = as_;
                            a.ints.push(varint_at(b, &mut p) as i64);
                        }
                        (8, 2) => {
                            let mut p = as_;
                            while p < ae {
                                a.ints.push(varint_at(b, &mut p) as i64);
                            }
                        }
                        _ => {}
                    }
                }
                n.attrs.insert(aname, a);
            }
            _ => {}
        }
    }
    Ok(n)
}

/// Walks ModelProto → GraphProto: every node (copied, they are small) and
/// every initializer's location (not its data).
pub fn parse<S: Source + ?Sized>(src: &S) -> Result<Graph, String> {
    let mut w = Window::new(src);
    let top = w.fields(0, src.len())?;
    let &(_, _, gs, ge) = top.iter().find(|f| f.0 == 7 && f.1 == 2).ok_or("no graph in the model")?;
    let mut g = Graph { nodes: Vec::new(), inits: HashMap::new() };
    for (field, wire, s, e) in w.fields(gs, ge)? {
        if wire != 2 {
            continue;
        }
        match field {
            1 => {
                let mut b = vec![0u8; e - s];
                src.read_into(s, &mut b);
                g.nodes.push(parse_node(&b)?);
            }
            5 => {
                let (name, t) = parse_tensor(&mut w, s, e)?;
                g.inits.insert(name, t);
            }
            _ => {}
        }
    }
    Ok(g)
}

/// A float32 initializer's values (read from the source now).
pub fn read_f32<S: Source + ?Sized>(src: &S, t: &Tensor) -> Result<Vec<f32>, String> {
    if t.data_type != 1 {
        return Err(format!("expected float32, got data_type {}", t.data_type));
    }
    match t.raw {
        Some((s, e)) => {
            let mut b = vec![0u8; e - s];
            src.read_into(s, &mut b);
            if b.len() != t.numel() * 4 {
                return Err("raw_data size does not match dims".into());
            }
            Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        None => Ok(t.inline.clone()),
    }
}
