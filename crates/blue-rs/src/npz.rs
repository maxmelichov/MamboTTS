//! Minimal reader for the `.npz` archives the v2.5 export ships.
//!
//! An `.npz` is a zip of `.npy` members. Only what the checkpoints actually
//! contain is supported: C-ordered little-endian `f4`/`f8` arrays. Anything
//! else is an error rather than a silent misread, because these arrays are
//! latent statistics — quietly transposing or truncating one would degrade
//! audio instead of failing.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};

/// One array from an `.npz`, flattened to f32 with its shape kept.
#[derive(Debug, Clone)]
pub struct NpyArray {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl NpyArray {
    /// The sole element, for the scalars stored as shape `()` or `[1]`.
    pub fn scalar(&self) -> Result<f32> {
        self.data
            .first()
            .copied()
            .context("expected at least one element")
    }
}

/// Reads every member of an `.npz` into memory, keyed by name without the
/// `.npy` suffix.
pub fn read_npz(path: impl AsRef<Path>) -> Result<HashMap<String, NpyArray>> {
    let path = path.as_ref();
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut archive =
        zip::ZipArchive::new(file).with_context(|| format!("read {} as npz", path.display()))?;

    let mut out = HashMap::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry
            .name()
            .strip_suffix(".npy")
            .unwrap_or(entry.name())
            .to_owned();
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        let array = parse_npy(&bytes).with_context(|| format!("parse {name} in {}", path.display()))?;
        out.insert(name, array);
    }
    Ok(out)
}

fn parse_npy(bytes: &[u8]) -> Result<NpyArray> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not an npy member");
    }
    // v1 headers carry a 2-byte length, v2 and later a 4-byte one.
    let major = bytes[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header_end = header_start + header_len;
    if header_end > bytes.len() {
        bail!("npy header runs past the end of the member");
    }
    let header = std::str::from_utf8(&bytes[header_start..header_end])?;

    if header.contains("'fortran_order': True") {
        bail!("Fortran-ordered npy is not supported");
    }
    let descr = field(header, "'descr':").context("npy header has no descr")?;
    let width = match descr.trim_matches(['\'', '"'].as_slice()) {
        "<f4" | "|f4" | "f4" => 4,
        "<f8" | "|f8" | "f8" => 8,
        other => bail!("unsupported npy dtype {other}"),
    };

    let shape: Vec<usize> = field(header, "'shape':")
        .context("npy header has no shape")?
        .trim_matches(['(', ')'].as_slice())
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect();

    let body = &bytes[header_end..];
    let count = shape.iter().product::<usize>().max(1);
    if body.len() < count * width {
        bail!("npy body is shorter than its shape declares");
    }
    let data = (0..count)
        .map(|i| {
            let at = i * width;
            if width == 4 {
                f32::from_le_bytes(body[at..at + 4].try_into().unwrap())
            } else {
                f64::from_le_bytes(body[at..at + 8].try_into().unwrap()) as f32
            }
        })
        .collect();
    Ok(NpyArray { shape, data })
}

/// Value of a key in the npy header dict, up to the delimiter that ends it.
fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let rest = header.split_once(key)?.1.trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')')? + 1
    } else {
        rest.find(',')?
    };
    Some(rest[..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(descr: &str, shape: &str, body: &[u8]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}"
        );
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend((header.len() as u16).to_le_bytes());
        out.extend(header.as_bytes());
        out.extend(body);
        out
    }

    #[test]
    fn reads_a_little_endian_f32_array() {
        let body: Vec<u8> = [1.5f32, -2.0, 0.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let array = parse_npy(&npy("<f4", "(3,)", &body)).unwrap();
        assert_eq!(array.shape, vec![3]);
        assert_eq!(array.data, vec![1.5, -2.0, 0.25]);
    }

    #[test]
    fn narrows_f64_to_f32() {
        let body: Vec<u8> = [0.5f64, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let array = parse_npy(&npy("<f8", "(2,)", &body)).unwrap();
        assert_eq!(array.data, vec![0.5, 4.0]);
    }

    #[test]
    fn reads_a_scalar_stored_with_an_empty_shape() {
        let body = 3.25f32.to_le_bytes();
        let array = parse_npy(&npy("<f4", "()", &body)).unwrap();
        assert!(array.shape.is_empty());
        assert_eq!(array.scalar().unwrap(), 3.25);
    }

    #[test]
    fn rejects_layouts_it_would_otherwise_misread() {
        let body = [0u8; 16];
        let fortran = {
            let header = "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }".to_string();
            let mut out = b"\x93NUMPY\x01\x00".to_vec();
            out.extend((header.len() as u16).to_le_bytes());
            out.extend(header.as_bytes());
            out.extend(body);
            out
        };
        assert!(parse_npy(&fortran).is_err());
        assert!(parse_npy(&npy("<i4", "(4,)", &body)).is_err());
        assert!(parse_npy(b"not npy at all").is_err());
    }

    #[test]
    fn rejects_a_truncated_body() {
        assert!(parse_npy(&npy("<f4", "(8,)", &[0u8; 4])).is_err());
    }
}
