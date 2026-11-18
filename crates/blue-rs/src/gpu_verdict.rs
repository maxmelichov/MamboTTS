//! What this machine's GPU did the last time the flow engine was tried on it,
//! so a driver that crashed, hung or computed wrongly is not tried on every
//! start.
//!
//! One JSON file (by default `gpu-verdicts.json` in the app's data directory)
//! holds an entry per key. A key names the backend, the adapter and its driver
//! as wgpu reports them, and the build (see [`build_id`]), so a driver update,
//! another GPU or another app build is probed afresh. An entry is written
//! `probing` before the risky part starts and `ok` or `failed` after it: a
//! process that dies in between leaves `probing` behind, which the next start
//! reads (see `gpu_flow.rs`).
//!
//! Before the worker has reported its adapter there is no adapter to key on;
//! that stage is recorded under the backend alone ([`backend_key`]).

use std::{
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use serde_json::{Map, Value, json};

use crate::ve_gpu::AdapterId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum State {
    /// A probe started and has not reported back. `isolated`: it ran in a
    /// worker process, so the parent outlived anything the driver did and a
    /// stale entry means the app itself was stopped mid-probe.
    Probing { isolated: bool },
    Ok,
    Failed { reason: String },
}

#[derive(Clone, Debug)]
pub struct VerdictFile {
    path: Option<PathBuf>,
}

impl VerdictFile {
    /// `MAMBOTTS_GPU_VERDICT_FILE`, else `gpu-verdicts.json` in the desktop
    /// app's data directory (Tauri's, `com.maxmelichov.mambotts`). An empty
    /// `MAMBOTTS_GPU_VERDICT_FILE` turns the cache off.
    pub fn from_env() -> Self {
        match std::env::var_os("MAMBOTTS_GPU_VERDICT_FILE") {
            Some(p) if p.is_empty() => Self { path: None },
            Some(p) => Self { path: Some(PathBuf::from(p)) },
            None => Self {
                path: app_data_dir().map(|d| d.join("gpu-verdicts.json")),
            },
        }
    }

    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: Some(path.into()) }
    }

    pub fn disabled() -> Self {
        Self { path: None }
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    pub fn get(&self, key: &str) -> Option<State> {
        let entry = self.read().remove(key)?;
        match entry.get("state")?.as_str()? {
            "probing" => Some(State::Probing {
                isolated: entry.get("isolated").and_then(Value::as_bool).unwrap_or(false),
            }),
            "ok" => Some(State::Ok),
            "failed" => Some(State::Failed {
                reason: entry.get("reason").and_then(Value::as_str).unwrap_or("").to_owned(),
            }),
            _ => None,
        }
    }

    pub fn set(&self, key: &str, state: &State) {
        let Some(path) = &self.path else { return };
        let mut all = self.read();
        let at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let entry = match state {
            State::Probing { isolated } => json!({"state": "probing", "isolated": isolated, "at": at}),
            State::Ok => json!({"state": "ok", "at": at}),
            State::Failed { reason } => json!({"state": "failed", "reason": reason, "at": at}),
        };
        all.insert(key.to_owned(), entry);
        if let Err(err) = write_atomically(path, &Value::Object(all)) {
            eprintln!("GPU verdict cache {}: {err}", path.display());
        }
    }

    pub fn remove(&self, key: &str) {
        let Some(path) = &self.path else { return };
        let mut all = self.read();
        if all.remove(key).is_some() {
            let _ = write_atomically(path, &Value::Object(all));
        }
    }

    fn read(&self) -> Map<String, Value> {
        let Some(path) = &self.path else { return Map::new() };
        std::fs::read(path)
            .ok()
            .and_then(|raw| serde_json::from_slice::<Value>(&raw).ok())
            .and_then(|v| match v {
                Value::Object(m) => Some(m),
                _ => None,
            })
            .unwrap_or_default()
    }
}

fn write_atomically(path: &Path, value: &Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    std::fs::write(&tmp, serde_json::to_vec_pretty(value).unwrap_or_default())?;
    std::fs::rename(&tmp, path)
}

/// The key of an adapter's verdict.
pub fn adapter_key(id: &AdapterId, build: &str) -> String {
    format!(
        "{}|{}|{:04x}:{:04x}|{}|{}|{}",
        id.backend, id.name, id.vendor, id.device, id.driver, id.driver_info, build
    )
}

/// The key of a backend before its adapter is known.
pub fn backend_key(backend: &str, build: &str) -> String {
    format!("{backend}|<before the adapter>|{build}")
}

/// This build of the app: the crate's version and the size and modification
/// time of `exe` (the executable hosting the engine), which change with every
/// app update.
pub fn build_id(exe: Option<&Path>) -> String {
    let stamp = exe
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| {
            let mtime = m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0);
            format!("{}-{}", m.len(), mtime)
        })
        .unwrap_or_else(|| "?".into());
    format!("blue-rs {} exe {stamp}", env!("CARGO_PKG_VERSION"))
}

/// The desktop app's data directory (Tauri's app data dir for
/// `com.maxmelichov.mambotts`), where its models are downloaded too.
fn app_data_dir() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        PathBuf::from(std::env::var_os("LOCALAPPDATA")?)
    } else if cfg!(target_os = "macos") {
        PathBuf::from(std::env::var_os("HOME")?).join("Library/Application Support")
    } else {
        match std::env::var_os("XDG_DATA_HOME") {
            Some(d) if !d.is_empty() => PathBuf::from(d),
            _ => PathBuf::from(std::env::var_os("HOME")?).join(".local/share"),
        }
    };
    Some(base.join("com.maxmelichov.mambotts"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_round_trip_and_survive_other_keys() {
        let dir = std::env::temp_dir().join(format!("gpu-verdict-test-{}", std::process::id()));
        let file = VerdictFile::at(dir.join("v.json"));
        assert_eq!(file.get("a"), None);
        file.set("a", &State::Probing { isolated: true });
        file.set("b", &State::Ok);
        assert_eq!(file.get("a"), Some(State::Probing { isolated: true }));
        file.set("a", &State::Failed { reason: "crashed".into() });
        assert_eq!(file.get("a"), Some(State::Failed { reason: "crashed".into() }));
        assert_eq!(file.get("b"), Some(State::Ok));
        file.remove("a");
        assert_eq!(file.get("a"), None);
        std::fs::write(dir.join("v.json"), b"not json").unwrap();
        assert_eq!(file.get("b"), None, "a damaged file reads as empty");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn keys_change_with_the_driver() {
        let id = AdapterId {
            backend: "vulkan".into(),
            name: "AMD Radeon(TM) Graphics".into(),
            vendor: 0x1002,
            device: 0x1638,
            driver: "AMD proprietary driver".into(),
            driver_info: "24.5.1".into(),
        };
        let newer = AdapterId { driver_info: "24.9.1".into(), ..id.clone() };
        assert_ne!(adapter_key(&id, "b"), adapter_key(&newer, "b"));
        assert_ne!(adapter_key(&id, "b"), adapter_key(&id, "c"));
    }
}
