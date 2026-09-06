use std::{env, path::PathBuf};
use tauri::Manager;

pub fn resolve_runner_binary(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let binary_name = runner_binary_name();

    if let Ok(resource_dir) = app.path().resource_dir() {
        let path = resource_dir.join(binary_name);
        if path.exists() {
            return Ok(path);
        }
    }

    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let path = exe_dir.join(binary_name);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // The Debian package puts the sidecar next to the shell in /usr/bin and
        // its ONNX Runtime resources under /usr/lib/MamboTTS, so the lookups
        // above normally win. These stay as a fallback for hand-rolled installs
        // that drop the payload somewhere else, and they cover both the
        // product-name and lowercase directory spellings.
        for base in [
            "/usr/lib/MamboTTS",
            "/usr/lib/MamboTTS/binaries",
            "/usr/lib/mambotts",
            "/usr/lib/mambotts/binaries",
            "/opt/mambotts",
            "/opt/mambotts/binaries",
        ] {
            let path = PathBuf::from(base).join(binary_name);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    if let Some(path) = find_in_path(binary_name) {
        return Ok(path);
    }

    Err("MamboTTS server sidecar not found".to_string())
}

fn runner_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "mambotts-server.exe"
    } else {
        "mambotts-server"
    }
}

fn find_in_path(binary_name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .map(|dir| dir.join(binary_name))
        .find(|path| path.exists() && path.is_file())
}
