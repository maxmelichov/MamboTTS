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
        for base in [
            "/usr/lib/mamborambo",
            "/usr/lib/mamborambo/binaries",
            "/opt/mamborambo",
            "/opt/mamborambo/binaries",
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

    Err("MamboRambo server sidecar not found".to_string())
}

fn runner_binary_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "mamborambo-server.exe"
    } else {
        "mamborambo-server"
    }
}

fn find_in_path(binary_name: &str) -> Option<PathBuf> {
    let path_var = env::var_os("PATH")?;
    env::split_paths(&path_var)
        .map(|dir| dir.join(binary_name))
        .find(|path| path.exists() && path.is_file())
}
