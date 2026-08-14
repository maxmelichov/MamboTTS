use std::{
    env,
    io::BufRead,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
};

use tauri::Manager;

use crate::model;

use super::dto::ReadySignal;

pub struct RunnerState {
    pub process: Mutex<Option<RunnerProcess>>,
}

pub struct RunnerProcess {
    port: u16,
    child: Child,
    client: reqwest::Client,
    stderr_buf: Arc<Mutex<String>>,
}

impl RunnerProcess {
    pub fn spawn(app: &tauri::AppHandle, binary_path: &Path) -> Result<Self, String> {
        let mut cmd = Command::new(binary_path);
        cmd.args(["serve", "--host", "127.0.0.1", "--port", "0"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Ok(bundle) = model::model_bundle(app) {
            if bundle.installed {
                cmd.env("MAMBORAMBO_RUNTIME", &bundle.runtime);
                cmd.env("MAMBORAMBO_BLUE_MODEL_DIR", &bundle.model_path);
                if !bundle.codec_path.is_empty() {
                    cmd.env("MAMBORAMBO_RENIKUD_PATH", &bundle.codec_path);
                }
            }
        }
        prepend_native_library_paths(&mut cmd, app, binary_path);

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "failed to spawn MamboRambo server at {}: {err}",
                binary_path.display()
            )
        })?;

        let mut stderr = child.stderr.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to capture MamboRambo server stdout".to_string())?;
        let mut reader = std::io::BufReader::new(stdout);
        let mut line = String::new();

        if let Err(err) = reader.read_line(&mut line) {
            let stderr_output = read_first_stderr_line(stderr.take());
            kill_child(&mut child);
            return Err(format_runner_start_error(
                "failed to read ready signal",
                &err.to_string(),
                &stderr_output,
            ));
        }

        let signal: ReadySignal = match serde_json::from_str(line.trim()) {
            Ok(signal) => signal,
            Err(err) => {
                let stderr_output = read_first_stderr_line(stderr.take());
                kill_child(&mut child);
                return Err(format_runner_start_error(
                    "failed to parse ready signal",
                    &err.to_string(),
                    &stderr_output,
                ));
            }
        };
        if signal.status != "ready" {
            kill_child(&mut child);
            return Err(format!(
                "unexpected MamboRambo server status: {}",
                signal.status
            ));
        }

        let client = match reqwest::Client::builder().no_proxy().build() {
            Ok(client) => client,
            Err(err) => {
                kill_child(&mut child);
                return Err(format!("failed to build HTTP client: {err}"));
            }
        };

        std::thread::spawn(move || {
            let mut buf = String::new();
            while reader.read_line(&mut buf).unwrap_or(0) > 0 {
                buf.clear();
            }
        });

        let stderr_buf = Arc::new(Mutex::new(String::new()));
        if let Some(stderr) = stderr {
            let buf = stderr_buf.clone();
            std::thread::spawn(move || {
                let mut reader = std::io::BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if let Ok(mut stored) = buf.lock() {
                        if stored.len() < 8192 {
                            stored.push_str(&line);
                        }
                    }
                    line.clear();
                }
            });
        }

        Ok(Self {
            port: signal.port,
            child,
            client,
            stderr_buf,
        })
    }

    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn client(&self) -> reqwest::Client {
        self.client.clone()
    }

    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn recent_stderr(&self) -> String {
        self.stderr_buf
            .lock()
            .map(|buf| buf.trim().to_string())
            .unwrap_or_default()
    }

    pub fn kill(&mut self) {
        kill_child(&mut self.child);
    }
}

impl Drop for RunnerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}

fn read_first_stderr_line(stderr: Option<impl std::io::Read>) -> String {
    let Some(stderr) = stderr else {
        return String::new();
    };
    let mut reader = std::io::BufReader::new(stderr);
    let mut buf = String::new();
    let _ = reader.read_line(&mut buf);
    buf.truncate(4096);
    buf
}

fn kill_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn format_runner_start_error(context: &str, error: &str, stderr: &str) -> String {
    if stderr.trim().is_empty() {
        format!("{context}: {error}")
    } else {
        format!("{context}: {error}\n\nrunner stderr: {}", stderr.trim())
    }
}

fn prepend_native_library_paths(cmd: &mut Command, app: &tauri::AppHandle, binary_path: &Path) {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(parent) = binary_path.parent() {
        dirs.push(parent.to_path_buf());
    }
    if let Ok(resource_dir) = app.path().resource_dir() {
        dirs.push(resource_dir.join("binaries"));
        dirs.push(resource_dir);
    }
    dirs.retain(|dir| dir.exists());
    if dirs.is_empty() {
        return;
    }

    #[cfg(target_os = "windows")]
    {
        let current = env::var_os("PATH").unwrap_or_default();
        let mut paths: Vec<PathBuf> = dirs;
        paths.extend(env::split_paths(&current));
        if let Ok(joined) = env::join_paths(paths) {
            cmd.env("PATH", joined);
        }
    }

    #[cfg(target_os = "linux")]
    {
        let current = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let mut paths: Vec<PathBuf> = dirs;
        paths.extend(env::split_paths(&current));
        if let Ok(joined) = env::join_paths(paths) {
            cmd.env("LD_LIBRARY_PATH", joined);
        }
    }

    #[cfg(target_os = "macos")]
    {
        let current = env::var_os("DYLD_LIBRARY_PATH").unwrap_or_default();
        let mut paths: Vec<PathBuf> = dirs;
        paths.extend(env::split_paths(&current));
        if let Ok(joined) = env::join_paths(paths) {
            cmd.env("DYLD_LIBRARY_PATH", joined);
        }
    }
}
