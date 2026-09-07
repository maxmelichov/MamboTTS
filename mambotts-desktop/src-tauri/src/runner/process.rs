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

/// espeak-rs reads this to find its phoneme data, ahead of any other probe.
const ESPEAK_DATA_ENV: &str = "PIPER_ESPEAKNG_DATA_DIRECTORY";
const ESPEAK_DATA_DIR_NAME: &str = "espeak-ng-data";

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
                cmd.env("MAMBOTTS_RUNTIME", &bundle.runtime);
                cmd.env("MAMBOTTS_BLUE_MODEL_DIR", &bundle.model_path);
                if !bundle.codec_path.is_empty() {
                    cmd.env("MAMBOTTS_RENIKUD_PATH", &bundle.codec_path);
                }
            }
        }
        prepend_native_library_paths(&mut cmd, app, binary_path);
        set_espeak_data_dir(&mut cmd, app, binary_path);

        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }

        let mut child = cmd.spawn().map_err(|err| {
            format!(
                "failed to spawn MamboTTS server at {}: {err}",
                binary_path.display()
            )
        })?;

        #[cfg(target_os = "windows")]
        confine_to_job_object(&child);

        let mut stderr = child.stderr.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "failed to capture MamboTTS server stdout".to_string())?;
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
                "unexpected MamboTTS server status: {}",
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
    let dirs = sidecar_asset_dirs(app, binary_path);
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

/// The directories that hold the assets shipped beside the sidecar: the ONNX
/// Runtime shared libraries and the espeak-ng data.
///
/// Tauri does not gather these in one place. `externalBin` puts the sidecar
/// next to the main executable, while `resources` keeps the declared
/// `binaries/` prefix underneath the resource directory, so both roots have to
/// be offered and the caller takes whichever one actually holds what it wants.
fn sidecar_asset_dirs(app: &tauri::AppHandle, binary_path: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(parent) = binary_path.parent() {
        dirs.push(parent.to_path_buf());
    }
    if let Ok(resource_dir) = app.path().resource_dir() {
        dirs.push(resource_dir.join("binaries"));
        dirs.push(resource_dir);
    }
    dirs.retain(|dir| dir.exists());
    dirs
}

/// Point espeak-ng at the phoneme data that ships with the app.
///
/// When it is given nothing, espeak-ng falls back to the data directory that
/// was baked in when it was compiled. That is a path on the build machine, so
/// on a user's machine it does not exist and initialisation fails. Everything
/// that goes through espeak then breaks, which is English, Spanish, German and
/// Italian; Hebrew survives only because it takes the separate Renikud
/// phonemizer. An explicit environment variable is used rather than relying on
/// the executable-directory probe in espeak-rs, because the sidecar and the
/// data can land in different directories depending on the bundle format.
fn set_espeak_data_dir(cmd: &mut Command, app: &tauri::AppHandle, binary_path: &Path) {
    // Respect a deliberate override from the surrounding environment.
    if env::var_os(ESPEAK_DATA_ENV).is_some() {
        return;
    }
    for dir in sidecar_asset_dirs(app, binary_path) {
        if dir.join(ESPEAK_DATA_DIR_NAME).is_dir() {
            cmd.env(ESPEAK_DATA_ENV, plain_path(&dir));
            return;
        }
    }
}

/// Tie the sidecar's lifetime to this process with a Windows job object.
///
/// `--exit-with-parent` is a no-op here: the server's watcher is `#[cfg(unix)]`,
/// so nothing on that side ever reaps the child. Stopping the runner on exit
/// covers a clean quit, but not a crash, not an abort (the release profile sets
/// `panic = "abort"`, so destructors never run) and not "End task" from Task
/// Manager. A job object with kill-on-close is enforced by the kernel, so the
/// sidecar goes away however this process dies.
///
/// The consequences of getting this wrong are not cosmetic. A survivor keeps
/// the whole model resident, is invisible because it was spawned with
/// CREATE_NO_WINDOW, and holds its own executable open, which makes the next
/// installer run fail with "file in use".
#[cfg(target_os = "windows")]
fn confine_to_job_object(child: &Child) {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    // One job for the whole app, deliberately never closed: the kernel closes
    // it when this process ends, and that close is what kills the children.
    static JOB: OnceLock<isize> = OnceLock::new();

    let job = *JOB.get_or_init(|| unsafe {
        let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if handle.is_null() {
            return 0;
        }
        let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        let applied = SetInformationJobObject(
            handle,
            JobObjectExtendedLimitInformation,
            std::ptr::addr_of!(limits).cast(),
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if applied == 0 { 0 } else { handle as isize }
    });

    if job == 0 {
        // Losing the job object is not worth refusing to start over. The
        // ordinary shutdown path still stops the sidecar; only the abnormal
        // exits leak, which is exactly where we were before.
        tracing::warn!("could not confine the MamboTTS server to a job object");
        return;
    }

    let assigned =
        unsafe { AssignProcessToJobObject(job as _, child.as_raw_handle() as _) };
    if assigned == 0 {
        tracing::warn!("could not assign the MamboTTS server to the job object");
    }
}

/// Remove the `\\?\` extended-length prefix from a path.
///
/// espeak-ng joins the data directory with its own relative paths using forward
/// slashes. Windows normalises those to backslashes for an ordinary path but
/// deliberately does not inside a verbatim `\\?\` path, where the string reaches
/// the filesystem untouched, so every lookup under the directory fails. Tauri
/// hands back verbatim paths on Windows, so the prefix has to come off before
/// the value goes to a C library that will build paths out of it.
#[cfg(target_os = "windows")]
fn plain_path(path: &Path) -> PathBuf {
    use std::path::{Component, Prefix};

    let mut components = path.components();
    if let Some(Component::Prefix(prefix)) = components.next() {
        // Only a plain drive has a shorter spelling. A verbatim UNC path does
        // not, so it is left exactly as it came.
        if let Prefix::VerbatimDisk(letter) = prefix.kind() {
            let root = format!("{}:\\", letter as char);
            return PathBuf::from(root).join(components.as_path());
        }
    }
    path.to_path_buf()
}

#[cfg(not(target_os = "windows"))]
fn plain_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}
