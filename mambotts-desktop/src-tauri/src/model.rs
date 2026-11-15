use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::{Emitter, Manager};

use crate::analytics;
use crate::download::{fetch_verified, has_expected_size};
use mambotts_registry::{
    HebrewG2pPlan, HebrewG2pVariant, ModelFile, hebrew_g2p_plan, hebrew_g2p_variant, runtimes,
};

/// One download into the models directory at a time: the onboarding's bundle
/// download and the background RenikudPlus upgrade write the same `.part`.
fn download_lock() -> &'static tokio::sync::Mutex<()> {
    static LOCK: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| tokio::sync::Mutex::new(()))
}

/// The RenikudPlus weights this machine runs, by bluetts.com's policy
/// (backend-policy.js `hebrewG2PVariant`): fp32 on a computer, int8 when it
/// has 4 GB of memory or less. `MAMBOTTS_HEBREW_G2P=fp32|int8` overrides it.
pub fn hebrew_g2p_for_this_machine() -> HebrewG2pVariant {
    let mut system = sysinfo::System::new();
    system.refresh_memory();
    let total = system.total_memory();
    let setting = std::env::var("MAMBOTTS_HEBREW_G2P").ok();
    hebrew_g2p_variant((total > 0).then_some(total), setting.as_deref())
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBundle {
    pub installed: bool,
    pub runtime: String,
    pub model_path: String,
    pub codec_path: String,
    /// Hebrew G2P model. BlueTTS consumes IPA rather than Hebrew script,
    /// so this is not optional.
    pub renikud_path: String,
    pub voices_path: Option<String>,
    pub model_dir: String,
    pub version: String,
    pub url: String,
    /// The RenikudPlus weights `renikud_path` names.
    pub hebrew_g2p: Option<HebrewG2pVariant>,
    /// Set while the machine's own RenikudPlus is still to come: an install
    /// from before the fp32 default runs int8 meanwhile, and
    /// `upgrade_hebrew_g2p` fetches fp32 in the background.
    pub hebrew_g2p_upgrade: Option<HebrewG2pVariant>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSourceFile {
    pub name: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSource {
    pub id: String,
    pub name: String,
    pub version: String,
    pub size: String,
    pub description: String,
    pub files: Vec<ModelSourceFile>,
    pub archive_url: Option<String>,
    pub directory: String,
    pub capabilities: RuntimeCapabilities,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCapabilities {
    pub hebrew: bool,
    pub streaming: bool,
    pub voice_reference: bool,
    pub fixed_voices: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSources {
    pub runtimes: Vec<ModelSource>,
    pub voices_url: String,
    pub default_paths: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ModelDownloadProgress {
    downloaded: u64,
    total: Option<u64>,
    progress: Option<f64>,
    stage: &'static str,
}

#[tauri::command]
pub async fn get_model_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    model_bundle_for_runtime(&app, "blue")
}

#[tauri::command]
pub async fn get_model_bundle_for_runtime(
    app: tauri::AppHandle,
    runtime: String,
) -> Result<ModelBundle, String> {
    model_bundle_for_runtime(&app, &runtime)
}

#[tauri::command]
pub fn get_model_sources() -> ModelSources {
    model_sources()
}

#[tauri::command]
pub async fn download_model_bundle(
    app: tauri::AppHandle,
    runtime: Option<String>,
) -> Result<ModelBundle, String> {
    let runtime = runtime.unwrap_or_else(|| "blue".to_string());
    download_model_bundle_inner(app.clone(), runtime.clone())
        .await
        .map_err(|err| {
            analytics::track_error(
                &app,
                analytics::events::ERROR_MODEL_DOWNLOAD_FAILED,
                err,
                serde_json::json!({"operation": "download_model_bundle", "runtime": runtime}),
            )
        })
}

/// Fetch the machine's RenikudPlus in the background while the app keeps
/// running on the variant it has; see [`ModelBundle::hebrew_g2p_upgrade`].
///
/// Progress goes out as `model_download_progress` with stage `upgrading`.
/// The download resumes from its `.part` after a dropped connection or a
/// restart, and the file takes its name only once its size and SHA-256 match
/// the registry. Then the unused variant is removed (int8, on an fp32
/// machine) and the returned bundle points at the new file, which the next
/// `load_model` picks up.
#[tauri::command]
pub async fn upgrade_hebrew_g2p(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    upgrade_hebrew_g2p_inner(&app).await.map_err(|err| {
        analytics::track_error(
            &app,
            analytics::events::ERROR_MODEL_DOWNLOAD_FAILED,
            err,
            serde_json::json!({"operation": "upgrade_hebrew_g2p", "runtime": "blue"}),
        )
    })
}

async fn upgrade_hebrew_g2p_inner(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let _guard = download_lock().lock().await;
    let dir = blue_dir(app)?;
    let client = http_client()?;
    upgrade_hebrew_g2p_in(&client, &dir, hebrew_g2p_for_this_machine(), |progress| {
        emit_progress(app, progress)
    })
    .await?;
    blue_bundle(app)
}

/// The upgrade itself, on a directory, for the command and the tests.
/// Returns the plan it leaves behind.
pub(crate) async fn upgrade_hebrew_g2p_in(
    client: &reqwest::Client,
    dir: &Path,
    policy: HebrewG2pVariant,
    mut emit: impl FnMut(ModelDownloadProgress),
) -> Result<HebrewG2pPlan, String> {
    let plan = hebrew_g2p_plan(dir, policy);
    if let Some(variant) = plan.upgrade {
        let file = variant.model_file();
        upgrade_file(client, &file, dir, &mut emit).await?;
    }
    finish_hebrew_g2p(dir, policy).await
}

async fn upgrade_file(
    client: &reqwest::Client,
    file: &ModelFile,
    dir: &Path,
    emit: &mut impl FnMut(ModelDownloadProgress),
) -> Result<(), String> {
    let total = file.size;
    fetch_verified(client, file, &dir.join(file.name), |downloaded| {
        emit(ModelDownloadProgress {
            downloaded,
            total: Some(total),
            progress: Some((downloaded as f64 / total as f64).clamp(0.0, 0.99)),
            stage: "upgrading",
        })
    })
    .await?;
    emit(ModelDownloadProgress {
        downloaded: total,
        total: Some(total),
        progress: Some(1.0),
        stage: "upgraded",
    });
    Ok(())
}

/// Remove the RenikudPlus variant the plan no longer needs.
async fn finish_hebrew_g2p(dir: &Path, policy: HebrewG2pVariant) -> Result<HebrewG2pPlan, String> {
    let plan = hebrew_g2p_plan(dir, policy);
    if let Some(unused) = plan.remove {
        let path = dir.join(unused.file_name());
        // On Windows a file the server still has open cannot be removed; the
        // next start removes it (see `blue_bundle`).
        let _ = tokio::fs::remove_file(&path).await;
        let _ = tokio::fs::remove_file(crate::download::part_path(&path)).await;
    }
    Ok(hebrew_g2p_plan(dir, policy))
}

fn http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))
}

async fn download_model_bundle_inner(
    app: tauri::AppHandle,
    runtime: String,
) -> Result<ModelBundle, String> {
    match runtime.as_str() {
        "blue" => download_runtime_bundle(app, &runtime).await,
        other => Err(format!("unsupported runtime `{other}`")),
    }
}

pub fn model_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    blue_bundle(app)
}

pub fn model_bundle_for_runtime(
    app: &tauri::AppHandle,
    runtime: &str,
) -> Result<ModelBundle, String> {
    match runtime {
        "blue" => blue_bundle(app),
        other => Err(format!("unsupported runtime `{other}`")),
    }
}

fn blue_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let source = runtime_source("blue").ok_or_else(|| "missing Blue source".to_string())?;
    Ok(models_root(app)?.join(&source.directory))
}

fn blue_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let source = runtime_source("blue").ok_or_else(|| "missing Blue source".to_string())?;
    let dir = blue_dir(app)?;
    Ok(blue_bundle_in(&dir, hebrew_g2p_for_this_machine(), source.version))
}

/// The bundle a directory holds, under a RenikudPlus policy.
pub(crate) fn blue_bundle_in(dir: &Path, policy: HebrewG2pVariant, version: String) -> ModelBundle {
    // Drop the old thewh1teagle Renikud file so we never load or re-count it.
    let legacy_renikud = dir.join("renikud.onnx");
    if legacy_renikud.is_file() {
        let _ = std::fs::remove_file(&legacy_renikud);
    }
    let mut plan = hebrew_g2p_plan(dir, policy);
    if let Some(unused) = plan.remove {
        // Left behind when an upgrade could not remove it (Windows keeps a
        // loaded file locked); nothing loads it any more.
        let _ = std::fs::remove_file(dir.join(unused.file_name()));
        plan = hebrew_g2p_plan(dir, policy);
    }
    let blue = mambotts_registry::runtime("blue").expect("the Blue manifest");
    let renikud_path = dir.join(plan.load.unwrap_or(policy).file_name());
    ModelBundle {
        // Every file of the manifest, and some RenikudPlus to load. An
        // install from before the fp32 default holds only the int8 file: it
        // is installed and runs int8 while `hebrew_g2p_upgrade` asks for
        // fp32 in the background, instead of sending the user back through
        // onboarding for a 1.2 GB download.
        installed: plan.load.is_some()
            && blue
                .files
                .iter()
                // An empty file is a download that never got its bytes.
                .all(|file| {
                    std::fs::metadata(dir.join(file.name))
                        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
                }),
        runtime: "blue".to_string(),
        model_path: path_string(dir),
        codec_path: path_string(&renikud_path),
        renikud_path: path_string(&renikud_path),
        voices_path: Some(path_string(&dir.join("voices"))),
        model_dir: path_string(dir),
        version,
        url: mambotts_registry::blue_model_base_url().to_string(),
        hebrew_g2p: plan.load,
        hebrew_g2p_upgrade: plan.upgrade,
    }
}

fn model_sources() -> ModelSources {
    let variant = hebrew_g2p_for_this_machine();
    ModelSources {
        runtimes: runtimes()
            .iter()
            .map(|runtime| ModelSource {
                id: runtime.id.into(),
                name: runtime.name.into(),
                version: runtime.version.into(),
                size: runtime.size_for(variant).into(),
                description: runtime.description.into(),
                files: runtime
                    .files_for(variant)
                    .iter()
                    .map(|file| ModelSourceFile {
                        name: file.name.into(),
                        url: file.url.into(),
                    })
                    .collect(),
                archive_url: None,
                directory: runtime.directory.into(),
                capabilities: RuntimeCapabilities {
                    hebrew: runtime.capabilities.hebrew,
                    streaming: runtime.capabilities.streaming,
                    voice_reference: runtime.capabilities.voice_reference,
                    fixed_voices: runtime.capabilities.fixed_voices,
                },
            })
            .collect(),
        voices_url: String::new(),
        default_paths: vec![
            "macOS: ~/Library/Application Support/com.maxmelichov.mambotts/models".to_string(),
            "Windows: %LOCALAPPDATA%\\com.maxmelichov.mambotts\\models".to_string(),
            "Linux: ~/.local/share/com.maxmelichov.mambotts/models".to_string(),
        ],
    }
}

fn runtime_source(runtime: &str) -> Option<ModelSource> {
    model_sources()
        .runtimes
        .into_iter()
        .find(|source| source.id == runtime)
}

async fn download_runtime_bundle(
    app: tauri::AppHandle,
    runtime: &str,
) -> Result<ModelBundle, String> {
    let _guard = download_lock().lock().await;
    let manifest =
        mambotts_registry::runtime(runtime).ok_or_else(|| format!("missing `{runtime}` source"))?;
    let dir = blue_dir(&app)?;
    let policy = hebrew_g2p_for_this_machine();
    let client = http_client()?;
    download_bundle_in(&client, &dir, &manifest.files_for(policy), |progress| {
        emit_progress(&app, progress)
    })
    .await?;
    finish_hebrew_g2p(&dir, policy).await?;
    let bundle = model_bundle_for_runtime(&app, runtime)?;
    if !bundle.installed {
        return Err(format!(
            "`{runtime}` download completed, but required files are missing"
        ));
    }
    Ok(bundle)
}

/// Download every file of a bundle that is not already in place.
///
/// No early return when every file is present: a download is always a
/// user's request, and checking each file against the registry's size is
/// what repairs a truncated or outdated file that would otherwise count as
/// installed for good. Files of the right size are kept (their hash was
/// checked when they got their name); the others are fetched, resuming any
/// `.part` an earlier attempt left, and verified.
pub(crate) async fn download_bundle_in(
    client: &reqwest::Client,
    dir: &Path,
    files: &[ModelFile],
    mut emit: impl FnMut(ModelDownloadProgress),
) -> Result<(), String> {
    tokio::fs::create_dir_all(dir)
        .await
        .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let total: u64 = files.iter().map(|file| file.size).sum();
    let progress = |downloaded: u64| ModelDownloadProgress {
        downloaded,
        total: Some(total),
        progress: Some((downloaded as f64 / total.max(1) as f64).clamp(0.0, 0.99)),
        stage: "downloading",
    };
    let mut done = 0_u64;
    for file in files {
        let destination = dir.join(file.name);
        if !has_expected_size(&destination, file) {
            fetch_verified(client, file, &destination, |bytes| emit(progress(done + bytes)))
                .await?;
        }
        done += file.size;
        emit(progress(done));
    }
    emit(ModelDownloadProgress {
        downloaded: total,
        total: Some(total),
        progress: Some(1.0),
        stage: "downloading",
    });
    Ok(())
}

fn models_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_local_data_dir()
        .map_err(|err| format!("failed to resolve app data dir: {err}"))?
        .join("models");
    Ok(dir)
}

pub(crate) fn emit_progress(app: &tauri::AppHandle, payload: ModelDownloadProgress) {
    let _ = app.emit("model_download_progress", payload);
}

fn path_string(path: &Path) -> String {
    path.as_os_str().to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    //! The upgrade an int8-only install goes through, against a local server
    //! (see `download::tests`), on a real directory.

    use super::*;
    use crate::download::tests::{ServerOptions, body, model_file, serve};
    use mambotts_registry::HebrewG2pVariant::{Fp32, Int8};

    /// A bundle directory with every manifest file present (tiny stand-ins)
    /// and the given RenikudPlus files at their registry sizes (sparse).
    fn bundle_dir(variants: &[HebrewG2pVariant]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        for file in mambotts_registry::runtime("blue").unwrap().files {
            let path = dir.path().join(file.name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"x").unwrap();
        }
        for variant in variants {
            std::fs::File::create(dir.path().join(variant.file_name()))
                .unwrap()
                .set_len(variant.model_file().size)
                .unwrap();
        }
        dir
    }

    #[test]
    fn an_int8_only_install_is_installed_and_asks_for_fp32() {
        let dir = bundle_dir(&[Int8]);
        let bundle = blue_bundle_in(dir.path(), Fp32, "v".into());
        assert!(bundle.installed, "no onboarding for an upgrader");
        assert_eq!(bundle.hebrew_g2p, Some(Int8));
        assert_eq!(bundle.hebrew_g2p_upgrade, Some(Fp32));
        assert!(bundle.renikud_path.ends_with(Int8.file_name()));

        // A low-memory machine is done: int8 is its own variant.
        let bundle = blue_bundle_in(dir.path(), Int8, "v".into());
        assert_eq!((bundle.hebrew_g2p, bundle.hebrew_g2p_upgrade), (Some(Int8), None));
    }

    #[test]
    fn without_any_renikud_the_bundle_is_not_installed() {
        let dir = bundle_dir(&[]);
        let bundle = blue_bundle_in(dir.path(), Fp32, "v".into());
        assert!(!bundle.installed);
        assert_eq!(bundle.hebrew_g2p_upgrade, None);
    }

    #[test]
    fn a_leftover_int8_goes_once_fp32_is_in_place() {
        // An upgrade that could not delete int8 (a locked file on Windows).
        let dir = bundle_dir(&[Int8, Fp32]);
        let bundle = blue_bundle_in(dir.path(), Fp32, "v".into());
        assert_eq!((bundle.hebrew_g2p, bundle.hebrew_g2p_upgrade), (Some(Fp32), None));
        assert!(!dir.path().join(Int8.file_name()).exists());
    }

    /// The whole flow with a small stand-in for the fp32 file: int8 keeps
    /// working, the download drops and resumes, the file is verified, and
    /// only then is int8 deleted and the bundle switched.
    #[tokio::test]
    async fn the_background_upgrade_resumes_verifies_switches_and_cleans_up() {
        let bytes = body(200_000);
        let server = serve(
            bytes.clone(),
            ServerOptions {
                cut_after: Some(80_000),
                ..Default::default()
            },
        )
        .await;
        let file = model_file(&server.url, &bytes);
        let dir = bundle_dir(&[Int8]);
        let client = reqwest::Client::new();
        let mut stages = Vec::new();

        // First attempt drops: still on int8, nothing deleted.
        let err = upgrade_file(&client, &file, dir.path(), &mut |p: ModelDownloadProgress| {
            stages.push((p.stage, p.downloaded))
        })
        .await
        .unwrap_err();
        assert!(err.contains("resume"), "{err}");
        assert!(dir.path().join(Int8.file_name()).exists());
        assert!(!dir.path().join(file.name).exists());
        assert!(stages.iter().all(|(stage, _)| *stage == "upgrading"));
        let have = std::fs::metadata(crate::download::part_path(&dir.path().join(file.name)))
            .unwrap()
            .len();
        assert!(have > 0 && have <= 80_000, "{have}");

        // Second attempt resumes from what arrived and finishes.
        server.options.lock().unwrap().cut_after = None;
        stages.clear();
        upgrade_file(&client, &file, dir.path(), &mut |p: ModelDownloadProgress| {
            stages.push((p.stage, p.downloaded))
        })
        .await
        .unwrap();
        assert_eq!(stages.first(), Some(&("upgrading", have)));
        assert_eq!(stages.last(), Some(&("upgraded", 200_000)));
        assert_eq!(
            server.ranges.lock().unwrap().last().unwrap().clone(),
            Some(format!("{have}-"))
        );
        assert_eq!(std::fs::read(dir.path().join(file.name)).unwrap(), bytes);
    }

    #[tokio::test]
    async fn finishing_deletes_int8_only_on_an_fp32_machine() {
        // fp32 machine with fp32 verified: int8 goes, the bundle switches.
        let dir = bundle_dir(&[Int8, Fp32]);
        let plan = finish_hebrew_g2p(dir.path(), Fp32).await.unwrap();
        assert_eq!(plan.load, Some(Fp32));
        assert!(!dir.path().join(Int8.file_name()).exists());
        let bundle = blue_bundle_in(dir.path(), Fp32, "v".into());
        assert!(bundle.renikud_path.ends_with(Fp32.file_name()));

        // fp32 machine still waiting for fp32: int8 stays.
        let dir = bundle_dir(&[Int8]);
        finish_hebrew_g2p(dir.path(), Fp32).await.unwrap();
        assert!(dir.path().join(Int8.file_name()).exists());

        // Low-memory machine with only fp32: nothing is deleted or fetched.
        let dir = bundle_dir(&[Fp32]);
        let plan = upgrade_hebrew_g2p_in(&reqwest::Client::new(), dir.path(), Int8, |_| {
            panic!("no download")
        })
        .await
        .unwrap();
        assert_eq!(plan.load, Some(Fp32));
        assert!(dir.path().join(Fp32.file_name()).exists());
    }

    #[tokio::test]
    async fn a_bundle_download_keeps_complete_files_and_fetches_the_rest() {
        let a = body(5_000);
        let b = body(7_000);
        let server_a = serve(a.clone(), ServerOptions::default()).await;
        let server_b = serve(b.clone(), ServerOptions::default()).await;
        let mut file_a = model_file(&server_a.url, &a);
        file_a.name = "a.onnx";
        let mut file_b = model_file(&server_b.url, &b);
        file_b.name = "voices/b.json";
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.onnx"), &a).unwrap();
        let mut last = None;
        download_bundle_in(&reqwest::Client::new(), dir.path(), &[file_a, file_b], |p| {
            last = Some((p.downloaded, p.total))
        })
        .await
        .unwrap();
        assert_eq!(server_a.requests.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(std::fs::read(dir.path().join("voices/b.json")).unwrap(), b);
        assert_eq!(last, Some((12_000, Some(12_000))));
    }
}
