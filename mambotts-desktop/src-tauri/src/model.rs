use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::{Emitter, Manager};

use crate::analytics;
use crate::download::{fetch_verified, has_expected_size, reuse_verified};
use mambotts_registry::{
    HebrewG2pPlan, HebrewG2pVariant, ModelFile, hebrew_g2p_plan, hebrew_g2p_variant, runtimes,
};

/// Bundles from before BlueV3 (MamboTTS 1.3.x). Their graphs and voices do
/// not run on BlueV3, so there is no background upgrade from them and the
/// user goes through onboarding; but RenikudPlus is the same revision, so
/// onboarding takes its file from here instead of fetching 1.2 GB again,
/// and the directory goes once the current bundle is installed.
const RETIRED_DIRECTORIES: &[&str] = &["bluetts-2.5"];

/// Models of engines MamboTTS no longer runs at all (the early Blue ONNX
/// bundle, the Qwen TTS experiment and the Phonikud G2P). Nothing reuses
/// their files; they go with the retired bundles once v3.0.1 is installed.
const OBSOLETE_DIRECTORIES: &[&str] = &["blue-onnx-v2", "qwentts-he-1.7b", "phonikud-v1"];

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
    /// Set while this is an earlier release's bundle (`version` names it):
    /// the release it is being upgraded to. The app keeps running on this
    /// one while `upgrade_model_bundle` fills the new directory.
    pub bundle_upgrade: Option<String>,
    /// The registry's optional files this machine would run faster with but
    /// does not have yet (the int8 vector estimator, off macOS):
    /// `fetch_optional_files` fetches them in the background. The bundle is
    /// installed and runs without them.
    pub optional_files_pending: Vec<String>,
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
    pub directory: String,
    pub capabilities: RuntimeCapabilities,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeCapabilities {
    pub hebrew: bool,
    pub streaming: bool,
    pub fixed_voices: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelSources {
    pub runtimes: Vec<ModelSource>,
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

/// Move an install of an earlier release (BlueV3 3.0.0) to the current one
/// in the background, while the app keeps running on the old bundle; see
/// [`ModelBundle::bundle_upgrade`].
///
/// Every file whose bytes did not change between the releases (the
/// vocabulary, the vocoder, polyg2p's data, RenikudPlus) is taken from the
/// old directory once its SHA-256 matches the registry, as a hard link where
/// possible; only the changed graphs and voices are downloaded, resumable
/// and verified like any download. Progress goes out as
/// `model_download_progress` with stage `upgrading`, then `upgraded`. The
/// returned bundle points at the new directory only once all of it is in
/// place and verified, and only then is the old directory removed.
#[tauri::command]
pub async fn upgrade_model_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    upgrade_model_bundle_inner(&app).await.map_err(|err| {
        analytics::track_error(
            &app,
            analytics::events::ERROR_MODEL_DOWNLOAD_FAILED,
            err,
            serde_json::json!({"operation": "upgrade_model_bundle", "runtime": "blue"}),
        )
    })
}

/// Fetch the optional files this machine runs faster with (see
/// [`ModelBundle::optional_files_pending`]) in the background, resumable and
/// verified like any download. No progress events: nothing waits for them,
/// and the engine picks them up the next time it loads the bundle.
#[tauri::command]
pub async fn fetch_optional_files(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    let inner = async {
        let _guard = download_lock().lock().await;
        let dir = blue_dir(&app)?;
        fetch_optional_files_in(&http_client()?, &dir).await?;
        blue_bundle(&app)
    };
    inner.await.map_err(|err| {
        analytics::track_error(
            &app,
            analytics::events::ERROR_MODEL_DOWNLOAD_FAILED,
            err,
            serde_json::json!({"operation": "fetch_optional_files", "runtime": "blue"}),
        )
    })
}

/// The optional files of the current release this build wants, fetched into
/// an installed bundle directory, for the command and the tests.
pub(crate) async fn fetch_optional_files_in(client: &reqwest::Client, dir: &Path) -> Result<(), String> {
    let blue = mambotts_registry::runtime("blue").expect("the Blue manifest");
    fetch_missing(client, dir, &blue.optional_files_here()).await
}

async fn fetch_missing(client: &reqwest::Client, dir: &Path, files: &[ModelFile]) -> Result<(), String> {
    for file in files {
        let destination = dir.join(file.name);
        if !has_expected_size(&destination, file) {
            fetch_verified(client, file, &destination, |_| {}).await?;
        }
    }
    Ok(())
}

async fn upgrade_model_bundle_inner(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let _guard = download_lock().lock().await;
    let root = models_root(app)?;
    let client = http_client()?;
    upgrade_bundle_in(&client, &root, hebrew_g2p_for_this_machine(), |progress| {
        emit_progress(app, progress)
    })
    .await?;
    blue_bundle(app)
}

/// The bundle upgrade itself, on a models directory, for the command and
/// the tests.
pub(crate) async fn upgrade_bundle_in(
    client: &reqwest::Client,
    root: &Path,
    policy: HebrewG2pVariant,
    mut emit: impl FnMut(ModelDownloadProgress),
) -> Result<(), String> {
    let blue = mambotts_registry::runtime("blue").expect("the Blue manifest");
    let dir = root.join(blue.directory);
    if blue_bundle_in(&dir, policy, blue.version.into()).installed {
        remove_previous_bundles(root);
        return Ok(());
    }
    let Some(old) = blue
        .previous_directories
        .iter()
        .map(|name| root.join(name))
        .find(|old| blue_bundle_in(old, policy, String::new()).installed)
    else {
        return Err("there is no earlier BlueTTS bundle to upgrade".into());
    };
    // RenikudPlus is the same revision in both releases: every variant the
    // old bundle holds comes along (an int8-only install stays int8 until
    // the RenikudPlus upgrade fetches fp32 in the new directory).
    let mut files = blue.files.to_vec();
    let carried: Vec<ModelFile> = [HebrewG2pVariant::Fp32, HebrewG2pVariant::Int8]
        .into_iter()
        .map(HebrewG2pVariant::model_file)
        .filter(|file| has_expected_size(&old.join(file.name), file))
        .collect();
    if carried.is_empty() {
        files.push(policy.model_file());
    }
    files.extend(carried);
    migrate_files(client, &old, &dir, &files, &mut emit).await?;
    if !blue_bundle_in(&dir, policy, blue.version.into()).installed {
        return Err(format!("the {} upgrade finished, but files are missing", blue.version));
    }
    emit(ModelDownloadProgress {
        downloaded: files.iter().map(|file| file.size).sum(),
        total: Some(files.iter().map(|file| file.size).sum()),
        progress: Some(1.0),
        stage: "upgraded",
    });
    remove_previous_bundles(root);
    Ok(())
}

/// Fill `dir` with `files`: kept when already there, taken from `old` when
/// its copy is byte-identical (verified), downloaded otherwise.
pub(crate) async fn migrate_files(
    client: &reqwest::Client,
    old: &Path,
    dir: &Path,
    files: &[ModelFile],
    emit: &mut impl FnMut(ModelDownloadProgress),
) -> Result<(), String> {
    let total: u64 = files.iter().map(|file| file.size).sum();
    let progress = |downloaded: u64| ModelDownloadProgress {
        downloaded,
        total: Some(total),
        progress: Some((downloaded as f64 / total.max(1) as f64).clamp(0.0, 0.99)),
        stage: "upgrading",
    };
    let mut done = 0_u64;
    for file in files {
        let destination = dir.join(file.name);
        if !has_expected_size(&destination, file)
            && !reuse_verified(file, &old.join(file.name), &destination).await?
        {
            fetch_verified(client, file, &destination, |bytes| emit(progress(done + bytes))).await?;
        }
        done += file.size;
        emit(progress(done));
    }
    Ok(())
}

/// Put a retired bundle's RenikudPlus in place for `policy` when its bytes
/// are the registry's (it is a hard link where possible, so the retired
/// directory's removal frees nothing twice). 1.3.x named its file
/// `renikud-plus.onnx` whichever weights it held, so both names are tried
/// and only the size and SHA-256 decide.
pub(crate) async fn adopt_retired_hebrew_g2p(
    root: &Path,
    dir: &Path,
    policy: HebrewG2pVariant,
) -> Result<(), String> {
    let file = policy.model_file();
    let dest = dir.join(file.name);
    if has_expected_size(&dest, &file) {
        return Ok(());
    }
    for retired in RETIRED_DIRECTORIES {
        for name in [HebrewG2pVariant::Int8, HebrewG2pVariant::Fp32].map(|v| v.model_file().name) {
            if reuse_verified(&file, &root.join(retired).join(name), &dest).await? {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Remove earlier releases' directories once the current one is installed.
/// A file a running server still holds open may not go on Windows; the
/// next start tries again.
fn remove_previous_bundles(root: &Path) {
    let blue = mambotts_registry::runtime("blue").expect("the Blue manifest");
    for name in blue
        .previous_directories
        .iter()
        .chain(RETIRED_DIRECTORIES)
        .chain(OBSOLETE_DIRECTORIES)
    {
        let dir = root.join(name);
        if dir.is_dir() {
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
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
    Ok(blue_bundle_at(&models_root(app)?, hebrew_g2p_for_this_machine()))
}

/// The bundle the app runs, in a models directory: the current release's
/// when it is installed (earlier releases' directories are then removed);
/// else the newest earlier release that is installed, marked for the
/// background upgrade; else the current one, not installed (onboarding).
pub(crate) fn blue_bundle_at(root: &Path, policy: HebrewG2pVariant) -> ModelBundle {
    let blue = mambotts_registry::runtime("blue").expect("the Blue manifest");
    let current = blue_bundle_in(&root.join(blue.directory), policy, blue.version.into());
    if current.installed {
        remove_previous_bundles(root);
        return current;
    }
    for name in blue.previous_directories {
        let mut previous = blue_bundle_in(&root.join(name), policy, (*name).into());
        if previous.installed {
            previous.bundle_upgrade = Some(blue.version.into());
            // RenikudPlus is upgraded in the new directory, after the bundle.
            previous.hebrew_g2p_upgrade = None;
            // ...and so are the optional files.
            previous.optional_files_pending.clear();
            return previous;
        }
    }
    current
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
        bundle_upgrade: None,
        optional_files_pending: blue
            .optional_files_here()
            .iter()
            .filter(|file| !has_expected_size(&dir.join(file.name), file))
            .map(|file| file.name.to_string())
            .collect(),
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
                directory: runtime.directory.into(),
                capabilities: RuntimeCapabilities {
                    hebrew: runtime.capabilities.hebrew,
                    streaming: runtime.capabilities.streaming,
                    fixed_voices: runtime.capabilities.fixed_voices,
                },
            })
            .collect(),
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
    adopt_retired_hebrew_g2p(&models_root(&app)?, &dir, policy).await?;
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
    fn a_retired_bundle_goes_once_the_current_one_is_installed() {
        let root = tempfile::tempdir().unwrap();
        let blue = mambotts_registry::runtime("blue").unwrap();
        let current = bundle_dir(&[Fp32]);
        std::fs::rename(current.path(), root.path().join(blue.directory)).unwrap();
        let retired = root.path().join(RETIRED_DIRECTORIES[0]);
        std::fs::create_dir_all(retired.join("voices")).unwrap();
        std::fs::write(retired.join("vocoder.onnx"), b"x").unwrap();

        assert!(blue_bundle_at(root.path(), Fp32).installed);
        assert!(!retired.exists());
    }

    #[test]
    fn other_engines_models_go_once_the_current_one_is_installed() {
        let root = tempfile::tempdir().unwrap();
        let blue = mambotts_registry::runtime("blue").unwrap();
        let current = bundle_dir(&[Fp32]);
        std::fs::rename(current.path(), root.path().join(blue.directory)).unwrap();
        for name in OBSOLETE_DIRECTORIES {
            let dir = root.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("model.onnx"), b"x").unwrap();
        }

        assert!(blue_bundle_at(root.path(), Fp32).installed);
        for name in OBSOLETE_DIRECTORIES {
            assert!(!root.path().join(name).exists(), "{name}");
        }
        assert!(root.path().join(blue.directory).is_dir());
    }

    /// A 1.3.x RenikudPlus of the right size but other bytes is not taken:
    /// onboarding downloads the registry's file instead.
    #[tokio::test]
    async fn a_retired_renikud_that_is_not_the_registrys_is_left_alone() {
        let root = tempfile::tempdir().unwrap();
        let retired = root.path().join(RETIRED_DIRECTORIES[0]);
        std::fs::create_dir_all(&retired).unwrap();
        std::fs::File::create(retired.join("renikud-plus.onnx"))
            .unwrap()
            .set_len(Int8.model_file().size)
            .unwrap();
        let dir = root.path().join("current");
        adopt_retired_hebrew_g2p(root.path(), &dir, Int8).await.unwrap();
        assert!(!dir.join(Int8.file_name()).exists());
    }

    /// 1.3.x's real RenikudPlus (fp32 under the int8 name) is taken over:
    /// `MAMBOTTS_RETIRED_RENIKUD` names a bluetts-2.5 `renikud-plus.onnx`.
    #[tokio::test]
    #[ignore = "needs MAMBOTTS_RETIRED_RENIKUD pointing at a 1.3.x renikud-plus.onnx"]
    async fn a_retired_renikud_with_the_registrys_bytes_is_taken_over() {
        let source = PathBuf::from(std::env::var("MAMBOTTS_RETIRED_RENIKUD").unwrap());
        let root = tempfile::tempdir().unwrap();
        let retired = root.path().join(RETIRED_DIRECTORIES[0]);
        std::fs::create_dir_all(&retired).unwrap();
        std::fs::copy(&source, retired.join("renikud-plus.onnx")).unwrap();
        let dir = root.path().join("current");
        adopt_retired_hebrew_g2p(root.path(), &dir, Fp32).await.unwrap();
        assert!(has_expected_size(&dir.join(Fp32.file_name()), &Fp32.model_file()));
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

    /// A models directory with an earlier release's complete bundle.
    fn root_with_previous(variants: &[HebrewG2pVariant]) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let previous = bundle_dir(variants);
        let blue = mambotts_registry::runtime("blue").unwrap();
        std::fs::rename(previous.path(), root.path().join(blue.previous_directories[0])).unwrap();
        root
    }

    #[test]
    fn an_earlier_release_keeps_running_and_asks_for_the_upgrade() {
        let blue = mambotts_registry::runtime("blue").unwrap();
        let root = root_with_previous(&[Int8]);
        let bundle = blue_bundle_at(root.path(), Fp32);
        assert!(bundle.installed, "no onboarding for an upgrader");
        assert_eq!(bundle.version, blue.previous_directories[0]);
        assert!(bundle.model_dir.ends_with(blue.previous_directories[0]));
        assert_eq!(bundle.bundle_upgrade.as_deref(), Some(blue.version));
        // RenikudPlus waits for the new directory.
        assert_eq!(bundle.hebrew_g2p_upgrade, None);
        assert_eq!(bundle.hebrew_g2p, Some(Int8));

        // The current release installed: it runs, and the old one goes.
        let current = root.path().join(blue.directory);
        std::fs::rename(bundle_dir(&[Fp32]).path(), &current).unwrap();
        let bundle = blue_bundle_at(root.path(), Fp32);
        assert!(bundle.installed);
        assert_eq!(bundle.version, blue.version);
        assert_eq!(bundle.bundle_upgrade, None);
        assert!(!root.path().join(blue.previous_directories[0]).exists());

        // Nothing at all: the current release, not installed.
        let empty = tempfile::tempdir().unwrap();
        let bundle = blue_bundle_at(empty.path(), Fp32);
        assert!(!bundle.installed);
        assert!(bundle.model_dir.ends_with(blue.directory));
    }

    /// The migration: unchanged files come from the old directory without a
    /// request, a changed one is downloaded, a corrupt old copy is not used,
    /// and the old directory is left alone.
    #[tokio::test]
    async fn the_upgrade_reuses_identical_files_and_downloads_the_rest() {
        let same = body(4_000);
        let changed_old = body(3_000);
        let changed_new: Vec<u8> = body(3_500).into_iter().rev().collect();
        let corrupt = body(2_000);
        let server_same = serve(same.clone(), ServerOptions::default()).await;
        let server_new = serve(changed_new.clone(), ServerOptions::default()).await;
        let server_corrupt = serve(corrupt.clone(), ServerOptions::default()).await;
        let mut vocoder = model_file(&server_same.url, &same);
        vocoder.name = "vocoder.onnx";
        let mut voice = model_file(&server_new.url, &changed_new);
        voice.name = "voices/noa.json";
        let mut lexicon = model_file(&server_corrupt.url, &corrupt);
        lexicon.name = "polyg2p/en.lex";

        let old = tempfile::tempdir().unwrap();
        let new = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(old.path().join("voices")).unwrap();
        std::fs::create_dir_all(old.path().join("polyg2p")).unwrap();
        std::fs::write(old.path().join("vocoder.onnx"), &same).unwrap();
        std::fs::write(old.path().join("voices/noa.json"), &changed_old).unwrap();
        // Right size, wrong bytes: must be downloaded, not linked.
        let mut flipped = corrupt.clone();
        flipped[0] ^= 0xff;
        std::fs::write(old.path().join("polyg2p/en.lex"), &flipped).unwrap();

        let mut stages = Vec::new();
        migrate_files(
            &reqwest::Client::new(),
            old.path(),
            new.path(),
            &[vocoder, voice, lexicon],
            &mut |p: ModelDownloadProgress| stages.push((p.stage, p.downloaded)),
        )
        .await
        .unwrap();
        use std::sync::atomic::Ordering::SeqCst;
        assert_eq!(server_same.requests.load(SeqCst), 0, "an identical file is not downloaded");
        assert_eq!(server_new.requests.load(SeqCst), 1);
        assert_eq!(server_corrupt.requests.load(SeqCst), 1);
        assert_eq!(std::fs::read(new.path().join("vocoder.onnx")).unwrap(), same);
        assert_eq!(std::fs::read(new.path().join("voices/noa.json")).unwrap(), changed_new);
        assert_eq!(std::fs::read(new.path().join("polyg2p/en.lex")).unwrap(), corrupt);
        // The old bundle is untouched (the app was running on it).
        assert_eq!(std::fs::read(old.path().join("voices/noa.json")).unwrap(), changed_old);
        assert_eq!(std::fs::read(old.path().join("vocoder.onnx")).unwrap(), same);
        assert!(stages.iter().all(|(stage, _)| *stage == "upgrading"));
        assert_eq!(stages.last().unwrap().1, 4_000 + 3_500 + 2_000);
    }

    /// An install without the int8 vector estimator is complete; off macOS
    /// it asks for the file in the background, and once it is there, not.
    #[test]
    fn an_install_without_the_optional_files_is_installed_and_asks_for_them_off_macos() {
        let dir = bundle_dir(&[Fp32]);
        let blue = mambotts_registry::runtime("blue").unwrap();
        let bundle = blue_bundle_in(dir.path(), Fp32, "v".into());
        assert!(bundle.installed);
        if cfg!(target_os = "macos") {
            assert!(bundle.optional_files_pending.is_empty());
            return;
        }
        assert_eq!(bundle.optional_files_pending, ["vector_estimator.int8.onnx"]);
        for file in blue.optional_files_here() {
            std::fs::File::create(dir.path().join(file.name)).unwrap().set_len(file.size).unwrap();
        }
        assert!(blue_bundle_in(dir.path(), Fp32, "v".into()).optional_files_pending.is_empty());
    }

    #[tokio::test]
    async fn optional_files_are_fetched_verified_and_kept() {
        let bytes = body(6_000);
        let server = serve(bytes.clone(), ServerOptions::default()).await;
        let mut file = model_file(&server.url, &bytes);
        file.name = "vector_estimator.int8.onnx";
        let dir = tempfile::tempdir().unwrap();
        let client = reqwest::Client::new();
        fetch_missing(&client, dir.path(), &[file]).await.unwrap();
        fetch_missing(&client, dir.path(), &[file]).await.unwrap();
        assert_eq!(std::fs::read(dir.path().join(file.name)).unwrap(), bytes);
        assert_eq!(server.requests.load(std::sync::atomic::Ordering::SeqCst), 1, "a complete file is kept");
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
