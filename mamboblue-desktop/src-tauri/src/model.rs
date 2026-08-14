use flate2::read::GzDecoder;
use futures_util::StreamExt;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use tar::Archive;
use tauri::{Emitter, Manager};
use tokio::io::AsyncWriteExt;

use crate::analytics;
use mamborambo_registry::runtimes;

const MODELS_TAG: &str = "mamborambo-models-v0.1.3";
const MODEL_DIR: &str = "mamborambo-models-q5_0";
const MODEL_FILE: &str = "qwen3-tts-model.gguf";
const CODEC_FILE: &str = "qwen3-tts-codec.gguf";
const MODEL_BASE_URL: &str = "https://huggingface.co/thewh1teagle/qwen3-tts-gguf/resolve/main";
const KOKORO_MODELS_TAG: &str = "kokoro-v1.0";
const KOKORO_MODEL_DIR: &str = "mamborambo-kokoro-models-kokoro-v1.0";
const KOKORO_MODEL_FILE: &str = "kokoro-v1.0.onnx";
const KOKORO_VOICES_FILE: &str = "voices-v1.0.bin";
const KOKORO_ESPEAK_DIR: &str = "espeak-ng-data";
const KOKORO_BUNDLE_URL: &str = "https://huggingface.co/maxmelichov/MamboRambo-kokoro-models/resolve/main/mamborambo-kokoro-models-kokoro-v1.0.tar.gz";
const BLUE_MODELS_TAG: &str = "blue-onnx-v2";
const BLUE_MODEL_DIR: &str = "blue-onnx-v2";
const BLUE_MODEL_BASE_URL: &str = "https://huggingface.co/notmax123/blue-onnx-v2/resolve/main";
const RENIKUD_URL: &str = "https://huggingface.co/notmax123/RenikudPlus/resolve/main/model.onnx";
const PHONIKUD_URL: &str = "https://huggingface.co/Phonikud/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx";

#[derive(Debug, Clone, Serialize)]
pub struct ModelBundle {
    pub installed: bool,
    pub runtime: String,
    pub model_path: String,
    pub codec_path: String,
    pub voices_path: Option<String>,
    pub espeak_data_path: Option<String>,
    pub model_dir: String,
    pub version: String,
    pub url: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PhonikudBundle {
    pub installed: bool,
    pub path: String,
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
struct ModelDownloadProgress {
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
pub fn get_phonikud_bundle(app: tauri::AppHandle) -> Result<PhonikudBundle, String> {
    phonikud_bundle(&app)
}

#[tauri::command]
pub async fn download_phonikud_bundle(app: tauri::AppHandle) -> Result<PhonikudBundle, String> {
    let bundle = phonikud_bundle(&app)?;
    if bundle.installed {
        return Ok(bundle);
    }
    let path = PathBuf::from(&bundle.path);
    tokio::fs::create_dir_all(path.parent().ok_or("invalid Phonikud model path")?)
        .await
        .map_err(|err| format!("failed to create Phonikud model directory: {err}"))?;
    let client = reqwest::Client::builder().no_proxy().build().map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let mut downloaded = 0;
    let mut total = remote_content_length(&client, PHONIKUD_URL).await;
    download_model_file(&app, &client, PHONIKUD_URL, &path, &mut downloaded, &mut total).await?;
    phonikud_bundle(&app)
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

async fn download_model_bundle_inner(
    app: tauri::AppHandle,
    runtime: String,
) -> Result<ModelBundle, String> {
    if runtime != "blue" {
        return Err(format!(
            "unsupported runtime `{runtime}`; BlueTTS is the only available runtime"
        ));
    }
    return download_blue_bundle(app).await;
    /*
    let bundle = qwen_bundle(&app)?;
    if bundle.installed {
        return Ok(bundle);
    }

    let models_root = models_root(&app)?;
    tokio::fs::create_dir_all(&models_root)
        .await
        .map_err(|err| format!("failed to create {}: {err}", models_root.display()))?;
    let dir = model_dir(&app)?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;

    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let mut downloaded = 0_u64;
    let source = runtime_source("qwen").ok_or_else(|| "missing qwen source".to_string())?;
    let model_url = source
        .files
        .iter()
        .find(|file| file.name == MODEL_FILE)
        .map(|file| file.url.clone())
        .ok_or_else(|| "missing qwen model source URL".to_string())?;
    let codec_url = source
        .files
        .iter()
        .find(|file| file.name == CODEC_FILE)
        .map(|file| file.url.clone())
        .ok_or_else(|| "missing qwen codec source URL".to_string())?;
    let model_total = remote_content_length(&client, &model_url).await;
    let codec_total = remote_content_length(&client, &codec_url).await;
    let total = match (model_total, codec_total) {
        (Some(model), Some(codec)) => Some(model + codec),
        _ => None,
    };

    download_model_file(
        &app,
        &client,
        &model_url,
        &dir.join(MODEL_FILE),
        &mut downloaded,
        total,
    )
    .await?;
    download_model_file(
        &app,
        &client,
        &codec_url,
        &dir.join(CODEC_FILE),
        &mut downloaded,
        total,
    )
    .await?;

    let bundle = qwen_bundle(&app)?;
    if !bundle.installed {
        return Err("model files downloaded, but expected GGUF files were not found".to_string());
    }
    Ok(bundle)*/
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
        other => Err(format!(
            "unsupported runtime `{other}`; BlueTTS is the only available runtime"
        )),
    }
}

fn blue_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let source = runtime_source("blue").ok_or_else(|| "missing Blue source".to_string())?;
    let dir = models_root(app)?.join(BLUE_MODEL_DIR);
    let renikud_path = dir.join("renikud-plus.onnx");
    // Drop the old thewh1teagle Renikud file so we never load or re-count it.
    let legacy_renikud = dir.join("renikud.onnx");
    if legacy_renikud.is_file() {
        let _ = std::fs::remove_file(&legacy_renikud);
    }
    let required = [
        "duration_predictor.onnx",
        "text_encoder.onnx",
        "vector_estimator.onnx",
        "vocoder.onnx",
        "vocab.json",
        "tts.json",
        "voices/female1.json",
        "voices/male1.json",
        "renikud-plus.onnx",
    ];
    Ok(ModelBundle {
        installed: required.iter().all(|file| dir.join(file).is_file()),
        runtime: "blue".to_string(),
        model_path: path_string(&dir),
        codec_path: path_string(&renikud_path),
        voices_path: Some(path_string(&dir.join("voices"))),
        espeak_data_path: None,
        model_dir: path_string(&dir),
        version: source.version,
        url: BLUE_MODEL_BASE_URL.to_string(),
    })
}

fn phonikud_bundle(app: &tauri::AppHandle) -> Result<PhonikudBundle, String> {
    let path = models_root(app)?
        .join("phonikud-v1")
        .join("phonikud-1.0.int8.onnx");
    Ok(PhonikudBundle {
        installed: path.is_file(),
        path: path_string(&path),
    })
}

fn qwen_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let source = runtime_source("qwen").ok_or_else(|| "missing qwen source".to_string())?;
    let dir = model_dir(app)?;
    let model_path = dir.join(MODEL_FILE);
    let codec_path = dir.join(CODEC_FILE);
    Ok(ModelBundle {
        installed: model_path.exists() && codec_path.exists(),
        runtime: "qwen".to_string(),
        model_path: path_string(&model_path),
        codec_path: path_string(&codec_path),
        voices_path: None,
        espeak_data_path: None,
        model_dir: path_string(&dir),
        version: source.version,
        url: MODEL_BASE_URL.to_string(),
    })
}

fn kokoro_bundle(app: &tauri::AppHandle) -> Result<ModelBundle, String> {
    let source = runtime_source("kokoro").ok_or_else(|| "missing kokoro source".to_string())?;
    let dir = models_root(app)?.join(KOKORO_MODEL_DIR);
    let model_path = dir.join(KOKORO_MODEL_FILE);
    let voices_path = dir.join(KOKORO_VOICES_FILE);
    let espeak_data_path = dir.join(KOKORO_ESPEAK_DIR);
    Ok(ModelBundle {
        installed: model_path.exists() && voices_path.exists() && espeak_data_path.is_dir(),
        runtime: "kokoro".to_string(),
        model_path: path_string(&model_path),
        codec_path: String::new(),
        voices_path: Some(path_string(&voices_path)),
        espeak_data_path: Some(path_string(&espeak_data_path)),
        model_dir: path_string(&dir),
        version: source.version,
        url: source
            .archive_url
            .unwrap_or_else(|| KOKORO_BUNDLE_URL.to_string()),
    })
}

fn model_sources() -> ModelSources {
    ModelSources {
        runtimes: runtimes()
            .iter()
            .map(|runtime| ModelSource {
                id: runtime.id.into(),
                name: runtime.name.into(),
                version: runtime.version.into(),
                size: runtime.size.into(),
                description: runtime.description.into(),
                files: runtime
                    .files
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
            "macOS: ~/Library/Application Support/com.maxmelichov.mamborambo/models".to_string(),
            "Windows: %LOCALAPPDATA%\\com.maxmelichov.mamborambo\\models".to_string(),
            "Linux: ~/.local/share/com.maxmelichov.mamborambo/models".to_string(),
        ],
    }
}

fn runtime_source(runtime: &str) -> Option<ModelSource> {
    model_sources()
        .runtimes
        .into_iter()
        .find(|source| source.id == runtime)
}

async fn download_kokoro_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    let bundle = kokoro_bundle(&app)?;
    if bundle.installed {
        return Ok(bundle);
    }
    let source = runtime_source("kokoro").ok_or_else(|| "missing kokoro source".to_string())?;
    let archive_url = source
        .archive_url
        .ok_or_else(|| "missing kokoro archive source URL".to_string())?;
    let models_root = models_root(&app)?;
    tokio::fs::create_dir_all(&models_root)
        .await
        .map_err(|err| format!("failed to create {}: {err}", models_root.display()))?;
    let archive_path = models_root.join("mamborambo-kokoro-models-kokoro-v1.0.tar.gz.part");
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let mut downloaded = 0_u64;
    let mut total = remote_content_length(&client, &archive_url).await;
    download_model_file(
        &app,
        &client,
        &archive_url,
        &archive_path,
        &mut downloaded,
        &mut total,
    )
    .await?;

    emit_progress(
        &app,
        ModelDownloadProgress {
            downloaded,
            total,
            progress: Some(1.0),
            stage: "extracting",
        },
    );
    let root = models_root.clone();
    let archive = archive_path.clone();
    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let file = fs::File::open(&archive)
            .map_err(|err| format!("failed to open {}: {err}", archive.display()))?;
        let decoder = GzDecoder::new(file);
        let mut tar = Archive::new(decoder);
        tar.unpack(&root)
            .map_err(|err| format!("failed to extract Kokoro model bundle: {err}"))?;
        let _ = fs::remove_file(&archive);
        Ok(())
    })
    .await
    .map_err(|err| format!("failed to join extraction task: {err}"))??;
    let bundle = kokoro_bundle(&app)?;
    if !bundle.installed {
        return Err("Kokoro bundle extracted, but expected files were not found".to_string());
    }
    Ok(bundle)
}

async fn download_blue_bundle(app: tauri::AppHandle) -> Result<ModelBundle, String> {
    let bundle = blue_bundle(&app)?;
    if bundle.installed {
        return Ok(bundle);
    }
    let source = runtime_source("blue").ok_or_else(|| "missing Blue source".to_string())?;
    let dir = PathBuf::from(&bundle.model_dir);
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|err| format!("failed to create {}: {err}", dir.display()))?;
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .map_err(|err| format!("failed to build HTTP client: {err}"))?;
    let totals = futures_util::future::join_all(
        source
            .files
            .iter()
            .map(|file| remote_content_length(&client, &file.url)),
    )
    .await;
    let known_sum: u64 = totals.iter().flatten().copied().sum();
    let known_count = totals.iter().flatten().count();
    let estimated = parse_size_label_bytes(&source.size).unwrap_or(BLUE_ESTIMATED_BYTES);
    // Prefer exact sum when every file reports a size; otherwise keep the bar moving
    // with the larger of known bytes vs the advertised bundle size (Blue + RenikudPlus).
    let mut total = Some(if known_count == source.files.len() && known_sum > 0 {
        known_sum
    } else {
        known_sum.max(estimated)
    });
    let mut downloaded = 0_u64;
    for file in source.files {
        let destination = dir.join(&file.name);
        if let Some(parent) = destination.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|err| format!("failed to create {}: {err}", parent.display()))?;
        }
        download_model_file(
            &app,
            &client,
            &file.url,
            &destination,
            &mut downloaded,
            &mut total,
        )
        .await?;
    }
    emit_progress(
        &app,
        ModelDownloadProgress {
            downloaded,
            total,
            progress: Some(1.0),
            stage: "downloading",
        },
    );
    let bundle = blue_bundle(&app)?;
    if !bundle.installed {
        return Err("Blue model download completed, but required files are missing".to_string());
    }
    Ok(bundle)
}

fn models_root(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_local_data_dir()
        .map_err(|err| format!("failed to resolve app data dir: {err}"))?
        .join("models");
    Ok(dir)
}

fn model_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(models_root(app)?.join(MODEL_DIR))
}

const BLUE_ESTIMATED_BYTES: u64 = 560 * 1024 * 1024;

fn parse_size_label_bytes(label: &str) -> Option<u64> {
    let lower = label.to_ascii_lowercase();
    let number: String = lower
        .chars()
        .filter(|ch| ch.is_ascii_digit() || *ch == '.')
        .collect();
    let value = number.parse::<f64>().ok()?;
    if lower.contains("gb") {
        Some((value * 1024.0 * 1024.0 * 1024.0) as u64)
    } else if lower.contains("mb") {
        Some((value * 1024.0 * 1024.0) as u64)
    } else if lower.contains("kb") {
        Some((value * 1024.0) as u64)
    } else {
        None
    }
}

async fn remote_content_length(client: &reqwest::Client, url: &str) -> Option<u64> {
    if let Some(length) = client
        .head(url)
        .send()
        .await
        .ok()
        .filter(|response| response.status().is_success())
        .and_then(|response| response_content_length(&response))
    {
        return Some(length);
    }

    // Hugging Face often omits Content-Length on HEAD; Range probes usually reveal size.
    let response = client
        .get(url)
        .header(reqwest::header::RANGE, "bytes=0-0")
        .send()
        .await
        .ok()?;
    if !(response.status().is_success() || response.status().as_u16() == 206) {
        return None;
    }
    // Prefer Content-Range total; Content-Length on a Range response is only the fragment size.
    content_range_total(&response).or_else(|| response_content_length(&response))
}

fn content_range_total(response: &reqwest::Response) -> Option<u64> {
    response
        .headers()
        .get(reqwest::header::CONTENT_RANGE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.rsplit('/').next())
        .filter(|value| *value != "*")
        .and_then(|value| value.parse::<u64>().ok())
}

fn response_content_length(response: &reqwest::Response) -> Option<u64> {
    response.content_length().or_else(|| {
        response
            .headers()
            .get("x-linked-size")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
    })
}

async fn download_model_file(
    app: &tauri::AppHandle,
    client: &reqwest::Client,
    url: &str,
    dest: &Path,
    downloaded: &mut u64,
    total: &mut Option<u64>,
) -> Result<(), String> {
    let part = dest.with_file_name(format!(
        "{}.part",
        dest.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("model.gguf")
    ));
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|err| format!("failed to download model file {url}: {err}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "model file download failed for {url}: {}",
            response.status(),
        ));
    }

    let prior_downloaded = *downloaded;
    let fallback_file_total = response_content_length(&response);
    if let Some(file_total) = fallback_file_total {
        // Expand overall total when we learn this file's true size (e.g. RenikudPlus).
        let minimum = prior_downloaded.saturating_add(file_total);
        *total = Some(total.unwrap_or(0).max(minimum));
    }

    emit_progress(
        app,
        ModelDownloadProgress {
            downloaded: *downloaded,
            total: *total,
            progress: total
                .filter(|total| *total > 0)
                .map(|total| (*downloaded as f64 / total as f64).clamp(0.0, 0.99)),
            stage: "downloading",
        },
    );

    let mut file_downloaded = 0_u64;
    let mut file = tokio::fs::File::create(&part)
        .await
        .map_err(|err| format!("failed to create {}: {err}", part.display()))?;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|err| format!("failed to read model download from {url}: {err}"))?;
        *downloaded += chunk.len() as u64;
        file_downloaded += chunk.len() as u64;
        file.write_all(&chunk)
            .await
            .map_err(|err| format!("failed to write {}: {err}", part.display()))?;
        if let Some(t) = total.as_mut() {
            if *downloaded > *t {
                *t = *downloaded;
            }
        }
        let progress_total = (*total).or(fallback_file_total);
        let progress_downloaded = if total.is_some() {
            *downloaded
        } else {
            file_downloaded
        };
        let ratio = progress_total.filter(|total| *total > 0).map(|total| {
            let value = progress_downloaded as f64 / total as f64;
            if progress_downloaded >= total {
                1.0
            } else {
                value.clamp(0.0, 0.99)
            }
        });
        emit_progress(
            app,
            ModelDownloadProgress {
                downloaded: progress_downloaded,
                total: progress_total,
                progress: ratio,
                stage: "downloading",
            },
        );
    }
    file.flush()
        .await
        .map_err(|err| format!("failed to flush {}: {err}", part.display()))?;
    tokio::fs::rename(&part, dest).await.map_err(|err| {
        format!(
            "failed to move {} to {}: {err}",
            part.display(),
            dest.display()
        )
    })
}

fn model_file_url(file_name: &str) -> String {
    format!("{MODEL_BASE_URL}/{file_name}")
}

fn emit_progress(app: &tauri::AppHandle, payload: ModelDownloadProgress) {
    let _ = app.emit("model_download_progress", payload);
}

fn path_string(path: &Path) -> String {
    path.as_os_str().to_string_lossy().into_owned()
}
