//! Python bindings for the BlueTTS engine.
//!
//! The same Rust inference path the desktop sidecar uses, exposed to
//! Python so a server can be written there without a second copy of the
//! pipeline. Audio comes back as `f32` samples in `[-1, 1]`; the caller
//! decides whether to write a WAV or stream it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

use blue_rs::{
    BlueTts, ChunkingOptions, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};
use mambotts_registry::{BLUE_VOICES, DEFAULT_BLUE_VOICE, blue_voice_name};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;

/// A loaded BlueTTS bundle.
///
/// The engine holds mutable ONNX session state and its phonemizer is
/// `Send` but not `Sync`, so it lives behind a mutex. That also makes the
/// class safe to hand between Python threads, which a threaded server
/// will do.
#[pyclass]
struct BlueTtsEngine {
    inner: Mutex<Engine>,
}

struct Engine {
    tts: BlueTts,
    phonemizer: Phonemizer,
    styles: HashMap<String, VoiceStyle>,
}

#[pymethods]
impl BlueTtsEngine {
    /// Loads a model directory, plus the RenikudPlus ONNX that turns
    /// Hebrew into the stressed IPA the model reads.
    #[new]
    #[pyo3(signature = (model_dir, renikud_path = None))]
    fn new(model_dir: PathBuf, renikud_path: Option<PathBuf>) -> PyResult<Self> {
        let tts = BlueTts::from_dir(&model_dir).map_err(to_py_err)?;
        let renikud = renikud_path.unwrap_or_else(|| model_dir.join("renikud-plus.onnx"));
        let phonemizer = if renikud.is_file() {
            Phonemizer::with_language(Some(&renikud), Language::English)
        } else {
            Phonemizer::with_language(None::<PathBuf>, Language::English)
        }
        .map_err(to_py_err)?;

        let voices_dir = model_dir.join("voices");
        let mut styles = HashMap::new();
        for voice in BLUE_VOICES {
            let path = voices_dir.join(voice.file);
            if !path.is_file() {
                continue;
            }
            styles.insert(
                voice.name.to_owned(),
                VoiceStyle::from_json(&path).map_err(to_py_err)?,
            );
        }
        if styles.is_empty() {
            return Err(PyValueError::new_err(format!(
                "no voice styles found under {}",
                voices_dir.display()
            )));
        }
        Ok(Self {
            inner: Mutex::new(Engine {
                tts,
                phonemizer,
                styles,
            }),
        })
    }

    /// Output sample rate in Hz.
    #[getter]
    fn sample_rate(&self) -> PyResult<u32> {
        Ok(self.lock()?.tts.sample_rate())
    }

    /// Names of the voices that loaded, in catalog order.
    fn voices(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.voices())
    }

    fn languages(&self) -> Vec<String> {
        ["he", "en", "es", "de", "it"]
            .into_iter()
            .map(str::to_owned)
            .collect()
    }

    /// Grapheme-to-IPA for a language, without synthesizing.
    #[pyo3(signature = (text, language = "auto"))]
    fn phonemize(&self, py: Python<'_>, text: &str, language: &str) -> PyResult<String> {
        let language = resolve_language(text, language)?;
        // G2P is CPU work with no Python involvement, so the GIL goes
        // back to the interpreter for its duration.
        py.detach(|| self.lock()?.phonemizer.g2p(text, language).map_err(to_py_err))
    }

    /// Synthesizes text, returning mono `f32` samples.
    #[pyo3(signature = (text, voice = None, language = "auto", speed = 0.95, cfg_scale = 4.0, total_step = 8))]
    fn synthesize(
        &self,
        py: Python<'_>,
        text: &str,
        voice: Option<&str>,
        language: &str,
        speed: f32,
        cfg_scale: f32,
        total_step: usize,
    ) -> PyResult<Vec<f32>> {
        let language = resolve_language(text, language)?;
        let options = options(language.code(), speed, cfg_scale, total_step);
        py.detach(|| {
            let mut engine = self.lock()?;
            let style = engine.style_for(voice)?.clone();
            let Engine { tts, phonemizer, .. } = &mut *engine;
            tts.synthesize_text(phonemizer, text, &style, options)
                .map_err(to_py_err)
        })
    }

    /// Synthesizes from IPA that has already been produced elsewhere.
    #[pyo3(signature = (phonemes, voice = None, language = "en", speed = 0.95, cfg_scale = 4.0, total_step = 8))]
    fn synthesize_phonemes(
        &self,
        py: Python<'_>,
        phonemes: &str,
        voice: Option<&str>,
        language: &str,
        speed: f32,
        cfg_scale: f32,
        total_step: usize,
    ) -> PyResult<Vec<f32>> {
        let options = options(language, speed, cfg_scale, total_step);
        py.detach(|| {
            let mut engine = self.lock()?;
            let style = engine.style_for(voice)?.clone();
            engine
                .tts
                .create(phonemes, &style, options)
                .map_err(to_py_err)
        })
    }
}

impl BlueTtsEngine {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Engine>> {
        self.inner
            .lock()
            .map_err(|_| PyValueError::new_err("engine is poisoned by an earlier panic"))
    }
}

impl Engine {
    fn voices(&self) -> Vec<String> {
        BLUE_VOICES
            .iter()
            .filter(|voice| self.styles.contains_key(voice.name))
            .map(|voice| voice.name.to_owned())
            .collect()
    }

    fn style_for(&self, voice: Option<&str>) -> PyResult<&VoiceStyle> {
        let requested = voice
            .map(str::trim)
            .filter(|voice| !voice.is_empty())
            .unwrap_or(DEFAULT_BLUE_VOICE);
        let name = blue_voice_name(requested);
        self.styles.get(name).ok_or_else(|| {
            PyKeyError::new_err(format!(
                "unknown voice `{requested}`; available: {}",
                self.voices().join(", ")
            ))
        })
    }
}

fn options(language: &str, speed: f32, cfg_scale: f32, total_step: usize) -> SynthesisOptions {
    SynthesisOptions {
        lang: language.to_owned(),
        total_step,
        cfg_scale,
        speed,
        chunking: Some(ChunkingOptions {
            enabled: true,
            silence_seconds: 0.15,
            max_chars: Some(200),
        }),
    }
}

/// Mirrors the sidecar: an empty or `auto` language is decided by whether
/// the text contains Hebrew letters.
fn resolve_language(text: &str, language: &str) -> PyResult<Language> {
    let code = language.trim().to_ascii_lowercase();
    let code = if code.is_empty() || code == "auto" {
        if text.chars().any(|c| ('\u{0590}'..='\u{05ff}').contains(&c)) {
            "he"
        } else {
            "en"
        }
    } else {
        code.as_str()
    };
    match code {
        "en" | "en-us" => Ok(Language::English),
        "he" => Ok(Language::Hebrew),
        "es" => Ok(Language::Spanish),
        "de" | "ge" => Ok(Language::German),
        "it" => Ok(Language::Italian),
        other => Err(PyValueError::new_err(format!(
            "unsupported language `{other}`; expected he, en, es, de, or it"
        ))),
    }
}

fn to_py_err(err: anyhow::Error) -> PyErr {
    PyValueError::new_err(format!("{err:#}"))
}

#[pymodule]
fn mambotts(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<BlueTtsEngine>()?;
    module.add("DEFAULT_VOICE", DEFAULT_BLUE_VOICE)?;
    module.add(
        "VOICES",
        BLUE_VOICES
            .iter()
            .map(|voice| voice.name)
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
