//! BlueV3 (v3.0.1) inference: text or phonemes in, 44.1 kHz audio out.
//!
//! The flow around the graphs is the BlueTTS site's (`js/onnx-web-worker.js`
//! `synthesize`, `chunkFront`, `chunkBack`; `js/bluetts-text.js`), step for
//! step: the text prepared as the site prepares it, G2P over the whole text
//! (Russian a sentence at a time), the phonemes cut into chunks of at most
//! 300 ids a sentence each, ids wrapped in language tags, the predicted
//! duration as it is (no pace blend, inline `<xx>` spans included) divided
//! in float32 by the speed times the voice's `speed_scale` for the language
//! (v3.0.1's per-voice calibration), the latent length and mask as the site
//! rounds them, numpy's `default_rng(42)` noise for every chunk, classifier
//! free guidance inside the vector estimator, the vocoder fed the normalised
//! latent, a peak limit of 0.95 per chunk, 0.1 s between chunks and no trim.

/// A regex compiled once per process, on first use.
macro_rules! regex {
    ($pattern:expr) => {{
        static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
        RE.get_or_init(|| regex::Regex::new($pattern).expect("valid regex"))
    }};
}

pub mod bot;
mod chunking;
pub mod handling;
pub mod hebrew_numbers;
mod ipa;
pub mod numpy_rng;
pub mod phonemize;
mod text;
pub mod gpu_flow;
pub mod gpu_verdict;
pub mod ve_gpu;
pub mod ve_worker;

use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array3};
use ort::{session::Session, value::Tensor};
use serde_json::Value;

use crate::{
    chunking::{pack_chunks, silence_samples},
    phonemize::{Language, Phonemizer, strip_lang_tags},
    text::Tokenizer,
};

const DEFAULT_SAMPLE_RATE: usize = 44_100;
const DEFAULT_BASE_CHUNK_SIZE: usize = 512;
const DEFAULT_CHUNK_COMPRESS_FACTOR: usize = 6;
const DEFAULT_LATENT_DIM: usize = 24;
/// The noise seed of every chunk: the site (and blue_onnx) start each
/// chunk's flow from a fresh `np.random.default_rng(42)`.
pub const SEED: u64 = 42;
/// The checkpoint BlueV3 v3.0.1's graphs and voices are labelled with
/// (`tts.json` and every voice JSON carry `"model"`).
pub const RELEASE_MODEL: &str = "bluev3-702000";
/// Earlier releases' checkpoints that can still run (graphs and voices of
/// the same one) while an app upgrades their bundle: BlueV3 v3.0.0.
pub const PREVIOUS_MODELS: &[&str] = &["bluev3-701500"];
/// The speed every language is read at by default: the release's own 1.0
/// (v3.0.1; the voices' `speed_scale` replaced v3.0.0's 0.9/0.95 tuning).
pub const DEFAULT_SPEED: f64 = 1.0;
/// The floor of the duration divisor (blue_onnx `speed_divisor`).
const SPEED_FLOOR: f64 = 1e-6;

#[derive(Clone, Debug)]
pub struct SynthesisOptions {
    pub lang: String,
    pub total_step: usize,
    pub cfg_scale: f32,
    /// The speed asked for. The voice's `speed_scale[lang]` multiplies it, so
    /// 1.0 is the voice's natural pace in every language.
    pub speed: f64,
}

impl Default for SynthesisOptions {
    /// The release's operating point: 8 Euler steps, guidance 4.0, speed 1.0.
    fn default() -> Self {
        Self {
            lang: "he".to_string(),
            total_step: 8,
            cfg_scale: 4.0,
            speed: DEFAULT_SPEED,
        }
    }
}

/// A voice JSON's `speed_scale`: language code to a positive factor.
pub type SpeedScale = BTreeMap<String, f64>;

#[derive(Clone)]
pub struct VoiceStyle {
    ttl: Array3<f32>,
    dp: Array3<f32>,
    /// The voice's per-language speed calibration (v3.0.1), if it has one.
    speed_scale: Option<SpeedScale>,
    /// The checkpoint the voice was encoded for (`"model"`), if stated.
    model: Option<String>,
}

impl VoiceStyle {
    pub fn new(ttl: Array3<f32>, dp: Array3<f32>) -> Self {
        Self {
            ttl,
            dp,
            speed_scale: None,
            model: None,
        }
    }

    /// The voice's `speed_scale` ([`parse_speed_scale`]).
    pub fn speed_scale(&self) -> Option<&SpeedScale> {
        self.speed_scale.as_ref()
    }

    /// The voice's `"model"` label, such as [`RELEASE_MODEL`].
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let raw = fs::read_to_string(path.as_ref())
            .with_context(|| format!("read voice style {}", path.as_ref().display()))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let json: Value = serde_json::from_str(raw)?;
        Ok(Self {
            ttl: read_style_tensor(&json["style_ttl"])?,
            dp: read_style_tensor(&json["style_dp"])?,
            speed_scale: parse_speed_scale(json.get("speed_scale"), "voice JSON")?,
            model: json["model"].as_str().map(str::to_owned),
        })
    }

    pub fn from_json_bytes(raw: &[u8]) -> Result<Self> {
        Self::from_json_str(std::str::from_utf8(raw)?)
    }
}

pub struct BlueTts {
    dp: Session,
    text_encoder: Session,
    vector_estimator: Session,
    /// Which vector estimator graph `vector_estimator` is.
    precision: Precision,
    vocoder: Session,
    tokenizer: Tokenizer,
    geometry: ModelGeometry,
    release: ReleasePoint,
    vocoder_input: String,
    /// `tts.json`'s `"model"`: the checkpoint the graphs are.
    model: Option<String>,
    last_chunks: Vec<ChunkInfo>,
    /// The site's Rust engine for the flow (wgpu: Metal, Vulkan or DX12),
    /// in-process or in a worker process (see `gpu_flow.rs`). `None` on
    /// Windows unless asked for (see [`gpu_flow_wanted`]), without a usable
    /// GPU, with `MAMBOTTS_VE_BACKEND=ort`, when it failed before on this
    /// adapter and driver, or after it failed once; the ONNX Runtime graph
    /// runs the flow then.
    gpu: Option<gpu_flow::GpuFlow>,
    /// The GPU engine while it is still opening. Compiling its kernels can
    /// take half a minute, so the model loads and speaks on
    /// ONNX Runtime meanwhile and switches at the next chunk once it is ready.
    gpu_opening: Option<std::thread::JoinHandle<Option<gpu_flow::GpuFlow>>>,
}

/// The bundle's dynamic-int8 vector estimator (BlueV3-onnx f0db559, BlueTTS
/// `blue_onnx/cpu.py`): MatMul weights int8 per output channel with
/// `reduce_range`, everything else float32. Optional: a bundle without it
/// runs `vector_estimator.onnx`.
pub const INT8_VECTOR_ESTIMATOR: &str = "vector_estimator.int8.onnx";

/// The float32 vector estimator, the graph every bundle has.
pub const FP32_VECTOR_ESTIMATOR: &str = "vector_estimator.onnx";

/// Which vector estimator graph ONNX Runtime runs the flow with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Int8,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Precision::Fp32 => "fp32",
            Precision::Int8 => "int8",
        }
    }
}

/// The vector estimator precision to ask for: `fp32`, `int8`, or `auto`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PrecisionSetting {
    /// int8 where it was measured to help (see [`PrecisionSetting::resolve`]).
    #[default]
    Auto,
    Fp32,
    Int8,
}

impl PrecisionSetting {
    /// `MAMBOTTS_PRECISION=fp32|int8|auto`; unset, empty or unknown is auto.
    pub fn from_env() -> Self {
        let value = std::env::var("MAMBOTTS_PRECISION").ok();
        let setting = Self::parse(value.as_deref().unwrap_or(""));
        if setting.is_none() {
            eprintln!("MAMBOTTS_PRECISION={value:?} is not fp32, int8 or auto: using auto");
        }
        setting.unwrap_or_default()
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Some(Self::Auto),
            "fp32" | "float32" | "float" => Some(Self::Fp32),
            "int8" | "quantized" => Some(Self::Int8),
            _ => None,
        }
    }

    /// `Auto` is int8 off macOS and float32 on it. Measured with ONNX Runtime
    /// on the CPU (BlueTTS PR #67): the int8 vector estimator was about 1.7x
    /// faster on an x86 Xeon and 2.5x on ARM Linux, with the same ASR error
    /// and a speaker similarity to float32 no lower than another noise draw's.
    /// macOS keeps float32: BlueTTS measured no gain on an Apple M5 (Python
    /// ONNX Runtime), and there the flow runs on the GPU engine by default
    /// (24.8x real time on an M5), which always reads float32. (blue-rs's own
    /// ONNX Runtime 1.23.2 on the M5's CPU did run int8 1.8x faster: 17.0x
    /// against 9.7x real time; `MAMBOTTS_PRECISION=int8` turns it on.)
    pub fn resolve(self, macos: bool) -> Precision {
        match self {
            Self::Fp32 => Precision::Fp32,
            Self::Int8 => Precision::Int8,
            Self::Auto if macos => Precision::Fp32,
            Self::Auto => Precision::Int8,
        }
    }
}

/// Open the vector estimator ONNX Runtime runs: the int8 graph when
/// `precision` asks for it and the bundle has it, else float32. A missing or
/// unloadable int8 graph is a warning, not an error.
fn load_vector_estimator(dir: &Path, precision: Precision) -> Result<(Session, Precision)> {
    let check = |session: Session, path: &Path| -> Result<Session> {
        if !session.inputs().iter().any(|input| input.name() == "cfg_scale") {
            bail!("{} is not a BlueV3 vector estimator: it has no `cfg_scale` input", path.display());
        }
        Ok(session)
    };
    if precision == Precision::Int8 {
        let path = dir.join(INT8_VECTOR_ESTIMATOR);
        if path.is_file() {
            match load_session(&path).and_then(|session| check(session, &path)) {
                Ok(session) => return Ok((session, Precision::Int8)),
                Err(err) => eprintln!("BlueV3 vector estimator: the int8 graph did not load, using float32: {err:#}"),
            }
        } else {
            eprintln!("BlueV3 vector estimator: no {INT8_VECTOR_ESTIMATOR} in the bundle, using float32");
        }
    }
    let path = dir.join(FP32_VECTOR_ESTIMATOR);
    Ok((check(load_session(&path)?, &path)?, Precision::Fp32))
}

/// Whether to open the GPU flow engine, from `MAMBOTTS_VE_BACKEND`: `ort`
/// never, `gpu` always, and otherwise everywhere but Windows.
///
/// Windows runs ONNX Runtime on the CPU by default. The GPU engine compiles
/// its kernels through whatever graphics driver the machine has, and on
/// Windows that went wrong in ways the app cannot recover from: an AMD Radeon
/// iGPU computes them wrongly on Vulkan (caught by the self-check), FXC takes
/// many minutes, and a driver that crashes or hangs takes the whole server
/// down with it, so generation stops working instead of falling back. The CPU
/// path is the one every Windows machine could already fall back to.
///
/// The engine now runs in a worker process by default (`gpu_flow.rs`), where a
/// driver crash or hang costs only the GPU; once that is verified on Windows,
/// the GPU becomes the default there by replacing `!cfg!(windows)` with `true`.
pub(crate) fn gpu_flow_wanted(setting: Option<&str>) -> bool {
    match setting.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("ort") => false,
        Some(value) if value.eq_ignore_ascii_case("gpu") => true,
        _ => !cfg!(windows),
    }
}

/// What one chunk of the last run was: its phonemes, ids, durations and
/// latent length (the site's per-chunk `info`).
#[derive(Clone, Debug)]
pub struct ChunkInfo {
    pub phonemes: String,
    pub tokens: usize,
    /// The duration predictor's seconds.
    pub raw_duration: f32,
    /// The voice's `speed_scale` for the chunk's language (1.0 without one).
    pub speed_scale: f64,
    /// Seconds after the speed: `raw / (speed * speed_scale)` in float32.
    pub duration: f32,
    /// Latent frames (`tLat`).
    pub frames: usize,
    /// Frames under the latent mask (`tMask`); the noise is zero past it.
    pub mask_frames: usize,
    /// Samples the vocoder returned for the chunk.
    pub samples: usize,
}

/// The site's front end over one request's text, before any graph runs.
#[derive(Clone, Debug)]
pub struct TextPlan {
    /// `prepareText`'s output.
    pub prepared: String,
    /// The G2P units (the whole text, or for Russian a sentence each), phonemized.
    pub phonemes: Vec<String>,
    /// The chunks, each at most the release's token budget.
    pub chunks: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
struct ModelGeometry {
    sample_rate: usize,
    base_chunk_size: usize,
    chunk_compress_factor: usize,
    latent_dim: usize,
}

impl Default for ModelGeometry {
    fn default() -> Self {
        Self {
            sample_rate: DEFAULT_SAMPLE_RATE,
            base_chunk_size: DEFAULT_BASE_CHUNK_SIZE,
            chunk_compress_factor: DEFAULT_CHUNK_COMPRESS_FACTOR,
            latent_dim: DEFAULT_LATENT_DIM,
        }
    }
}

/// The parts of the bundle's operating point (`tts.json`, `blue_release`)
/// that are the bundle's to decide rather than the caller's.
#[derive(Clone, Copy, Debug)]
struct ReleasePoint {
    /// Seconds of silence between two chunks.
    join_silence: f64,
    /// Ids one chunk may hold, language tags included.
    max_chunk_tokens: usize,
    /// A chunk louder than this is scaled down to it; quieter is left alone.
    peak_limit: f64,
}

impl Default for ReleasePoint {
    fn default() -> Self {
        Self {
            join_silence: 0.1,
            max_chunk_tokens: 300,
            peak_limit: 0.95,
        }
    }
}

impl BlueTts {
    /// The bundle in `dir`, with the GPU flow engine as the environment
    /// says (see [`gpu_flow::GpuOptions::from_env`]) and the vector
    /// estimator's precision from `MAMBOTTS_PRECISION` (default auto: see
    /// [`PrecisionSetting::resolve`]).
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        Self::from_dir_with_options(dir, gpu_flow::GpuOptions::from_env(), PrecisionSetting::from_env())
    }

    /// [`BlueTts::from_dir`] with the GPU flow engine opened as `gpu` says.
    pub fn from_dir_with(dir: impl AsRef<Path>, gpu: gpu_flow::GpuOptions) -> Result<Self> {
        Self::from_dir_with_options(dir, gpu, PrecisionSetting::from_env())
    }

    /// Opens a bundle with the given vector estimator precision, whatever
    /// the environment says (the parity checks need float32).
    pub fn from_dir_with_precision(dir: impl AsRef<Path>, precision: PrecisionSetting) -> Result<Self> {
        Self::from_dir_with_options(dir, gpu_flow::GpuOptions::from_env(), precision)
    }

    /// Opens a bundle with the GPU flow engine as `gpu` says and the given
    /// vector estimator precision for ONNX Runtime.
    pub fn from_dir_with_options(
        dir: impl AsRef<Path>,
        gpu: gpu_flow::GpuOptions,
        precision: PrecisionSetting,
    ) -> Result<Self> {
        let dir = dir.as_ref();
        let tokenizer = Tokenizer::from_json(dir.join("vocab.json"))?;
        if !tokenizer.is_tagged() {
            bail!(
                "{} is not a BlueV3 bundle: its vocab.json has no `lang_tag`. BlueTTS 2.5 and older bundles are no longer supported; download the current model",
                dir.display()
            );
        }
        let (vector_estimator, precision) =
            load_vector_estimator(dir, precision.resolve(cfg!(target_os = "macos")))?;
        eprintln!(
            "BlueV3 vector estimator on ONNX Runtime: {} ({} threads)",
            precision.as_str(),
            ort_threads()
        );
        let vocoder = load_session(dir.join("vocoder.onnx"))?;
        let vocoder_input = vocoder
            .inputs()
            .first()
            .map(|input| input.name().to_owned())
            .unwrap_or_else(|| "latent".to_owned());
        let (geometry, release, model) = load_config(dir.join("tts.json"))?;
        // The GPU engine runs the float32 graph whatever ONNX Runtime runs.
        let gpu_path = dir.join(FP32_VECTOR_ESTIMATOR);
        let gpu_opening = if gpu.wanted {
            std::thread::Builder::new()
                .name("bluev3-gpu-open".into())
                .spawn(move || gpu_flow::open(&gpu_path, &gpu))
                .map_err(|err| eprintln!("BlueV3 flow on ONNX Runtime: could not start the GPU engine: {err}"))
                .ok()
        } else {
            eprintln!("BlueV3 flow on ONNX Runtime (MAMBOTTS_VE_BACKEND=gpu tries the GPU engine)");
            None
        };

        Ok(Self {
            dp: load_session(dir.join("duration_predictor_style.onnx"))?,
            text_encoder: load_session(dir.join("text_encoder.onnx"))?,
            vector_estimator,
            precision,
            vocoder,
            tokenizer,
            geometry,
            release,
            vocoder_input,
            model,
            last_chunks: Vec::new(),
            gpu: None,
            gpu_opening,
        })
    }

    /// Takes over the GPU engine once its background open has finished.
    fn adopt_gpu(&mut self, wait: bool) {
        if self.gpu_opening.as_ref().is_some_and(|handle| wait || handle.is_finished()) {
            let handle = self.gpu_opening.take().expect("checked above");
            // A panic while opening is one more reason to stay on ONNX Runtime.
            self.gpu = handle.join().ok().flatten();
        }
    }

    /// Blocks until the GPU engine has opened or given up, for callers that
    /// need to know where the flow runs (see [`BlueTts::gpu_adapter`]).
    pub fn wait_for_gpu(&mut self) {
        self.adopt_gpu(true);
    }

    pub fn sample_rate(&self) -> u32 {
        self.geometry.sample_rate as u32
    }

    /// Where the flow runs: the GPU adapter's name, or `None` for ONNX
    /// Runtime (no adapter, `MAMBOTTS_VE_BACKEND=ort`, a failed self-check,
    /// a failure since, or still opening: see [`BlueTts::wait_for_gpu`]).
    pub fn gpu_adapter(&self) -> Option<&str> {
        self.gpu.as_ref().map(|gpu| gpu.adapter())
    }

    /// Whether the GPU engine in use runs in this process or a worker.
    pub fn gpu_isolation(&self) -> Option<gpu_flow::Isolation> {
        self.gpu.as_ref().map(|gpu| gpu.isolation())
    }

    /// The vector estimator graph ONNX Runtime runs (the GPU engine, when
    /// [`BlueTts::gpu_adapter`] names one, runs float32).
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// The checkpoint the bundle's graphs are (`tts.json` `"model"`).
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// A voice can be spoken with these graphs only when it was encoded for
    /// the same checkpoint: a voice JSON states its `"model"`, which has to
    /// be the bundle's.
    pub fn check_voice(&self, style: &VoiceStyle) -> Result<()> {
        match (self.model(), style.model()) {
            (Some(graphs), Some(voice)) if graphs == voice => Ok(()),
            (Some(graphs), Some(voice)) => {
                bail!("this voice was made for {voice}, but the bundle's graphs are {graphs}")
            }
            (Some(graphs), None) => {
                bail!("this voice states no model; the bundle's graphs are {graphs}")
            }
            (None, _) => Ok(()),
        }
    }

    /// Characters accepted by the loaded model vocabulary, including IPA symbols.
    pub fn supported_phonemes(&self) -> Vec<char> {
        self.tokenizer.supported_characters()
    }

    /// Phoneme-level synthesis, seeded as the site seeds it ([`SEED`]).
    pub fn create(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
    ) -> Result<Vec<f32>> {
        self.create_seeded(phonemes, style, opts, SEED)
    }

    /// Run phoneme-level synthesis with an explicit latent seed.
    pub fn create_seeded(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
        seed: u64,
    ) -> Result<Vec<f32>> {
        self.create_seeded_streaming(phonemes, style, opts, seed, |_| Ok(()))
    }

    /// Phoneme-level synthesis that hands each chunk to `on_chunk` as soon as
    /// it is ready, before the next one starts. The callback sees exactly the
    /// samples appended to the returned audio, and returning an error from it
    /// stops the run, which is how a caller cancels a long synthesis.
    pub fn create_streaming<F>(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
        on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        self.create_seeded_streaming(phonemes, style, opts, SEED, on_chunk)
    }

    /// Phonemes a caller wrote: any routing tags dropped (the chunk is
    /// tagged with the request's language), then cut into chunks as the text
    /// path cuts its phonemes.
    fn create_seeded_streaming<F>(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
        seed: u64,
        on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let opts = canonical_options(opts)?;
        let phonemes = strip_lang_tags(phonemes);
        let chunks = pack_chunks([phonemes.as_str()], &opts.lang, self.release.max_chunk_tokens);
        self.speak_chunks(&chunks, style, &opts, seed, on_chunk)
    }

    /// The site's front end over `text`: `prepareText`, G2P over the whole
    /// prepared text (Russian a sentence at a time), the phonemes cut into
    /// chunks.
    pub fn plan_text(
        &self,
        phonemizer: &mut Phonemizer,
        text: &str,
        lang: &str,
    ) -> Result<TextPlan> {
        plan_text(phonemizer, text, lang, self.release.max_chunk_tokens)
    }

    /// The ids a chunk of phonemes is fed to the graphs as.
    pub fn token_ids(&self, phonemes: &str, lang: &str) -> Result<Vec<i64>> {
        token_ids(&self.tokenizer, phonemes, lang)
    }

    /// What each chunk of the last run was.
    pub fn last_chunks(&self) -> &[ChunkInfo] {
        &self.last_chunks
    }

    /// Prepare, phonemize, and synthesize raw multilingual text.
    ///
    /// `create` remains available for callers that already have IPA.
    pub fn synthesize_text(
        &mut self,
        phonemizer: &mut Phonemizer,
        text: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
    ) -> Result<Vec<f32>> {
        self.synthesize_text_streaming(phonemizer, text, style, opts, |_| Ok(()))
    }

    /// Prepare, phonemize, and synthesize raw text one playable chunk at a time.
    ///
    /// The callback is invoked as soon as each chunk has been synthesized,
    /// before the next chunk starts, with the join silence before it
    /// included. A chunk is final when it is made: the returned samples are
    /// exactly the chunks the callback saw, joined. An error from the
    /// callback stops the run, which is how a caller cancels.
    pub fn synthesize_text_streaming<F>(
        &mut self,
        phonemizer: &mut Phonemizer,
        text: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
        on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let opts = canonical_options(opts)?;
        let plan = self.plan_text(phonemizer, text, &opts.lang)?;
        self.speak_chunks(&plan.chunks, style, &opts, SEED, on_chunk)
    }

    /// Every chunk through the graphs, in order, each after the join
    /// silence (none before the first), handed to `on_chunk` as it is made.
    fn speak_chunks<F>(
        &mut self,
        chunks: &[String],
        style: &VoiceStyle,
        opts: &SynthesisOptions,
        seed: u64,
        mut on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        self.check_voice(style)?;
        if !(opts.speed.is_finite() && opts.speed > 0.0) {
            bail!("speed must be a positive number, got {}", opts.speed);
        }
        self.last_chunks.clear();
        let mut ledger = ChunkLedger::default();
        ledger.expect(chunks.len());
        let gap = silence_samples(self.sample_rate(), self.release.join_silence);
        let mut audio = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            let mut piece = Vec::new();
            if idx != 0 {
                piece.resize(gap, 0.0);
            }
            let spoken = self.synthesize_chunk(chunk, style, opts, seed)?;
            ledger.spoken_chunk(spoken.requested_samples, spoken.samples.len());
            piece.extend(spoken.samples);
            on_chunk(&piece)?;
            audio.extend(piece);
        }
        ledger.verify(self.sample_rate())?;
        Ok(audio)
    }

    /// `inferChunk`: `chunkFront` (ids, duration, text encoder, noise), the
    /// flow, `chunkBack` (vocoder), then `limitPeak`.
    fn synthesize_chunk(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: &SynthesisOptions,
        seed: u64,
    ) -> Result<ChunkAudio> {
        let (text_ids, text_mask) = self.tokenizer.encode_batch(&[phonemes], &[&opts.lang])?;
        let n_text = text_ids.len();

        // Scoped so the session outputs, which borrow self, are released
        // before the vocoder call needs self again.
        let raw_duration = {
            let dur = self.dp.run(ort::inputs! {
                "text_ids" => Tensor::from_array(text_ids.clone())?,
                "style_dp" => Tensor::from_array(style.dp.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
            })?;
            output_vec_f32(&dur[0])?
                .first()
                .copied()
                .context("duration output was empty")?
        };
        if !raw_duration.is_finite() {
            bail!("the duration predictor returned {raw_duration} s for this text");
        }
        let speed_scale = speed_scale_for(style.speed_scale(), &opts.lang);
        let duration = chunk_duration(raw_duration, opts.speed, style.speed_scale(), &opts.lang);

        let text_emb = {
            let out = self.text_encoder.run(ort::inputs! {
                "text_ids" => Tensor::from_array(text_ids)?,
                "style_ttl" => Tensor::from_array(style.ttl.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
            })?;
            output_array3(&out[0])?
        };

        let requested_samples = requested_sample_count(duration, self.geometry.sample_rate);
        let latent = sample_noisy_latent(duration, self.geometry, seed);
        let (frames, mask_frames) = (latent.frames, latent.mask_frames);
        let mut xt = latent.noise;
        let latent_mask = latent.mask;
        let total_step = Array1::from_vec(vec![opts.total_step as f32]);
        let cfg_scale = Array1::from_vec(vec![opts.cfg_scale]);

        // The site's engine runs the whole flow in one go; the same latent
        // as the graph's to ~1e-5. If it fails (in a worker: crashes, hangs,
        // answers wrongly), ONNX Runtime takes over for this chunk and every
        // later one, from the same noise.
        let mut gpu_done = false;
        self.adopt_gpu(false);
        if let Some(engine) = self.gpu.as_mut() {
            let flat = |array: &ndarray::Array3<f32>| -> Vec<f32> { array.iter().copied().collect() };
            match engine.flow(
                &flat(&text_emb),
                &flat(&text_mask),
                &flat(&style.ttl),
                &flat(&latent_mask),
                &flat(&xt),
                opts.total_step,
                opts.cfg_scale,
            ) {
                Ok(latent) => {
                    xt = ndarray::Array3::from_shape_vec(xt.raw_dim(), latent)
                        .context("GPU flow returned a latent of the wrong size")?;
                    gpu_done = true;
                }
                Err(err) => {
                    eprintln!("GPU flow failed, this chunk and the rest on ONNX Runtime: {err:#}");
                    self.gpu = None;
                }
            }
        }

        // One Euler step a run, with both guidance branches inside the graph.
        for step in (0..opts.total_step).filter(|_| !gpu_done) {
            let current_step = Array1::from_vec(vec![step as f32]);
            let out = self.vector_estimator.run(ort::inputs! {
                "noisy_latent" => Tensor::from_array(xt)?,
                "text_emb" => Tensor::from_array(text_emb.clone())?,
                "style_ttl" => Tensor::from_array(style.ttl.clone())?,
                "latent_mask" => Tensor::from_array(latent_mask.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
                "current_step" => Tensor::from_array(current_step)?,
                "total_step" => Tensor::from_array(total_step.clone())?,
                "cfg_scale" => Tensor::from_array(cfg_scale.clone())?,
            })?;
            xt = output_array3(&out[0])?;
        }

        // The vocoder takes the normalised flow output as it is.
        let vocoder_input = self.vocoder_input.clone();
        let wav = self.vocoder.run(ort::inputs! {
            vocoder_input.as_str() => Tensor::from_array(xt)?,
        })?;
        let wav = output_array3(&wav[0])?;
        // The full decoded length of the chunk: the release trims no edge.
        let mut samples: Vec<f32> = wav.iter().copied().collect();
        limit_peak(&mut samples, self.release.peak_limit);
        self.last_chunks.push(ChunkInfo {
            phonemes: phonemes.to_owned(),
            tokens: n_text,
            raw_duration,
            speed_scale,
            duration,
            frames,
            mask_frames,
            samples: samples.len(),
        });
        Ok(ChunkAudio {
            samples,
            requested_samples,
        })
    }
}

/// A voice JSON's optional `speed_scale` (BlueTTS main, PR #61,
/// `blue_onnx.parse_speed_scale`): `null` or missing is no calibration;
/// otherwise an object whose every value reads as a finite positive number
/// (as Python's `float()` reads it: a number, a numeric string, a boolean).
/// Anything else refuses the voice.
pub fn parse_speed_scale(value: Option<&Value>, source: &str) -> Result<Option<SpeedScale>> {
    let map = match value {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Object(map)) => map,
        Some(other) => bail!(
            "{source}: speed_scale must be an object of language -> number, got {}",
            json_type_name(other)
        ),
    };
    let mut out = SpeedScale::new();
    for (lang, factor) in map {
        let number = match factor {
            Value::Number(n) => n.as_f64(),
            Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            Value::String(s) => s.trim().replace('_', "").parse::<f64>().ok(),
            _ => None,
        };
        let Some(number) = number else {
            bail!("{source}: speed_scale[{lang:?}] is not a number: {factor}");
        };
        if !(number.is_finite() && number > 0.0) {
            bail!("{source}: speed_scale[{lang:?}] must be a positive finite number, got {factor}");
        }
        out.insert(lang.clone(), number);
    }
    Ok(Some(out))
}

fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

/// The voice's factor for `lang` (`speed_scale_for`): the entry for the code
/// as given, else for its canonical code (`en-us` is `en`, `ge` is `de`),
/// else 1.0, as it is with no `speed_scale` at all.
pub fn speed_scale_for(scale: Option<&SpeedScale>, lang: &str) -> f64 {
    let Some(scale) = scale.filter(|scale| !scale.is_empty()) else {
        return 1.0;
    };
    scale
        .get(lang)
        .or_else(|| scale.get(&handling::canonical_lang(lang)))
        .copied()
        .unwrap_or(1.0)
}

/// What the duration predictor's seconds are divided by (`speed_divisor`):
/// `max(speed * speed_scale[lang], 1e-6)`, the product in float64.
pub fn speed_divisor(speed: f64, scale: Option<&SpeedScale>, lang: &str) -> f64 {
    (speed * speed_scale_for(scale, lang)).max(SPEED_FLOOR)
}

/// Seconds for one chunk, as the site's `chunkFront` computes them:
/// `float32(raw) / float32(speed_divisor(...))`, a float32 division (JS
/// `Math.fround(raw / Math.fround(divisor))`, the same value). Release
/// bundles blend no pace, inline `<xx>…</xx>` spans included.
pub fn chunk_duration(raw: f32, speed: f64, scale: Option<&SpeedScale>, lang: &str) -> f32 {
    raw / speed_divisor(speed, scale, lang) as f32
}

/// The duration predictor alone (`duration_predictor_style.onnx`): the
/// seconds a chunk's ids take for a voice, before the speed.
pub struct DurationModel {
    session: Session,
}

impl DurationModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            session: load_session(path)?,
        })
    }

    /// The predictor's raw seconds for one chunk's ids.
    pub fn raw_seconds(&mut self, ids: &[i64], style: &VoiceStyle) -> Result<f32> {
        let n = ids.len();
        let text_ids = ndarray::Array2::from_shape_vec((1, n), ids.to_vec())?;
        let text_mask = Array3::<f32>::ones((1, 1, n));
        let out = self.session.run(ort::inputs! {
            "text_ids" => Tensor::from_array(text_ids)?,
            "style_dp" => Tensor::from_array(style.dp.clone())?,
            "text_mask" => Tensor::from_array(text_mask)?,
        })?;
        output_vec_f32(&out[0])?
            .first()
            .copied()
            .context("duration output was empty")
    }
}

/// `opts` with its language code made canonical (`ge` is `de`, `EN` is
/// `en`), so the ids are tagged with a code the model knows. An unknown code
/// fails here rather than tagging the ids with it.
fn canonical_options(mut opts: SynthesisOptions) -> Result<SynthesisOptions> {
    opts.lang = Language::try_from(opts.lang.as_str())?.code().to_owned();
    Ok(opts)
}

/// One chunk's audio and the number of samples the model was asked to fill.
struct ChunkAudio {
    samples: Vec<f32>,
    requested_samples: usize,
}

/// ONNX Runtime's intra-op thread count: `MAMBOTTS_ORT_THREADS`, else the
/// CPUs this process may use ([`std::thread::available_parallelism`]: the
/// affinity mask and, on Linux, the cgroup quota), at most 8.
///
/// Measured on an Apple M5 with ONNX Runtime 1.23.2 (57.7 s of English audio,
/// flow on ONNX Runtime): 1 thread 3.2x real time, 2 5.7x, 4 8.8x, 6 8.6x,
/// 8 10.2x, 10 9.0x. More threads than CPUs is far worse than fewer: on an
/// x86 Xeon limited to 2 CPUs, 8 threads ran 3x slower than 2 (BlueTTS PR
/// #67). The variable is there to try other counts per machine
/// (`examples/bench.rs`).
fn ort_threads() -> usize {
    let setting = std::env::var("MAMBOTTS_ORT_THREADS").ok();
    let cpus = std::thread::available_parallelism().map(usize::from).ok();
    threads_for(setting.as_deref(), cpus)
}

const MAX_DEFAULT_THREADS: usize = 8;

fn threads_for(setting: Option<&str>, cpus: Option<usize>) -> usize {
    setting
        .and_then(|v| v.trim().parse().ok())
        .filter(|n| *n > 0)
        .unwrap_or_else(|| cpus.unwrap_or(MAX_DEFAULT_THREADS).clamp(1, MAX_DEFAULT_THREADS))
}

fn load_session(path: impl AsRef<Path>) -> Result<Session> {
    let path = path.as_ref();
    Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(ort_threads())
        .map_err(|e| anyhow!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow!("{e}"))?
        .commit_from_file(path)
        .map_err(|e| anyhow!("{e}"))
        .with_context(|| format!("load ONNX session {}", path.display()))
}

fn output_array3(value: &ort::value::DynValue) -> Result<Array3<f32>> {
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    match dims.as_slice() {
        [a, b, c] => Ok(Array3::from_shape_vec((*a, *b, *c), data.to_vec())?),
        [a, b] => Ok(Array3::from_shape_vec((*a, 1, *b), data.to_vec())?),
        _ => bail!("expected rank-2/rank-3 f32 tensor, got shape {shape}"),
    }
}

/// The site's front end over `text`, without the graphs: the text prepared
/// ([`Phonemizer::prepare`], bot rules included unless turned off), G2P over
/// the whole prepared text (Russian a sentence at a time), the phonemes cut
/// into chunks of at most `max_ids` ids.
pub fn plan_text(
    phonemizer: &mut Phonemizer,
    text: &str,
    lang: &str,
    max_ids: usize,
) -> Result<TextPlan> {
    let language = Language::try_from(lang)?;
    let prepared = phonemizer.prepare(text, language);
    let phonemes = phonemizer.phonemize_units(&prepared, language)?;
    if phonemes.iter().all(String::is_empty) && regex!(r"\p{L}").is_match(&prepared) {
        // Letters that came back as nothing are a G2P failure, not
        // silence: say so instead of returning an empty recording.
        bail!(
            "no phonemes were produced for this text, so it cannot be spoken: {:?}",
            prepared
        );
    }
    let chunks = pack_chunks(phonemes.iter().map(String::as_str), language.code(), max_ids);
    Ok(TextPlan {
        prepared,
        phonemes,
        chunks,
    })
}

/// `ChunkPacker` over G2P units (each the phonemes of a whole text or, for
/// Russian, of one sentence): chunks of at most `max_ids` ids.
pub fn chunk_phonemes<'a>(
    units: impl IntoIterator<Item = &'a str>,
    lang: &str,
    max_ids: usize,
) -> Vec<String> {
    pack_chunks(units, lang, max_ids)
}

fn token_ids(tokenizer: &Tokenizer, phonemes: &str, lang: &str) -> Result<Vec<i64>> {
    let (ids, _) = tokenizer.encode_batch(&[phonemes], &[lang])?;
    Ok(ids.iter().copied().collect())
}

/// The ids of a bundle's `vocab.json` without its graphs, for checking the
/// front end alone.
pub struct TokenEncoder(Tokenizer);

impl TokenEncoder {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self(Tokenizer::from_json(path)?))
    }

    /// The ids a chunk of phonemes is fed to the graphs as
    /// ([`BlueTts::token_ids`]).
    pub fn token_ids(&self, phonemes: &str, lang: &str) -> Result<Vec<i64>> {
        token_ids(&self.0, phonemes, lang)
    }
}

/// `(tLat, tMask)` for a chunk of `duration` seconds (after the speed), as
/// the site rounds them: the width from `trunc(seconds * sr)` in float64,
/// at least one frame; the mask from the float32 product, within the width.
pub fn latent_geometry(duration: f32, sample_rate: usize, frame_len: usize) -> (usize, usize) {
    let rate = sample_rate as f64;
    let frame = frame_len as f64;
    let frames = ((f64::from(duration) * rate).trunc() + frame - 1.0) / frame;
    let frames = frames.floor().max(1.0) as usize;
    let mask_frames = (f64::from((duration * sample_rate as f32).trunc()) + frame - 1.0) / frame;
    let mask_frames = (mask_frames.floor().max(0.0) as usize).min(frames);
    (frames, mask_frames)
}

/// Samples of speech the model is being asked for at this duration.
fn requested_sample_count(duration: f32, sample_rate: usize) -> usize {
    (duration * sample_rate as f32).max(1.0).ceil() as usize
}

/// The flow's starting point for one chunk.
struct NoisyLatent {
    /// `[1, latent_dim * compress, frames]`, zero past `mask_frames`.
    noise: Array3<f32>,
    /// `[1, 1, frames]`: one over the first `mask_frames` frames.
    mask: Array3<f32>,
    frames: usize,
    mask_frames: usize,
}

/// `sample_noisy_latent` as the site computes it: the width from
/// `int(seconds * sr)` in float64 (`latent_frames_for_duration`), the mask
/// from `duration * sr` in float32, as the two numpy expressions compute
/// them; the noise from `default_rng(seed)`, zeroed past the mask.
fn sample_noisy_latent(duration: f32, geometry: ModelGeometry, seed: u64) -> NoisyLatent {
    let (frames, mask_frames) = latent_geometry(
        duration,
        geometry.sample_rate,
        geometry.base_chunk_size * geometry.chunk_compress_factor,
    );
    let channels = geometry.latent_dim * geometry.chunk_compress_factor;

    let mut noise = numpy_rng::standard_normal_f32(seed, channels * frames);
    if mask_frames < frames {
        for row in noise.chunks_mut(frames) {
            row[mask_frames..].fill(0.0);
        }
    }
    let mask = (0..frames)
        .map(|t| if t < mask_frames { 1.0 } else { 0.0 })
        .collect();
    NoisyLatent {
        noise: Array3::from_shape_vec((1, channels, frames), noise).expect("noise shape"),
        mask: Array3::from_shape_vec((1, 1, frames), mask).expect("mask shape"),
        frames,
        mask_frames,
    }
}

/// blue_onnx's `limit_peak(audio, 0.95)`, per chunk, as the site computes it:
/// one gain (float64 `limit / peak`, then float32) when the peak is above the
/// limit, attenuation only; quieter audio is untouched.
fn limit_peak(audio: &mut [f32], limit: f64) {
    let mut peak = 0.0f32;
    for sample in audio.iter() {
        let a = sample.abs();
        if a > peak {
            peak = a;
        }
    }
    let peak = f64::from(peak);
    if peak <= limit || peak < 1e-9 {
        return;
    }
    let gain = (limit / peak) as f32;
    for sample in audio {
        *sample *= gain;
    }
}

/// Chunks the text asked for against the audio the model gave back.
///
/// Issue #9 was a document that came back as seven percent of itself with a
/// success at the end. Nothing in the pipeline compared what the splitter
/// produced with what was spoken, so any chunk that went missing went missing
/// quietly. This counts both sides and refuses to return a recording that does
/// not cover the text.
#[derive(Default, Debug)]
struct ChunkLedger {
    expected: usize,
    spoken: usize,
    requested_samples: usize,
    produced_samples: usize,
}

impl ChunkLedger {
    fn expect(&mut self, chunks: usize) {
        self.expected += chunks;
    }

    fn spoken_chunk(&mut self, requested_samples: usize, produced_samples: usize) {
        self.spoken += 1;
        self.requested_samples += requested_samples;
        self.produced_samples += produced_samples;
    }

    /// Fail loudly when the recording cannot account for the text.
    fn verify(&self, sample_rate: u32) -> Result<()> {
        if self.spoken != self.expected {
            bail!(
                "synthesis lost text: {} of {} chunks were spoken",
                self.spoken,
                self.expected
            );
        }
        // A chunk decodes to whole latent frames, so the recording is never
        // shorter than the sum of the requested durations, let alone by half.
        if self.produced_samples * 2 < self.requested_samples {
            let seconds = |samples: usize| samples as f32 / sample_rate.max(1) as f32;
            bail!(
                "synthesis produced {:.1} s of audio for text that needs about {:.1} s across {} chunks",
                seconds(self.produced_samples),
                seconds(self.requested_samples),
                self.spoken
            );
        }
        Ok(())
    }
}

/// The latent geometry and the release's operating point from `tts.json`.
fn load_config(path: impl AsRef<Path>) -> Result<(ModelGeometry, ReleasePoint, Option<String>)> {
    let path = path.as_ref();
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read model config {}", path.display()))?;
    let json: Value = serde_json::from_str(&raw)
        .with_context(|| format!("parse model config {}", path.display()))?;
    let mut geometry = ModelGeometry::default();
    geometry.sample_rate = json["ae"]["sample_rate"]
        .as_u64()
        .unwrap_or(geometry.sample_rate as u64) as usize;
    geometry.base_chunk_size = json["ae"]["base_chunk_size"]
        .as_u64()
        .unwrap_or(geometry.base_chunk_size as u64) as usize;
    geometry.chunk_compress_factor = json["ttl"]["chunk_compress_factor"]
        .as_u64()
        .unwrap_or(geometry.chunk_compress_factor as u64)
        as usize;
    geometry.latent_dim = json["ttl"]["latent_dim"]
        .as_u64()
        .unwrap_or(geometry.latent_dim as u64) as usize;

    let inference = &json["blue_release"]["inference"];
    let mut release = ReleasePoint::default();
    if let Some(value) = inference["join_silence_s"].as_f64() {
        release.join_silence = value;
    }
    if let Some(value) = inference["max_chunk_tokens"].as_u64() {
        release.max_chunk_tokens = value as usize;
    }
    if let Some(value) = inference["peak_limit"].as_f64() {
        release.peak_limit = value;
    }
    let model = json["blue_release"]["model"].as_str().map(str::to_owned);
    Ok((geometry, release, model))
}

fn output_vec_f32(value: &ort::value::DynValue) -> Result<Vec<f32>> {
    let (_, data) = value.try_extract_tensor::<f32>()?;
    Ok(data.to_vec())
}

fn read_style_tensor(value: &Value) -> Result<Array3<f32>> {
    let dims = value["dims"].as_array().context("style dims missing")?;
    let shape = [
        dims[0].as_u64().context("bad style dim 0")? as usize,
        dims[1].as_u64().context("bad style dim 1")? as usize,
        dims[2].as_u64().context("bad style dim 2")? as usize,
    ];
    let data = flatten_f32(&value["data"]);
    Ok(Array3::from_shape_vec(shape, data)?)
}

fn flatten_f32(value: &Value) -> Vec<f32> {
    match value {
        Value::Array(items) => items.iter().flat_map(flatten_f32).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap_or(0.0) as f32],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_codes_are_canonical_before_tagging() {
        let with = |lang: &str| SynthesisOptions {
            lang: lang.to_owned(),
            ..SynthesisOptions::default()
        };
        for (lang, expected) in [("ge", "de"), ("EN", "en"), ("en-US", "en"), ("he", "he")] {
            assert_eq!(
                canonical_options(with(lang)).unwrap().lang,
                expected,
                "{lang}"
            );
        }
        assert!(canonical_options(with("xx")).is_err());
    }

    /// The bundle's engine and front end:
    /// `MAMBOTTS_BLUE_MODEL_DIR=<bundle> cargo test -- --ignored`.
    fn bundle() -> (BlueTts, Phonemizer, VoiceStyle) {
        let dir = std::path::PathBuf::from(
            std::env::var("MAMBOTTS_BLUE_MODEL_DIR").expect("set MAMBOTTS_BLUE_MODEL_DIR"),
        );
        let voice = fs::read_dir(dir.join("voices"))
            .unwrap()
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .find(|path| path.extension().is_some_and(|ext| ext == "json"))
            .expect("a voice in the bundle");
        let mut tts = BlueTts::from_dir(&dir).unwrap();
        // Every run on one engine: otherwise the GPU engine, opening in the
        // background, takes over between two runs and they differ by its ~1e-5.
        tts.wait_for_gpu();
        (
            tts,
            Phonemizer::open(None::<&Path>, Some(dir.join("polyg2p")), Language::English).unwrap(),
            VoiceStyle::from_json(voice).unwrap(),
        )
    }

    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn two_runs_are_bit_identical() {
        let (mut tts, mut phonemizer, style) = bundle();
        let english = || SynthesisOptions {
            lang: "en".to_owned(),
            speed: 0.95,
            ..SynthesisOptions::default()
        };
        let text = "Hi. This is the first sentence of the test. And here is a second one!";
        let first = tts
            .synthesize_text(&mut phonemizer, text, &style, english())
            .unwrap();
        let info = tts.last_chunks().to_vec();
        let second = tts
            .synthesize_text(&mut phonemizer, text, &style, english())
            .unwrap();
        assert_eq!(first, second);
        // "Hi." is a fragment and joins the sentence after it.
        assert_eq!(info.len(), 2, "{info:?}");
        let gap = silence_samples(tts.sample_rate(), 0.1);
        assert_eq!(gap, 4410);
        assert_eq!(first.len(), info[0].samples + gap + info[1].samples);
        for chunk in &info {
            assert_eq!(chunk.samples, chunk.frames * 3072);
            assert_eq!(chunk.mask_frames, chunk.frames);
        }
        // Phoneme input is seeded the same way.
        let phonemes = phonemizer.g2p(text, Language::English).unwrap();
        let a = tts.create(&phonemes, &style, english()).unwrap();
        let b = tts.create(&phonemes, &style, english()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, first, "the phoneme path speaks the text path's chunks");
    }

    #[test]
    fn threads_default_to_the_usable_cpus_at_most_eight() {
        assert_eq!(threads_for(None, Some(2)), 2);
        assert_eq!(threads_for(None, Some(1)), 1);
        assert_eq!(threads_for(None, Some(16)), 8);
        assert_eq!(threads_for(None, None), 8);
        assert_eq!(threads_for(Some(""), Some(4)), 4);
        assert_eq!(threads_for(Some("0"), Some(4)), 4);
        assert_eq!(threads_for(Some("x"), Some(4)), 4);
        assert_eq!(threads_for(Some(" 12 "), Some(4)), 12, "the override is not capped");
        assert!(ort_threads() >= 1);
    }

    #[test]
    fn int8_is_the_default_off_macos_only() {
        assert_eq!(PrecisionSetting::Auto.resolve(true), Precision::Fp32);
        assert_eq!(PrecisionSetting::Auto.resolve(false), Precision::Int8);
        assert_eq!(PrecisionSetting::Fp32.resolve(false), Precision::Fp32);
        assert_eq!(PrecisionSetting::Int8.resolve(true), Precision::Int8);
        assert_eq!(PrecisionSetting::parse(""), Some(PrecisionSetting::Auto));
        assert_eq!(PrecisionSetting::parse(" AUTO "), Some(PrecisionSetting::Auto));
        assert_eq!(PrecisionSetting::parse("FP32"), Some(PrecisionSetting::Fp32));
        assert_eq!(PrecisionSetting::parse("int8"), Some(PrecisionSetting::Int8));
        assert_eq!(PrecisionSetting::parse("fp16"), None);
    }

    /// A bundle without the int8 graph runs float32 even when int8 is asked
    /// for (an install from before it was published).
    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn a_bundle_without_int8_runs_float32() {
        let src = std::path::PathBuf::from(std::env::var("MAMBOTTS_BLUE_MODEL_DIR").unwrap());
        let dir = tempfile_dir();
        for name in [FP32_VECTOR_ESTIMATOR, "vocoder.onnx", "text_encoder.onnx", "duration_predictor_style.onnx", "vocab.json", "tts.json"] {
            link_or_copy(&src.join(name), &dir.join(name));
        }
        let tts = BlueTts::from_dir_with_precision(&dir, PrecisionSetting::Int8).unwrap();
        assert_eq!(tts.precision(), Precision::Fp32);
        if src.join(INT8_VECTOR_ESTIMATOR).is_file() {
            link_or_copy(&src.join(INT8_VECTOR_ESTIMATOR), &dir.join(INT8_VECTOR_ESTIMATOR));
            let tts = BlueTts::from_dir_with_precision(&dir, PrecisionSetting::Int8).unwrap();
            assert_eq!(tts.precision(), Precision::Int8);
            let tts = BlueTts::from_dir_with_precision(&dir, PrecisionSetting::Fp32).unwrap();
            assert_eq!(tts.precision(), Precision::Fp32);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    fn tempfile_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("blue-rs-precision-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn link_or_copy(from: &Path, to: &Path) {
        if fs::hard_link(from, to).is_err() {
            fs::copy(from, to).unwrap();
        }
    }

    #[test]
    fn windows_runs_onnx_runtime_unless_the_gpu_is_asked_for() {
        assert_eq!(gpu_flow_wanted(None), !cfg!(windows));
        assert_eq!(gpu_flow_wanted(Some("")), !cfg!(windows));
        assert!(!gpu_flow_wanted(Some("ort")));
        assert!(!gpu_flow_wanted(Some(" ORT ")));
        assert!(gpu_flow_wanted(Some("gpu")));
        assert!(gpu_flow_wanted(Some("GPU")));
    }

    /// The desktop's default off Windows (and with MAMBOTTS_VE_BACKEND=gpu
    /// on it): the flow on the GPU engine, its start-up self-check passing
    /// against this release's graph (a stale golden from another vector
    /// estimator would fall back to ONNX Runtime quietly).
    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle and a GPU"]
    fn the_flow_runs_on_the_gpu_by_default() {
        if !gpu_flow_wanted(std::env::var("MAMBOTTS_VE_BACKEND").ok().as_deref()) {
            eprintln!("the GPU engine is not wanted here (Windows, or MAMBOTTS_VE_BACKEND=ort): skipped");
            return;
        }
        let (mut tts, _, _) = bundle();
        tts.wait_for_gpu();
        let adapter = tts.gpu_adapter().expect("the GPU flow engine did not open (see stderr)");
        eprintln!("flow on {adapter}");
    }

    #[test]
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn punctuation_alone_is_read_as_the_site_reads_it() {
        let (tts, mut phonemizer, _) = bundle();
        // A unit with no letter is not spoken (a bot rule, BlueV3 v3.0.1):
        // the model is never asked to say bare punctuation.
        let plan = tts.plan_text(&mut phonemizer, "...", "en").unwrap();
        assert!(plan.chunks.is_empty(), "{:?}", plan.chunks);
        // Without the bot rules the site hands polyg2p nothing for a piece
        // with no letters, and the marks are the chunk.
        phonemizer.set_bot_rules(false);
        let plan = tts.plan_text(&mut phonemizer, "...", "en").unwrap();
        assert_eq!(plan.chunks, vec!["..."]);
        phonemizer.set_bot_rules(true);
        let plan = tts
            .plan_text(&mut phonemizer, "Hello there. ...", "en")
            .unwrap();
        assert_eq!(plan.chunks.len(), 1, "{:?}", plan.chunks);
        assert!(tts.plan_text(&mut phonemizer, "  ", "en").unwrap().chunks.is_empty());
    }

    #[test]
    fn seeded_latents_are_reproducible() {
        let geometry = ModelGeometry::default();
        let first = sample_noisy_latent(1.0, geometry, 42);
        let second = sample_noisy_latent(1.0, geometry, 42);
        assert_eq!(first.noise, second.noise);
        assert_eq!(first.mask, second.mask);
    }

    #[test]
    fn the_latent_covers_the_duration_in_whole_frames() {
        // T = floor((int(duration * 44100) + 3071) / 3072), 144 channels.
        let latent = sample_noisy_latent(3.62, ModelGeometry::default(), 7);
        assert_eq!(latent.noise.dim(), (1, 144, 52));
        assert_eq!(latent.mask.dim(), (1, 1, 52));
        assert_eq!(latent.mask_frames, 52);
        assert!(latent.mask.iter().all(|value| *value == 1.0));
        // The noise is numpy's default_rng(seed) stream in C order.
        let latent = sample_noisy_latent(1.0, ModelGeometry::default(), 42);
        assert_eq!(latent.frames, 15);
        assert_eq!(latent.noise[[0, 0, 0]], 0.304_717_1_f32);
        assert_eq!(latent.noise[[0, 0, 1]], -1.039_984_1_f32);
    }

    #[test]
    fn the_duration_is_the_models_divided_by_speed_times_scale_in_float32() {
        assert_eq!(chunk_duration(2.0, 0.9, None, "he"), 2.0f32 / 0.9f32);
        assert_eq!(chunk_duration(2.0, 1.0, None, "he"), 2.0);
        assert_eq!(chunk_duration(2.0, 0.0, None, "he"), 2.0f32 / 1e-6f32);
        let scale: SpeedScale = [("he".to_owned(), 1.063), ("de".to_owned(), 0.995)].into();
        assert_eq!(speed_scale_for(Some(&scale), "he"), 1.063);
        assert_eq!(speed_scale_for(Some(&scale), "ge"), 0.995);
        assert_eq!(speed_scale_for(Some(&scale), "fr"), 1.0);
        assert_eq!(speed_scale_for(Some(&SpeedScale::new()), "he"), 1.0);
        assert_eq!(
            chunk_duration(2.5, 0.9, Some(&scale), "he"),
            2.5f32 / (0.9f64 * 1.063) as f32
        );
    }

    #[test]
    fn speed_scale_is_an_object_of_positive_numbers() {
        let parse = |raw: &str| parse_speed_scale(Some(&serde_json::from_str(raw).unwrap()), "v");
        assert_eq!(parse("null").unwrap(), None);
        assert_eq!(parse("{}").unwrap(), Some(SpeedScale::new()));
        assert_eq!(parse(r#"{"he": 1.05, "en": "0.9"}"#).unwrap().unwrap()["en"], 0.9);
        for bad in [r#"[1.0]"#, r#"1.0"#, r#"{"he": 0}"#, r#"{"he": -1}"#, r#"{"he": "fast"}"#, r#"{"he": null}"#, r#"{"he": [1]}"#] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        assert_eq!(parse_speed_scale(None, "v").unwrap(), None);
    }

    #[test]
    fn a_loud_chunk_is_scaled_to_the_limit_with_one_gain() {
        let mut audio = vec![0.5, -1.9, 0.95];
        limit_peak(&mut audio, 0.95);
        assert!((audio[1] + 0.95).abs() < 1e-6);
        assert!((audio[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn a_quiet_chunk_is_left_exactly_as_it_is() {
        let mut audio = vec![0.001, -0.002, 0.9];
        limit_peak(&mut audio, 0.95);
        assert_eq!(audio, vec![0.001, -0.002, 0.9]);
    }

    #[test]
    fn the_ledger_accepts_a_fully_spoken_document() {
        let mut ledger = ChunkLedger::default();
        ledger.expect(3);
        ledger.spoken_chunk(44_100, 46_080);
        ledger.spoken_chunk(44_100, 46_080);
        ledger.spoken_chunk(44_100, 46_080);
        assert!(ledger.verify(44_100).is_ok());
    }

    #[test]
    fn the_ledger_refuses_a_document_that_lost_chunks() {
        let mut ledger = ChunkLedger::default();
        ledger.expect(50);
        ledger.spoken_chunk(44_100, 46_080);
        let error = ledger.verify(44_100).unwrap_err().to_string();
        assert!(error.contains("1 of 50 chunks"), "{error}");
    }

    #[test]
    fn the_ledger_refuses_a_recording_far_shorter_than_the_text() {
        let mut ledger = ChunkLedger::default();
        ledger.expect(1);
        ledger.spoken_chunk(44_100 * 10, 44_100);
        let error = ledger.verify(44_100).unwrap_err().to_string();
        assert!(error.contains("1.0 s of audio"), "{error}");
        assert!(error.contains("10.0 s"), "{error}");
    }
}
