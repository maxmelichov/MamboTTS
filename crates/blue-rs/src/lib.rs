//! BlueV3 (v3.0.0) inference: text or phonemes in, 44.1 kHz audio out.
//!
//! The flow around the graphs is the BlueTTS site's (`js/onnx-web-worker.js`
//! `synthesize`, `chunkFront`, `chunkBack`; `js/bluetts-text.js`), step for
//! step: the text prepared as the site prepares it, G2P over the whole text
//! (Russian a sentence at a time), the phonemes cut into chunks of at most
//! 300 ids a sentence each, ids wrapped in language tags, the predicted
//! duration as it is (no pace blend, inline `<xx>` spans included) divided
//! by the speed in float32, the latent length and mask as the site
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

mod chunking;
pub mod handling;
pub mod hebrew_numbers;
mod ipa;
pub mod numpy_rng;
pub mod phonemize;
mod text;
pub mod ve_gpu;

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array1, Array3};
use ort::{session::Session, value::Tensor};
use serde_json::Value;

use crate::{
    chunking::{pack_chunks, silence_samples},
    handling::prepare_text,
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

#[derive(Clone, Debug)]
pub struct SynthesisOptions {
    pub lang: String,
    pub total_step: usize,
    pub cfg_scale: f32,
    pub speed: f32,
}

impl Default for SynthesisOptions {
    /// The release's operating point: 8 Euler steps, guidance 4.0, speed 1.0.
    fn default() -> Self {
        Self {
            lang: "he".to_string(),
            total_step: 8,
            cfg_scale: 4.0,
            speed: 1.0,
        }
    }
}

#[derive(Clone)]
pub struct VoiceStyle {
    ttl: Array3<f32>,
    dp: Array3<f32>,
}

impl VoiceStyle {
    pub fn new(ttl: Array3<f32>, dp: Array3<f32>) -> Self {
        Self { ttl, dp }
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
    vocoder: Session,
    tokenizer: Tokenizer,
    geometry: ModelGeometry,
    release: ReleasePoint,
    vocoder_input: String,
    last_chunks: Vec<ChunkInfo>,
    /// The site's Rust engine for the flow (wgpu: Metal, Vulkan or DX12).
    /// `None` without a usable GPU, with `MAMBOTTS_VE_BACKEND=ort`, or after
    /// it failed once; the ONNX Runtime graph runs the flow then.
    gpu: Option<ve_gpu::GpuVectorEstimator>,
}

/// Open the GPU flow engine unless disabled; any failure means ONNX Runtime.
fn open_gpu_flow(path: &Path) -> Option<ve_gpu::GpuVectorEstimator> {
    if std::env::var("MAMBOTTS_VE_BACKEND").is_ok_and(|value| value.eq_ignore_ascii_case("ort")) {
        return None;
    }
    match ve_gpu::GpuVectorEstimator::open(path) {
        Ok(Some(engine)) => {
            eprintln!("BlueV3 flow on the GPU engine ({})", engine.adapter());
            Some(engine)
        }
        Ok(None) => {
            eprintln!("BlueV3 flow on ONNX Runtime: no GPU adapter");
            None
        }
        Err(err) => {
            eprintln!("BlueV3 flow on ONNX Runtime: the GPU engine did not open: {err:#}");
            None
        }
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
    /// Seconds after the speed.
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
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let tokenizer = Tokenizer::from_json(dir.join("vocab.json"))?;
        if !tokenizer.is_tagged() {
            bail!(
                "{} is not a BlueV3 bundle: its vocab.json has no `lang_tag`. BlueTTS 2.5 and older bundles are no longer supported; download the current model",
                dir.display()
            );
        }
        let vector_estimator = load_session(dir.join("vector_estimator.onnx"))?;
        if !vector_estimator
            .inputs()
            .iter()
            .any(|input| input.name() == "cfg_scale")
        {
            bail!(
                "{} is not a BlueV3 vector estimator: it has no `cfg_scale` input",
                dir.join("vector_estimator.onnx").display()
            );
        }
        let vocoder = load_session(dir.join("vocoder.onnx"))?;
        let vocoder_input = vocoder
            .inputs()
            .first()
            .map(|input| input.name().to_owned())
            .unwrap_or_else(|| "latent".to_owned());
        let (geometry, release) = load_config(dir.join("tts.json"))?;
        let gpu = open_gpu_flow(&dir.join("vector_estimator.onnx"));

        Ok(Self {
            dp: load_session(dir.join("duration_predictor_style.onnx"))?,
            text_encoder: load_session(dir.join("text_encoder.onnx"))?,
            vector_estimator,
            vocoder,
            tokenizer,
            geometry,
            release,
            vocoder_input,
            last_chunks: Vec::new(),
            gpu,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.geometry.sample_rate as u32
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
        let language = Language::try_from(lang)?;
        let prepared = prepare_text(text, language.code());
        let phonemes = phonemizer.phonemize_units(&prepared, language)?;
        if phonemes.iter().all(String::is_empty) && prepared.chars().any(char::is_alphabetic) {
            // Letters that came back as nothing are a G2P failure, not
            // silence: say so instead of returning an empty recording.
            bail!(
                "no phonemes were produced for this text, so it cannot be spoken: {:?}",
                prepared
            );
        }
        let chunks = pack_chunks(
            phonemes.iter().map(String::as_str),
            language.code(),
            self.release.max_chunk_tokens,
        );
        Ok(TextPlan {
            prepared,
            phonemes,
            chunks,
        })
    }

    /// The ids a chunk of phonemes is fed to the graphs as.
    pub fn token_ids(&self, phonemes: &str, lang: &str) -> Result<Vec<i64>> {
        let (ids, _) = self.tokenizer.encode_batch(&[phonemes], &[lang])?;
        Ok(ids.iter().copied().collect())
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
        let duration = chunk_duration(raw_duration, opts.speed);

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
        // as the graph's to ~1e-5. If it fails, ONNX Runtime takes over for
        // this chunk and every later one.
        let mut gpu_done = false;
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
                    eprintln!("GPU flow failed, using ONNX Runtime from now on: {err:#}");
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

/// The model's own duration divided by the speed in float32, as the site's
/// `chunkFront` computes it: release bundles blend no pace, inline
/// `<xx>…</xx>` spans included (BlueTTS a6cc157).
fn chunk_duration(raw: f32, speed: f32) -> f32 {
    raw / speed.max(1e-6)
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

fn load_session(path: impl AsRef<Path>) -> Result<Session> {
    let path = path.as_ref();
    Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(8)
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
    let rate = geometry.sample_rate as f64;
    let frame = (geometry.base_chunk_size * geometry.chunk_compress_factor) as f64;
    let frames = ((f64::from(duration) * rate).trunc() + frame - 1.0) / frame;
    let frames = frames.floor().max(1.0) as usize;
    let mask_frames = (f64::from((duration * geometry.sample_rate as f32).trunc()) + frame - 1.0) / frame;
    let mask_frames = (mask_frames.floor().max(0.0) as usize).min(frames);
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
fn load_config(path: impl AsRef<Path>) -> Result<(ModelGeometry, ReleasePoint)> {
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
    Ok((geometry, release))
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
        (
            BlueTts::from_dir(&dir).unwrap(),
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
    #[ignore = "requires MAMBOTTS_BLUE_MODEL_DIR pointing to an installed BlueV3 bundle"]
    fn punctuation_alone_is_read_as_the_site_reads_it() {
        let (tts, mut phonemizer, _) = bundle();
        // The site hands polyg2p nothing for a piece with no letters: the
        // marks are the chunk.
        let plan = tts.plan_text(&mut phonemizer, "...", "en").unwrap();
        assert_eq!(plan.chunks, vec!["..."]);
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
    fn the_duration_is_the_models_divided_by_speed_in_float32() {
        assert_eq!(chunk_duration(2.0, 0.9), 2.0f32 / 0.9f32);
        assert_eq!(chunk_duration(2.0, 1.0), 2.0);
        assert_eq!(chunk_duration(2.0, 0.0), 2.0f32 / 1e-6f32);
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
