//! BlueV3 (v3.0.0) inference: text or phonemes in, 44.1 kHz audio out.
//!
//! The engine runs a BlueV3 release bundle (`notmax123/BlueV3-onnx`) the way
//! the release's own runtime does: ids wrapped in language tags, classifier
//! free guidance inside the vector estimator, the vocoder fed the normalised
//! latent as it is, the full decoded length of every chunk and a peak limit
//! of 0.95 on what leaves. The front end is in [`phonemize`].

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
pub mod phonemize;
mod text;

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array, Array1, Array3};
use ort::{session::Session, value::Tensor};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::Value;

use crate::{chunking::append_silence, text::Tokenizer};
use crate::{
    handling::{prepare_text_for_synthesis, split_prepared_by_reference_codes},
    phonemize::{Language, Phonemizer},
};

const DEFAULT_SAMPLE_RATE: usize = 44_100;
const DEFAULT_BASE_CHUNK_SIZE: usize = 512;
const DEFAULT_CHUNK_COMPRESS_FACTOR: usize = 6;
const DEFAULT_LATENT_DIM: usize = 24;
const REFERENCE_CODE_SPEED_SCALE: f32 = 0.90;
const REFERENCE_CODE_SILENCE: f32 = 0.12;
/// Raw characters a sentence is cut to when its phonemes overflow the token
/// budget, before the cut pieces are measured again.
const OVERLONG_SENTENCE_CHARS: usize = 160;

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
    join_silence: f32,
    /// Ids one chunk may hold, language tags included.
    max_chunk_tokens: usize,
    /// A chunk louder than this is scaled down to it; quieter is left alone.
    peak_limit: f32,
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

        Ok(Self {
            dp: load_session(dir.join("duration_predictor_style.onnx"))?,
            text_encoder: load_session(dir.join("text_encoder.onnx"))?,
            vector_estimator,
            vocoder,
            tokenizer,
            geometry,
            release,
            vocoder_input,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.geometry.sample_rate as u32
    }

    /// Characters accepted by the loaded model vocabulary, including IPA symbols.
    pub fn supported_phonemes(&self) -> Vec<char> {
        self.tokenizer.supported_characters()
    }

    pub fn create(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
    ) -> Result<Vec<f32>> {
        self.create_seeded(phonemes, style, opts, rand::random())
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
        self.create_seeded_streaming(phonemes, style, opts, rand::random(), on_chunk)
    }

    fn create_seeded_streaming<F>(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
        seed: u64,
        mut on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let opts = canonical_options(opts)?;
        let chunks = self.phoneme_chunks(phonemes, &opts.lang)?;
        let mut ledger = ChunkLedger::default();
        ledger.expect(chunks.len());
        let mut audio = Vec::new();
        for (idx, chunk) in chunks.iter().enumerate() {
            let mut piece = Vec::new();
            if idx != 0 {
                append_silence(&mut piece, self.sample_rate(), self.release.join_silence);
            }
            let spoken =
                self.synthesize_chunk(chunk, style, &opts, seed.wrapping_add(idx as u64))?;
            ledger.spoken_chunk(spoken.requested_samples, spoken.samples.len());
            piece.extend(spoken.samples);
            on_chunk(&piece)?;
            audio.extend(piece);
        }
        ledger.verify(self.sample_rate())?;
        Ok(audio)
    }

    /// Phonemes a caller wrote, one sentence a chunk (a fragment of a few
    /// words joins the sentence after it, as on the text path), each inside
    /// the token budget. A chunk with nothing to say (punctuation alone) is
    /// dropped rather than spoken as noise.
    fn phoneme_chunks(&self, phonemes: &str, lang: &str) -> Result<Vec<String>> {
        let mut chunks = Vec::new();
        let sentences =
            chunking::join_phoneme_fragments(chunking::split_phoneme_sentences(phonemes));
        for sentence in sentences {
            if !chunking::is_speakable(&sentence) {
                continue;
            }
            if self.tokenizer.token_count(&sentence, lang)? <= self.release.max_chunk_tokens {
                chunks.push(sentence);
                continue;
            }
            // Every character is at most one id and the tags and final mark
            // add fewer than twenty, so pieces this long always fit.
            let budget = self.release.max_chunk_tokens.saturating_sub(20).max(1);
            chunks.extend(chunking::split_phonemes(&sentence, Some(budget)));
        }
        Ok(chunks)
    }

    /// Prepare, phonemize, and synthesize raw multilingual text.
    ///
    /// `create` remains available for callers that already have IPA. Each call
    /// uses a fresh random latent seed.
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
    /// before the next chunk starts. A chunk is final when it is made: the
    /// returned samples are exactly the chunks the callback saw, joined.
    ///
    /// One sentence is one chunk. The release packs sentences up to its token
    /// budget, and the model then runs them together with almost no pause
    /// (seven Hebrew sentences: 1.6 s of pauses in 37 s). Given one sentence,
    /// it ends it the way it ends an utterance, and the join silence follows.
    pub fn synthesize_text_streaming<F>(
        &mut self,
        phonemizer: &mut Phonemizer,
        text: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
        mut on_chunk: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let opts = canonical_options(opts)?;
        let language = Language::try_from(opts.lang.as_str())?;
        let prepared = prepare_text_for_synthesis(text, language.code());
        let segments = split_prepared_by_reference_codes(&prepared);
        let mut output = Vec::new();
        let mut previous_was_reference = false;
        let mut ledger = ChunkLedger::default();
        let base_seed = rand::random::<u64>();

        for (segment_index, segment) in segments.iter().enumerate() {
            if segment.text.is_empty() {
                continue;
            }
            let mut segment_opts = opts.clone();
            if segment.is_reference_code {
                segment_opts.speed *= REFERENCE_CODE_SPEED_SCALE;
            }
            // Chunk the raw text (not the phonemes) and phonemize each chunk.
            let mut pending: std::collections::VecDeque<String> =
                chunking::split_sentence_chunks(&segment.text).into();
            ledger.expect(pending.len());
            let mut chunk_index = 0u64;

            while let Some(raw_chunk) = pending.pop_front() {
                // Punctuation alone has nothing to say, whatever the G2P
                // makes of it: the model would read it as a second of noise.
                if !chunking::is_speakable(&raw_chunk) {
                    ledger.silent_chunk();
                    continue;
                }
                let chunk = phonemizer.g2p(&raw_chunk, language)?;
                if chunk.is_empty() {
                    // A chunk of nothing but punctuation or whitespace has no
                    // speech in it and is fine to skip. A chunk that had
                    // letters and came back empty is a phonemizer failure,
                    // and skipping it was how a document could lose most of
                    // its text with no error and a "success" at the end.
                    if raw_chunk.chars().any(char::is_alphabetic) {
                        anyhow::bail!(
                            "no phonemes were produced for this text, so it cannot be spoken: {:?}",
                            raw_chunk.trim()
                        );
                    }
                    ledger.silent_chunk();
                    continue;
                }
                // The budget is in ids after G2P and tagging, so a sentence
                // that overflows it is cut smaller and read again.
                if self.tokenizer.token_count(&chunk, language.code())?
                    > self.release.max_chunk_tokens
                {
                    let limit = OVERLONG_SENTENCE_CHARS.min(raw_chunk.chars().count() * 2 / 3);
                    let parts = chunking::hard_split_words(&raw_chunk, limit.max(16));
                    if parts.len() > 1 {
                        ledger.expect(parts.len() - 1);
                        for part in parts.into_iter().rev() {
                            pending.push_front(part);
                        }
                        continue;
                    }
                }
                let spoken = self.synthesize_chunk(
                    &chunk,
                    style,
                    &segment_opts,
                    base_seed
                        .wrapping_add((segment_index as u64) << 32)
                        .wrapping_add(chunk_index),
                )?;
                chunk_index += 1;
                ledger.spoken_chunk(spoken.requested_samples, spoken.samples.len());
                let mut audio = spoken.samples;
                if !output.is_empty() {
                    let gap = if segment.is_reference_code || previous_was_reference {
                        REFERENCE_CODE_SILENCE
                    } else {
                        self.release.join_silence
                    };
                    let mut playable = Vec::new();
                    append_silence(&mut playable, self.sample_rate(), gap);
                    playable.append(&mut audio);
                    audio = playable;
                }
                on_chunk(&audio)?;
                output.extend_from_slice(&audio);
                previous_was_reference = segment.is_reference_code;
            }
        }
        ledger.verify(self.sample_rate())?;
        Ok(output)
    }

    fn synthesize_chunk(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: &SynthesisOptions,
        seed: u64,
    ) -> Result<ChunkAudio> {
        let (text_ids, text_mask) = self.tokenizer.encode_batch(&[phonemes], &[&opts.lang])?;

        // Scoped so the session outputs, which borrow self, are released
        // before the vocoder call needs self again.
        let predicted_duration = {
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
        if !predicted_duration.is_finite() || predicted_duration <= 0.0 {
            bail!("the duration predictor returned {predicted_duration} s for this text");
        }
        // The predicted duration as it is: the release blends no reference pace.
        let duration = predicted_duration / opts.speed.max(1e-6);

        let text_emb = {
            let out = self.text_encoder.run(ort::inputs! {
                "text_ids" => Tensor::from_array(text_ids)?,
                "style_ttl" => Tensor::from_array(style.ttl.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
            })?;
            output_array3(&out[0])?
        };

        let requested_samples = requested_sample_count(duration, self.geometry.sample_rate);
        let (mut xt, latent_mask) = sample_noisy_latent(duration, self.geometry, seed);
        let total_step = Array1::from_vec(vec![opts.total_step as f32]);
        let cfg_scale = Array1::from_vec(vec![opts.cfg_scale]);

        // One Euler step a run, with both guidance branches inside the graph.
        for step in 0..opts.total_step {
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
        Ok(ChunkAudio {
            samples,
            requested_samples,
        })
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

/// Noise for `ceil(duration * rate / frame)` latent frames and a mask of ones
/// over them, as the release samples it.
fn sample_noisy_latent(
    duration: f32,
    geometry: ModelGeometry,
    seed: u64,
) -> (Array3<f32>, Array3<f32>) {
    let wav_len = requested_sample_count(duration, geometry.sample_rate);
    let chunk = geometry.base_chunk_size * geometry.chunk_compress_factor;
    let latent_len = wav_len.div_ceil(chunk).max(1);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    let xt = Array::from_shape_fn(
        (
            1,
            geometry.latent_dim * geometry.chunk_compress_factor,
            latent_len,
        ),
        |_| normal.sample(&mut rng),
    );
    (xt, Array3::ones((1, 1, latent_len)))
}

/// blue_onnx's `limit_peak`: one gain for the whole chunk, attenuation only.
/// Audio whose peak is above `limit` is scaled so the peak is `limit`; quieter
/// audio is untouched, so nothing is boosted, compressed or clipped.
fn limit_peak(audio: &mut [f32], limit: f32) {
    let peak = audio
        .iter()
        .filter(|sample| sample.is_finite())
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    if peak > limit && limit > 0.0 {
        let gain = limit / peak;
        for sample in audio {
            *sample *= gain;
        }
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
    /// Chunks with nothing to say in them (punctuation or whitespace only).
    silent: usize,
    requested_samples: usize,
    produced_samples: usize,
}

impl ChunkLedger {
    fn expect(&mut self, chunks: usize) {
        self.expected += chunks;
    }

    fn silent_chunk(&mut self) {
        self.silent += 1;
    }

    fn spoken_chunk(&mut self, requested_samples: usize, produced_samples: usize) {
        self.spoken += 1;
        self.requested_samples += requested_samples;
        self.produced_samples += produced_samples;
    }

    /// Fail loudly when the recording cannot account for the text.
    fn verify(&self, sample_rate: u32) -> Result<()> {
        if self.spoken + self.silent != self.expected {
            bail!(
                "synthesis lost text: {} of {} chunks were spoken and {} had nothing to say",
                self.spoken,
                self.expected,
                self.silent
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
        release.join_silence = value as f32;
    }
    if let Some(value) = inference["max_chunk_tokens"].as_u64() {
        release.max_chunk_tokens = value as usize;
    }
    if let Some(value) = inference["peak_limit"].as_f64() {
        release.peak_limit = value as f32;
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
    fn punctuation_alone_is_silence_not_noise() {
        let (mut tts, mut phonemizer, style) = bundle();
        let english = || SynthesisOptions {
            lang: "en".to_owned(),
            ..SynthesisOptions::default()
        };
        for text in ["...", " ?! ", "…"] {
            let audio = tts
                .synthesize_text(&mut phonemizer, text, &style, english())
                .unwrap();
            assert!(audio.is_empty(), "{text:?} gave {} samples", audio.len());
            let audio = tts.create(text, &style, english()).unwrap();
            assert!(audio.is_empty(), "{text:?} gave {} samples", audio.len());
        }
        let audio = tts
            .synthesize_text(&mut phonemizer, "Hello there. ...", &style, english())
            .unwrap();
        assert!(!audio.is_empty());
    }

    #[test]
    fn seeded_latents_are_reproducible() {
        let geometry = ModelGeometry::default();
        let (first, first_mask) = sample_noisy_latent(1.0, geometry, 42);
        let (second, second_mask) = sample_noisy_latent(1.0, geometry, 42);
        assert_eq!(first, second);
        assert_eq!(first_mask, second_mask);
    }

    #[test]
    fn the_latent_covers_the_duration_in_whole_frames() {
        // T = ceil(duration * 44100 / 3072), 144 channels, a mask of ones.
        let (latent, mask) = sample_noisy_latent(3.62, ModelGeometry::default(), 7);
        assert_eq!(latent.dim(), (1, 144, 52));
        assert_eq!(mask.dim(), (1, 1, 52));
        assert!(mask.iter().all(|value| *value == 1.0));
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
        ledger.silent_chunk();
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
