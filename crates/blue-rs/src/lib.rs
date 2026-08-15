mod chunking;
mod npz;
pub mod handling;
pub mod phonemize;
pub mod style;
mod text;

use std::{fs, path::Path};

use anyhow::{Context, Result, anyhow, bail};
use ndarray::{Array, Array1, Array3};
use ort::{session::Session, value::Tensor};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde_json::Value;

pub use crate::chunking::ChunkingOptions;
use crate::{chunking::append_silence, text::Tokenizer};
use crate::{
    handling::{prepare_text_for_synthesis, split_prepared_by_reference_codes},
    phonemize::{Language, Phonemizer},
};

const DEFAULT_SAMPLE_RATE: usize = 44_100;
const DEFAULT_BASE_CHUNK_SIZE: usize = 512;
const DEFAULT_CHUNK_COMPRESS_FACTOR: usize = 6;
const DEFAULT_LATENT_DIM: usize = 24;
const DEFAULT_PACE_BLEND: f32 = 0.30;
const MIXED_PACE_BLEND: f32 = 0.25;
const REFERENCE_CODE_SPEED_SCALE: f32 = 0.90;
const REFERENCE_CODE_SILENCE: f32 = 0.12;

#[derive(Clone, Debug)]
pub struct SynthesisOptions {
    pub lang: String,
    pub total_step: usize,
    pub cfg_scale: f32,
    pub speed: f32,
    pub chunking: Option<ChunkingOptions>,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            lang: "he".to_string(),
            total_step: 5,
            cfg_scale: 4.0,
            speed: 1.0,
            chunking: None,
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
    /// v2 baked classifier-free guidance into the vector estimator. v2.5
    /// dropped that input and ships the unconditional embeddings instead,
    /// so guidance becomes two runs blended by the caller.
    guidance: Guidance,
    /// v2 folded and denormalized the latents inside the vocoder graph.
    /// v2.5 exports the bare vocoder, so that has to happen here, and it
    /// is only valid against the stats the checkpoint was normalized with.
    latent_stats: Option<LatentStats>,
    vocoder_input: String,
}

/// How to steer sampling toward the prompt.
enum Guidance {
    /// The estimator takes `cfg_scale` directly (v2).
    Baked,
    /// Blend a conditional and an unconditional run (v2.5).
    Uncond { u_text: Array3<f32>, u_ref: Array3<f32> },
    /// Neither available; sampling runs unguided.
    None,
}

/// Latent normalization constants from `stats.npz`.
struct LatentStats {
    mean: Vec<f32>,
    std: Vec<f32>,
    normalizer_scale: f32,
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

impl BlueTts {
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        // Voices here are precomputed styles, so the style-conditioned
        // duration predictor is the one that applies. v2 shipped only one
        // predictor and it took the style; v2.5 adds a second, z_ref
        // conditioned one under the original name, so prefer the explicit
        // style export whenever it is present.
        let style_dp = dir.join("duration_predictor_style.onnx");
        let dp = if style_dp.is_file() {
            style_dp
        } else {
            dir.join("duration_predictor.onnx")
        };

        let vector_estimator = load_session(dir.join("vector_estimator.onnx"))?;
        let vocoder = load_session(dir.join("vocoder.onnx"))?;
        let vocoder_input = vocoder
            .inputs()
            .first()
            .map(|input| input.name().to_owned())
            .unwrap_or_else(|| "latent".to_owned());
        let guidance = load_guidance(&vector_estimator, dir.join("uncond.npz"))?;
        let latent_stats = load_latent_stats(dir.join("stats.npz"))?;

        Ok(Self {
            dp: load_session(dp)?,
            text_encoder: load_session(dir.join("text_encoder.onnx"))?,
            vector_estimator,
            vocoder,
            tokenizer: Tokenizer::from_json(dir.join("vocab.json"))?,
            geometry: load_geometry(dir.join("tts.json"))?,
            guidance,
            latent_stats,
            vocoder_input,
        })
    }

    pub fn from_model_bytes(models: BlueTtsModelBytes<'_>) -> Result<Self> {
        Ok(Self {
            dp: load_session_from_memory(models.duration_predictor)?,
            text_encoder: load_session_from_memory(models.text_encoder)?,
            vector_estimator: load_session_from_memory(models.vector_estimator)?,
            vocoder: load_session_from_memory(models.vocoder)?,
            tokenizer: Tokenizer::from_json_bytes(models.vocab)?,
            geometry: ModelGeometry::default(),
            // The embedded-bytes path carries the v2 pipeline only.
            guidance: Guidance::Baked,
            latent_stats: None,
            vocoder_input: "latent".to_owned(),
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
        if let Some(chunking) = &opts.chunking {
            if chunking.enabled {
                let chunks = chunking::split_phonemes(phonemes, chunking.max_chars);
                let mut audio = Vec::new();
                let last_idx = chunks.len().saturating_sub(1);
                for (idx, chunk) in chunks.iter().enumerate() {
                    audio.extend(self.synthesize_chunk(
                        chunk,
                        style,
                        &opts,
                        seed.wrapping_add(idx as u64),
                    )?);
                    if idx != last_idx {
                        append_silence(&mut audio, self.sample_rate(), chunking.silence_seconds);
                    }
                }
                return Ok(audio);
            }
        }
        self.synthesize_chunk(phonemes, style, &opts, seed)
    }

    /// Prepare, phonemize, and synthesize raw multilingual text.
    ///
    /// `create` remains available for callers that already have IPA. This path
    /// mirrors the Space's text-facing behavior, including slow reference-code
    /// segments. Each call uses a fresh random latent seed.
    pub fn synthesize_text(
        &mut self,
        phonemizer: &mut Phonemizer,
        text: &str,
        style: &VoiceStyle,
        opts: SynthesisOptions,
    ) -> Result<Vec<f32>> {
        let language = Language::try_from(opts.lang.as_str())?;
        let prepared = prepare_text_for_synthesis(text, language.code());
        let segments = split_prepared_by_reference_codes(&prepared);
        let mut output = Vec::new();
        let mut previous_was_reference = false;
        let base_seed = rand::random::<u64>();
        let chunking = opts.chunking.clone().unwrap_or(ChunkingOptions {
            enabled: true,
            silence_seconds: 0.15,
            max_chars: Some(200),
        });

        for (segment_index, segment) in segments.iter().enumerate() {
            if segment.text.is_empty() {
                continue;
            }
            let mut segment_opts = opts.clone();
            segment_opts.chunking = None;
            if segment.is_reference_code {
                segment_opts.speed *= REFERENCE_CODE_SPEED_SCALE;
            }
            // Chunk raw text then phonemize each chunk (reference parity).
            let raw_chunks = if chunking.enabled {
                chunking::split_text(&segment.text, chunking.max_chars.unwrap_or(200))
            } else {
                vec![segment.text.clone()]
            };

            for (chunk_index, raw_chunk) in raw_chunks.iter().enumerate() {
                let chunk = phonemizer.g2p(raw_chunk, language)?;
                if chunk.is_empty() {
                    continue;
                }
                let audio = self.synthesize_chunk(
                    &chunk,
                    style,
                    &segment_opts,
                    base_seed
                        .wrapping_add((segment_index as u64) << 32)
                        .wrapping_add(chunk_index as u64),
                )?;
                if !output.is_empty() {
                    let gap = if segment.is_reference_code || previous_was_reference {
                        REFERENCE_CODE_SILENCE
                    } else {
                        chunking.silence_seconds
                    };
                    append_silence(&mut output, self.sample_rate(), gap);
                }
                output.extend(audio);
                previous_was_reference = segment.is_reference_code;
            }
        }
        Ok(normalize_generated_audio(output))
    }

    /// Prepare, phonemize, and synthesize raw text one playable chunk at a time.
    ///
    /// The callback is invoked as soon as each chunk has been synthesized, before
    /// the next chunk starts. The returned samples are the complete, normalized
    /// recording and are intended for saving after streaming playback has begun.
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
        let language = Language::try_from(opts.lang.as_str())?;
        let prepared = prepare_text_for_synthesis(text, language.code());
        let segments = split_prepared_by_reference_codes(&prepared);
        let mut output = Vec::new();
        let mut previous_was_reference = false;
        let base_seed = rand::random::<u64>();
        let chunking = opts.chunking.clone().unwrap_or(ChunkingOptions {
            enabled: true,
            silence_seconds: 0.15,
            max_chars: Some(200),
        });

        for (segment_index, segment) in segments.iter().enumerate() {
            if segment.text.is_empty() {
                continue;
            }
            let mut segment_opts = opts.clone();
            segment_opts.chunking = None;
            if segment.is_reference_code {
                segment_opts.speed *= REFERENCE_CODE_SPEED_SCALE;
            }
            // Chunk the raw text (not the phonemes) and phonemize each chunk,
            // exactly like the reference pipeline. Splitting phonemes over-splits
            // short inputs into tiny trailing fragments the vocoder renders as
            // noise; chunking raw text keeps short inputs whole.
            let raw_chunks = if chunking.enabled {
                chunking::split_text(&segment.text, chunking.max_chars.unwrap_or(200))
            } else {
                vec![segment.text.clone()]
            };

            for (chunk_index, raw_chunk) in raw_chunks.iter().enumerate() {
                let chunk = phonemizer.g2p(raw_chunk, language)?;
                if chunk.is_empty() {
                    continue;
                }
                let mut audio = self.synthesize_chunk(
                    &chunk,
                    style,
                    &segment_opts,
                    base_seed
                        .wrapping_add((segment_index as u64) << 32)
                        .wrapping_add(chunk_index as u64),
                )?;
                if !output.is_empty() {
                    let gap = if segment.is_reference_code || previous_was_reference {
                        REFERENCE_CODE_SILENCE
                    } else {
                        chunking.silence_seconds
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
        Ok(normalize_generated_audio(output))
    }

    fn synthesize_chunk(
        &mut self,
        phonemes: &str,
        style: &VoiceStyle,
        opts: &SynthesisOptions,
        seed: u64,
    ) -> Result<Vec<f32>> {
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
        let duration = blend_duration_pace(
            predicted_duration,
            text_mask.sum(),
            if has_mixed_language_tags(phonemes) {
                MIXED_PACE_BLEND
            } else {
                DEFAULT_PACE_BLEND
            },
        ) / opts.speed.max(1e-6);

        let text_emb = {
            let out = self.text_encoder.run(ort::inputs! {
                "text_ids" => Tensor::from_array(text_ids)?,
                "style_ttl" => Tensor::from_array(style.ttl.clone())?,
                "text_mask" => Tensor::from_array(text_mask.clone())?,
            })?;
            output_array3(&out[0])?
        };

        let (mut xt, latent_mask) = sample_noisy_latent(duration, self.geometry, seed);
        let total_step = Array1::from_vec(vec![opts.total_step as f32]);
        let cfg_scale = Array1::from_vec(vec![opts.cfg_scale]);

        for step in 0..opts.total_step {
            let current_step = Array1::from_vec(vec![step as f32]);
            let cond = match &self.guidance {
                // v2 steers inside the graph, so one run is the whole step.
                Guidance::Baked => {
                    let out = self.vector_estimator.run(ort::inputs! {
                        "noisy_latent" => Tensor::from_array(xt.clone())?,
                        "text_emb" => Tensor::from_array(text_emb.clone())?,
                        "style_ttl" => Tensor::from_array(style.ttl.clone())?,
                        "latent_mask" => Tensor::from_array(latent_mask.clone())?,
                        "text_mask" => Tensor::from_array(text_mask.clone())?,
                        "current_step" => Tensor::from_array(current_step.clone())?,
                        "total_step" => Tensor::from_array(total_step.clone())?,
                        "cfg_scale" => Tensor::from_array(cfg_scale.clone())?,
                    })?;
                    output_array3(&out[0])?
                }
                _ => {
                    let out = self.vector_estimator.run(ort::inputs! {
                        "noisy_latent" => Tensor::from_array(xt.clone())?,
                        "text_emb" => Tensor::from_array(text_emb.clone())?,
                        "style_ttl" => Tensor::from_array(style.ttl.clone())?,
                        "latent_mask" => Tensor::from_array(latent_mask.clone())?,
                        "text_mask" => Tensor::from_array(text_mask.clone())?,
                        "current_step" => Tensor::from_array(current_step.clone())?,
                        "total_step" => Tensor::from_array(total_step.clone())?,
                    })?;
                    output_array3(&out[0])?
                }
            };

            xt = match &self.guidance {
                Guidance::Uncond { u_text, u_ref } if opts.cfg_scale != 1.0 => {
                    // The unconditional prompt is a single frame, so its text
                    // mask is one step rather than the real one.
                    let out = self.vector_estimator.run(ort::inputs! {
                        "noisy_latent" => Tensor::from_array(xt.clone())?,
                        "text_emb" => Tensor::from_array(u_text.clone())?,
                        "style_ttl" => Tensor::from_array(u_ref.clone())?,
                        "latent_mask" => Tensor::from_array(latent_mask.clone())?,
                        "text_mask" => Tensor::from_array(Array3::<f32>::ones((1, 1, 1)))?,
                        "current_step" => Tensor::from_array(current_step)?,
                        "total_step" => Tensor::from_array(total_step.clone())?,
                    })?;
                    let uncond = output_array3(&out[0])?;
                    // The graph returns the stepped latent x + v/T, not v.
                    // Both branches share x and 1/T, so blending the outputs
                    // is exactly the Euler step on the guided velocity.
                    &uncond + opts.cfg_scale * (&cond - &uncond)
                }
                _ => cond,
            };
        }

        let latent = self.to_vocoder_input(xt)?;
        let vocoder_input = self.vocoder_input.clone();
        let wav = self.vocoder.run(ort::inputs! {
            vocoder_input.as_str() => Tensor::from_array(latent)?,
        })?;
        let wav = output_array3(&wav[0])?;
        let mut audio: Vec<f32> = wav.iter().copied().collect();
        // Match the reference pipeline exactly: drop one full latent frame from
        // each end. The trailing frame is the vocoder's noisy edge; removing it
        // is what keeps the end of every chunk (and the final chunk) clean.
        let frame_len = self.geometry.base_chunk_size * self.geometry.chunk_compress_factor;
        if audio.len() > 2 * frame_len {
            audio = audio[frame_len..audio.len() - frame_len].to_vec();
        }
        Ok(audio)
    }
}

pub struct BlueTtsModelBytes<'a> {
    pub duration_predictor: &'a [u8],
    pub text_encoder: &'a [u8],
    pub vector_estimator: &'a [u8],
    pub vocoder: &'a [u8],
    pub vocab: &'a [u8],
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

pub(crate) fn load_onnx_session(path: impl AsRef<Path>) -> Result<Session> {
    load_session(path)
}

pub(crate) fn output_array3(value: &ort::value::DynValue) -> Result<Array3<f32>> {
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    let dims: Vec<usize> = shape.iter().map(|d| *d as usize).collect();
    match dims.as_slice() {
        [a, b, c] => Ok(Array3::from_shape_vec((*a, *b, *c), data.to_vec())?),
        [a, b] => Ok(Array3::from_shape_vec((*a, 1, *b), data.to_vec())?),
        _ => bail!("expected rank-2/rank-3 f32 tensor, got shape {shape}"),
    }
}

fn load_session_from_memory(bytes: &[u8]) -> Result<Session> {
    let builder = Session::builder()
        .map_err(|e| anyhow!("{e}"))?
        .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("{e}"))?
        .with_intra_threads(8)
        .map_err(|e| anyhow!("{e}"))?
        .with_inter_threads(1)
        .map_err(|e| anyhow!("{e}"))?;
    builder
        .commit_from_memory(bytes)
        .map_err(|e| anyhow!("{e}"))
}

fn sample_noisy_latent(
    duration: f32,
    geometry: ModelGeometry,
    seed: u64,
) -> (Array3<f32>, Array3<f32>) {
    let wav_len = (duration * geometry.sample_rate as f32).max(1.0).ceil() as usize;
    let chunk = geometry.base_chunk_size * geometry.chunk_compress_factor;
    let latent_len = wav_len.div_ceil(chunk).max(1);
    let valid_latent_len = wav_len.div_ceil(chunk).max(1);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;
    let mut xt = Array::from_shape_fn(
        (
            1,
            geometry.latent_dim * geometry.chunk_compress_factor,
            latent_len,
        ),
        |_| normal.sample(&mut rng),
    );
    let mut mask = Array3::zeros((1, 1, latent_len));
    for index in 0..valid_latent_len {
        mask[[0, 0, index]] = 1.0;
    }
    for channel in 0..xt.shape()[1] {
        for index in 0..latent_len {
            xt[[0, channel, index]] *= mask[[0, 0, index]];
        }
    }
    (xt, mask)
}

fn blend_duration_pace(duration: f32, text_token_count: f32, pace_blend: f32) -> f32 {
    let blend = pace_blend.clamp(0.0, 1.0);
    let token_count = text_token_count.max(1.0);
    let predicted_dpt = duration / token_count;
    let blended_dpt = (1.0 - blend) * predicted_dpt + blend * 0.0625;
    blended_dpt * token_count
}

fn has_mixed_language_tags(phonemes: &str) -> bool {
    ["en", "es", "de", "it"]
        .iter()
        .any(|language| phonemes.contains(&format!("<{language}>")))
        && phonemes.contains("<he>")
}

fn normalize_generated_audio(mut audio: Vec<f32>) -> Vec<f32> {
    if audio.is_empty() || audio.iter().any(|sample| !sample.is_finite()) {
        return audio;
    }
    let peak = audio
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0f32, f32::max);
    if peak < 1e-6 {
        return audio;
    }
    let threshold = (peak * 0.02).max(1e-4);
    let active: Vec<f32> = audio
        .iter()
        .copied()
        .filter(|sample| sample.abs() > threshold)
        .collect();
    let source = if active.is_empty() { &audio } else { &active };
    let rms =
        (source.iter().map(|sample| sample * sample).sum::<f32>() / source.len() as f32).sqrt();
    if rms < 1e-6 {
        return audio;
    }
    let gain = (0.08 / rms).min(0.95 / peak).min(4.0);
    if gain > 1.0 {
        for sample in &mut audio {
            *sample *= gain;
        }
    }
    audio
}

impl BlueTts {
    /// Prepares the sampled latents for whichever vocoder export is loaded.
    ///
    /// v2 vocoders take the folded, still-normalized latents straight. v2.5
    /// exports the bare vocoder, so the caller denormalizes with the
    /// checkpoint's own stats and interleaves the compression factor back
    /// into the time axis: `[1, ldim*f, T]` becomes `[1, ldim, f*T]`.
    fn to_vocoder_input(&self, xt: Array3<f32>) -> Result<Array3<f32>> {
        let Some(stats) = self.latent_stats.as_ref() else {
            return Ok(xt);
        };
        let (batch, channels, frames) = xt.dim();
        let ldim = self.geometry.latent_dim;
        let factor = self.geometry.chunk_compress_factor;
        if channels != ldim * factor {
            bail!("latent has {channels} channels, expected {ldim} x {factor}");
        }
        if stats.mean.len() != channels || stats.std.len() != channels {
            bail!(
                "stats.npz describes {} channels but the latent has {channels}",
                stats.mean.len()
            );
        }

        let mut out = Array3::<f32>::zeros((batch, ldim, factor * frames));
        for b in 0..batch {
            for c in 0..ldim {
                for k in 0..factor {
                    let channel = c * factor + k;
                    let mean = stats.mean[channel];
                    let deviation = stats.std[channel];
                    for t in 0..frames {
                        let z = (xt[[b, channel, t]] / stats.normalizer_scale) * deviation + mean;
                        out[[b, c, t * factor + k]] = z;
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Picks the guidance strategy the loaded estimator supports.
fn load_guidance(estimator: &Session, uncond_path: impl AsRef<Path>) -> Result<Guidance> {
    if estimator.inputs().iter().any(|input| input.name() == "cfg_scale") {
        return Ok(Guidance::Baked);
    }
    let uncond_path = uncond_path.as_ref();
    if !uncond_path.is_file() {
        return Ok(Guidance::None);
    }
    let arrays = npz::read_npz(uncond_path)?;
    let (Some(u_text), Some(u_ref)) = (arrays.get("u_text"), arrays.get("u_ref")) else {
        return Ok(Guidance::None);
    };
    Ok(Guidance::Uncond {
        u_text: to_array3(u_text)?,
        u_ref: to_array3(u_ref)?,
    })
}

fn to_array3(array: &npz::NpyArray) -> Result<Array3<f32>> {
    match array.shape.as_slice() {
        [a, b, c] => Ok(Array3::from_shape_vec((*a, *b, *c), array.data.clone())?),
        other => bail!("expected a 3-D array, found shape {other:?}"),
    }
}

/// Latent normalization stats, present only on exports whose vocoder needs
/// the caller to denormalize.
fn load_latent_stats(path: impl AsRef<Path>) -> Result<Option<LatentStats>> {
    let path = path.as_ref();
    if !path.is_file() {
        return Ok(None);
    }
    let arrays = npz::read_npz(path)?;
    let (Some(mean), Some(std)) = (arrays.get("mean"), arrays.get("std")) else {
        bail!("{} has no mean/std", path.display());
    };
    Ok(Some(LatentStats {
        mean: mean.data.clone(),
        std: std.data.clone(),
        normalizer_scale: arrays
            .get("normalizer_scale")
            .map(|scale| scale.scalar())
            .transpose()?
            .unwrap_or(1.0),
    }))
}

fn load_geometry(path: impl AsRef<Path>) -> Result<ModelGeometry> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(ModelGeometry::default());
    }
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
    Ok(geometry)
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
    fn duration_pace_blend_moves_toward_reference() {
        let blended = blend_duration_pace(1.0, 10.0, 0.30);
        assert!((blended - 0.8875).abs() < 1e-6);
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
    fn audio_normalization_does_not_clip() {
        let audio = normalize_generated_audio(vec![0.001, -0.001, 0.002]);
        assert!(audio.iter().all(|sample| sample.abs() <= 0.95));
    }
}
