//! Qwen3-TTS with the Hebrew LoRA merged into the talker.
//!
//! The checkpoint reads *stressed IPA*, not Hebrew script, so this
//! runtime runs the same RenikudPlus grapheme-to-phoneme front end that
//! Blue uses and hands the resulting IPA to qwentts.cpp. Voices are not
//! a fixed list here: the speaker comes from a reference recording, so
//! the `voice` argument carries a path to a WAV file rather than a name.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use blue_rs::phonemize::{Language, Phonemizer};
use qwentts_rs::{QwenTts, QwenTtsConfig, SynthesizeRequest};

use super::{Language as RuntimeLanguage, Runtime};

pub struct QwenRuntime {
    tts: QwenTts,
    phonemizer: Phonemizer,
    languages: Vec<RuntimeLanguage>,
    /// Reference audio decoded to the codec rate, cached by source path
    /// so repeated synthesis with one voice decodes the WAV once.
    reference: Option<(PathBuf, Vec<f32>)>,
}

impl QwenRuntime {
    pub fn load(talker_path: PathBuf, codec_path: PathBuf, renikud_path: PathBuf) -> Result<Self> {
        if !renikud_path.is_file() {
            bail!(
                "RenikudPlus is required for Hebrew phonemes but is missing at {}",
                renikud_path.display()
            );
        }
        let tts = QwenTts::load(QwenTtsConfig::new(&talker_path, &codec_path))
            .with_context(|| format!("load Qwen talker from {}", talker_path.display()))?;
        let phonemizer = Phonemizer::with_language(Some(&renikud_path), Language::Hebrew)
            .with_context(|| format!("load RenikudPlus ONNX from {}", renikud_path.display()))?;

        Ok(Self {
            tts,
            phonemizer,
            // The adapter was trained on Hebrew alone. The base model
            // speaks ten languages, but its output heads were retrained
            // for Hebrew phonotactics, so only Hebrew is advertised.
            languages: vec![RuntimeLanguage {
                name: "he".into(),
                id: 0,
            }],
            reference: None,
        })
    }

    /// Decodes a reference WAV to the mono 24 kHz buffer the codec wants.
    fn reference_audio(&mut self, voice: Option<&str>) -> Result<Option<Vec<f32>>> {
        let Some(voice) = voice.map(str::trim).filter(|voice| !voice.is_empty()) else {
            return Ok(None);
        };
        let path = PathBuf::from(voice);
        if !path.is_file() {
            bail!(
                "QwenTTS clones a voice from a reference recording; \
                 `{voice}` is not a readable WAV file"
            );
        }
        if let Some((cached, samples)) = self.reference.as_ref() {
            if cached == &path {
                return Ok(Some(samples.clone()));
            }
        }
        let samples = decode_reference(&path)?;
        self.reference = Some((path, samples.clone()));
        Ok(Some(samples))
    }

    fn request(&mut self, ipa: &str, voice: Option<&str>) -> Result<SynthesizeRequest> {
        let mut request = SynthesizeRequest::new(ipa);
        // The model card drives generation with language "Auto" and
        // greedy decoding, which an empty language string selects here.
        request.reference_audio = self.reference_audio(voice)?;
        Ok(request)
    }

    fn to_ipa(&mut self, text: &str) -> Result<String> {
        let ipa = self.phonemizer.g2p(text, Language::Hebrew)?;
        Ok(strip_language_tags(ipa))
    }
}

impl Runtime for QwenRuntime {
    fn languages(&self) -> &[RuntimeLanguage] {
        &self.languages
    }

    fn voices(&self) -> Option<Vec<String>> {
        // Base checkpoints carry no named speakers; the voice is the
        // reference recording, so the UI must not offer a picker.
        let speakers = self.tts.speakers();
        (!speakers.is_empty()).then_some(speakers)
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    fn phonemize(&mut self, text: &str, _language: &str) -> Result<String> {
        self.to_ipa(text)
    }

    fn diacritize(&mut self, _text: &str) -> Result<String> {
        bail!("QwenTTS does not add diacritics; select BlueTTS with Phonikud for that")
    }

    fn supported_phonemes(&self) -> Vec<char> {
        // qwentts.cpp exposes no phoneme inventory: the talker consumes
        // IPA through its BPE vocabulary, so there is no closed set to
        // validate the advanced editor against.
        Vec::new()
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        language: &str,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let ipa = self.to_ipa(text)?;
        self.synthesize_phonemes_streaming(&ipa, voice, language, on_chunk)
    }

    fn synthesize_phonemes_streaming(
        &mut self,
        phonemes: &str,
        voice: Option<&str>,
        _language: &str,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let request = self.request(phonemes, voice)?;
        let sample_rate = self.tts.sample_rate();
        self.tts
            .synthesize_streaming(request, &mut |chunk| on_chunk(chunk, sample_rate))
    }

    fn synthesize_to_file(
        &mut self,
        text: &str,
        voice: Option<&str>,
        output_path: &Path,
        _language: &str,
    ) -> Result<()> {
        let ipa = self.to_ipa(text)?;
        let request = self.request(&ipa, voice)?;
        let audio = self.tts.synthesize(request)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.tts.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(output_path, spec)?;
        for sample in audio {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        Ok(())
    }
}

/// Reads a WAV of any rate or channel count into mono at the codec rate.
fn decode_reference(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("open reference audio {}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|sample| sample as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };
    if samples.is_empty() {
        bail!("reference audio {} is empty", path.display());
    }

    let mono = downmix(&samples, spec.channels.max(1) as usize);
    Ok(resample(&mono, spec.sample_rate, qwentts_rs::SAMPLE_RATE))
}

fn downmix(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Linear resampling, which is adequate for conditioning audio: the
/// speaker encoder reads timbre, not fine spectral detail.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || from_rate == 0 || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let out_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    (0..out_len)
        .map(|index| {
            let position = index as f64 / ratio;
            let left = position.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let fraction = (position - left as f64) as f32;
            samples[left.min(samples.len() - 1)] * (1.0 - fraction) + samples[right] * fraction
        })
        .collect()
}

fn strip_language_tags(phonemes: String) -> String {
    ["en", "es", "de", "it", "he"]
        .into_iter()
        .fold(phonemes, |output, language| {
            output
                .replace(&format!("<{language}>"), "")
                .replace(&format!("</{language}>"), "")
        })
}

#[cfg(test)]
mod tests {
    use super::{downmix, resample, strip_language_tags};

    #[test]
    fn downmixes_interleaved_stereo_to_mono() {
        assert_eq!(downmix(&[1.0, -1.0, 0.5, 0.5], 2), vec![0.0, 0.5]);
        assert_eq!(downmix(&[0.25, 0.75], 1), vec![0.25, 0.75]);
    }

    #[test]
    fn resampling_keeps_rate_matched_audio_untouched() {
        let samples = [0.1, 0.2, 0.3];
        assert_eq!(resample(&samples, 24_000, 24_000), samples.to_vec());
    }

    #[test]
    fn resampling_scales_length_by_the_rate_ratio() {
        let samples = vec![0.0; 16_000];
        assert_eq!(resample(&samples, 16_000, 24_000).len(), 24_000);
        assert_eq!(resample(&samples, 48_000, 24_000).len(), 8_000);
    }

    #[test]
    fn strips_the_language_tags_renikud_emits() {
        assert_eq!(strip_language_tags("<he>ʃalˈom</he>".into()), "ʃalˈom");
    }
}
