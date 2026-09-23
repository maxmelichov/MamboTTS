//! Audio file encoding shared by the server and the desktop app.
//!
//! Synthesis produces mono float samples. Inside the app they always travel as
//! 16-bit WAV, because every webview can decode it and nothing is lost. Only a
//! file that leaves the app (a saved recording, an HTTP response that asked for
//! it) is re-encoded, and this crate is the one place that knows how.
//!
//! MP3 comes from LAME, vendored and built from source by `mp3lame-sys`
//! (autotools on macOS and Linux, `cc` on Windows), so it adds no system
//! dependency on any platform.

use std::{io::Cursor, path::Path, str::FromStr};

use anyhow::{Context, Result, anyhow, bail};
use mp3lame_encoder::{Bitrate, Builder, FlushGap, Mode, MonoPcm, Quality};
use serde::{Deserialize, Serialize};

/// A container and codec a recording can be written as.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioFormat {
    /// 16-bit PCM WAV at the synthesis rate. Lossless and large.
    Wav,
    /// Constant bitrate MP3 at the given kilobits per second.
    Mp3 { kbps: u32 },
}

/// The MP3 bitrate an HTTP caller gets when it asks for `mp3` without saying more.
pub const DEFAULT_MP3_KBPS: u32 = 128;

impl AudioFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 { .. } => "mp3",
        }
    }

    pub fn content_type(self) -> &'static str {
        match self {
            Self::Wav => "audio/wav",
            Self::Mp3 { .. } => "audio/mpeg",
        }
    }
}

impl FromStr for AudioFormat {
    type Err = anyhow::Error;

    /// Parse an OpenAI-style `response_format`. Empty means WAV, so callers
    /// that never sent the field keep getting what they always got.
    fn from_str(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "wav" => Ok(Self::Wav),
            "mp3" => Ok(Self::Mp3 {
                kbps: DEFAULT_MP3_KBPS,
            }),
            other => bail!("unsupported response_format `{other}`; expected wav or mp3"),
        }
    }
}

/// The five sizes the app offers when a recording is saved.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportQuality {
    VerySmall,
    Small,
    #[default]
    Normal,
    Large,
    VeryLarge,
}

impl ExportQuality {
    /// Speech is mono, so even the smallest of these stays clear; the step
    /// from one to the next is mostly about how much air and sibilance
    /// survives. The largest is the untouched synthesis output.
    pub fn format(self) -> AudioFormat {
        match self {
            Self::VerySmall => AudioFormat::Mp3 { kbps: 64 },
            Self::Small => AudioFormat::Mp3 { kbps: 96 },
            Self::Normal => AudioFormat::Mp3 { kbps: 128 },
            Self::Large => AudioFormat::Mp3 { kbps: 192 },
            Self::VeryLarge => AudioFormat::Wav,
        }
    }
}

/// Encode mono float samples in `[-1, 1]`.
pub fn encode(samples: &[f32], sample_rate: u32, format: AudioFormat) -> Result<Vec<u8>> {
    match format {
        AudioFormat::Wav => encode_wav(samples, sample_rate),
        AudioFormat::Mp3 { kbps } => encode_mp3(samples, sample_rate, kbps),
    }
}

/// 16-bit mono PCM WAV, the format synthesis output travels in.
pub fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(cursor.into_inner())
}

fn bitrate(kbps: u32) -> Result<Bitrate> {
    Ok(match kbps {
        32 => Bitrate::Kbps32,
        48 => Bitrate::Kbps48,
        64 => Bitrate::Kbps64,
        96 => Bitrate::Kbps96,
        128 => Bitrate::Kbps128,
        160 => Bitrate::Kbps160,
        192 => Bitrate::Kbps192,
        256 => Bitrate::Kbps256,
        320 => Bitrate::Kbps320,
        other => bail!("unsupported MP3 bitrate {other} kbps"),
    })
}

/// Constant bitrate mono MP3.
///
/// LAME picks the output sample rate for the bitrate, which at the low end
/// means it resamples on its own rather than spending bits on content speech
/// does not have.
pub fn encode_mp3(samples: &[f32], sample_rate: u32, kbps: u32) -> Result<Vec<u8>> {
    let mut builder = Builder::new().ok_or_else(|| anyhow!("could not start the MP3 encoder"))?;
    builder
        .set_num_channels(1)
        .map_err(|err| anyhow!("MP3 channels: {err}"))?;
    builder
        .set_mode(Mode::Mono)
        .map_err(|err| anyhow!("MP3 mode: {err}"))?;
    builder
        .set_sample_rate(sample_rate)
        .map_err(|err| anyhow!("MP3 sample rate {sample_rate}: {err}"))?;
    builder
        .set_brate(bitrate(kbps)?)
        .map_err(|err| anyhow!("MP3 bitrate {kbps}: {err}"))?;
    // 2 is what LAME itself recommends: near the best quality it can reach at
    // a fraction of the time, which matters on a long recording.
    builder
        .set_quality(Quality::NearBest)
        .map_err(|err| anyhow!("MP3 quality: {err}"))?;
    // The Xing header would have to be patched in after the fact; a constant
    // bitrate file is seekable without it.
    builder
        .set_to_write_vbr_tag(false)
        .map_err(|err| anyhow!("MP3 tag: {err}"))?;
    let mut encoder = builder
        .build()
        .map_err(|err| anyhow!("MP3 encoder: {err}"))?;

    let clamped: Vec<f32> = samples
        .iter()
        .map(|sample| sample.clamp(-1.0, 1.0))
        .collect();
    let mut output = Vec::new();
    // Encoding in slices keeps the scratch space LAME asks for bounded.
    for block in clamped.chunks(1 << 16) {
        output.reserve(mp3lame_encoder::max_required_buffer_size(block.len()));
        encoder
            .encode_to_vec(MonoPcm(block), &mut output)
            .map_err(|err| anyhow!("MP3 encode: {err}"))?;
    }
    // The flush that pads the last frame with silence rather than cutting it,
    // so the end of the take is never lost.
    output.reserve(7200);
    encoder
        .flush_to_vec::<FlushGap>(&mut output)
        .map_err(|err| anyhow!("MP3 flush: {err}"))?;
    Ok(output)
}

/// Read a WAV into mono float samples, mixing down anything with more channels.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).context("read WAV header")?;
    let spec = reader.spec();
    let channels = usize::from(spec.channels.max(1));
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample.saturating_sub(1))) as f32;
            reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / scale))
                .collect::<Result<_, _>>()?
        }
    };
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    };
    Ok((samples, spec.sample_rate))
}

/// Write the WAV at `source` to `destination` in `format`. A WAV destination
/// is a plain copy, so the largest size stays byte for byte what was synthesized.
pub fn export_wav_file(source: &Path, destination: &Path, format: AudioFormat) -> Result<()> {
    if format == AudioFormat::Wav {
        std::fs::copy(source, destination)
            .with_context(|| format!("copy {} to {}", source.display(), destination.display()))?;
        return Ok(());
    }
    let bytes = std::fs::read(source).with_context(|| format!("read {}", source.display()))?;
    let (samples, sample_rate) = decode_wav(&bytes)?;
    let encoded = encode(&samples, sample_rate, format)?;
    std::fs::write(destination, encoded)
        .with_context(|| format!("write {}", destination.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(seconds: f32, sample_rate: u32) -> Vec<f32> {
        let count = (seconds * sample_rate as f32) as usize;
        (0..count)
            .map(|index| {
                let t = index as f32 / sample_rate as f32;
                0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                    + 0.2 * (2.0 * std::f32::consts::PI * 1760.0 * t).sin()
            })
            .collect()
    }

    #[test]
    fn response_formats_parse_and_default_to_wav() {
        assert_eq!("".parse::<AudioFormat>().unwrap(), AudioFormat::Wav);
        assert_eq!("WAV".parse::<AudioFormat>().unwrap(), AudioFormat::Wav);
        assert_eq!(
            "mp3".parse::<AudioFormat>().unwrap(),
            AudioFormat::Mp3 { kbps: 128 }
        );
        assert!("opus".parse::<AudioFormat>().is_err());
    }

    #[test]
    fn quality_names_match_what_the_app_sends() {
        let parsed: ExportQuality = serde_json::from_str("\"very_small\"").unwrap();
        assert_eq!(parsed, ExportQuality::VerySmall);
        assert_eq!(ExportQuality::default(), ExportQuality::Normal);
        assert_eq!(ExportQuality::VeryLarge.format(), AudioFormat::Wav);
    }

    #[test]
    fn mp3_sizes_follow_the_bitrate_and_are_valid_frames() {
        let sample_rate = 44_100;
        let samples = tone(10.0, sample_rate);
        let wav = encode_wav(&samples, sample_rate).unwrap();
        let mut previous = 0;
        for quality in [
            ExportQuality::VerySmall,
            ExportQuality::Small,
            ExportQuality::Normal,
            ExportQuality::Large,
        ] {
            let AudioFormat::Mp3 { kbps } = quality.format() else {
                panic!("expected MP3 for {quality:?}");
            };
            let mp3 = encode(&samples, sample_rate, quality.format()).unwrap();
            // An MPEG audio frame starts with 11 set sync bits.
            assert_eq!(mp3[0], 0xFF, "{quality:?}");
            assert_eq!(mp3[1] & 0xE0, 0xE0, "{quality:?}");
            let expected = kbps as usize * 1000 / 8 * 10;
            assert!(
                mp3.len() > expected * 9 / 10 && mp3.len() < expected * 11 / 10,
                "{quality:?}: {} bytes, expected about {expected}",
                mp3.len()
            );
            assert!(mp3.len() > previous);
            assert!(mp3.len() < wav.len() / 3);
            previous = mp3.len();
        }
    }

    #[test]
    fn wav_round_trips_through_decode() {
        let samples = tone(0.5, 22_050);
        let wav = encode_wav(&samples, 22_050).unwrap();
        let (decoded, sample_rate) = decode_wav(&wav).unwrap();
        assert_eq!(sample_rate, 22_050);
        assert_eq!(decoded.len(), samples.len());
        for (a, b) in decoded.iter().zip(&samples) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn exporting_wav_copies_and_mp3_shrinks() {
        let dir = std::env::temp_dir().join(format!("mambotts-audio-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let source = dir.join("take.wav");
        std::fs::write(&source, encode_wav(&tone(3.0, 44_100), 44_100).unwrap()).unwrap();

        let copy = dir.join("copy.wav");
        export_wav_file(&source, &copy, AudioFormat::Wav).unwrap();
        assert_eq!(
            std::fs::read(&copy).unwrap(),
            std::fs::read(&source).unwrap()
        );

        let mp3 = dir.join("small.mp3");
        export_wav_file(&source, &mp3, ExportQuality::Small.format()).unwrap();
        let mp3_len = std::fs::metadata(&mp3).unwrap().len();
        let wav_len = std::fs::metadata(&source).unwrap().len();
        assert!(mp3_len * 5 < wav_len, "{mp3_len} vs {wav_len}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
