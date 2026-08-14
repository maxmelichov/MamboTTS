use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{Context, Result, bail};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};

use super::{Language as RuntimeLanguage, Runtime};

const VOICES: [(&str, &str); 2] = [("Rotem", "female1.json"), ("Roi", "male1.json")];

pub struct BlueRuntime {
    tts: BlueTts,
    phonemizer: Phonemizer,
    styles: HashMap<String, VoiceStyle>,
    languages: Vec<RuntimeLanguage>,
    hebrew_g2p_engine: String,
    phonikud_path: Option<PathBuf>,
}

impl BlueRuntime {
    pub fn load(
        model_dir: PathBuf,
        renikud_path: PathBuf,
        hebrew_g2p_engine: String,
        phonikud_path: Option<PathBuf>,
        speaker: u8,
        target_speaker: u8,
    ) -> Result<Self> {
        if hebrew_g2p_engine == "phonikud" && !phonikud_path.as_ref().is_some_and(|path| path.is_file()) {
            bail!("Phonikud is selected but its ONNX model has not been downloaded");
        }
        if hebrew_g2p_engine == "renikud" && !renikud_path.is_file() {
            bail!(
                "RenikudPlus is selected but its ONNX model is missing at {}",
                renikud_path.display()
            );
        }
        let tts = BlueTts::from_dir(&model_dir)
            .with_context(|| format!("load Blue ONNX models from {}", model_dir.display()))?;
        // RenikudPlus runs natively in-process (ONNX). Do not shell out to uv/git —
        // that downloaded a second package copy and hung first Hebrew synth.
        let mut phonemizer = if hebrew_g2p_engine == "renikud" {
            Phonemizer::with_language(Some(&renikud_path), Language::English).with_context(|| {
                format!("load RenikudPlus ONNX from {}", renikud_path.display())
            })?
        } else {
            Phonemizer::with_language(None::<PathBuf>, Language::English)?
        };
        phonemizer.set_speakers(speaker, target_speaker);

        let voices_dir = model_dir.join("voices");
        let mut styles = HashMap::new();
        for (id, file) in VOICES {
            let path = voices_dir.join(file);
            let style = VoiceStyle::from_json(&path)
                .with_context(|| format!("load Blue voice style {}", path.display()))?;
            styles.insert(id.to_owned(), style);
        }
        // Keep legacy ids working for existing clients.
        styles.insert(
            "female1".to_owned(),
            VoiceStyle::from_json(voices_dir.join("female1.json"))?,
        );
        styles.insert(
            "male1".to_owned(),
            VoiceStyle::from_json(voices_dir.join("male1.json"))?,
        );

        Ok(Self {
            tts,
            phonemizer,
            styles,
            languages: vec![
                language("en", 0),
                language("he", 1),
                language("es", 2),
                language("de", 3),
                language("it", 4),
            ],
            hebrew_g2p_engine,
            phonikud_path,
        })
    }

    fn language_for(text: &str, requested: &str) -> Result<(Language, &'static str)> {
        let requested = requested.trim().to_ascii_lowercase();
        let code = if requested.is_empty() || requested == "auto" {
            if text
                .chars()
                .any(|character| ('\u{0590}'..='\u{05ff}').contains(&character))
            {
                "he"
            } else {
                "en"
            }
        } else {
            requested.as_str()
        };
        let language = match code {
            "en" | "en-us" => Language::English,
            "he" => Language::Hebrew,
            "es" => Language::Spanish,
            "de" | "ge" => Language::German,
            "it" => Language::Italian,
            _ => bail!("unsupported Blue language `{code}`; expected he, en, es, de, or it"),
        };
        Ok((language, language.code()))
    }

    fn normalize_voice(voice: &str) -> &str {
        match voice.trim() {
            "female1" => "Rotem",
            "male1" => "Roi",
            other => other,
        }
    }

    fn phonikud(&self, text: &str, mode: &str) -> Result<String> {
        let model = self.phonikud_path.as_ref().context("Phonikud model is unavailable")?;
        let script = r#"import sys
from phonikud_onnx import Phonikud
from phonikud import phonemize
model = Phonikud(sys.argv[2])
text = sys.stdin.read()
vocalized = model.add_diacritics(text)
print(vocalized if sys.argv[1] == "diacritize" else phonemize(vocalized))"#;
        let uv = std::env::var("MAMBORAMBO_UV_PATH").unwrap_or_else(|_| {
            let homebrew_uv = Path::new("/opt/homebrew/bin/uv");
            if homebrew_uv.is_file() {
                homebrew_uv.display().to_string()
            } else {
                "uv".to_owned()
            }
        });
        let mut child = Command::new(uv)
            .args(["run", "--with", "phonikud==0.4.1", "--with", "phonikud-onnx", "python", "-c", script, mode])
            .arg(model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("start Phonikud via uv; install uv to use this engine")?;
        use std::io::Write;
        child.stdin.take().context("open Phonikud stdin")?.write_all(text.as_bytes())?;
        let output = child.wait_with_output().context("run Phonikud")?;
        if !output.status.success() {
            bail!("Phonikud failed: {}", String::from_utf8_lossy(&output.stderr).trim());
        }
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
    }
}

impl Runtime for BlueRuntime {
    fn languages(&self) -> &[RuntimeLanguage] {
        &self.languages
    }

    fn voices(&self) -> Option<Vec<String>> {
        Some(VOICES.into_iter().map(|(id, _)| id.to_owned()).collect())
    }

    fn sample_rate(&self) -> u32 {
        self.tts.sample_rate()
    }

    fn phonemize(&mut self, text: &str, language: &str) -> Result<String> {
        let (language, _) = Self::language_for(text, language)?;
        if language == Language::Hebrew && self.hebrew_g2p_engine == "phonikud" {
            return self.phonikud(text, "phonemize");
        }
        self.phonemizer.g2p(text, language).map(strip_language_tags)
    }

    fn diacritize(&mut self, text: &str) -> Result<String> {
        if self.hebrew_g2p_engine != "phonikud" {
            bail!("Diatrics are available only when Phonikud is selected");
        }
        self.phonikud(text, "diacritize")
    }

    fn supported_phonemes(&self) -> Vec<char> {
        self.tts.supported_phonemes()
    }

    fn synthesize_streaming(
        &mut self,
        text: &str,
        voice: Option<&str>,
        language: &str,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let (detected_language, language_code) = Self::language_for(text, language)?;
        let voice = Self::normalize_voice(voice.unwrap_or("Rotem"));
        let style = self.styles.get(voice).ok_or_else(|| {
            anyhow::anyhow!("unknown Blue voice `{voice}`; expected Rotem or Roi")
        })?;
        if detected_language == Language::Hebrew && self.hebrew_g2p_engine == "phonikud" {
            let phonemes = self.phonikud(text, "phonemize")?;
            let audio = self.tts.create(
                &phonemes,
                style,
                SynthesisOptions {
                    lang: language_code.to_owned(),
                    total_step: 8,
                    cfg_scale: 4.0,
                    speed: 0.95,
                    chunking: Some(blue_rs::ChunkingOptions {
                        enabled: true,
                        silence_seconds: 0.15,
                        max_chars: Some(200),
                    }),
                },
            )?;
            on_chunk(&audio, self.tts.sample_rate())?;
            return Ok(audio);
        }
        let sample_rate = self.tts.sample_rate();
        self.tts.synthesize_text_streaming(
            &mut self.phonemizer,
            text,
            style,
            SynthesisOptions {
                lang: language_code.to_owned(),
                total_step: 8,
                cfg_scale: 4.0,
                speed: 0.95,
                chunking: Some(blue_rs::ChunkingOptions {
                    enabled: true,
                    silence_seconds: 0.15,
                    max_chars: Some(200),
                }),
            },
            |chunk| on_chunk(chunk, sample_rate),
        )
    }

    fn synthesize_phonemes_streaming(
        &mut self,
        phonemes: &str,
        voice: Option<&str>,
        language: &str,
        on_chunk: &mut dyn FnMut(&[f32], u32) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let (_, language_code) = Self::language_for(phonemes, language)?;
        let voice = Self::normalize_voice(voice.unwrap_or("Rotem"));
        let style = self.styles.get(voice).ok_or_else(|| {
            anyhow::anyhow!("unknown Blue voice `{voice}`; expected Rotem or Roi")
        })?;
        let audio = self.tts.create(
            phonemes,
            style,
            SynthesisOptions {
                lang: language_code.to_owned(),
                total_step: 8,
                cfg_scale: 4.0,
                speed: 0.95,
                chunking: Some(blue_rs::ChunkingOptions {
                    enabled: true,
                    silence_seconds: 0.15,
                    max_chars: Some(200),
                }),
            },
        )?;
        on_chunk(&audio, self.tts.sample_rate())?;
        Ok(audio)
    }

    fn synthesize_to_file(
        &mut self,
        text: &str,
        voice: Option<&str>,
        output_path: &Path,
        language: &str,
    ) -> Result<()> {
        let (_language, language_code) = Self::language_for(text, language)?;
        let voice = Self::normalize_voice(voice.unwrap_or("Rotem"));
        let style = self.styles.get(voice).ok_or_else(|| {
            anyhow::anyhow!("unknown Blue voice `{voice}`; expected Rotem or Roi")
        })?;
        let audio = self.tts.synthesize_text(
            &mut self.phonemizer,
            text,
            style,
            SynthesisOptions {
                lang: language_code.to_owned(),
                total_step: 8,
                cfg_scale: 4.0,
                speed: 0.95,
                chunking: None,
            },
        )?;
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

fn language(name: &str, id: i32) -> RuntimeLanguage {
    RuntimeLanguage {
        name: name.to_owned(),
        id,
    }
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
    use super::{BlueRuntime, strip_language_tags};

    #[test]
    fn detects_hebrew_and_rejects_unsupported_languages() {
        assert_eq!(BlueRuntime::language_for("שלום", "auto").unwrap().1, "he");
        assert_eq!(BlueRuntime::language_for("Hello", "auto").unwrap().1, "en");
        assert_eq!(BlueRuntime::language_for("Hola", "es").unwrap().1, "es");
        assert_eq!(BlueRuntime::language_for("Hallo", "de").unwrap().1, "de");
        assert_eq!(BlueRuntime::language_for("Ciao", "it").unwrap().1, "it");
        assert!(BlueRuntime::language_for("Bonjour", "fr").is_err());
    }

    #[test]
    fn strips_internal_language_tags_from_preview_ipa() {
        assert_eq!(
            strip_language_tags("<he>ʃalˈom</he> , <en>həlˈoʊ</en>".into()),
            "ʃalˈom , həlˈoʊ"
        );
    }
}
