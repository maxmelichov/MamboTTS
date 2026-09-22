use anyhow::{Result, bail};
use blue_rs::{
    BlueTts, BlueTtsModelBytes, ChunkingOptions, SynthesisOptions, VoiceStyle,
    phonemize::{Language, Phonemizer},
};

static RENIKUD_MODEL: &[u8] = include_bytes!("../renikud.onnx");
static VOCAB: &[u8] = include_bytes!("../onnx_models/vocab.json");
static STYLE: &[u8] = include_bytes!("../voices/female1.json");
static TEXT_ENCODER: &[u8] = include_bytes!("../onnx_models/text_encoder.onnx");
static VECTOR_ESTIMATOR: &[u8] = include_bytes!("../onnx_models/vector_estimator.onnx");
static VOCODER: &[u8] = include_bytes!("../onnx_models/vocoder.onnx");
static DURATION_PREDICTOR: &[u8] = include_bytes!("../onnx_models/duration_predictor.onnx");

fn main() -> Result<()> {
    let args = Args::parse()?;

    let mut app = EmbeddedDemo::new(args.language)?;
    match (args.text.as_deref(), args.output.as_deref()) {
        (Some(text), Some(path)) => app.save(text, path)?,
        (Some(text), None) => app.speak(text)?,
        (None, _) => app.stdin_loop()?,
    }

    Ok(())
}

struct Args {
    language: Language,
    text: Option<String>,
    output: Option<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut language = Language::Hebrew;
        let mut positional = Vec::new();
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-l" | "--language" => {
                    let Some(code) = args.next() else {
                        bail!("{arg} requires a language code: he, en, es, de, or it");
                    };
                    language = Language::try_from(code.as_str())?;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ if arg.starts_with('-') => bail!("unknown option `{arg}`"),
                _ => positional.push(arg),
            }
        }

        Ok(Self {
            language,
            text: positional.first().cloned(),
            output: positional.get(1).cloned(),
        })
    }
}

fn print_usage() {
    eprintln!(
        "Usage: cargo run --release --example embedded --features embedded-example -- [--language he|en|es|de|it] <text> [output.wav]"
    );
}

struct EmbeddedDemo {
    phonemizer: Phonemizer,
    tts: BlueTts,
    style: VoiceStyle,
    language: Language,
}

impl EmbeddedDemo {
    fn new(language: Language) -> Result<Self> {
        Ok(Self {
            phonemizer: Phonemizer::from_renikud_bytes(RENIKUD_MODEL, language)?,
            tts: BlueTts::from_model_bytes(BlueTtsModelBytes {
                duration_predictor: DURATION_PREDICTOR,
                text_encoder: TEXT_ENCODER,
                vector_estimator: VECTOR_ESTIMATOR,
                vocoder: VOCODER,
                vocab: VOCAB,
            })?,
            style: VoiceStyle::from_json_bytes(STYLE)?,
            language,
        })
    }

    fn create(&mut self, text: &str) -> Result<Vec<f32>> {
        let phonemes = self.phonemizer.phonemize(text)?;
        eprintln!("Phonemes: {phonemes}");
        self.tts.create(
            &phonemes,
            &self.style,
            SynthesisOptions {
                lang: self.language.code().to_string(),
                total_step: 5,
                cfg_scale: 4.0,
                speed: 1.0,
                chunking: Some(ChunkingOptions {
                    enabled: true,
                    silence_seconds: 0.2,
                    max_chars: None,
                }),
            },
        )
    }

    fn speak(&mut self, text: &str) -> Result<()> {
        let samples = self.create(text)?;
        let (_stream, handle) = rodio::OutputStream::try_default()?;
        let sink = rodio::Sink::try_new(&handle)?;
        sink.append(rodio::buffer::SamplesBuffer::new(
            1,
            self.tts.sample_rate(),
            samples,
        ));
        sink.sleep_until_end();
        Ok(())
    }

    fn save(&mut self, text: &str, path: &str) -> Result<()> {
        let samples = self.create(text)?;
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.tts.sample_rate(),
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec)?;
        for sample in samples {
            writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
        eprintln!("Saved to {path}");
        Ok(())
    }

    fn stdin_loop(&mut self) -> Result<()> {
        use std::io::BufRead;

        let stdin = std::io::stdin();
        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Err(e) = self.speak(line) {
                eprintln!("Error: {e}");
            }
        }
        Ok(())
    }
}
