//! Print prepared text, phonemes and audio duration for the issue #8 corpus.
//!
//! Run it once on this branch and once on the branch point to see what the
//! Hebrew number pass changed. Set `RENIKUD_MODEL`, and set `TTS_MODELS` as
//! well to also synthesize and report durations.
//!
//! ```console
//! RENIKUD_MODEL=crates/blue-rs/renikud.onnx cargo run -p blue-rs --example issue8_numbers
//! ```

use anyhow::{Context, Result};
use blue_rs::{
    BlueTts, SynthesisOptions, VoiceStyle,
    handling::prepare_text_for_synthesis,
    phonemize::{Language, Phonemizer},
};

const CORPUS: [&str; 13] = [
    "זמן קיבוץ, גיליון 2011, 3 בספטמבר 2026",
    "4500 דונם",
    "6.4 מיליארד שקלים",
    "30,000 שקלים",
    "כ-150 דונם",
    "7,400 מיני צמחים",
    "12.3% רייטינג",
    "346 אלף צופים",
    "בשנים 2022 - 2025",
    "7 באוקטובר",
    "כ-600 בני אדם",
    "בן 90",
    "תשפ\"ז",
];

/// Lines that also get synthesized when a TTS model directory is given.
const SPOKEN: [&str; 4] = [
    "זמן קיבוץ, גיליון 2011, 3 בספטמבר 2026",
    "4500 דונם",
    "בשנים 2022 - 2025",
    "346 אלף צופים",
];

fn main() -> Result<()> {
    let renikud = std::env::var("RENIKUD_MODEL")
        .context("set RENIKUD_MODEL=/path/to/renikud.onnx")?;
    let mut phonemizer = Phonemizer::new(Some(&renikud))?;
    let models = std::env::var("TTS_MODELS").ok();
    let voice = std::env::var("VOICE").ok();

    for line in CORPUS {
        println!("input:    {line}");
        println!("prepared: {}", prepare_text_for_synthesis(line, "he"));
        println!("phonemes: {}", phonemizer.g2p(line, Language::Hebrew)?);
        println!();
    }

    let (Some(models), Some(voice)) = (models, voice) else {
        println!("no TTS_MODELS/VOICE set, skipping synthesis");
        return Ok(());
    };
    let mut tts = BlueTts::from_dir(&models)?;
    let style = VoiceStyle::from_json(&voice)?;
    let rate = tts.sample_rate() as f32;
    for line in SPOKEN {
        let audio = tts.synthesize_text(
            &mut phonemizer,
            line,
            &style,
            SynthesisOptions {
                lang: "he".to_string(),
                ..SynthesisOptions::default()
            },
        )?;
        println!(
            "audio:    {:.2}s for {line}",
            audio.len() as f32 / rate
        );
    }
    Ok(())
}
