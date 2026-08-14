use std::path::{Path, PathBuf};

use anyhow::{Result, bail};
use qwentts_rs::{
    QwenTts, QwenTtsConfig, SynthesizeRequest, ar::ArConfig, audio::wav::write_mono_i16_wav,
};

use super::{Language, Runtime, language_code_alias};

pub struct QwenRuntime {
    tts: QwenTts,
    languages: Vec<Language>,
}

unsafe impl Send for QwenRuntime {}

impl QwenRuntime {
    pub fn load(
        model_path: PathBuf,
        codec_path: PathBuf,
        max_tokens: i32,
        temperature: f32,
        top_k: i32,
    ) -> Result<Self> {
        let mut config = QwenTtsConfig::new(model_path).codec_path(codec_path);
        if max_tokens > 0 {
            config.max_tokens = max_tokens;
        }
        if temperature >= 0.0 {
            config.temperature = temperature;
        }
        if top_k > 0 {
            config.top_k = top_k;
        }
        let tts = QwenTts::load(config)?;
        let languages = ArConfig::load(tts.model())?
            .languages
            .into_iter()
            .map(|language| Language {
                name: language.name,
                id: language.id,
            })
            .collect();
        Ok(Self { tts, languages })
    }
}

impl Runtime for QwenRuntime {
    fn languages(&self) -> &[Language] {
        &self.languages
    }

    fn voices(&self) -> Option<Vec<String>> {
        None
    }

    fn synthesize_to_file(
        &mut self,
        text: &str,
        voice_or_reference: Option<&Path>,
        output_path: &Path,
        language: &str,
    ) -> Result<()> {
        if text.is_empty() {
            bail!("text is required");
        }
        let mut request = SynthesizeRequest::new(text);
        request.ref_wav_path = voice_or_reference.map(Path::to_path_buf);
        request.language_id = qwen_language_id(&self.languages, language)?;
        let audio = self.tts.synthesize(request)?;
        write_mono_i16_wav(output_path, &audio)?;
        Ok(())
    }
}

pub fn qwen_language_id(languages: &[Language], language: &str) -> Result<Option<i32>> {
    let language = language_code_alias(language);
    if language.is_empty() || language == "auto" {
        return Ok(None);
    }
    languages
        .iter()
        .find(|item| item.name == language)
        .map(|item| Some(item.id))
        .ok_or_else(|| anyhow::anyhow!("unsupported language {language:?}"))
}
