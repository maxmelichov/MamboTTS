//! Safe Rust bindings for the qwentts.cpp C ABI.
//!
//! qwentts.cpp runs Qwen3-TTS on GGML: a *talker* GGUF turns text into
//! 12.5 Hz codec frames and a *codec* GGUF decodes those frames into
//! 24 kHz mono audio. Both files are required.
//!
//! The C ABI reports failure as a negative `qt_status` plus a thread
//! local message, which every entry point here converts into an
//! [`anyhow::Error`].

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case)]

use std::ffi::{CStr, CString};
use std::os::raw::{c_int, c_void};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};

mod sys {
    #![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Output rate of the 12 Hz codec, fixed by the model.
pub const SAMPLE_RATE: u32 = 24_000;

/// Reads the thread local diagnostic left behind by a failing entry point.
fn last_error() -> String {
    // SAFETY: the pointer is owned by the library and stays valid until
    // the next failing call on this thread; it is copied out immediately.
    unsafe {
        let message = sys::qt_last_error();
        if message.is_null() {
            return "unknown qwentts error".into();
        }
        CStr::from_ptr(message).to_string_lossy().into_owned()
    }
}

fn check(status: sys::qt_status, action: &str) -> Result<()> {
    if (status as i32) < 0 {
        bail!("{action} failed: {}", last_error());
    }
    Ok(())
}

fn c_string(value: &str, field: &str) -> Result<CString> {
    CString::new(value).with_context(|| format!("{field} contains an interior NUL byte"))
}

fn path_string(path: &Path, field: &str) -> Result<CString> {
    let text = path
        .to_str()
        .ok_or_else(|| anyhow!("{field} is not valid UTF-8: {}", path.display()))?;
    c_string(text, field)
}

/// Where the two GGUF files live, plus the backend knobs worth exposing.
pub struct QwenTtsConfig {
    pub talker_path: PathBuf,
    pub codec_path: PathBuf,
    /// Fused flash attention, used only when a GPU backend is present.
    pub use_fa: bool,
    /// Guards FP16 matmul accumulation on pre-Ampere CUDA targets.
    pub clamp_fp16: bool,
}

impl QwenTtsConfig {
    pub fn new(talker_path: impl Into<PathBuf>, codec_path: impl Into<PathBuf>) -> Self {
        Self {
            talker_path: talker_path.into(),
            codec_path: codec_path.into(),
            use_fa: true,
            clamp_fp16: false,
        }
    }
}

/// One synthesis call.
#[derive(Default)]
pub struct SynthesizeRequest {
    /// For the Hebrew checkpoint this is stressed IPA, not Hebrew script.
    pub text: String,
    /// Upstream language name; empty selects the model's own detection.
    pub language: String,
    /// Mono 24 kHz PCM of the voice to clone.
    pub reference_audio: Option<Vec<f32>>,
    /// Transcript of `reference_audio`, which switches cloning to ICL mode.
    pub reference_text: Option<String>,
    pub seed: i64,
    pub max_new_tokens: i32,
    pub temperature: f32,
    pub top_k: i32,
}

impl SynthesizeRequest {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            // Zero means "leave the upstream default in place"; the
            // defaults are filled in by qt_tts_default_params.
            seed: -1,
            ..Default::default()
        }
    }
}

/// Reusable voice-clone conditioning extracted from a reference recording.
pub struct VoiceRef {
    inner: sys::qt_voice_ref,
}

impl VoiceRef {
    pub fn speaker_dim(&self) -> usize {
        self.inner.ref_spk_dim.max(0) as usize
    }
}

impl Drop for VoiceRef {
    fn drop(&mut self) {
        // SAFETY: the struct was filled by qt_extract_voice_ref and owns
        // its buffers; freeing also resets it, so a double drop is safe.
        unsafe { sys::qt_voice_ref_free(&mut self.inner) };
    }
}

/// A loaded talker + codec pair.
pub struct QwenTts {
    ctx: *mut sys::qt_context,
}

// SAFETY: the handle owns its GGML backends and the C ABI documents
// qt_synthesize as thread safe. Rust's own borrow rules keep calls
// serialised because every synthesis takes `&mut self`.
unsafe impl Send for QwenTts {}

impl QwenTts {
    pub fn load(config: QwenTtsConfig) -> Result<Self> {
        if !config.talker_path.is_file() {
            bail!("talker GGUF is missing at {}", config.talker_path.display());
        }
        if !config.codec_path.is_file() {
            bail!("codec GGUF is missing at {}", config.codec_path.display());
        }
        let talker = path_string(&config.talker_path, "talker_path")?;
        let codec = path_string(&config.codec_path, "codec_path")?;

        // SAFETY: default params fill in abi_version, which the library
        // reads first to route the struct layout. The CStrings outlive
        // the qt_init call that copies from them.
        let ctx = unsafe {
            let mut params: sys::qt_init_params = std::mem::zeroed();
            sys::qt_init_default_params(&mut params);
            params.talker_path = talker.as_ptr();
            params.codec_path = codec.as_ptr();
            params.use_fa = config.use_fa;
            params.clamp_fp16 = config.clamp_fp16;
            sys::qt_init(&params)
        };
        if ctx.is_null() {
            bail!("failed to load Qwen TTS: {}", last_error());
        }
        Ok(Self { ctx })
    }

    /// Named speakers, empty for the Base checkpoints used here.
    pub fn speakers(&self) -> Vec<String> {
        // SAFETY: the handle is live and the returned pointers are owned
        // by the library, valid for the duration of the copy.
        unsafe {
            let count = sys::qt_n_speakers(self.ctx);
            (0..count)
                .filter_map(|index| {
                    let name = sys::qt_speaker_name(self.ctx, index);
                    (!name.is_null())
                        .then(|| CStr::from_ptr(name).to_string_lossy().into_owned())
                })
                .collect()
        }
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Extracts speaker conditioning from mono 24 kHz reference audio.
    pub fn extract_voice_ref(&mut self, reference_audio_24k: &[f32]) -> Result<VoiceRef> {
        if reference_audio_24k.is_empty() {
            bail!("reference audio is empty");
        }
        // SAFETY: the input slice is only read for the duration of the
        // call; `out` is zeroed first as the ABI requires.
        let inner = unsafe {
            let mut out: sys::qt_voice_ref = std::mem::zeroed();
            let status = sys::qt_extract_voice_ref(
                self.ctx,
                reference_audio_24k.as_ptr(),
                reference_audio_24k.len() as c_int,
                &mut out,
            );
            check(status, "extract voice reference")?;
            out
        };
        Ok(VoiceRef { inner })
    }

    /// Synthesises into a single buffer.
    pub fn synthesize(&mut self, request: SynthesizeRequest) -> Result<Vec<f32>> {
        self.run(request, None::<&mut dyn FnMut(&[f32]) -> Result<()>>)
    }

    /// Synthesises, handing each decoded chunk to `on_chunk` as it lands.
    ///
    /// The chunk width ramps from a single 12.5 Hz frame up to eight, so
    /// the first callback fires as early as the model allows. The full
    /// audio is also accumulated and returned.
    pub fn synthesize_streaming(
        &mut self,
        request: SynthesizeRequest,
        on_chunk: &mut dyn FnMut(&[f32]) -> Result<()>,
    ) -> Result<Vec<f32>> {
        self.run(request, Some(on_chunk))
    }

    fn run(
        &mut self,
        request: SynthesizeRequest,
        on_chunk: Option<&mut dyn FnMut(&[f32]) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        if request.text.trim().is_empty() {
            bail!("text is required");
        }
        let text = c_string(&request.text, "text")?;
        let language = (!request.language.is_empty())
            .then(|| c_string(&request.language, "language"))
            .transpose()?;
        let reference_text = request
            .reference_text
            .as_deref()
            .filter(|value| !value.is_empty())
            .map(|value| c_string(value, "reference_text"))
            .transpose()?;

        let mut state = ChunkState {
            sink: on_chunk,
            collected: Vec::new(),
            error: None,
        };
        let streaming = state.sink.is_some();

        // SAFETY: every pointer below outlives the qt_synthesize call.
        // The callback trampoline receives `state` as its user data and
        // is only invoked from inside that call because max_batch is 1.
        let mut audio: sys::qt_audio = unsafe { std::mem::zeroed() };
        let status = unsafe {
            let mut params: sys::qt_tts_params = std::mem::zeroed();
            sys::qt_tts_default_params(&mut params);
            params.text = text.as_ptr();
            if let Some(language) = language.as_ref() {
                params.lang = language.as_ptr();
            }
            if let Some(reference) = request.reference_audio.as_ref() {
                params.ref_audio_24k = reference.as_ptr();
                params.ref_n_samples = reference.len() as c_int;
            }
            if let Some(reference_text) = reference_text.as_ref() {
                params.ref_text = reference_text.as_ptr();
            }
            params.seed = request.seed;
            if request.max_new_tokens > 0 {
                params.max_new_tokens = request.max_new_tokens;
            }
            if request.temperature > 0.0 {
                params.temperature = request.temperature;
            }
            if request.top_k > 0 {
                params.top_k = request.top_k;
            }
            if streaming {
                params.on_chunk = Some(chunk_trampoline);
                params.on_chunk_user_data = &mut state as *mut ChunkState as *mut c_void;
            }
            sys::qt_synthesize(self.ctx, &params, &mut audio)
        };

        // A callback error is the real cause of the cancellation the
        // library reports, so it wins over the generic status message.
        if let Some(error) = state.error.take() {
            // SAFETY: qt_synthesize either filled the struct or left it
            // zeroed; freeing a zeroed struct is documented as safe.
            unsafe { sys::qt_audio_free(&mut audio) };
            return Err(error);
        }
        if let Err(err) = check(status, "synthesize") {
            unsafe { sys::qt_audio_free(&mut audio) };
            return Err(err);
        }

        if streaming {
            // Streaming leaves `out` empty by design.
            unsafe { sys::qt_audio_free(&mut audio) };
            return Ok(std::mem::take(&mut state.collected));
        }

        // SAFETY: on success the samples pointer is a malloc'd buffer of
        // n_samples floats owned by the struct; it is copied out and then
        // released.
        let samples = unsafe {
            if audio.samples.is_null() || audio.n_samples <= 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(audio.samples, audio.n_samples as usize).to_vec()
            }
        };
        unsafe { sys::qt_audio_free(&mut audio) };
        Ok(samples)
    }
}

impl Drop for QwenTts {
    fn drop(&mut self) {
        // SAFETY: qt_free is documented as safe on NULL and the handle is
        // not used afterwards.
        unsafe { sys::qt_free(self.ctx) };
    }
}

struct ChunkState<'a> {
    sink: Option<&'a mut dyn FnMut(&[f32]) -> Result<()>>,
    collected: Vec<f32>,
    error: Option<anyhow::Error>,
}

/// Bridges the C callback back into the Rust closure.
///
/// Returning `false` aborts the synthesis, which the library surfaces as
/// a cancellation; the underlying error is carried out through the state.
unsafe extern "C" fn chunk_trampoline(
    samples: *const f32,
    n_samples: c_int,
    user_data: *mut c_void,
) -> bool {
    if user_data.is_null() {
        return false;
    }
    // SAFETY: user_data is the &mut ChunkState handed to qt_synthesize,
    // which outlives the call, and the library invokes this callback
    // serially from inside that call.
    let state = unsafe { &mut *(user_data as *mut ChunkState) };
    let chunk = if samples.is_null() || n_samples <= 0 {
        &[][..]
    } else {
        // SAFETY: valid only for the duration of the call, which is
        // exactly the lifetime of this borrow.
        unsafe { std::slice::from_raw_parts(samples, n_samples as usize) }
    };
    state.collected.extend_from_slice(chunk);
    let Some(sink) = state.sink.as_mut() else {
        return true;
    };
    match sink(chunk) {
        Ok(()) => true,
        Err(err) => {
            state.error = Some(err);
            false
        }
    }
}
