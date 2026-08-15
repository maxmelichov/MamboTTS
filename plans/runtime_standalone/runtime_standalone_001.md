# Runtime Standalone 001

## Goal

Create a standalone `mambotts/runtime` project for Qwen3-TTS inference, separate from
`llama.cpp/tools`, with no top-level `mambotts/CMakeLists.txt`. The runtime builds
directly from `mambotts/runtime` and exposes a simple CLI plus a C API.

## Repositories And Sources Used

- Local workspace repo:
  - `/home/yakov/Documents/audio/mambotts`
- Qwen Python/reference implementation:
  - local folder `Qwen3-TTS/`
  - used for understanding prompt formatting, tokenizer behavior, model conversion,
    and validating native speaker embedding extraction.
- llama.cpp:
  - local source folder `llama.cpp/`
  - used as the original development location for `tools/qwen3-tts-v1`,
    `tools/qwen3-tts-codec`, and `tools/qwen3-tts-v2`.
  - standalone runtime fetches upstream `ggml-org/llama.cpp` with CMake
    `FetchContent`.
  - latest release tag was obtained with:
    ```bash
    gh api repos/ggml-org/llama.cpp/releases/latest --jq .tag_name
    ```
  - tag used: `b8941`
- dr_wav:
  - upstream single-header WAV library:
    `https://raw.githubusercontent.com/mackron/dr_libs/master/dr_wav.h`
  - fetched with CMake `FetchContent`.
- kissfft:
  - upstream FFT library:
    `https://github.com/mborgerding/kissfft`
  - fetched with CMake `FetchContent`.
  - pinned commit: `6e9e673e420c4bf47d4a60c57c578f93e4ec192f`
  - used for the speaker encoder STFT.
- libsoxr:
  - upstream SoX Resampler library:
    `https://github.com/chirlu/soxr`
  - fetched with CMake `FetchContent`.
  - pinned commit: `945b592b70470e29f917f4de89b4281fbbd540c0`
  - used for reference WAV resampling to 24 kHz.

## Runtime Layout Created

```text
mambotts/
  .gitignore
  plans/
    runtime_standalone/
      runtime_standalone_001.md
  runtime/
    CMakeLists.txt
    README.md
    scripts/
      convert_model_to_gguf.py
      convert_codec_to_gguf.py
    src/
      main.cpp
      qwen3_tts.cpp
      qwen3_tts.h
      ar/
      decoder/
      encoder/
      speaker/
      text/
```

No CMake files were added at `mambotts/` top level.

## How Runtime Was Assembled

1. Started from the latest working in-tree v2 implementation:
   - `llama.cpp/tools/qwen3-tts-v2`

2. Copied v2 runtime sources into:
   - `mambotts/runtime/src`

3. Renamed the public API from v2-specific names to stable names:
   - `qwen3_tts_v2.h` -> `qwen3_tts.h`
   - `qwen3_tts_v2.cpp` -> `qwen3_tts.cpp`
   - API symbols now use `qwen3_tts_*`.

4. Kept runtime source split:
   - `src/ar/`: autoregressive Qwen3-TTS codebook generator and code predictor.
   - `src/decoder/`: codec decoder and WAV output.
   - `src/encoder/`: native Mimi/Qwen codec encoder code retained for later use.
   - `src/text/`: native Qwen text tokenizer added later.

5. Copied converter scripts into:
   - `mambotts/runtime/scripts/convert_model_to_gguf.py`
   - `mambotts/runtime/scripts/convert_codec_to_gguf.py`

6. Removed `prepare_inputs.py` after porting both runtime input-preparation
   paths to C++:
   - Text tokenization is native.
   - Reference speaker embedding extraction is native.

## CMake Setup

`mambotts/runtime/CMakeLists.txt` is the standalone build entry point.

It uses `FetchContent` for llama.cpp:

```cmake
FetchContent_Declare(
    llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG b8941
)
```

It does not build all llama.cpp tools. It populates the dependency and adds only
the `ggml` subdirectory:

```cmake
FetchContent_Populate(llama_cpp)
add_subdirectory(${llama_cpp_SOURCE_DIR}/ggml ${llama_cpp_BINARY_DIR}/ggml)
```

It also fetches `dr_wav.h`:

```cmake
FetchContent_Declare(
    dr_libs
    URL https://raw.githubusercontent.com/mackron/dr_libs/master/dr_wav.h
    DOWNLOAD_NO_EXTRACT TRUE
)
```

It fetches `kissfft` for FFT and `libsoxr` for resampling:

```cmake
FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG 6e9e673e420c4bf47d4a60c57c578f93e4ec192f
)

FetchContent_Declare(
    soxr
    GIT_REPOSITORY https://github.com/chirlu/soxr.git
    GIT_TAG 945b592b70470e29f917f4de89b4281fbbd540c0
)
```

The built executable is:

```text
mambotts-runtime
```

## C API Added

Public header:

```text
mambotts/runtime/src/qwen3_tts.h
```

Public functions:

```c
qwen3_tts_params qwen3_tts_default_params(void);
qwen3_tts_context * qwen3_tts_init(const qwen3_tts_params * params);
void qwen3_tts_free(qwen3_tts_context * ctx);
const char * qwen3_tts_get_error(const qwen3_tts_context * ctx);
int qwen3_tts_synthesize_to_file(
    qwen3_tts_context * ctx,
    const char * text,
    const char * ref_wav_path,
    const char * output_wav_path
);
```

Generation progress callback:

```c
typedef int (*qwen3_tts_generate_progress_callback)(
    void * user_data,
    int32_t current_frame,
    int32_t max_frames
);
```

Return convention:

- `1`: continue generation.
- `0`: cancel generation.

## CLI

The CLI is a thin wrapper over the C API:

```console
mambotts/runtime/build/mambotts-runtime \
  --model models/qwen3-tts-0.6b-f16.gguf \
  --codec models/qwen3-tts-tokenizer-f16.gguf \
  --text "hello" \
  --ref tmp/refs/female1.wav \
  --output tmp/mambotts_runtime.wav
```

`--ref` is optional. Without `--ref`, runtime does not invoke Python.

## Text Tokenizer Work

Native tokenizer added:

```text
mambotts/runtime/src/text/qwen_tokenizer.h
mambotts/runtime/src/text/qwen_tokenizer.cpp
```

It reads tokenizer metadata embedded in the model GGUF:

- `tokenizer.ggml.tokens`
- `tokenizer.ggml.merges`

It recreates Qwen TTS prompt formatting:

```text
<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
```

It implements byte-level BPE for the Qwen tokenizer and maps special tokens:

- `<|im_start|>` -> `151644`
- `<|im_end|>` -> `151645`

Verified native token IDs matched Python for:

- `hello`
- `Hello from Qwen three TTS.`
- `Standalone mambotts runtime test.`

## Python Dependency Status

Current runtime dependency split:

```text
With or without --ref:
  C++ text tokenizer
  C++ speaker embedding extraction when --ref is provided
  C++ AR/codebook generation
  C++ codec decode
  C++ WAV read/write
  no Python invocation
```

Python is now only used for converter scripts and validation scripts, not runtime
inference.

## Speaker Encoder Work

Native speaker/x-vector extraction added:

```text
mambotts/runtime/src/speaker/speaker_encoder.h
mambotts/runtime/src/speaker/speaker_encoder.cpp
```

It loads the `spk_enc.*` tensors embedded in the model GGUF and implements the
Qwen3-TTS ECAPA-TDNN speaker encoder. The audio frontend matches the Python
reference:

- 24 kHz mono input
- 24 kHz resampling through `libsoxr`
- reflected STFT padding
- 1024-point FFT through `kissfft`
- 256-sample hop
- periodic Hann window
- librosa-compatible Slaney mel filterbank

Validation against Python `extract_speaker_embedding` on the original 44.1 kHz
`female1.wav` reference after replacing linear resampling with `libsoxr`:

- cosine similarity: `1.0`
- mean absolute error: `1.8726e-7`
- max absolute error: `7.6294e-6`

## WAV I/O

Manual WAV parsing/writing was replaced with `dr_wav`.

Runtime wrapper names stayed stable:

```cpp
read_wav_mono(...)
write_wav_mono16(...)
resample_linear(...)
```

## Build Verification

Standalone runtime build:

```console
cmake -S mambotts/runtime -B mambotts/runtime/build
cmake --build mambotts/runtime/build -j 8
```

This completed successfully.

## Inference Verification

Standalone inference with reference voice:

```console
mambotts/runtime/build/mambotts-runtime \
  --model models/qwen3-tts-0.6b-f16.gguf \
  --codec models/qwen3-tts-tokenizer-f16.gguf \
  --text 'Standalone mambotts runtime test.' \
  --ref tmp/refs/female1.wav \
  --output tmp/mambotts_runtime_standalone.wav \
  --max-tokens 40 \
  --temperature 0 \
  --top-k 1
```

Generated:

```text
/home/yakov/Documents/audio/mambotts/tmp/mambotts_runtime_standalone.wav
```

Validation:

- sample rate: `24000`
- duration: `2.46s`
- finite audio: `true`
- RMS: `0.0890`

Native-tokenizer no-ref inference:

```console
mambotts/runtime/build/mambotts-runtime \
  --model models/qwen3-tts-0.6b-f16.gguf \
  --codec models/qwen3-tts-tokenizer-f16.gguf \
  --text 'Native tokenizer test without python.' \
  --output tmp/mambotts_native_tokenizer_no_ref.wav \
  --max-tokens 32 \
  --temperature 0 \
  --top-k 1
```

No Python output appeared for this no-ref path, confirming text preparation is native.

Native speaker-reference inference after removing `prepare_inputs.py`:

```console
mambotts/runtime/build/mambotts-runtime \
  --model models/qwen3-tts-0.6b-f16.gguf \
  --codec models/qwen3-tts-tokenizer-f16.gguf \
  --text 'Native speaker encoder is now running inside the mambotts C plus plus runtime.' \
  --ref tmp/refs/female1.wav \
  --output tmp/mambotts_native_speaker_runtime.wav \
  --max-tokens 48 \
  --temperature 0 \
  --top-k 1
```

Generated:

```text
/home/yakov/Documents/audio/mambotts/tmp/mambotts_native_speaker_runtime.wav
```

Validation:

- sample rate: `24000`
- channels: `1`
- duration: `3.816875s`
- finite audio: `true`
- peak: `0.3723`
- RMS: `0.0712`

Native speaker-reference inference after replacing custom FFT/resampling with
`kissfft` and `libsoxr`:

```console
mambotts/runtime/build/mambotts-runtime \
  --model models/qwen3-tts-0.6b-f16.gguf \
  --codec models/qwen3-tts-tokenizer-f16.gguf \
  --text 'The mambotts runtime now uses library based resampling and Fourier transforms.' \
  --ref tmp/refs/female1.wav \
  --output tmp/mambotts_soxr_kissfft_runtime.wav \
  --max-tokens 48 \
  --temperature 0 \
  --top-k 1
```

Generated:

```text
/home/yakov/Documents/audio/mambotts/tmp/mambotts_soxr_kissfft_runtime.wav
```

Validation:

- sample rate: `24000`
- channels: `1`
- duration: `3.816875s`
- finite audio: `true`
- peak: `0.4725`
- RMS: `0.0814`

## Important Notes

- `models/qwen3-tts-0.6b-f16.gguf` must include tokenizer metadata.
- Current converter scripts already embed tokenizer metadata in the model GGUF.
- The codec GGUF is separate and passed as `--codec`.
- `mambotts/.gitignore` ignores runtime build outputs, fetched deps, Python caches,
  and generated audio/code/token artifacts.
