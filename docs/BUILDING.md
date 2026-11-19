# Building and reinstalling

MamboTTS packages a Tauri desktop application with a Rust inference sidecar
(`mambotts-server`). Use `pnpm` for the desktop shell and Cargo for the Rust
workspace.

The desktop app builds for macOS on Apple Silicon, Linux on x86_64, and Windows
on x86_64. Every platform follows the same two steps: `scripts/pre_build.py`
compiles the sidecar and stages the matching ONNX Runtime shared library into
`mambotts-desktop/src-tauri/binaries/`, and then `tauri build` bundles the shell
around it. Tauri picks up `tauri.macos.conf.json`, `tauri.linux.conf.json`, or
`tauri.windows.conf.json` automatically, so no extra flag is needed to get the
right ONNX Runtime library into the installer.

## What is verified

- **Linux x86_64**: built end to end in the container from `scripts/Dockerfile.linux-build`, producing `.deb`, `.rpm`, and `.AppImage` bundles. The bundles have not been launched on a Linux desktop from this repository.
- **macOS Apple Silicon**: the shipping platform. Built here into a `.dmg` and a `.app`, with the sidecar and ONNX Runtime staged inside the bundle. The bundle was not launched as part of that check.
- **Windows x86_64**: built, installed, and run on Windows 10. The NSIS installer produced by `tauri build --bundles nsis` was installed, the app launched, the sidecar started under it, and all five languages were synthesized to WAV through the running app. Killing the app from Task Manager takes the sidecar with it.

## Shared prerequisites

- Rust stable, installed through [rustup](https://rustup.rs)
- Node.js 22 and pnpm 10
- Python with [uv](https://docs.astral.sh/uv/), used by `scripts/pre_build.py`
- A C toolchain, because a few dependencies (the LAME MP3 encoder among them) compile C sources

`scripts/pre_build.py` downloads ONNX Runtime 1.23.2 into
`crates/blue-rs/.ort/` on first use and sets `ORT_STRATEGY`,
`ORT_LIB_LOCATION`, and `ORT_PREFER_DYNAMIC_LINK` for the Cargo build it runs.
If you invoke Cargo directly against `mambotts-server`, set those three
yourself, because the `ort` crate otherwise tries to download a runtime of its
own.

## Install dependencies

```console
pnpm install
cd mambotts-desktop
pnpm install
```

## macOS on Apple Silicon

Prerequisites beyond the shared list: the Xcode command line tools
(`xcode-select --install`), which supply clang and the linker.

```console
uv run scripts/pre_build.py
cd mambotts-desktop
pnpm tauri build
```

Bundles land in:

```text
target/release/bundle/dmg/
target/release/bundle/macos/
```

Open the generated `.dmg`, drag MamboTTS into Applications, and replace the
existing copy. If macOS says the app is still running, quit it first.

A build produced this way is ad-hoc signed, not Developer ID signed (see
[docs/code-signing/macos.md](code-signing/macos.md) for the alternative), so
a copy dragged out of the DMG keeps the quarantine attribute and macOS
refuses to open it with no useful error. Clear it and, if that alone is not
enough, re-seal the ad-hoc signature over the bundle:

```console
xattr -dr com.apple.quarantine "target/release/bundle/macos/MamboTTS.app"
codesign --force --deep --sign - "target/release/bundle/macos/MamboTTS.app"
```

This is the same fix documented in the [README](../README.md#install) for
anyone who installs the DMG manually instead of through `install.sh`.

For a faster local reinstallable debug build:

```console
cd mambotts-desktop
pnpm tauri build --debug
```

## Linux on x86_64

### System packages

On Ubuntu 22.04 or Debian, these are what the build actually needs. The list
matches `scripts/Dockerfile.linux-build`, which is the environment the Linux
bundles are verified in.

```console
sudo apt-get update
sudo apt-get install -y \
  build-essential git curl wget file patchelf ca-certificates \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
  libssl-dev libgtk-3-dev libxdo-dev libasound2-dev \
  pkg-config xdg-utils
```

`patchelf` is not optional. `scripts/pre_build.py` uses it to set an rpath on
the sidecar so the packaged ONNX Runtime is found at runtime, and it warns and
carries on if the tool is missing, which produces a bundle that fails to load
the model.

### Build on Linux

```console
export ORT_LIB_DIR="$PWD/crates/blue-rs/.ort/onnxruntime-linux-x64-1.23.2/lib"
export LD_LIBRARY_PATH="$ORT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uv run scripts/pre_build.py --target x86_64-unknown-linux-gnu
pnpm --dir mambotts-desktop install
pnpm --dir mambotts-desktop exec tauri build --target x86_64-unknown-linux-gnu
```

`LD_LIBRARY_PATH` matters for the AppImage. Tauri hands the AppDir to
linuxdeploy, which walks the sidecar's shared library dependencies through the
normal loader search path and copies each one into the bundle. ONNX Runtime is
not installed system wide, so without that variable linuxdeploy stops with
`Could not find dependency: libonnxruntime.so.1` and no AppImage is produced.
The directory only exists after `scripts/pre_build.py` has downloaded ONNX
Runtime once, which the first line of the block above arranges for by pointing
at where the script puts it.

Building inside a container usually also needs `APPIMAGE_EXTRACT_AND_RUN=1`,
because linuxdeploy and its plugins are AppImages themselves and there is
normally no FUSE available to mount them.

Bundles land in:

```text
target/x86_64-unknown-linux-gnu/release/bundle/deb/
target/x86_64-unknown-linux-gnu/release/bundle/rpm/
target/x86_64-unknown-linux-gnu/release/bundle/appimage/
```

The Debian package installs the shell at `/usr/bin/mambotts`, the sidecar next
to it at `/usr/bin/mambotts-server`, and ONNX Runtime under
`/usr/lib/MamboTTS/binaries/`. The desktop shell adds that resource directory to
`LD_LIBRARY_PATH` before it spawns the sidecar.

### Build Linux bundles from macOS

You do not need a Linux machine. This runs the same steps inside an
`ubuntu:22.04` container under `linux/amd64` emulation, which is slower than a
native build but needs nothing installed beyond Docker:

```console
scripts/build_linux_docker.sh
```

The script builds a cached `mambotts-linux-builder:22.04` image the first time
it runs, then prints the bundle paths when it finishes.

If you want to be sure the bundle carries a sidecar built from your current
tree rather than a leftover artifact, delete the staged one first:

```console
rm -f mambotts-desktop/src-tauri/binaries/mambotts-server-x86_64-unknown-linux-gnu
```

## Windows on x86_64

### Prerequisites

Beyond the shared list, a Windows machine needs:

- Visual Studio Build Tools with the "Desktop development with C++" workload, which supplies the MSVC linker and the Windows SDK
- The WebView2 runtime, which is already present on Windows 11 and current Windows 10

### Build on Windows

Run this from a Developer PowerShell or any shell where the MSVC toolchain is on
`PATH`:

```console
uv run scripts/pre_build.py --target x86_64-pc-windows-msvc
pnpm --dir mambotts-desktop install
pnpm --dir mambotts-desktop exec tauri build --target x86_64-pc-windows-msvc
```

Installers land in:

```text
target\x86_64-pc-windows-msvc\release\bundle\msi\
target\x86_64-pc-windows-msvc\release\bundle\nsis\
```

`onnxruntime.dll` and `onnxruntime_providers_shared.dll` are bundled as
resources directly beside the installed executables rather than into a
subfolder, and that placement is load-bearing. Windows ships its own much
older `onnxruntime.dll` in `System32` for Windows ML, and the executable's
own directory is the only step of the DLL search order that comes before the
system directories. A copy anywhere else, including one reachable through
`PATH`, loses to the System32 file, and the sidecar then dies inside the
loader with `0xC000007B` (`STATUS_INVALID_IMAGE_FORMAT`) before it can run a
line of its own code or write a word to stderr.

On Windows the flow runs on ONNX Runtime on the CPU; the GPU engine (Metal on
macOS, Vulkan on Linux) is not opened there. It compiles its kernels through
whatever graphics driver the machine has, and on Windows that failed in ways
the app could not recover from: an AMD Radeon iGPU computes them wrongly on
Vulkan, wgpu's FXC compiler for DX12 takes many minutes, and a driver that
crashes or hangs takes the speech server down with it. To try the GPU engine
on Windows anyway, start the server with `MAMBOTTS_VE_BACKEND=gpu`; DX12 needs
`dxcompiler.dll` and `dxil.dll` from the DirectX Shader Compiler (1.8.2502 or
newer) beside `mambotts-server.exe`, and otherwise only Vulkan is tried.
`MAMBOTTS_VE_BACKEND=ort` keeps any platform on ONNX Runtime.

ONNX Runtime's CPU path has two speed settings:

- Threads: `min(8, usable CPUs)` (the affinity mask and, on Linux, the cgroup
  quota). `MAMBOTTS_ORT_THREADS=<n>` overrides it. More threads than CPUs is
  much slower: 8 threads on 2 usable x86 CPUs ran 3x slower than 2.
- The vector estimator graph: off macOS, ONNX Runtime runs
  `vector_estimator.int8.onnx` (dynamic int8 MatMuls, BlueV3-onnx f0db559) when
  the bundle has it, and `vector_estimator.onnx` otherwise. That is about 1.7x
  faster on x86 and 2.5x on ARM Linux, with the same ASR error and speaker
  similarity. The file is optional: the desktop app fetches it in the
  background off macOS (`optional_files` in the registry), and `install.ps1`
  fetches it after the bundle. macOS and the GPU engine keep float32.
  `MAMBOTTS_PRECISION=fp32|int8|auto` overrides the choice.

### Cross-compile the Windows sidecar from macOS or Linux

The installer has to be produced on Windows, but the sidecar can be
cross-compiled with [cargo-xwin](https://github.com/rust-cross/cargo-xwin),
which downloads the Windows SDK import libraries and links with `lld-link`.
This is useful for catching Windows compile errors without leaving your
machine.

```console
cargo install cargo-xwin
rustup target add x86_64-pc-windows-msvc
uv run scripts/pre_build.py --target x86_64-pc-windows-msvc
```

`scripts/pre_build.py` switches to `cargo xwin build` on its own whenever the
target is Windows and the host is not, so the command is the same one you would
run on Windows. To drive Cargo directly instead:

```console
export ORT_DIR="$PWD/crates/blue-rs/.ort/onnxruntime-win-x64-1.23.2/lib"
export ORT_STRATEGY=system
export ORT_PREFER_DYNAMIC_LINK=1
export ORT_LIB_LOCATION="$ORT_DIR"
export RUSTFLAGS="-L native=$ORT_DIR"
cargo xwin build -p mambotts-server --release --target x86_64-pc-windows-msvc
```

## Development

Run the desktop application with hot reload:

```console
cd mambotts-desktop
pnpm tauri dev
```

The pre-build hook compiles the matching `mambotts-server` sidecar and stages
ONNX Runtime before Tauri starts.

## Validate

```console
cargo test --workspace --exclude mambotts-py
cargo build -p mambotts-server --release --bin mambotts-server
cd mambotts-desktop
pnpm build
```

The tests that need real model files are `#[ignore]`d. Run them with the
downloaded bundle:

```console
MAMBOTTS_BLUE_MODEL_DIR=/path/to/models/bluev3-3.0.1 \
MAMBOTTS_RENIKUD_PATH=/path/to/models/bluev3-3.0.1/renikud-plus.onnx \
  cargo test -p blue-rs -p renikud-plus-rs -- --ignored
```

The ONNX Runtime download that `ort` does on a plain `cargo build` can be
refused by its CDN. Point the build at the copy `scripts/pre_build.py` keeps in
`crates/blue-rs/.ort` instead: `ORT_STRATEGY=system`, `ORT_LIB_LOCATION=<that
copy's lib directory>` and `ORT_PREFER_DYNAMIC_LINK=1`, with the same directory
on `DYLD_LIBRARY_PATH` (macOS) or `LD_LIBRARY_PATH` (Linux) to run the tests.

To hear the engine without the app, speak a line straight from a bundle:

```console
cargo run -p blue-rs --example speak -- /path/to/models/bluev3-3.0.1 noa en "Hello there." out.wav
```

### Validate on Windows

Cargo does not set the ONNX Runtime variables for you here, so set them the
way `scripts/pre_build.py` does, from Git Bash:

```console
export ORT_DIR="$PWD/crates/blue-rs/.ort/onnxruntime-win-x64-1.23.2/lib"
export ORT_STRATEGY=system ORT_PREFER_DYNAMIC_LINK=1 ORT_LIB_LOCATION="$ORT_DIR"
export RUSTFLAGS="-L native=$ORT_DIR"
mkdir -p target/debug/deps
cp "$ORT_DIR"/onnxruntime*.dll target/debug/deps/
cargo test --workspace --exclude mambotts-py
```

The copy into `target/debug/deps` is the same System32 problem the installer
works around: the test executables live there, and adding the ONNX Runtime
directory to `PATH` is not enough, because the old `onnxruntime.dll` in
`System32` is found first and every test binary that links `ort` dies with
`0xC000007B` before a single test runs. The GPU tests among the `--ignored`
ones try DX12 only with `dxcompiler.dll` and `dxil.dll` copied there as well;
without them they test Vulkan alone.

## Models

Model files are downloaded on first use into the application-local data
directory:

- macOS: `~/Library/Application Support/com.maxmelichov.mambotts/models`
- Windows: `%LOCALAPPDATA%\com.maxmelichov.mambotts\models`
- Linux: `~/.local/share/com.maxmelichov.mambotts/models`

BlueV3 v3.0.1 is the only model: the app and the server offer nothing else.
Its manifest, files and voices live in `crates/mambotts-registry`.

## Upstream projects

- [BlueTTS](https://github.com/maxmelichov/BlueTTS) provides the ONNX text-to-speech model.
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) provides Hebrew grapheme-to-IPA inference, Hebrew diacritization, and optional speaker conditioning.
