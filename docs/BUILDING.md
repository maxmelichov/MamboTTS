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
- CMake and a C toolchain, because espeak-ng is compiled from source
- Clang, because `espeak-rs-sys` generates its bindings with bindgen

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

For a faster local reinstallable debug build:

```console
cd mambotts-desktop
pnpm tauri build --debug
```

### If the espeak-ng build fails with "Failed to open"

espeak-ng compiles its phoneme data with fixed 180 byte path buffers, so a deep
checkout path can push the generated file names past the end of those buffers.
The build then stops on a truncated path, for example:

```text
Failed to open: '.../out/build/espeak-ng-data/../phsource/vwl_en_us_nyc/a_rais'Error: Compile error.
```

Note the missing `ed` at the end of `a_raised`. Adding `--target
aarch64-apple-darwin` makes this more likely, because the triple adds another
twenty one characters to every artifact path. Point Cargo at a shorter build
directory and it goes away:

```console
export CARGO_TARGET_DIR=/tmp/mambotts-build
uv run scripts/pre_build.py
```

`scripts/pre_build.py` reads `CARGO_TARGET_DIR`, so it still finds the sidecar
and stages it correctly. Moving the checkout itself somewhere shorter works too.

## Linux on x86_64

### System packages

On Ubuntu 22.04 or Debian, these are what the build actually needs. The list
matches `scripts/Dockerfile.linux-build`, which is the environment the Linux
bundles are verified in.

```console
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git curl wget file patchelf ca-certificates \
  libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev \
  libssl-dev libgtk-3-dev libxdo-dev clang libclang-dev libasound2-dev \
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
- LLVM, so bindgen can find `libclang.dll`. Set `LIBCLANG_PATH` to its `bin` directory if the build cannot locate it.
- CMake on `PATH`
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

`espeak-ng-data` ships beside the executables too, and the desktop shell
also passes `PIPER_ESPEAKNG_DATA_DIRECTORY` explicitly when it spawns the
sidecar. That value has any `\\?\` prefix stripped first: espeak-ng joins
the directory with its own relative paths using forward slashes, and Windows
does not normalise those inside a verbatim path the way it does for an
ordinary one, so a `\\?\` value makes every lookup under the data
directory fail. Without it espeak-ng falls back to the path baked in when it
was compiled, which belongs to the build machine, and every language except
Hebrew stops working.

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
export RUSTFLAGS="-L native=$ORT_DIR -C link-arg=advapi32.lib"
cargo xwin build -p mambotts-server --release --target x86_64-pc-windows-msvc
```

`advapi32.lib` is required because espeak-ng calls the Windows registry APIs.

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
cargo test --workspace --exclude mambotts-py --exclude espeak-rs
cargo build -p mambotts-server --release --bin mambotts-server
cd mambotts-desktop
pnpm build
```

`crates/espeak-rs` and `crates/espeak-rs-sys` are a vendored fork that nothing
in the product depends on, because `blue-rs` takes espeak from a git dependency
on piper-rs instead. Excluding `espeak-rs` from the test run keeps that unwired
copy out of the way.

## Models

Model files are downloaded on first use into the application-local data
directory:

- macOS: `~/Library/Application Support/com.maxmelichov.mambotts/models`
- Windows: `%LOCALAPPDATA%\com.maxmelichov.mambotts\models`
- Linux: `~/.local/share/com.maxmelichov.mambotts/models`

The current release provides the BlueTTS model bundle. New models must be
registered in `crates/mambotts-registry` and included in a sidecar build before
the desktop will offer them. Full instructions for contributors:
[ADDING_MODELS.md](./ADDING_MODELS.md).

## Upstream projects

- [BlueTTS](https://github.com/maxmelichov/BlueTTS) provides the ONNX text-to-speech model.
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) provides Hebrew grapheme-to-IPA inference and optional speaker conditioning.
- [Phonikud](https://github.com/phonikud/phonikud) provides optional Hebrew diacritics and vocalization tooling.
