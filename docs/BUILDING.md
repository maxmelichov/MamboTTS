# Building and reinstalling

MamboBlue packages a Tauri desktop application with a Rust inference sidecar. Use `pnpm` for the desktop and Cargo for the Rust workspace.

## Prerequisites

- Rust toolchain for your platform
- Node.js and pnpm
- Python + uv (used by the sidecar preparation script)
- Platform build dependencies required by Tauri

On Linux, install `patchelf`; it is required to package ONNX Runtime correctly.

## Install dependencies

```console
pnpm install
cd mamboblue-desktop
pnpm install
```

## Development

Run the desktop application with hot reload:

```console
cd mamboblue-desktop
pnpm tauri dev
```

The pre-build hook compiles the matching `mamboblue-server` sidecar and stages ONNX Runtime before Tauri starts.

## Build an installable desktop app

```console
cd mamboblue-desktop
pnpm tauri build
```

On macOS, the installer is normally written under:

```text
src-tauri/target/release/bundle/dmg/
```

Open the generated `.dmg`, drag MamboBlue into Applications, and replace the existing copy. If macOS says the app is still running, quit it first. Windows installers and Linux packages are emitted in the corresponding `src-tauri/target/release/bundle/` subdirectories.

For a faster local reinstallable debug build:

```console
cd mamboblue-desktop
pnpm tauri build --debug
```

## Validate

```console
cargo test --workspace
cargo build -p mamboblue-server --release --bin mamboblue-server
cd mamboblue-desktop
pnpm build
```

## Models

Model files are downloaded on first use into the application-local data directory:

- macOS: `~/Library/Application Support/com.maxmelichov.mamboblue/models`
- Windows: `%LOCALAPPDATA%\com.maxmelichov.mamboblue\models`
- Linux: `~/.local/share/com.maxmelichov.mamboblue/models`

The current release provides the BlueTTS model bundle. New models must be registered in `crates/mamboblue-registry` and included in a sidecar build before the desktop will offer them. Full instructions for contributors: [ADDING_MODELS.md](./ADDING_MODELS.md).

## Upstream projects

- [BlueTTS](https://github.com/maxmelichov/BlueTTS) provides the ONNX text-to-speech model.
- [RenikudPlus](https://github.com/maxmelichov/RenikudPlus) provides Hebrew grapheme-to-IPA inference and optional speaker conditioning.
- [Phonikud](https://github.com/phonikud/phonikud) provides optional Hebrew diacritics and vocalization tooling.
