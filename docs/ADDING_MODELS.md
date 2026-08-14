# Adding open-source models to MamboBlue

MamboBlue downloads model assets at runtime, but **inference engines are compiled into the app**. There is no plugin/DLL loader for third-party TTS engines. New open-source models ship only after they are registered in the repo and merged through a pull request.

This guide covers two levels of contribution:

1. **Bundled assets for an existing runtime** (for example extra BlueTTS voices or an updated Hebrew G2P ONNX file)
2. **A new TTS runtime** (a new engine with its own loader, API fields, and download catalog entry)

If you only want to try a model locally without publishing it, see [Local experiments](#local-experiments). For shipping it to all users, follow the PR path below.

## How model delivery works

```text
crates/mamboblue-registry   ← catalog: names, sizes, download URLs, capabilities
        ↓
desktop downloader (Tauri)   ← fetches files into the app data `models/` folder
        ↓
mamboblue-server sidecar    ← loads files through a compiled Runtime adapter
        ↓
React UI                     ← picks a runtime from the catalog, then synthesizes
```

Important consequences:

- Updating a download URL or file list means changing `mamboblue-registry` and usually cutting a new app release.
- Adding a brand-new engine also means Rust server work and packaging native dependencies with the sidecar.
- Dynamic native plugins are intentionally unsupported (code signing, ABI drift, and dependency resolution across macOS/Windows/Linux). See [ARCHITECTURE.md](./ARCHITECTURE.md).

Model files land on disk in:

| Platform | Directory |
| --- | --- |
| macOS | `~/Library/Application Support/com.maxmelichov.mamboblue/models` |
| Windows | `%LOCALAPPDATA%\com.maxmelichov.mamboblue\models` |
| Linux | `~/.local/share/com.maxmelichov.mamboblue/models` |

## Requirements before you open a PR

Your contribution must meet all of these:

1. **Open-source license** for weights, configs, and any inference code you add. Prefer OSI-approved licenses that allow redistribution (MIT, Apache-2.0, BSD, etc.). State the license clearly in the PR and in any Hugging Face / GitHub model card.
2. **Stable public download URLs** for every file the app will fetch. Hugging Face `resolve/main/...` (or a pinned commit/tag URL) is the usual pattern. Avoid links that require login, paid quotas, or brittle redirects.
3. **Reproducible bundle layout** — list every required file, expected relative path, and approximate download size.
4. **Offline use after download** — no phone-home inference API. Optional analytics in the desktop shell do not replace a local model.
5. **Cross-platform story** — macOS (Apple Silicon at minimum), Windows, and Linux, or an explicit limitation called out in the PR.
6. **Hebrew / multilingual claims must be honest** — set registry capabilities (`hebrew`, `streaming`, `voice_reference`, `fixed_voices`) to match real behavior.
7. **No license washing** — do not re-host incompatible weights under MamboBlue’s name. If upstream forbids redistribution, do not propose bundling them.

Maintainers may reject models that are too large for the desktop UX, require proprietary runtimes, or cannot be signed/packaged safely.

## Contribution path (summary)

1. Fork [maxmelichov/MamboBlue](https://github.com/maxmelichov/MamboBlue) and create a branch (`feat/add-<runtime-id>`).
2. Host or point to open-source model files with stable URLs.
3. Register the model in `crates/mamboblue-registry`.
4. Wire load + inference in `mamboblue-server` (new runtime) or verify paths still load (asset-only change).
5. Teach the desktop downloader / onboard flow about the files if needed.
6. Update docs (`README.md`, this file, [ARCHITECTURE.md](./ARCHITECTURE.md) when capabilities change).
7. Run the checks in [Validation](#validation).
8. Open a pull request against `main` with the checklist below.

## Option A — Add or update assets for BlueTTS (existing runtime)

Use this when you are not introducing a new engine. Examples:

- New fixed voice embedding JSON next to `voices/female1.json`
- Newer `renikud-plus.onnx` (or equivalent) with a compatible API
- Replacement BlueTTS ONNX pipeline pieces that keep the same IO contract

### Steps

1. **Publish the files** under a public repo (Hugging Face or GitHub Releases) with an open-source license and a model card.
2. **Edit** [`crates/mamboblue-registry/src/lib.rs`](../crates/mamboblue-registry/src/lib.rs):
   - Add or change entries in `BLUE_FILES` (`name` = relative path under the install directory, `url` = direct download).
   - Keep `BLUE_REQUIRED_FILES` in sync so install detection stays correct.
   - Bump `version` / `size` / `description` on the `BLUE` manifest when the bundle meaningfully changes.
3. **Keep desktop paths aligned** if the desktop still hardcodes Blue filenames or helper URLs (see [`mamboblue-desktop/src-tauri/src/model.rs`](../mamboblue-desktop/src-tauri/src/model.rs)). Prefer reading from the registry where possible; if a temporary constant remains, update it in the same PR.
4. **Server loading** — if filenames change, update Blue load validation and any path defaults in:
   - [`mamboblue-server/src/server/handlers/load.rs`](../mamboblue-server/src/server/handlers/load.rs)
   - [`mamboblue-server/src/runtime/blue.rs`](../mamboblue-server/src/runtime/blue.rs)
   - desktop runner DTOs / load requests under `mamboblue-desktop/src-tauri/src/runner/`
5. **UI** — only if users need a new control (for example a new voice name or G2P option). Fixed voices are usually discovered from the loaded runtime’s voice list.
6. **Document** the upstream project in `README.md` → “Models and phonemizers” when it is a distinct open-source dependency.

### BlueTTS bundle expectations today

Registered Blue files currently include the ONNX pipeline (`duration_predictor`, `text_encoder`, `vector_estimator`, `vocoder`), `vocab.json`, `tts.json`, voice styles under `voices/`, and `renikud-plus.onnx` for Hebrew G2P. Optional Phonikud is downloaded separately when selected.

If your change breaks the existing Blue IO contract, treat it as a new runtime (Option B) or a major version bump of the Blue directory name (for example a new `directory` / `version` so old and new installs do not collide).

## Option B — Add a new TTS runtime

A “runtime” is a full inference backend with a stable string id (Blue’s id is `"blue"`).

### 1. Registry manifest

In [`crates/mamboblue-registry/src/lib.rs`](../crates/mamboblue-registry/src/lib.rs):

1. Define `ModelFile` slices and `required_files`.
2. Create a `RuntimeManifest` with:
   - `id` — stable snake-case id used by the API (`"blue"`, `"your_engine"`)
   - `name` — UI display name
   - `version` — bundle tag shown in the catalog
   - `size` — approximate download size string for the onboard UI
   - `description` — one short sentence
   - `directory` — folder under `models/`
   - `install_kind` — `Files` for per-file downloads, or `Archive` if you ship a tarball
   - `capabilities` — honest feature flags
3. Append the manifest to the `RUNTIMES` array.
4. Add a unit test that asserts required files / capabilities for the new runtime.

The desktop `get_model_sources` command and server `GET /v1/models/sources` both serialize this registry. **Do not invent a second catalog.**

### 2. Server `Runtime` adapter

Implement [`Runtime`](../mamboblue-server/src/runtime/mod.rs) in a new module under `mamboblue-server/src/runtime/`, then export it from `runtime/mod.rs`.

You must implement:

| Method | Role |
| --- | --- |
| `languages` / `voices` / `sample_rate` | Metadata for UI and API |
| `phonemize` / `supported_phonemes` | Advanced IPA editor support (return empty / error if unsupported) |
| `diacritize` | Hebrew diacritics helper (no-op/`Err` if unsupported) |
| `synthesize_streaming` | Primary playback path (framed WAV chunks) |
| `synthesize_phonemes_streaming` | Advanced-mode phoneme synthesis |
| `synthesize_to_file` | Non-streaming export |

Add a `RuntimeParams` variant for the paths/config your loader needs, and branch on it in [`Server::load_model`](../mamboblue-server/src/server/state.rs).

Update [`POST /v1/models/load`](../mamboblue-server/src/server/handlers/load.rs) so the new `runtime` id is accepted and validated (Blue-only checks must become a match on runtime id).

If you introduce a Rust crate for the engine (like `crates/blue-rs`), keep ONNX / native deps explicit and document packaging needs for `scripts/pre_build.py`.

### 3. Desktop download and load wiring

Minimum desktop work:

1. [`mamboblue-desktop/src-tauri/src/model.rs`](../mamboblue-desktop/src-tauri/src/model.rs) — implement download for the new runtime id (today Blue is special-cased; generalize or add a clear branch). Respect `InstallKind::Files` vs archive extraction.
2. Runner client / DTOs — send the correct load payload (`runtime`, `model_path`, and any engine-specific fields).
3. TypeScript types in [`mamboblue-desktop/src/lib/types.ts`](../mamboblue-desktop/src/lib/types.ts) if new fields appear.
4. Onboard / settings UI — the catalog cards come from registry sources; ensure selecting the new runtime downloads, persists `mamboblue.runtime`, and reloads the sidecar.

Do not hardcode a one-off marketing card that disagrees with the registry.

### 4. Packaging native dependencies

Runtimes are compiled into `mamboblue-server` and bundled as a Tauri sidecar. If your engine needs extra shared libraries (ONNX Runtime, espeak data, etc.):

1. Extend [`scripts/pre_build.py`](../scripts/pre_build.py) (and Linux/Windows CI if needed) so those libraries ship beside the sidecar.
2. Confirm macOS codesigning / Windows/Linux packaging still work. See [BUILDING.md](./BUILDING.md) and `docs/code-signing/`.

PRs that cannot ship a signed, offline desktop build will not be merged.

### 5. Documentation

Update:

- `README.md` feature / models list when the runtime is user-facing
- [ARCHITECTURE.md](./ARCHITECTURE.md) “Current runtime” section
- This guide if you introduce a new contribution pattern

## Pull request checklist

Open the PR against `main` on GitHub. Title it clearly, for example:

`Add <RuntimeName> open-source TTS runtime`

or

`Update BlueTTS bundle to include <asset>`

PR description should include:

```markdown
## Summary
- What model/runtime is being added or updated
- Upstream project links + licenses
- Bundle size and install directory

## Model hosting
- Hugging Face / release URLs for every file
- Confirm redistributable under an OSI-friendly license

## Implementation
- [ ] Registry manifest updated (`crates/mamboblue-registry`)
- [ ] Server load + Runtime path works
- [ ] Desktop download + onboard selection works
- [ ] Docs updated

## Test plan
- [ ] Fresh download on a clean models directory
- [ ] `POST /v1/models/load` succeeds for the new/updated runtime
- [ ] Synthesis from the desktop UI (and streaming if claimed)
- [ ] Advanced phoneme mode behavior verified or documented as unsupported
- [ ] macOS / Windows / Linux notes (what you tested)
```

### Review expectations

Reviewers will check licensing, URL stability, registry honesty, API compatibility, and whether the desktop + sidecar stay in lockstep. Expect requests for smaller bundles, clearer model cards, or follow-up packaging fixes.

After merge, the feature ships in a **versioned desktop release**. Until users update the app, they will not see registry-only changes.

## Validation

From the repo root (see also [BUILDING.md](./BUILDING.md)):

```console
cargo test --workspace
cargo build -p mamboblue-server --release --bin mamboblue-server
cd mamboblue-desktop
pnpm install
pnpm build
pnpm tauri dev
```

Manual smoke test:

1. Wipe or rename the local `models/` directory (paths above).
2. Launch the app and download the runtime from onboard.
3. Load a fixed voice, synthesize a short Hebrew and English phrase if claimed.
4. Hit `GET /v1/models/sources` on the local sidecar and confirm the manifest matches the registry.

## Local experiments

You can prototype without a PR:

1. Build the server/desktop from a private branch.
2. Point load paths at local ONNX/GGUF files via the load API fields or environment variables currently recognized by the Blue loader (`MAMBOBLUE_BLUE_MODEL_DIR`, `MAMBOBLUE_RENIKUD_PATH`, etc.).
3. Keep proprietary or non-redistributable weights off the public registry.

Local success is not enough for merge — public URLs, license clarity, and packaging still required.

## What not to send

- Closed-weight models without redistribution rights
- “Just add this Hugging Face repo link” with no registry/server/desktop wiring
- Runtime PRs that only update the React UI
- Unsigned binary blobs or pretrained weights checked into git (host them externally; reference URLs from the registry)
- Engines that require a cloud API key for every utterance

## Reference map

| Concern | Primary location |
| --- | --- |
| Catalog / URLs / capabilities | `crates/mamboblue-registry/src/lib.rs` |
| HTTP sources API | `mamboblue-server/src/server/sources.rs` |
| Load API | `mamboblue-server/src/server/handlers/load.rs` |
| Runtime trait + params | `mamboblue-server/src/runtime/mod.rs` |
| Blue reference adapter | `mamboblue-server/src/runtime/blue.rs` |
| Downloads | `mamboblue-desktop/src-tauri/src/model.rs` |
| Sidecar packaging | `scripts/pre_build.py` |
| Architecture overview | [ARCHITECTURE.md](./ARCHITECTURE.md) |
| Build / reinstall | [BUILDING.md](./BUILDING.md) |

Questions that do not fit a PR description can go in a GitHub Discussion or Issue tagged for models. Prefer a draft PR with WIP commits over a speculative issue that cannot show URLs and licenses.
