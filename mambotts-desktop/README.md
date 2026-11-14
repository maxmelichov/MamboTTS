# MamboTTS desktop

Tauri 2 + React + TypeScript shell around the local MamboTTS server.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## App icons

The brand mark is a rounded yellow tile (`#FCD423`) carrying "Mt" in Space
Grotesk Bold, dark brown (`#2E2300`). The letters are outlined to paths, so no
font is needed to render them. The SVGs in `icon-src/` are the source of truth.

`icon-src/` holds one SVG per pixel tier, because Windows shows the icon at
16-32 px far more often than at 256 px. The tiers decide the margin and how
many letters fit:

| Source | Used for | Detail |
| --- | --- | --- |
| `icon-16.svg` | up to 20 px | tile and a single M; two letters do not fit |
| `icon-24.svg` | 21-28 px | tile and Mt |
| `icon-32.svg` | 29-40 px | tile and Mt |
| `icon-48.svg` | 41-80 px | tile and Mt |
| `icon-128.svg` | 81-199 px | tile and Mt |
| `icon-256.svg` | 200 px and up | tile and Mt |
| `icon-mac-16/32.svg`, `app-icon.svg` | `icon.icns` | the same tile; macOS needs no separate shape |
| `nsis-header.svg`, `nsis-sidebar.svg` | NSIS wizard | exported as 24-bit BMPs at 150x57 and 164x314 |

After editing an `icon-src/*.svg`, render the shipped assets:

```console
uv run scripts/render-icons.py
```

`scripts/trace-mark.py` belongs to the old raster artwork and is obsolete; do
not run it, it would overwrite these sources.

Do not run `pnpm tauri icon`. It downscales one image for every size, which is
the problem this directory exists to solve, and it writes every `icon.ico` frame
as a PNG blob. Explorer reads those, but GDI+ does not: `System.Drawing.Icon`
reads the deflate stream as if it were a DIB and returns coloured static, so
anything on that path shows noise where the app icon should be. `render-icons.py`
writes the frames below 256 px as 32-bit DIBs instead.
