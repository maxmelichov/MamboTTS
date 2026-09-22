# MamboTTS desktop

Tauri 2 + React + TypeScript shell around the local MamboTTS server.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## App icons

The brand mark is `src/assets/mambotts-logo.png`: a rounded yellow tile, a chunky
dark-brown "MT" inside a cream keyline, a cyan sparkle and an orange dot. That
PNG is the design; everything else is derived from it.

`icon-src/` holds one SVG per level of detail, because Windows shows the icon at
16-32 px far more often than at 256 px and a single big drawing scaled down to
16 px is a blob. The tiers only decide how much of the mark survives:

| Source | Used for | Detail |
| --- | --- | --- |
| `icon-16.svg` | up to 20 px | tile and MT only, at 1.08x so it fills the room the rest leaves |
| `icon-24.svg` | 21-28 px | same: the sparkle is ~3 px here and the dot ~1.5 px |
| `icon-32.svg` | 29-40 px | plus the sparkle and dot; the keyline is 0.9 px, so it stays off |
| `icon-48.svg` | 41-80 px | the full mark - 1.3 px is the floor for the cream keyline |
| `icon-128.svg` | 81-199 px | the full mark |
| `icon-256.svg` | 200 px and up | the full mark |
| `icon-mac-16/32.svg`, `app-icon.svg` | `icon.icns` | the same tile; macOS needs no separate shape |
| `nsis-header.svg`, `nsis-sidebar.svg` | NSIS wizard | exported as 24-bit BMPs at 150x57 and 164x314 |

Retrace the SVGs after the mark itself changes, then render the shipped assets:

```console
uv run scripts/trace-mark.py
uv run scripts/render-icons.py
```

Editing an `icon-src/*.svg` by hand works too - just skip the first command, or
it will overwrite your edit.

Do not run `pnpm tauri icon`. It downscales one image for every size, which is
the problem this directory exists to solve, and it writes every `icon.ico` frame
as a PNG blob. Explorer reads those, but GDI+ does not: `System.Drawing.Icon`
reads the deflate stream as if it were a DIB and returns coloured static, so
anything on that path shows noise where the app icon should be. `render-icons.py`
writes the frames below 256 px as 32-bit DIBs instead.
