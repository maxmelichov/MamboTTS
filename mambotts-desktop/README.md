# MamboTTS desktop

Tauri 2 + React + TypeScript shell around the local MamboTTS server.

## Recommended IDE Setup

- [VS Code](https://code.visualstudio.com/) + [Tauri](https://marketplace.visualstudio.com/items?itemName=tauri-apps.tauri-vscode) + [rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)

## App icons

`icon-src/` holds one SVG per level of detail, because Windows shows the icon at
16-32 px far more often than at 256 px and a single big drawing scaled down to
16 px is a blob:

| Source | Used for | Detail |
| --- | --- | --- |
| `icon-16.svg` | up to 20 px | badge, record, label |
| `icon-24.svg` | 21-28 px | same |
| `icon-32.svg` | 29-40 px | plus the spindle hole |
| `icon-48.svg` | 41-80 px | plus one groove, the highlight arc, the gold rim and the MT |
| `icon-128.svg` | 81-199 px | plus the six rays and the second groove |
| `icon-256.svg` | 200 px and up | the full drawing, tonearm and accent dots included |
| `icon-mac-16/32.svg`, `app-icon.svg` | `icon.icns` | macOS keeps the full-bleed rounded square |
| `nsis-header.svg`, `nsis-sidebar.svg` | NSIS wizard | exported as 24-bit BMPs at 150x57 and 164x314 |

The Windows and Linux art is the mark on transparency: a full-bleed rounded
square is macOS icon language and looks like a sticker on a Windows taskbar.

Regenerate everything after editing any of those files:

```console
uv run scripts/render-icons.py
```

Do not run `pnpm tauri icon`: it downscales one image for every size, which is
the problem this directory exists to solve.
