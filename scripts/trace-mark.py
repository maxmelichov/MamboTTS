#!/usr/bin/env python3
"""Regenerate mambotts-desktop/icon-src/*.svg from the brand mark PNG.

    uv run scripts/trace-mark.py

The design belongs to mambotts-desktop/src/assets/mambotts-logo.png: a rounded
yellow tile, a chunky dark-brown "MT" with a cream keyline, a cyan sparkle and
an orange dot. This script traces that PNG into four vector layers and writes
one SVG per pixel tier, deciding only how much of the mark survives at each
size. Nothing here invents artwork - re-run it whenever the mark changes, then
run scripts/render-icons.py to turn the SVGs into the shipped assets.

The tiers exist because Windows shows the app icon at 16-32 px far more often
than at 256 (taskbar, Alt-Tab, Explorer details, the tray, the shortcut), and
details that carry the mark at 256 px are sub-pixel down there: the cream
keyline is 2.7% of the side, so it is 0.9 px at 32 and renders as grey mush.
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "numpy", "scipy", "vtracer"]
# ///

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import vtracer
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "mambotts-desktop"
MARK = DESKTOP / "src" / "assets" / "mambotts-logo.png"
SRC = DESKTOP / "icon-src"

YELLOW, BROWN, CREAM, CYAN, ORANGE = "#FCD423", "#35220F", "#FAF3E6", "#06BEC4", "#FA5E3E"
PAL = {"brown": (53, 34, 15), "cream": (250, 243, 230), "cyan": (6, 190, 195),
       "orange": (250, 94, 62), "yellow": (252, 212, 35)}

# All measured off the PNG rather than guessed: the tile's corner radius is 20%
# of the side, and the MT block sits just left of and just above centre.
R = 205
CX, CY = 519.5, 522.0

# The mark PNG is full-bleed, but every icon the owner generated from it sits in
# 9.8% of transparent margin - `tauri icon` adds it, and it is what both the
# Windows taskbar and the macOS Dock expect, so the shipped icons keep it. It is
# tapered away below 48 px: a tenth of a 16 px box is 1.6 px off each side of an
# already tiny glyph, and there the legibility is worth more than the margin.
OWNER_INSET = 0.098

TOKEN = re.compile(r"[-+]?\d*\.?\d+|[A-Za-z]")


def shift(d: str, tx: float, ty: float) -> str:
    """Add (tx, ty) to every coordinate. vtracer only emits absolute M/C/L/Z."""
    out: list[str] = []
    nums: list[float] = []

    def flush() -> None:
        for i in range(0, len(nums), 2):
            out.append(f"{nums[i] + tx:.1f}")
            out.append(f"{nums[i + 1] + ty:.1f}")
        nums.clear()

    for tok in TOKEN.findall(d):
        if tok.isalpha():
            flush()
            assert tok in "MCLZ", f"unexpected path command {tok!r}"
            out.append(tok)
        else:
            nums.append(float(tok))
    flush()
    s = re.sub(r"\.0\b", "", " ".join(out))
    return re.sub(r"\s*([MCLZ])\s*", r"\1", s)


def declutter(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop specks and pinholes that nearest-colour classification leaves behind."""
    lab, n = ndimage.label(mask)
    if not n:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= min_px])
    holes, hn = ndimage.label(~keep)
    if hn:
        hsizes = ndimage.sum(~keep, holes, range(1, hn + 1))
        edge = set(np.unique(np.concatenate(
            [holes[0], holes[-1], holes[:, 0], holes[:, -1]])))
        small = [i + 1 for i, s in enumerate(hsizes)
                 if s < min_px and (i + 1) not in edge]
        if small:
            keep |= np.isin(holes, small)
    return keep


def trace(mask: np.ndarray, name: str) -> list[str]:
    img = Image.fromarray(np.where(declutter(mask, 60), 0, 255).astype(np.uint8))
    with tempfile.TemporaryDirectory() as td:
        ip, op = Path(td) / "in.png", Path(td) / "out.svg"
        img.convert("RGB").save(ip)
        vtracer.convert_image_to_svg_py(
            str(ip), str(op), colormode="binary", mode="spline", filter_speckle=8,
            corner_threshold=55, length_threshold=3.5, splice_threshold=45,
            path_precision=2)
        svg = op.read_text(encoding="utf-8")
    # vtracer draws each contour at its own origin and puts the placement in a
    # translate; bake that in so the result is a plain 1024-canvas path.
    paths = [shift(d, float(tx), float(ty)) for d, tx, ty in re.findall(
        r'<path\s+d="([^"]+)"[^>]*?transform="translate\(([-\d.]+),([-\d.]+)\)"', svg)]
    assert paths, f"nothing traced for {name}"
    print(f"  {name}: {len(paths)} contour(s)")
    return paths


def layers() -> dict[str, list[str]]:
    a = np.array(Image.open(MARK).convert("RGBA")).astype(np.int32)
    rgb, alpha = a[:, :, :3], a[:, :, 3]
    names = list(PAL)
    cls = np.stack([((rgb - np.array(PAL[n])) ** 2).sum(-1) for n in names]).argmin(0)
    cls[alpha < 128] = -1
    m = {n: cls == i for i, n in enumerate(names)}
    # The glyph silhouette is cream plus brown: drawing cream first and brown on
    # top is how the mark is built, a dark letter inside a light keyline.
    return {"silhouette": trace(m["brown"] | m["cream"], "silhouette"),
            "brown": trace(m["brown"], "brown"),
            "cyan": trace(m["cyan"], "cyan"),
            "orange": trace(m["orange"], "orange")}


LAYERS: dict[str, list[str]] = {}


def group(layer: str, fill: str) -> str:
    body = "".join(f'\n    <path d="{p}"/>' for p in LAYERS[layer])
    return f'  <g fill="{fill}">{body}\n  </g>'


def art(size: int, comment: str, *, mt_scale: float = 1.0, cream: bool = True,
        accents: bool = True, inset: float = OWNER_INSET) -> str:
    parts = [f'  <rect width="1024" height="1024" rx="{R}" fill="{YELLOW}"/>']
    if accents:
        parts += [group("cyan", CYAN), group("orange", ORANGE)]
    inner = ([group("silhouette", CREAM)] if cream else []) + [group("brown", BROWN)]
    body = "\n".join(inner)
    if abs(mt_scale - 1.0) > 1e-9:
        tx, ty = CX * (1 - mt_scale), CY * (1 - mt_scale)
        body = (f'  <g transform="translate({tx:.1f},{ty:.1f}) scale({mt_scale:g})">\n'
                + "\n".join("  " + ln for ln in body.splitlines()) + "\n  </g>")
    parts.append(body)
    if inset:
        p = 1024 * inset
        parts = [f'  <g transform="translate({p:.1f},{p:.1f}) scale({1 - 2 * inset:g})">'] \
            + ["  " + ln for b in parts for ln in b.splitlines()] + ["  </g>"]
    return (f"<!-- {comment} -->\n<svg width=\"{size}\" height=\"{size}\""
            ' viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">\n'
            + "\n".join(parts) + "\n</svg>\n")


HEAD = ("MamboTTS mark, traced from src/assets/mambotts-logo.png by\n"
        "     scripts/trace-mark.py. The artwork is the owner's; the only thing that\n"
        "     changes between tiers is how much of it survives at that pixel size.")

FULL = f"{HEAD}\n     %s: the mark in full."
TIERS: dict[str, tuple[int, str, dict]] = {
    "icon-256.svg": (256, FULL % "256 px tier (200 px and up)", {}),
    "icon-128.svg": (128, FULL % "128 px tier (81-199 px)", {}),
    "icon-48.svg": (48, f"{HEAD}\n     48 px tier (41-80 px): still the full mark. The sparkle is ~7 px\n"
                        "     here and the keyline 1.3 px, which is the floor for both. The margin\n"
                        "     starts closing from here down.",
                    dict(inset=0.08)),
    "icon-32.svg": (32, f"{HEAD}\n     32 px tier (29-40 px): the cream keyline is 2.7% of the side, so\n"
                        "     0.9 px here - it renders as grey mush rather than as a keyline, and\n"
                        "     goes. The MT grows into the room it leaves. The sparkle stays: at\n"
                        "     ~4.5 px it still reads as a sparkle.",
                    dict(mt_scale=1.08, cream=False, inset=0.06)),
    "icon-24.svg": (24, f"{HEAD}\n     24 px tier (21-28 px): the sparkle is down to ~3 px and the dot to\n"
                        "     ~1.5 px, so they go with the keyline. Dark brown on yellow carries\n"
                        "     the contrast on its own.",
                    dict(mt_scale=1.08, cream=False, accents=False, inset=0.04)),
    "icon-16.svg": (16, f"{HEAD}\n     16 px tier (up to 20 px): the yellow tile and the MT, nothing else,\n"
                        "     and the margin down to half a pixel. The letters are deliberately not\n"
                        "     fattened - at 16 px the strokes are already 2-3 px and any dilation\n"
                        "     closes the M's counters into a blob.",
                    dict(mt_scale=1.08, cream=False, accents=False, inset=0.03)),
}
# macOS wants the same tile and the same margin, so its two simplified tiers are
# the same drawings as their Windows counterparts.
TIERS["icon-mac-32.svg"] = (32, TIERS["icon-32.svg"][1].replace(
    "32 px tier (29-40 px)", "macOS 17-48 px tier"), TIERS["icon-32.svg"][2])
TIERS["icon-mac-16.svg"] = (16, TIERS["icon-16.svg"][1].replace(
    "16 px tier (up to 20 px)", "macOS 16 px tier"), TIERS["icon-16.svg"][2])


def tile(x: float, y: float, side: float) -> str:
    """The mark as a rounded tile of `side` px at (x, y), for the wizard art."""
    inner = [f'<rect width="1024" height="1024" rx="{R}" fill="{YELLOW}"/>',
             group("cyan", CYAN), group("orange", ORANGE),
             group("silhouette", CREAM), group("brown", BROWN)]
    body = "\n".join("  " + ln for b in inner for ln in b.splitlines())
    return (f'  <g transform="translate({x},{y}) scale({side / 1024:.6g})">\n'
            f"{body}\n  </g>")


def wizard() -> dict[str, str]:
    return {
        "nsis-header.svg": f"""<!-- Source for src-tauri/installer/header.bmp - the 150x57 banner NSIS shows in
     the top-right of every wizard page. Exported as a 24-bit BMP at exactly
     150x57; NSIS will not accept any other size or depth. -->
<svg width="150" height="57" viewBox="0 0 150 57" xmlns="http://www.w3.org/2000/svg">
  <rect width="150" height="57" fill="{BROWN}"/>
  <rect y="54" width="150" height="3" fill="{YELLOW}"/>
{tile(9, 6.5, 44)}
  <!-- 15/8 px, not 17/9: at 17 px "MamboTTS" runs past x=150 and the banner
       loses its last letter. Measured on the rendered BMP, not guessed. -->
  <text x="58" y="27" font-family="Segoe UI, Arial, sans-serif" font-size="15"
        font-weight="700" fill="{YELLOW}">MamboTTS</text>
  <text x="58" y="41" font-family="Segoe UI, Arial, sans-serif" font-size="8"
        fill="{CREAM}" opacity="0.72">Hebrew text to speech</text>
</svg>
""",
        "nsis-sidebar.svg": f"""<!-- Source for src-tauri/installer/sidebar.bmp - the 164x314 panel NSIS shows on
     the welcome and finish pages. Exported as a 24-bit BMP at exactly 164x314. -->
<svg width="164" height="314" viewBox="0 0 164 314" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3D2F12"/>
      <stop offset="1" stop-color="#241A08"/>
    </linearGradient>
  </defs>
  <rect width="164" height="314" fill="url(#bg)"/>
  <g stroke="{YELLOW}" fill="none" opacity="0.10">
    <circle cx="82" cy="116" r="86"/>
    <circle cx="82" cy="116" r="102"/>
    <circle cx="82" cy="116" r="120"/>
  </g>
{tile(34, 60, 96)}
  <text x="82" y="222" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif"
        font-size="24" font-weight="700" fill="{YELLOW}">MamboTTS</text>
  <rect x="62" y="232" width="40" height="3" rx="1.5" fill="{CYAN}"/>
  <text x="82" y="256" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif"
        font-size="11" fill="{CREAM}" opacity="0.75">Hebrew text to speech</text>
  <text x="82" y="272" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif"
        font-size="11" fill="{CREAM}" opacity="0.75">that runs on your machine</text>
  <circle cx="72" cy="294" r="3" fill="{ORANGE}"/>
  <circle cx="82" cy="294" r="3" fill="{YELLOW}" opacity="0.6"/>
  <circle cx="92" cy="294" r="3" fill="{CYAN}"/>
</svg>
""",
    }


def main() -> None:
    print(f"Tracing {MARK.relative_to(ROOT)}")
    LAYERS.update(layers())

    print("Icon tiers")
    for name, (size, comment, kw) in TIERS.items():
        (SRC / name).write_text(art(size, comment, **kw), encoding="utf-8")
        print(f"  {name}")

    print("macOS master")
    (DESKTOP / "app-icon.svg").write_text(art(
        1024, "MamboTTS mark, traced from src/assets/mambotts-logo.png by\n"
              "     scripts/trace-mark.py. Feeds icon.icns above 48 px, and keeps the\n"
              "     margin the Dock expects. The in-app header and the website use the\n"
              "     full-bleed src/assets/mambotts-mark.png instead."), encoding="utf-8")
    print("  app-icon.svg")

    print("NSIS wizard art")
    for name, text in wizard().items():
        (SRC / name).write_text(text, encoding="utf-8")
        print(f"  {name}")


if __name__ == "__main__":
    main()
