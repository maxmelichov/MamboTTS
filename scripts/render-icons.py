#!/usr/bin/env python3
"""Render every shipped icon from the size-specific SVGs in mambotts-desktop/icon-src.

Windows shows the app icon at 16-32 px far more often than at 256 px (taskbar,
Alt-Tab, Explorer details view, the tray, the desktop shortcut). A single 1024 px
drawing scaled down to 16 px turns into a blob, which is exactly what the old
`tauri icon` output did. So the art exists at several levels of detail and each
output is rendered from the drawing made for its size - never downscaled from a
bigger one.

    uv run scripts/render-icons.py

The drawings in icon-src are traced from the brand mark in
mambotts-desktop/src/assets/mambotts-logo.png - see scripts/trace-mark.py, which
regenerates them. The artwork is the owner's; the tiers only decide how much of
it survives at a given size.

Outputs (all under mambotts-desktop/src-tauri/icons unless noted):
  * the Windows/Linux PNG set and the Windows Store Square*/StoreLogo PNGs
  * icon.ico, with 16/24/32/48/64 BMP frames plus a 256 PNG frame, each frame
    rendered from its own SVG
  * icon.icns, which keeps the macOS full-bleed rounded square
  * android/ mipmaps
  * ../installer/header.bmp and ../installer/sidebar.bmp for the NSIS wizard
  * mambotts-desktop/app-icon.png, mambotts-desktop/public/icon.png and
    mambotts-website/public/favicon.ico
"""

# /// script
# requires-python = ">=3.11"
# dependencies = ["pillow", "resvg-py"]
# ///

from __future__ import annotations

import io
import struct
from pathlib import Path

import resvg_py
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "mambotts-desktop"
WEBSITE = ROOT / "mambotts-website"
SRC = DESKTOP / "icon-src"
ICONS = DESKTOP / "src-tauri" / "icons"
INSTALLER = DESKTOP / "src-tauri" / "installer"

# Windows/Linux art: the mark on transparency, no macOS-style background tile.
# Each entry is the largest pixel size the drawing is meant for.
WIN_TIERS = [(20, "icon-16.svg"), (28, "icon-24.svg"), (40, "icon-32.svg"),
             (80, "icon-48.svg"), (199, "icon-128.svg"), (10_000, "icon-256.svg")]

# macOS art: keeps the full-bleed rounded square, simplified at the two sizes
# where the full drawing collapses.
MAC_TIERS = [(16, "icon-mac-16.svg"), (48, "icon-mac-32.svg"), (10_000, "app-icon.svg")]


def pick(tiers: list[tuple[int, str]], size: int) -> Path:
    for limit, name in tiers:
        if size <= limit:
            base = DESKTOP if name == "app-icon.svg" else SRC
            return base / name
    raise AssertionError("unreachable")


def render(svg_path: Path, size: int) -> Image.Image:
    svg = svg_path.read_text(encoding="utf-8")
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size))
    return Image.open(io.BytesIO(png)).convert("RGBA")


def win(size: int) -> Image.Image:
    return render(pick(WIN_TIERS, size), size)


def mac(size: int) -> Image.Image:
    return render(pick(MAC_TIERS, size), size)


def write_png(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "PNG")
    print(f"  {path.relative_to(ROOT)} ({img.width}x{img.height})")


def ico_frame(img: Image.Image) -> tuple[bytes, int]:
    """One ICO frame: a PNG blob at 256, a 32-bit DIB below that.

    `pnpm tauri icon` writes *every* frame as a PNG blob, which Explorer and
    modern shell paths do read - but GDI+ does not. `System.Drawing.Icon` on
    such a file does not fail; it reads the deflate stream as if it were a DIB
    and hands back noise, so anything going through GDI+ (WinForms, older
    installers, several screenshot and shortcut tools) shows a field of
    coloured static instead of the app icon. Below 256 the frames therefore
    stay as 32-bit DIBs, which every reader agrees on.
    """
    w, h = img.size
    if w >= 256:
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue(), 32

    px = img.load()
    xor = bytearray()
    for y in range(h - 1, -1, -1):  # DIBs are stored bottom-up
        for x in range(w):
            r, g, b, a = px[x, y]
            xor += bytes((b, g, r, a))
    row = ((w + 31) // 32) * 4  # 1bpp AND mask, rows padded to 4 bytes
    and_mask = bytearray()
    for y in range(h - 1, -1, -1):
        bits = bytearray(row)
        for x in range(w):
            if px[x, y][3] == 0:
                bits[x // 8] |= 0x80 >> (x % 8)
        and_mask += bits
    header = struct.pack("<IiiHHIIiiII", 40, w, h * 2, 1, 32, 0,
                         len(xor) + len(and_mask), 0, 0, 0, 0)
    return bytes(header + xor + and_mask), 32


def write_ico(path: Path, sizes: list[int]) -> None:
    frames = [(s, *ico_frame(win(s))) for s in sizes]
    out = bytearray(struct.pack("<HHH", 0, 1, len(frames)))
    offset = 6 + 16 * len(frames)
    for size, blob, bpp in frames:
        out += struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, bpp,
                           len(blob), offset)
        offset += len(blob)
    for _, blob, _ in frames:
        out += blob
    path.write_bytes(out)
    print(f"  {path.relative_to(ROOT)} (frames: {', '.join(str(s) for s in sizes)})")


# (OSType, pixel size). PNG payloads; every type here accepts PNG on 10.7+.
ICNS_TYPES = [("icp4", 16), ("icp5", 32), ("icp6", 64), ("ic07", 128),
              ("ic08", 256), ("ic09", 512), ("ic10", 1024), ("ic11", 32),
              ("ic12", 64), ("ic13", 256), ("ic14", 512)]


def write_icns(path: Path) -> None:
    body = bytearray()
    for ostype, size in ICNS_TYPES:
        buf = io.BytesIO()
        mac(size).save(buf, "PNG")
        data = buf.getvalue()
        body += ostype.encode("ascii") + struct.pack(">I", len(data) + 8) + data
    path.write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + bytes(body))
    print(f"  {path.relative_to(ROOT)} ({len(ICNS_TYPES)} entries)")


def write_bmp(svg_name: str, path: Path, width: int, height: int) -> None:
    """24-bit BMP at an exact size - NSIS rejects anything else."""
    svg = (SRC / svg_name).read_text(encoding="utf-8")
    png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height))
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    flat = Image.new("RGB", img.size, (53, 34, 15))  # the mark's dark brown
    flat.paste(img, (0, 0), img)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat.save(path, "BMP")
    print(f"  {path.relative_to(ROOT)} ({flat.width}x{flat.height}, 24-bit)")


def main() -> None:
    print("Windows / Linux PNGs")
    for name, size in [("32x32.png", 32), ("64x64.png", 64), ("128x128.png", 128),
                       ("128x128@2x.png", 256), ("icon.png", 512)]:
        write_png(win(size), ICONS / name)

    print("Windows Store logos")
    write_png(win(50), ICONS / "StoreLogo.png")
    for size in (30, 44, 71, 89, 107, 142, 150, 284, 310):
        write_png(win(size), ICONS / f"Square{size}x{size}Logo.png")

    print("Windows icon")
    write_ico(ICONS / "icon.ico", [16, 24, 32, 48, 64, 256])

    print("macOS icon")
    write_icns(ICONS / "icon.icns")

    print("Android mipmaps")
    for folder, legacy, foreground in [("mdpi", 48, 108), ("hdpi", 49, 162),
                                       ("xhdpi", 96, 216), ("xxhdpi", 144, 324),
                                       ("xxxhdpi", 192, 432)]:
        out = ICONS / "android" / f"mipmap-{folder}"
        write_png(mac(legacy), out / "ic_launcher.png")
        write_png(win(legacy), out / "ic_launcher_round.png")
        # Adaptive icons crop to the middle ~66%, so the mark is inset.
        inner = round(foreground * 0.66)
        canvas = Image.new("RGBA", (foreground, foreground), (0, 0, 0, 0))
        art = win(inner)
        canvas.paste(art, ((foreground - inner) // 2, (foreground - inner) // 2), art)
        write_png(canvas, out / "ic_launcher_foreground.png")

    print("NSIS wizard art")
    write_bmp("nsis-header.svg", INSTALLER / "header.bmp", 150, 57)
    write_bmp("nsis-sidebar.svg", INSTALLER / "sidebar.bmp", 164, 314)

    print("Web favicons")
    write_png(mac(1024), DESKTOP / "app-icon.png")
    write_png(win(64), DESKTOP / "public" / "icon.png")
    write_ico(WEBSITE / "public" / "favicon.ico", [16, 24, 32, 48])


if __name__ == "__main__":
    main()
