#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RELEASES_PATH = ROOT / "mamboblue-website" / "src" / "lib" / "latest_release.json"
REPO = "maxmelichov/MamboBlue"
DESKTOP_TAG_RE = re.compile(r"^mamboblue-desktop-v")


def gh_json(*args: str) -> Any:
    result = subprocess.run(
        ["gh", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def latest_desktop_release() -> dict[str, Any]:
    releases = gh_json(
        "release",
        "list",
        "--repo",
        REPO,
        "--limit",
        "100",
        "--json",
        "tagName,publishedAt,name,isPrerelease",
    )
    for release in releases:
        tag = release.get("tagName", "")
        if DESKTOP_TAG_RE.match(tag):
            return gh_json(
                "release",
                "view",
                tag,
                "--repo",
                REPO,
                "--json",
                "tagName,publishedAt,isPrerelease,url,assets",
            )
    raise RuntimeError("No mamboblue-desktop-v* release found")


def asset_info(name: str) -> dict[str, str] | None:
    lower = name.lower()

    if lower.endswith(".dmg"):
        arch = "darwin-aarch64" if "aarch64" in lower or "arm64" in lower else "darwin-x86_64"
        return {"platform": "macos", "arch": arch, "kind": "dmg"}

    if lower.endswith("-setup.exe"):
        return {"platform": "windows", "arch": "windows-x86_64", "kind": "exe"}

    if lower.endswith(".msi"):
        return {"platform": "windows", "arch": "windows-x86_64", "kind": "msi"}

    if lower.endswith("-portable.zip") or (lower.endswith(".zip") and "portable" in lower):
        return {"platform": "windows", "arch": "windows-x86_64", "kind": "zip"}

    if lower.endswith(".appimage"):
        return {"platform": "linux", "arch": "linux-x86_64", "kind": "appimage"}

    if lower.endswith(".deb"):
        return {"platform": "linux", "arch": "linux-x86_64", "kind": "deb"}

    if lower.endswith(".rpm"):
        return {"platform": "linux", "arch": "linux-x86_64", "kind": "rpm"}

    return None


def map_assets(assets: list[dict[str, Any]]) -> list[dict[str, str]]:
    mapped: list[dict[str, str]] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        info = asset_info(name)
        if not info:
            continue
        mapped.append(
            {
                "url": str(asset.get("url", "")),
                "name": name,
                **info,
            }
        )
    return mapped


def main() -> int:
    release = latest_desktop_release()
    data = {
        "version": release["tagName"],
        "url": release["url"],
        "publishedAt": release["publishedAt"],
        "assets": map_assets(release.get("assets", [])),
    }
    RELEASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    RELEASES_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {RELEASES_PATH}")
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
