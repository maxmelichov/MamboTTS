#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ORT_VERSION = "1.23.2"
ORT_CACHE = ROOT / "crates" / "blue-rs" / ".ort"

HOST_TRIPLE_MAP = {
    ("Darwin", "arm64"): "aarch64-apple-darwin",
    ("Darwin", "x86_64"): "x86_64-apple-darwin",
    ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
    ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
    ("Windows", "AMD64"): "x86_64-pc-windows-msvc",
}

ORT_PACKAGES = {
    "aarch64-apple-darwin": (
        f"onnxruntime-osx-arm64-{ORT_VERSION}",
        f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-osx-arm64-{ORT_VERSION}.tgz",
        "tgz",
        [f"lib/libonnxruntime.{ORT_VERSION}.dylib"],
    ),
    "x86_64-apple-darwin": (
        f"onnxruntime-osx-x86_64-{ORT_VERSION}",
        f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-osx-x86_64-{ORT_VERSION}.tgz",
        "tgz",
        [f"lib/libonnxruntime.{ORT_VERSION}.dylib"],
    ),
    "x86_64-unknown-linux-gnu": (
        f"onnxruntime-linux-x64-{ORT_VERSION}",
        f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-x64-{ORT_VERSION}.tgz",
        "tgz",
        [
            f"lib/libonnxruntime.so.{ORT_VERSION}",
            "lib/libonnxruntime_providers_shared.so",
        ],
    ),
    "aarch64-unknown-linux-gnu": (
        f"onnxruntime-linux-aarch64-{ORT_VERSION}",
        f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-linux-aarch64-{ORT_VERSION}.tgz",
        "tgz",
        [
            f"lib/libonnxruntime.so.{ORT_VERSION}",
            "lib/libonnxruntime_providers_shared.so",
        ],
    ),
    "x86_64-pc-windows-msvc": (
        f"onnxruntime-win-x64-{ORT_VERSION}",
        f"https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/onnxruntime-win-x64-{ORT_VERSION}.zip",
        "zip",
        ["lib/onnxruntime.dll", "lib/onnxruntime_providers_shared.dll"],
    ),
}


def detect_host_target() -> str | None:
    return HOST_TRIPLE_MAP.get((platform.system(), platform.machine()))


def download_ort(target: str) -> Path:
    if target not in ORT_PACKAGES:
        raise KeyError(f"no ONNX Runtime package mapping for target {target}")
    name, url, archive_kind, _ = ORT_PACKAGES[target]
    dest = ORT_CACHE / name
    marker = dest / "VERSION.txt"
    if marker.exists() and marker.read_text(encoding="utf-8").strip() == ORT_VERSION:
        return dest

    ORT_CACHE.mkdir(parents=True, exist_ok=True)
    archive = ORT_CACHE / f"{name}.{archive_kind}"
    print(f"+ downloading {url}")
    urllib.request.urlretrieve(url, archive)

    if dest.exists():
        shutil.rmtree(dest)
    if archive_kind == "tgz":
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(ORT_CACHE)
    else:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(ORT_CACHE)
    if not dest.exists():
        raise FileNotFoundError(f"expected extracted ORT directory at {dest}")
    marker.write_text(ORT_VERSION + "\n", encoding="utf-8")
    return dest


def install_ort_libs(target: str, dest_dir: Path) -> list[Path]:
    name, _, _, relative_libs = ORT_PACKAGES[target]
    ort_root = download_ort(target)
    installed: list[Path] = []
    for relative in relative_libs:
        source = ort_root / relative
        if not source.exists():
            print(f"warning: missing ORT library {source}")
            continue
        resolved = source.resolve()
        target_path = dest_dir / source.name
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        shutil.copy2(resolved, target_path)
        installed.append(target_path)

    if "linux" in target:
        versioned = dest_dir / f"libonnxruntime.so.{ORT_VERSION}"
        if versioned.exists():
            for link_name in ("libonnxruntime.so.1", "libonnxruntime.so"):
                link_path = dest_dir / link_name
                if link_path.exists() or link_path.is_symlink():
                    link_path.unlink()
                link_path.symlink_to(versioned.name)
                installed.append(link_path)

    if not installed:
        raise FileNotFoundError(f"no ONNX Runtime libraries found under {ort_root}")
    print(f"Installed ONNX Runtime libs from {name}: {[p.name for p in installed]}")
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MamboTTS server sidecar for Tauri builds")
    parser.add_argument("--target", help="Rust target triple, for example x86_64-unknown-linux-gnu")
    parser.add_argument("--profile", default="release", choices=["debug", "release"])
    args = parser.parse_args()

    target = args.target or detect_host_target()
    if not target:
        print("Warning: could not detect host target; pass --target or place the server sidecar manually.")
        return 0

    is_windows = target.endswith("windows-msvc")
    is_macos = "apple-darwin" in target
    is_linux = "linux" in target
    sidecar_name = f"mambotts-server-{target}" + (".exe" if is_windows else "")
    dest_dir = ROOT / "mambotts-desktop" / "src-tauri" / "binaries"
    dest = dest_dir / sidecar_name
    profile_args = [] if args.profile == "debug" else ["--release"]
    cmd = [
        "cargo",
        "build",
        "-p",
        "mambotts-server",
        "--bin",
        "mambotts-server",
        "--target",
        target,
        *profile_args,
    ]
    print("+", " ".join(cmd))
    build_env = os.environ.copy()
    ort_root = download_ort(target)
    lib_dir = ort_root / "lib"
    build_env["ORT_STRATEGY"] = "system"
    # ort-sys treats ORT_LIB_LOCATION as a direct -L path for the shared library.
    build_env["ORT_LIB_LOCATION"] = str(lib_dir if lib_dir.exists() else ort_root)
    build_env["ORT_PREFER_DYNAMIC_LINK"] = "1"
    if lib_dir.exists():
        path_key = "PATH" if is_windows else "LD_LIBRARY_PATH" if is_linux else "DYLD_LIBRARY_PATH"
        current = build_env.get(path_key, "")
        build_env[path_key] = f"{lib_dir}{os.pathsep}{current}" if current else str(lib_dir)
        library_path = build_env.get("LIBRARY_PATH", "")
        build_env["LIBRARY_PATH"] = f"{lib_dir}{os.pathsep}{library_path}" if library_path else str(lib_dir)
        rustflags = build_env.get("RUSTFLAGS", "")
        build_env["RUSTFLAGS"] = f"{rustflags} -L native={lib_dir}".strip()
    if is_macos:
        build_env["RUSTFLAGS"] = f"{build_env.get('RUSTFLAGS', '')} -C link-arg=-Wl,-headerpad_max_install_names".strip()
    if is_windows:
        # espeak-ng pulls registry APIs that need advapi32 on MSVC.
        build_env["RUSTFLAGS"] = f"{build_env.get('RUSTFLAGS', '')} -C link-arg=advapi32.lib".strip()
    subprocess.run(cmd, cwd=ROOT, env=build_env, check=True)

    source = ROOT / "target" / target / args.profile / ("mambotts-server.exe" if is_windows else "mambotts-server")
    if not source.exists():
        raise FileNotFoundError(source)
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    if not is_windows:
        dest.chmod(dest.stat().st_mode | 0o111)

    install_ort_libs(target, dest_dir)

    if is_macos:
        for rpath in (
            "@loader_path",
            "@loader_path/../Resources",
            "@loader_path/../Resources/binaries",
        ):
            subprocess.run(
                ["install_name_tool", "-add_rpath", rpath, str(dest)],
                check=False,
            )
    elif is_linux:
        patchelf = shutil.which("patchelf")
        if patchelf:
            subprocess.run(
                [patchelf, "--set-rpath", "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../Resources", str(dest)],
                check=True,
            )
        else:
            print("warning: patchelf not found; packaged Linux sidecar may fail to load ONNX Runtime")

    print(f"Installed MamboTTS server sidecar at {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
