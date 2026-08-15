#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(src: Path, dst: Path) -> int:
    if not src.is_file():
        raise SystemExit(f"missing file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst.stat().st_size


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Kokoro runtime model files")
    parser.add_argument("--model", type=Path, default=Path("/tmp/mambotts-kokoro-c-assets/kokoro-v1.0.onnx"))
    parser.add_argument("--voices", type=Path, default=Path("/tmp/mambotts-kokoro-c-assets/voices-v1.0.bin"))
    parser.add_argument("--espeak-data", type=Path, default=Path("/tmp/mambotts-kokoro-c-assets/espeak-ng-data"))
    parser.add_argument("--version", default="kokoro-v1.0")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    if not args.espeak_data.is_dir():
        raise SystemExit(f"missing espeak-ng-data dir: {args.espeak_data}")

    staging = args.out_dir / f"mambotts-kokoro-models-{args.version}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model_size = copy_file(args.model, staging / "kokoro-v1.0.onnx")
    voices_size = copy_file(args.voices, staging / "voices-v1.0.bin")
    shutil.copytree(args.espeak_data, staging / "espeak-ng-data")
    espeak_size = dir_size(staging / "espeak-ng-data")

    manifest = {
        "runtime": "kokoro",
        "version": args.version,
        "files": {
            "model": "kokoro-v1.0.onnx",
            "voices": "voices-v1.0.bin",
            "espeak_data": "espeak-ng-data",
        },
        "sizes": {
            "model": model_size,
            "voices": voices_size,
            "espeak_data": espeak_size,
            "total_uncompressed": model_size + voices_size + espeak_size,
        },
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    archive = args.out_dir / f"{staging.name}.tar.gz"
    if archive.exists():
        archive.unlink()
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(staging, arcname=staging.name)
    print(archive)


if __name__ == "__main__":
    main()
