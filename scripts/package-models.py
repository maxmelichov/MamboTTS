#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path, archive_name: str) -> dict[str, object]:
    return {
        "name": archive_name,
        "source": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def package(quant: str, model: Path, codec: Path, out_dir: Path, version: str) -> Path:
    if not model.exists():
        raise FileNotFoundError(model)
    if not codec.exists():
        raise FileNotFoundError(codec)

    package_name = f"mambotts-models-{quant}"
    archive = out_dir / f"{package_name}.tar.gz"
    out_dir.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        archive.unlink()

    with tempfile.TemporaryDirectory(prefix="mambotts-models-") as td:
        stage = Path(td) / package_name
        stage.mkdir()
        files = [
            (model, "qwen3-tts-model.gguf"),
            (codec, "qwen3-tts-codec.gguf"),
        ]
        metadata = {
            "component": "mambotts-models",
            "version": version,
            "quantization": quant,
            "source_models": [
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                "Qwen/Qwen3-TTS speech_tokenizer",
            ],
            "runtime": "mambotts-server",
            "files": [],
        }
        for src, dst_name in files:
            print(f"hashing {src.name}")
            metadata["files"].append(file_info(src, dst_name))
            shutil.copy2(src, stage / dst_name)
        (stage / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run(["tar", "-C", str(Path(td)), "-czf", str(archive), package_name])

    print(archive)
    return archive


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Qwen3-TTS GGUF model bundles")
    parser.add_argument("--quant", required=True, choices=["q8_0", "q5_0", "q4_k"])
    parser.add_argument("--version", default="mambotts-models-v0.1.3")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--model", type=Path, help="Override AR model GGUF path")
    parser.add_argument("--codec", type=Path, help="Override codec GGUF path")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    package(
        quant=args.quant,
        model=args.model or args.models_dir / f"qwen3-tts-0.6b-{args.quant}.gguf",
        codec=args.codec or args.models_dir / "qwen3-tts-tokenizer-f16.gguf",
        out_dir=args.out_dir,
        version=args.version,
    )


if __name__ == "__main__":
    main()
