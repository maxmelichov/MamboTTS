#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def package(
    binary: Path,
    out: Path,
    platform: str,
    version: str,
) -> None:
    if not binary.exists():
        raise FileNotFoundError(binary)

    with tempfile.TemporaryDirectory(prefix="mamboblue-server-package-") as td:
        stage = Path(td) / out.stem.removesuffix(".tar")
        stage.mkdir(parents=True)

        target_name = "mamboblue-server.exe" if platform.startswith("windows-") else "mamboblue-server"
        target = stage / target_name
        shutil.copy2(binary, target)
        target.chmod(0o755)

        (stage / "metadata.json").write_text(
            json.dumps(
                {
                    "component": "mamboblue-server",
                    "version": version,
                    "platform": platform,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".zip":
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for item in sorted(stage.iterdir()):
                    zf.write(item, arcname=f"{stage.name}/{item.name}")
        else:
            with tarfile.open(out, "w:gz") as tf:
                for item in sorted(stage.iterdir()):
                    tf.add(item, arcname=f"{stage.name}/{item.name}")

    print(f"packaged {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a mamboblue server release archive")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    package(args.binary, args.out, args.platform, args.version)


if __name__ == "__main__":
    main()
