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


def add_to_zip(zf: zipfile.ZipFile, item: Path, arcname: str) -> None:
    """Add a file or a whole directory tree to the zip.

    `tarfile.add` walks directories on its own but `ZipFile.write` only ever
    writes the single entry it is given, so the espeak-ng-data tree would end up
    as an empty directory in the Windows archive without this.
    """
    zf.write(item, arcname=arcname)
    if item.is_dir():
        for child in sorted(item.iterdir()):
            add_to_zip(zf, child, f"{arcname}/{child.name}")


def package(
    binary: Path,
    espeak_data: Path,
    out: Path,
    platform: str,
    version: str,
) -> None:
    if not binary.exists():
        raise FileNotFoundError(binary)
    if not espeak_data.is_dir() or not any(espeak_data.iterdir()):
        raise FileNotFoundError(
            f"espeak-ng-data directory missing or empty at {espeak_data}. espeak-rs looks for it "
            "beside the executable at runtime and otherwise falls back to a path baked in at "
            "compile time, which does not exist on end-user machines, so every non-Hebrew language "
            "(English, Spanish, German, Italian) would fail to synthesize in this archive."
        )

    with tempfile.TemporaryDirectory(prefix="mambotts-server-package-") as td:
        stage = Path(td) / out.stem.removesuffix(".tar")
        stage.mkdir(parents=True)

        target_name = "mambotts-server.exe" if platform.startswith("windows-") else "mambotts-server"
        target = stage / target_name
        shutil.copy2(binary, target)
        target.chmod(0o755)

        # espeak-rs resolves the data directory relative to the executable, so it
        # has to sit directly beside the binary rather than under a subdirectory.
        shutil.copytree(espeak_data, stage / "espeak-ng-data", dirs_exist_ok=True)

        (stage / "metadata.json").write_text(
            json.dumps(
                {
                    "component": "mambotts-server",
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
                    add_to_zip(zf, item, f"{stage.name}/{item.name}")
        else:
            with tarfile.open(out, "w:gz") as tf:
                for item in sorted(stage.iterdir()):
                    tf.add(item, arcname=f"{stage.name}/{item.name}")

    print(f"packaged {out} ({out.stat().st_size // 1024} KB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Package a mambotts server release archive")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--espeak-data",
        type=Path,
        required=True,
        help="espeak-ng-data directory to ship beside the binary, normally "
        "<target>/<profile>/build/espeak-rs-sys-<hash>/out/share/espeak-ng-data",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    package(args.binary, args.espeak_data, args.out, args.platform, args.version)


if __name__ == "__main__":
    main()
