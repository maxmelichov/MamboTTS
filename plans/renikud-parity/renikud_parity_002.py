#!/usr/bin/env python
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""What pointing Hebrew and reading it back costs, over the parity corpus.

`/v1/diacritize` hands the desktop pointed Hebrew, the user edits it, and
`/v1/phonemize` reads it back — so the question this answers is how often that
round trip changes the reading, and how much the two marks behind
`Options::round_trip` (a rafe on a silent ו, a mapiq in a final ה) recover.

    MAMBOTTS_RENIKUD_PATH=.../renikud-plus.onnx \
      uv run plans/renikud-parity/renikud_parity_002.py

Rust only: upstream `vocalize` has no stress mark and no round-trip marks, so
there is nothing on the Python side to compare against.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from renikud_parity_001 import MODEL, OUT_DIR, ROOT, corpus, run_rust  # noqa: E402

VARIANTS = [
    ("upstream vocalize  ", {}),
    ("+ stress           ", {"with_stress": True}),
    ("+ stress+roundtrip ", {"with_stress": True, "round_trip": True}),
]


def main() -> int:
    # Hebrew and IPA go to the console, which is not UTF-8 by default on Windows.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = [text for text in corpus() if len(text) < 2000]
    cases = []
    for variant, (_, flags) in enumerate(VARIANTS):
        for text in texts:
            cases.append({"i": len(cases), "op": "round_trip", "text": text,
                          "niqqud": "use", "variant": variant, **flags})
    results = run_rust(cases, OUT_DIR / "roundtrip_requests.jsonl", OUT_DIR / "roundtrip.jsonl")
    by_index = {r["i"]: r for r in results}

    drifted: list[dict] = []
    for variant, (label, _) in enumerate(VARIANTS):
        same = stress_same = total = 0
        for case in (c for c in cases if c["variant"] == variant):
            out = by_index[case["i"]].get("out")
            if out is None:
                continue
            total += 1
            if out["ipa"] == out["read_back"]:
                same += 1
            elif variant == len(VARIANTS) - 1:
                drifted.append({"text": case["text"], **out})
            if out["ipa"].count("ˈ") == out["read_back"].count("ˈ"):
                stress_same += 1
        print(f"{label} identical {same}/{total}   same stress count {stress_same}/{total}")

    (OUT_DIR / "roundtrip_drift.json").write_text(
        json.dumps(drifted, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for item in drifted[:15]:
        print("\n", item["text"], "\n  ", item["pointed"], "\n  ", item["ipa"], "\n  ", item["read_back"])
    print(f"\n{len(drifted)} texts still drift; written to {OUT_DIR / 'roundtrip_drift.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
