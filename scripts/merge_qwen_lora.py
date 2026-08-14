"""Merge the QwenTTS-he LoRA adapter into Qwen3-TTS-12Hz-1.7B-Base.

The adapter wraps the talker submodule, so every adapter key maps onto the
base checkpoint by swapping the `base_model.model.` prefix for `talker.`.
Two kinds of entries are present:

  * LoRA pairs (lora_A / lora_B) over the attention and MLP projections,
    folded in as W += (B @ A) * (alpha / r).
  * 20 fully retrained tensors (codec_head, text_projection, the 15
    code_predictor lm_heads) listed in the config's modules_to_save,
    which replace the base weight outright.

Math runs in float32 and is cast back to the base dtype so the result
stays a drop-in replacement for the original file.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True, help="Qwen3-TTS-12Hz-1.7B-Base checkout")
parser.add_argument("--adapter", type=Path, required=True, help="QwenTTS-he-1.7B checkout")
parser.add_argument("--out", type=Path, required=True, help="directory to write the merged checkpoint to")
args = parser.parse_args()
BASE, ADAPTER, OUT = args.base, args.adapter, args.out

cfg = json.loads((ADAPTER / "adapter_config.json").read_text())
scale = cfg["lora_alpha"] / cfg["r"]
print(f"lora_alpha={cfg['lora_alpha']} r={cfg['r']} scale={scale}", flush=True)

base = load_file(BASE / "model.safetensors")
adapter = load_file(ADAPTER / "adapter_model.safetensors")
print(f"loaded base={len(base)} adapter={len(adapter)}", flush=True)


def to_base_key(key: str) -> str:
    assert key.startswith("base_model.model."), key
    return "talker." + key[len("base_model.model.") :]


pairs = sorted({k.split(".lora_A.")[0] for k in adapter if ".lora_A." in k})
replaced = [k for k in adapter if ".lora_A." not in k and ".lora_B." not in k]

merged_count = 0
for stem in pairs:
    a = adapter[f"{stem}.lora_A.weight"]
    b = adapter[f"{stem}.lora_B.weight"]
    target = to_base_key(stem) + ".weight"
    if target not in base:
        raise KeyError(f"no base weight for {stem} -> {target}")
    w = base[target]
    delta = (b.float() @ a.float()) * scale
    if delta.shape != w.shape:
        raise ValueError(f"{target}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
    base[target] = (w.float() + delta).to(w.dtype)
    merged_count += 1

replaced_count = 0
for key in replaced:
    target = to_base_key(key)
    if target not in base:
        raise KeyError(f"no base weight for retrained module {key} -> {target}")
    if adapter[key].shape != base[target].shape:
        raise ValueError(f"{target}: shape mismatch")
    base[target] = adapter[key].to(base[target].dtype)
    replaced_count += 1

print(f"merged {merged_count} LoRA pairs, replaced {replaced_count} retrained tensors", flush=True)

OUT.mkdir(exist_ok=True)
save_file(base, OUT / "model.safetensors", metadata={"format": "pt"})

# convert.py consumes the whole checkpoint directory, so carry across the
# configs, tokenizer files and the separate speech tokenizer untouched.
for item in BASE.iterdir():
    if item.name in {"model.safetensors", ".cache", ".gitattributes"}:
        continue
    dest = OUT / item.name
    if item.is_dir():
        shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
        shutil.copy2(item, dest)

print("MERGED_OK", OUT, flush=True)
