#!/usr/bin/env python3
"""Export an X2 Turn Hugging Face checkpoint into a vLLM-friendly directory.

Input layout::

    X2-Turn-4B-0812/
      model.safetensors   # base_model.* + vad_lm_head.weight
      params.json / tekken.json / config.json / ...

Output layout::

    X2-Turn-4B-0812-vllm/
      consolidated.safetensors  # mistral keys + vad_lm_head.weight
      params.json, tekken.json, ...

Loading either the Hugging Face checkpoint or the exported directory works with
this fork; the export is mainly for faster cold-start and easier inspection.
"""

from __future__ import annotations

import argparse

# Allow running without installing the package: load remap helpers directly.
import importlib.util
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
_UTILS = REPO_ROOT / "vllm/model_executor/models/voxtral_mtp_utils.py"
_spec = importlib.util.spec_from_file_location("voxtral_mtp_utils", _UTILS)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
remap_mtp_weights = _mod.remap_mtp_weights


CONFIG_FILES = [
    "params.json",
    "tekken.json",
    "generation_config.json",
    "processor_config.json",
]


def export_mtp(src: Path, dst: Path, base: Path | None = None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    src_weights = src / "model.safetensors"
    if not src_weights.exists():
        raise FileNotFoundError(f"missing {src_weights}")

    sd = load_file(str(src_weights))
    remapped, vad = remap_mtp_weights(sd.items())
    out = {k: v for k, v in remapped}
    if vad is not None:
        out["vad_lm_head.weight"] = vad

    out_path = dst / "consolidated.safetensors"
    save_file(out, str(out_path))
    print(f"wrote {len(out)} tensors -> {out_path}")

    for name in CONFIG_FILES:
        src_f = src / name
        if not src_f.exists() and base is not None:
            src_f = base / name
        if src_f.exists():
            shutil.copy2(src_f, dst / name)
            print(f"copied {name}")

    # Do NOT copy HF config.json — it lacks mistral-remapped fields
    # (e.g. audio_config.downsample_factor) and would break vLLM init.
    # params.json is the source of truth for Voxtral Realtime in vLLM.

    if not (dst / "params.json").exists():
        raise FileNotFoundError(
            f"{src}/params.json missing; pass --base with the official "
            "Voxtral-Mini-4B-Realtime directory"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src",
        type=Path,
        required=True,
        help="MTP final/ checkpoint directory",
    )
    p.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output directory for vLLM",
    )
    p.add_argument(
        "--base",
        type=Path,
        default=None,
        help=(
            "Optional official base model directory for missing params.json "
            "or tekken.json"
        ),
    )
    args = p.parse_args()
    export_mtp(args.src, args.dst, args.base)


if __name__ == "__main__":
    main()
