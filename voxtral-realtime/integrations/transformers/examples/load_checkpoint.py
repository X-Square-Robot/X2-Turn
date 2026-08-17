#!/usr/bin/env python3
"""Load a Voxtral MTP checkpoint from Hugging Face or a local directory."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import load_mtp_checkpoint

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="x-square-robot/X2-Turn-4B-0812",
        help="Hugging Face model ID or local checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model)
    model = load_mtp_checkpoint(
        args.model,
        device=args.device,
        dtype=DTYPES[args.dtype],
    ).eval()
    print(f"loaded {args.model} on {args.device} ({args.dtype})")
    print(f"processor={type(processor).__name__}")
    print(f"model={type(model).__name__}")
    print(f"turn_head={tuple(model.vad_lm_head.weight.shape)}")


if __name__ == "__main__":
    main()
