#!/usr/bin/env python3
"""Run local ASR and frame-level turn inference for one audio file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import (
    infer_asr_turn,
    load_mtp_checkpoint,
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hub model ID or local path")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="optional frame JSON output")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        help="target ASR delay; defaults to the checkpoint configuration",
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
    result = infer_asr_turn(
        model,
        processor,
        args.audio,
        delay_ms=args.delay_ms,
    )

    for frame in result.turn_frames:
        print(
            f"{frame.start_ms:>6}-{frame.end_ms:<6} ms "
            f"{frame.label:<12} p={frame.confidence:.3f}"
        )
    print(f"\ntranscript: {result.transcript}")

    if args.output is not None:
        payload = {
            "transcript": result.transcript,
            "frame_ms": result.frame_ms,
            "generated_token_ids": result.generated_token_ids,
            "frames": [
                {
                    "frame": frame.index,
                    "start_ms": frame.start_ms,
                    "end_ms": frame.end_ms,
                    "turn": frame.label,
                    "confidence": round(frame.confidence, 6),
                    "probabilities": {
                        name: round(probability, 6)
                        for name, probability in frame.probabilities.items()
                    },
                }
                for frame in result.turn_frames
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"frames: {args.output}")


if __name__ == "__main__":
    main()
