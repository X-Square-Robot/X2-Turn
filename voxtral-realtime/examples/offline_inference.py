#!/usr/bin/env python3
"""Replay one PCM WAV through Voxtral ASR and turn-taking inference."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from pathlib import Path

from voxtral_realtime.config import RealtimeConfig
from voxtral_realtime.offline import infer_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True, help="input PCM WAV")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("offline_output"),
        help="directory for states.json and transcript.json",
    )
    parser.add_argument("--chunk-ms", type=int, default=80)
    parser.add_argument("--flush-ms", type=int, default=1200)
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="pace chunks at their original wall-clock rate",
    )
    parser.add_argument(
        "--bot-speaking",
        action="store_true",
        help="evaluate the input as an interruption while the bot is speaking",
    )
    parser.add_argument("--vllm-url")
    parser.add_argument("--model")
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    config = RealtimeConfig.from_env()
    config = replace(
        config,
        **{
            key: value
            for key, value in {
                "vllm_url": args.vllm_url,
                "model_id": args.model,
            }.items()
            if value is not None
        },
    )
    states, transcripts = await infer_wav(
        args.audio,
        output_dir=args.output_dir,
        config=config,
        chunk_ms=args.chunk_ms,
        flush_ms=args.flush_ms,
        realtime=args.realtime,
        bot_speaking=args.bot_speaking,
    )
    print(f"processed {args.audio}")
    print(f"states: {len(states)} -> {args.output_dir / 'states.json'}")
    print(f"accepted turns: {len(transcripts)}")
    for item in transcripts:
        print(f"[{item['time_ms'] / 1000:.2f}s] {item['text']}")


if __name__ == "__main__":
    asyncio.run(run())
