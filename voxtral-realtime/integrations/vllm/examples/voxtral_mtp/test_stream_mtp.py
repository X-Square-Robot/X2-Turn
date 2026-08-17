#!/usr/bin/env python3
"""Minimal smoke client for turn.delta from the custom vLLM server."""

import argparse
import asyncio
import base64
import json
import wave

import numpy as np
import websockets


async def run(url: str, model: str, audio: str) -> None:
    with wave.open(audio, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError("audio must be mono 16-bit PCM WAV")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    async with websockets.connect(url, max_size=None) as ws:
        print(await ws.recv())
        await ws.send(json.dumps({"type": "session.update", "model": model}))
        for start in range(0, len(pcm), 1280):
            chunk = base64.b64encode(pcm[start : start + 1280].tobytes()).decode(
                "ascii"
            )
            await ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": chunk})
            )
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            for _ in range(2):
                try:
                    print(await asyncio.wait_for(ws.recv(), 0.1))
                except TimeoutError:
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8011/v1/realtime")
    parser.add_argument("--model", default="Kaiqfu/X2-Turn-4B-0812")
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.url, args.model, args.audio))
