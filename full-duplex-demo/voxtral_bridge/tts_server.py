#!/usr/bin/env python3
"""Minimal TTS server compatible with X Square TTS client (POST /tts -> wav bytes).

Uses Microsoft Edge TTS (edge-tts) + bundled ffmpeg for mp3->wav conversion.
No GPU required.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import subprocess
import tempfile

import edge_tts
import imageio_ffmpeg
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

app = FastAPI(title="Dialogue Demo TTS")

# Chinese female neural voice; override with EDGE_TTS_VOICE
DEFAULT_VOICE = os.environ.get("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


async def synthesize_wav(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    text = (text or "").strip()
    if not text:
        return b""

    communicate = edge_tts.Communicate(text, voice)
    mp3_buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buf.write(chunk["data"])
    mp3_bytes = mp3_buf.getvalue()
    if not mp3_bytes:
        return b""

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f_mp3:
        f_mp3.write(mp3_bytes)
        mp3_path = f_mp3.name
    wav_path = mp3_path + ".wav"
    try:
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-i",
                mp3_path,
                "-ac",
                "1",
                "-ar",
                "24000",
                "-sample_fmt",
                "s16",
                wav_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        with open(wav_path, "rb") as f:
            return f.read()
    finally:
        for p in (mp3_path, wav_path):
            try:
                os.unlink(p)
            except OSError:
                pass


@app.get("/health")
def health():
    return {"ok": True, "voice": DEFAULT_VOICE}


@app.post("/tts")
async def tts(request: Request):
    data = await request.json()
    text = data.get("text", "")
    # X Square TTS client also sends "character"; ignore and use EDGE voice
    wav = await synthesize_wav(text)
    if not wav:
        return Response(content=b"", media_type="audio/wav", status_code=204)
    return Response(content=wav, media_type="audio/wav")


@app.post("/tts_stream")
async def tts_stream(request: Request):
    """Streaming synthesis: chunked raw PCM s16le mono 24 kHz.

    edge-tts mp3 chunks are piped into ffmpeg as they arrive; decoded PCM is
    yielded immediately, so first audio reaches the client ~1s earlier than
    the whole-file /tts endpoint.
    """
    data = await request.json()
    text = (data.get("text") or "").strip()
    voice = data.get("voice", DEFAULT_VOICE)
    if not text:
        return Response(content=b"", status_code=204)

    async def gen():
        proc = await asyncio.create_subprocess_exec(
            FFMPEG,
            "-loglevel", "quiet",
            "-i", "pipe:0",
            "-f", "s16le", "-ac", "1", "-ar", "24000",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )

        async def feeder():
            try:
                communicate = edge_tts.Communicate(text, voice)
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        proc.stdin.write(chunk["data"])
                        await proc.stdin.drain()
            except Exception as e:
                print(f"[TTS] stream feeder error: {e}", flush=True)
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        feed_task = asyncio.create_task(feeder())
        try:
            while True:
                buf = await proc.stdout.read(9600)  # 200 ms @ 24 kHz s16
                if not buf:
                    break
                yield buf
        finally:
            await feed_task
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    return StreamingResponse(gen(), media_type="application/octet-stream")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6016)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[TTS] voice={DEFAULT_VOICE} ffmpeg={FFMPEG}", flush=True)
    # Warm-up
    try:
        asyncio.get_event_loop().run_until_complete(synthesize_wav("你好"))
        print("[TTS] warm-up ok", flush=True)
    except Exception as e:
        print(f"[TTS] warm-up skipped: {e}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
