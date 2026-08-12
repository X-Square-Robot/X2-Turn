"""Utilities for replaying PCM WAV files through the realtime turn stack."""

from __future__ import annotations

import asyncio
import json
import wave
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import RealtimeConfig
from .realtime import RealtimeVLLMSession
from .server import SessionFactory, TurnBridge


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        return (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
    if sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        values = np.where(values & 0x800000, values - 0x1000000, values)
        return values.astype(np.float32) / 8388608
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
    raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")


def resample_linear(
    samples: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    """Resample a mono float32 signal without adding an audio dependency."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or samples.size == 0:
        return samples.copy()
    if samples.size == 1:
        return np.repeat(samples, max(1, round(target_rate / source_rate)))

    output_size = max(1, round(samples.size * target_rate / source_rate))
    positions = np.arange(output_size, dtype=np.float64) * source_rate / target_rate
    return np.interp(positions, np.arange(samples.size), samples).astype(np.float32)


def load_pcm_wav(path: str | Path, target_rate: int = 16000) -> np.ndarray:
    """Load an uncompressed PCM WAV, mix channels, and resample to target_rate."""
    path = Path(path)
    with wave.open(str(path), "rb") as reader:
        if reader.getcomptype() != "NONE":
            raise ValueError(f"{path} is compressed; use an uncompressed PCM WAV")
        channels = reader.getnchannels()
        source_rate = reader.getframerate()
        sample_width = reader.getsampwidth()
        raw = reader.readframes(reader.getnframes())

    if channels <= 0:
        raise ValueError(f"{path} has no audio channels")
    samples = _decode_pcm(raw, sample_width)
    if samples.size % channels:
        raise ValueError(f"{path} contains an incomplete PCM frame")
    samples = samples.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return resample_linear(samples, source_rate, target_rate)


def iter_chunks(samples: np.ndarray, chunk_samples: int) -> Iterator[np.ndarray]:
    """Yield fixed-size float32 chunks, padding the last chunk with silence."""
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    for start in range(0, samples.size, chunk_samples):
        chunk = samples[start : start + chunk_samples]
        if chunk.size < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - chunk.size))
        yield chunk.astype(np.float32, copy=False)


async def infer_samples(
    samples: np.ndarray,
    config: RealtimeConfig,
    *,
    chunk_ms: int = 80,
    flush_ms: int = 1200,
    realtime: bool = False,
    bot_speaking: bool = False,
    session_factory: SessionFactory = RealtimeVLLMSession,
    sleep: Callable[[float], Any] = asyncio.sleep,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay samples through the same TurnSession policy used by the server."""
    if chunk_ms <= 0 or flush_ms < 0:
        raise ValueError("chunk_ms must be positive and flush_ms cannot be negative")
    chunk_samples = max(1, round(config.sample_rate * chunk_ms / 1000))
    flush_samples = round(config.sample_rate * flush_ms / 1000)
    source = np.asarray(samples, dtype=np.float32).reshape(-1)
    replay = np.concatenate([source, np.zeros(flush_samples, dtype=np.float32)])

    bridge = TurnBridge(config, session_factory=session_factory)
    session = bridge.get_session("offline")
    states: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []

    try:
        for index, chunk in enumerate(iter_chunks(replay, chunk_samples)):
            state = await session.feed(chunk, bot_speaking=bot_speaking)
            timestamp_ms = min(
                (index + 1) * chunk_ms,
                round(replay.size * 1000 / config.sample_rate),
            )
            row = {"chunk_index": index, "time_ms": timestamp_ms, **state}
            states.append(row)
            if state.get("state") == "speak" and (state.get("text") or "").strip():
                transcripts.append(
                    {
                        "time_ms": timestamp_ms,
                        "text": state["text"].strip(),
                        "event": state.get("event"),
                        "reason": state.get("reason"),
                    }
                )
                if index * chunk_samples >= source.size:
                    break
            # The realtime server emits frames asynchronously. Even an offline
            # replay must yield for at least one polling interval; --realtime
            # additionally preserves the source chunk's wall-clock duration.
            delay = (
                chunk_samples / config.sample_rate
                if realtime
                else config.commit_ms / 1000
            )
            await sleep(delay)
    finally:
        await session.reset()

    return states, transcripts


async def infer_wav(
    audio_path: str | Path,
    *,
    output_dir: str | Path,
    config: RealtimeConfig | None = None,
    chunk_ms: int = 80,
    flush_ms: int = 1200,
    realtime: bool = False,
    bot_speaking: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run offline inference for one WAV and write JSON artifacts."""
    config = config or RealtimeConfig.from_env()
    config = replace(config, sample_rate=16000)
    samples = load_pcm_wav(audio_path, config.sample_rate)
    states, transcripts = await infer_samples(
        samples,
        config,
        chunk_ms=chunk_ms,
        flush_ms=flush_ms,
        realtime=realtime,
        bot_speaking=bot_speaking,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "states.json").write_text(
        json.dumps(states, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "transcript.json").write_text(
        json.dumps(transcripts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return states, transcripts
