"""Lightweight prediction types shared by local and vLLM backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import soundfile as sf

TURN_CLASS_IDS = (35, 36, 37, 38, 39, 40)
TURN_CLASS_NAMES = (
    "idle",
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
)


@dataclass
class FramePred:
    frame: int
    t0: float
    t1: float
    asr: str
    turn: str
    turn_prob: float
    probs: dict[str, float]


@dataclass
class UtterancePred:
    wav_path: str
    asr_text: str
    seconds_per_token: float
    delay_ms: int
    turn_label_delay_frames: int
    frames: list[FramePred]
    duration_s: float


@dataclass
class StreamUpdate:
    kind: str  # partial | final
    asr_text: str
    last_turn: str
    duration_s: float
    n_frames: int
    turn_hist: dict[str, int]
    timeline_html: str
    frames_html: str
    turns: list[str]
    elapsed_infer_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load mono audio and resample without importing the local model stack."""
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if source_rate == sample_rate or audio.size == 0:
        return audio

    target_size = max(1, round(audio.size * sample_rate / source_rate))
    source_positions = np.arange(audio.size, dtype=np.float64) / source_rate
    target_positions = np.arange(target_size, dtype=np.float64) / sample_rate
    return np.interp(target_positions, source_positions, audio).astype(np.float32)
