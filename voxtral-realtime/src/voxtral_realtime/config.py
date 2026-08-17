"""Environment-backed runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, get_type_hints

DEFAULT_MODEL_ID = "Kaiqfu/X2-Turn-4B-0812"
DEFAULT_VLLM_URL = "ws://127.0.0.1:8011/v1/realtime"


@dataclass(frozen=True)
class RealtimeConfig:
    model_id: str = DEFAULT_MODEL_ID
    vllm_url: str = DEFAULT_VLLM_URL
    host: str = "127.0.0.1"
    port: int = 8000
    sample_rate: int = 16000
    commit_ms: int = 80
    barge_commit_ms: int = 80
    turn_label_delay_frames: int = 0
    lead_in_gate: bool = True
    lead_in_rms: float = 0.012
    lead_in_preroll_ms: int = 320
    lead_in_hold_ms: int = 80
    suppress_idle_text: bool = True
    trace_jsonl: str = ""
    session_ttl_sec: float = 60.0
    gc_interval_sec: float = 10.0
    end_confirm_frames: int = 1
    silence_end_frames: int = 3
    min_asr_chars: int = 1
    tail_min_frames: int = 2
    tail_max_frames: int = 5
    tail_stable_frames: int = 2
    backchannel_confirm_frames: int = 2
    backchannel_fallback_idle_frames: int = 3
    short_asr_chars: int = 4
    short_tail_min_frames: int = 4
    short_tail_max_frames: int = 7
    acoustic_vad_rms_threshold: float = 0.010
    acoustic_vad_peak_threshold: float = 0.050
    acoustic_vad_hangover_ms: int = 200
    acoustic_vad_max_hold_frames: int = 8

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> RealtimeConfig:
        env = os.environ if environ is None else environ
        hints = get_type_hints(cls)
        aliases = {"model_id": "VOXTRAL_MODEL_ID", "vllm_url": "VLLM_URL"}
        values: dict[str, Any] = {}
        defaults = cls()
        for item in fields(cls):
            key = aliases.get(item.name, "VOXTRAL_" + item.name.upper())
            if key not in env:
                continue
            raw = env[key]
            typ = hints[item.name]
            if typ is bool:
                values[item.name] = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif typ in {int, float, str}:
                values[item.name] = typ(raw)
            else:
                values[item.name] = raw
        return cls(**{**defaults.__dict__, **values})
