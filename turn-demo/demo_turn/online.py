"""Online streaming session: microphone/chunked PCM → incremental ASR + turn states.

Implementation notes
--------------------
This module is the demo path for the local Transformers backend. It uses:

  **chunked ingestion with the HF ONLINE processor + incremental decoding of
  the accumulated buffer**
  (suitable for observing raw ASR and Turn states for short utterances, with
  semantics consistent with offline inference)

Every ``commit_ms`` (320 ms by default), ``engine.infer_wav`` runs on the
current buffer and pushes new ASR and turn states to the frontend.

For production streaming, use vLLM ``/v1/realtime`` with the X2 Turn overlay.
The corresponding implementation is in ``online_vllm.py`` and can receive all
six ``turn.delta`` classes directly.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import numpy as np

from demo_turn.engine import TurnDemoEngine, UtterancePred
from demo_turn.predictions import StreamUpdate
from demo_turn.viz import frame_table_html, timeline_html


class OnlineTurnSession:
    """Single session: push_pcm → optional StreamUpdate; finish → final update."""

    def __init__(
        self,
        engine: TurnDemoEngine,
        commit_ms: int = 320,
        max_buffer_s: float = 20.0,
        lock: Optional[threading.Lock] = None,
    ):
        self.engine = engine
        self.sr = int(engine.sr)
        self.commit_samples = max(
            int(self.sr * commit_ms / 1000.0), int(self.sr * 0.08)
        )
        self.max_buffer_samples = int(self.sr * max_buffer_s)
        self.lock = lock

        self._chunks: List[np.ndarray] = []
        self._n_samples = 0
        self._since_commit = 0
        self._started = time.time()
        self.last_update: Optional[StreamUpdate] = None

    def _wav(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks, axis=0)

    def push_pcm(self, pcm: np.ndarray, force: bool = False) -> Optional[StreamUpdate]:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return None
        self._chunks.append(pcm)
        self._n_samples += int(pcm.size)
        self._since_commit += int(pcm.size)

        # Use a rolling window to avoid repeatedly decoding long utterances in full.
        if self._n_samples > self.max_buffer_samples:
            wav = self._wav()[-self.max_buffer_samples :]
            self._chunks = [wav]
            self._n_samples = int(wav.size)

        if (not force) and self._since_commit < self.commit_samples:
            return None
        if self._n_samples < int(self.sr * 0.25):
            return None
        return self._decode(kind="partial")

    def finish(self) -> StreamUpdate:
        # Append silence to help flush delayed ASR and turn output.
        pad_s = max(self.engine.delay_ms / 1000.0, 0.48)
        if self.engine.turn_delay > 0:
            pad_s += self.engine.turn_delay * self.engine.seconds_per_token
        pad = np.zeros(int(self.sr * pad_s), dtype=np.float32)
        self._chunks.append(pad)
        self._n_samples += int(pad.size)
        return self._decode(kind="final")

    def _decode(self, kind: str) -> StreamUpdate:
        self._since_commit = 0
        wav = self._wav()
        t0 = time.time()
        if self.lock is not None:
            with self.lock:
                pred = self.engine.infer_wav(wav, wav_path="<online>")
        else:
            pred = self.engine.infer_wav(wav, wav_path="<online>")
        infer_ms = (time.time() - t0) * 1000.0
        update = self._pack(pred, kind=kind, infer_ms=infer_ms)
        self.last_update = update
        return update

    def _pack(self, pred: UtterancePred, kind: str, infer_ms: float) -> StreamUpdate:
        turns = [f.turn for f in pred.frames]
        hist = {}
        for t in turns:
            hist[t] = hist.get(t, 0) + 1
        return StreamUpdate(
            kind=kind,
            asr_text=pred.asr_text,
            last_turn=turns[-1] if turns else "idle",
            duration_s=pred.duration_s,
            n_frames=len(pred.frames),
            turn_hist=hist,
            timeline_html=timeline_html(
                turns,
                seconds_per_token=pred.seconds_per_token,
            ),
            frames_html=frame_table_html(pred.frames),
            turns=turns,
            elapsed_infer_ms=round(infer_ms, 1),
        )


def chunk_wav_for_online(
    wav: np.ndarray,
    sr: int,
    chunk_ms: int = 80,
):
    """Split a WAV array into chunks for online pushes (tests/playback)."""
    n = int(sr * chunk_ms / 1000.0)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    for i in range(0, len(wav), n):
        yield wav[i : i + n]
