"""Online streaming session over vLLM ``/v1/realtime`` (ASR + turn.delta).

Browser still talks to ``demo_turn.server`` WebSocket; this class forwards PCM
to vLLM and packs the same ``StreamUpdate`` shape as the HF online path.

vLLM ASR deltas are irreversible. Leading mic silence / early idle frames often
hallucinate; this session applies two mitigations by default:

1. **Lead-in energy gate** — do not forward PCM until short speech is detected
   (keeps a small preroll so the true onset is not clipped).
2. **Idle-text suppress** — do not accumulate ``transcription.delta`` text until
   the first non-``idle`` ``turn.delta`` (drops pre-speech junk tokens).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import List, Optional, Set

import numpy as np
import websockets

from demo_turn.engine_vllm import (
    DEFAULT_SECONDS_PER_TOKEN,
    DEFAULT_VLLM_URL,
    frames_to_utterance,
)
from demo_turn.predictions import StreamUpdate, UtterancePred
from demo_turn.viz import frame_table_html, timeline_html

# Turns that mean "user has started speaking" for text gating.
_SPEECH_TURNS: Set[str] = {
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
}


class OnlineVLLMSession:
    """Async online session: push float32 PCM → StreamUpdate via vLLM."""

    def __init__(
        self,
        *,
        vllm_url: str = DEFAULT_VLLM_URL,
        model: str,
        commit_ms: int = 320,
        sr: int = 16000,
        seconds_per_token: float = DEFAULT_SECONDS_PER_TOKEN,
        delay_ms: int = 480,
        turn_label_delay_frames: int = 0,
        lead_in_gate: bool = True,
        lead_in_rms: float = 0.012,
        lead_in_preroll_ms: int = 320,
        lead_in_hold_ms: int = 80,
        suppress_idle_text: bool = True,
    ):
        self.vllm_url = vllm_url
        self.model = model
        self.sr = int(sr)
        self.seconds_per_token = float(seconds_per_token)
        self.delay_ms = int(delay_ms)
        self.turn_delay = max(0, int(turn_label_delay_frames))
        self.commit_ms = max(int(commit_ms), 80)

        self.lead_in_gate = bool(lead_in_gate)
        self.lead_in_rms = float(lead_in_rms)
        self.lead_in_preroll_samples = int(self.sr * lead_in_preroll_ms / 1000.0)
        self.lead_in_hold_samples = int(self.sr * lead_in_hold_ms / 1000.0)
        self.suppress_idle_text = bool(suppress_idle_text)

        self._ws = None
        self._reader: Optional[asyncio.Task] = None
        self._parts: List[str] = []
        self._turn_frames: List[dict] = []
        self._pending_text = ""
        self._speech_opened = False  # first non-idle turn seen
        self._done = asyncio.Event()
        self._error: Optional[BaseException] = None
        self._n_samples = 0  # samples actually sent to vLLM
        self._mic_samples = 0  # all mic samples (incl. gated)
        self._started = time.time()
        self._last_pack_t = 0.0
        self._last_n_frames = 0
        self._consumed_frames = 0
        self.last_update: Optional[StreamUpdate] = None

        self._gate_open = not self.lead_in_gate
        self._preroll: List[np.ndarray] = []
        self._preroll_n = 0
        self._speech_run = 0

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.vllm_url, max_size=None)
        created = json.loads(await self._ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"unexpected handshake: {created}")
        await self._ws.send(json.dumps({"type": "session.update", "model": self.model}))
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self._reader = asyncio.create_task(self._recv_loop())
        self._started = time.time()
        self._last_pack_t = 0.0

    def _accept_text(self, text: str, turn: str) -> None:
        """Accumulate ASR text with optional idle-leading suppress."""
        if not self.suppress_idle_text:
            self._parts.append(text)
            return
        if not self._speech_opened:
            if turn in _SPEECH_TURNS:
                self._speech_opened = True
                if text:
                    self._parts.append(text)
            # else: drop pre-speech / idle hallucination
            return
        self._parts.append(text)

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                data = json.loads(message)
                mtype = data.get("type")
                if mtype == "transcription.delta":
                    # Defer append until paired turn.delta (know idle vs speech).
                    self._pending_text = data.get("delta") or ""
                elif mtype == "turn.delta":
                    turn = data.get("turn_class") or "idle"
                    text = self._pending_text
                    self._accept_text(text, turn)
                    self._turn_frames.append(
                        {
                            "turn": turn,
                            "turn_id": int(data.get("turn_id") or 35),
                            "probs": data.get("probs"),
                            "frame_index": data.get("frame_index"),
                            "text": text
                            if (self._speech_opened or not self.suppress_idle_text)
                            else "",
                        }
                    )
                    self._pending_text = ""
                elif mtype == "transcription.done":
                    # Prefer our gated stream; only fall back if empty.
                    if not self._parts and data.get("text"):
                        self._parts.append(data["text"])
                    self._done.set()
                    return
                elif mtype == "error":
                    self._error = RuntimeError(data)
                    self._done.set()
                    return
        except Exception as e:  # noqa: BLE001 — surface to finish/push
            self._error = e
            self._done.set()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    async def _send_pcm(self, pcm: np.ndarray) -> None:
        assert self._ws is not None
        i16 = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
        self._n_samples += int(pcm.size)
        await self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(i16.tobytes()).decode(),
                }
            )
        )

    def _trim_preroll(self) -> None:
        while self._preroll_n > self.lead_in_preroll_samples and self._preroll:
            drop = self._preroll.pop(0)
            self._preroll_n -= int(drop.size)

    async def push_pcm(self, pcm: np.ndarray) -> Optional[StreamUpdate]:
        self._raise_if_error()
        assert self._ws is not None
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return None
        self._mic_samples += int(pcm.size)

        if not self._gate_open:
            self._preroll.append(pcm.copy())
            self._preroll_n += int(pcm.size)
            self._trim_preroll()
            rms = float(np.sqrt(np.mean(pcm * pcm) + 1e-12))
            if rms >= self.lead_in_rms:
                self._speech_run += int(pcm.size)
            else:
                self._speech_run = 0
            if self._speech_run < self.lead_in_hold_samples:
                return None
            self._gate_open = True
            # Flush preroll (includes onset) then continue with current already in preroll.
            flush = (
                np.concatenate(self._preroll, axis=0)
                if self._preroll
                else np.zeros(0, dtype=np.float32)
            )
            self._preroll.clear()
            self._preroll_n = 0
            if flush.size:
                await self._send_pcm(flush)
        else:
            await self._send_pcm(pcm)

        now = time.time()
        if (now - self._last_pack_t) * 1000.0 < self.commit_ms:
            return None
        if self._n_samples < int(self.sr * 0.25):
            return None
        # Burst upload may outrun the engine; wait briefly for new frames.
        deadline = time.time() + min(self.commit_ms / 1000.0, 0.35)
        while time.time() < deadline and len(self._turn_frames) <= self._last_n_frames:
            self._raise_if_error()
            await asyncio.sleep(0.01)
        if len(self._turn_frames) <= self._last_n_frames:
            return None
        return self._pack(kind="partial")

    async def finish(self) -> StreamUpdate:
        self._raise_if_error()
        assert self._ws is not None
        # If user never crossed the energy gate, still flush preroll so final
        # commit is not empty (offline-style short clip / whisper).
        if not self._gate_open and self._preroll:
            flush = np.concatenate(self._preroll, axis=0)
            self._preroll.clear()
            self._preroll_n = 0
            self._gate_open = True
            await self._send_pcm(flush)
        await self._ws.send(
            json.dumps({"type": "input_audio_buffer.commit", "final": True})
        )
        try:
            await asyncio.wait_for(self._done.wait(), timeout=60.0)
        except asyncio.TimeoutError as e:
            raise TimeoutError("vLLM transcription.done timed out") from e
        self._raise_if_error()
        upd = self._pack(kind="final")
        await self.close()
        return upd

    async def close(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
            self._reader = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    @property
    def asr_text(self) -> str:
        return "".join(self._parts)

    def consume_turn_frames(self) -> List[dict]:
        """Return newly received raw turn.delta rows (may include frame_index)."""
        new = self._turn_frames[self._consumed_frames :]
        self._consumed_frames = len(self._turn_frames)
        return list(new)

    def _pack(self, kind: str) -> StreamUpdate:
        t0 = time.time()
        duration_s = self._mic_samples / float(self.sr)
        pred: UtterancePred = frames_to_utterance(
            wav_path="<online-vllm>",
            asr_text="".join(self._parts),
            turn_frames=list(self._turn_frames),
            duration_s=duration_s,
            seconds_per_token=self.seconds_per_token,
            delay_ms=self.delay_ms,
            turn_label_delay_frames=self.turn_delay,
        )
        turns = [f.turn for f in pred.frames]
        hist = {}
        for t in turns:
            hist[t] = hist.get(t, 0) + 1
        infer_ms = (time.time() - t0) * 1000.0
        wall_ms = (time.time() - self._started) * 1000.0
        upd = StreamUpdate(
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
            elapsed_infer_ms=round(
                max(infer_ms, wall_ms / max(len(pred.frames), 1)), 1
            ),
        )
        self.last_update = upd
        self._last_pack_t = time.time()
        self._last_n_frames = len(self._turn_frames)
        return upd
