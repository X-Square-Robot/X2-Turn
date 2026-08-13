"""Self-contained client for the vLLM ``/v1/realtime`` WebSocket API."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any

import numpy as np
import websockets

SPEECH_TURNS = frozenset({"noidle", "speaking", "turn_end", "backchannel", "uncertain"})


class RealtimeVLLMSession:
    """Stream float32 PCM and expose gated ASR text plus raw turn frames."""

    def __init__(
        self,
        *,
        vllm_url: str,
        model: str,
        bot_speaking: bool = False,
        commit_ms: int = 80,
        barge_commit_ms: int = 80,
        sample_rate: int = 16000,
        turn_label_delay_frames: int = 0,
        lead_in_gate: bool = True,
        lead_in_rms: float = 0.012,
        lead_in_preroll_ms: int = 320,
        lead_in_hold_ms: int = 80,
        suppress_idle_text: bool = True,
    ) -> None:
        self.vllm_url, self.model = vllm_url, model
        self.sample_rate = int(sample_rate)
        self.turn_label_delay_frames = max(0, int(turn_label_delay_frames))
        self.normal_commit_ms = max(80, int(commit_ms))
        self.barge_commit_ms = max(80, int(barge_commit_ms))
        self.bot_speaking = bool(bot_speaking)
        self.commit_ms = (
            self.barge_commit_ms if self.bot_speaking else self.normal_commit_ms
        )
        self.lead_in_gate, self.lead_in_rms = bool(lead_in_gate), float(lead_in_rms)
        self.preroll_limit = int(self.sample_rate * lead_in_preroll_ms / 1000)
        self.hold_samples = int(self.sample_rate * lead_in_hold_ms / 1000)
        self.suppress_idle_text = bool(suppress_idle_text)
        self._ws: Any = None
        self._reader: asyncio.Task[None] | None = None
        self._parts: list[str] = []
        self._turn_frames: list[dict[str, Any]] = []
        self._consumed_frames = 0
        self._pending_text = ""
        self._speech_opened = False
        self._gate_open = self.bot_speaking or not self.lead_in_gate
        self._preroll: list[np.ndarray] = []
        self._preroll_n = self._speech_run = self._sent_samples = 0
        self._last_poll = 0.0
        self._last_observed_frames = 0
        self._done = asyncio.Event()
        self._error: BaseException | None = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.vllm_url, max_size=None)
        created = json.loads(await self._ws.recv())
        if created.get("type") != "session.created":
            await self.close()
            raise RuntimeError(f"unexpected realtime handshake: {created}")
        await self._ws.send(json.dumps({"type": "session.update", "model": self.model}))
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self._reader = asyncio.create_task(self._recv_loop())

    def set_bot_speaking(self, speaking: bool) -> None:
        speaking = bool(speaking)
        if speaking == self.bot_speaking:
            return
        self.bot_speaking = speaking
        self.commit_ms = (
            self.barge_commit_ms if self.bot_speaking else self.normal_commit_ms
        )
        if self.bot_speaking:
            self._gate_open = True
            self._preroll.clear()
            self._preroll_n = self._speech_run = 0
            self._last_poll = 0.0

    def handle_message(self, data: dict[str, Any]) -> None:
        """Apply one decoded server event; public to support deterministic tests."""
        kind = data.get("type")
        if kind == "transcription.delta":
            self._pending_text = data.get("delta") or ""
        elif kind == "turn.delta":
            turn = data.get("turn_class") or "idle"
            text = self._pending_text
            if (
                not self.suppress_idle_text
                or self._speech_opened
                or turn in SPEECH_TURNS
            ):
                self._speech_opened = True
                self._parts.append(text)
            self._turn_frames.append(
                {
                    "turn": turn,
                    "turn_id": int(data.get("turn_id") or 35),
                    "probs": data.get("probs"),
                    "frame_index": data.get("frame_index"),
                    "text": text if self._speech_opened else "",
                }
            )
            self._pending_text = ""
        elif kind == "transcription.done":
            if not self._parts and data.get("text"):
                self._parts.append(str(data["text"]))
            self._done.set()
        elif kind == "error":
            self._error = RuntimeError(str(data))
            self._done.set()

    async def _recv_loop(self) -> None:
        try:
            async for message in self._ws:
                self.handle_message(json.loads(message))
                if self._done.is_set():
                    return
        except Exception as exc:
            self._error = exc
            self._done.set()

    async def _send_pcm(self, pcm: np.ndarray) -> None:
        encoded = np.clip(pcm * 32767.0, -32768, 32767).astype(np.int16)
        self._sent_samples += int(pcm.size)
        await self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(encoded.tobytes()).decode("ascii"),
                }
            )
        )

    async def push_pcm(self, pcm: np.ndarray) -> None:
        self._raise_if_error()
        if self._ws is None:
            raise RuntimeError("session is not connected")
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if not samples.size:
            return
        if not self._gate_open:
            self._preroll.append(samples.copy())
            self._preroll_n += samples.size
            while self._preroll_n > self.preroll_limit and len(self._preroll) > 1:
                self._preroll_n -= self._preroll.pop(0).size
            rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
            self._speech_run = (
                self._speech_run + samples.size if rms >= self.lead_in_rms else 0
            )
            if self._speech_run < self.hold_samples:
                return
            self._gate_open = True
            samples = np.concatenate(self._preroll)
            self._preroll.clear()
            self._preroll_n = 0
        await self._send_pcm(samples)
        now = time.monotonic()
        if self._sent_samples < self.sample_rate // 4:
            return
        if (now - self._last_poll) * 1000 < self.commit_ms:
            return

        # A non-final commit starts generation once during connect(). Further
        # commits are ignored by the pinned vLLM realtime server and only add
        # log noise. Instead, briefly yield here so turn.delta events produced
        # from the appended audio are visible to the bridge in this call.
        deadline = time.monotonic() + min(self.commit_ms / 1000.0, 0.35)
        while (
            time.monotonic() < deadline
            and len(self._turn_frames) <= self._last_observed_frames
        ):
            self._raise_if_error()
            await asyncio.sleep(0.01)
        self._last_observed_frames = len(self._turn_frames)
        self._last_poll = time.monotonic()

    async def finish(self, timeout: float = 60.0) -> str:
        if not self._gate_open and self._preroll:
            await self._send_pcm(np.concatenate(self._preroll))
            self._preroll.clear()
        await self._ws.send(
            json.dumps({"type": "input_audio_buffer.commit", "final": True})
        )
        await asyncio.wait_for(self._done.wait(), timeout=timeout)
        self._raise_if_error()
        return self.asr_text

    async def close(self) -> None:
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
        self._reader = None
        if self._ws is not None:
            await self._ws.close()
        self._ws = None

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    @property
    def asr_text(self) -> str:
        return "".join(self._parts)

    def consume_turn_frames(self) -> list[dict[str, Any]]:
        rows = self._turn_frames[self._consumed_frames :]
        self._consumed_frames = len(self._turn_frames)
        return list(rows)
