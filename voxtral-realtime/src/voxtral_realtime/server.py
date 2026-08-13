"""FastAPI bridge from application audio to the vLLM realtime service."""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .audio import AcousticVoiceGate
from .config import RealtimeConfig
from .realtime import RealtimeVLLMSession
from .trace import JSONLTraceWriter
from .turn import FrameTurnConfig, FrameTurnController

logger = logging.getLogger(__name__)
SessionFactory = Callable[..., RealtimeVLLMSession]


class TurnSession:
    def __init__(self, bridge: TurnBridge, session_id: str) -> None:
        self.bridge, self.session_id = bridge, session_id
        self.inner: RealtimeVLLMSession | None = None
        self.controller = FrameTurnController(bridge.frame_config)
        cfg = bridge.config
        self.acoustic_gate = AcousticVoiceGate(
            cfg.sample_rate,
            cfg.acoustic_vad_rms_threshold,
            cfg.acoustic_vad_peak_threshold,
            cfg.acoustic_vad_hangover_ms,
        )
        self.bot_speaking = False
        self.trace_frame_index = 0
        self.trace_segment = 0
        self.last_active = time.monotonic()
        self.last_output: dict = {
            "state": "idle",
            "turn_class": "idle",
            "asr_segment": "",
            "asr_buffer": "",
        }

    async def ensure_inner(self) -> RealtimeVLLMSession:
        if self.inner is None:
            cfg = self.bridge.config
            self.inner = self.bridge.session_factory(
                vllm_url=cfg.vllm_url,
                model=cfg.model_id,
                bot_speaking=self.bot_speaking,
                commit_ms=cfg.commit_ms,
                barge_commit_ms=cfg.barge_commit_ms,
                sample_rate=cfg.sample_rate,
                turn_label_delay_frames=cfg.turn_label_delay_frames,
                lead_in_gate=cfg.lead_in_gate,
                lead_in_rms=cfg.lead_in_rms,
                lead_in_preroll_ms=cfg.lead_in_preroll_ms,
                lead_in_hold_ms=cfg.lead_in_hold_ms,
                suppress_idle_text=cfg.suppress_idle_text,
            )
            await self.inner.connect()
        return self.inner

    def set_bot_speaking(self, speaking: bool) -> None:
        changed = self.bot_speaking != bool(speaking)
        self.bot_speaking = bool(speaking)
        self.controller.set_bot_speaking(self.bot_speaking)
        if self.inner is not None:
            self.inner.set_bot_speaking(self.bot_speaking)
        if changed:
            self.bridge.trace(
                "control",
                self.session_id,
                segment=self.trace_segment,
                bot_speaking=self.bot_speaking,
            )

    async def reset(self, reason: str = "reset") -> None:
        if self.inner is not None:
            with suppress(Exception):
                await self.inner.close()
        self.inner = None
        self.controller.reset()
        self.acoustic_gate.reset()
        self.bridge.trace(
            "session_reset",
            self.session_id,
            segment=self.trace_segment,
            reason=reason,
        )
        self.trace_segment += 1
        self.trace_frame_index = 0

    async def feed(self, pcm: np.ndarray, bot_speaking: bool | None = None) -> dict:
        self.last_active = time.monotonic()
        if bot_speaking is not None:
            self.set_bot_speaking(bot_speaking)
        acoustic_active = self.acoustic_gate.update(pcm)
        inner = await self.ensure_inner()
        await inner.push_pcm(pcm)
        output = None
        accepted = False
        trace_records: list[dict] = []
        for frame in inner.consume_turn_frames():
            raw_turn = frame.get("turn") or "idle"
            frame_index = frame.get("frame_index")
            if frame_index is None:
                frame_index = self.trace_frame_index
            frame_index = int(frame_index)
            self.trace_frame_index = max(self.trace_frame_index + 1, frame_index + 1)
            output = self.controller.on_frame(
                raw_turn,
                inner.asr_text,
                frame_index,
                acoustic_active,
            )
            probabilities = frame.get("probs")
            if isinstance(probabilities, (list, tuple)):
                probabilities = [float(value) for value in probabilities]
            else:
                probabilities = None
            trace_records.append(
                {
                    "segment": self.trace_segment,
                    "frame_index": frame_index,
                    "turn_class": raw_turn,
                    "turn_id": frame.get("turn_id"),
                    "probabilities": probabilities,
                    "asr_buffer": inner.asr_text,
                    "bot_speaking": self.bot_speaking,
                    "acoustic_active": acoustic_active,
                    "acoustic_rms": round(self.acoustic_gate.last_rms, 5),
                    "acoustic_peak": round(self.acoustic_gate.last_peak, 5),
                    "dialogue_output": dict(output),
                }
            )
            if output.get("state") == "speak":
                text = (output.get("text") or inner.asr_text).strip()
                reason = output.get("reason", "")
                output = {
                    "state": "speak",
                    "turn_class": "turn_end",
                    "event": "accept",
                    "text": text,
                    "asr_segment": "",
                    "asr_buffer": text,
                    "reason": reason,
                }
                accepted = True
                break
        if output is None:
            state = self.controller.st
            output = {
                "state": state.last_state,
                "turn_class": state.last_turn,
                "asr_segment": "",
                "asr_buffer": inner.asr_text,
                "acoustic_active": acoustic_active,
                "acoustic_rms": round(self.acoustic_gate.last_rms, 5),
            }
            if state.last_reason:
                output["note"] = state.last_reason
        stop_tts = bool(
            trace_records
            and self.bot_speaking
            and output.get("state") == "nonidle"
            and output.get("turn_class") == "speaking"
        )
        for index, record in enumerate(trace_records):
            self.bridge.trace(
                "frame",
                self.session_id,
                **record,
                dialogue_stop_tts=stop_tts and index == len(trace_records) - 1,
            )
        if accepted:
            await self.reset(reason="accept")
        self.last_output = output
        return dict(output)


class TurnBridge:
    def __init__(
        self,
        config: RealtimeConfig | None = None,
        session_factory: SessionFactory = RealtimeVLLMSession,
    ) -> None:
        self.config = config or RealtimeConfig.from_env()
        self.session_factory = session_factory
        cfg = self.config
        self.frame_config = FrameTurnConfig(
            end_confirm_frames=cfg.end_confirm_frames,
            silence_end_frames=cfg.silence_end_frames,
            min_asr_chars=cfg.min_asr_chars,
            tail_min_frames=cfg.tail_min_frames,
            tail_max_frames=cfg.tail_max_frames,
            tail_stable_frames=cfg.tail_stable_frames,
            backchannel_confirm_frames=cfg.backchannel_confirm_frames,
            backchannel_fallback_idle_frames=cfg.backchannel_fallback_idle_frames,
            short_asr_chars=cfg.short_asr_chars,
            short_tail_min_frames=cfg.short_tail_min_frames,
            short_tail_max_frames=cfg.short_tail_max_frames,
            acoustic_vad_max_hold_frames=cfg.acoustic_vad_max_hold_frames,
        )
        self.trace_writer = (
            JSONLTraceWriter(cfg.trace_jsonl) if cfg.trace_jsonl.strip() else None
        )
        self.sessions: dict[str, TurnSession] = {}

    def get_session(self, session_id: str) -> TurnSession:
        session = self.sessions.get(session_id)
        if session is None:
            session = TurnSession(self, session_id)
            self.sessions[session_id] = session
            self.trace(
                "session_start",
                session_id,
                segment=session.trace_segment,
                model=self.config.model_id,
            )
        return session

    def trace(self, record_type: str, session_id: str, **fields) -> None:
        if self.trace_writer is not None:
            try:
                self.trace_writer.write(record_type, session_id, **fields)
            except (OSError, TypeError, ValueError):
                logger.exception("disabling turn trace after write failure")
                self.trace_writer = None

    async def gc(self) -> None:
        now = time.monotonic()
        expired = [
            key
            for key, value in self.sessions.items()
            if now - value.last_active > self.config.session_ttl_sec
        ]
        for key in expired:
            await self.sessions.pop(key).reset(reason="ttl")


async def _gc_loop(bridge: TurnBridge) -> None:
    while True:
        await asyncio.sleep(bridge.config.gc_interval_sec)
        await bridge.gc()


def create_app(bridge: TurnBridge | None = None) -> FastAPI:
    bridge = bridge or TurnBridge()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = asyncio.create_task(_gc_loop(bridge))
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        for session in list(bridge.sessions.values()):
            await session.reset(reason="shutdown")

    app = FastAPI(title="Voxtral Realtime", version="0.1.0", lifespan=lifespan)
    app.state.bridge = bridge

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "backend": "vllm",
            "sessions": len(bridge.sessions),
            "model": bridge.config.model_id,
            "vllm_url": bridge.config.vllm_url,
        }

    @app.websocket("/turn")
    async def turn_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_json()
                session_id = str(data.get("session_id") or "")
                if not session_id:
                    await websocket.send_json(
                        {"type": "error", "error": "session_id is required"}
                    )
                    continue
                session = bridge.get_session(session_id)
                if data.get("type") == "control":
                    if "bot_speaking" in data:
                        session.set_bot_speaking(bool(data["bot_speaking"]))
                    await websocket.send_json(
                        {"type": "control.ack", "session_id": session_id}
                    )
                    continue
                if data.get("type") != "audio":
                    await websocket.send_json(
                        {"type": "error", "error": "unsupported message type"}
                    )
                    continue
                try:
                    pcm = np.frombuffer(
                        base64.b64decode(data["audio"], validate=True), dtype=np.float32
                    )
                    state = await session.feed(pcm, data.get("bot_speaking"))
                except (KeyError, ValueError) as exc:
                    await websocket.send_json(
                        {"type": "error", "error": f"invalid audio: {exc}"}
                    )
                    continue
                await websocket.send_json(
                    {
                        "type": "turn_state",
                        "session_id": session_id,
                        "state": state,
                        "ts": time.time(),
                    }
                )
        except WebSocketDisconnect:
            logger.debug("turn websocket disconnected")

    return app
