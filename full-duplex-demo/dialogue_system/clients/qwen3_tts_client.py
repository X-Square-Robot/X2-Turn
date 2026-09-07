"""Native Qwen3TTS-Streaming client with one engine session per dialogue turn."""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import uuid

client_src = os.environ.get("QWEN3TTS_CLIENT_SRC", "").strip()
if client_src and client_src not in sys.path:
    sys.path.insert(0, client_src)

try:
    from qwen3tts import (
        AudioChunk,
        AudioFormat,
        SessionStartRequest,
        StreamEvent,
        SynthesisConfig,
        TTSClient,
    )
except ImportError as exc:
    raise RuntimeError(
        "Qwen3TTS SDK is unavailable. Install qwen3tts or set "
        "QWEN3TTS_CLIENT_SRC to Qwen3TTS-Streaming/client/src."
    ) from exc


def _config(speaker: str) -> SynthesisConfig:
    sample_rate = int(os.environ.get("QWEN3_TTS_SAMPLE_RATE", "24000"))
    if sample_rate != 24000:
        raise ValueError("The dialogue frontend currently requires 24000 Hz TTS audio")
    return SynthesisConfig(
        task_type=os.environ.get("QWEN3_TTS_TASK_TYPE", "custom_voice"),
        language=os.environ.get("QWEN3_TTS_LANGUAGE", "Auto"),
        speaker=speaker,
        input_mode=os.environ.get("QWEN3_TTS_INPUT_MODE", "token"),
        audio=AudioFormat(
            encoding="pcm_s16le",
            sample_rate=sample_rate,
            channels=1,
        ),
    )


class Qwen3TurnSession:
    def __init__(self, client: TTSClient, speaker: str):
        self._session = client.open_stream(
            SessionStartRequest(
                session_id=str(uuid.uuid4()),
                config=_config(speaker),
            )
        )
        self._send_closed = False
        self._cancelled = False
        self._seq = 0
        self._opened_at = time.monotonic()

    def send_text(self, text: str) -> None:
        text = text or ""
        if text == "" or self._send_closed:
            return
        self._seq += 1
        self._session.send_text(text, seq_no=self._seq)

    def end(self) -> None:
        if self._send_closed:
            return
        self._send_closed = True
        self._session.end()

    def cancel(self, reason: str = "barge_in") -> None:
        self._cancelled = True
        self._send_closed = True
        try:
            # close() is intentionally stronger than cancel(): it also
            # terminates a stream whose text input was already ended.
            self._session.close(reason=reason)
        except Exception:
            pass

    def expired(self, max_age_s: float) -> bool:
        return time.monotonic() - self._opened_at >= max_age_s

    def iter_pcm(self, idle_s: float | None = None):
        idle_s = float(idle_s if idle_s is not None else os.environ.get("QWEN3_TTS_IDLE_S", "20"))
        messages: queue.Queue[object] = queue.Queue()
        sentinel = object()

        def drain() -> None:
            try:
                for message in self._session.iter_messages():
                    messages.put(message)
            except Exception as exc:
                messages.put(exc)
            finally:
                messages.put(sentinel)

        threading.Thread(target=drain, daemon=True).start()
        while True:
            try:
                message = messages.get(timeout=idle_s)
            except queue.Empty:
                self.cancel("idle_timeout")
                return
            if message is sentinel:
                return
            if isinstance(message, Exception):
                if self._cancelled:
                    return
                raise message
            if isinstance(message, AudioChunk) and message.pcm_bytes:
                yield message.pcm_bytes
            elif isinstance(message, StreamEvent) and message.type == "error":
                if self._cancelled:
                    return
                raise RuntimeError(message.message or "Qwen3TTS stream failed")


class Qwen3TTSClient:
    def __init__(self, speaker: str | None = None, ws_url: str | None = None):
        self.speaker = speaker or os.environ.get("QWEN3_TTS_SPEAKER", "serena")
        self.ws_url = ws_url or os.environ.get("QWEN3_TTS_WS_URL", "ws://127.0.0.1:50052/v1/ws")
        self._client = TTSClient.connect(
            self.ws_url,
            timeout=float(os.environ.get("QWEN3_TTS_TIMEOUT_S", "180")),
        )
        self._lock = threading.Lock()
        self._warm: Qwen3TurnSession | None = None
        print(
            f"[qwen3tts] native stream ws={self.ws_url} speaker={self.speaker}",
            flush=True,
        )
        self._spawn_warm()
        deadline = time.time() + float(os.environ.get("QWEN3_TTS_PREWARM_WAIT_S", "5"))
        while time.time() < deadline:
            with self._lock:
                if self._warm is not None:
                    break
            time.sleep(0.02)
        with self._lock:
            ready = self._warm is not None
        if not ready:
            # Fail application startup instead of reporting success with an
            # unreachable endpoint, incompatible task, or unknown speaker.
            session = Qwen3TurnSession(self._client, self.speaker)
            with self._lock:
                if self._warm is None:
                    self._warm = session
                    session = None
            if session is not None:
                session.cancel("duplicate prewarm")

    def _spawn_warm(self) -> None:
        def create() -> None:
            try:
                session = Qwen3TurnSession(self._client, self.speaker)
            except Exception as exc:
                print(f"[qwen3tts] prewarm failed: {exc}", flush=True)
                return
            with self._lock:
                old = self._warm
                self._warm = session
            if old is not None:
                old.cancel("replaced")

        threading.Thread(target=create, daemon=True, name="qwen3-prewarm").start()

    def open_turn(self) -> Qwen3TurnSession:
        with self._lock:
            session = self._warm
            self._warm = None
        if session is not None and session.expired(
            float(os.environ.get("QWEN3_TTS_PREWARM_MAX_AGE_S", "240"))
        ):
            session.cancel("expired_prewarm")
            session = None
        self._spawn_warm()
        if session is not None:
            return session
        return Qwen3TurnSession(self._client, self.speaker)
