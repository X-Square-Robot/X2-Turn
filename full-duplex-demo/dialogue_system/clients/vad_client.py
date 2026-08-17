import time
import uuid
import base64
import os

import json
import websocket
import numpy as np


class TurnTaking:
    def __init__(
        self,
        status_callback=None,
        transcription_callback=None,
        circle_callback=None,
        server_url=None,
        client_id=None,
        timeout=20.0,
    ):
        self.client_id = client_id or uuid.uuid4().hex
        self.timeout = timeout
        self.status_callback = status_callback
        self.transcription_callback = transcription_callback
        self.circle_callback = circle_callback
        self.server_url = server_url or os.environ.get(
            "TURN_API_URL", "ws://localhost:8000/turn"
        )
        self.ws = None
        self.bot_speaking = False
        self._last_status = None
        self._last_circle_status = None
        self._last_transcription = None
        self.last_detail = None  # full state dict of the latest /turn response

    def set_bot_speaking(self, speaking: bool) -> None:
        self.bot_speaking = bool(speaking)

    def _emit_status(self, state, message, force=False):
        if not self.status_callback:
            return
        payload = (state, message)
        if not force and self._last_status == payload:
            return
        self._last_status = payload
        try:
            self.status_callback(state, message)
        except Exception as exc:
            print(f"[VadClient] status callback failed: {exc}")

    def _emit_circle_status(self, status, force=False):
        if not self.circle_callback:
            return
        if not force and self._last_circle_status == status:
            return
        self._last_circle_status = status
        try:
            self.circle_callback(status)
        except Exception as exc:
            print(f"[VadClient] circle callback failed: {exc}")

    def _emit_transcription(self, text, force=False):
        if not self.transcription_callback:
            return
        cleaned = text.strip()
        if not cleaned and not force:
            return
        if not force and self._last_transcription == cleaned:
            return
        self._last_transcription = cleaned
        try:
            self.transcription_callback(cleaned)
        except Exception as exc:
            print(f"[VadClient] transcription callback failed: {exc}")

    def connect(self):
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
        self.ws = websocket.create_connection(self.server_url, timeout=self.timeout)
        self.ws.settimeout(self.timeout)

    def reset(self, client_id=None):
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None
        self.client_id = client_id or uuid.uuid4().hex
        self.bot_speaking = False
        self._last_status = None
        self._last_circle_status = None
        self._last_transcription = None
        self.last_detail = None
        # Defer VAD WS connect to process() — avoid blocking pool release on :8000 down.

    def _ensure_connected(self):
        if self.ws is None:
            self.connect()

    def process(self, audio_chunk: np.ndarray, bot_speaking=None):
        if audio_chunk is None:
            return None
        if bot_speaking is not None:
            self.bot_speaking = bool(bot_speaking)
        payload = {
            "type": "audio",
            "session_id": self.client_id,
            "bot_speaking": self.bot_speaking,
            "audio": base64.b64encode(
                np.asarray(audio_chunk, dtype=np.float32).tobytes()
            ).decode(),
        }
        try:
            self._ensure_connected()
            self.ws.send(json.dumps(payload))
            response = self.ws.recv()
            data = json.loads(response)
        except Exception:
            self.connect()
            self.ws.send(json.dumps(payload))
            response = self.ws.recv()
            data = json.loads(response)

        assert isinstance(data, dict), ValueError(
            f"Unexpected response type: {type(data)}"
        )

        self.last_detail = data.get("state") if isinstance(data.get("state"), dict) else None

        if data["state"]["state"] == "idle":
            self._emit_status("silence", "Silence detected")
            return None
        elif data["state"]["state"] == "nonidle":
            self._emit_status("listening", "Voice detected")
            self._emit_circle_status("LISTENING")
            asr_buffer = data["state"].get("asr_buffer", "")
            if asr_buffer.strip():
                self._emit_transcription(asr_buffer)
            return [None]
        elif data["state"]["state"] == "speak":
            self._emit_status("taking turn", "User finished, taking turn")
            asr_buffer = data["state"].get("asr_buffer", "")
            if asr_buffer.strip():
                self._emit_transcription(asr_buffer)
            return data["state"].get("text", "")
