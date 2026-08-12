import base64

import numpy as np
from fastapi.testclient import TestClient

from voxtral_realtime.config import RealtimeConfig
from voxtral_realtime.server import TurnBridge, create_app


class FakeRealtimeSession:
    def __init__(self, **kwargs):
        self.frames = []
        self.asr_text = "hello"

    async def connect(self):
        return None

    async def close(self):
        return None

    def set_bot_speaking(self, speaking):
        return None

    async def push_pcm(self, pcm):
        self.frames.append({"turn": "speaking", "frame_index": 1})

    def consume_turn_frames(self):
        frames, self.frames = self.frames, []
        return frames


def test_health_and_mocked_websocket():
    config = RealtimeConfig(
        acoustic_vad_rms_threshold=1.0, acoustic_vad_peak_threshold=1.0
    )
    app = create_app(TurnBridge(config, session_factory=FakeRealtimeSession))
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True
        with client.websocket_connect("/turn") as socket:
            socket.send_json(
                {"type": "control", "session_id": "test", "bot_speaking": False}
            )
            assert socket.receive_json()["type"] == "control.ack"
            audio = base64.b64encode(
                np.zeros(1280, dtype=np.float32).tobytes()
            ).decode()
            socket.send_json({"type": "audio", "session_id": "test", "audio": audio})
            response = socket.receive_json()
            assert response["type"] == "turn_state"
            assert response["state"]["turn_class"] == "speaking"
