import base64
import json

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
        self.frames.append(
            {
                "turn": "speaking",
                "turn_id": 37,
                "frame_index": 1,
                "probs": [0.0, 0.0, 0.9, 0.1, 0.0, 0.0],
            }
        )

    def consume_turn_frames(self):
        frames, self.frames = self.frames, []
        return frames


class FailingTraceWriter:
    def write(self, *args, **kwargs):
        raise OSError("disk full")


class MultiFrameRealtimeSession(FakeRealtimeSession):
    async def push_pcm(self, pcm):
        self.frames.extend(
            [
                {"turn": "speaking", "frame_index": 1},
                {"turn": "idle", "frame_index": 2},
            ]
        )


def test_health_and_mocked_websocket(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    config = RealtimeConfig(
        acoustic_vad_rms_threshold=1.0,
        acoustic_vad_peak_threshold=1.0,
        trace_jsonl=str(trace_path),
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

    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    frame = next(record for record in records if record["record_type"] == "frame")
    assert frame["schema"] == "x2-turn-trace"
    assert frame["version"] == 1
    assert frame["session_id"] == "test"
    assert frame["turn_class"] == "speaking"
    assert frame["probabilities"][2] == 0.9
    assert frame["dialogue_output"]["state"] == "nonidle"
    assert frame["dialogue_stop_tts"] is False
    assert "audio" not in frame


def test_trace_failure_does_not_break_bridge():
    bridge = TurnBridge(RealtimeConfig())
    bridge.trace_writer = FailingTraceWriter()

    bridge.trace("frame", "test", frame_index=1)

    assert bridge.trace_writer is None


async def test_trace_tts_stop_matches_the_state_delivered_to_dialogue(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    bridge = TurnBridge(
        RealtimeConfig(trace_jsonl=str(trace_path)),
        session_factory=MultiFrameRealtimeSession,
    )

    output = await bridge.get_session("multi").feed(
        np.zeros(1280, dtype=np.float32),
        bot_speaking=True,
    )

    assert output["turn_class"] == "idle"
    frames = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["record_type"] == "frame"
    ]
    assert len(frames) == 2
    assert all(record["dialogue_stop_tts"] is False for record in frames)
