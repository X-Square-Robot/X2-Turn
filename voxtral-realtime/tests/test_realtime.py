import json

import numpy as np
import pytest

from voxtral_realtime.realtime import RealtimeVLLMSession


def make_session(**kwargs):
    return RealtimeVLLMSession(
        vllm_url="ws://127.0.0.1:8011/v1/realtime", model="model", **kwargs
    )


def test_idle_text_is_dropped_until_speech_turn():
    session = make_session()
    session.handle_message({"type": "transcription.delta", "delta": "junk"})
    session.handle_message({"type": "turn.delta", "turn_class": "idle"})
    assert session.asr_text == ""
    session.handle_message({"type": "transcription.delta", "delta": "hello"})
    session.handle_message(
        {"type": "turn.delta", "turn_class": "speaking", "frame_index": 3}
    )
    assert session.asr_text == "hello"
    assert session.consume_turn_frames()[1]["frame_index"] == 3
    assert session.consume_turn_frames() == []


def test_text_suppression_can_be_disabled():
    session = make_session(suppress_idle_text=False)
    session.handle_message({"type": "transcription.delta", "delta": "kept"})
    session.handle_message({"type": "turn.delta", "turn_class": "idle"})
    assert session.asr_text == "kept"


def test_bot_speaking_uses_fast_commit_and_opens_gate():
    session = make_session(commit_ms=320, barge_commit_ms=80)
    session.set_bot_speaking(True)
    assert session.commit_ms == 80
    assert session._gate_open
    session._last_poll = 123.0
    session._speech_run = 42

    session.set_bot_speaking(True)

    assert session._last_poll == 123.0
    assert session._speech_run == 42


def test_session_created_during_tts_starts_with_open_gate():
    session = make_session(bot_speaking=True)

    assert session._gate_open
    assert session.commit_ms == 80


@pytest.mark.asyncio
async def test_streaming_append_does_not_repeat_generation_commit():
    class FakeWebSocket:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    session = make_session(lead_in_gate=False)
    session._ws = FakeWebSocket()
    session._sent_samples = session.sample_rate // 4
    session._turn_frames.append({"turn": "idle"})

    await session.push_pcm(np.zeros(80, dtype=np.float32))

    assert [message["type"] for message in session._ws.messages] == [
        "input_audio_buffer.append"
    ]
