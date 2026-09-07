import importlib.util
import sys
import types
from pathlib import Path


def load_client_module():
    sdk = types.ModuleType("qwen3tts")
    for name in (
        "AudioChunk",
        "AudioFormat",
        "SessionStartRequest",
        "StreamEvent",
        "SynthesisConfig",
        "TTSClient",
    ):
        sdk_type = type(
            name,
            (),
            {
                "__init__": lambda self, **kwargs: self.__dict__.update(kwargs),
            },
        )
        setattr(sdk, name, sdk_type)
    sys.modules["qwen3tts"] = sdk

    path = Path(__file__).parents[1] / "dialogue_system" / "clients" / "qwen3_tts_client.py"
    spec = importlib.util.spec_from_file_location("qwen3_tts_client_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cancel_force_closes_after_input_has_ended():
    module = load_client_module()

    class FakeSession:
        def __init__(self):
            self.ended = False
            self.closed_reason = None

        def end(self):
            self.ended = True

        def close(self, reason):
            self.closed_reason = reason

    wrapped = module.Qwen3TurnSession.__new__(module.Qwen3TurnSession)
    wrapped._session = FakeSession()
    wrapped._send_closed = False
    wrapped._seq = 0

    wrapped.end()
    wrapped.cancel("barge_in")

    assert wrapped._session.ended
    assert wrapped._session.closed_reason == "barge_in"


def test_streaming_tokens_preserve_whitespace():
    module = load_client_module()

    class FakeSession:
        def __init__(self):
            self.sent = []

        def send_text(self, text, seq_no):
            self.sent.append((text, seq_no))

    wrapped = module.Qwen3TurnSession.__new__(module.Qwen3TurnSession)
    wrapped._session = FakeSession()
    wrapped._send_closed = False
    wrapped._seq = 0

    wrapped.send_text(" hello ")
    wrapped.send_text(" ")

    assert wrapped._session.sent == [(" hello ", 1), (" ", 2)]


def test_default_language_is_automatic(monkeypatch):
    module = load_client_module()
    monkeypatch.delenv("QWEN3_TTS_LANGUAGE", raising=False)

    config = module._config("serena")

    assert config.language == "Auto"


def test_cancelled_stream_error_is_expected():
    module = load_client_module()

    class FakeSession:
        def iter_messages(self):
            yield module.StreamEvent(type="error", message="stream already closed")

    wrapped = module.Qwen3TurnSession.__new__(module.Qwen3TurnSession)
    wrapped._session = FakeSession()
    wrapped._cancelled = True

    assert list(wrapped.iter_pcm(idle_s=0.1)) == []


def test_non_24khz_audio_is_rejected(monkeypatch):
    module = load_client_module()
    monkeypatch.setenv("QWEN3_TTS_SAMPLE_RATE", "16000")

    try:
        module._config("serena")
    except ValueError as exc:
        assert "24000 Hz" in str(exc)
    else:
        raise AssertionError("Expected non-24kHz audio to be rejected")
