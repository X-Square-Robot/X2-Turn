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
        setattr(sdk, name, type(name, (), {}))
    sys.modules["qwen3tts"] = sdk

    path = (
        Path(__file__).parents[1]
        / "dialogue_system"
        / "clients"
        / "qwen3_tts_client.py"
    )
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
