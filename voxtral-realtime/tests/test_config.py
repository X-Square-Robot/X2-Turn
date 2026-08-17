from voxtral_realtime.config import RealtimeConfig


def test_environment_config_and_public_model_default():
    config = RealtimeConfig.from_env(
        {
            "VOXTRAL_PORT": "9000",
            "VOXTRAL_LEAD_IN_GATE": "false",
            "VOXTRAL_TRACE_JSONL": "/tmp/turn-trace.jsonl",
        }
    )
    assert config.port == 9000
    assert config.lead_in_gate is False
    assert config.host == "127.0.0.1"
    assert config.model_id == "x-square-robot/X2-Turn-4B-0812"
    assert config.trace_jsonl == "/tmp/turn-trace.jsonl"
