from voxtral_realtime.config import RealtimeConfig


def test_environment_config_and_public_model_default():
    config = RealtimeConfig.from_env(
        {"VOXTRAL_PORT": "9000", "VOXTRAL_LEAD_IN_GATE": "false"}
    )
    assert config.port == 9000
    assert config.lead_in_gate is False
    assert config.model_id == "Kaiqfu/X2-Turn-4B-0812"
