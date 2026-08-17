import wave

from demo_turn.scenarios import BUILTIN_TEXT, build_scenarios


def test_builtin_sample_is_ready_to_run():
    scenario = build_scenarios()[0]

    assert scenario.key == "builtin_sample_en"
    assert scenario.text == BUILTIN_TEXT
    with wave.open(scenario.wav, "rb") as stream:
        assert stream.getframerate() == 16000
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        assert stream.getnframes() > 16000
