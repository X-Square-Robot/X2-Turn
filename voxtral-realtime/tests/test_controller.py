from voxtral_realtime.turn import FrameTurnConfig, FrameTurnController


def test_noidle_cannot_barge_but_speaking_can():
    controller = FrameTurnController()
    controller.set_bot_speaking(True)
    noidle = controller.on_frame("noidle", "noise")
    assert noidle["state"] == "nonidle"
    assert "event" not in noidle
    speaking = controller.on_frame("speaking", "hello")
    assert speaking["event"] == "barge_in"


def test_turn_end_confirms_then_accepts_after_tail():
    controller = FrameTurnController(
        FrameTurnConfig(
            end_confirm_frames=1,
            tail_min_frames=1,
            tail_max_frames=2,
            tail_stable_frames=1,
        )
    )
    assert controller.on_frame("turn_end", "hello")["state"] == "nonidle"
    assert controller.on_frame("idle", "hello")["note"].startswith("tail_start")
    result = controller.on_frame("idle", "hello")
    assert result["state"] == "speak"
    assert result["event"] == "accept"
    assert result["text"] == "hello"


def test_soft_endpoint_after_semantic_speaking():
    controller = FrameTurnController(
        FrameTurnConfig(
            silence_end_frames=1,
            tail_min_frames=1,
            tail_max_frames=1,
            tail_stable_frames=0,
            short_asr_chars=0,
        )
    )
    controller.on_frame("speaking", "hello world")
    assert controller.on_frame("idle", "hello world")["note"].startswith("tail_start")
    assert controller.on_frame("idle", "hello world")["state"] == "speak"
