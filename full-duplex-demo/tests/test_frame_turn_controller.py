from voxtral_realtime import FrameTurnConfig, FrameTurnController


def test_only_semantic_speech_barges_during_bot_audio():
    controller = FrameTurnController()
    controller.set_bot_speaking(True)

    acoustic_only = controller.on_frame("noidle", "um", frame_index=1)
    assert acoustic_only.get("event") is None

    semantic = controller.on_frame("speaking", "hello", frame_index=2)
    assert semantic["event"] == "barge_in"


def test_confirmed_endpoint_keeps_asr_tail_before_accept():
    controller = FrameTurnController(
        FrameTurnConfig(
            end_confirm_frames=1,
            tail_min_frames=1,
            tail_max_frames=1,
            tail_stable_frames=0,
        )
    )

    pending = controller.on_frame("turn_end", "hello", frame_index=1)
    assert pending["state"] == "nonidle"

    tail = controller.on_frame("idle", "hello", frame_index=2)
    assert tail["note"].startswith("tail_start:")

    accepted = controller.on_frame("idle", "hello!", frame_index=3)
    assert accepted["state"] == "speak"
    assert accepted["text"] == "hello!"


def test_pure_backchannel_is_rejected():
    controller = FrameTurnController(
        FrameTurnConfig(
            backchannel_confirm_frames=1,
            backchannel_fallback_idle_frames=1,
        )
    )

    controller.on_frame("backchannel", "嗯", frame_index=1)
    rejected = controller.on_frame("idle", "嗯", frame_index=2)
    assert rejected["event"] == "reject"
    assert rejected["reason"] == "backchannel_filler"
