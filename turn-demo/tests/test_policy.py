from demo_turn.policy import PolicyConfig, run_policy_on_frames


def decide(turns, *, bot_speaking=False, barge_in_frames=4):
    return run_policy_on_frames(
        turns=turns,
        turn_probs=[0.9] * len(turns),
        asr_tokens=[""] * len(turns),
        bot_speaking=bot_speaking,
        cfg=PolicyConfig(barge_in_frames=barge_in_frames),
    )


def test_turn_end_is_accepted():
    assert decide(["speaking", "turn_end", "idle"]).action == "ACCEPT"


def test_backchannel_is_rejected():
    assert decide(["noidle", "backchannel", "idle"]).action == "REJECT"


def test_incomplete_speech_is_held():
    assert decide(["noidle", "speaking", "idle"]).action == "HOLD"


def test_sustained_speech_barges_in():
    decision = decide(
        ["speaking", "speaking", "speaking", "speaking", "turn_end"],
        bot_speaking=True,
    )
    assert decision.barge_in_at_s is not None
