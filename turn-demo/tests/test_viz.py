from demo_turn.policy import DemoDecision, FrameEvent
from demo_turn.viz import decision_banner, events_table


def test_decision_banner_escapes_asr_text():
    decision = DemoDecision(
        action="ACCEPT",
        reason="<reason>",
        last_turn="turn_end",
        asr_text="<script>alert(1)</script>",
        barge_in_at_s=None,
        events=[],
    )

    rendered = decision_banner(decision)

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "&lt;reason&gt;" in rendered


def test_events_table_escapes_dynamic_text():
    event = FrameEvent(
        frame=0,
        t0=0.0,
        t1=0.08,
        asr="<img src=x>",
        turn="turn_end",
        turn_prob=0.9,
        bot_speaking=False,
        action="候选ACCEPT",
        note="<note>",
    )

    rendered = events_table([event])

    assert "<img" not in rendered
    assert "&lt;img src=x&gt;" in rendered
    assert "&lt;note&gt;" in rendered
