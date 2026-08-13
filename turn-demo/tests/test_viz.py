from types import SimpleNamespace

from demo_turn.viz import frame_table_html, timeline_html


def test_timeline_renders_raw_turn_states():
    rendered = timeline_html(["idle", "speaking", "turn_end"])

    assert "idle" in rendered
    assert "speaking" in rendered
    assert "turn_end" in rendered
    assert "barge-in" not in rendered


def test_frame_table_renders_raw_predictions_and_escapes_text():
    frame = SimpleNamespace(
        frame=7,
        t0=0.56,
        t1=0.64,
        asr="<script>alert(1)</script>",
        turn="speaking",
        turn_prob=0.875,
    )

    rendered = frame_table_html([frame])

    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered
    assert "speaking" in rendered
    assert "0.875" in rendered
    assert "<th>action</th>" not in rendered
