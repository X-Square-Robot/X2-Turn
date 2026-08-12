"""把逐帧 turn / 决策事件渲染成 HTML 色条 + 事件表。"""

from __future__ import annotations

from typing import List, Optional

try:
    from demo_turn.policy import DemoDecision, FrameEvent
except ImportError:
    from policy import DemoDecision, FrameEvent  # type: ignore

TURN_COLOR = {
    "idle": "#9ca3af",
    "noidle": "#60a5fa",
    "speaking": "#3b82f6",
    "turn_end": "#22c55e",
    "backchannel": "#a855f7",
    "uncertain": "#eab308",
}

ACTION_COLOR = {
    "ACCEPT": "#16a34a",
    "REJECT": "#9333ea",
    "HOLD": "#ca8a04",
    "NONE": "#6b7280",
    "BARGE_IN": "#dc2626",
}


def timeline_html(
    turns: List[str],
    seconds_per_token: float = 0.08,
    barge_in_at_s: Optional[float] = None,
    max_cells: int = 240,
) -> str:
    n = len(turns)
    step = max(1, n // max_cells) if n > max_cells else 1
    cells = []
    for i in range(0, n, step):
        t = turns[i]
        color = TURN_COLOR.get(t, "#ddd")
        t0 = i * seconds_per_token
        title = f"f{i} {t0:.2f}s {t}"
        cells.append(
            f'<div title="{title}" style="flex:1;min-width:2px;height:28px;'
            f'background:{color};"></div>'
        )
    barge_mark = ""
    if barge_in_at_s is not None and n > 0:
        pct = min(100.0, max(0.0, 100.0 * barge_in_at_s / (n * seconds_per_token)))
        barge_mark = (
            f'<div style="position:absolute;left:{pct:.2f}%;top:0;bottom:0;'
            f'width:2px;background:#dc2626;z-index:2;" title="barge-in '
            f'@{barge_in_at_s:.2f}s"></div>'
        )
    legend = " ".join(
        f'<span style="display:inline-block;width:10px;height:10px;'
        f'background:{c};margin-right:4px;border-radius:2px;"></span>{k}'
        for k, c in TURN_COLOR.items()
    )
    return f"""
<div style="font-family:ui-sans-serif,system-ui;font-size:13px;">
  <div style="margin-bottom:6px;color:#374151;">{legend}</div>
  <div style="position:relative;display:flex;width:100%;border:1px solid #e5e7eb;
              border-radius:6px;overflow:hidden;">
    {barge_mark}
    {''.join(cells)}
  </div>
</div>
"""


def decision_banner(decision: DemoDecision) -> str:
    color = ACTION_COLOR.get(decision.action, "#6b7280")
    barge = (
        f'<div style="margin-top:4px;color:#dc2626;">⚡ barge-in @ '
        f"{decision.barge_in_at_s:.2f}s</div>"
        if decision.barge_in_at_s is not None
        else ""
    )
    return f"""
<div style="padding:14px 16px;border-radius:10px;background:{color}18;
            border:1px solid {color};font-family:ui-sans-serif,system-ui;">
  <div style="font-size:22px;font-weight:700;color:{color};">{decision.action}</div>
  <div style="margin-top:6px;color:#111827;">末态 turn = <b>{decision.last_turn}</b></div>
  <div style="margin-top:4px;color:#4b5563;">{decision.reason}</div>
  <div style="margin-top:4px;color:#111827;">ASR: {decision.asr_text or '(empty)'}</div>
  {barge}
</div>
"""


def events_table(events: List[FrameEvent], only_interesting: bool = True) -> str:
    rows = []
    interesting = {
        "barge_in",
        "候选ACCEPT",
        "候选REJECT",
        "HOLD",
        "listening_tts",
    }
    for ev in events:
        if only_interesting and ev.action not in interesting and ev.turn == "idle":
            continue
        if only_interesting and ev.action in ("none", "idle", "tts_playing") and ev.asr in (
            "[PAD]",
            "",
        ):
            continue
        if only_interesting and ev.action in ("idle", "tts_playing") and ev.turn == "idle":
            continue
        rows.append(
            f"<tr>"
            f"<td>{ev.frame}</td>"
            f"<td>{ev.t0:.2f}-{ev.t1:.2f}</td>"
            f"<td>{ev.asr}</td>"
            f"<td style='color:{TURN_COLOR.get(ev.turn,'#000')}'>{ev.turn}</td>"
            f"<td>{ev.turn_prob:.2f}</td>"
            f"<td>{ev.action}</td>"
            f"<td>{ev.note}</td>"
            f"</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan=7>(无关键事件)</td></tr>")
    return f"""
<div style="max-height:320px;overflow:auto;font-family:ui-monospace,monospace;font-size:12px;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="text-align:left;border-bottom:1px solid #ddd;">
<th>f</th><th>time</th><th>ASR</th><th>turn</th><th>p</th><th>action</th><th>note</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
"""
