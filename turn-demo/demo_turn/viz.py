"""Render frame-level turn states as an HTML timeline."""

from __future__ import annotations

from html import escape
from typing import Any, Iterable, List

TURN_COLOR = {
    "idle": "#9ca3af",
    "noidle": "#60a5fa",
    "speaking": "#3b82f6",
    "turn_end": "#22c55e",
    "backchannel": "#a855f7",
    "uncertain": "#eab308",
}


def timeline_html(
    turns: List[str],
    seconds_per_token: float = 0.08,
    max_cells: int = 240,
) -> str:
    n = len(turns)
    step = max(1, n // max_cells) if n > max_cells else 1
    cells = []
    for i in range(0, n, step):
        t = turns[i]
        color = TURN_COLOR.get(t, "#ddd")
        t0 = i * seconds_per_token
        title = escape(f"f{i} {t0:.2f}s {t}", quote=True)
        cells.append(
            f'<div title="{title}" style="flex:1;min-width:2px;height:28px;'
            f'background:{color};"></div>'
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
    {"".join(cells)}
  </div>
</div>
"""


def frame_table_html(frames: Iterable[Any]) -> str:
    """Render raw frame-level ASR and Turn predictions without policy actions."""
    rows = []
    for frame in frames:
        rows.append(
            "<tr>"
            f"<td>{int(frame.frame)}</td>"
            f"<td>{float(frame.t0):.2f}-{float(frame.t1):.2f}</td>"
            f"<td>{escape(str(frame.asr))}</td>"
            f"<td style='color:{TURN_COLOR.get(frame.turn, '#000')}'>"
            f"{escape(str(frame.turn))}</td>"
            f"<td>{float(frame.turn_prob):.3f}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan=5>(no frames)</td></tr>")
    return f"""
<div style="max-height:320px;overflow:auto;font-family:ui-monospace,monospace;font-size:12px;">
<table style="width:100%;border-collapse:collapse;">
<thead><tr style="text-align:left;border-bottom:1px solid #ddd;">
<th>frame</th><th>time</th><th>ASR token</th><th>turn</th><th>prob</th>
</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
"""
