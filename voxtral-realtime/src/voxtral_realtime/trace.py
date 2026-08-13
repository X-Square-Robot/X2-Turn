"""Structured JSONL traces for replaying turn-controller decisions."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

TRACE_SCHEMA = "x2-turn-trace"
TRACE_VERSION = 1


class JSONLTraceWriter:
    """Append small, audio-free trace records to one JSONL file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = threading.Lock()

    def write(self, record_type: str, session_id: str, **fields: Any) -> None:
        payload = {
            "schema": TRACE_SCHEMA,
            "version": TRACE_VERSION,
            "record_type": record_type,
            "ts": time.time(),
            "session_id": session_id,
            **fields,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
