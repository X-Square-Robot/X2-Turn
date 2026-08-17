"""Enforce English release text outside the Chinese root README.

Files containing Chinese-language normalization or test data are explicitly
allowlisted because translating those literals would change product behavior.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
TEXT_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".yaml",
    ".yml",
}

ALLOWED_CJK_FILES = {
    Path("README_zh.md"),  # The Chinese repository entry point.
    # Functional Chinese-language data; translating literals changes behavior.
    Path("full-duplex-demo/dialogue_system/clients/tts_client.py"),
    Path("full-duplex-demo/dialogue_system/modules/utils/MyTn/cn_tn.py"),
    Path("full-duplex-demo/dialogue_system/modules/utils/backchannel_utils.py"),
    Path("full-duplex-demo/tests/test_frame_turn_controller.py"),
    Path("full-duplex-demo/voxtral_bridge/tts_server.py"),
    Path("full-duplex-demo/voxtral_bridge/tts_server_cosyvoice.py"),
    Path("voxtral-realtime/src/voxtral_realtime/turn/controller.py"),
}


def main() -> int:
    violations: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in TEXT_SUFFIXES
            or ".git" in path.parts
            or "__pycache__" in path.parts
        ):
            continue
        relative = path.relative_to(ROOT)
        if relative in ALLOWED_CJK_FILES:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if CJK_RE.search(line):
                violations.append(f"{relative}:{line_number}: {line.strip()}")

    if violations:
        print("Release-language check failed: non-English text found.")
        print("\n".join(violations))
        return 1

    print("Release-language check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
