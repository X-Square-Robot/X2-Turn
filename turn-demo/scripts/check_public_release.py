#!/usr/bin/env python3
"""Reject common private-environment artifacts before a public release."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {Path(__file__).resolve(), ROOT / ".git"}
PRIVATE_ROOT = "/" + "mnt" + "/"
SECRET_WORDS = (
    "AKIA" + r"[A-Z0-9]{16}",
    r"""(?i)(api[_-]?key|secret|token)\s*=\s*['"][^'"]{8,}""",
)
PATTERNS = [
    ("private absolute path", re.compile(re.escape(PRIVATE_ROOT))),
    (
        "private host",
        re.compile(
            r"(?<![\w.])(10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3})(?![\w.])"
        ),
    ),
]
PATTERNS.extend(("possible secret", re.compile(item)) for item in SECRET_WORDS)
TEXT_SUFFIXES = {
    ".py",
    ".toml",
    ".md",
    ".sh",
    ".txt",
    ".yml",
    ".yaml",
    ".json",
    ".ini",
    ".cfg",
    ".example",
    "",
}


def main() -> int:
    failures = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.resolve() in SKIP or ".git" in path.parts:
            continue
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(ROOT)}:{line}: {label}")
    if failures:
        print("Public release check failed:\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("Public release check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
