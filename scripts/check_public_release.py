"""Reject private infrastructure and likely secrets before publication."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "certs",
    "CosyVoice_official",
    "logs",
    "node_modules",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".css",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PATTERNS = [
    ("private storage path", re.compile(r"/(?:mnt|workspace)/[^ \n\"']+")),
    ("internal GitLab host", re.compile(r"\bgitlab\.zbl\.local\b")),
    ("legacy deployment address", re.compile(r"\b39\.101\.65\.229\b")),
    (
        "private IPv4 address",
        re.compile(
            r"(?<![\w.])(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3})(?![\w.])"
        ),
    ),
    (
        "possible secret",
        re.compile(
            r"""(?i)(api[_-]?key|access[_-]?token|secret|token)\s*=\s*['"][^'"]{8,}"""
        ),
    ),
    ("AWS access key", re.compile(r"AKIA[A-Z0-9]{16}")),
]


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.resolve() == SELF
            or any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts)
            or path.suffix.lower() not in TEXT_SUFFIXES
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
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
