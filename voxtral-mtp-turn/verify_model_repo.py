#!/usr/bin/env python3
"""Validate the model repository staging layout."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REQUIRED_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "MODEL_RELEASE_CHECKLIST.md",
    ".gitattributes",
    ".gitignore",
    "config.example.json",
    "verify_model_repo.py",
}

WEIGHT_SUFFIXES = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".h5",
    ".msgpack",
}

EXPECTED_OUTPUTS = [
    "idle",
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
]

EXPECTED_FRONTMATTER = {
    "license": "apache-2.0",
    "base_model": "mistralai/Voxtral-Mini-4B-Realtime-2602",
    "pipeline_tag": "automatic-speech-recognition",
    "library_name": "vllm",
}

TEXT_SUFFIXES = {".md", ".json", ".py", ".txt", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="repository directory (defaults to the script directory)",
    )
    parser.add_argument(
        "--allow-weights",
        action="store_true",
        help="allow model-weight files after owner-approved release staging",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="release mode: fail while evaluation placeholders remain",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def frontmatter(readme: str) -> str:
    match = re.match(r"\A---\n(.*?)\n---\n", readme, flags=re.DOTALL)
    if not match:
        raise ValueError("README.md has no leading YAML frontmatter")
    return match.group(1)


def scalar_value(yaml_text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*([^\n]+)\s*$", yaml_text)
    return match.group(1).strip() if match else None


def list_values(yaml_text: str, key: str) -> list[str]:
    match = re.search(rf"(?ms)^{re.escape(key)}:\s*\n((?:  - [^\n]+\n?)+)", yaml_text)
    if not match:
        return []
    return [line.removeprefix("  - ").strip() for line in match.group(1).splitlines()]


def iter_repository_files(repo: Path):
    for path in repo.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(repo).parts:
            yield path


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    errors: list[str] = []
    notes: list[str] = []

    missing = sorted(name for name in REQUIRED_FILES if not (repo / name).is_file())
    if missing:
        errors.append("missing required files: " + ", ".join(missing))

    if not any((parent / ".git").exists() for parent in (repo, *repo.parents)):
        errors.append("git repository is not initialized")

    readme_path = repo / "README.md"
    if readme_path.is_file():
        readme = read_text(readme_path)
        try:
            yaml_text = frontmatter(readme)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            for key, expected in EXPECTED_FRONTMATTER.items():
                actual = scalar_value(yaml_text, key)
                if actual != expected:
                    errors.append(
                        f"frontmatter {key!r}: expected {expected!r}, got {actual!r}"
                    )
            if list_values(yaml_text, "language") != ["zh", "en"]:
                errors.append("frontmatter language must be exactly [zh, en]")
            if list_values(yaml_text, "tags") != ["realtime", "turn-taking"]:
                errors.append(
                    "frontmatter tags must be exactly [realtime, turn-taking]"
                )

        required_readme_phrases = [
            "x-square/voxtral-mtp-turn-v3-delay0-zhen",
            "80 ms",
            "voxtral-realtime",
            "final release license",
            "training-data",
            "TBD before release",
        ]
        for phrase in required_readme_phrases:
            if phrase not in readme:
                errors.append(f"README.md missing required phrase: {phrase!r}")
        for output in EXPECTED_OUTPUTS:
            if f"`{output}`" not in readme:
                errors.append(f"README.md does not document output {output!r}")
        if re.search(r"(?i)\bbalanced\b", readme):
            errors.append("README.md still contains obsolete balanced wording")

        placeholder_count = readme.count("TBD before release")
        if args.release and placeholder_count:
            errors.append(
                f"README.md has {placeholder_count} unresolved evaluation placeholders"
            )
        elif not args.release:
            if placeholder_count < 6:
                errors.append("README.md evaluation placeholders are incomplete")
            else:
                notes.append(
                    f"staging mode: {placeholder_count} evaluation placeholders retained"
                )

    example_path = repo / "config.example.json"
    if example_path.is_file():
        try:
            example = json.loads(read_text(example_path))
        except json.JSONDecodeError as exc:
            errors.append(f"config.example.json is invalid JSON: {exc}")
        else:
            if example.get("_documentation_only") is not True:
                errors.append("config.example.json must be marked documentation-only")
            if example.get("_loadable") is not False:
                errors.append("config.example.json must explicitly be non-loadable")
            custom = example.get("required_custom_architecture_fields", {})
            required_custom = {
                "wrapper_class",
                "wrapper_definition",
                "base_architecture",
                "base_model_type",
                "state_dict_base_prefix",
                "asr_head_parameter",
                "turn_head_parameter",
                "heads_share_backbone",
                "turn_label_delay_frames",
                "frame_duration_ms",
                "turn_output_token_ids",
            }
            absent = sorted(required_custom - custom.keys())
            if absent:
                errors.append(
                    "config.example.json missing custom fields: " + ", ".join(absent)
                )
            turn_ids = custom.get("turn_output_token_ids")
            if isinstance(turn_ids, dict) and list(turn_ids) != EXPECTED_OUTPUTS:
                errors.append("turn output order does not match the release contract")
            if custom.get("frame_duration_ms") != 80:
                errors.append("frame_duration_ms must be 80")
            if custom.get("wrapper_definition") != (
                "voxtral-realtime/integrations/transformers/"
                "modeling_voxtral_mtp.py"
            ):
                errors.append("wrapper_definition must point to voxtral-realtime")

    weight_files = [
        path.relative_to(repo)
        for path in iter_repository_files(repo)
        if path.suffix.lower() in WEIGHT_SUFFIXES
    ]
    if weight_files and not args.allow_weights:
        errors.append(
            "weight-like files found (use --allow-weights only after approval): "
            + ", ".join(map(str, sorted(weight_files)))
        )
    elif not weight_files:
        notes.append("no model weights found")

    private_path_patterns = [
        re.compile(re.escape("/") + r"(?:mnt|home|workspace|Users)/"),
        re.compile(r"[A-Za-z]:\\Users\\"),
        re.compile(r"(?i)\b(?:ssh|s3)://"),
    ]
    leaks: list[str] = []
    for path in iter_repository_files(repo):
        if path.name == Path(__file__).name:
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in {
            "LICENSE",
            "NOTICE",
            ".gitignore",
            ".gitattributes",
        }:
            continue
        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in private_path_patterns):
                leaks.append(f"{path.relative_to(repo)}:{line_number}")
    if leaks:
        errors.append("possible private path leakage: " + ", ".join(leaks))
    else:
        notes.append("no private path leakage found")

    if errors:
        print("MODEL REPOSITORY VERIFICATION: FAIL")
        for error in errors:
            print(f"[ERROR] {error}")
        for note in notes:
            print(f"[INFO] {note}")
        return 1

    print("MODEL REPOSITORY VERIFICATION: PASS")
    for note in notes:
        print(f"[INFO] {note}")
    print(f"[INFO] checked {repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
