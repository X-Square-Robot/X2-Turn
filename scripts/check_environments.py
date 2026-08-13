"""Validate the checked-in Conda environment definitions."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_DIR = ROOT / "environments"
EXPECTED = {
    "environment-transformers.yml",
    "environment-vllm.yml",
    "environment-dialogue.yml",
}


def main() -> int:
    actual = {path.name for path in ENVIRONMENT_DIR.glob("environment-*.yml")}
    if actual != EXPECTED:
        missing = sorted(EXPECTED - actual)
        extra = sorted(actual - EXPECTED)
        raise SystemExit(f"environment file mismatch: missing={missing}, extra={extra}")

    for filename in sorted(EXPECTED):
        path = ENVIRONMENT_DIR / filename
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise SystemExit(f"{filename}: expected a YAML mapping")
        if not isinstance(data.get("name"), str):
            raise SystemExit(f"{filename}: missing environment name")
        if "conda-forge" not in data.get("channels", []):
            raise SystemExit(f"{filename}: conda-forge must be an explicit channel")
        dependencies = data.get("dependencies")
        if not isinstance(dependencies, list) or "pip" not in dependencies:
            raise SystemExit(f"{filename}: dependencies must include pip")
        if not any(
            isinstance(item, dict)
            and isinstance(item.get("pip"), list)
            and item["pip"]
            for item in dependencies
        ):
            raise SystemExit(f"{filename}: missing pip package list")

    print("Conda environment definitions are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
