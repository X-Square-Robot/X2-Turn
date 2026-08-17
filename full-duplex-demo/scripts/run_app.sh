#!/usr/bin/env bash
# Launch X Square dialogue system frontend/app (backends must already be up).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: Python executable '$PY' was not found." >&2
  exit 2
fi
cd "$ROOT/dialogue_system"
export PYTHONPATH="$ROOT/dialogue_system:${PYTHONPATH:-}"
exec "$PY" app.py
