#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-hf}"
MODEL="${MODEL:-x-square-robot/X2-Turn-4B-0812}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"
VLLM_URL="${VLLM_URL:-ws://127.0.0.1:8011/v1/realtime}"
VLLM_MODEL="${VLLM_MODEL:-x-square-robot/X2-Turn-4B-0812}"
PYTHON="${PYTHON:-python}"
SSL_CERTFILE="${SSL_CERTFILE:-}"
SSL_KEYFILE="${SSL_KEYFILE:-}"

args=(
  --backend "$BACKEND"
  --model "$MODEL"
  --device "$DEVICE"
  --host "$HOST"
  --port "$PORT"
)

if [[ "$BACKEND" == "vllm" ]]; then
  args+=(--vllm-url "$VLLM_URL" --vllm-model "$VLLM_MODEL")
fi

if [[ -n "$SSL_CERTFILE" || -n "$SSL_KEYFILE" ]]; then
  if [[ -z "$SSL_CERTFILE" || -z "$SSL_KEYFILE" ]]; then
    echo "SSL_CERTFILE and SSL_KEYFILE must be set together." >&2
    exit 2
  fi
  args+=(--ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE")
fi

exec "$PYTHON" -m demo_turn.server "${args[@]}" "$@"
