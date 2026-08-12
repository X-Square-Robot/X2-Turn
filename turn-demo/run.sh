#!/usr/bin/env bash
set -euo pipefail

BACKEND="${BACKEND:-hf}"
MODEL="${MODEL:-Kaiqfu/X2-Turn-4B-0812}"
DEVICE="${DEVICE:-cuda:0}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
VLLM_URL="${VLLM_URL:-ws://127.0.0.1:8010/v1/realtime}"
VLLM_MODEL="${VLLM_MODEL:-$MODEL}"
PYTHON="${PYTHON:-python}"

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

exec "$PYTHON" -m demo_turn.server "${args[@]}" "$@"
