#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to the exported final_vllm directory}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8011}"
GPU_MEM="${GPU_MEM:-0.8}"
exec "$VLLM_BIN" serve "$MODEL" --served-model-name "${SERVED_MODEL_NAME:-x-square/voxtral-mtp-turn-v3-delay0-zhen}" \
  --host "${HOST:-0.0.0.0}" --port "$PORT" --tokenizer-mode mistral --enforce-eager \
  --gpu-memory-utilization "$GPU_MEM"
