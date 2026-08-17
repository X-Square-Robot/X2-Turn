#!/usr/bin/env bash
set -euo pipefail
: "${MODEL:?Set MODEL to the exported X2 Turn vLLM directory}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PORT="${PORT:-8011}"
GPU_MEM="${GPU_MEM:-0.8}"
exec "$VLLM_BIN" serve "$MODEL" --served-model-name "${SERVED_MODEL_NAME:-Kaiqfu/X2-Turn-4B-0812}" \
  --host "${HOST:-127.0.0.1}" --port "$PORT" --tokenizer-mode mistral --enforce-eager \
  --gpu-memory-utilization "$GPU_MEM"
