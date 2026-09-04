#!/usr/bin/env bash
# Start the external Qwen3TTS-Streaming standalone engine used by this demo.
set -euo pipefail

DEMO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$DEMO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$DEMO_ROOT/.env"
  set +a
fi

QWEN3TTS_ROOT="${QWEN3TTS_ROOT:-}"
ENGINE_PYTHON="${ENGINE_PYTHON:-python3}"
ENGINE_MODEL_PACKAGE_DIR="${ENGINE_MODEL_PACKAGE_DIR:-}"
QWEN3_TTS_GPU="${QWEN3_TTS_GPU:-0}"
QWEN3_TTS_GRPC_PORT="${QWEN3_TTS_GRPC_PORT:-50051}"
QWEN3_TTS_WS_PORT="${QWEN3_TTS_WS_PORT:-50052}"
QWEN3_TTS_MAX_BATCH="${QWEN3_TTS_MAX_BATCH:-4}"
QWEN3_TTS_MAX_SESSIONS="${QWEN3_TTS_MAX_SESSIONS:-4}"
LOG_DIR="$DEMO_ROOT/logs"
PID_FILE="$LOG_DIR/qwen3tts-engine.pid"
LOG_FILE="$LOG_DIR/qwen3tts-engine.log"
mkdir -p "$LOG_DIR"

health_ok() {
  curl -sf --connect-timeout 1 --max-time 2 \
    "http://127.0.0.1:$QWEN3_TTS_WS_PORT/v1/capabilities" >/dev/null
}

stop_engine() {
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 10); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  fi
}

case "${1:-start}" in
  stop)
    stop_engine
    exit 0
    ;;
  status)
    health_ok
    echo "Qwen3TTS engine ready: ws://127.0.0.1:$QWEN3_TTS_WS_PORT/v1/ws"
    exit 0
    ;;
esac

if health_ok; then
  echo "Qwen3TTS engine already ready: ws://127.0.0.1:$QWEN3_TTS_WS_PORT/v1/ws"
  exit 0
fi
if [[ -z "$QWEN3TTS_ROOT" || ! -d "$QWEN3TTS_ROOT/engine" ]]; then
  echo "ERROR: set QWEN3TTS_ROOT to a Qwen3TTS-Streaming checkout." >&2
  exit 2
fi
if [[ -z "$ENGINE_MODEL_PACKAGE_DIR" || ! -f "$ENGINE_MODEL_PACKAGE_DIR/runtime/model.plan" ]]; then
  echo "ERROR: missing TensorRT package: $ENGINE_MODEL_PACKAGE_DIR/runtime/model.plan" >&2
  exit 2
fi
if ! command -v "$ENGINE_PYTHON" >/dev/null 2>&1; then
  echo "ERROR: ENGINE_PYTHON was not found: $ENGINE_PYTHON" >&2
  exit 2
fi
if ! "$ENGINE_PYTHON" -c "import torch, tensorrt" >/dev/null 2>&1; then
  echo "ERROR: ENGINE_PYTHON must provide torch and tensorrt." >&2
  exit 2
fi

stop_engine
export PYTHONPATH="$QWEN3TTS_ROOT:$QWEN3TTS_ROOT/client/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${QWEN3_TTS_TENSORRT_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="$QWEN3_TTS_TENSORRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
cd "$QWEN3TTS_ROOT"
nohup env CUDA_VISIBLE_DEVICES="$QWEN3_TTS_GPU" "$ENGINE_PYTHON" -m engine.server \
  --model-package-dir "$ENGINE_MODEL_PACKAGE_DIR" \
  --device 0 \
  --max-batch "$QWEN3_TTS_MAX_BATCH" \
  --max-sessions "$QWEN3_TTS_MAX_SESSIONS" \
  --port "$QWEN3_TTS_GRPC_PORT" \
  --ws-port "$QWEN3_TTS_WS_PORT" \
  >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 120); do
  if health_ok; then
    echo "Qwen3TTS engine ready: ws://127.0.0.1:$QWEN3_TTS_WS_PORT/v1/ws"
    exit 0
  fi
  if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "ERROR: Qwen3TTS engine exited. See $LOG_FILE" >&2
    tail -40 "$LOG_FILE" >&2 || true
    exit 1
  fi
  sleep 1
done

echo "ERROR: timed out waiting for Qwen3TTS engine. See $LOG_FILE" >&2
exit 1
