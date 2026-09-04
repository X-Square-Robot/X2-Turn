#!/usr/bin/env bash
# Launch the full-duplex dialogue demo:
#   X2 Turn ASR + turn-taking, Qwen2.5-3B LLM, and Qwen3TTS-Streaming.
# Run this script from any directory; paths are resolved relative to this file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
PY="${PY:-python3}"
VLLM_PY="${VLLM_PY:-python3}"
VOXTRAL_CLI="${VOXTRAL_CLI:-voxtral-realtime}"
VOXTRAL_MODEL="${VOXTRAL_MODEL:-x-square-robot/X2-Turn-4B-0812}"
VOXTRAL_VLLM_MODEL="${VOXTRAL_VLLM_MODEL:-}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
# Local services bind to loopback unless remote access is requested.
BIND_HOST="${BIND_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8011}"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

stop_pid() {
  local name="$1" pidfile="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name (pid=$pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

if [[ "${1:-}" == "stop" || "${1:-}" == "stop-all" ]]; then
  stop_pid llm "$LOG_DIR/llm.pid"
  stop_pid vad "$LOG_DIR/vad.pid"
  stop_pid app "$LOG_DIR/app.pid"
  if [[ "${1:-}" == "stop-all" ]]; then
    stop_pid vllm "$LOG_DIR/vllm.pid"
  fi
  echo "Stopped."
  exit 0
fi

# GPU assignment (override freely). TURN_GPU is preferred; VAD_GPU is a legacy alias.
TURN_GPU="${TURN_GPU:-${VAD_GPU:-0}}"
LLM_GPU="${LLM_GPU:-2}"
QWEN3_TTS_WS_URL="${QWEN3_TTS_WS_URL:-ws://127.0.0.1:50052/v1/ws}"
QWEN3_TTS_HEALTH_URL="${QWEN3_TTS_HEALTH_URL:-http://127.0.0.1:50052/v1/capabilities}"
QWEN3_TTS_SPEAKER="${QWEN3_TTS_SPEAKER:-serena}"
LLM_TOKEN_STREAM="${LLM_TOKEN_STREAM:-1}"

# Browsers permit microphone access on localhost; use a real certificate when
# exposing the demo on another host.
DEMO_PORT="${DEMO_PORT:-8443}"
DEMO_PUBLIC_HOST="${DEMO_PUBLIC_HOST:-localhost}"
export DEMO_PORT
export DEMO_SSL_CERT="${DEMO_SSL_CERT:-$ROOT/certs/cert.pem}"
export DEMO_SSL_KEY="${DEMO_SSL_KEY:-$ROOT/certs/key.pem}"
need_cert=0
if [[ ! -f "$DEMO_SSL_CERT" || ! -f "$DEMO_SSL_KEY" ]]; then
  need_cert=1
elif ! openssl x509 -in "$DEMO_SSL_CERT" -noout -text 2>/dev/null \
  | grep -q "$DEMO_PUBLIC_HOST"; then
  need_cert=1
fi
if [[ "$need_cert" == "1" ]]; then
  echo "Generating self-signed TLS cert for $DEMO_PUBLIC_HOST ..."
  mkdir -p "$ROOT/certs"
  if [[ "$DEMO_PUBLIC_HOST" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    PUBLIC_SAN="IP:$DEMO_PUBLIC_HOST"
  else
    PUBLIC_SAN="DNS:$DEMO_PUBLIC_HOST"
  fi
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$DEMO_SSL_KEY" -out "$DEMO_SSL_CERT" -days 825 \
    -subj "/CN=$DEMO_PUBLIC_HOST" \
    -addext "subjectAltName=$PUBLIC_SAN,DNS:localhost,IP:127.0.0.1"
  chmod 600 "$DEMO_SSL_KEY"
fi

require_value() {
  local name="$1" value="$2"
  if [[ -z "$value" ]]; then
    echo "ERROR: $name is required. Set it in the environment or .env." >&2
    exit 2
  fi
}

require_command() {
  local command_name="$1"
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "ERROR: required command '$command_name' was not found." >&2
    exit 2
  fi
}

require_value VOXTRAL_MODEL "$VOXTRAL_MODEL"
require_command "$PY"
if ! PYTHONPATH="${QWEN3TTS_CLIENT_SRC:-}${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" -c "import qwen3tts" >/dev/null 2>&1; then
  echo "ERROR: qwen3tts SDK is unavailable." >&2
  echo "Install it or set QWEN3TTS_CLIENT_SRC to Qwen3TTS-Streaming/client/src." >&2
  exit 2
fi

echo "==== Dialogue Demo ===="
echo "ROOT         : $ROOT"
echo "Voxtral model: $VOXTRAL_MODEL"
echo "vLLM artifact: ${VOXTRAL_VLLM_MODEL:-<reuse existing :$VLLM_PORT if healthy>}"
echo "Turn GPU     : $TURN_GPU"
echo "LLM model    : $LLM_MODEL      (GPU $LLM_GPU)"
echo "Bind host    : $BIND_HOST"
echo "Python       : $PY"
echo "UI           : https://$DEMO_PUBLIC_HOST:$DEMO_PORT  (TLS)"
echo "Logs         : $LOG_DIR"

# Optional: stop previous
bash "$ROOT/start_demo.sh" stop || true

export PYTHONUNBUFFERED=1

# Timed health probe.
health_ok() {
  local url="$1"
  curl -ksf --connect-timeout 1 --max-time 2 "$url" >/dev/null 2>&1
}

port_open() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

if ! health_ok "$QWEN3_TTS_HEALTH_URL"; then
  echo "ERROR: local Qwen3TTS-Streaming is not ready: $QWEN3_TTS_WS_URL" >&2
  echo "Start the local engine first, then rerun this launcher." >&2
  exit 2
fi
if ! curl -ksf --connect-timeout 1 --max-time 3 "$QWEN3_TTS_HEALTH_URL" \
    | "$PY" -c '
import json, sys
caps = json.load(sys.stdin)
assert str(caps.get("protocol_version", "")).startswith("tts-session-")
assert "custom_voice" in caps.get("declared_supported_task_types", [])
'; then
  echo "ERROR: Qwen3TTS endpoint lacks the required custom_voice protocol." >&2
  exit 2
fi
echo "[1/4] TTS  $QWEN3_TTS_WS_URL (local Qwen3TTS-Streaming)"

if health_ok "http://127.0.0.1:6007/health"; then
  echo "[2/4] LLM  :6007 already running"
else
  echo "[2/4] LLM  :$BIND_HOST:6007 (GPU $LLM_GPU)"
  CUDA_VISIBLE_DEVICES="$LLM_GPU" nohup "$PY" -u "$ROOT/voxtral_bridge/llm_server.py" \
    --model_dir "$LLM_MODEL" --host "$BIND_HOST" --port 6007 \
    >"$LOG_DIR/llm.log" 2>&1 &
  echo $! >"$LOG_DIR/llm.pid"
fi

if health_ok "http://127.0.0.1:$VLLM_PORT/health" || port_open "$VLLM_PORT"; then
  echo "[3a/4] vLLM :$VLLM_PORT already running"
else
  require_command "$VLLM_PY"
  require_value VOXTRAL_VLLM_MODEL "${VOXTRAL_VLLM_MODEL:-}"
  if [[ ! -f "$VOXTRAL_VLLM_MODEL/consolidated.safetensors" ]]; then
    echo "ERROR: VOXTRAL_VLLM_MODEL must be an exported vLLM directory containing consolidated.safetensors." >&2
    echo "Do not pass the Hugging Face model ID. See ../voxtral-realtime/integrations/vllm/README.md" >&2
    exit 2
  fi
  echo "[3a/4] vLLM :$BIND_HOST:$VLLM_PORT (GPU $TURN_GPU)  <-- X2 Turn realtime"
  CUDA_VISIBLE_DEVICES="$TURN_GPU" nohup "$VLLM_PY" -u -m vllm.entrypoints.cli.main \
    serve "$VOXTRAL_VLLM_MODEL" \
    --served-model-name "$VOXTRAL_MODEL" \
    --host "$BIND_HOST" --port "$VLLM_PORT" --tokenizer-mode mistral --enforce-eager \
    --gpu-memory-utilization "${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.35}" \
    >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! >"$LOG_DIR/vllm.pid"
fi

require_command "$VOXTRAL_CLI"
if [[ -z "${VOXTRAL_TRACE_JSONL+x}" ]]; then
  VOXTRAL_TRACE_JSONL="$LOG_DIR/turn_trace.jsonl"
fi
export VOXTRAL_TRACE_JSONL
if health_ok "http://127.0.0.1:8000/health"; then
  echo "[3b/4] Turn bridge :8000 already running"
else
  echo "[3b/4] Turn bridge :$BIND_HOST:8000 (voxtral-realtime, forwards to vLLM)"
  nohup "$VOXTRAL_CLI" serve \
    --vllm-url "ws://127.0.0.1:$VLLM_PORT/v1/realtime" \
    --model "$VOXTRAL_MODEL" --host "$BIND_HOST" --port 8000 \
    >"$LOG_DIR/vad.log" 2>&1 &
  echo $! >"$LOG_DIR/vad.pid"
fi

echo "Waiting for vLLM / turn bridge / LLM / TTS to become healthy ..."
for i in $(seq 1 300); do
  ok=0
  if health_ok "$QWEN3_TTS_HEALTH_URL"; then ok=$((ok+1)); fi
  if health_ok "http://127.0.0.1:6007/health"; then ok=$((ok+1)); fi
  if health_ok "http://127.0.0.1:8000/health"; then ok=$((ok+1)); fi
  if health_ok "http://127.0.0.1:$VLLM_PORT/health"; then ok=$((ok+1)); fi
  if [[ "$ok" -eq 4 ]]; then
    echo "Backends ready."
    break
  fi
  if [[ "$i" -eq 300 ]]; then
    echo "Timeout waiting for backends. Check logs in $LOG_DIR"
    exit 1
  fi
  sleep 2
done

echo "[4/4] Dialogue UI :$BIND_HOST:$DEMO_PORT (HTTPS)"
cd "$ROOT/dialogue_system"
export PYTHONPATH="$ROOT/dialogue_system:${PYTHONPATH:-}"
export QWEN3_TTS_WS_URL QWEN3_TTS_SPEAKER
export QWEN3TTS_CLIENT_SRC LLM_TOKEN_STREAM
export DEMO_PORT DEMO_SSL_CERT DEMO_SSL_KEY
export DEMO_BIND_HOST="$BIND_HOST"
nohup "$PY" app.py >"$LOG_DIR/app.log" 2>&1 &
echo $! >"$LOG_DIR/app.pid"

app_ready=0
for _ in $(seq 1 30); do
  if ! kill -0 "$(cat "$LOG_DIR/app.pid")" 2>/dev/null; then
    echo "ERROR: dialogue app exited during startup." >&2
    tail -40 "$LOG_DIR/app.log" >&2 || true
    exit 1
  fi
  if health_ok "https://127.0.0.1:$DEMO_PORT/"; then
    app_ready=1
    break
  fi
  sleep 1
done
if [[ "$app_ready" != "1" ]]; then
  echo "ERROR: dialogue app did not become ready." >&2
  stop_pid app "$LOG_DIR/app.pid"
  tail -40 "$LOG_DIR/app.log" >&2 || true
  exit 1
fi
echo
echo "Done. Open:  https://$DEMO_PUBLIC_HOST:$DEMO_PORT"
echo "             (browser will warn on self-signed cert — click Advanced → Proceed)"
echo "Stop with:   $0 stop"
echo "Logs:        $LOG_DIR/{llm,vad,app}.log"
