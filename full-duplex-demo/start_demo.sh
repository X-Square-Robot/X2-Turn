#!/usr/bin/env bash
# Launch full-duplex dialogue demo:
#   Voxtral turn model as X Square VAD + Qwen2.5-3B LLM + CosyVoice2
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
PY="${PY:-python3}"
COSY_PY="${COSY_PY:-python3}"
VLLM_PY="${VLLM_PY:-python3}"
VOXTRAL_CLI="${VOXTRAL_CLI:-voxtral-realtime}"
VOXTRAL_MODEL="${VOXTRAL_MODEL:-x-square/voxtral-mtp-turn-v3-delay0-zhen}"
VOXTRAL_VLLM_MODEL="${VOXTRAL_VLLM_MODEL:-$VOXTRAL_MODEL}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
COSY_MODEL="${COSY_MODEL:-FunAudioLLM/CosyVoice2-0.5B}"
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
    stop_pid tts "$LOG_DIR/tts.pid"
  fi
  echo "Stopped."
  exit 0
fi

# GPU assignment (override freely)
VAD_GPU="${VAD_GPU:-0}"
LLM_GPU="${LLM_GPU:-2}"
TTS_GPU="${TTS_GPU:-1}"
TTS_BACKEND="${TTS_BACKEND:-cosyvoice}"
if [[ "$TTS_BACKEND" == "cosyvoice" ]]; then
  TTS_PORT="${TTS_PORT:-6017}"
else
  TTS_PORT="${TTS_PORT:-6016}"
fi
TTS_API_URL="${TTS_API_URL:-http://127.0.0.1:$TTS_PORT/tts}"

# Browsers permit microphone access on localhost; use a real certificate when
# exposing the demo on another host.
DEMO_PORT="${DEMO_PORT:-8443}"
DEMO_PUBLIC_HOST="${DEMO_PUBLIC_HOST:-localhost}"
export DEMO_PORT
export DEMO_SSL_CERT="${DEMO_SSL_CERT:-$ROOT/certs/cert.pem}"
export DEMO_SSL_KEY="${DEMO_SSL_KEY:-$ROOT/certs/key.pem}"
if [[ ! -f "$DEMO_SSL_CERT" || ! -f "$DEMO_SSL_KEY" ]]; then
  echo "Generating self-signed TLS cert for $DEMO_PUBLIC_HOST ..."
  mkdir -p "$ROOT/certs"
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$DEMO_SSL_KEY" -out "$DEMO_SSL_CERT" -days 825 \
    -subj "/CN=$DEMO_PUBLIC_HOST" \
    -addext "subjectAltName=DNS:$DEMO_PUBLIC_HOST,DNS:localhost,IP:127.0.0.1"
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

echo "==== Dialogue Demo ===="
echo "ROOT         : $ROOT"
echo "Voxtral model: $VOXTRAL_MODEL"
echo "vLLM artifact: $VOXTRAL_VLLM_MODEL  (GPU $VAD_GPU)"
echo "LLM model    : $LLM_MODEL      (GPU $LLM_GPU)"
echo "Python       : $PY"
echo "UI           : https://$DEMO_PUBLIC_HOST:$DEMO_PORT  (TLS)"
echo "Logs         : $LOG_DIR"

# Optional: stop previous
bash "$ROOT/start_demo.sh" stop || true

export PYTHONUNBUFFERED=1

# Timed health probe (CosyVoice /health can hang under load).
health_ok() {
  local url="$1"
  curl -sf --connect-timeout 1 --max-time 2 "$url" >/dev/null 2>&1
}

port_open() {
  local port="$1"
  (echo >/dev/tcp/127.0.0.1/"$port") >/dev/null 2>&1
}

if health_ok "http://127.0.0.1:$TTS_PORT/health" || port_open "$TTS_PORT"; then
  echo "[1/4] TTS  :$TTS_PORT already running ($TTS_BACKEND)"
else
  if [[ "$TTS_BACKEND" == "cosyvoice" ]]; then
    require_value COSY_ROOT "${COSY_ROOT:-}"
    if [[ ! -d "$COSY_ROOT/cosyvoice" ]]; then
      echo "ERROR: COSY_ROOT must be an external CosyVoice checkout containing cosyvoice/." >&2
      exit 2
    fi
    require_command "$COSY_PY"
    export COSY_ROOT
    echo "[1/4] TTS  :0.0.0.0:$TTS_PORT ($COSY_MODEL, GPU $TTS_GPU, vLLM+JIT+TRT)"
    # Ensure CosyVoice2ForCausalLM is registered inside vLLM EngineCore subprocess.
    export VLLM_PLUGINS="${VLLM_PLUGINS:-cosyvoice2_register}"
    export COSY_ROOT
    export PYTHONPATH="${COSY_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
    CUDA_VISIBLE_DEVICES="$TTS_GPU" nohup "$COSY_PY" -u "$ROOT/voxtral_bridge/tts_server_cosyvoice.py" \
      --model-dir "$COSY_MODEL" --host 0.0.0.0 --port "$TTS_PORT" --vllm --jit --trt \
      >"$LOG_DIR/tts.log" 2>&1 &
  else
    echo "[1/4] TTS  :0.0.0.0:$TTS_PORT (Edge-TTS fallback)"
    nohup "$PY" -u "$ROOT/voxtral_bridge/tts_server.py" \
      --host 0.0.0.0 --port "$TTS_PORT" \
      >"$LOG_DIR/tts.log" 2>&1 &
  fi
  echo $! >"$LOG_DIR/tts.pid"
fi

echo "[2/4] LLM  :0.0.0.0:6007 (GPU $LLM_GPU)"
CUDA_VISIBLE_DEVICES="$LLM_GPU" nohup "$PY" -u "$ROOT/voxtral_bridge/llm_server.py" \
  --model_dir "$LLM_MODEL" --host 0.0.0.0 --port 6007 \
  >"$LOG_DIR/llm.log" 2>&1 &
echo $! >"$LOG_DIR/llm.pid"

VLLM_PORT="${VLLM_PORT:-8011}"

if health_ok "http://127.0.0.1:$VLLM_PORT/health" || port_open "$VLLM_PORT"; then
  echo "[3a/4] vLLM :$VLLM_PORT already running"
else
  require_command "$VLLM_PY"
  echo "[3a/4] vLLM :0.0.0.0:$VLLM_PORT (GPU $VAD_GPU)  <-- Voxtral MTP realtime"
  CUDA_VISIBLE_DEVICES="$VAD_GPU" nohup "$VLLM_PY" -u -m vllm.entrypoints.cli.main \
    serve "$VOXTRAL_VLLM_MODEL" \
    --served-model-name "$VOXTRAL_MODEL" \
    --host 0.0.0.0 --port "$VLLM_PORT" --tokenizer-mode mistral --enforce-eager \
    --gpu-memory-utilization "${VOXTRAL_GPU_MEMORY_UTILIZATION:-0.35}" \
    >"$LOG_DIR/vllm.log" 2>&1 &
  echo $! >"$LOG_DIR/vllm.pid"
fi

require_command "$VOXTRAL_CLI"
echo "[3b/4] Turn bridge :0.0.0.0:8000 (voxtral-realtime, forwards to vLLM)"
nohup "$VOXTRAL_CLI" serve \
  --vllm-url "ws://127.0.0.1:$VLLM_PORT/v1/realtime" \
  --model "$VOXTRAL_MODEL" --host 0.0.0.0 --port 8000 \
  >"$LOG_DIR/vad.log" 2>&1 &
echo $! >"$LOG_DIR/vad.pid"

echo "Waiting for vLLM/VAD/LLM/TTS to become healthy ..."
for i in $(seq 1 300); do
  ok=0
  # TTS: accept open port if /health is wedged
  if health_ok "http://127.0.0.1:$TTS_PORT/health" || port_open "$TTS_PORT"; then ok=$((ok+1)); fi
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

echo "[4/4] Dialogue UI :0.0.0.0:$DEMO_PORT (HTTPS)"
cd "$ROOT/dialogue_system"
export PYTHONPATH="$ROOT/dialogue_system:${PYTHONPATH:-}"
export TTS_API_URL DEMO_PORT DEMO_SSL_CERT DEMO_SSL_KEY
nohup "$PY" app.py >"$LOG_DIR/app.log" 2>&1 &
echo $! >"$LOG_DIR/app.pid"

sleep 3
echo
echo "Done. Open:  https://$DEMO_PUBLIC_HOST:$DEMO_PORT"
echo "             (browser will warn on self-signed cert — click Advanced → Proceed)"
echo "Stop with:   $0 stop"
echo "Logs:        $LOG_DIR/{tts,llm,vad,app}.log"
