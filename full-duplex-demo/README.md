# X Square Full-Duplex Dialogue Demo

This page is for people who already finished the repository
[Quick start](../README.md#quick-start) and now want a conversational stack.
It is not the first path in this repository.

The demo streams ASR and turn states, a reply LLM, TTS, and barge-in during
playback. The frontend comes from
[SoulX-Duplug dialogue-system](https://github.com/Soul-AILab/SoulX-Duplug/tree/dialogue-system).
Model weights, Qwen3TTS-Streaming, recordings, logs, and evaluation datasets
are not in this tree.

## Before you start

Work through this checklist first:

1. The Turn Demo from the root README produces the expected built-in result.
2. Patched vLLM is installed from
   [`voxtral-realtime/integrations/vllm/README.md`](../voxtral-realtime/integrations/vllm/README.md).
3. `VOXTRAL_VLLM_MODEL` is an **exported directory** that contains
   `consolidated.safetensors`, not the Hugging Face model ID.
4. A local Qwen3TTS-Streaming standalone engine is running, and
   `QWEN3TTS_CLIENT_SRC` points to its `client/src` directory. See
   [`../environments/README.md`](../environments/README.md).
5. Linux, extra GPU memory, `curl`, and `openssl`.

## Service map

```text
Browser (HTTPS :8443)
  └─ dialogue_system/app.py
       ├─ Turn WebSocket :8000
       │    └─ voxtral-realtime bridge
       │         └─ patched vLLM realtime :8011
       ├─ Streaming LLM HTTP :6007
       └─ Qwen3TTS-Streaming WebSocket :50052 (local)
```

Each browser session keeps its own ASR buffer, LLM history, and a generation
epoch. Stale LLM/TTS chunks from an interrupted turn are discarded. The demo
owns orchestration only; ASR, the acoustic gate, and the frame controller live
in `voxtral-realtime`.

Local `start_demo.sh` binds these ports to `127.0.0.1` unless `BIND_HOST` is
set. Compose publishes the same ports from container `0.0.0.0`.

## How turns are used

The model emits one of six labels every 80 ms: `idle`, `noidle`, `speaking`,
`turn_end`, `backchannel`, `uncertain`. `idle` between those labels is normal.
The app does **not** treat a single frame as an action.

While TTS is playing, the demo stops playback only after audio has started
**and** `turn_class == speaking`. `noidle` is not an interrupt. Endpoint
confirmation, ASR tail wait, and backchannel policy are in
`voxtral_realtime.turn.controller`.

## Install

The Python packages here are not on PyPI. From `full-duplex-demo/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ../voxtral-realtime
python -m pip install -e '.[llm]'
```

The Qwen3TTS Python SDK can be installed, or loaded directly by setting
`QWEN3TTS_CLIENT_SRC=/path/to/Qwen3TTS-Streaming/client/src`.

### Build and start local Qwen3TTS-Streaming

The TensorRT engine is GPU-specific; a prebuilt `model.plan` from another GPU
may not work. Build and deploy the recommended `custom-1.7b` engine on the
target machine:

```bash
git clone https://github.com/X-Square-Robot/Qwen3TTS-Streaming.git
cd Qwen3TTS-Streaming
bash scripts/bash/autorun.sh all -m custom-1.7b
curl -f http://127.0.0.1:50052/v1/capabilities
```

Follow that repository's prerequisites if the build reports missing
TensorRT, CUDA, Git LFS, or model access. If the engine package is already
built, this repository also provides a small standalone launcher:

```bash
cd full-duplex-demo
cp .env.example .env
# Set QWEN3TTS_ROOT, ENGINE_PYTHON, ENGINE_MODEL_PACKAGE_DIR and QWEN3_TTS_GPU.
bash scripts/start_qwen3tts_engine.sh
```

## Model setup

Defaults may be replaced with local paths:

- Turn/ASR: `x-square-robot/X2-Turn-4B-0812`
- LLM: `Qwen/Qwen2.5-3B-Instruct`
- TTS: local Qwen3TTS-Streaming `custom-1.7b`

```bash
# from full-duplex-demo/
cp .env.example .env
# edit QWEN3TTS_ROOT and QWEN3TTS_CLIENT_SRC
# edit VOXTRAL_VLLM_MODEL=/absolute/path/to/X2-Turn-4B-0812-vllm
```

`VOXTRAL_MODEL` and `LLM_MODEL` accept hub IDs or directories.
`start_demo.sh` will not start vLLM unless `VOXTRAL_VLLM_MODEL` is a directory
containing `consolidated.safetensors`. A healthy vLLM already on `:8011` is
reused. The GPU for that service is `TURN_GPU` (`VAD_GPU` is a legacy alias).

## Start

```bash
# from full-duplex-demo/, after editing .env
bash start_demo.sh
```

Open `https://localhost:8443`. The script may create a development-only
self-signed certificate. For a remote host, set `DEMO_PUBLIC_HOST`,
`DEMO_SSL_CERT`, and `DEMO_SSL_KEY`, and use a trusted certificate.

Set `BIND_HOST=0.0.0.0` only when another machine must connect.

```bash
bash start_demo.sh stop      # stop bridge, LLM, and app
bash start_demo.sh stop-all  # also stop the locally launched Voxtral service
```

Logs go under `logs/`. To attach the UI to services that are already running:

```bash
bash scripts/run_app.sh
```

The local Qwen3TTS-Streaming engine must be healthy before the launcher starts
the dialogue app.

The launcher writes an audio-free turn trace to `logs/turn_trace.jsonl`. Keep
it private. Override or disable it with `VOXTRAL_TRACE_JSONL`.

## Containers

`docker-compose.yml` is a deployment template, not a turnkey image. Set
`DEMO_IMAGE` to an image that already contains this checkout. Inside Compose,
vLLM uses `--enforce-eager`, matching `serve.sh`.

The Compose template connects to a Qwen3TTS-Streaming engine running on the
host at `host.docker.internal:50052`. Set `QWEN3TTS_ROOT` to the host checkout
so the app container can mount the SDK, and set `VOXTRAL_VLLM_MODEL` to the
exported model directory. Override `QWEN3_TTS_DOCKER_WS_URL` if the engine is
not running on the Compose host.

## Licensing

Apache License 2.0. Third-party software and models keep their own terms; see
`THIRD_PARTY_NOTICES.md`.

The X Square name and logo in `dialogue_system/frontend/x-square-logo.png` are
not licensed under Apache-2.0. Forks and redistributed products should remove
or replace the logo unless the owner has approved their use. Reasonable
attribution in this demo is allowed.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`SECURITY.md`](../SECURITY.md)
before reporting changes or issues.
