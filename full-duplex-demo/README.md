# X Square Full-Duplex Dialogue Demo

A browser-based, full-duplex speech dialogue pipeline with streaming
transcription and turn prediction, streaming LLM output, low-latency TTS, and
barge-in playback control.

The frontend and dialogue service were originally based on
[SoulX-Duplug dialogue-system](https://github.com/Soul-AILab/SoulX-Duplug/tree/dialogue-system).
Model weights, CosyVoice source, recordings, logs, and evaluation datasets are
not distributed in this repository.

## Service map

```text
Browser (HTTPS :8443)
  └─ dialogue_system/app.py
       ├─ Turn WebSocket :8000
       │    └─ voxtral-realtime bridge
       │         └─ patched vLLM realtime :8011
       ├─ Streaming LLM HTTP :6007
       └─ Streaming TTS HTTP :6017
```

Detailed design:

- [`docs/ARCHITECTURE.zh.md`](docs/ARCHITECTURE.zh.md): end-to-end pipeline,
  generation epochs, streaming TTS, and current limitations.
- [`docs/STATE_MACHINE.md`](docs/STATE_MACHINE.md): 80 ms turn labels,
  endpoint confirmation, acoustic veto, and backchannel policy.
- [`docs/BRAND_ASSETS.md`](docs/BRAND_ASSETS.md): logo and trademark boundary.

## Requirements

- Linux, Python 3.10+, `curl`, and `openssl`
- NVIDIA GPUs and compatible CUDA libraries for the default model stack
- The standalone `voxtral-realtime>=0.1.0` Python package and CLI
- An external [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) checkout
  when using the default TTS backend

Install the core demo and the components needed on each service environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ../voxtral-realtime
pip install -e '.[demo,llm]'

# In the separate environment used for CosyVoice:
pip install -e '.[tts]'
pip install -e ./cosyvoice_vllm_plugin
```

For Miniforge users, the monorepo provides separate dialogue and patched-vLLM
environments in [`../environments/`](../environments/README.md). CosyVoice
should still use its upstream-recommended environment.

Follow the upstream CosyVoice installation instructions inside its own
checkout. Do not copy that source tree into this repository.

## Model setup

Defaults are public model IDs and may be replaced with local paths:

- Turn/ASR: `Kaiqfu/X2-Turn-4B-0812`
- LLM: `Qwen/Qwen2.5-3B-Instruct`
- TTS: `FunAudioLLM/CosyVoice2-0.5B`

Copy the environment template and point `COSY_ROOT` at the external checkout:

```bash
cp .env.example .env
# edit COSY_ROOT=/absolute/path/to/CosyVoice
```

`VOXTRAL_MODEL`, `LLM_MODEL`, and `COSY_MODEL` accept either hub IDs or model
directories. Review every model's license and access requirements separately.

The model hub repository contains the canonical Hugging Face `final`
checkpoint. The patched vLLM runtime requires a generated `final_vllm`
directory. Follow `../voxtral-realtime/integrations/vllm/README.md` to apply the
pinned vLLM overlay and export the weights, then set:

```bash
VOXTRAL_VLLM_MODEL=/path/to/voxtral-mtp-turn-v3-delay0-zhen/final_vllm
```

## Quickstart

```bash
bash start_demo.sh
```

Open `https://localhost:8443`. The script creates a development-only,
self-signed localhost certificate if none exists. Use a trusted certificate
and set `DEMO_PUBLIC_HOST`, `DEMO_SSL_CERT`, and `DEMO_SSL_KEY` for remote
deployment.

The launcher starts and health-checks TTS, LLM, the Voxtral server, the turn
bridge, and the web app. Existing healthy Voxtral and TTS services are reused.

`docker-compose.yml` is a deployment template, not a turnkey image build. Set
`DEMO_IMAGE` to an image that already contains this checkout and its Python
dependencies before using Compose. The source-based `start_demo.sh` path above
is the supported quickstart.

```bash
bash start_demo.sh stop      # stop bridge, LLM, and app
bash start_demo.sh stop-all  # also stop Voxtral and TTS
```

Logs are written under `logs/`. To run only the frontend/app against existing
services, set `TURN_API_URL`, `LLM_API_URL`, and `TTS_API_URL`, then run:

```bash
bash scripts/run_app.sh
```

Set `TTS_BACKEND=edge` to use the lightweight Edge TTS fallback without
`COSY_ROOT`.

## Containers

`docker-compose.yml` separates Voxtral, turn bridge, LLM, TTS, and app
services, with configurable GPU IDs. Set `DEMO_IMAGE` to an image containing
this checkout and the selected dependency extras. Start CosyVoice with its
profile:

```bash
COSY_ROOT=/path/to/CosyVoice docker compose --profile cosyvoice up
```

Compose is a deployment template; CUDA images and model caches vary by host.

## Offline inference

The demo does not bundle evaluation datasets or a second offline dialogue
pipeline. For a standalone PCM WAV example that produces ASR and turn-state
JSON, use
[`../voxtral-realtime/examples/README.md`](../voxtral-realtime/examples/README.md).

## Licensing and brand assets

Code is provided under Apache License 2.0. Third-party software and models
retain their own terms; see `THIRD_PARTY_NOTICES.md`.

The included X Square logo and related brand assets are not granted under the
Apache License. Their use requires owner approval except for reasonable
attribution. See `docs/BRAND_ASSETS.md`.

See `CONTRIBUTING.md` and `SECURITY.md` before reporting changes or issues.
