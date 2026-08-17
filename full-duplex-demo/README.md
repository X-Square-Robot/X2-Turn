# X Square Full-Duplex Dialogue Demo

This page is for people who already finished the repository
[Quick start](../README.md#quick-start) and now want a conversational stack.
It is not the first path in this repository.

The demo streams ASR and turn states, a reply LLM, TTS, and barge-in during
playback. The frontend comes from
[SoulX-Duplug dialogue-system](https://github.com/Soul-AILab/SoulX-Duplug/tree/dialogue-system).
Model weights, CosyVoice source, recordings, logs, and evaluation datasets are
not in this tree.

## Before you start

Work through this checklist first:

1. The Turn Demo from the root README produces the expected built-in result.
2. Patched vLLM is installed from
   [`voxtral-realtime/integrations/vllm/README.md`](../voxtral-realtime/integrations/vllm/README.md).
3. `VOXTRAL_VLLM_MODEL` is an **exported directory** that contains
   `consolidated.safetensors`, not the Hugging Face model ID.
4. You have separate environments for patched vLLM, this demo, and (unless
   `TTS_BACKEND=edge`) upstream CosyVoice. See
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
       └─ Streaming TTS HTTP :6017 (CosyVoice) or :6016 (Edge-TTS default)
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
python -m pip install -e '.[demo,llm]'
```

CosyVoice stays in its upstream environment. Do not copy that source tree into
this repository. Install the TTS extra there:

```bash
pip install -e '.[tts]'
pip install -e ./cosyvoice_vllm_plugin
```

## Model setup

Defaults may be replaced with local paths:

- Turn/ASR: `x-square-robot/X2-Turn-4B-0812`
- LLM: `Qwen/Qwen2.5-3B-Instruct`
- TTS: `FunAudioLLM/CosyVoice2-0.5B`

```bash
# from full-duplex-demo/
cp .env.example .env
# edit COSY_ROOT=/absolute/path/to/CosyVoice
# edit VOXTRAL_VLLM_MODEL=/absolute/path/to/X2-Turn-4B-0812-vllm
```

`VOXTRAL_MODEL`, `LLM_MODEL`, and `COSY_MODEL` accept hub IDs or directories.
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
bash start_demo.sh stop-all  # also stop Voxtral and TTS
```

Logs go under `logs/`. To attach the UI to services that are already running:

```bash
bash scripts/run_app.sh
```

Set `TTS_BACKEND=edge` to skip CosyVoice. Edge-TTS then listens on `:6016`
unless `TTS_PORT` is set. Copying `.env.example` sets `TTS_PORT=6017` for both
backends.

The launcher writes an audio-free turn trace to `logs/turn_trace.jsonl`. Keep
it private. Override or disable it with `VOXTRAL_TRACE_JSONL`.

## Containers

`docker-compose.yml` is a deployment template, not a turnkey image. Set
`DEMO_IMAGE` to an image that already contains this checkout. Inside Compose,
vLLM uses `--enforce-eager`, matching `serve.sh`.

```bash
COSY_ROOT=/path/to/CosyVoice docker compose --profile cosyvoice up
```

## Licensing

Apache License 2.0. Third-party software and models keep their own terms; see
`THIRD_PARTY_NOTICES.md`.

The X Square name and logo in `dialogue_system/frontend/x-square-logo.png` are
not licensed under Apache-2.0. Forks and redistributed products should remove
or replace the logo unless the owner has approved their use. Reasonable
attribution in this demo is allowed.

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) and [`SECURITY.md`](../SECURITY.md)
before reporting changes or issues.
