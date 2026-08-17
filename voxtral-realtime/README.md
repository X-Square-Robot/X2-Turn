# Voxtral Realtime

This package is the model wrapper and realtime `/turn` bridge. New users should
finish the repository [Quick start](../README.md#quick-start) before serving
vLLM.

The bridge does realtime ASR and frame-level turn taking with Voxtral on
vLLM's `/v1/realtime` WebSocket API. Install it from this checkout; it is not
on PyPI.

## Two inference entry points

Do not confuse these scripts. Both mention offline WAV input, but they do
different work:

| Script | Needs patched vLLM? | What it does |
|--------|---------------------|--------------|
| [`integrations/transformers/examples/offline_inference.py`](integrations/transformers/examples/offline_inference.py) | No | Local Transformers ASR + 80 ms turn frames |
| [`examples/offline_inference.py`](examples/offline_inference.py) | Yes | Replay a WAV through the production `/turn` controller |

For a single-file transcript without a server, use the Transformers script.
That is the path shown in the repository-root README.

## Architecture

```text
application WebSocket (/turn, float32 PCM)
  -> AcousticVoiceGate (energy endpoint veto)
  -> RealtimeVLLMSession (vLLM protocol, ASR and turn.delta)
  -> FrameTurnController (barge-in and endpoint state machine)
  -> turn_state response
```

The client has no dependency on demo UI or policy code. It suppresses leading
idle transcription and uses a configurable energy lead-in gate. The controller
preserves these rules: `noidle` alone cannot barge in, `speaking` and `turn_end`
can barge while the bot is speaking, and endpoint candidates retain a short ASR
tail before acceptance.

## Install

Run from the `voxtral-realtime/` directory:

```bash
python -m pip install -e .
# development
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required.
For a ready-made Miniforge environment, see
[`../environments/environment-transformers.yml`](../environments/environment-transformers.yml).

## Model definition

The training-compatible `VoxtralMTP` wrapper and manual checkpoint loader live
in the optional `voxtral_realtime.transformers` module:

```bash
python -m pip install -e ".[transformers]"
```

See [`integrations/transformers/`](integrations/transformers/README.md) for
local and Hugging Face loading examples. The loader uses the unmodified stock
Transformers Voxtral implementation; it does not require `trust_remote_code`.

## vLLM prerequisite

Standard vLLM cannot emit `turn.delta`. Apply the pinned Apache-2.0-compatible
overlay documented in [`integrations/vllm/README.md`](integrations/vllm/README.md)
before serving. That guide owns the pinned vLLM revision, overlay installation,
checkpoint export, and serving commands. Model weights are distributed
separately.

The bridge defaults to `ws://127.0.0.1:8011/v1/realtime`. Configure the served
model and URL without embedding local model paths:

```bash
export VOXTRAL_MODEL_ID=x-square-robot/X2-Turn-4B-0812
export VLLM_URL=ws://127.0.0.1:8011/v1/realtime
```

## CLI

The server binds to `127.0.0.1` by default. Pass `--host 0.0.0.0` only when
another machine must connect.

```bash
voxtral-realtime serve --host 127.0.0.1 --port 8000
voxtral-realtime serve --model x-square-robot/X2-Turn-4B-0812 \
  --vllm-url ws://127.0.0.1:8011/v1/realtime
```

See `.env.example` for common environment settings.

Set `VOXTRAL_TRACE_JSONL=/private/path/turn_trace.jsonl` to append a versioned
record for every consumed turn frame. The trace contains probabilities,
incremental ASR, acoustic activity, bot state, and controller output, but no raw
audio. Tracing is disabled by default in the core package.

## Replay a WAV through the turn bridge

Replay a PCM WAV through the production acoustic gate and turn controller
without starting the dialogue demo. Patched vLLM must already be running:

```bash
python examples/offline_inference.py \
  --audio ../turn-demo/assets/sample_en.wav \
  --output-dir offline_output
```

See [`examples/README.md`](examples/README.md) for vLLM prerequisites, output
schemas, realtime pacing, and barge-in evaluation.

## Python API

```python
from voxtral_realtime import RealtimeConfig, TurnBridge, create_app

config = RealtimeConfig.from_env()
app = create_app(TurnBridge(config))
```

For a direct backend connection, instantiate `RealtimeVLLMSession`, call
`connect()`, stream one-dimensional float32 arrays with `push_pcm()`, consume
`turn.delta` rows with `consume_turn_frames()`, and finally call `finish()` and
`close()`.

## Application WebSocket protocol

Connect to `ws://HOST:PORT/turn`. Audio is little-endian float32 PCM (normally
16 kHz), base64 encoded in JSON.

Audio request:

```json
{"type":"audio","session_id":"demo","audio":"BASE64_FLOAT32","bot_speaking":false}
```

Control request:

```json
{"type":"control","session_id":"demo","bot_speaking":true}
```

Audio responses use this shape:

```json
{"type":"turn_state","session_id":"demo","state":{"state":"nonidle","turn_class":"speaking","asr_buffer":"hello"},"ts":0.0}
```

Controller states are `idle`, `nonidle`, or `speak`. A `speak` state includes
`event: "accept"` and finalized `text`; `event: "barge_in"` indicates a valid
interruption. Invalid messages receive `{ "type": "error", "error": "..." }`.

## License and attribution

Apache-2.0. See `NOTICE` and `THIRD_PARTY_NOTICES.md` for derivation and model
weight notices.
