# X2 Turn Demo

A standalone browser demo for inspecting X2 Turn without starting an LLM or
TTS service. Upload or record speech to see:

- streaming ASR text;
- one six-class turn prediction every 80 ms;
- the turn timeline and class histogram;
- a raw frame table with frame time, ASR token, Turn class, and probability.

The six model outputs are `idle`, `noidle`, `speaking`, `turn_end`,
`backchannel`, and `uncertain`.

## Install

These packages are not published to PyPI. From the `X2-Turn` repository root:

```bash
python -m pip install -e "./voxtral-realtime[transformers]"
python -m pip install -e "./turn-demo[dev]"
```

Alternatively, create the shared Miniforge environment from the monorepo root:

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn
```

## Start with local Transformers

Use either a Hugging Face model ID or a local checkpoint directory:

```bash
cd turn-demo
MODEL=Kaiqfu/X2-Turn-4B-0812 bash run.sh

# For a local or private checkpoint:
MODEL=/path/to/X2-Turn-4B-0812 bash run.sh
```

Open <http://localhost:7860>. The model loads on first startup. Set
`DEVICE=cpu` only for small tests; the 4B checkpoint is intended for a
CUDA-capable machine.

Select **[built-in] English question** and click **Run scenario** to test the
model without recording or uploading audio. The browser player below the preset
selector lets you hear the exact input before or after inference. The bundled
16 kHz mono sample is synthetic; its text, provenance, license, and regeneration
command are in [`assets/README.md`](assets/README.md).

The server binds to `127.0.0.1` by default and has no authentication. For
development TLS, provide a certificate and key:

```bash
HOST=0.0.0.0 \
SSL_CERTFILE=/path/to/cert.pem \
SSL_KEYFILE=/path/to/key.pem \
bash run.sh
```

For public deployments, prefer an authenticated HTTPS reverse proxy and do not
expose uploaded speech to an untrusted network.

## Start with realtime vLLM

Stock vLLM does not emit `turn.delta`. First start the patched runtime described
in the
[`voxtral-realtime` vLLM guide](../voxtral-realtime/integrations/vllm/README.md),
then run:

```bash
cd turn-demo
BACKEND=vllm \
VLLM_URL=ws://127.0.0.1:8011/v1/realtime \
VLLM_MODEL=Kaiqfu/X2-Turn-4B-0812 \
bash run.sh
```

The vLLM backend forwards microphone PCM to `/v1/realtime` and displays
incremental ASR and turn frames. The local Transformers backend repeatedly
decodes the accumulated microphone buffer and is intended for demonstration,
not latency benchmarking.

## Raw model outputs

The demo intentionally does not convert Turn states into product actions. It
shows the model's raw ASR text, six-class frame timeline, latest state, and
class histogram. The frame-level text analysis table remains available, but it
contains only frame time, ASR token, Turn class, and probability—no action or
decision column. Applications should define their own response, rejection, and
barge-in policies for their latency and interaction requirements.

Optional scenario JSONL files can be supplied with `--test_jsonl`. The built-in
synthetic Quickstart sample remains available without an external dataset.
Upload and microphone inference are also available. Uploads are limited to
20 MiB.

## Validate

```bash
pytest
```
