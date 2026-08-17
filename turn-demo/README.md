# X2 Turn Demo

Follow the repository [Quick start](../README.md#quick-start) first. This page
is extra detail: flags, TLS, microphone upload, and the optional vLLM backend.

The demo shows streaming ASR, one six-class turn prediction every 80 ms, a
timeline and histogram, and a raw frame table. It does not start an LLM or TTS
service, and it does not turn frames into product actions.

The six labels are `idle`, `noidle`, `speaking`, `turn_end`, `backchannel`, and
`uncertain`.

## Local Transformers

This is the same path as the root Quick start. From the repository root,
activate `x2-turn`, then:

```bash
cd turn-demo
MODEL=x-square-robot/X2-Turn-4B-0812 bash run.sh

# After huggingface-cli download, or a private checkpoint:
MODEL=/path/to/X2-Turn-4B-0812 bash run.sh
```

Open <http://localhost:7860>. The 4B checkpoint needs about **24 GB+ VRAM**.
The model loads on the first **Run scenario**, upload, or microphone request,
not when the process starts. That first inference can take several minutes.

Choose **[built-in] English question** and click **Run scenario**. You do not
need a microphone. Typical numbers for that clip are in the root
[Quick start](../README.md#quick-start). Sample provenance is in
[`assets/README.md`](assets/README.md).

The server binds to `127.0.0.1` and has no authentication. For development TLS:

```bash
HOST=0.0.0.0 \
SSL_CERTFILE=/path/to/cert.pem \
SSL_KEYFILE=/path/to/key.pem \
bash run.sh
```

Do not expose uploaded speech on an untrusted network. Prefer an authenticated
HTTPS reverse proxy for anything beyond localhost.

`DEVICE=cpu` is only for tiny tests. Uploads are capped at 20 MiB. Optional
scenario JSONL files can be passed with `--test_jsonl`.

## After Transformers works: realtime vLLM

Stock vLLM does not emit `turn.delta`. Start the patched runtime in
[`voxtral-realtime/integrations/vllm/README.md`](../voxtral-realtime/integrations/vllm/README.md),
then:

```bash
cd turn-demo
BACKEND=vllm \
VLLM_URL=ws://127.0.0.1:8011/v1/realtime \
VLLM_MODEL=x-square-robot/X2-Turn-4B-0812 \
bash run.sh
```

The vLLM backend forwards microphone PCM to `/v1/realtime`. The Transformers
backend re-decodes the accumulated buffer and is for inspection, not latency
benchmarks.

## Validate

```bash
pytest
```
