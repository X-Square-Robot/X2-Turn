# Custom vLLM inference overlay

Do this after the Turn Demo in the repository
[Quick start](../../../README.md#quick-start) works. Stock vLLM does **not**
emit `turn.delta`.

This overlay is derived from an Apache-2.0 vLLM 0.19.1 checkout pinned at commit
`b1388b1fbf5aaef47937fabe98931211684666a6` and adds the Voxtral MTP turn head
and realtime event propagation.

Run every command in this file from the `voxtral-realtime/` directory unless a
snippet says otherwise.

## Reproduce a source checkout

```bash
# from voxtral-realtime/
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout b1388b1fbf5aaef47937fabe98931211684666a6 && cd ..
bash scripts/install_vllm_overlay.sh ./vllm
```

The installer refuses a different commit, version, or a dirty checkout, checks
that the patch applies, then installs the new model utility, export tool, and
examples. It intentionally does not install dependencies or model weights.

The patch contains all ten tracked modifications (179 insertions, 6 deletions):
realtime connection/protocol, model registry/implementation, output types,
scheduler, engine exports/output processor, and GPU model runner. New source is
under `integrations/vllm/`; `tests/test_overlay.py` guards the expected file set.

## Convert canonical weights

Download `Kaiqfu/X2-Turn-4B-0812` from the Model Hub or use an equivalent local
checkpoint directory. vLLM should load the exported directory, which uses
Mistral keys in `consolidated.safetensors` and includes `vad_lm_head.weight`.

```bash
# from voxtral-realtime/
python integrations/vllm/tools/export_mtp_for_vllm.py \
  --src /path/to/X2-Turn-4B-0812 \
  --dst /path/to/X2-Turn-4B-0812-vllm \
  --base /path/to/Voxtral-Mini-4B-Realtime-2602
```

`--base` is optional only when the source checkpoint already contains
`params.json` and `tekken.json`. Do not copy the HF `config.json` into the
exported directory; vLLM uses `params.json` for the required audio
configuration.

Serve the exported directory while exposing the public model name expected by
the bridge. The server binds to `127.0.0.1` by default; set `HOST=0.0.0.0` only
when another machine must connect.

```bash
# from voxtral-realtime/
MODEL=/path/to/X2-Turn-4B-0812-vllm \
  bash integrations/vllm/examples/voxtral_mtp/serve.sh
```

A smoke client is documented in
[`examples/voxtral_mtp/README.md`](examples/voxtral_mtp/README.md).
