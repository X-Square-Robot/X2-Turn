# Custom vLLM inference overlay

Stock vLLM does **not** emit `turn.delta`. This overlay is derived from an
Apache-2.0 vLLM 0.19.1 checkout pinned at commit
`b1388b1fbf5aaef47937fabe98931211684666a6` and adds the Voxtral MTP turn head
and realtime event propagation.

## Reproduce a source checkout

```bash
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

The canonical open-source training checkpoint is the directory named `final`
for `voxtral-mtp-turn-v3-delay0-zhen`. vLLM should load its generated sibling
`final_vllm`, which uses Mistral keys in `consolidated.safetensors` and includes
`vad_lm_head.weight`.

```bash
python integrations/vllm/tools/export_mtp_for_vllm.py \
  --src /path/to/voxtral-mtp-turn-v3-delay0-zhen/final \
  --dst /path/to/voxtral-mtp-turn-v3-delay0-zhen/final_vllm \
  --base /path/to/Voxtral-Mini-4B-Realtime-2602
```

`--base` is optional only when `final` already contains `params.json` and
`tekken.json`. Do not copy the HF `config.json` into `final_vllm`; vLLM uses
`params.json` for the required audio configuration.

Serve `final_vllm` while exposing the public model name expected by the bridge:

```bash
MODEL=/path/to/final_vllm bash integrations/vllm/examples/voxtral_mtp/serve.sh
```
