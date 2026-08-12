# X2 Turn Demo

A standalone browser demo for inspecting X2 Turn without starting an LLM or
TTS service. Upload or record speech to see:

- streaming ASR text;
- one six-class turn prediction every 80 ms;
- the turn timeline and class histogram;
- `ACCEPT`, `REJECT`, `HOLD`, and simulated barge-in decisions.

The six model outputs are `idle`, `noidle`, `speaking`, `turn_end`,
`backchannel`, and `uncertain`.

## Install

From the `x2_turn_opensource` monorepo:

```bash
python -m pip install -e "voxtral-realtime[transformers]"
python -m pip install -e "turn-demo[dev]"
```

## Start with local Transformers

Use either a Hugging Face model ID or a local checkpoint directory:

```bash
cd turn-demo
MODEL=Kaiqfu/X2-Turn-4B-0812 bash run.sh

# Before the Hub upload is available:
MODEL=/path/to/voxtral-mtp-turn bash run.sh
```

Open <http://localhost:7860>. The model loads on first startup. Set
`DEVICE=cpu` only for small tests; the 4B checkpoint is intended for a
CUDA-capable machine.

## Start with realtime vLLM

Stock vLLM does not emit `turn.delta`. First start the patched runtime described
in the
[`voxtral-realtime` vLLM guide](../voxtral-realtime/integrations/vllm/README.md),
then run:

```bash
cd turn-demo
BACKEND=vllm \
VLLM_URL=ws://127.0.0.1:8010/v1/realtime \
VLLM_MODEL=x2-turn-vllm \
bash run.sh
```

The vLLM backend forwards microphone PCM to `/v1/realtime` and displays
incremental ASR and turn frames. The local Transformers backend repeatedly
decodes the accumulated microphone buffer and is intended for demonstration,
not latency benchmarking.

## Decision policy

- While the bot is not speaking, the final non-idle class determines the
  result: `turn_end` becomes `ACCEPT`, `backchannel` becomes `REJECT`, and
  `speaking`/`uncertain`/`noidle` becomes `HOLD`.
- While simulated bot TTS is active, four consecutive `noidle` or `speaking`
  frames trigger barge-in by default. A `backchannel` does not interrupt.
- These are demo defaults, not universal product thresholds.

Optional scenario JSONL files can be supplied with `--test_jsonl`. No dataset,
audio, or internal evaluation path is bundled with this repository.

## Validate

```bash
pytest
python scripts/check_public_release.py
```
