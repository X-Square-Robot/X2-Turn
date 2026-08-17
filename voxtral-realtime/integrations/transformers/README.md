# Transformers loader

For a browser check, use the repository
[Quick start](../../../README.md#quick-start). This page is the Python loader.

The installable model definition lives in
[`src/voxtral_realtime/transformers/`](../../src/voxtral_realtime/transformers/).
It is a regular PyTorch `nn.Module` wrapper around the unmodified
`VoxtralRealtimeForConditionalGeneration`, with an independent full-vocabulary
`vad_lm_head`.

Install the optional dependencies from the `voxtral-realtime/` directory
without changing Transformers source code. This package is not published to
PyPI:

```bash
# from voxtral-realtime/
python -m pip install -e ".[transformers]"
```

## Load a local or Hugging Face checkpoint

The audio path in this snippet is relative to the X2-Turn repository root.

```python
import torch
from transformers import AutoProcessor

from voxtral_realtime.transformers import (
    infer_asr_turn,
    load_mtp_checkpoint,
)

model_id = "Kaiqfu/X2-Turn-4B-0812"
processor = AutoProcessor.from_pretrained(model_id)
model = load_mtp_checkpoint(
    model_id,
    device="cuda",
    dtype=torch.bfloat16,
).eval()

result = infer_asr_turn(model, processor, "turn-demo/assets/sample_en.wav")
print("ASR:", result.transcript)
for frame in result.turn_frames:
    print(frame.start_ms, frame.end_ms, frame.label, frame.confidence)
```

`load_mtp_checkpoint()` downloads `config.json` and `model.safetensors` when
given a Hub model ID. It then creates the stock Voxtral model, wraps it with
`VoxtralMTP`, and loads the canonical `base_model.*` plus
`vad_lm_head.weight` state dict. It does not register an AutoModel class and
does not require `trust_remote_code`.

`infer_asr_turn()` generates ASR with the stock model and performs a second
aligned forward pass over `vad_lm_head`. It returns the transcript, generated
token ids, and all six-class turn predictions on the 80 ms timeline.

The same loading example is available as a script. Run from
`voxtral-realtime/`:

```bash
python integrations/transformers/examples/load_checkpoint.py \
  --model Kaiqfu/X2-Turn-4B-0812
```

To transcribe one file and print all six-class turn predictions on the 80 ms
timeline (local Transformers, no vLLM):

```bash
python integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio ../turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

This local example first generates the aligned ASR sequence, then runs a second
forward pass through `vad_lm_head`. The second pass is required because stock
Transformers generation only uses the ASR head. Production realtime inference
should use the vLLM integration, which emits `turn.delta` incrementally.

Run the CPU-only wrapper tests from `integrations/transformers/` in an
environment containing PyTorch and Transformers:

```bash
# from voxtral-realtime/integrations/transformers/
python -m pytest -q tests/test_modeling_voxtral_mtp.py
```

Production vLLM serving does not import this wrapper. It uses the equivalent
turn-head loading and prediction logic in
[`../vllm/`](../vllm/README.md).
