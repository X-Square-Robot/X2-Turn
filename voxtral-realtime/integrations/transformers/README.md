# Transformers model definition

`modeling_voxtral_mtp.py` is the training-compatible definition of the
shared-backbone Voxtral ASR + turn model. It adds an independent
full-vocabulary `vad_lm_head` to
`VoxtralRealtimeForConditionalGeneration`.

This integration is intentionally separate from the lightweight
`voxtral-realtime` runtime package because it requires PyTorch, Transformers,
and safetensors.

## Rebuild a canonical checkpoint

```python
import torch

from modeling_voxtral_mtp import load_mtp_checkpoint

model = load_mtp_checkpoint(
    "/path/to/voxtral-mtp-turn-v3-delay0-zhen/final",
    device="cuda",
    dtype=torch.bfloat16,
).eval()
```

Run the CPU-only wrapper tests from this directory in an environment containing
PyTorch and Transformers:

```bash
python -m pytest -q tests/test_modeling_voxtral_mtp.py
```

Production vLLM serving does not import this wrapper. It uses the equivalent
turn-head loading and prediction logic in
[`../vllm/`](../vllm/README.md).
