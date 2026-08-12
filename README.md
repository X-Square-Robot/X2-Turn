# X2 Turn

X2 Turn extends Voxtral Realtime with two synchronized outputs: streaming
automatic speech recognition and one turn-taking prediction every 80 ms.

The turn head predicts `idle`, `noidle`, `speaking`, `turn_end`,
`backchannel`, or `uncertain`. Applications should smooth these frame-level
predictions instead of treating a single frame as an irreversible action.

## Quick start: one audio file

Install the local Transformers integration:

```bash
cd voxtral-realtime
python -m pip install -e ".[transformers]"
```

Load the model without modifying Transformers or enabling
`trust_remote_code`, then obtain both the transcript and all aligned turn
frames:

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

result = infer_asr_turn(model, processor, "/path/to/input.wav")

print("ASR:", result.transcript)
for frame in result.turn_frames:
    print(frame.start_ms, frame.end_ms, frame.label, frame.confidence)
```

Until the model is available on the Hub, `model_id` can be a local
`voxtral-mtp-turn/final` directory.

For a ready-to-run command that also writes JSON:

```bash
python integrations/transformers/examples/offline_inference.py \
  --model /path/to/voxtral-mtp-turn \
  --audio /path/to/input.wav \
  --output offline_frames.json
```

## Repository layout

- [`voxtral-realtime/`](voxtral-realtime/README.md) contains the model wrapper,
  local ASR + turn inference, the realtime controller, and the patched vLLM
  integration. Start here when integrating the model into another project.
- [`voxtral-mtp-turn/`](voxtral-mtp-turn/README.md) is the Model Hub release
  staging directory: Model Card, configuration, tokenizer metadata, release
  checks, and optionally the approved weights.
- [`full-duplex-demo/`](full-duplex-demo/README.md) is the browser-based
  full-duplex dialogue demo that combines X2 Turn with optional LLM and TTS
  services.

For realtime serving, follow the
[`vLLM integration guide`](voxtral-realtime/integrations/vllm/README.md).
Stock vLLM does not emit the custom `turn.delta` events.

## Validate

```bash
cd voxtral-realtime
python -m pip install -e ".[dev]"
pytest
python scripts/check_public_release.py

cd ../voxtral-mtp-turn
python verify_model_repo.py --allow-weights

cd ../full-duplex-demo
python -m pip install -e ../voxtral-realtime
pytest
python scripts/check_public_release.py
```

## Release boundary

Each component retains its own license and notices so it can be published
separately. The canonical model weight may exist locally, but this GitLab
repository ignores it. Publish weights to the approved Model Hub only after
completing
[`MODEL_RELEASE_CHECKLIST.md`](voxtral-mtp-turn/MODEL_RELEASE_CHECKLIST.md).
Never publish local logs, certificates, datasets, external checkouts, or
credentials.
