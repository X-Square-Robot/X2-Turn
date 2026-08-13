<div align="center">
  <h1>
    <img
      src="full-duplex-demo/dialogue_system/frontend/x-square-logo.png"
      alt="X Square mascot"
      width="72"
      align="center"
    >
    X2-Turn
  </h1>
  <p>
    <strong>Frame-synchronous streaming ASR and Turn-state prediction</strong>
  </p>
  <p>
    One Voxtral-based model, two synchronized outputs, and one Turn prediction
    every 80 ms.
  </p>
  <p>
    <a href="https://huggingface.co/Kaiqfu/X2-Turn-4B-0812"><img src="https://img.shields.io/badge/Hugging%20Face-X2--Turn--4B--0812-yellow" alt="Hugging Face model"></a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="Apache-2.0">
  </p>
</div>

[English](README.md) | [Chinese](README_zh.md)

X2 Turn extends Voxtral Realtime with two synchronized outputs: streaming
automatic speech recognition and one turn-taking prediction every 80 ms.

The turn head predicts `idle`, `noidle`, `speaking`, `turn_end`,
`backchannel`, or `uncertain`. Applications should smooth these frame-level
predictions instead of treating a single frame as an irreversible action.

## Quick start: one audio file

Recommended Miniforge setup, run from the repository root:

```bash
conda env create -f environments/environment-transformers.yml
conda activate x2-turn
```

Or install the local Transformers integration with pip:

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

result = infer_asr_turn(model, processor, "turn-demo/assets/sample_en.wav")

print("ASR:", result.transcript)
for frame in result.turn_frames:
    print(frame.start_ms, frame.end_ms, frame.label, frame.confidence)
```

The published Hub model is `Kaiqfu/X2-Turn-4B-0812`. For offline or private
deployments, `model_id` can instead be a local `voxtral-mtp-turn/final`
directory.

For a ready-to-run command that also writes JSON:

```bash
python voxtral-realtime/integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio turn-demo/assets/sample_en.wav \
  --output offline_frames.json
```

The bundled sample is synthetic 16 kHz mono speech. Its text, provenance,
license, and reproducible FFmpeg command are documented in
[`turn-demo/assets/README.md`](turn-demo/assets/README.md).

## Repository layout

- [`voxtral-realtime/`](voxtral-realtime/README.md) contains the model wrapper,
  local ASR + turn inference, the realtime controller, and the patched vLLM
  integration. Start here when integrating the model into another project.
- [`voxtral-mtp-turn/`](voxtral-mtp-turn/README.md) is the Model Hub release
  staging directory: Model Card, configuration, tokenizer metadata, release
  checks, and optionally the approved weights.
- [`turn-demo/`](turn-demo/README.md) is the focused browser demo for testing
  raw ASR, 80 ms Turn states, and the frame-level token/class/probability table
  without an LLM, TTS service, or product decision policy.
- [`full-duplex-demo/`](full-duplex-demo/README.md) is the browser-based
  full-duplex dialogue demo that combines X2 Turn with optional LLM and TTS
  services.

[`environments/`](environments/README.md) provides separate Miniforge
environments for local Transformers inference, patched vLLM, and the
full-duplex dialogue stack. Keeping these environments separate avoids most
Torch and CUDA dependency conflicts.

For realtime serving, follow the
[`vLLM integration guide`](voxtral-realtime/integrations/vllm/README.md).
Stock vLLM does not emit the custom `turn.delta` events.

## Release boundary

Each component retains its own license and notices so it can be published
separately. The canonical model weight may exist locally, but this source
repository ignores it. Publish weights to the approved Model Hub only after
completing
[`MODEL_RELEASE_CHECKLIST.md`](voxtral-mtp-turn/MODEL_RELEASE_CHECKLIST.md).
Never publish local logs, certificates, datasets, external checkouts, or
credentials.
