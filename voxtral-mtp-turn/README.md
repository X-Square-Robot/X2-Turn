---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
pipeline_tag: automatic-speech-recognition
language:
  - zh
  - en
library_name: transformers
tags:
  - realtime
  - turn-taking
---

# X2 Turn 4B

X2 Turn listens to speech and produces two synchronized results:

- streaming Chinese/English transcription; and
- one prediction every 80 ms describing whether the user is silent, still
  speaking, finished, or only giving a short acknowledgment.

It is designed for voice assistants that need to decide when to wait, reply,
ignore a backchannel, or allow the user to interrupt.

Model ID: `Kaiqfu/X2-Turn-4B-0812`

## Quick start

The model uses a small wrapper from the `voxtral-realtime` code repository. It
does not modify Transformers and does not require `trust_remote_code`.

```bash
cd /path/to/x2_turn_opensource
python -m pip install -e "./voxtral-realtime[transformers]"
```

The integration requires Transformers 5.10 or newer. The
`transformers_version` stored in the checkpoint JSON records the export
environment; it is not the supported minimum runtime version.

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

`result.transcript` is the recognized text. `result.turn_frames` contains the
turn prediction and confidence for each 80 ms frame.

The loader also accepts a local checkpoint directory. A complete command-line
example is available in the `voxtral-realtime` repository:

```bash
python integrations/transformers/examples/offline_inference.py \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --audio /path/to/input.wav \
  --output offline_frames.json
```

## What the six outputs mean

- `idle`: no useful speech is present.
- `noidle`: acoustic activity is present, but intent is not yet clear.
- `speaking`: the user is still speaking.
- `turn_end`: the user appears to have finished and the assistant may reply.
- `backchannel`: a short acknowledgment such as “嗯”, “对”, or “okay”.
- `uncertain`: the model is not confident that the turn has ended.

These are predictions, not commands. Applications should smooth several frames
and apply a policy instead of acting on one frame.

## When to use it

Good fits include:

- low-latency Mandarin, English, and mixed-language ASR;
- deciding when a voice assistant should respond;
- distinguishing a real request from a backchannel;
- endpointing, barge-in, and response-timing experiments; and
- controlled research or product evaluation with monitoring.

Do not use this model as the sole basis for safety-critical decisions, speaker
identity, emotion inference, legal transcription, or covert surveillance.

## Results at a glance

The following provisional results were traced to this checkpoint:

- EasyTurn Chinese: 89.46% endpoint-category accuracy over 579 scored
  utterances; 93.17% frame accuracy over 29,936 aligned frames.
- EasyTurn English: 88.49% last-non-idle endpoint-category accuracy over 617
  utterances.
- Full-Duplex-Bench Chinese turn-taking: 94.19% turn-end detection over 155
  examples.
- Full-Duplex-Bench Chinese interruption: 99.38% interruption detection over
  161 examples, with 160 ms median and 480 ms p90 latency.
- FastTurn stratified set: 71.13% category accuracy over 800 utterances.

All results used an 80 ms frame rate and `turn_label_delay_frames=0`. They are
provisional evidence rather than final release claims. EasyTurn and FastTurn
provenance and redistribution rights still require owner review.
Full-Duplex-Bench is CC BY-NC 4.0 and is not distributed with this repository.

Evaluation details still required for a final public release:

- ASR datasets and splits: **TBD before release**
- ASR metrics such as CER/WER: **TBD before release**
- Turn-taking datasets, label policy, and splits: **TBD before release**
- Per-class and timing metrics: **TBD before release**
- Streaming latency and hardware/software configuration: **TBD before release**
- Subgroup, robustness, and mixed-language analysis: **TBD before release**

## Limitations

- Noise, reverberation, overlapping speakers, accents, dialects, code-switching,
  far-field microphones, and packet loss may reduce accuracy.
- Turn predictions may flicker or arrive early/late and need temporal
  smoothing.
- ASR errors and turn errors can interact. Incomplete text does not always mean
  an incomplete turn, and silence does not guarantee that the user yielded.
- Performance may vary across demographic groups, speaking styles, languages,
  microphones, environments, and conversational norms.
- Real-time latency depends on hardware, serving configuration, transport, and
  policy buffering.

Speech may contain personal, biometric, confidential, or copyrighted
information. Obtain appropriate consent, minimize collection and retention,
protect stored data, and avoid logging raw audio or transcripts by default.

## Realtime serving

Production realtime serving uses the separate `voxtral-realtime` package and a
pinned vLLM overlay. Stock vLLM does not emit the custom `turn.delta` events.
Follow the vLLM integration guide in the code repository before serving.

For a focused browser visualization of raw ASR, 80 ms Turn frames, and the
frame-level token/class/probability table, use the standalone `turn-demo`
component. It intentionally applies no response or barge-in policy. The full
dialogue demo additionally connects an LLM and TTS.

## Technical details

The checkpoint is derived from
`mistralai/Voxtral-Mini-4B-Realtime-2602`. It keeps the original shared
backbone and ASR `lm_head`, and adds an independent full-vocabulary
`vad_lm_head`.

Turn labels occupy reserved tokenizer IDs 35 through 40 in this exact order:
`idle`, `noidle`, `speaking`, `turn_end`, `backchannel`, `uncertain`.

The canonical `model.safetensors` stores:

- the backbone and ASR head under `base_model.*`; and
- the turn head as `vad_lm_head.weight`.

The checkpoint keeps the stock Voxtral `config.json`. The
`voxtral_realtime.transformers` loader creates the `VoxtralMTP` wrapper before
loading both heads. `config.example.json` documents this contract but is not a
loadable replacement for `config.json`.

The initial release should remain unsharded because the repository loader
targets the canonical single-file layout. Runtime metadata also includes
`params.json`, `tekken.json`, `processor_config.json`, and
`generation_config.json`.

## Release status and license

The legacy proposed ID `x-square/voxtral-mtp-turn-v3-delay0-zhen` refers to the
same selected `voxtral-mtp-turn-v3-delay0-zhen/final` checkpoint. The canonical
published Hub repository is `Kaiqfu/X2-Turn-4B-0812`.

The release owner remains responsible for the final release license, Mistral
base-model terms, attribution, and all training-data collection, processing,
privacy, consent, and redistribution rights. The Apache-2.0 files in this
staging repository do not by themselves establish rights to the model weights
or training data.

Do not upload optimizer state, trainer state, datasets, raw audio, transcripts,
credentials, logs, caches, private paths, or intermediate checkpoints. See
`MODEL_RELEASE_CHECKLIST.md`, `LICENSE`, and `NOTICE`.
