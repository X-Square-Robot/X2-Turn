---
license: apache-2.0
base_model: mistralai/Voxtral-Mini-4B-Realtime-2602
pipeline_tag: automatic-speech-recognition
language:
  - zh
  - en
library_name: vllm
tags:
  - realtime
  - turn-taking
---

# Voxtral MTP Turn

Proposed model ID: `x-square/voxtral-mtp-turn-v3-delay0-zhen`

This repository is a release staging area for the `final` checkpoint of
`voxtral-mtp-turn-v3-delay0-zhen`, a derivative of
`mistralai/Voxtral-Mini-4B-Realtime-2602`. It combines streaming automatic
speech recognition (ASR) with a frame-level turn-taking head.

> **Release gate:** the repository owner must review and approve the final release license
> and all training-data collection, processing, and
> redistribution rights before any public upload. The Apache-2.0 files in this
> staging repository do not establish rights to training data or to artifacts
> not yet added.

## Model behavior

Audio is processed on an approximately 80 ms frame timeline. The model emits:

- streaming ASR tokens for Chinese and English speech; and
- one turn-taking output per frame from `idle`, `noidle`, `speaking`,
  `turn_end`, and `backchannel`.

The training head also reserves token id 40 for `uncertain`. The production
turn controller does not expose that class as an application action.

The ASR and turn heads share the acoustic/language backbone and run on the same
frame sequence. Turn outputs are predictions, not deterministic voice-activity
or endpoint guarantees. Applications should apply a separate turn controller
or policy rather than treating a single frame as an irrevocable action.

## Intended use

- Research and evaluation of low-latency Mandarin, English, and mixed-language
  streaming ASR.
- Turn-taking experiments such as endpointing, barge-in, response timing, and
  backchannel handling.
- Controlled deployment behind an application-level policy, monitoring, and
  human-reviewed privacy and safety controls.

This model is not intended as the sole basis for safety-critical decisions,
speaker identity, emotion inference, legal transcription, or covert
surveillance.

## Limitations

- Accuracy may degrade with noise, reverberation, overlap, accents, dialects,
  code-switching, far-field microphones, packet loss, or domains unlike the
  training data.
- Frame outputs may flicker or lag and require temporal smoothing and
  application-specific thresholds.
- ASR errors and turn errors can interact: incomplete transcription does not
  necessarily imply an incomplete turn, and silence does not guarantee that a
  speaker has yielded.
- Real-time latency and throughput depend on hardware, serving configuration,
  audio transport, and policy buffering.
- Language coverage declared in this card is Chinese and English; it does not
  imply equal performance across languages or varieties.

## Privacy

Speech can contain personal, biometric, confidential, and copyrighted
information. Obtain appropriate consent and legal authority, minimize
collection and retention, encrypt transport and storage, restrict access, and
provide deletion and incident-response procedures. Do not log raw audio or
transcripts by default. The owner must complete a training-data privacy and
rights review before release.

## Bias and fairness

Performance may vary by accent, dialect, age, gender presentation, speaking
style, disability, microphone, environment, language mix, and conversational
norms. Turn-taking conventions and backchannels are culturally and
contextually dependent. Evaluate representative subgroups and interaction
settings, document material gaps, and avoid using model outputs to infer
sensitive attributes.

## Evaluation

The following existing results were traced to the selected
`voxtral-mtp-turn-v3-delay0-zhen/final` checkpoint. They are included as
provisional evidence, not as a final release claim:

- EasyTurn Chinese, vLLM streaming, 480 ms ASR delay: 579 scored utterances,
  89.46% endpoint-category accuracy; 29,936 aligned frames from 533 eligible
  examples reached 93.17% frame accuracy.
- EasyTurn English, 480 ms ASR delay: 617 utterances reached 88.49%
  last-non-idle endpoint-category accuracy.
- Full-Duplex-Bench Chinese turn-taking subset: turn-end detection was 94.19%
  over 155 examples. This is an offline turn-state surrogate, and its mean
  timing was 309 ms before the annotated speech end.
- Full-Duplex-Bench Chinese interruption subset: interruption detection was
  99.38% over 161 examples, with 160 ms median and 480 ms p90 latency.
- FastTurn stratified set: 800 utterances reached 71.13% category accuracy,
  including 90% backchannel, 69% complete, 61% incomplete, and 89% wait.

These evaluations used an 80 ms frame rate and `turn_label_delay_frames=0`.
EasyTurn and FastTurn provenance and redistribution rights still require
release-owner review. Full-Duplex-Bench is CC BY-NC 4.0 and is not distributed
with this Apache-2.0 model repository. The reported Full-Duplex-Bench numbers
must not be interpreted as granting commercial rights to its data or code.

- ASR datasets and splits: **TBD before release**
- ASR metrics (for example, CER/WER): **TBD before release**
- Turn-taking datasets, label policy, and splits: **TBD before release**
- Per-class and timing metrics: **TBD before release**
- Streaming latency and hardware/software configuration: **TBD before release**
- Subgroup, robustness, and mixed-language analysis: **TBD before release**

All reported results must identify dataset rights, preprocessing, frame
alignment, decoding settings, policy thresholds, hardware, and package
versions.

## Model definition

The `voxtral-realtime` code repository contains the training-side definition at
[`integrations/transformers/modeling_voxtral_mtp.py`](../voxtral-realtime/integrations/transformers/modeling_voxtral_mtp.py).
`VoxtralMTP` wraps
`VoxtralRealtimeForConditionalGeneration` with:

- the original `lm_head` for streaming ASR;
- an independent full-vocabulary `vad_lm_head` initialized from `lm_head`;
- joint ASR and turn losses, including safe all-masked turn batches; and
- a turn-head-only mode that freezes the shared backbone in the graph.

Rebuild the canonical `final` checkpoint with:

```python
import torch

from modeling_voxtral_mtp import load_mtp_checkpoint

model = load_mtp_checkpoint(
    "/path/to/voxtral-mtp-turn-v3-delay0-zhen/final",
    device="cuda",
    dtype=torch.bfloat16,
).eval()
```

This definition is intended for checkpoint inspection, evaluation, and
training-compatible reconstruction. Production serving still uses the pinned
vLLM overlay described below.

## Serving

Serving support lives in the separate `voxtral-realtime` package; this model
repository does not bundle the vLLM server. Use a version of that package that
explicitly supports the MTP turn head, and follow its installation and server
documentation. Do not assume that generic `transformers` or stock vLLM loading
will automatically select the custom wrapper.

Before serving, add the checkpoint's validated runtime metadata and the
complete artifact set described below; keep `config.example.json` as
documentation only. Pin the serving package version and validate output-label
ordering end-to-end.

## Expected release artifacts

The local staging directory may contain the selected canonical
`model.safetensors` from `voxtral-mtp-turn-v3-delay0-zhen/final`, but the
private GitLab validation repository excludes it. Do not upload the weight
until every ownership, training-data, privacy, and license gate in
`MODEL_RELEASE_CHECKLIST.md` has been approved.

1. **Model weights:** `model.safetensors` containing the base model under the
   `base_model.*` parameter namespace and the separate
   `vad_lm_head.weight` turn head. If an approved release process shards this
   file, include all generated shards and `model.safetensors.index.json`; do
   not mix unsharded and sharded layouts.
2. **Architecture metadata:** `config.json` for the Hugging Face Voxtral
   backbone and `params.json` for the Mistral/vLLM architecture metadata.
   `config.example.json` documents the custom wrapper contract but is not a
   runtime configuration.
3. **Tokenizer:** `tekken.json`.
4. **Audio/processor metadata:** `processor_config.json`.
5. **Generation defaults:** `generation_config.json`.
6. **Repository documentation:** `README.md`, `LICENSE`, `NOTICE`, and
   `MODEL_RELEASE_CHECKLIST.md`.

Do not upload optimizer states, trainer states, datasets, raw audio, local
paths, credentials, logs, caches, or intermediate checkpoints.

## Configuration contract

The inspected source checkpoint metadata identifies
`VoxtralRealtimeForConditionalGeneration`, `model_type: voxtral_realtime`,
`audio_length_per_tok: 8`, `default_num_delay_tokens: 6`, 16 kHz audio, and a
12.5 Hz frame rate. It does not itself declare the additional turn head.

`config.example.json` records the machine-readable wrapper contract. Its
executable definition is versioned with `voxtral-realtime`, alongside the
offline and vLLM inference integrations. The example configuration remains
intentionally non-loadable and must not replace the checkpoint's real
`config.json`.

## License and attribution

See `LICENSE` and `NOTICE`. The model weight is derivative of the named Mistral
base model. Final licensing and training-data rights remain release blockers
requiring owner review.
