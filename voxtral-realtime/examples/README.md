# Offline ASR and turn inference

`offline_inference.py` replays one PCM WAV through the same acoustic gate,
realtime vLLM client, and frame-level turn controller used by the `/turn`
service. It does not start the browser demo, LLM, or TTS.

## Prerequisites

1. Install this package:

   ```bash
   pip install -e .
   ```

2. Apply the pinned vLLM overlay and convert the canonical checkpoint by
   following [`../integrations/vllm/README.md`](../integrations/vllm/README.md).

3. Start patched vLLM:

   ```bash
   MODEL=/path/to/X2-Turn-4B-0812-vllm \
     bash integrations/vllm/examples/voxtral_mtp/serve.sh
   ```

## Run

From the `voxtral-realtime` directory:

```bash
python examples/offline_inference.py \
  --audio /path/to/input.wav \
  --output-dir offline_output \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --vllm-url ws://127.0.0.1:8011/v1/realtime
```

The input must be an uncompressed PCM WAV. Multichannel audio is mixed to mono
and resampled to 16 kHz. The default 80 ms chunks match the model turn-frame
period. Add `--realtime` to preserve wall-clock pacing, or `--bot-speaking` to
evaluate the WAV as a barge-in attempt.

## Outputs

- `states.json`: one row per audio chunk, including `time_ms`, controller
  state, turn class, incremental ASR, acoustic activity, event, and reason when
  available.
- `transcript.json`: accepted utterances with their endpoint time and reason.

The script appends 1.2 seconds of silence by default so endpoint confirmation
can complete. Adjust this with `--flush-ms`.
