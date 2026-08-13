# Full-Duplex Demo Architecture

This demo combines continuous microphone input, incremental ASR, turn-taking
decisions, streaming LLM inference, streaming TTS, and user interruption in a
single real-time pipeline.

## Data Flow

```text
Browser microphone (16 kHz PCM)
  → dialogue_system/app.py
  → voxtral-realtime /turn bridge
  → patched vLLM /v1/realtime
  → ASR text + 80 ms turn labels
  → Qwen streaming HTTP
  → CosyVoice or Edge-TTS streaming HTTP
  → Browser SharedArrayBuffer + AudioWorklet
```

Service boundaries:

- `:8011`: vLLM Realtime service with the MTP overlay.
- `:8000`: `voxtral-realtime` turn bridge.
- `:6007`: replaceable streaming LLM service.
- `:6017`: CosyVoice TTS; the Edge-TTS fallback uses `:6016` by default.
- `:8443`: FastAPI WebSocket and static frontend.

The demo owns only application orchestration and adapters. ASR/turn inference,
the acoustic gate, and the frame controller belong to the separate
`voxtral-realtime` package. Model weights and third-party service weights are
not distributed with the demo.

## Sessions and Generation

Each browser WebSocket corresponds to an independent session that maintains:

- a turn bridge client and incremental ASR state;
- LLM conversation history;
- the current generation epoch and cancellation event;
- pending assistant text and its estimated playback duration;
- a lock that serializes WebSocket sends.

When a new turn begins or a valid interruption occurs, the generation epoch is
incremented. Even if stale LLM/TTS tasks return later, they are discarded
because their epoch is no longer current, preventing delayed PCM from being
mixed into a new response.

## LLM and TTS Pipeline

The LLM producer continuously generates text and segments it by punctuation and
length. The TTS consumer reads text segments from a bounded queue and produces
24 kHz, mono, 16-bit PCM. The session enters `bot_speaking` only after the first
PCM chunk is sent; the browser plays audio from a shared ring buffer through an
AudioWorklet.

This boundary serves two purposes:

1. TTS does not block the generation of subsequent LLM text.
2. The epoch is checked again before each PCM chunk is sent, ensuring safe
   cancellation.

## User Interruption

The browser explicitly requests echo cancellation and noise suppression. The
application clears the audio only after TTS playback has begun and the turn
model has continuously classified the user as speaking. Residual echo during
generation therefore does not prematurely cancel a first TTS packet that has
not yet arrived.

When a valid interruption occurs:

1. Set the current cancellation event and increment the epoch.
2. Send `stop_audio` to the browser and clear the playback buffer.
3. Stop consuming the stale LLM/TTS output.
4. Commit assistant history in proportion to the audio already played.
5. Start the next generation after accepting the complete new user turn.

For detailed turn labels and endpoint rules, see
[`STATE_MACHINE.md`](STATE_MACHINE.md).

## Deployment and Dependencies

`start_demo.sh` can start the complete service stack and reuse a healthy vLLM
or TTS service. All model paths, GPU indices, and ports are provided through
`.env` or environment variables. Production deployments should use trusted TLS
certificates; automatically generated certificates are suitable only for local
development.

The CosyVoice source must be installed as an external checkout and should not
be copied into this repository. Model weights, runtime logs, certificates,
recordings, and evaluation datasets are likewise outside the public source
boundary.

## Current Limitations

- ASR may still produce homophone errors, duplicate transcriptions, and missing
  characters at the end of an utterance.
- Endpoint speed and false-cut rates must be tuned for each use case.
- TTS time to first packet depends on the backend, GPU load, and warm-up state.
- Browser echo cancellation quality varies by device and browser implementation.
- Standardized accuracy and latency should be reproduced in a separate
  evaluation repository rather than embedding datasets in the demo.
