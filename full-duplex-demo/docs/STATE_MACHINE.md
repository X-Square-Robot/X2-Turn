# X Square Demo State Transition Logic (Based on the Voxtral 80 ms Turn Stream)

Model: `Kaiqfu/X2-Turn-4B-0812`
Frame interval: **80 ms per state** (`audio_length_per_tok=8`, 16 kHz, hop=160 → 8×160/16000 = 0.08 s)

Current demo stack:

```
Browser (:8443)
  └─ dialogue_system/app.py
       ├─ Turn ws://127.0.0.1:8000/turn   ← voxtral-realtime
       │         └─ vLLM ws://127.0.0.1:8011/v1/realtime
       ├─ LLM  http://127.0.0.1:6007/chat
       └─ TTS  http://127.0.0.1:6017/tts
```

---

## 1. What Is the Granularity of the Model Output?

### 1.1 Inference: One State per 80 ms Frame

During streaming inference, the Voxtral MTP turn head shares the ASR timeline, which advances in **80 ms latency-stream frames**.
Each vLLM Realtime `turn.delta` includes a `frame_index` identifying the corresponding **80 ms frame** and one of six states:

| ID | Class | Meaning (training semantics) |
|----|-------|------------------------------|
| 35 | `idle` | No meaningful turn-taking signal; **frame-level padding** between characters, between sentences, or during silence |
| 36 | `noidle` | Speech is present in this 80 ms window (coarse acoustic signal) |
| 37 | `speaking` | The user is speaking and the utterance is not complete |
| 38 | `turn_end` | The system may take the turn; end of utterance |
| 39 | `backchannel` | Acknowledgment that does not constitute a complete turn |
| 40 | `uncertain` | Uncertain state retained from training; the application layer takes no direct action |

This is **not** one state per character. ASR still emits `transcription.delta` events by character or word, while turn states form a **parallel sequence of 80 ms frames**.

### 1.2 What Do the Training Labels Look Like?

The training code is not distributed with the inference repository. The public label contract is:

1. Create an array whose length equals the total number of generated frames, initialized entirely to **`idle`**.
2. Write each character's or word's turn class (bc / noidle / speaking / turn_end) into the **frame interval to which ASR aligns it**.
3. Leave inter-character gaps, PAD frames, and trailing silence as **`idle`**.

As a result, an actual gold timeline looks like this (illustrative):

```
Frame: idle idle idle | bc  | idle | noidle | idle | speaking speaking | turn_end    | idle idle
       ─── silence ──   ack.   gap    speech    gap    speaking          utterance end  trailing silence
```

Key points:

- **`idle` may appear between bc / noidle / speaking / turn_end labels**. This is expected labeling behavior, not noise.
- A rule such as "four consecutive non-idle frames" is invalid because any intervening `idle` frame resets the count.
- Decisions must operate on the **80 ms frame sequence with a frame-by-frame state machine**, not on the "last non-idle character."

---

## 2. Implemented: Frame-Level Controller

Implementation files:

| File | Purpose |
|------|---------|
| `voxtral_realtime.turn.controller` | Frame-level controller with tail wait and backchannel policy |
| `voxtral_realtime.server` | WebSocket bridge that invokes the controller frame by frame |
| `voxtral_realtime.realtime` | `consume_turn_frames()` drains frames individually |
| `dialogue_system/clients/vad_client.py` | Turn-bridge client; attaches `bot_speaking` to each audio packet |
| `dialogue_system/app.py` | Sets `bot_speaking=True` on the first TTS packet and clears it on interruption |

### 2.1 Default Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `N` (`end_confirm_frames`) | 1 | After `turn_end`, wait N additional frames (×80 ms) without resumed speech, then start the ASR tail |
| `K` (`silence_end_frames`) | 3 | After semantic speech has begun, K model-idle frames trigger a soft endpoint, then the ASR tail |
| `tail_min_frames` / `tail_max_frames` | 2 / 5 | Keep the session 160–400 ms so late ASR characters can arrive before ACCEPT |
| `tail_stable_frames` | 2 | Unchanged ASR text required after the minimum tail |
| `backchannel_confirm_frames` | 2 | Consecutive sentence-initial `backchannel` frames before the class is confirmed |
| `backchannel_fallback_idle_frames` | 3 | Idle frames after a confirmed backchannel that may ACCEPT or reject a filler |
| `commit_ms` | 80 | Polling/gating interval while the bot is not speaking |
| `barge_commit_ms` | 80 | Switches to an 80 ms interval and skips the lead-in gate **while the bot is speaking** |
| `min_asr_chars` | 1 | Minimum number of ASR characters required before ACCEPT |
| `acoustic_vad_max_hold_frames` | 8 | Acoustic activity may veto an endpoint for at most 640 ms |

Configuration is provided through `voxtral-realtime` environment variables, for
example `VOXTRAL_BARGE_COMMIT_MS` and `VOXTRAL_TAIL_MIN_FRAMES`.

### 2.2 Rules (Applied to Each 80 ms Frame)

```python
SPEECH = {noidle, speaking}
BARGE_IMMEDIATE = {speaking, turn_end}   # While the bot is speaking
```

**① TTS is playing (`bot_speaking=True`)**

- Switch to `barge_commit_ms=80`, **skip lead-in energy gating**, and send user audio to vLLM immediately.
- The controller sets `event=barge_in` for `speaking` and `turn_end`. `noidle` stays `state=nonidle` without that event.
- The demo application stops playback only after PCM has begun **and** `turn_class == "speaking"`. A `turn_end` barge event is therefore not enough by itself.
- The decision still comes from the **turn model**, not a frontend RMS threshold.

**② Backchannel**

- Sentence-initial `backchannel` must run for `backchannel_confirm_frames` before it is confirmed.
- Confirmed backchannel while the bot is speaking is rejected (`event=reject`).
- After confirmation, trailing idle frames may ACCEPT (`backchannel_fallback`) unless the ASR is a pure filler listed in `PURE_BACKCHANNEL_FILLERS`.
- Once semantic speech has started, `backchannel` is ignored and does not advance the soft-endpoint countdown.

**③ Hard endpoint (the model emits `turn_end` while listening)**

- On `turn_end`, enter the pending state and count **N frames**.
- If `noidle`/`speaking` occurs while pending, cancel the pending state and continue listening.
- When the N-frame countdown expires and ASR is nonempty, start the ASR tail (`reason=turn_end_confirmed`). ACCEPT uses `turn_end_confirmed_tail`.

**④ Soft endpoint (fallback when `turn_end` is missing)**

- After entering a semantic speech segment, observe **K consecutive** model-idle frames (`idle`).
- Acoustic activity can veto that countdown for up to `acoustic_vad_max_hold_frames`.
- If ASR is nonempty, start the ASR tail (`reason=silence_end`). ACCEPT uses `silence_end_tail`.

**⑤ Other SPEECH states**

- `speaking` / `noidle` → `nonidle` (HOLD with streaming ASR)
- `uncertain` does not barge, does not start speech, and does not count toward silence

### 2.3 Mapping to the Three X Square States

| Controller output | Bridge `state` | Application behavior |
|-------------------|----------------|----------------------|
| idle / reject | `idle` | No action |
| hold / pending / tail | `nonidle` | Streaming ASR display. This is **not** an interrupt by itself. |
| `event=barge_in` | `nonidle` | Interrupt candidate. The app currently interrupts only when `bot_speaking` and `turn_class == "speaking"`. |
| speak / accept | `speak` + text | `pipeline_worker` → LLM→TTS |

### 2.4 Example Timeline

```
Frame:     idle idle | bc bc | idle idle idle | noidle | speaking … | turn_end | idle | idle idle
X Square:  idle idle | pending/confirm | fallback | nonidle | nonidle … | pending_end | tail_start | speak
```

If the model does not emit `turn_end`:

```
Frame:     … speaking speaking | idle idle idle …
X Square:  … nonidle …         | silence_run 1..3 → tail → speak (silence_end_tail)
```

---

## 3. Application Layer (L3)

Logic in `dialogue_system/app.py`:

| Bridge | Application |
|--------|-------------|
| `nonidle` without `barge_in` | Streaming ASR display |
| `nonidle` + `event=barge_in` and `turn_class=speaking` while TTS is playing | `interrupt()` |
| `speak` | `pipeline_worker(text)` |
| `idle` | No action |

**Synchronizing `bot_speaking`:**

1. When the first TTS `audio_chunk` is sent: `session.bot_speaking = True`, `vad.set_bot_speaking(True)`
2. On user barge-in / `interrupt()`: `bot_speaking = False`
3. For every microphone chunk: `vad.process(chunk, bot_speaking=session.bot_speaking)`

The bridge can also receive `type=control` + `bot_speaking` as a fallback.

---

## 4. Source Code Index

| File | Contents |
|------|----------|
| `voxtral_realtime.realtime` | vLLM streaming session + `consume_turn_frames()` |
| `voxtral_realtime.turn.controller` | **Live frame state machine** |
| `voxtral_realtime.server` | X Square WebSocket bridge |
| `dialogue_system/app.py` | L3 barge-in / LLM pipeline / bot_speaking |

---

## 5. Deployment Notes

- Turn/ASR model: `Kaiqfu/X2-Turn-4B-0812`
- TTS: CosyVoice2 `:6017` (`TTS_API_URL`); Edge-TTS `:6016` unless `TTS_PORT` is set
- UI: `https://localhost:8443`

Example bridge startup command:

```bash
VOXTRAL_END_CONFIRM_FRAMES=1 \
VOXTRAL_SILENCE_END_FRAMES=3 \
voxtral-realtime serve \
  --host 127.0.0.1 \
  --model Kaiqfu/X2-Turn-4B-0812 \
  --vllm-url ws://127.0.0.1:8011/v1/realtime
```
