"""vLLM realtime backend: ASR + 6-class turn via ``/v1/realtime``.

Requires the Voxtral-MTP fork (``vllm-voxtral-mtp``) serving a checkpoint
that includes ``vad_lm_head`` (for example, ``X2-Turn-4B-0812-vllm``).
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import os
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from demo_turn.predictions import (
    TURN_CLASS_IDS,
    TURN_CLASS_NAMES,
    FramePred,
    UtterancePred,
    load_audio,
)

TURN_ID_TO_CLASS = dict(zip(TURN_CLASS_IDS, TURN_CLASS_NAMES, strict=True))


def align_turn_preds_to_asr(
    turn_ids: list[int], delay_frames: int
) -> list[int]:
    if delay_frames <= 0:
        return turn_ids
    idle = TURN_CLASS_IDS[0]
    return turn_ids[delay_frames:] + [idle] * min(delay_frames, len(turn_ids))


def pad_wav_for_turn_delay(
    wav: np.ndarray,
    delay_frames: int,
    sample_rate: int,
    seconds_per_token: float,
) -> np.ndarray:
    if delay_frames <= 0:
        return wav
    padding = np.zeros(
        round(delay_frames * seconds_per_token * sample_rate),
        dtype=np.float32,
    )
    return np.concatenate([wav, padding])

# Same order as vLLM ``TURN_CLASS_NAMES`` / ``turn.delta.probs``.
_TURN_NAMES: List[str] = [TURN_ID_TO_CLASS[i] for i in TURN_CLASS_IDS]

_SPEECH_TURNS: Set[str] = {
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
}

DEFAULT_VLLM_URL = "ws://127.0.0.1:8011/v1/realtime"
DEFAULT_SECONDS_PER_TOKEN = 0.08  # 8 mel hops @ 16 kHz
CHUNK_BYTES = 4096
DEFAULT_RECV_TIMEOUT_S = 60.0


def _probs_dict(probs: Optional[Sequence[float]]) -> Dict[str, float]:
    out = {n: 0.0 for n in _TURN_NAMES}
    if not probs:
        out["idle"] = 1.0
        return out
    for i, name in enumerate(_TURN_NAMES):
        if i < len(probs):
            out[name] = float(probs[i])
    return out


def _pcm16_bytes(wav: np.ndarray) -> bytes:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    pcm = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def _run_coro(coro):
    """Run ``coro`` even if a loop is already running (e.g. FastAPI async)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


async def stream_wav_to_vllm(
    wav: np.ndarray,
    *,
    url: str,
    model: str,
    sr: int = 16000,
    chunk_bytes: int = CHUNK_BYTES,
    timeout_s: float = DEFAULT_RECV_TIMEOUT_S,
    suppress_idle_text: bool = True,
) -> tuple[str, List[dict]]:
    """Stream float32 wav → (asr_text, list of turn-frame dicts)."""
    import websockets

    pcm = _pcm16_bytes(wav)
    frames: List[dict] = []
    parts: List[str] = []
    pending_text = ""
    speech_opened = not suppress_idle_text
    done = asyncio.Event()
    err: Optional[BaseException] = None

    async with websockets.connect(url, max_size=None) as ws:
        created = json.loads(await ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"unexpected handshake: {created}")

        await ws.send(json.dumps({"type": "session.update", "model": model}))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        async def _recv() -> None:
            nonlocal pending_text, speech_opened, err
            try:
                async for message in ws:
                    data = json.loads(message)
                    mtype = data.get("type")
                    if mtype == "transcription.delta":
                        pending_text = data.get("delta") or ""
                    elif mtype == "turn.delta":
                        turn = data.get("turn_class") or "idle"
                        text = pending_text
                        keep_text = text
                        if suppress_idle_text and not speech_opened:
                            if turn in _SPEECH_TURNS:
                                speech_opened = True
                            else:
                                keep_text = ""
                        if keep_text:
                            parts.append(keep_text)
                        frames.append(
                            {
                                "turn": turn,
                                "turn_id": int(data.get("turn_id") or 35),
                                "probs": data.get("probs"),
                                "frame_index": data.get("frame_index"),
                                "text": keep_text,
                            }
                        )
                        pending_text = ""
                    elif mtype == "transcription.done":
                        if not parts and data.get("text"):
                            parts.append(data["text"])
                        done.set()
                        return
                    elif mtype == "error":
                        err = RuntimeError(data)
                        done.set()
                        return
            except BaseException as e:  # noqa: BLE001
                err = e
                done.set()

        reader = asyncio.create_task(_recv())
        try:
            for i in range(0, len(pcm), chunk_bytes):
                chunk = pcm[i : i + chunk_bytes]
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode(),
                        }
                    )
                )
            await ws.send(
                json.dumps({"type": "input_audio_buffer.commit", "final": True})
            )
            await asyncio.wait_for(done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError as e:
            reader.cancel()
            raise TimeoutError(
                f"vLLM transcription.done timed out after {timeout_s:.0f}s"
            ) from e
        finally:
            if not reader.done():
                reader.cancel()
                try:
                    await reader
                except (asyncio.CancelledError, Exception):
                    pass

    if err is not None:
        raise err
    return "".join(parts), frames


def frames_to_utterance(
    *,
    wav_path: str,
    asr_text: str,
    turn_frames: List[dict],
    duration_s: float,
    seconds_per_token: float,
    delay_ms: int,
    turn_label_delay_frames: int,
) -> UtterancePred:
    """Build the same ``UtterancePred`` shape as the HF engine."""
    raw_ids = [int(f.get("turn_id") or 35) for f in turn_frames]
    aligned_ids = align_turn_preds_to_asr(raw_ids, turn_label_delay_frames)
    score_len = (
        max(len(aligned_ids) - turn_label_delay_frames, 0)
        if turn_label_delay_frames > 0
        else len(aligned_ids)
    )

    frames: List[FramePred] = []
    for f in range(score_len):
        tid = aligned_ids[f]
        turn = TURN_ID_TO_CLASS.get(tid, "idle")
        src = f + turn_label_delay_frames if turn_label_delay_frames > 0 else f
        src = min(src, len(turn_frames) - 1) if turn_frames else 0
        src_f = turn_frames[src] if turn_frames else {}
        pd = _probs_dict(src_f.get("probs"))
        p = float(pd.get(turn, 0.0))
        t0 = f * seconds_per_token
        frames.append(
            FramePred(
                frame=f,
                t0=t0,
                t1=t0 + seconds_per_token,
                asr=str(src_f.get("text") or ""),
                turn=turn,
                turn_prob=p,
                probs=pd,
            )
        )

    return UtterancePred(
        wav_path=wav_path,
        asr_text=asr_text,
        seconds_per_token=seconds_per_token,
        delay_ms=delay_ms,
        turn_label_delay_frames=turn_label_delay_frames,
        frames=frames,
        duration_s=duration_s,
    )


class TurnDemoVLLMEngine:
    """Drop-in offline engine that talks to a running vLLM realtime server."""

    def __init__(
        self,
        vllm_url: str = DEFAULT_VLLM_URL,
        model: str = "",
        delay_ms: int = 480,
        turn_label_delay_frames: int = 0,
        seconds_per_token: float = DEFAULT_SECONDS_PER_TOKEN,
        sr: int = 16000,
        suppress_idle_text: bool = True,
        recv_timeout_s: float = DEFAULT_RECV_TIMEOUT_S,
    ):
        self.vllm_url = vllm_url
        self.model = model
        self.delay_ms = int(delay_ms)
        self.turn_delay = max(0, int(turn_label_delay_frames))
        self.seconds_per_token = float(seconds_per_token)
        self.sr = int(sr)
        self.suppress_idle_text = bool(suppress_idle_text)
        self.recv_timeout_s = float(recv_timeout_s)
        if not self.model:
            raise ValueError("--vllm-model / model path is required for backend=vllm")
        print(
            f"[demo] vLLM backend url={vllm_url} model={model} "
            f"delay_ms={self.delay_ms} turn_D={self.turn_delay}",
            flush=True,
        )

    def infer_file(self, wav_path: str) -> UtterancePred:
        wav = load_audio(wav_path, sample_rate=self.sr)
        return self.infer_wav(wav, wav_path=wav_path)

    def infer_wav(self, wav, wav_path: str = "<memory>") -> UtterancePred:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        duration_s = float(len(wav) / self.sr)
        if self.turn_delay > 0:
            wav = pad_wav_for_turn_delay(
                wav, self.turn_delay, self.sr, self.seconds_per_token
            )
        asr_text, turn_frames = _run_coro(
            stream_wav_to_vllm(
                wav,
                url=self.vllm_url,
                model=self.model,
                sr=self.sr,
                timeout_s=self.recv_timeout_s,
                suppress_idle_text=self.suppress_idle_text,
            )
        )
        return frames_to_utterance(
            wav_path=wav_path,
            asr_text=asr_text,
            turn_frames=turn_frames,
            duration_s=duration_s,
            seconds_per_token=self.seconds_per_token,
            delay_ms=self.delay_ms,
            turn_label_delay_frames=self.turn_delay,
        )


def resolve_vllm_model(model: str) -> str:
    """Prefer ``*_vllm`` export dir when present next to an HF ``final/``."""
    if not model:
        return model
    if not os.path.isabs(model):
        # leave relative; caller may still pass absolute served path
        pass
    if model.rstrip("/").endswith("_vllm") or os.path.isfile(
        os.path.join(model, "consolidated.safetensors")
    ):
        return model
    sibling = model.rstrip("/") + "_vllm"
    if os.path.isdir(sibling) and os.path.isfile(
        os.path.join(sibling, "consolidated.safetensors")
    ):
        return sibling
    return model
