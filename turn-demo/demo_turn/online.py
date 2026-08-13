"""Online streaming session: 麦克风/分块 PCM → 增量 ASR + turn 状态。

实现说明
---------
本模块是本地 Transformers 后端的演示路径，使用:

  **HF ONLINE processor 分块 ingest + 对累计缓冲做增量解码**
  （适合观察短句的原始 ASR 与 Turn 状态；与 offline 对齐口径一致）

每 ``commit_ms``（默认 320ms）对当前缓冲跑一次 ``engine.infer_wav``，
把新的 ASR / turn 状态推给前端。

生产流式路径请使用带 X2 Turn overlay 的 vLLM ``/v1/realtime``；对应实现位于
``online_vllm.py``，可以直接接收 6 类 ``turn.delta``。
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from demo_turn.engine import TurnDemoEngine, UtterancePred
from demo_turn.viz import frame_table_html, timeline_html


@dataclass
class StreamUpdate:
    kind: str  # partial | final
    asr_text: str
    last_turn: str
    duration_s: float
    n_frames: int
    turn_hist: Dict[str, int]
    timeline_html: str
    frames_html: str
    turns: List[str]
    elapsed_infer_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OnlineTurnSession:
    """单路会话: push_pcm → (optional) StreamUpdate; finish → final update."""

    def __init__(
        self,
        engine: TurnDemoEngine,
        commit_ms: int = 320,
        max_buffer_s: float = 20.0,
        lock: Optional[threading.Lock] = None,
    ):
        self.engine = engine
        self.sr = int(engine.sr)
        self.commit_samples = max(
            int(self.sr * commit_ms / 1000.0), int(self.sr * 0.08)
        )
        self.max_buffer_samples = int(self.sr * max_buffer_s)
        self.lock = lock

        self._chunks: List[np.ndarray] = []
        self._n_samples = 0
        self._since_commit = 0
        self._started = time.time()
        self.last_update: Optional[StreamUpdate] = None

    def _wav(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks, axis=0)

    def push_pcm(self, pcm: np.ndarray, force: bool = False) -> Optional[StreamUpdate]:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return None
        self._chunks.append(pcm)
        self._n_samples += int(pcm.size)
        self._since_commit += int(pcm.size)

        # 滚动窗口, 避免超长句反复全量解码过慢
        if self._n_samples > self.max_buffer_samples:
            wav = self._wav()[-self.max_buffer_samples :]
            self._chunks = [wav]
            self._n_samples = int(wav.size)

        if (not force) and self._since_commit < self.commit_samples:
            return None
        if self._n_samples < int(self.sr * 0.25):
            return None
        return self._decode(kind="partial")

    def finish(self) -> StreamUpdate:
        # 句尾再补一点静音, 帮助 ASR delay / turn flush
        pad_s = max(self.engine.delay_ms / 1000.0, 0.48)
        if self.engine.turn_delay > 0:
            pad_s += self.engine.turn_delay * self.engine.seconds_per_token
        pad = np.zeros(int(self.sr * pad_s), dtype=np.float32)
        self._chunks.append(pad)
        self._n_samples += int(pad.size)
        return self._decode(kind="final")

    def _decode(self, kind: str) -> StreamUpdate:
        self._since_commit = 0
        wav = self._wav()
        t0 = time.time()
        if self.lock is not None:
            with self.lock:
                pred = self.engine.infer_wav(wav, wav_path="<online>")
        else:
            pred = self.engine.infer_wav(wav, wav_path="<online>")
        infer_ms = (time.time() - t0) * 1000.0
        update = self._pack(pred, kind=kind, infer_ms=infer_ms)
        self.last_update = update
        return update

    def _pack(self, pred: UtterancePred, kind: str, infer_ms: float) -> StreamUpdate:
        turns = [f.turn for f in pred.frames]
        hist = {}
        for t in turns:
            hist[t] = hist.get(t, 0) + 1
        return StreamUpdate(
            kind=kind,
            asr_text=pred.asr_text,
            last_turn=turns[-1] if turns else "idle",
            duration_s=pred.duration_s,
            n_frames=len(pred.frames),
            turn_hist=hist,
            timeline_html=timeline_html(
                turns,
                seconds_per_token=pred.seconds_per_token,
            ),
            frames_html=frame_table_html(pred.frames),
            turns=turns,
            elapsed_infer_ms=round(infer_ms, 1),
        )


def chunk_wav_for_online(
    wav: np.ndarray,
    sr: int,
    chunk_ms: int = 80,
):
    """把整段 wav 切成 online push 用的小块 (测试/回放)。"""
    n = int(sr * chunk_ms / 1000.0)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    for i in range(0, len(wav), n):
        yield wav[i : i + n]
