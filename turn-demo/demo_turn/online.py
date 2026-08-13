"""Online streaming session: 麦克风/分块 PCM → 增量 ASR + turn 决策。

实现说明
---------
本模块是本地 Transformers 后端的演示路径，使用:

  **HF ONLINE processor 分块 ingest + 对累计缓冲做增量解码**
  （短句/体验打断拒识足够；与 offline 对齐口径一致）

每 ``commit_ms``（默认 320ms）对当前缓冲跑一次 ``engine.infer_wav``，
把新的 ASR / turn / policy 事件推给前端。

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
from demo_turn.policy import DemoDecision, PolicyConfig, run_policy_on_frames
from demo_turn.viz import decision_banner, events_table, timeline_html


@dataclass
class StreamUpdate:
    kind: str  # partial | final
    asr_text: str
    action: str
    last_turn: str
    reason: str
    barge_in_at_s: Optional[float]
    duration_s: float
    n_frames: int
    turn_hist: Dict[str, int]
    banner_html: str
    timeline_html: str
    events_html: str
    turns: List[str]
    elapsed_infer_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OnlineTurnSession:
    """单路会话: push_pcm → (optional) StreamUpdate; finish → final update."""

    def __init__(
        self,
        engine: TurnDemoEngine,
        bot_speaking: bool = False,
        barge_in_frames: int = 4,
        commit_ms: int = 320,
        max_buffer_s: float = 20.0,
        lock: Optional[threading.Lock] = None,
    ):
        self.engine = engine
        self.sr = int(engine.sr)
        self.bot_speaking = bool(bot_speaking)
        self.cfg = PolicyConfig(barge_in_frames=int(barge_in_frames))
        self.commit_samples = max(int(self.sr * commit_ms / 1000.0), int(self.sr * 0.08))
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
        decision: DemoDecision = run_policy_on_frames(
            turns=turns,
            turn_probs=[f.turn_prob for f in pred.frames],
            asr_tokens=[f.asr for f in pred.frames],
            seconds_per_token=pred.seconds_per_token,
            bot_speaking=self.bot_speaking,
            cfg=self.cfg,
            asr_text=pred.asr_text,
        )
        hist = {}
        for t in turns:
            hist[t] = hist.get(t, 0) + 1
        return StreamUpdate(
            kind=kind,
            asr_text=pred.asr_text,
            action=decision.action,
            last_turn=decision.last_turn,
            reason=decision.reason,
            barge_in_at_s=decision.barge_in_at_s,
            duration_s=pred.duration_s,
            n_frames=len(pred.frames),
            turn_hist=hist,
            banner_html=decision_banner(decision),
            timeline_html=timeline_html(
                turns,
                seconds_per_token=pred.seconds_per_token,
                barge_in_at_s=decision.barge_in_at_s,
            ),
            events_html=events_table(decision.events, only_interesting=True),
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
