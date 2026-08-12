"""Turn demo 决策状态机: 打断 (barge-in) / 拒识 / 接话 / 等待。

两套通道 (勿混同一阈值):
  - Bot 正在播 TTS → 用 noidle/speaking 做打断
  - Bot 在听       → 用 turn_end / backchannel / uncertain|speaking 做接话决策
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

# 6 类
IDLE = "idle"
NOIDLE = "noidle"
SPEAKING = "speaking"
TURN_END = "turn_end"
BACKCHANNEL = "backchannel"
UNCERTAIN = "uncertain"

SPEECH_FOR_BARGE = {NOIDLE, SPEAKING}
HOLD_CLASSES = {UNCERTAIN, SPEAKING, NOIDLE}


@dataclass
class PolicyConfig:
    """打断: 连续 speech 帧数; 接话: 末字/稳定类规则。"""

    barge_in_frames: int = 4          # 4 * 80ms = 320ms
    barge_ignore_backchannel: bool = True
    # 句末决策: 从尾部往前找最后一个非 idle
    accept_classes: frozenset = frozenset({TURN_END})
    reject_classes: frozenset = frozenset({BACKCHANNEL})
    hold_classes: frozenset = frozenset(HOLD_CLASSES)


@dataclass
class FrameEvent:
    frame: int
    t0: float
    t1: float
    turn: str
    turn_prob: float
    asr: str
    bot_speaking: bool
    action: str           # none / barge_in / listening / ...
    note: str = ""


@dataclass
class DemoDecision:
    """整句结束后的产品决策。"""

    action: str                 # ACCEPT / REJECT / HOLD / NONE
    last_turn: str
    reason: str
    barge_in_at_s: Optional[float] = None
    events: List[FrameEvent] = field(default_factory=list)
    asr_text: str = ""


class TurnPolicy:
    """逐帧推进的状态机。"""

    def __init__(self, cfg: Optional[PolicyConfig] = None, bot_speaking: bool = False):
        self.cfg = cfg or PolicyConfig()
        self.bot_speaking = bool(bot_speaking)
        self._speech_run = 0
        self.barge_in_at_s: Optional[float] = None
        self.events: List[FrameEvent] = []

    def reset(self, bot_speaking: bool = False):
        self.bot_speaking = bool(bot_speaking)
        self._speech_run = 0
        self.barge_in_at_s = None
        self.events.clear()

    def step(
        self,
        frame: int,
        t0: float,
        t1: float,
        turn: str,
        turn_prob: float,
        asr: str,
    ) -> FrameEvent:
        action = "none"
        note = ""

        if self.bot_speaking:
            is_speech = turn in SPEECH_FOR_BARGE
            if self.cfg.barge_ignore_backchannel and turn == BACKCHANNEL:
                is_speech = False
                note = "backchannel ignored while TTS"
            if is_speech:
                self._speech_run += 1
                if self._speech_run >= self.cfg.barge_in_frames:
                    action = "barge_in"
                    note = f"stop TTS ({self._speech_run} speech frames)"
                    self.bot_speaking = False
                    self.barge_in_at_s = t0
                    self._speech_run = 0
                else:
                    action = "listening_tts"
                    note = f"speech_run={self._speech_run}/{self.cfg.barge_in_frames}"
            else:
                self._speech_run = 0
                action = "tts_playing"
        else:
            if turn == TURN_END:
                action = "候选ACCEPT"
                note = "turn_end seen"
            elif turn == BACKCHANNEL:
                action = "候选REJECT"
                note = "backchannel seen"
            elif turn in HOLD_CLASSES:
                action = "HOLD"
                note = "still speaking / uncertain"
            else:
                action = "idle"

        ev = FrameEvent(
            frame=frame,
            t0=t0,
            t1=t1,
            turn=turn,
            turn_prob=turn_prob,
            asr=asr,
            bot_speaking=self.bot_speaking if action != "barge_in" else False,
            action=action,
            note=note,
        )
        # barge_in 发生在本帧: 记下后 bot 已 False
        if action == "barge_in":
            ev.bot_speaking = False
        self.events.append(ev)
        return ev

    def finalize(self, asr_text: str = "") -> DemoDecision:
        """句末决策: 取最后一个非 idle turn。"""
        last_turn = IDLE
        for ev in reversed(self.events):
            if ev.turn != IDLE:
                last_turn = ev.turn
                break

        if last_turn in self.cfg.accept_classes:
            action, reason = "ACCEPT", f"末字/末态 = {last_turn} → 可以回复"
        elif last_turn in self.cfg.reject_classes:
            action, reason = "REJECT", f"末字/末态 = {last_turn} → 拒识(不当一轮请求)"
        elif last_turn in self.cfg.hold_classes:
            action, reason = "HOLD", f"末字/末态 = {last_turn} → 未说完, 继续听"
        else:
            action, reason = "NONE", "全程 idle / 无有效 turn"

        if self.barge_in_at_s is not None:
            reason = f"barge-in @{self.barge_in_at_s:.2f}s; " + reason

        return DemoDecision(
            action=action,
            last_turn=last_turn,
            reason=reason,
            barge_in_at_s=self.barge_in_at_s,
            events=list(self.events),
            asr_text=asr_text,
        )


def run_policy_on_frames(
    turns: Sequence[str],
    turn_probs: Sequence[float],
    asr_tokens: Sequence[str],
    seconds_per_token: float = 0.08,
    bot_speaking: bool = False,
    cfg: Optional[PolicyConfig] = None,
    asr_text: str = "",
) -> DemoDecision:
    """离线: 对整段逐帧预测跑一遍策略。"""
    pol = TurnPolicy(cfg=cfg, bot_speaking=bot_speaking)
    n = len(turns)
    for i in range(n):
        t0 = i * seconds_per_token
        pol.step(
            frame=i,
            t0=t0,
            t1=t0 + seconds_per_token,
            turn=turns[i],
            turn_prob=float(turn_probs[i]) if i < len(turn_probs) else 0.0,
            asr=asr_tokens[i] if i < len(asr_tokens) else "",
        )
    return pol.finalize(asr_text=asr_text)
