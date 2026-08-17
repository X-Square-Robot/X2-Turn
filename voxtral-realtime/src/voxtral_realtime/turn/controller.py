"""Frame-level turn controller for Voxtral realtime sessions.

Rules (user-approved):
  - Bot speaking: only ``speaking`` / ``turn_end`` → immediate barge (nonidle)
  - ``turn_end`` while listening → wait N acoustically silent frames; speech cancels
  - After semantic speech, K model-idle + acoustic-silence frames → soft endpoint
  - only sustained sentence-initial ``backchannel`` is actionable
  - sentence-internal ``backchannel`` is ignored for endpoint counting
  - ``noidle`` alone does not barge; can cancel pending_end / reset silence counter
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPEECH = frozenset({"noidle", "speaking"})
SEMANTIC = frozenset({"speaking", "turn_end"})
BARGE_WHEN_BOT = frozenset({"speaking", "turn_end"})
NON_SPEECH = frozenset({"idle"})
PURE_BACKCHANNEL_FILLERS = frozenset(
    {
        "嗯",
        "嗯嗯",
        "哦",
        "噢",
        "啊",
        "好",
        "好的",
        "对",
        "对的",
        "是的",
        "有的",
        "ok",
        "okay",
        "ok嗯",
        "嗯ok",
    }
)


@dataclass
class FrameTurnConfig:
    end_confirm_frames: int = 1  # N: frames after turn_end before ACCEPT
    silence_end_frames: int = 3  # K: non-speech frames after speaking → soft end
    min_asr_chars: int = 1  # reject empty ACCEPT
    tail_min_frames: int = 2  # collect delayed ASR tail for at least 160 ms
    tail_max_frames: int = 5  # force decision after at most 400 ms
    tail_stable_frames: int = 2  # unchanged ASR frames required after minimum
    backchannel_confirm_frames: int = 2
    backchannel_fallback_idle_frames: int = 3
    short_asr_chars: int = 4
    short_tail_min_frames: int = 4  # K + tail = ~560 ms minimum
    short_tail_max_frames: int = 7  # K + tail = ~800 ms maximum
    acoustic_vad_max_hold_frames: int = 8  # fail-safe: at most 640 ms veto


@dataclass
class ControllerState:
    bot_speaking: bool = False
    user_active: bool = False
    semantic_seen: bool = False
    pending_end_left: int = 0
    silence_run: int = 0
    tail_pending: bool = False
    tail_frames: int = 0
    tail_stable: int = 0
    tail_last_asr: str = ""
    tail_reason: str = ""
    backchannel_run: int = 0
    backchannel_idle_run: int = 0
    backchannel_asr: str = ""
    backchannel_confirmed: bool = False
    backchannel_rejected: bool = False
    last_turn: str = "idle"
    last_state: str = "idle"
    last_reason: str = ""
    acoustic_active: bool = False
    acoustic_hold_run: int = 0


class FrameTurnController:
    def __init__(self, cfg: FrameTurnConfig | None = None):
        self.cfg = cfg or FrameTurnConfig()
        self.st = ControllerState()

    def reset(self) -> None:
        bot = self.st.bot_speaking
        self.st = ControllerState(bot_speaking=bot)

    def set_bot_speaking(self, speaking: bool) -> None:
        self.st.bot_speaking = bool(speaking)

    def on_frame(
        self,
        turn: str,
        asr_text: str,
        frame_index: int | None = None,
        acoustic_active: bool = False,
    ) -> dict:
        """Process one 80ms turn label; return a generic bridge state dictionary."""
        turn = turn or "idle"
        self.st.last_turn = turn
        self.st.acoustic_active = bool(acoustic_active)
        asr = (asr_text or "").strip()
        idx = frame_index if frame_index is not None else "?"

        # --- Candidate endpoint: retain the realtime session briefly so late
        # ASR deltas can append sentence-final characters before ACCEPT/reset.
        if self.st.tail_pending:
            if (
                acoustic_active
                and self.st.acoustic_hold_run < self.cfg.acoustic_vad_max_hold_frames
            ):
                self.st.acoustic_hold_run += 1
                self._clear_tail()
                self.st.silence_run = 0
                return self._out("nonidle", turn, asr, note=f"tail_cancel_vad@{idx}")
            if not acoustic_active:
                self.st.acoustic_hold_run = 0
            if turn == "backchannel":
                return self._out(
                    "nonidle", turn, asr, note=f"tail_ignore_backchannel@{idx}"
                )
            if turn in SPEECH:
                self._clear_tail()
                self.st.silence_run = 0
                return self._out("nonidle", turn, asr, note=f"tail_cancel@{idx}:{turn}")

            self.st.tail_frames += 1
            if asr == self.st.tail_last_asr:
                self.st.tail_stable += 1
            else:
                self.st.tail_last_asr = asr
                self.st.tail_stable = 0

            tail_min, tail_max = self._tail_limits(
                self.st.tail_last_asr, self.st.tail_reason
            )
            tail_ready = (
                self.st.tail_frames >= tail_min
                and self.st.tail_stable >= self.cfg.tail_stable_frames
            )
            tail_timeout = self.st.tail_frames >= tail_max
            if tail_ready or tail_timeout:
                reason = self.st.tail_reason
                final_asr = self.st.tail_last_asr
                self._clear_tail()
                if len(final_asr) >= self.cfg.min_asr_chars:
                    return self._final_speak(final_asr, reason=f"{reason}_tail")
                return self._out("idle", turn, final_asr, note="tail_empty_asr")

            return self._out(
                "nonidle",
                turn,
                asr,
                note=(
                    f"tail_wait:{self.st.tail_frames}/{tail_max}"
                    f":stable={self.st.tail_stable}"
                ),
            )

        # --- Bot TTS: only semantic user speech may interrupt.
        # ``noidle`` is acoustic speech without semantic intent.
        if self.st.bot_speaking and turn in BARGE_WHEN_BOT:
            return self._out(
                "nonidle",
                turn,
                asr,
                event="barge_in",
                reason=f"barge@{idx}:{turn}",
            )

        # --- Backchannel is meaningful only as a sustained sentence-initial
        # run. Once semantic speech starts it is neutral and cannot advance an
        # endpoint countdown.
        if turn == "backchannel":
            if self.st.semantic_seen:
                return self._out("nonidle", turn, asr, note=f"bc_inside_ignored@{idx}")

            was_confirmed = self.st.backchannel_confirmed
            self.st.backchannel_run += 1
            self.st.backchannel_idle_run = 0
            if asr:
                self.st.backchannel_asr = asr
            if self.st.backchannel_run >= self.cfg.backchannel_confirm_frames:
                self.st.backchannel_confirmed = True

            if not self.st.backchannel_confirmed:
                return self._out(
                    "idle",
                    turn,
                    asr,
                    note=(
                        f"bc_pending:{self.st.backchannel_run}/"
                        f"{self.cfg.backchannel_confirm_frames}"
                    ),
                )

            if self.st.bot_speaking:
                self.st.backchannel_rejected = True
                return self._out(
                    "idle",
                    turn,
                    asr,
                    event=None if was_confirmed else "reject",
                    reason=f"bc_bot_only@{idx}",
                )
            return self._out("nonidle", turn, asr, note=f"bc_confirmed@{idx}")

        if self.st.backchannel_confirmed and turn == "idle":
            if self.st.backchannel_rejected:
                self._clear_backchannel()
                return self._out("idle", turn, asr, note="bc_bot_done")
            if (
                acoustic_active
                and self.st.acoustic_hold_run < self.cfg.acoustic_vad_max_hold_frames
            ):
                self.st.acoustic_hold_run += 1
                self.st.backchannel_idle_run = 0
                return self._out("nonidle", turn, asr, note=f"bc_vad_hold@{idx}")
            if not acoustic_active:
                self.st.acoustic_hold_run = 0
            self.st.backchannel_idle_run += 1
            if asr:
                self.st.backchannel_asr = asr
            if (
                self.st.backchannel_idle_run
                >= self.cfg.backchannel_fallback_idle_frames
            ):
                final_asr = self.st.backchannel_asr
                self._clear_backchannel()
                if len(final_asr) >= self.cfg.min_asr_chars:
                    if self._is_pure_backchannel_filler(final_asr):
                        return self._out(
                            "idle",
                            turn,
                            final_asr,
                            event="reject",
                            reason="backchannel_filler",
                        )
                    return self._final_speak(final_asr, reason="backchannel_fallback")
                return self._out("idle", turn, final_asr, note="backchannel_empty_asr")
            return self._out(
                "nonidle",
                turn,
                asr,
                note=(
                    f"bc_idle:{self.st.backchannel_idle_run}/"
                    f"{self.cfg.backchannel_fallback_idle_frames}"
                ),
            )

        if self.st.backchannel_run:
            # A non-backchannel continuation means this is an ordinary
            # utterance; remove the tentative prefix classification.
            self._clear_backchannel()

        # --- turn_end: start N-frame confirmation (listening) ---
        if turn == "turn_end" and not self.st.bot_speaking:
            self.st.user_active = True
            self.st.semantic_seen = True
            self.st.pending_end_left = self.cfg.end_confirm_frames
            self.st.silence_run = 0
            return self._out(
                "nonidle",
                turn,
                asr,
                note=f"pending_end_start:N={self.cfg.end_confirm_frames}",
            )

        # --- Pending end countdown ---
        if self.st.pending_end_left > 0:
            if turn in SPEECH:
                self.st.pending_end_left = 0
                self.st.silence_run = 0
                self.st.acoustic_hold_run = 0
                return self._out(
                    "nonidle", turn, asr, note=f"pending_cancel@{idx}:{turn}"
                )
            if (
                acoustic_active
                and self.st.acoustic_hold_run < self.cfg.acoustic_vad_max_hold_frames
            ):
                self.st.acoustic_hold_run += 1
                return self._out(
                    "nonidle",
                    turn,
                    asr,
                    note=f"pending_vad_hold:{self.st.pending_end_left}@{idx}",
                )
            if not acoustic_active:
                self.st.acoustic_hold_run = 0
            self.st.pending_end_left -= 1
            if self.st.pending_end_left <= 0:
                if len(asr) >= self.cfg.min_asr_chars:
                    return self._start_tail(asr, reason="turn_end_confirmed")
                return self._out("idle", turn, asr, note="turn_end_empty_asr")
            return self._out(
                "nonidle",
                turn,
                asr,
                note=f"pending_end:{self.st.pending_end_left}",
            )

        # --- Semantic speaking / hold ---
        if turn == "speaking":
            self.st.user_active = True
            self.st.semantic_seen = True
            self.st.silence_run = 0
            self.st.acoustic_hold_run = 0
            return self._out("nonidle", turn, asr, note=f"hold@{idx}:{turn}")

        if turn in SPEECH:
            # noidle: acoustic only — keeps listening, no barge by itself
            self.st.user_active = True
            self.st.silence_run = 0
            self.st.acoustic_hold_run = 0
            return self._out("nonidle", turn, asr, note=f"speech@{idx}:noidle")

        # --- Soft end: K frames without speech after semantic content ---
        if self.st.user_active and self.st.semantic_seen:
            model_idle = turn in NON_SPEECH or turn == "idle"
            vad_can_hold = (
                acoustic_active
                and self.st.acoustic_hold_run < self.cfg.acoustic_vad_max_hold_frames
            )
            if model_idle and vad_can_hold:
                self.st.acoustic_hold_run += 1
                self.st.silence_run = 0
            elif model_idle:
                if not acoustic_active:
                    self.st.acoustic_hold_run = 0
                self.st.silence_run += 1
            else:
                self.st.acoustic_hold_run = 0
                self.st.silence_run = 0
            if self.st.silence_run >= self.cfg.silence_end_frames:
                if len(asr) >= self.cfg.min_asr_chars:
                    return self._start_tail(asr, reason="silence_end")
                self.st.silence_run = 0
                return self._out("idle", turn, asr, note="silence_end_empty_asr")
            if self.st.user_active:
                return self._out(
                    "nonidle", turn, asr, note=f"silence_run:{self.st.silence_run}"
                )

        return self._out("idle", turn, asr)

    def _start_tail(self, asr: str, reason: str) -> dict:
        tail_min, tail_max = self._tail_limits(asr, reason)
        self.st.pending_end_left = 0
        self.st.silence_run = 0
        self.st.tail_pending = True
        self.st.tail_frames = 0
        self.st.tail_stable = 0
        self.st.tail_last_asr = asr
        self.st.tail_reason = reason
        return self._out(
            "nonidle",
            "turn_end",
            asr,
            note=f"tail_start:{reason}:range={tail_min}-{tail_max}",
        )

    def _clear_tail(self) -> None:
        self.st.tail_pending = False
        self.st.tail_frames = 0
        self.st.tail_stable = 0
        self.st.tail_last_asr = ""
        self.st.tail_reason = ""

    def _clear_backchannel(self) -> None:
        self.st.backchannel_run = 0
        self.st.backchannel_idle_run = 0
        self.st.backchannel_asr = ""
        self.st.backchannel_confirmed = False
        self.st.backchannel_rejected = False

    def _tail_limits(self, asr: str, reason: str = "") -> tuple[int, int]:
        if reason == "turn_end_confirmed":
            return self.cfg.tail_min_frames, self.cfg.tail_max_frames
        meaningful = re.sub(r"[\W_]+", "", asr or "", flags=re.UNICODE)
        if len(meaningful) <= self.cfg.short_asr_chars:
            return self.cfg.short_tail_min_frames, self.cfg.short_tail_max_frames
        return self.cfg.tail_min_frames, self.cfg.tail_max_frames

    @staticmethod
    def _is_pure_backchannel_filler(asr: str) -> bool:
        normalized = re.sub(r"[\s,，。.!！?？、]+", "", (asr or "").lower())
        return normalized in PURE_BACKCHANNEL_FILLERS

    def _final_speak(self, asr: str, reason: str) -> dict:
        return {
            "state": "speak",
            "turn_class": "turn_end",
            "event": "accept",
            "text": asr,
            "asr_segment": "",
            "asr_buffer": asr,
            "reason": reason,
        }

    def _out(
        self,
        state: str,
        turn: str,
        asr: str,
        event: str | None = None,
        note: str = "",
        reason: str = "",
    ) -> dict:
        self.st.last_state = state
        self.st.last_reason = reason or note
        out = {
            "state": state,
            "turn_class": turn,
            "asr_segment": "",
            "asr_buffer": asr,
            "acoustic_active": self.st.acoustic_active,
        }
        if event:
            out["event"] = event
        if note:
            out["note"] = note
        if reason:
            out["reason"] = reason
        return out
