"""Local Transformers backend for the standalone turn demo."""

from __future__ import annotations

import numpy as np
import torch
from mistral_common.tokens.tokenizers.audio import Audio
from transformers import AutoProcessor

from demo_turn.predictions import FramePred, UtterancePred, load_audio
from voxtral_realtime.transformers import (
    TURN_CLASS_IDS,
    infer_asr_turn,
    load_mtp_checkpoint,
)

STREAMING_PAD_ID = 32
STREAMING_WORD_ID = 33


def _token_kind(token_id: int, tokenizer) -> str:
    if token_id == STREAMING_PAD_ID:
        return "[PAD]"
    if token_id == STREAMING_WORD_ID:
        return "[WORD]"
    if token_id in TURN_CLASS_IDS:
        return "[TURN]"
    if token_id == (tokenizer.bos_token_id or 1):
        return "[BOS]"
    if token_id == (tokenizer.eos_token_id or 2):
        return "[EOS]"
    try:
        return tokenizer.decode([token_id], skip_special_tokens=False)
    except Exception:
        return f"<{token_id}>"


class TurnDemoEngine:
    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        delay_ms: int | None = None,
        turn_label_delay_frames: int = 0,
    ) -> None:
        self.model_dir = model_dir
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        )
        self.turn_delay = max(0, int(turn_label_delay_frames))
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer
        self.model = load_mtp_checkpoint(
            model_dir,
            device=self.device,
            dtype=self.dtype,
        ).eval()

        self.sr = int(self.processor.feature_extractor.sampling_rate)
        hop = int(self.processor.feature_extractor.hop_length)
        audio_length = int(getattr(self.model.config, "audio_length_per_tok", 8))
        self.seconds_per_token = audio_length * hop / self.sr
        default_tokens = int(
            getattr(self.model.config, "default_num_delay_tokens", 6)
        )
        self.delay_ms = (
            int(delay_ms)
            if delay_ms is not None
            else round(default_tokens * self.seconds_per_token * 1000)
        )
        print(
            f"[turn-demo] backend=transformers model={model_dir} "
            f"device={self.device} delay_ms={self.delay_ms}",
            flush=True,
        )

    def infer_file(self, wav_path: str) -> UtterancePred:
        wav = load_audio(wav_path, self.sr)
        return self.infer_wav(wav, wav_path=wav_path)

    def infer_wav(
        self, wav: np.ndarray, wav_path: str = "<memory>"
    ) -> UtterancePred:
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        duration_s = float(wav.size / self.sr)
        if self.turn_delay:
            silence = np.zeros(
                round(self.turn_delay * self.seconds_per_token * self.sr),
                dtype=np.float32,
            )
            wav = np.concatenate([wav, silence])
        audio = Audio(audio_array=wav, sampling_rate=self.sr, format="wav")
        result = infer_asr_turn(
            self.model,
            self.processor,
            audio,
            delay_ms=self.delay_ms,
        )

        score_count = max(len(result.turn_frames) - self.turn_delay, 0)
        frames: list[FramePred] = []
        for index in range(score_count):
            source_index = index + self.turn_delay
            turn_frame = result.turn_frames[source_index]
            token_id = (
                result.generated_token_ids[index]
                if index < len(result.generated_token_ids)
                else STREAMING_PAD_ID
            )
            frames.append(
                FramePred(
                    frame=index,
                    t0=index * self.seconds_per_token,
                    t1=(index + 1) * self.seconds_per_token,
                    asr=_token_kind(token_id, self.tokenizer),
                    turn=turn_frame.label,
                    turn_prob=turn_frame.confidence,
                    probs=turn_frame.probabilities,
                )
            )
        return UtterancePred(
            wav_path=wav_path,
            asr_text=result.transcript,
            seconds_per_token=self.seconds_per_token,
            delay_ms=self.delay_ms,
            turn_label_delay_frames=self.turn_delay,
            frames=frames,
            duration_s=duration_s,
        )
