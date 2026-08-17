"""Local two-pass ASR and frame-level turn inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from mistral_common.protocol.instruct.chunk import RawAudio
from mistral_common.protocol.transcription.request import (
    StreamingMode,
    TranscriptionRequest,
)
from mistral_common.tokens.tokenizers.audio import Audio
from transformers import BatchFeature

from .modeling import TURN_CLASS_IDS, TURN_CLASS_NAMES, VoxtralMTP

STREAMING_PAD_ID = 32
STREAMING_WORD_ID = 33


@dataclass(frozen=True)
class TurnFrame:
    """One turn-head prediction on the model's frame timeline."""

    index: int
    start_ms: int
    end_ms: int
    label: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ASRTurnResult:
    """Transcript and aligned frame-level turn predictions."""

    transcript: str
    turn_frames: list[TurnFrame]
    generated_token_ids: list[int]
    frame_ms: int


def _encode_audio(
    processor: Any,
    audio: Audio,
    delay_ms: int,
) -> BatchFeature:
    request = TranscriptionRequest(
        audio=RawAudio.from_audio(audio),
        streaming=StreamingMode.OFFLINE,
        target_streaming_delay_ms=delay_ms,
    )
    tokenized = processor.tokenizer.tokenizer.encode_transcription(request)
    text_encoding = processor.tokenizer(
        tokenized.tokens,
        add_special_tokens=False,
        return_tensors="pt",
    )
    audio_encoding = processor.feature_extractor(
        [item.audio_array for item in tokenized.audios],
        center=True,
        sampling_rate=audio.sampling_rate,
        return_tensors="pt",
    )
    num_delay_tokens = processor.mistral_common_audio_config.get_num_delay_tokens(
        delay_ms
    )
    return BatchFeature(
        {
            **text_encoding,
            **audio_encoding,
            "num_delay_tokens": num_delay_tokens,
        },
        tensor_type="pt",
    )


def _predict_turn(
    logits: torch.Tensor,
) -> tuple[str, float, dict[str, float]]:
    probabilities = F.softmax(logits[list(TURN_CLASS_IDS)].float(), dim=-1)
    best_index = int(probabilities.argmax().item())
    return (
        TURN_CLASS_NAMES[best_index],
        float(probabilities[best_index].item()),
        {
            name: float(probabilities[index].item())
            for index, name in enumerate(TURN_CLASS_NAMES)
        },
    )


@torch.inference_mode()
def infer_asr_turn(
    model: VoxtralMTP,
    processor: Any,
    audio: str | Path | Audio,
    *,
    delay_ms: int | None = None,
) -> ASRTurnResult:
    """Infer one transcript and all aligned 80 ms turn frames.

    ASR generation uses the stock Voxtral head. A second forward pass over the
    aligned generated sequence evaluates the independent MTP turn head.
    """
    if isinstance(audio, (str, Path)):
        audio = Audio.from_file(str(audio), strict=False)
    if not isinstance(audio, Audio):
        raise TypeError("audio must be a path or mistral_common Audio")

    sample_rate = int(processor.feature_extractor.sampling_rate)
    audio.resample(sample_rate)
    audio_length_per_token = int(getattr(model.config, "audio_length_per_tok", 8))
    hop_length = int(processor.feature_extractor.hop_length)
    frame_ms = round(audio_length_per_token * hop_length * 1000 / sample_rate)
    if delay_ms is None:
        delay_ms = int(getattr(model.config, "default_num_delay_tokens", 6)) * frame_ms
    if delay_ms < frame_ms or delay_ms % frame_ms:
        raise ValueError(
            f"delay_ms must be a positive multiple of {frame_ms}, got {delay_ms}"
        )

    encoded = _encode_audio(processor, audio, delay_ms)
    device = next(model.parameters()).device
    input_ids = encoded.input_ids.to(device)
    input_features = encoded.input_features.to(device=device, dtype=model.dtype)
    num_delay_tokens = int(encoded.num_delay_tokens)
    prefix_length = int(input_ids.shape[1])
    num_audio_tokens = math.ceil(input_features.shape[-1] / audio_length_per_token)
    frame_count = max(num_audio_tokens - prefix_length, 0)

    generated = model.generate(
        input_ids=input_ids,
        input_features=input_features,
        num_delay_tokens=num_delay_tokens,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max(frame_count, 1),
    )[0]
    aligned_ids = generated.unsqueeze(0)
    if aligned_ids.shape[1] < num_audio_tokens:
        padding = torch.full(
            (1, num_audio_tokens - aligned_ids.shape[1]),
            STREAMING_PAD_ID,
            dtype=aligned_ids.dtype,
            device=device,
        )
        aligned_ids = torch.cat([aligned_ids, padding], dim=1)
    else:
        aligned_ids = aligned_ids[:, :num_audio_tokens]

    output = model(
        input_ids=aligned_ids,
        input_features=input_features,
        num_delay_tokens=num_delay_tokens,
    )
    turn_logits = output.vad_logits[0]
    frames: list[TurnFrame] = []
    for frame_index in range(frame_count):
        prediction_index = prefix_length + frame_index - 1
        if not 0 <= prediction_index < turn_logits.shape[0]:
            continue
        label, confidence, probabilities = _predict_turn(turn_logits[prediction_index])
        frames.append(
            TurnFrame(
                index=frame_index,
                start_ms=frame_index * frame_ms,
                end_ms=(frame_index + 1) * frame_ms,
                label=label,
                confidence=confidence,
                probabilities=probabilities,
            )
        )

    generated_token_ids = [int(token) for token in generated[prefix_length:].tolist()]
    transcript_ids = [
        token
        for token in generated_token_ids
        if token not in (STREAMING_PAD_ID, STREAMING_WORD_ID, *TURN_CLASS_IDS)
    ]
    transcript = processor.tokenizer.decode(
        transcript_ids,
        skip_special_tokens=True,
    ).strip()
    return ASRTurnResult(
        transcript=transcript,
        turn_frames=frames,
        generated_token_ids=generated_token_ids,
        frame_ms=frame_ms,
    )
