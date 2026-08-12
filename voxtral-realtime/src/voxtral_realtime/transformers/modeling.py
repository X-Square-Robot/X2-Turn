# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 X Square contributors
"""Training-side definition for the Voxtral ASR + turn dual-head model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers.utils import ModelOutput

TURN_CLASS_IDS = (35, 36, 37, 38, 39, 40)
TURN_CLASS_NAMES = (
    "idle",
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
)


@dataclass
class VoxtralMTPOutput(ModelOutput):
    """Losses and logits produced by the shared-backbone dual-head wrapper."""

    loss: torch.Tensor | None = None
    asr_loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    vad_logits: torch.Tensor | None = None
    vad_loss: torch.Tensor | None = None


class VoxtralMTP(nn.Module):
    """Shared Voxtral backbone with independent ASR and turn heads.

    ``base_model`` must expose the public Transformers Voxtral realtime
    contract: ``model``, ``lm_head``, ``loss_function``, and
    ``config.text_config``. The turn head has the full vocabulary dimension so
    checkpoints remain compatible with standard language-model loss tooling;
    turn supervision uses token ids 35 through 40.
    """

    def __init__(
        self,
        base_model: nn.Module,
        vad_loss_weight: float = 0.1,
        train_vad_head_only: bool = False,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.vad_loss_weight = float(vad_loss_weight)
        self.train_vad_head_only = bool(train_vad_head_only)

        vocab_size = int(base_model.config.text_config.vocab_size)
        hidden_size = int(base_model.config.text_config.hidden_size)
        reference = base_model.lm_head.weight
        self.vad_lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
            device=reference.device,
            dtype=reference.dtype,
        )
        with torch.no_grad():
            self.vad_lm_head.weight.copy_(reference)

    @property
    def config(self) -> Any:
        return self.base_model.config

    @property
    def dtype(self) -> torch.dtype | None:
        return self.base_model.dtype

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> Any:
        return self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> Any:
        return self.base_model.gradient_checkpointing_disable()

    def _cast_float_inputs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        dtype = self.dtype
        if dtype is None:
            return kwargs
        for key, value in list(kwargs.items()):
            if torch.is_tensor(value) and value.is_floating_point():
                kwargs[key] = value.to(dtype=dtype)
        return kwargs

    def _compute_vad_loss(
        self, vad_logits: torch.Tensor, vad_labels: torch.Tensor
    ) -> torch.Tensor:
        vad_labels = vad_labels.to(vad_logits.device)
        if bool((vad_labels != -100).any()):
            return self.base_model.loss_function(
                logits=vad_logits,
                labels=vad_labels,
                vocab_size=self.base_model.config.text_config.vocab_size,
            )

        # Mean cross entropy over an all-ignored target is NaN. Keep the turn
        # head in the autograd graph with an exact zero so DDP/DeepSpeed still
        # sees populated gradient buckets on ASR-only micro-batches.
        return vad_logits.sum() * 0.0

    def forward(
        self,
        *args: Any,
        vad_labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> VoxtralMTPOutput:
        labels = kwargs.pop("labels", None)
        model_kwargs = dict(kwargs)
        model_kwargs.pop("vad_labels", None)
        model_kwargs = self._cast_float_inputs(model_kwargs)

        if self.train_vad_head_only:
            with torch.no_grad():
                outputs = self.base_model.model(*args, **model_kwargs)
                hidden_states = outputs.last_hidden_state
                logits = self.base_model.lm_head(hidden_states)
                asr_loss = None
                if labels is not None:
                    asr_loss = self.base_model.loss_function(
                        logits=logits,
                        labels=labels,
                        vocab_size=self.base_model.config.text_config.vocab_size,
                    )

            vad_logits = self.vad_lm_head(
                hidden_states.to(dtype=self.vad_lm_head.weight.dtype)
            )
            vad_loss = (
                self._compute_vad_loss(vad_logits, vad_labels)
                if vad_labels is not None
                else None
            )
            loss = (
                self.vad_loss_weight * vad_loss
                if vad_loss is not None and self.vad_loss_weight > 0
                else None
            )
            return VoxtralMTPOutput(
                loss=loss,
                asr_loss=asr_loss,
                logits=logits,
                vad_logits=vad_logits,
                vad_loss=vad_loss,
            )

        outputs = self.base_model.model(*args, **model_kwargs)
        hidden_states = outputs.last_hidden_state
        logits = self.base_model.lm_head(hidden_states)
        vad_logits = self.vad_lm_head(
            hidden_states.to(dtype=self.vad_lm_head.weight.dtype)
        )

        asr_loss = None
        loss = None
        if labels is not None:
            asr_loss = self.base_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.base_model.config.text_config.vocab_size,
            )
            loss = asr_loss

        vad_loss = None
        if vad_labels is not None:
            vad_loss = self._compute_vad_loss(vad_logits, vad_labels)
            if loss is not None and self.vad_loss_weight > 0:
                loss = loss + self.vad_loss_weight * vad_loss

        return VoxtralMTPOutput(
            loss=loss,
            asr_loss=asr_loss,
            logits=logits,
            vad_logits=vad_logits,
            vad_loss=vad_loss,
        )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate ASR generation to the wrapped Transformers model."""
        return self.base_model.generate(*args, **kwargs)


def _resolve_model_dir(
    model_id_or_path: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    candidate = Path(model_id_or_path).expanduser()
    if candidate.is_dir():
        return candidate

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(model_id_or_path),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
            allow_patterns=["config.json", "model.safetensors"],
        )
    )


def _validate_load_result(missing: list[str], unexpected: list[str]) -> None:
    missing = [name for name in missing if name != "base_model.lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(
            f"incompatible checkpoint: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )


def load_mtp_checkpoint(
    model_id_or_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = None,
    vad_loss_weight: float = 0.1,
    train_vad_head_only: bool = False,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> VoxtralMTP:
    """Load ``VoxtralMTP`` from a local directory or Hugging Face Hub."""
    from safetensors.torch import load_file
    from transformers import (
        VoxtralRealtimeConfig,
        VoxtralRealtimeForConditionalGeneration,
    )

    model_dir = _resolve_model_dir(
        model_id_or_path,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    config = VoxtralRealtimeConfig.from_pretrained(model_dir)
    base_model = VoxtralRealtimeForConditionalGeneration(config)
    model = VoxtralMTP(
        base_model,
        vad_loss_weight=vad_loss_weight,
        train_vad_head_only=train_vad_head_only,
    )
    state_dict = load_file(str(model_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _validate_load_result(missing, unexpected)
    if dtype is None:
        return model.to(device=device)
    return model.to(device=device, dtype=dtype)
