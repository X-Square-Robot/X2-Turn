# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for Voxtral MTP (ASR + turn) dual-head support."""

from __future__ import annotations

import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F

TURN_CLASS_IDS: list[int] = [35, 36, 37, 38, 39, 40]
TURN_CLASS_NAMES: list[str] = [
    "idle",
    "noidle",
    "speaking",
    "turn_end",
    "backchannel",
    "uncertain",
]
TURN_ID_TO_NAME: dict[int, str] = dict(
    zip(TURN_CLASS_IDS, TURN_CLASS_NAMES, strict=True)
)
TURN_NAME_TO_ID: dict[str, int] = dict(
    zip(TURN_CLASS_NAMES, TURN_CLASS_IDS, strict=True)
)


def turn_preds_from_hidden(
    vad_lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
) -> tuple[list[int], list[list[float]]]:
    """Run vad head and argmax over the 6 turn slots.

    Returns per-row (class_id, probs[6]).
    """
    out = vad_lm_head(hidden_states)
    if isinstance(out, tuple):
        logits = out[0]
    else:
        logits = out
    # logits: [N, vocab]
    slot = logits[:, TURN_CLASS_IDS]
    probs = F.softmax(slot.float(), dim=-1)
    idx = probs.argmax(dim=-1)
    class_ids = [TURN_CLASS_IDS[int(i)] for i in idx.tolist()]
    probs_list = probs.detach().cpu().tolist()
    return class_ids, probs_list


def _mtp_to_hf_name(name: str) -> str:
    """Normalize training-checkpoint keys to official HF layout."""
    if name.startswith("base_model."):
        name = name[len("base_model.") :]

    # MTP saves nested under model.*; official HF drops that wrapper.
    if name.startswith("model."):
        name = name[len("model.") :]

    # MTP: language_model.X  vs official: language_model.model.X
    if name.startswith("language_model.") and not name.startswith(
        "language_model.model."
    ):
        if not name.startswith("language_model.lm_head"):
            name = "language_model.model." + name[len("language_model.") :]

    return name


def hf_to_mistral_name(name: str) -> str | None:
    """Map official HF VoxtralRealtime keys to mistral consolidated names.

    Returns None to drop the weight (e.g. tied lm_head duplicate).
    """
    if name in ("lm_head.weight", "language_model.lm_head.weight"):
        # Tied with tok_embeddings in stock Voxtral / our MTP ASR head.
        return None

    if name.startswith("language_model.model.layers."):
        rest = name[len("language_model.model.") :]
        repls = [
            ("self_attn.q_proj", "attention.wq"),
            ("self_attn.k_proj", "attention.wk"),
            ("self_attn.v_proj", "attention.wv"),
            ("self_attn.o_proj", "attention.wo"),
            ("input_layernorm", "attention_norm"),
            ("post_attention_layernorm", "ffn_norm"),
            ("mlp.gate_proj", "feed_forward.w1"),
            ("mlp.down_proj", "feed_forward.w2"),
            ("mlp.up_proj", "feed_forward.w3"),
            ("ada_rms_norm.linear1", "ada_rms_norm_t_cond.0"),
            ("ada_rms_norm.linear2", "ada_rms_norm_t_cond.2"),
        ]
        for a, b in repls:
            if a in rest:
                rest = rest.replace(a, b, 1)
                break
        return rest

    if name == "language_model.model.embed_tokens.weight":
        return "mm_streams_embeddings.embedding_module.tok_embeddings.weight"
    if name == "language_model.model.norm.weight":
        return "norm.weight"
    if name == "multi_modal_projector.linear_1.weight":
        return (
            "mm_streams_embeddings.embedding_module.audio_language_projection.0.weight"
        )
    if name == "multi_modal_projector.linear_2.weight":
        return (
            "mm_streams_embeddings.embedding_module.audio_language_projection.2.weight"
        )

    m = re.match(r"audio_tower\.embedder\.conv([12])\.(weight|bias)", name)
    if m:
        idx = int(m.group(1)) - 1
        return (
            "mm_streams_embeddings.embedding_module.whisper_encoder."
            f"conv_layers.{idx}.conv.{m.group(2)}"
        )

    if name == "audio_tower.norm.weight":
        return (
            "mm_streams_embeddings.embedding_module.whisper_encoder."
            "transformer.norm.weight"
        )

    if name.startswith("audio_tower.layers."):
        rest = name[len("audio_tower.") :]
        repls = [
            ("self_attn.q_proj", "attention.wq"),
            ("self_attn.k_proj", "attention.wk"),
            ("self_attn.v_proj", "attention.wv"),
            ("self_attn.o_proj", "attention.wo"),
            ("self_attn_layer_norm", "attention_norm"),
            ("mlp.gate_proj", "feed_forward.w1"),
            ("mlp.down_proj", "feed_forward.w2"),
            ("mlp.up_proj", "feed_forward.w3"),
            ("final_layer_norm", "ffn_norm"),
        ]
        for a, b in repls:
            if a in rest:
                rest = rest.replace(a, b, 1)
                break
        rest = "transformer." + rest
        return "mm_streams_embeddings.embedding_module.whisper_encoder." + rest

    return name


def _inv_mistral_permute(
    w: torch.Tensor,
    n_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Inverse of vLLM MistralForCausalLM.maybe_remap_mistral permute.

    vLLM load path does ``param = permute(mistral_ckpt_weight)`` into HF-style
    modules.  Therefore HF -> mistral export must apply this inverse.
    """
    attn_out = w.shape[1]
    attn_in = head_dim * n_heads
    assert w.shape[0] == attn_in, (w.shape, n_heads, head_dim)
    return (
        w.reshape(n_heads, 2, attn_in // n_heads // 2, attn_out)
        .transpose(1, 2)
        .reshape(attn_in, attn_out)
        .contiguous()
    )


def _inv_mistral_permute_bias(
    b: torch.Tensor, n_heads: int, head_dim: int
) -> torch.Tensor:
    return b.reshape(n_heads, 2, head_dim // 2).transpose(1, 2).reshape(-1).contiguous()


def _maybe_inv_permute_for_mistral(
    mis_name: str, tensor: torch.Tensor, *, source_was_hf: bool
) -> torch.Tensor:
    """Apply HF->mistral layout fix for Q/K attention weights (and Q bias).

    Only when ``source_was_hf``: tensors already stored under mistral keys
    (exported consolidated) must not be inv-permuted again.
    """
    if not source_was_hf:
        return tensor

    # Text LLM: dim=3072, head_dim=128, n_heads=32, n_kv=8
    if re.fullmatch(r"layers\.\d+\.attention\.wq\.weight", mis_name):
        return _inv_mistral_permute(tensor, n_heads=32, head_dim=128)
    if re.fullmatch(r"layers\.\d+\.attention\.wk\.weight", mis_name):
        return _inv_mistral_permute(tensor, n_heads=8, head_dim=128)

    # Whisper encoder: encoder_head_dim=64, n_heads=n_kv=32
    if ".whisper_encoder.transformer.layers." in mis_name:
        if mis_name.endswith(".attention.wq.weight") or mis_name.endswith(
            ".attention.wk.weight"
        ):
            return _inv_mistral_permute(tensor, n_heads=32, head_dim=64)
        if mis_name.endswith(".attention.wq.bias"):
            return _inv_mistral_permute_bias(tensor, n_heads=32, head_dim=64)

    return tensor


def _source_name_is_hf(name: str) -> bool:
    return name.startswith(
        (
            "base_model.",
            "model.",
            "audio_tower.",
            "language_model.",
            "multi_modal_projector.",
        )
    )


def remap_mtp_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> tuple[list[tuple[str, torch.Tensor]], torch.Tensor | None]:
    """Remap MTP / HF weights to mistral keys; peel off vad_lm_head.

    Also converts HF Q/K layouts to mistral layouts (inverse of vLLM's
    load-time permute) so ``consolidated.safetensors`` matches stock Voxtral.
    """
    remapped: list[tuple[str, torch.Tensor]] = []
    vad_weight: torch.Tensor | None = None
    for name, tensor in weights:
        if name.endswith("vad_lm_head.weight") or name == "vad_lm_head.weight":
            vad_weight = tensor
            continue
        # After stripping base_model., training keys still look HF.
        hf_name = _mtp_to_hf_name(name)
        source_was_hf = _source_name_is_hf(name) or _source_name_is_hf(hf_name)
        mis_name = hf_to_mistral_name(hf_name)
        if mis_name is None:
            continue
        tensor = _maybe_inv_permute_for_mistral(
            mis_name, tensor, source_was_hf=source_was_hf
        )
        remapped.append((mis_name, tensor))
    return remapped, vad_weight
