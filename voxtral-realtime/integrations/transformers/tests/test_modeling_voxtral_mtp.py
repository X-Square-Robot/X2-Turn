import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from mistral_common.tokens.tokenizers.audio import Audio
from torch import nn
from transformers import BatchFeature

PACKAGE_SRC = Path(__file__).resolve().parents[3] / "src"
sys.path.insert(0, str(PACKAGE_SRC))

from voxtral_realtime.transformers import VoxtralMTP, VoxtralMTPOutput  # noqa: E402
from voxtral_realtime.transformers import inference as inference_module  # noqa: E402
from voxtral_realtime.transformers import modeling as modeling_module  # noqa: E402

VOCAB_SIZE = 64
HIDDEN_SIZE = 16


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, inputs_embeds=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.projection(inputs_embeds))


class StubVoxtral(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            text_config = SimpleNamespace(
                vocab_size=VOCAB_SIZE,
                hidden_size=HIDDEN_SIZE,
            )
            config = SimpleNamespace(text_config=text_config)
        self.config = config
        self.model = StubBackbone()
        self.lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    @property
    def dtype(self):
        return self.lm_head.weight.dtype

    def loss_function(self, logits=None, labels=None, vocab_size=None, **kwargs):
        return F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )


def run_forward(model, vad_labels):
    inputs = torch.randn(2, 8, HIDDEN_SIZE)
    labels = torch.randint(0, VOCAB_SIZE, (2, 8))
    return model(inputs_embeds=inputs, labels=labels, vad_labels=vad_labels)


def test_output_is_transformers_model_output():
    output = VoxtralMTPOutput(loss=torch.tensor(1.0))

    assert isinstance(output, dict)
    assert output["loss"].item() == 1.0


def test_turn_head_starts_from_asr_head():
    base = StubVoxtral()
    model = VoxtralMTP(base)

    torch.testing.assert_close(model.vad_lm_head.weight, base.lm_head.weight)
    assert model.vad_lm_head.weight is not base.lm_head.weight


def test_all_masked_turn_batch_has_finite_zero_turn_loss():
    model = VoxtralMTP(StubVoxtral())
    labels = torch.full((2, 8), -100, dtype=torch.long)

    output = run_forward(model, labels)
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.vad_loss.item() == 0.0
    assert model.vad_lm_head.weight.grad is not None


def test_head_only_training_does_not_update_backbone():
    model = VoxtralMTP(StubVoxtral(), train_vad_head_only=True)
    labels = torch.full((2, 8), -100, dtype=torch.long)
    labels[0] = 37

    output = run_forward(model, labels)
    output.loss.backward()

    assert model.vad_lm_head.weight.grad is not None
    assert model.base_model.model.projection.weight.grad is None
    assert model.base_model.lm_head.weight.grad is None


def test_resolve_model_dir_accepts_local_directory(tmp_path):
    assert modeling_module._resolve_model_dir(tmp_path) == tmp_path


def test_resolve_model_dir_downloads_hub_snapshot(tmp_path, monkeypatch):
    import huggingface_hub

    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    resolved = modeling_module._resolve_model_dir(
        "x-square-robot/X2-Turn-4B-0812",
        revision="release",
        local_files_only=True,
    )

    assert resolved == tmp_path
    assert calls["revision"] == "release"
    assert calls["local_files_only"] is True
    assert calls["allow_patterns"] == ["config.json", "model.safetensors"]


def test_loader_rejects_missing_turn_head():
    with pytest.raises(RuntimeError, match="vad_lm_head.weight"):
        modeling_module._validate_load_result(["vad_lm_head.weight"], [])


def test_loader_allows_tied_asr_head():
    modeling_module._validate_load_result(["base_model.lm_head.weight"], [])


def test_infer_asr_turn_returns_transcript_and_frames(monkeypatch):
    encoded = BatchFeature(
        {
            "input_ids": torch.tensor([[1]]),
            "input_features": torch.zeros((1, 4, 32)),
            "num_delay_tokens": 6,
        }
    )
    monkeypatch.setattr(
        inference_module,
        "_encode_audio",
        lambda processor, audio, delay_ms: encoded,
    )

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(
                audio_length_per_tok=8,
                default_num_delay_tokens=6,
            )

        @property
        def dtype(self):
            return self.anchor.dtype

        def generate(self, input_ids, **kwargs):
            return torch.cat(
                [input_ids, torch.tensor([[33, 55, 2]])],
                dim=1,
            )

        def forward(self, input_ids, **kwargs):
            logits = torch.zeros((1, input_ids.shape[1], VOCAB_SIZE))
            logits[:, :, 38] = 10
            return SimpleNamespace(vad_logits=logits)

    processor = SimpleNamespace(
        feature_extractor=SimpleNamespace(sampling_rate=16000, hop_length=160),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens: "hello"),
    )
    audio = Audio(
        audio_array=np.zeros(1600, dtype=np.float32),
        sampling_rate=16000,
        format="wav",
    )

    result = inference_module.infer_asr_turn(FakeModel(), processor, audio)

    assert result.transcript == "hello"
    assert result.frame_ms == 80
    assert len(result.turn_frames) == 3
    assert result.turn_frames[0].label == "turn_end"
