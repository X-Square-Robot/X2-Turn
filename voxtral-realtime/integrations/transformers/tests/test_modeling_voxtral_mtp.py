from types import SimpleNamespace

import torch
import torch.nn.functional as F
from modeling_voxtral_mtp import VoxtralMTP, VoxtralMTPOutput
from torch import nn

VOCAB_SIZE = 64
HIDDEN_SIZE = 16


class StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

    def forward(self, inputs_embeds=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.projection(inputs_embeds))


class StubVoxtral(nn.Module):
    def __init__(self):
        super().__init__()
        text_config = SimpleNamespace(
            vocab_size=VOCAB_SIZE,
            hidden_size=HIDDEN_SIZE,
        )
        self.config = SimpleNamespace(text_config=text_config)
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
