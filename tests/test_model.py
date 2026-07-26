import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from para_audio_id.metrics import ranking_metrics
from para_audio_id.model import BeamResult, DigitDecoder, MuQEncoder, ParametricAudioIdentifier


class FakeEncoder(nn.Module):
    def forward(self, audio):
        return torch.zeros(audio.shape[0], 3, 8, device=audio.device)


def config():
    return {
        "model": {
            "muq_name": "unused",
            "encoder_dim": 8,
            "decoder": {
                "model_dim": 16,
                "layers": 1,
                "heads": 4,
                "feedforward_dim": 32,
                "dropout": 0.0,
            },
        }
    }


def test_decoder_shape_and_exact_five_step_greedy():
    model = ParametricAudioIdentifier(config(), encoder=FakeEncoder()).eval()
    audio = torch.zeros(2, 100)
    targets = torch.ones(2, 5, dtype=torch.long)
    assert model(audio, targets).shape == (2, 5, 11)
    outputs = model.greedy_decode(audio)
    assert len(outputs) == 2
    assert all(len(code) == 5 and code.isdecimal() for code in outputs)


def test_digit_decoder_causal_prefix_shapes():
    decoder = DigitDecoder(
        encoder_dim=8, model_dim=16, layers=1, heads=4, feedforward_dim=32, dropout=0
    )
    assert decoder(torch.zeros(2, 3, 8), torch.zeros(2, 4, dtype=torch.long)).shape == (
        2,
        4,
        11,
    )


def test_ranking_metrics():
    rankings = [
        [BeamResult("12345", -1), BeamResult("00000", -2)],
        [BeamResult("11111", -1), BeamResult("22222", -2)],
    ]
    result = ranking_metrics(["12345", "22222"], rankings)
    assert result["beam_top1"] == 0.5
    assert result["beam_top5"] == 1.0
    assert result["beam_mrr"] == 0.75


class DummyMuQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.9)) for _ in range(4)]
        )

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def forward(self, audio, output_hidden_states=True):
        assert audio.ndim == 2
        hidden = audio
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden.unsqueeze(1))


def test_muq_wrapper_requires_waveform_rank_and_validates_output(monkeypatch):
    monkeypatch.setitem(sys.modules, "muq", SimpleNamespace(MuQ=DummyMuQ))
    encoder = MuQEncoder("dummy", encoder_dim=4)
    assert encoder(torch.zeros(2, 4)).shape == (2, 1, 4)
    with pytest.raises(ValueError, match=r"\[batch, time\]"):
        encoder(torch.zeros(2, 1, 4))


def test_frozen_muq_is_deterministic_and_only_upper_blocks_train(monkeypatch):
    monkeypatch.setitem(sys.modules, "muq", SimpleNamespace(MuQ=DummyMuQ))
    encoder = MuQEncoder("dummy", encoder_dim=4)
    encoder.freeze_all()
    encoder.train()
    audio = torch.ones(2, 4)
    assert torch.equal(encoder(audio), encoder(audio))
    assert not encoder.model.training
    assert not any(parameter.requires_grad for parameter in encoder.parameters())

    upper = encoder.unfreeze_upper_fraction(0.25)
    encoder.train()
    assert len(upper) == 1
    assert upper[0].training
    assert all(not block.training for block in encoder.model.layers[:-1])
    assert any(parameter.requires_grad for parameter in upper[0].parameters())
