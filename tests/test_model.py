import torch
from torch import nn

from para_audio_id.metrics import ranking_metrics
from para_audio_id.model import BeamResult, DigitDecoder, ParametricAudioIdentifier


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
