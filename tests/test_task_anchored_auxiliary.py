from __future__ import annotations

import pytest
import torch
from torch import nn

from para_audio_id.audio_lm.auxiliary import (
    TaskAnchoredAuxiliary,
    task_anchored_total_loss,
)
from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.losses import degraded_causal_base_losses
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


def _batch(codes: list[str]) -> dict:
    vocabulary = AudioLMVocabulary()
    return collate_causal_documents(
        [
            {
                "audio_tokens": torch.tensor([1, 1025]),
                "code": code,
                "track_id": f"track-{index // 2}",
                "document_index": index,
            }
            for index, code in enumerate(codes)
        ],
        vocabulary,
        32,
    )


def _auxiliary(hidden_size: int = 8) -> TaskAnchoredAuxiliary:
    return TaskAnchoredAuxiliary(
        hidden_size,
        id_token_id=AudioLMVocabulary().id_token_id,
        projector_hidden_size=16,
        projection_size=4,
    )


def test_identifier_digit_targets_and_summary_shape():
    batch = _batch(["01234", "98765"])
    assert batch["identifier_digits"].tolist() == [
        [0, 1, 2, 3, 4],
        [9, 8, 7, 6, 5],
    ]
    auxiliary = _auxiliary()
    logits = auxiliary.summary_head(torch.randn(2, 8)).view(2, 5, 10)
    assert logits.shape == (2, 5, 10)
    assert any(isinstance(module, nn.LayerNorm) for module in auxiliary.modules())
    assert not any(isinstance(module, nn.BatchNorm1d) for module in auxiliary.modules())
    with pytest.raises(ValueError, match="five decimal digits"):
        _batch(["12x45"])


def test_id_state_is_selected_before_any_identifier_digit():
    batch = _batch(["01234"])
    hidden = torch.zeros(1, batch["input_ids"].shape[1], 3)
    id_column = int(batch["id_target_mask"][0].long().argmax())
    hidden[0, id_column] = torch.tensor([1.0, 2.0, 3.0])
    hidden[0, id_column + 1] = torch.tensor([9.0, 9.0, 9.0])
    selected = _auxiliary(3).id_states(
        hidden, batch["input_ids"], batch["id_target_mask"]
    )
    assert selected.tolist() == [[1.0, 2.0, 3.0]]

    invalid_input_ids = batch["input_ids"].clone()
    invalid_input_ids[0, id_column] = 0
    with pytest.raises(ValueError, match=r"\[ID\] input state"):
        _auxiliary(3).id_states(
            hidden, invalid_input_ids, batch["id_target_mask"]
        )


def test_summary_exact_requires_all_five_digits():
    auxiliary = TaskAnchoredAuxiliary(
        5,
        id_token_id=AudioLMVocabulary().id_token_id,
        projector_hidden_size=8,
        projection_size=4,
    )
    with torch.no_grad():
        auxiliary.summary_head.weight.zero_()
        auxiliary.summary_head.bias.fill_(-10.0)
        for position, digit in enumerate((0, 1, 2, 3, 9)):
            auxiliary.summary_head.bias[position * 10 + digit] = 10.0
    batch = _batch(["01234"])
    hidden = torch.zeros(1, batch["input_ids"].shape[1], 5)
    metrics = auxiliary(
        hidden,
        batch["input_ids"],
        batch["id_target_mask"],
        batch["identifier_digits"],
        torch.tensor([False]),
        batch["track_id"],
    )
    assert metrics["summary_digit_accuracy"] == pytest.approx(0.8)
    assert metrics["summary_exact_accuracy"] == pytest.approx(0.0)


def test_predictive_pairing_and_gradient_asymmetry():
    torch.manual_seed(3)
    auxiliary = _auxiliary()
    batch = _batch(["01234", "01234", "56789", "56789"])
    hidden = torch.randn(
        4, batch["input_ids"].shape[1], 8, requires_grad=True
    )
    metrics = auxiliary(
        hidden,
        batch["input_ids"],
        batch["id_target_mask"],
        batch["identifier_digits"],
        torch.tensor([False, True, False, True]),
        batch["track_id"],
    )
    assert torch.isfinite(metrics["predictive_loss"])
    assert torch.isfinite(metrics["shuffled_prediction_cosine"])
    assert torch.isfinite(metrics["prediction_cosine_margin"])
    metrics["predictive_loss"].backward()
    id_column = int(batch["id_target_mask"][0].long().argmax())
    assert not hidden.grad[0, id_column].any()
    assert hidden.grad[1, id_column].any()
    assert not hidden.grad[2, id_column].any()
    assert hidden.grad[3, id_column].any()
    assert any(
        parameter.grad is not None and parameter.grad.any()
        for parameter in auxiliary.projector.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.any()
        for parameter in auxiliary.predictor.parameters()
    )
    assert all(parameter.grad is None for parameter in auxiliary.summary_head.parameters())


def test_summary_gradient_does_not_reach_projector_or_predictor():
    torch.manual_seed(4)
    auxiliary = _auxiliary()
    batch = _batch(["01234", "01234"])
    hidden = torch.randn(
        2, batch["input_ids"].shape[1], 8, requires_grad=True
    )
    metrics = auxiliary(
        hidden,
        batch["input_ids"],
        batch["id_target_mask"],
        batch["identifier_digits"],
        torch.tensor([False, True]),
        batch["track_id"],
    )
    metrics["summary_loss"].backward()
    id_column = int(batch["id_target_mask"][0].long().argmax())
    assert hidden.grad[:, id_column].any()
    assert any(parameter.grad is not None for parameter in auxiliary.summary_head.parameters())
    assert all(parameter.grad is None for parameter in auxiliary.projector.parameters())
    assert all(parameter.grad is None for parameter in auxiliary.predictor.parameters())


def test_no_degradation_returns_differentiable_zero_and_unavailable_pairs():
    auxiliary = _auxiliary()
    batch = _batch(["01234", "01234"])
    hidden = torch.randn(
        2, batch["input_ids"].shape[1], 8, requires_grad=True
    )
    metrics = auxiliary(
        hidden,
        batch["input_ids"],
        batch["id_target_mask"],
        batch["identifier_digits"],
        torch.tensor([False, False]),
        batch["track_id"],
    )
    assert metrics["predictive_loss"].requires_grad
    assert float(metrics["predictive_loss"].detach()) == pytest.approx(0.0)
    assert torch.isnan(metrics["paired_prediction_cosine"])
    assert torch.isnan(metrics["prediction_cosine_margin"])
    metrics["predictive_loss"].backward()


def test_pair_contract_rejects_wrong_layout_or_identity():
    auxiliary = _auxiliary()
    batch = _batch(["01234", "01234"])
    hidden = torch.randn(2, batch["input_ids"].shape[1], 8)
    with pytest.raises(ValueError, match="second row"):
        auxiliary(
            hidden,
            batch["input_ids"],
            batch["id_target_mask"],
            batch["identifier_digits"],
            torch.tensor([True, False]),
            batch["track_id"],
        )
    with pytest.raises(ValueError, match="share a track"):
        auxiliary(
            hidden,
            batch["input_ids"],
            batch["id_target_mask"],
            batch["identifier_digits"],
            torch.tensor([False, True]),
            ["first", "second"],
        )


def test_degraded_base_loss_masks_audio_and_preserves_tc13_coefficients():
    vocabulary = AudioLMVocabulary()
    audio = torch.tensor(
        [value for frame in range(125) for value in (frame, 1024 + frame)]
    )
    batch = collate_causal_documents(
        [
            {
                "audio_tokens": audio,
                "code": "01234",
                "track_id": "track",
                "document_index": index,
            }
            for index in range(2)
        ],
        vocabulary,
        512,
    )
    logits = torch.randn(
        2, batch["input_ids"].shape[1], vocabulary.size, requires_grad=True
    )
    base, metrics = degraded_causal_base_losses(
        logits,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        torch.tensor([False, True]),
        id_digit_weight=20.0,
    )
    assert metrics["audio_family_coefficient"] == pytest.approx(250 / 352)
    assert metrics["digit_family_coefficient"] == pytest.approx(100 / 352)
    assert metrics["boundary_family_coefficient"] == pytest.approx(2 / 352)
    base.backward()
    noisy_audio = batch["audio_target_mask"][1].nonzero().flatten()
    assert not logits.grad[1, noisy_audio].any()
    assert logits.grad[1, batch["id_target_mask"][1].nonzero().flatten()].any()
    assert logits.grad[1, batch["boundary_target_mask"][1].nonzero().flatten()].any()


def test_total_loss_uses_only_base_summary_and_predictive_terms():
    base = torch.tensor(2.0)
    metrics = {
        "summary_loss": torch.tensor(3.0),
        "predictive_loss": torch.tensor(4.0),
    }
    total, contributions = task_anchored_total_loss(
        base, metrics, summary_weight=0.1, predictive_weight=0.05
    )
    assert total == pytest.approx(2.5)
    assert contributions["summary_contribution"] == pytest.approx(0.3)
    assert contributions["predictive_contribution"] == pytest.approx(0.2)
