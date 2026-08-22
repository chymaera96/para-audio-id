from __future__ import annotations

import pytest
import torch

import para_audio_id.audio_lm.losses as loss_module
from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.losses import (
    degraded_causal_base_losses,
    identifier_logit_distillation_loss,
)
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


def _batch(codes: list[str]) -> tuple[dict, AudioLMVocabulary]:
    vocabulary = AudioLMVocabulary(num_codebooks=4)
    batch = collate_causal_documents(
        [
            {
                "audio_tokens": torch.tensor([1, 1025, 2049, 3073]),
                "code": code,
                "track_id": f"track-{index // 2}",
                "document_index": index,
            }
            for index, code in enumerate(codes)
        ],
        vocabulary,
        32,
    )
    return batch, vocabulary


def _kd(logits, batch, vocabulary, is_degraded):
    return identifier_logit_distillation_loss(
        logits,
        batch["id_target_mask"],
        is_degraded,
        batch["track_id"],
        list(vocabulary.digit_token_ids),
        temperature=2.0,
    )


def test_distillation_selects_next_digit_positions_and_digit_vocabulary_only():
    batch, vocabulary = _batch(["01234", "01234"])
    shifted_targets = batch["input_ids"][:, 1:]
    assert shifted_targets[batch["id_target_mask"]].reshape(2, 5).tolist() == [
        vocabulary.encode_code("01234").tolist(),
        vocabulary.encode_code("01234").tolist(),
    ]
    first_digit_position = int(batch["id_target_mask"][0].nonzero()[0])
    assert batch["input_ids"][0, first_digit_position] == vocabulary.id_token_id
    logits = torch.randn(
        2, batch["input_ids"].shape[1], vocabulary.size, requires_grad=True
    )
    loss = _kd(logits, batch, vocabulary, torch.tensor([False, True]))
    loss.backward()

    selected_columns = batch["id_target_mask"][1].nonzero().flatten()
    assert len(selected_columns) == 5
    assert logits.grad[1, selected_columns][:, list(vocabulary.digit_token_ids)].any()
    unselected = torch.ones_like(logits.grad, dtype=torch.bool)
    unselected[1, selected_columns[:, None], torch.tensor(list(vocabulary.digit_token_ids))] = False
    assert not logits.grad[unselected].any()


def test_identical_digit_logits_have_zero_distillation_loss():
    batch, vocabulary = _batch(["01234", "01234"])
    logits = torch.randn(1, batch["input_ids"].shape[1], vocabulary.size).repeat(
        2, 1, 1
    )
    loss = _kd(logits, batch, vocabulary, torch.tensor([False, True]))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_distillation_detaches_clean_teacher_but_trains_degraded_logits():
    batch, vocabulary = _batch(["01234", "01234"])
    logits = torch.randn(
        2, batch["input_ids"].shape[1], vocabulary.size, requires_grad=True
    )
    loss = _kd(logits, batch, vocabulary, torch.tensor([False, True]))
    loss.backward()
    assert not logits.grad[0].any()
    assert logits.grad[1].any()


def test_clean_only_distillation_is_differentiable_zero():
    batch, vocabulary = _batch(["01234", "01234"])
    logits = torch.randn(
        2, batch["input_ids"].shape[1], vocabulary.size, requires_grad=True
    )
    loss = _kd(logits, batch, vocabulary, torch.tensor([False, False]))
    assert loss.requires_grad
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert not logits.grad.any()


def test_distillation_rejects_invalid_pair_layout_or_track():
    batch, vocabulary = _batch(["01234", "01234"])
    logits = torch.randn(2, batch["input_ids"].shape[1], vocabulary.size)
    with pytest.raises(ValueError, match="second row"):
        _kd(logits, batch, vocabulary, torch.tensor([True, False]))
    batch["track_id"] = ["first", "second"]
    with pytest.raises(ValueError, match="share a track"):
        _kd(logits, batch, vocabulary, torch.tensor([False, True]))


def test_eight_codebook_base_loss_masks_degraded_audio_and_has_expected_coefficients():
    vocabulary = AudioLMVocabulary(num_codebooks=8)
    audio = torch.tensor(
        [
            value
            for frame in range(50)
            for value in (
                frame,
                1024 + frame,
                2048 + frame,
                3072 + frame,
                4096 + frame,
                5120 + frame,
                6144 + frame,
                7168 + frame,
            )
        ]
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
        id_digit_weight=32.0,
    )
    assert metrics["audio_family_coefficient"] == pytest.approx(400 / 562)
    assert metrics["digit_family_coefficient"] == pytest.approx(160 / 562)
    assert metrics["boundary_family_coefficient"] == pytest.approx(2 / 562)
    base.backward()
    degraded_audio = batch["audio_target_mask"][1].nonzero().flatten()
    assert not logits.grad[1, degraded_audio].any()
    assert logits.grad[1, batch["id_target_mask"][1].nonzero().flatten()].any()
    assert logits.grad[1, batch["boundary_target_mask"][1].nonzero().flatten()].any()


def test_tc18_base_computes_vocabulary_cross_entropy_once(monkeypatch):
    batch, vocabulary = _batch(["01234", "01234"])
    logits = torch.randn(2, batch["input_ids"].shape[1], vocabulary.size)
    calls = 0
    original = loss_module.F.cross_entropy

    def counted_cross_entropy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(loss_module.F, "cross_entropy", counted_cross_entropy)
    degraded_causal_base_losses(
        logits,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        torch.tensor([False, True]),
        id_digit_weight=32.0,
    )
    assert calls == 1
