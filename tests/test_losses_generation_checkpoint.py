from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from para_audio_id.audio_lm.checkpoint import (
    ARCHITECTURE,
    load_audio_lm,
    validate_checkpoint_metadata,
)
from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.evaluation import _generation_metrics, select_checkpoint_cohort
from para_audio_id.audio_lm.generation import (
    batched_beam_generate,
    batched_greedy_generate,
    beam_generate,
    greedy_generate,
)
from para_audio_id.audio_lm.losses import (
    causal_audio_id_losses,
    causal_losses_by_view,
    noise_consistency_losses,
)
from para_audio_id.audio_lm.model import AudioCausalLM
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


def tiny_config():
    return {
        "architecture": ARCHITECTURE,
        "model": {
            "architecture": "gpt2",
            "num_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "max_position_embeddings": 32,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "tie_word_embeddings": True,
        },
    }


def test_single_weighted_causal_loss():
    vocabulary = AudioLMVocabulary()
    batch = collate_causal_documents(
        [
            {
                "audio_tokens": torch.tensor([1, 1025]),
                "code": "12345",
                "track_id": "track",
                "document_index": 0,
            }
        ],
        vocabulary,
        32,
    )
    torch.manual_seed(4)
    logits = torch.randn(1, batch["input_ids"].shape[1], vocabulary.size)
    results = causal_audio_id_losses(
        logits,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        id_digit_weight=20.0,
    )
    token_losses = F.cross_entropy(
        logits[:, :-1].transpose(1, 2),
        batch["input_ids"][:, 1:],
        reduction="none",
    )
    weights = (
        batch["audio_target_mask"].float()
        + 20.0 * batch["id_target_mask"].float()
        + batch["boundary_target_mask"].float()
    )
    assert torch.allclose(results["loss"], (token_losses * weights).sum() / weights.sum())
    separately_normalized = results["audio_loss"] + 20.0 * results["id_loss"]
    assert not torch.allclose(results["loss"], separately_normalized)


def test_generation_emits_exactly_five_digits_then_eos():
    vocabulary = AudioLMVocabulary()
    model = AudioCausalLM(tiny_config(), vocabulary).eval()
    prompt = torch.tensor([vocabulary.bos_token_id, 1, 1025, vocabulary.id_token_id])
    greedy = greedy_generate(model, prompt, vocabulary)
    beam = beam_generate(model, prompt, vocabulary, width=10)
    assert len(greedy.code) == 5 and greedy.code.isdecimal()
    assert greedy.ended_with_eos
    assert len(beam) == 10
    assert all(len(result.code) == 5 and result.code.isdecimal() for result in beam)
    assert all(result.ended_with_eos for result in beam)
    prompts = prompt.repeat(2, 1)
    batched_greedy = batched_greedy_generate(model, prompts, vocabulary)
    batched_beam = batched_beam_generate(model, prompts, vocabulary, width=5)
    assert len(batched_greedy) == 2
    assert [len(results) for results in batched_beam] == [5, 5]


def test_cached_greedy_matches_full_prefix_decoding():
    vocabulary = AudioLMVocabulary()
    torch.manual_seed(12)
    model = AudioCausalLM(tiny_config(), vocabulary).eval()
    prompt = torch.tensor(
        [[vocabulary.bos_token_id, 1, 1025, vocabulary.id_token_id]]
    )
    sequence = prompt
    expected_digits = []
    expected_score = torch.tensor(0.0)
    with torch.inference_mode():
        for _ in range(5):
            log_probs = model(sequence)[
                :, -1, vocabulary.digit_offset : vocabulary.digit_offset + 10
            ].log_softmax(dim=-1)
            digit = log_probs.argmax(dim=-1)
            expected_score += log_probs[0, int(digit.item())]
            expected_digits.append(int(digit.item()))
            sequence = torch.cat(
                (sequence, (digit + vocabulary.digit_offset)[:, None]), dim=1
            )
        expected_score += model(sequence)[:, -1, :].log_softmax(dim=-1)[
            0, vocabulary.eos_token_id
        ]
    result = batched_greedy_generate(model, prompt, vocabulary)[0]
    expected_code = "".join(str(digit) for digit in expected_digits)
    assert result.code == expected_code
    assert result.log_probability == pytest.approx(float(expected_score), abs=1e-5)


def test_paired_loss_is_equal_mean_of_independent_view_losses():
    vocabulary = AudioLMVocabulary()
    examples = [
        {
            "audio_tokens": torch.tensor([1, 1025]),
            "code": "12345",
            "track_id": "track",
            "document_index": index,
            "view_type": view,
        }
        for index, view in enumerate(("canonical", "shifted"))
    ]
    batch = collate_causal_documents(examples, vocabulary, 32)
    torch.manual_seed(8)
    logits = torch.randn(2, batch["input_ids"].shape[1], vocabulary.size)
    loss, _, per_view = causal_losses_by_view(
        logits,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        batch["view_type"],
        view_mode="paired",
        id_digit_weight=20.0,
    )
    expected = 0.5 * (
        per_view["canonical"]["loss"] + per_view["shifted"]["loss"]
    )
    assert torch.allclose(loss, expected)


def test_anchor_secondary_loss_is_equal_mean_after_noisy_replacement():
    vocabulary = AudioLMVocabulary()
    examples = [
        {
            "audio_tokens": torch.tensor([1, 1025]),
            "code": "12345",
            "track_id": "track",
            "document_index": index,
            "view_type": view,
        }
        for index, view in enumerate(("canonical", "noisy"))
    ]
    batch = collate_causal_documents(examples, vocabulary, 32)
    torch.manual_seed(9)
    logits = torch.randn(2, batch["input_ids"].shape[1], vocabulary.size)
    loss, _, per_role = causal_losses_by_view(
        logits,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        ["anchor", "secondary"],
        view_mode="paired_roles",
        id_digit_weight=20.0,
    )
    assert torch.allclose(
        loss,
        0.5 * (
            per_role["anchor"]["loss"] + per_role["secondary"]["loss"]
        ),
    )


def test_tc6_masks_noisy_audio_and_preserves_id_boundary_gradients():
    vocabulary = AudioLMVocabulary()
    examples = [
        {
            "audio_tokens": torch.tensor([1, 1025]),
            "code": "12345",
            "track_id": "track",
            "document_index": index,
            "view_type": view,
        }
        for index, view in enumerate(("canonical", "noisy"))
    ]
    batch = collate_causal_documents(examples, vocabulary, 32)
    torch.manual_seed(17)
    logits = torch.randn(
        2,
        batch["input_ids"].shape[1],
        vocabulary.size,
        requires_grad=True,
    )
    hidden = torch.randn(
        2,
        batch["input_ids"].shape[1],
        8,
        requires_grad=True,
    )
    is_noisy = torch.tensor([False, True])
    loss, metrics = noise_consistency_losses(
        logits,
        hidden,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        is_noisy,
        batch["track_id"],
        id_digit_weight=20.0,
        consistency_weight=0.1,
    )
    expected = (
        (
            2 * metrics["clean_audio_loss"]
            + 100 * metrics["digit_loss"]
            + 2 * metrics["boundary_loss"]
        )
        / 104
        + 0.1 * metrics["consistency_loss"]
    )
    assert torch.allclose(loss, expected)
    loss.backward()
    noisy_audio_positions = batch["audio_target_mask"][1].nonzero().flatten()
    assert not logits.grad[1, noisy_audio_positions].any()
    assert logits.grad[1, batch["id_target_mask"][1].nonzero().flatten()].any()
    assert logits.grad[
        1, batch["boundary_target_mask"][1].nonzero().flatten()
    ].any()
    id_column = int(batch["id_target_mask"][0].long().argmax())
    assert not hidden.grad[0, id_column].any()
    assert hidden.grad[1, id_column].any()
    assert not metrics["legacy_weighted_token_loss"].requires_grad


def test_tc6_clean_base_loss_exactly_matches_tc5_weighted_loss():
    vocabulary = AudioLMVocabulary()
    audio = torch.tensor(
        [
            value
            for frame in range(125)
            for value in (frame % 1024, 1024 + frame % 1024)
        ]
    )
    examples = [
        {
            "audio_tokens": audio,
            "code": "12345",
            "track_id": f"track-{index}",
            "document_index": index,
        }
        for index in range(2)
    ]
    batch = collate_causal_documents(examples, vocabulary, 512)
    torch.manual_seed(23)
    logits = torch.randn(
        2, batch["input_ids"].shape[1], vocabulary.size
    )
    hidden = torch.randn(2, batch["input_ids"].shape[1], 8)
    total, metrics = noise_consistency_losses(
        logits,
        hidden,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        torch.tensor([False, False]),
        batch["track_id"],
        id_digit_weight=20.0,
        consistency_weight=0.0,
    )
    assert torch.allclose(total, metrics["legacy_weighted_token_loss"])
    assert metrics["audio_family_coefficient"] == pytest.approx(250 / 352)
    assert metrics["digit_family_coefficient"] == pytest.approx(100 / 352)
    assert metrics["boundary_family_coefficient"] == pytest.approx(2 / 352)


def test_tc6_consistency_is_zero_for_identical_id_states():
    vocabulary = AudioLMVocabulary()
    examples = [
        {
            "audio_tokens": torch.tensor([1, 1025]),
            "code": "12345",
            "track_id": "track",
            "document_index": index,
        }
        for index in range(2)
    ]
    batch = collate_causal_documents(examples, vocabulary, 32)
    logits = torch.randn(2, batch["input_ids"].shape[1], vocabulary.size)
    hidden = torch.randn(1, batch["input_ids"].shape[1], 8).repeat(2, 1, 1)
    _, metrics = noise_consistency_losses(
        logits,
        hidden,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        torch.tensor([False, True]),
        batch["track_id"],
        id_digit_weight=20.0,
        consistency_weight=0.1,
    )
    assert metrics["consistency_loss"] == pytest.approx(0.0, abs=1e-6)


def test_tc8_two_second_family_weights_match_reported_objective():
    vocabulary = AudioLMVocabulary()
    audio = torch.tensor(
        [
            value
            for frame in range(50)
            for value in (frame % 1024, 1024 + frame % 1024)
        ]
    )
    examples = [
        {
            "audio_tokens": audio,
            "code": "12345",
            "track_id": f"tc8-track-{index}",
            "document_index": index,
        }
        for index in range(2)
    ]
    batch = collate_causal_documents(examples, vocabulary, 512)
    torch.manual_seed(29)
    logits = torch.randn(2, batch["input_ids"].shape[1], vocabulary.size)
    hidden = torch.randn(2, batch["input_ids"].shape[1], 8)
    total, metrics = noise_consistency_losses(
        logits,
        hidden,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        torch.tensor([False, False]),
        batch["track_id"],
        id_digit_weight=8.0,
        consistency_weight=0.0,
    )
    assert batch["input_ids"].shape[1] == 108
    assert batch["audio_target_mask"].sum(dim=1).tolist() == [100, 100]
    assert torch.allclose(total, metrics["legacy_weighted_token_loss"])
    assert metrics["audio_family_coefficient"] == pytest.approx(100 / 142)
    assert metrics["digit_family_coefficient"] == pytest.approx(40 / 142)
    assert metrics["boundary_family_coefficient"] == pytest.approx(2 / 142)
    assert (
        metrics["audio_family_coefficient"]
        / metrics["digit_family_coefficient"]
    ) == pytest.approx(2.5)


def test_tc6_noisy_id_path_reaches_input_embeddings_and_transformer():
    vocabulary = AudioLMVocabulary()
    cfg = tiny_config()
    cfg["model"]["tie_word_embeddings"] = False
    model = AudioCausalLM(cfg, vocabulary)
    examples = [
        {
            "audio_tokens": torch.tensor(tokens),
            "code": "12345",
            "track_id": "track",
            "document_index": index,
        }
        for index, tokens in enumerate(([1, 1025], [77, 1101]))
    ]
    batch = collate_causal_documents(examples, vocabulary, 32)
    logits, hidden = model(
        batch["input_ids"],
        batch["attention_mask"],
        return_final_hidden_state=True,
    )
    loss, _ = noise_consistency_losses(
        logits,
        hidden,
        batch["input_ids"],
        batch["audio_target_mask"],
        batch["id_target_mask"],
        batch["boundary_target_mask"],
        torch.tensor([False, True]),
        batch["track_id"],
        id_digit_weight=20.0,
        consistency_weight=0.1,
    )
    loss.backward()
    embeddings = model.network.transformer.wte.weight.grad
    assert embeddings[77].abs().sum() > 0
    assert model.network.transformer.h[0].attn.c_attn.weight.grad.abs().sum() > 0


def test_checkpoint_identity_and_inference_loader_need_no_token_store(tmp_path):
    vocabulary = AudioLMVocabulary()
    cfg = tiny_config()
    model = AudioCausalLM(cfg, vocabulary)
    metadata = {
        "architecture": ARCHITECTURE,
        "tokenizer_spec": {"architecture": "dummy"},
        "tokenizer_fingerprint": "fingerprint",
        "vocabulary": vocabulary.to_dict(),
        "model_config": cfg["model"],
        "code_mapping_fingerprint": "mapping",
        "validation_probe": ["track"],
        "training_track_ids": ["track"],
        "training_corpus_fingerprint": "corpus",
        "view_policy": {"view_mode": "paired"},
    }
    validate_checkpoint_metadata(metadata, tokenizer_fingerprint="fingerprint")
    with pytest.raises(ValueError, match="architecture"):
        validate_checkpoint_metadata({**metadata, "architecture": "legacy"})
    checkpoint = {
        **metadata,
        "hyper_parameters": cfg,
        "state_dict": {f"model.{key}": value for key, value in model.state_dict().items()},
    }
    path = Path(tmp_path) / "model.ckpt"
    torch.save(checkpoint, path)
    loaded, loaded_vocabulary, _, _ = load_audio_lm(path)
    assert loaded_vocabulary == vocabulary
    assert all(
        torch.equal(left, right)
        for left, right in zip(model.parameters(), loaded.parameters(), strict=True)
    )


def test_checkpoint_training_cohort_selection_is_exact():
    checkpoint = {
        "validation_probe": ["probe"],
        "training_track_ids": ["track-2", "track-1"],
    }
    assert select_checkpoint_cohort(
        checkpoint, cohort="training", expected_tracks=2
    ) == ["track-2", "track-1"]
    with pytest.raises(ValueError, match="Expected exactly 1000"):
        select_checkpoint_cohort(
            checkpoint, cohort="training", expected_tracks=1000
        )
    first = select_checkpoint_cohort(
        checkpoint, cohort="training", sample_tracks=1, sample_seed=7
    )
    second = select_checkpoint_cohort(
        checkpoint, cohort="training", sample_tracks=1, sample_seed=7
    )
    assert first == second
    assert len(first) == 1 and first[0] in checkpoint["training_track_ids"]
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_checkpoint_cohort(
            checkpoint,
            cohort="training",
            max_tracks=1,
            sample_tracks=1,
        )


def test_position_generation_metrics_are_free_running():
    rows = [
        {
            "code": "00001",
            "greedy": "00001",
            "greedy_ended_with_eos": True,
            "beam": [{"code": "00001"}],
        },
        {
            "code": "00002",
            "greedy": "99999",
            "greedy_ended_with_eos": True,
            "beam": [{"code": "99999"}, {"code": "00002"}],
        },
    ]
    metrics = _generation_metrics(rows)
    assert metrics["greedy_top1"] == 0.5
    assert metrics["beam_top1"] == 0.5
    assert metrics["beam_top5"] == 1.0
