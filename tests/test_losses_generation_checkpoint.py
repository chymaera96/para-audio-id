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
from para_audio_id.audio_lm.evaluation import (
    _generation_metrics,
    joint_window_starts,
    select_checkpoint_cohort,
)
from para_audio_id.audio_lm.generation import (
    batched_beam_generate,
    batched_greedy_generate,
    batched_joint_beam_generate,
    beam_generate,
    greedy_generate,
    prompt_from_audio_tokens,
)
from para_audio_id.audio_lm.losses import (
    causal_audio_id_losses,
    causal_losses_by_view,
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


def test_tc16_four_codebook_generation_fits_512_position_context():
    vocabulary = AudioLMVocabulary(num_codebooks=4)
    cfg = tiny_config()
    cfg["model"]["max_position_embeddings"] = 512
    model = AudioCausalLM(cfg, vocabulary).eval()
    audio_tokens = torch.arange(200) % vocabulary.audio_size
    prompt = prompt_from_audio_tokens(audio_tokens, vocabulary)
    assert prompt.shape == (202,)
    result = greedy_generate(model, prompt, vocabulary)
    assert len(result.code) == 5
    assert result.ended_with_eos


def test_one_window_joint_beam_matches_regular_beam():
    vocabulary = AudioLMVocabulary()
    torch.manual_seed(17)
    model = AudioCausalLM(tiny_config(), vocabulary).eval()
    prompts = torch.tensor(
        [
            [vocabulary.bos_token_id, 1, 1025, vocabulary.id_token_id],
            [vocabulary.bos_token_id, 7, 1031, vocabulary.id_token_id],
        ]
    )
    regular = batched_beam_generate(model, prompts, vocabulary, width=10)
    joint = batched_joint_beam_generate(
        model, prompts[:, None, :], vocabulary, width=10
    )
    assert [[result.code for result in row] for row in joint] == [
        [result.code for result in row] for row in regular
    ]
    for joint_row, regular_row in zip(joint, regular, strict=True):
        assert [result.log_probability for result in joint_row] == pytest.approx(
            [result.log_probability for result in regular_row], abs=1e-5
        )


def test_joint_beam_scores_are_mean_window_log_probabilities():
    vocabulary = AudioLMVocabulary()
    torch.manual_seed(18)
    model = AudioCausalLM(tiny_config(), vocabulary).eval()
    prompts = torch.tensor(
        [
            [vocabulary.bos_token_id, 1, 1025, vocabulary.id_token_id],
            [vocabulary.bos_token_id, 9, 1033, vocabulary.id_token_id],
        ]
    )[None, :, :]
    ranking = batched_joint_beam_generate(model, prompts, vocabulary, width=5)[0]
    for candidate in ranking:
        per_window = []
        for prompt in prompts[0]:
            sequence = prompt[None, :]
            score = torch.tensor(0.0)
            for digit in candidate.code:
                log_probs = model(sequence)[
                    :, -1, vocabulary.digit_offset : vocabulary.digit_offset + 10
                ].log_softmax(dim=-1)
                score += log_probs[0, int(digit)]
                token = torch.tensor(
                    [[vocabulary.digit_offset + int(digit)]], dtype=torch.long
                )
                sequence = torch.cat((sequence, token), dim=1)
            score += model(sequence)[:, -1, :].log_softmax(dim=-1)[
                0, vocabulary.eos_token_id
            ]
            per_window.append(score)
        expected = torch.stack(per_window).mean()
        assert candidate.log_probability == pytest.approx(float(expected), abs=1e-5)


def test_joint_query_window_grid_uses_overlap_and_tail_alignment():
    assert joint_window_starts(48_000, 48_000, 24_000) == [0]
    assert joint_window_starts(72_000, 48_000, 24_000) == [0, 24_000]
    assert joint_window_starts(120_000, 48_000, 24_000) == [
        0,
        24_000,
        48_000,
        72_000,
    ]
    assert joint_window_starts(240_000, 48_000, 24_000) == list(
        range(0, 192_001, 24_000)
    )
    assert joint_window_starts(50_000, 48_000, 24_000) == [0, 2_000]


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


def test_checkpoint_identity_and_inference_loader_need_no_token_store(tmp_path):
    vocabulary = AudioLMVocabulary(num_codebooks=3)
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
        "state_dict": {
            f"model.{key}": value for key, value in model.state_dict().items()
        },
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
