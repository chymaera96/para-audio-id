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
from para_audio_id.audio_lm.generation import beam_generate, greedy_generate
from para_audio_id.audio_lm.losses import causal_audio_id_losses
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
