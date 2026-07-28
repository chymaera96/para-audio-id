import os

import pytest
import torch

from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.generation import greedy_generate, prompt_from_audio_tokens
from para_audio_id.audio_lm.losses import causal_losses_by_view
from para_audio_id.audio_lm.model import AudioCausalLM
from para_audio_id.audio_lm.noise import mix_background_noise
from para_audio_id.audio_lm.tokenizer import MuQRVQTokenizer


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MUQ_INTEGRATION") != "1",
    reason="set RUN_MUQ_INTEGRATION=1 to load the real MuQ checkpoint",
)
def test_real_muq_rvq_probe():
    tokenizer = MuQRVQTokenizer(
        "OpenMuQ/MuQ-large-msd-iter",
        selected_codebooks=2,
        device=os.environ.get("MUQ_DEVICE", "cuda"),
    )
    waveform = torch.sin(
        2 * torch.pi * 220 * torch.arange(120_000) / 24_000
    ).unsqueeze(0)
    report = tokenizer.probe(waveform)
    assert report["raw_shape"][1] == 2
    assert report["serialized_tokens_per_example"] + 8 <= 512
    audio_tokens = tokenizer.tokenize(waveform)[0].cpu()
    lightweight = MuQRVQTokenizer(
        "OpenMuQ/MuQ-large-msd-iter",
        revision=tokenizer.revision,
        selected_codebooks=2,
        device=os.environ.get("MUQ_DEVICE", "cuda"),
        lightweight=True,
    )
    lightweight_tokens = lightweight.tokenize(waveform)
    assert torch.equal(lightweight_tokens[0].cpu(), audio_tokens)
    mixed, valid = mix_background_noise(
        waveform.to(lightweight.device),
        waveform.roll(7_000, dims=1).to(lightweight.device),
        torch.tensor([5.0], device=lightweight.device),
    )
    assert valid.all()
    noisy_tokens = lightweight.tokenize(mixed)[0].cpu()
    cfg = {
        "model": {
            "architecture": "gpt2",
            "num_layers": 1,
            "hidden_size": 64,
            "num_attention_heads": 4,
            "max_position_embeddings": 512,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "tie_word_embeddings": True,
        }
    }
    model = AudioCausalLM(cfg, tokenizer.vocabulary).to(tokenizer.device)
    batch = collate_causal_documents(
        [
            {
                "audio_tokens": noisy_tokens,
                "code": "01234",
                "track_id": "integration",
                "document_index": 0,
                "view_type": "canonical",
            },
            {
                "audio_tokens": audio_tokens,
                "code": "01234",
                "track_id": "integration",
                "document_index": 1,
                "view_type": "noisy",
            },
        ],
        tokenizer.vocabulary,
        512,
    )
    logits = model(
        batch["input_ids"].to(tokenizer.device),
        batch["attention_mask"].to(tokenizer.device),
    )
    loss, _, per_view = causal_losses_by_view(
        logits,
        batch["input_ids"].to(tokenizer.device),
        batch["audio_target_mask"].to(tokenizer.device),
        batch["id_target_mask"].to(tokenizer.device),
        batch["boundary_target_mask"].to(tokenizer.device),
        ["anchor", "secondary"],
        view_mode="paired_roles",
        id_digit_weight=20.0,
    )
    loss.backward()
    assert set(per_view) == {"anchor", "secondary"}
    generated = greedy_generate(
        model.eval(),
        prompt_from_audio_tokens(audio_tokens.to(tokenizer.device), tokenizer.vocabulary),
        tokenizer.vocabulary,
    )
    assert len(generated.code) == 5
    assert generated.ended_with_eos
