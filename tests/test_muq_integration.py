import os

import pytest
import torch

from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.generation import greedy_generate, prompt_from_audio_tokens
from para_audio_id.audio_lm.losses import causal_losses_by_view
from para_audio_id.audio_lm.model import AudioCausalLM
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
    report = tokenizer.probe(torch.zeros(1, 120_000))
    assert report["raw_shape"][1] == 2
    assert report["serialized_tokens_per_example"] + 8 <= 512
    audio_tokens = tokenizer.tokenize(torch.zeros(1, 120_000))[0].cpu()
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
                "audio_tokens": audio_tokens,
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
                "view_type": "shifted",
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
        batch["view_type"],
        view_mode="paired",
        id_digit_weight=20.0,
    )
    loss.backward()
    assert set(per_view) == {"canonical", "shifted"}
    generated = greedy_generate(
        model.eval(),
        prompt_from_audio_tokens(audio_tokens.to(tokenizer.device), tokenizer.vocabulary),
        tokenizer.vocabulary,
    )
    assert len(generated.code) == 5
    assert generated.ended_with_eos
