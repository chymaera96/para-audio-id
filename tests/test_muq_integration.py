import os

import pytest
import torch

from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.generation import greedy_generate, prompt_from_audio_tokens
from para_audio_id.audio_lm.losses import noise_consistency_losses
from para_audio_id.audio_lm.model import AudioCausalLM
from para_audio_id.audio_lm.noise import mix_background_noise
from para_audio_id.audio_lm.tokenizer import MuQRVQTokenizer


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("RUN_MUQ_INTEGRATION") != "1",
    reason="set RUN_MUQ_INTEGRATION=1 to load the real MuQ checkpoint",
)
def test_real_muq_tc13_eight_document_probe():
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
    assert report["serialized_tokens_per_example"] == 250
    audio_tokens = tokenizer.tokenize(waveform)[0].cpu()
    lightweight = MuQRVQTokenizer(
        "OpenMuQ/MuQ-large-msd-iter",
        revision=tokenizer.revision,
        selected_codebooks=2,
        device=os.environ.get("MUQ_DEVICE", "cuda"),
        lightweight=True,
    )
    anchors = torch.cat(
        [waveform.roll(index * 1_000, dims=1) for index in range(4)]
    ).to(lightweight.device)
    noises = anchors.roll(7_000, dims=1)
    mixed, valid = mix_background_noise(
        anchors,
        noises,
        torch.tensor(
            [0.0, 10.0, 20.0, 30.0],
            device=lightweight.device,
        ),
    )
    assert valid.all()
    online_waveforms = torch.stack(
        [
            value
            for pair in zip(anchors, mixed, strict=True)
            for value in pair
        ]
    )
    lightweight_tokens = lightweight.tokenize(online_waveforms)
    assert torch.equal(lightweight_tokens[0].cpu(), audio_tokens)
    assert lightweight_tokens.shape == (8, 250)
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
    examples = []
    is_noisy = []
    for pair in range(4):
        for role, tokens in enumerate(
            lightweight_tokens[pair * 2 : pair * 2 + 2].cpu()
        ):
            examples.append(
                {
                    "audio_tokens": tokens,
                    "code": f"{pair:05d}",
                    "track_id": f"integration-{pair}",
                    "document_index": pair * 2 + role,
                    "view_type": "noisy" if role else "random_clean",
                }
            )
            is_noisy.append(bool(role))
    batch = collate_causal_documents(
        examples,
        tokenizer.vocabulary,
        512,
    )
    assert batch["input_ids"].shape == (8, 258)
    assert int(batch["audio_target_mask"].sum()) == 2_000
    logits, hidden = model(
        batch["input_ids"].to(tokenizer.device),
        batch["attention_mask"].to(tokenizer.device),
        return_final_hidden_state=True,
    )
    loss, metrics = noise_consistency_losses(
        logits,
        hidden,
        batch["input_ids"].to(tokenizer.device),
        batch["audio_target_mask"].to(tokenizer.device),
        batch["id_target_mask"].to(tokenizer.device),
        batch["boundary_target_mask"].to(tokenizer.device),
        torch.tensor(is_noisy, device=tokenizer.device),
        batch["track_id"],
        id_digit_weight=20.0,
        consistency_weight=0.1,
    )
    loss.backward()
    assert torch.isfinite(metrics["consistency_loss"])
    generated = greedy_generate(
        model.eval(),
        prompt_from_audio_tokens(audio_tokens.to(tokenizer.device), tokenizer.vocabulary),
        tokenizer.vocabulary,
    )
    assert len(generated.code) == 5
    assert generated.ended_with_eos
