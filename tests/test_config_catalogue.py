from pathlib import Path

import yaml

from para_audio_id.codes import assign_codes


def test_primary_config_is_audio_lm_and_matches_logical_batch():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
    )
    assert cfg["architecture"] == "audio_lm_v1"
    assert cfg["tokenizer"]["selected_codebooks"] == 2
    assert cfg["model"]["max_position_embeddings"] == 512
    physical_documents = (
        cfg["train"]["tracks_per_microbatch"] * cfg["train"]["segments_per_track"]
    )
    assert physical_documents == 8
    assert physical_documents * cfg["trainer"]["accumulate_grad_batches"] == 64
    assert cfg["train"]["deterministic"]
    assert cfg["train"]["id_digit_weight"] == 20.0
    assert cfg["data"]["max_training_tracks"] == 10_000
    assert cfg["train"]["max_steps"] == 20_000
    assert cfg["train"]["warmup_steps"] == 200
    assert cfg["train"]["evaluation_interval"] == 500
    assert not any(key.startswith("id_loss_") for key in cfg["train"])


def test_random_code_mapping_is_complete_unique_and_seeded():
    first = assign_codes(100_000, seed=1337)
    second = assign_codes(100_000, seed=1337)
    assert first == second
    assert len(set(first)) == 100_000
    assert all(len(code) == 5 for code in first)
