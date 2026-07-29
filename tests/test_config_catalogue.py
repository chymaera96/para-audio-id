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
    assert cfg["train"]["max_steps"] == 70_000
    assert cfg["train"]["warmup_steps"] == 200
    assert cfg["train"]["evaluation_interval"] == 2_500
    assert cfg["train"]["checkpoint_interval"] == 500
    assert (
        cfg["train"]["curriculum"]["protocol"]
        == "noise_consistency_curriculum_v1"
    )
    assert (
        cfg["train"]["curriculum"]["loss_protocol"]
        == "tc5_family_weighted_consistency_v2"
    )
    assert cfg["train"]["curriculum"]["gate_threshold"] == 0.5
    assert cfg["evaluation"]["monitor_tracks"] == 100
    assert cfg["evaluation"]["noise_snr_db"] == [0, 5, 10, 20, 30]
    assert cfg["data"]["background_noise"]["training_root"].endswith(
        "/bg_noise/train"
    )
    assert cfg["data"]["background_noise"]["validation_root"].endswith(
        "/bg_noise/test"
    )
    assert cfg["data"]["view_mode"] == "paired"
    assert len(cfg["data"]["canonical_starts"]) == 6
    assert len(cfg["data"]["shifted_training_starts"]) == 20
    assert len(cfg["data"]["shifted_evaluation_starts"]) == 5
    assert not (
        set(cfg["data"]["shifted_training_starts"])
        & set(cfg["data"]["shifted_evaluation_starts"])
    )
    assert not any(key.startswith("id_loss_") for key in cfg["train"])


def test_random_code_mapping_is_complete_unique_and_seeded():
    first = assign_codes(100_000, seed=1337)
    second = assign_codes(100_000, seed=1337)
    assert first == second
    assert len(set(first)) == 100_000
    assert all(len(code) == 5 for code in first)
