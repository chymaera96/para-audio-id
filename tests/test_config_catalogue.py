from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from para_audio_id.audio_lm.tokenizer import TokenizerSpec
from para_audio_id.audio_lm.training import validate_tc8_query_configuration
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary
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
    assert cfg["data"]["segment_duration"] == 2.0
    assert cfg["tokenizer"]["sample_rate"] * cfg["data"]["segment_duration"] == 48_000
    assert cfg["train"]["id_digit_weight"] == 8.0
    assert cfg["data"]["max_training_tracks"] == 10_000
    assert cfg["train"]["max_steps"] == 70_000
    assert cfg["train"]["warmup_steps"] == 200
    assert cfg["train"]["evaluation_interval"] == 2_500
    assert cfg["train"]["checkpoint_interval"] == 500
    assert (
        cfg["train"]["schedule"]["protocol"]
        == "two_second_online_random_crop_noise_consistency_v1"
    )
    assert (
        cfg["train"]["schedule"]["loss_protocol"]
        == "tc5_family_weighted_consistency_v2"
    )
    assert cfg["train"]["schedule"]["clean_until_step"] == 20_000
    assert cfg["train"]["schedule"]["ramp_until_step"] == 25_000
    assert cfg["evaluation"]["monitor_tracks"] == 100
    assert cfg["evaluation"]["noise_snr_db"] == [0, 5, 10, 20, 30]
    assert cfg["data"]["background_noise"]["training_root"].endswith(
        "/bg_noise/train"
    )
    assert cfg["data"]["background_noise"]["validation_root"].endswith(
        "/bg_noise/test"
    )
    assert cfg["data"]["crop_retries"] == 4
    assert cfg["data"]["replacement_retries"] == 32
    assert not any("token_root" in key for key in cfg["data"])
    assert not any(key.startswith("id_loss_") for key in cfg["train"])


def test_tc8_startup_query_invariants_are_enforced():
    cfg = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
    )
    tokenizer_spec = TokenizerSpec(
        architecture="muq_mel_rvq",
        model_name="OpenMuQ/MuQ-large-msd-iter",
        revision="test",
        package_version="test",
        sample_rate=24_000,
        frame_rate=25.0,
        waveform_normalization="none_before_muq_internal_preprocessing",
        num_available_codebooks=8,
        selected_codebooks=2,
        codebook_size=1024,
        serialization="time_major_codebook_interleaved",
        preprocessing_version=1,
    )
    query_spec = validate_tc8_query_configuration(
        cfg, tokenizer_spec, AudioLMVocabulary()
    )
    assert query_spec["waveform_samples"] == 48_000
    assert query_spec["frames"] == 50
    assert query_spec["audio_targets"] == 100
    assert query_spec["digit_targets"] == 5
    assert query_spec["boundary_targets"] == 2
    assert query_spec["document_tokens"] == 108

    wrong_duration = deepcopy(cfg)
    wrong_duration["data"]["segment_duration"] = 5.0
    with pytest.raises(ValueError, match="segment_duration=2.0"):
        validate_tc8_query_configuration(
            wrong_duration, tokenizer_spec, AudioLMVocabulary()
        )

    wrong_weight = deepcopy(cfg)
    wrong_weight["train"]["id_digit_weight"] = 20.0
    with pytest.raises(ValueError, match="id_digit_weight=8"):
        validate_tc8_query_configuration(
            wrong_weight, tokenizer_spec, AudioLMVocabulary()
        )


def test_random_code_mapping_is_complete_unique_and_seeded():
    first = assign_codes(100_000, seed=1337)
    second = assign_codes(100_000, seed=1337)
    assert first == second
    assert len(set(first)) == 100_000
    assert all(len(code) == 5 for code in first)
