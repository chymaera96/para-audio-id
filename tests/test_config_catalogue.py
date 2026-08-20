from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch
import yaml

from para_audio_id.audio_lm.generation import prompt_from_audio_tokens
from para_audio_id.audio_lm.tokenizer import TokenizerSpec
from para_audio_id.audio_lm.profiles import resolve_training_config
from para_audio_id.audio_lm.training import (
    validate_tc16_batch_configuration,
    validate_tc16_query_configuration,
)
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary
from para_audio_id.codes import assign_codes
from prepare_training_cohort import prepare_training_cohort


def test_primary_config_is_audio_lm_and_matches_logical_batch():
    cfg = resolve_training_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
        )
    )
    assert cfg["architecture"] == "audio_lm_v1"
    assert cfg["tokenizer"]["selected_codebooks"] == 4
    assert cfg["model"]["max_position_embeddings"] == 512
    physical_documents = (
        cfg["train"]["tracks_per_microbatch"] * cfg["train"]["segments_per_track"]
    )
    assert physical_documents == 80
    assert physical_documents * cfg["trainer"]["accumulate_grad_batches"] == 160
    assert cfg["train"]["tracks_per_microbatch"] == 40
    assert cfg["train"]["deterministic"]
    assert cfg["data"]["segment_duration"] == 2.0
    assert cfg["tokenizer"]["sample_rate"] * cfg["data"]["segment_duration"] == 48_000
    assert cfg["train"]["id_digit_weight"] == 16.0
    assert cfg["resolved_query_profile"] == {
        "selected_codebooks": 4,
        "id_digit_weight": 16.0,
    }
    assert cfg["data"]["max_training_tracks"] == 25_000
    assert cfg["train"]["max_steps"] == 225_000
    assert cfg["train"]["warmup_steps"] == 500
    assert cfg["train"]["evaluation_interval"] == 2_500
    assert cfg["train"]["checkpoint_interval"] == 2_500
    assert cfg["resolved_training_profile"]["decoder"]["name"] == "small"
    assert (
        cfg["resolved_training_profile"]["variant"]
        == "tc16-two-second-four-codebook-logit-distillation"
    )
    assert (
        cfg["train"]["distillation"]["protocol"]
        == "tc16_two_second_four_codebook_logit_distillation_v1"
    )
    assert cfg["train"]["distillation"]["maximum_weight"] == 0.1
    assert cfg["train"]["schedule"]["name"] == "noise-rir"
    assert (
        cfg["train"]["schedule"]["loss_protocol"]
        == "tc16_two_second_four_codebook_logit_distillation_v1"
    )
    assert cfg["train"]["schedule"]["clean_until_step"] == 10_000
    assert cfg["train"]["schedule"]["degradation_ramp_until_step"] == 30_000
    assert cfg["train"]["schedule"]["combined_ramp_until_step"] == 60_000
    assert cfg["evaluation"]["monitor_tracks"] == 100
    assert cfg["evaluation"]["noise_snr_db"] == [0, 5, 10, 20, 30]
    assert cfg["data"]["background_noise"]["training_root"].endswith(
        "/bg_noise/train"
    )
    assert cfg["data"]["background_noise"]["validation_root"].endswith(
        "/bg_noise/test"
    )
    assert cfg["data"]["room_ir"]["training_root"].endswith("/room_ir/train")
    assert cfg["data"]["room_ir"]["validation_root"].endswith("/room_ir/test")
    assert cfg["data"]["room_ir"]["past_context_duration"] == 2.0
    assert cfg["data"]["crop_retries"] == 4
    assert cfg["data"]["replacement_retries"] == 32
    assert not any("token_root" in key for key in cfg["data"])
    assert not any(key.startswith("id_loss_") for key in cfg["train"])


def test_tc16_startup_query_invariants_are_enforced():
    cfg = resolve_training_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
        )
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
        selected_codebooks=4,
        codebook_size=1024,
        serialization="time_major_codebook_interleaved",
        preprocessing_version=1,
    )
    query_spec = validate_tc16_query_configuration(
        cfg, tokenizer_spec, AudioLMVocabulary(num_codebooks=4)
    )
    batch_spec = validate_tc16_batch_configuration(cfg, query_spec)
    assert query_spec["waveform_samples"] == 48_000
    assert query_spec["frames"] == 50
    assert query_spec["audio_targets"] == 200
    assert query_spec["digit_targets"] == 5
    assert query_spec["boundary_targets"] == 2
    assert query_spec["document_tokens"] == 208
    prompt = prompt_from_audio_tokens(
        torch.zeros(200, dtype=torch.long), AudioLMVocabulary(num_codebooks=4)
    )
    assert len(prompt) == 202
    assert len(prompt) + 6 == 208
    assert len(prompt) + 6 <= cfg["model"]["max_position_embeddings"]
    assert batch_spec["documents_per_microbatch"] == 80
    assert batch_spec["audio_targets_per_microbatch"] == 16_000
    assert batch_spec["causal_tokens_per_microbatch"] == 16_640
    assert batch_spec["waveform_seconds_per_microbatch"] == 160.0
    assert batch_spec["tracks_per_optimizer_step"] == 80
    assert batch_spec["documents_per_optimizer_step"] == 160
    assert batch_spec["audio_targets_per_optimizer_step"] == 32_000
    assert batch_spec["waveform_seconds_per_optimizer_step"] == 320.0

    wrong_batch = deepcopy(cfg)
    wrong_batch["train"]["tracks_per_microbatch"] = 10
    with pytest.raises(ValueError, match="40 tracks per microbatch"):
        validate_tc16_batch_configuration(wrong_batch, query_spec)

    wrong_duration = deepcopy(cfg)
    wrong_duration["data"]["segment_duration"] = 5.0
    with pytest.raises(ValueError, match="segment_duration=2.0"):
        validate_tc16_query_configuration(
            wrong_duration, tokenizer_spec, AudioLMVocabulary(num_codebooks=4)
        )

    wrong_weight = deepcopy(cfg)
    wrong_weight["train"]["id_digit_weight"] = 40.0
    with pytest.raises(ValueError, match="id_digit_weight=16"):
        validate_tc16_query_configuration(
            wrong_weight, tokenizer_spec, AudioLMVocabulary(num_codebooks=4)
        )


def test_tc16_rejects_three_codebook_query_profile():
    with pytest.raises(ValueError, match="four MuQ codebooks"):
        resolve_training_config(
            yaml.safe_load(
                (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
            ),
            selected_codebooks=3,
        )


def test_random_code_mapping_is_complete_unique_and_seeded():
    first = assign_codes(100_000, seed=1337)
    second = assign_codes(100_000, seed=1337)
    assert first == second
    assert len(set(first)) == 100_000
    assert all(len(code) == 5 for code in first)


def test_fresh_training_cohort_is_seeded_and_preserves_catalogue_codes(tmp_path):
    catalogue = tmp_path / "catalogue.jsonl"
    rows = [
        {
            "path": f"{index}.mp3",
            "track_id": f"track-{index:03d}",
            "code": f"{index:05d}",
            "duration": 30.0,
        }
        for index in range(10_000)
    ]
    catalogue.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "cohort.json"
    cfg = {
        "data": {
            "catalogue": str(catalogue),
            "max_training_tracks": 10_000,
            "training_tracks_manifest": str(output),
        },
        "train": {"seed": 1337},
    }
    first = prepare_training_cohort(cfg)
    second = prepare_training_cohort(cfg)
    assert first == second
    assert len(first["track_ids"]) == 10_000
    assert len(set(first["track_ids"])) == 10_000
    assert first["protocol"] == "fresh_seeded_catalogue_cohort_v1"
