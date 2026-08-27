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
    validate_tc18_batch_configuration,
    validate_tc18_query_configuration,
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
    assert cfg["tokenizer"]["selected_codebooks"] == 8
    assert cfg["model"]["max_position_embeddings"] == 512
    physical_documents = (
        cfg["train"]["tracks_per_microbatch"] * cfg["train"]["segments_per_track"]
    )
    assert physical_documents == 32
    assert physical_documents * cfg["trainer"]["accumulate_grad_batches"] == 32
    assert cfg["train"]["tracks_per_microbatch"] == 16
    assert cfg["train"]["deterministic"]
    assert cfg["data"]["segment_duration"] == 2.0
    assert cfg["tokenizer"]["sample_rate"] * cfg["data"]["segment_duration"] == 48_000
    assert cfg["train"]["id_digit_weight"] == 32.0
    assert cfg["resolved_query_profile"] == {
        "selected_codebooks": 8,
        "id_digit_weight": 32.0,
    }
    assert cfg["data"]["max_training_tracks"] == 100_000
    assert cfg["train"]["max_steps"] == 1_125_000
    assert cfg["train"]["warmup_steps"] == 500
    assert cfg["train"]["evaluation_interval"] == 5_000
    assert cfg["train"]["checkpoint_interval"] == 10_000
    assert cfg["resolved_training_profile"]["decoder"]["name"] == "medium"
    assert (
        cfg["resolved_training_profile"]["variant"]
        == "scale-100k-medium-4gpu-eight-codebook-throughput"
    )
    assert (
        cfg["train"]["distillation"]["protocol"]
        == "tc18_two_second_eight_codebook_logit_distillation_v1"
    )
    assert cfg["train"]["distillation"]["maximum_weight"] == 0.1
    assert cfg["train"]["schedule"]["name"] == "noise-rir"
    assert (
        cfg["train"]["schedule"]["loss_protocol"]
        == "tc18_two_second_eight_codebook_logit_distillation_v1"
    )
    assert cfg["train"]["schedule"]["clean_until_step"] == 50_000
    assert cfg["train"]["schedule"]["degradation_ramp_until_step"] == 150_000
    assert cfg["train"]["schedule"]["combined_ramp_until_step"] == 300_000
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


def test_100k_override_selects_existing_size_specific_manifest():
    cfg = resolve_training_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
        ),
        database_size=100_000,
    )
    assert cfg["data"]["database_size"] == 100_000
    assert cfg["data"]["max_training_tracks"] == 100_000
    assert cfg["data"]["training_tracks_manifest"] == (
        "data/training_tracks_100k.json"
    )
    assert cfg["train"]["max_steps"] == 1_125_000
    assert cfg["train"]["evaluation_interval"] == 5_000
    assert cfg["train"]["checkpoint_interval"] == 10_000


def test_four_gpu_batch_uses_probe_selected_scale_layout():
    cfg = resolve_training_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
        ),
        database_size=100_000,
        devices=4,
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
        selected_codebooks=8,
        codebook_size=1024,
        serialization="time_major_codebook_interleaved",
        preprocessing_version=1,
    )
    query = validate_tc18_query_configuration(
        cfg, tokenizer_spec, AudioLMVocabulary(num_codebooks=8)
    )
    batch = validate_tc18_batch_configuration(cfg, query)
    assert batch["tracks_per_microbatch"] == 16
    assert batch["gradient_accumulation_steps"] == 1
    assert batch["tracks_per_optimizer_step"] == 64
    assert batch["documents_per_optimizer_step"] == 128
    assert batch["audio_targets_per_optimizer_step"] == 51_200
    assert batch["waveform_seconds_per_optimizer_step"] == 256.0


def test_medium_four_gpu_batch_is_fixed_not_cli_tunable():
    cfg = resolve_training_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
        ),
        database_size=100_000,
        decoder="medium",
        devices=4,
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
        selected_codebooks=8,
        codebook_size=1024,
        serialization="time_major_codebook_interleaved",
        preprocessing_version=1,
    )
    query = validate_tc18_query_configuration(
        cfg, tokenizer_spec, AudioLMVocabulary(num_codebooks=8)
    )
    batch = validate_tc18_batch_configuration(cfg, query)
    assert batch["tracks_per_microbatch"] == 16
    assert batch["documents_per_microbatch"] == 32
    assert batch["gradient_accumulation_steps"] == 1
    assert batch["tracks_per_optimizer_step"] == 64
    assert batch["documents_per_optimizer_step"] == 128
    assert batch["audio_targets_per_optimizer_step"] == 51_200
    assert batch["waveform_seconds_per_optimizer_step"] == 256.0


def test_tc18_startup_query_invariants_are_enforced():
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
        selected_codebooks=8,
        codebook_size=1024,
        serialization="time_major_codebook_interleaved",
        preprocessing_version=1,
    )
    vocabulary = AudioLMVocabulary(num_codebooks=8)
    assert vocabulary.audio_size == 8_192
    assert vocabulary.size == 8_205
    query_spec = validate_tc18_query_configuration(cfg, tokenizer_spec, vocabulary)
    batch_spec = validate_tc18_batch_configuration(cfg, query_spec)
    assert query_spec["waveform_samples"] == 48_000
    assert query_spec["frames"] == 50
    assert query_spec["audio_targets"] == 400
    assert query_spec["digit_targets"] == 5
    assert query_spec["boundary_targets"] == 2
    assert query_spec["document_tokens"] == 408
    prompt = prompt_from_audio_tokens(
        torch.zeros(400, dtype=torch.long), vocabulary
    )
    assert len(prompt) == 402
    assert len(prompt) + 6 == 408
    assert len(prompt) + 6 <= cfg["model"]["max_position_embeddings"]
    assert batch_spec["documents_per_microbatch"] == 32
    assert batch_spec["audio_targets_per_microbatch"] == 12_800
    assert batch_spec["causal_tokens_per_microbatch"] == 13_056
    assert batch_spec["waveform_seconds_per_microbatch"] == 64.0
    assert batch_spec["tracks_per_optimizer_step"] == 64
    assert batch_spec["documents_per_optimizer_step"] == 128
    assert batch_spec["audio_targets_per_optimizer_step"] == 51_200
    assert batch_spec["waveform_seconds_per_optimizer_step"] == 256.0

    wrong_batch = deepcopy(cfg)
    wrong_batch["train"]["tracks_per_microbatch"] = 10
    with pytest.raises(ValueError, match="16 tracks per microbatch"):
        validate_tc18_batch_configuration(wrong_batch, query_spec)

    wrong_duration = deepcopy(cfg)
    wrong_duration["data"]["segment_duration"] = 5.0
    with pytest.raises(ValueError, match="segment_duration=2.0"):
        validate_tc18_query_configuration(
            wrong_duration, tokenizer_spec, AudioLMVocabulary(num_codebooks=8)
        )

    wrong_weight = deepcopy(cfg)
    wrong_weight["train"]["id_digit_weight"] = 24.0
    with pytest.raises(ValueError, match="id_digit_weight=32"):
        validate_tc18_query_configuration(
            wrong_weight, tokenizer_spec, AudioLMVocabulary(num_codebooks=8)
        )


def test_ablation_rejects_other_codebook_query_profile():
    with pytest.raises(ValueError, match="all eight MuQ codebooks"):
        resolve_training_config(
            yaml.safe_load(
                (Path(__file__).parents[1] / "configs" / "fma_large.yaml").read_text()
            ),
            selected_codebooks=6,
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
