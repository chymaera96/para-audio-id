from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from para_audio_id.audio_lm.tokenizer import TokenizerSpec
from para_audio_id.audio_lm.profiles import resolve_training_config
from para_audio_id.audio_lm.profiles import resolve_capacity_config
from para_audio_id.audio_lm.training import (
    AudioLMDataModule,
    validate_tc9_query_configuration,
    validate_tc9_batch_configuration,
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
    assert cfg["tokenizer"]["selected_codebooks"] == 2
    assert cfg["model"]["max_position_embeddings"] == 512
    physical_documents = (
        cfg["train"]["tracks_per_microbatch"] * cfg["train"]["segments_per_track"]
    )
    assert physical_documents == 20
    assert physical_documents * cfg["trainer"]["accumulate_grad_batches"] == 160
    assert cfg["train"]["tracks_per_microbatch"] == 10
    assert cfg["train"]["deterministic"]
    assert cfg["data"]["segment_duration"] == 2.0
    assert cfg["tokenizer"]["sample_rate"] * cfg["data"]["segment_duration"] == 48_000
    assert cfg["train"]["id_digit_weight"] == 8.0
    assert cfg["data"]["max_training_tracks"] == 25_000
    assert cfg["train"]["max_steps"] == 175_000
    assert cfg["train"]["warmup_steps"] == 200
    assert cfg["train"]["evaluation_interval"] == 2_500
    assert cfg["train"]["checkpoint_interval"] == 500
    assert cfg["resolved_training_profile"]["decoder"]["name"] == "small"
    assert cfg["train"]["schedule"]["name"] == "noise"
    assert (
        cfg["train"]["schedule"]["loss_protocol"]
        == "tc5_family_weighted_consistency_v2"
    )
    assert cfg["train"]["schedule"]["clean_until_step"] == 50_000
    assert cfg["train"]["schedule"]["noise_ramp_until_step"] == 62_500
    assert "noise_steady_until_step" not in cfg["train"]["schedule"]
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


def test_tc9_startup_query_invariants_are_enforced():
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
        selected_codebooks=2,
        codebook_size=1024,
        serialization="time_major_codebook_interleaved",
        preprocessing_version=1,
    )
    query_spec = validate_tc9_query_configuration(
        cfg, tokenizer_spec, AudioLMVocabulary()
    )
    batch_spec = validate_tc9_batch_configuration(cfg, query_spec)
    assert query_spec["waveform_samples"] == 48_000
    assert query_spec["frames"] == 50
    assert query_spec["audio_targets"] == 100
    assert query_spec["digit_targets"] == 5
    assert query_spec["boundary_targets"] == 2
    assert query_spec["document_tokens"] == 108
    assert batch_spec["documents_per_microbatch"] == 20
    assert batch_spec["audio_targets_per_microbatch"] == 2_000
    assert batch_spec["causal_tokens_per_microbatch"] == 2_160
    assert batch_spec["waveform_seconds_per_microbatch"] == 40.0
    assert batch_spec["tracks_per_optimizer_step"] == 80
    assert batch_spec["documents_per_optimizer_step"] == 160
    assert batch_spec["audio_targets_per_optimizer_step"] == 16_000
    assert batch_spec["waveform_seconds_per_optimizer_step"] == 320.0


def test_capacity_distributed_batch_preserves_global_optimizer_batch():
    cfg = resolve_capacity_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "capacity.yaml").read_text()
        ),
        devices=2,
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
    query_spec = validate_tc9_query_configuration(
        cfg, tokenizer_spec, AudioLMVocabulary()
    )
    batch_spec = validate_tc9_batch_configuration(cfg, query_spec)
    assert batch_spec["tracks_per_microbatch"] == 40
    assert batch_spec["gradient_accumulation_steps"] == 1
    assert batch_spec["world_size"] == 2
    assert batch_spec["tracks_per_device_optimizer_step"] == 40
    assert batch_spec["tracks_per_optimizer_step"] == 80
    assert batch_spec["documents_per_optimizer_step"] == 160

    wrong_batch = deepcopy(cfg)
    wrong_batch["train"]["tracks_per_microbatch"] = 4
    with pytest.raises(ValueError, match="40 tracks per microbatch"):
        validate_tc9_batch_configuration(wrong_batch, query_spec)

    wrong_duration = deepcopy(cfg)
    wrong_duration["data"]["segment_duration"] = 5.0
    with pytest.raises(ValueError, match="segment_duration=2.0"):
        validate_tc9_query_configuration(
            wrong_duration, tokenizer_spec, AudioLMVocabulary()
        )

    wrong_weight = deepcopy(cfg)
    wrong_weight["train"]["id_digit_weight"] = 20.0
    with pytest.raises(ValueError, match="id_digit_weight=8"):
        validate_tc9_query_configuration(
            wrong_weight, tokenizer_spec, AudioLMVocabulary()
        )


def test_capacity_batch_is_fixed_at_40_tracks_with_two_accumulation_steps():
    cfg = resolve_capacity_config(
        yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "capacity.yaml").read_text()
        )
    )
    query_spec = {
        "audio_targets": 100,
        "document_tokens": 108,
        "segment_duration_seconds": 2.0,
    }
    batch = validate_tc9_batch_configuration(cfg, query_spec)
    assert batch["tracks_per_microbatch"] == 40
    assert batch["documents_per_microbatch"] == 80
    assert batch["audio_targets_per_microbatch"] == 8_000
    assert batch["causal_tokens_per_microbatch"] == 8_640
    assert batch["gradient_accumulation_steps"] == 2
    assert batch["tracks_per_optimizer_step"] == 80
    assert batch["documents_per_optimizer_step"] == 160
    assert batch["audio_targets_per_optimizer_step"] == 16_000


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
    original_bytes = output.read_bytes()
    second = prepare_training_cohort(cfg)
    assert first == second
    assert output.read_bytes() == original_bytes
    assert len(first["track_ids"]) == 10_000
    assert len(set(first["track_ids"])) == 10_000
    assert first["protocol"] == "fresh_seeded_catalogue_cohort_v1"

    broken = json.loads(output.read_text())
    broken["count"] = 9_999
    output.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="10,000 unique IDs|10000 unique IDs"):
        prepare_training_cohort(cfg)


def test_capacity_datamodule_never_constructs_degradation_assets(tmp_path, monkeypatch):
    catalogue = tmp_path / "catalogue.jsonl"
    rows = [
        {
            "path": f"{index}.mp3",
            "track_id": f"track-{index:05d}",
            "code": f"{index:05d}",
            "duration": 30.0,
        }
        for index in range(10_000)
    ]
    catalogue.write_text("\n".join(json.dumps(row) for row in rows))
    manifest = tmp_path / "training_tracks_10k.json"
    cfg = resolve_capacity_config(
        {
            "data": {
                "database_size": 10_000,
                "catalogue": str(catalogue),
                "audio_root": str(tmp_path),
                "segment_duration": 2.0,
            },
            "model": {},
            "train": {
                "seed": 1337,
                "target_exposures": 560,
                "tracks_per_microbatch": 40,
            },
            "trainer": {"accumulate_grad_batches": 2},
            "evaluation": {"monitor_tracks": 100},
        }
    )
    cfg["data"]["training_tracks_manifest"] = str(manifest)
    prepare_training_cohort(cfg, manifest)

    def forbidden(*args, **kwargs):
        raise AssertionError("capacity setup accessed degradation assets")

    monkeypatch.setattr(
        "para_audio_id.audio_lm.training.BackgroundNoiseAssets", forbidden
    )
    monkeypatch.setattr(
        "para_audio_id.audio_lm.training.RoomImpulseResponseAssets", forbidden
    )
    spec = TokenizerSpec(
        architecture="muq_mel_rvq",
        model_name="test",
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
    datamodule = AudioLMDataModule(cfg, spec, AudioLMVocabulary())
    datamodule.setup("fit")
    assert datamodule.noise_assets is None
    assert datamodule.rir_assets is None


def test_legacy_track_id_manifest_is_reused_without_rewrite(tmp_path):
    catalogue = tmp_path / "catalogue.jsonl"
    rows = [
        {
            "path": f"{index}.mp3",
            "track_id": f"track-{index:05d}",
            "code": f"{index:05d}",
            "duration": 30.0,
        }
        for index in range(10_000)
    ]
    catalogue.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "training_tracks_10k.json"
    track_ids = [row["track_id"] for row in rows]
    output.write_text(json.dumps(track_ids, indent=2) + "\n")
    original = output.read_bytes()
    result = prepare_training_cohort(
        {
            "data": {
                "catalogue": str(catalogue),
                "max_training_tracks": 10_000,
                "training_tracks_manifest": str(output),
            },
            "train": {"seed": 1337},
        }
    )
    assert result["protocol"] == "legacy_track_id_list_v1"
    assert result["reused_without_rewrite"]
    assert output.read_bytes() == original
