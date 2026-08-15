import hashlib
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from para_audio_id.audio_lm.tokenizer import TokenizerSpec
from para_audio_id.audio_lm.profiles import catalogue_fingerprint
from para_audio_id.audio_lm.training import (
    AUGMENTATION_METRICS,
    TRAIN_LOG_LEVELS,
    TRAIN_METRICS,
    tc6_probe_wandb_keys,
    tc6_training_wandb_keys,
    train,
)
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary
from para_audio_id.catalogue import load_catalogue


def test_one_step_training_smoke(tmp_path, monkeypatch):
    spec = TokenizerSpec(
        architecture="muq_mel_rvq",
        model_name="dummy",
        revision="resolved",
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
    vocabulary = AudioLMVocabulary()
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(6 * 8_000, dtype=np.float32) / 8_000
    )
    catalogue = tmp_path / "catalogue.jsonl"
    rows = []
    track_ids = []
    for index in range(10):
        path = f"{index}.wav"
        track_id = f"track-{index}"
        sf.write(audio_root / path, waveform, 8_000)
        rows.append(
            {
                "path": path,
                "track_id": track_id,
                "code": f"{index:05d}",
                "duration": 6.0,
            }
        )
        track_ids.append(track_id)
    catalogue.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    manifest = tmp_path / "training_tracks.json"
    records = load_catalogue(catalogue)
    manifest.write_text(
        json.dumps(
            {
                "protocol": "fresh_seeded_catalogue_cohort_v1",
                "seed": 7,
                "count": 10,
                "catalogue_fingerprint": catalogue_fingerprint(records),
                "track_ids": track_ids,
                "code_mapping_fingerprint": hashlib.sha256(
                    "\n".join(
                        f"{record.track_id}:{record.code}" for record in records
                    ).encode()
                ).hexdigest(),
            }
        )
    )
    noise_train = tmp_path / "noise_train"
    noise_validation = tmp_path / "noise_validation"
    noise_train.mkdir()
    noise_validation.mkdir()
    sf.write(noise_train / "train.wav", waveform[: 5 * 8_000], 8_000)
    sf.write(noise_validation / "validation.wav", waveform[: 5 * 8_000], 8_000)
    rir_train = tmp_path / "rir_train" / "OpenAIR" / "train-room"
    rir_validation = tmp_path / "rir_validation" / "OpenAIR" / "test-room"
    rir_train.mkdir(parents=True)
    rir_validation.mkdir(parents=True)
    impulse = np.zeros(800, dtype=np.float32)
    impulse[0] = 1.0
    sf.write(rir_train / "ir.wav", impulse, 8_000)
    sf.write(rir_validation / "ir.wav", impulse[::-1], 8_000)

    class FakeOnlineTokenizer:
        def __init__(self, *args, **kwargs):
            self.spec = spec
            self.vocabulary = vocabulary
            self.device = torch.device(kwargs.get("device", "cpu"))

        def tokenize(self, waveforms):
            frame = torch.tensor([3, 1027], device=waveforms.device)
            return frame.repeat(waveforms.shape[0], 125)

    monkeypatch.setattr(
        "para_audio_id.audio_lm.training.MuQRVQTokenizer",
        FakeOnlineTokenizer,
    )
    auxiliary_profile = {
        "protocol": "tc13_task_anchored_simsiam_v1",
        "summary_weight": 0.1,
        "maximum_predictive_weight": 0.1,
        "predictive_weight_schedule": {
            "zero_until_step": 10_000,
            "ramp_until_step": 30_000,
            "maximum_weight": 0.1,
        },
        "summary_digits": 5,
        "digit_classes": 10,
        "projector": [32, 64, 16],
        "predictor": [16, 64, 16],
        "normalization": "layer_norm",
        "clean_target_detached_after_projector": True,
        "ema_target_encoder": False,
        "contrastive_negatives": False,
    }
    cfg = {
        "architecture": "audio_lm_v1",
        "tokenizer": {
            "model_name": "dummy",
            "revision": "resolved",
            "selected_codebooks": 2,
            "sample_rate": 24_000,
            "device": "cpu",
        },
        "model": {
            "architecture": "gpt2",
            "num_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "max_position_embeddings": 512,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "tie_word_embeddings": True,
        },
        "data": {
            "audio_root": str(audio_root),
            "catalogue": str(catalogue),
            "training_tracks_manifest": str(manifest),
            "database_size": 10,
            "max_training_tracks": 10,
            "segment_duration": 5.0,
            "crop_retries": 4,
            "replacement_retries": 32,
            "background_noise": {
                "training_root": str(noise_train),
                "validation_root": str(noise_validation),
            },
            "room_ir": {
                "training_root": str(rir_train.parent.parent),
                "validation_root": str(rir_validation.parent.parent),
                "past_context_duration": 2.0,
            },
        },
        "train": {
            "seed": 7,
            "deterministic": True,
            "deterministic_warn_only": True,
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "run_id": "smoke",
            "max_steps": 2,
            "tracks_per_microbatch": 4,
            "segments_per_track": 2,
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "evaluation_interval": 1,
            "checkpoint_interval": 1,
            "id_digit_weight": 20.0,
            "gradient_clip_norm": 1.0,
            "schedule": {
                "name": "noise-rir",
                "protocol": "tc13_five_second_task_anchored_simsiam_v1",
                "loss_protocol": "tc13_task_anchored_simsiam_v1",
                "curriculum": "tc12_noise_rir_curriculum_v1",
                "clean_until_step": 10_000,
                "degradation_ramp_until_step": 30_000,
                "combined_ramp_until_step": 60_000,
                "predictive_weight": 0.1,
                "snr_bin_probabilities": [0.4, 0.3, 0.2, 0.1],
                "exact_zero_fraction_in_first_bin": 0.25,
            },
            "auxiliary": auxiliary_profile,
            "wandb": {"enabled": False},
        },
        "resolved_training_profile": {
            "version": 4,
            "variant": "tc13-task-anchored",
            "database_size": 10,
            "training_tracks_manifest": str(manifest),
            "decoder": {
                "name": "small",
                "num_layers": 1,
                "hidden_size": 32,
                "num_attention_heads": 4,
            },
            "schedule": {
                "name": "noise-rir",
                "protocol": "tc13_five_second_task_anchored_simsiam_v1",
                "loss_protocol": "tc13_task_anchored_simsiam_v1",
                "max_steps": 2,
                "curriculum": "tc12_noise_rir_curriculum_v1",
                "clean_until_step": 10_000,
                "degradation_ramp_until_step": 30_000,
                "combined_ramp_until_step": 60_000,
                "predictive_weight": 0.1,
                "snr_bin_probabilities": [0.4, 0.3, 0.2, 0.1],
                "exact_zero_fraction_in_first_bin": 0.25,
            },
            "auxiliary": auxiliary_profile,
        },
        "evaluation": {
            "online_monitor_enabled": False,
            "monitor_tracks": 2,
            "probe_batch_size": 2,
            "generation_batch_size": 2,
            "beam_width": 10,
            "noise_snr_db": [0, 5, 10, 20, 30],
        },
        "dataloader": {
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "strategy": "auto",
            "precision": "32-true",
            "accumulate_grad_batches": 20,
            "log_every_n_steps": 1,
        },
    }
    train(cfg)
    uninterrupted = torch.load(
        tmp_path / "checkpoints" / "smoke" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    interrupted = tmp_path / "checkpoints" / "smoke" / "step-1.ckpt"
    assert interrupted.exists()
    train(cfg, checkpoint=interrupted)
    last = tmp_path / "checkpoints" / "smoke" / "last.ckpt"
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    for key, value in uninterrupted["state_dict"].items():
        assert torch.equal(value, checkpoint["state_dict"][key])
    assert checkpoint["global_step"] == 2
    assert (
        checkpoint["training_protocol"]
        == "tc13_five_second_task_anchored_simsiam_v1"
    )
    assert checkpoint["loss_protocol"] == "tc13_task_anchored_simsiam_v1"
    assert checkpoint["auxiliary_protocol"] == "tc13_task_anchored_simsiam_v1"
    assert checkpoint["auxiliary_profile"] == auxiliary_profile
    assert any(key.startswith("task_auxiliary.") for key in checkpoint["state_dict"])
    assert checkpoint["monitor_protocol"] == "compact_beam_monitor_v2"
    assert checkpoint["crop_policy"] == "tc13_five_second_online_random_crop_24k_v1"
    assert checkpoint["room_ir_manifest"]["training_files"] == 1
    assert len(checkpoint["monitor_recipes"]) == 6
    assert {
        row["view_type"] for row in checkpoint["monitor_recipes"]
    } == {"canonical", "shifted", "heldout"}
    assert all(
        row["crop_duration"] == 5.0
        for row in checkpoint["monitor_recipes"]
    )
    assert checkpoint["query_spec"] == {
        "segment_duration_seconds": 5.0,
        "sample_rate": 24_000,
        "waveform_samples": 120_000,
        "frame_rate": 25.0,
        "selected_codebooks": 2,
        "frames": 125,
        "audio_targets": 250,
        "digit_targets": 5,
        "boundary_targets": 2,
        "document_tokens": 258,
        "id_digit_weight": 20.0,
        "max_position_embeddings": 512,
    }
    assert checkpoint["batch_spec"] == {
        "tracks_per_microbatch": 4,
        "documents_per_track": 2,
        "documents_per_microbatch": 8,
        "audio_targets_per_microbatch": 2_000,
        "causal_tokens_per_microbatch": 2_064,
        "waveform_seconds_per_microbatch": 40.0,
        "gradient_accumulation_steps": 20,
        "tracks_per_optimizer_step": 80,
        "documents_per_optimizer_step": 160,
        "audio_targets_per_optimizer_step": 40_000,
        "waveform_seconds_per_optimizer_step": 800.0,
    }
    auxiliary_rows = [
        json.loads(line)
        for line in (
            tmp_path / "logs" / "smoke" / "auxiliary_metrics.jsonl"
        ).read_text().splitlines()
    ]
    assert [row["global_step"] for row in auxiliary_rows] == [0, 1]
    assert all(row["summary_loss"] is not None for row in auxiliary_rows)
    assert all(row["predictive_loss"] == 0.0 for row in auxiliary_rows)
    assert all(row["paired_prediction_cosine"] is None for row in auxiliary_rows)

    invalid = tmp_path / "invalid.ckpt"
    checkpoint["resolved_training_profile"]["decoder"]["name"] = "medium"
    torch.save(checkpoint, invalid)
    with pytest.raises(ValueError, match="different training profile"):
        train(cfg, checkpoint=invalid)


def test_tc13_wandb_keys_are_compact_and_task_anchored():
    assert TRAIN_METRICS == {
        "clean_audio_loss",
        "digit_loss",
        "teacher_forced_exact_accuracy",
        "summary_loss",
        "summary_exact_accuracy",
        "predictive_loss",
        "paired_prediction_cosine",
        "prediction_cosine_margin",
    }
    assert AUGMENTATION_METRICS == {
        "scheduled_probability",
        "realized_noisy_fraction",
        "mean_snr_db",
        "online_tokenization_seconds",
    }
    assert TRAIN_LOG_LEVELS == {"on_step": True, "on_epoch": True}
    assert tc6_training_wandb_keys() == {
        "train/clean_audio_loss_step",
        "train/clean_audio_loss_epoch",
        "train/digit_loss_step",
        "train/digit_loss_epoch",
        "train/summary_loss_step",
        "train/summary_loss_epoch",
        "train/summary_exact_accuracy_step",
        "train/summary_exact_accuracy_epoch",
        "train/predictive_loss_step",
        "train/predictive_loss_epoch",
        "train/paired_prediction_cosine_step",
        "train/paired_prediction_cosine_epoch",
        "train/prediction_cosine_margin_step",
        "train/prediction_cosine_margin_epoch",
        "train/teacher_forced_exact_accuracy_step",
        "train/teacher_forced_exact_accuracy_epoch",
        "train/loss_step",
        "train/loss_epoch",
    }
    assert tc6_probe_wandb_keys([0.0, 5.0, 10.0, 20.0, 30.0]) == {
        "probe/clean/canonical/beam_top1",
        "probe/clean/shifted/beam_top1",
        "probe/clean/heldout/beam_top1",
        "probe/noise/snr_0/beam_top1",
        "probe/noise/snr_5/beam_top1",
        "probe/noise/snr_10/beam_top1",
        "probe/noise/snr_20/beam_top1",
        "probe/noise/snr_30/beam_top1",
        "probe/noise/aggregate/beam_top1",
        "probe/noise/online_tokenization_seconds",
        "probe/clean/shifted/teacher_forced_exact_accuracy",
        "probe/rir/aggregate/beam_top1",
        "probe/noise_rir/aggregate/beam_top1",
    }
