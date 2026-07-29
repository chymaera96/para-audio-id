import hashlib
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from para_audio_id.audio_lm.curriculum import AdaptiveCurriculum
from para_audio_id.audio_lm.noise import NoiseConsistencySchedule
from para_audio_id.audio_lm.token_store import TokenRecord, write_shard
from para_audio_id.audio_lm.tokenizer import TokenizerSpec
from para_audio_id.audio_lm.training import train
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


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
    track_ids = [f"track-{track}" for track in range(4)]
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(30 * 8_000, dtype=np.float32) / 8_000
    )
    for track in range(4):
        sf.write(audio_root / f"{track}.wav", waveform, 8_000)
    noise_train = tmp_path / "noise_train"
    noise_validation = tmp_path / "noise_validation"
    noise_train.mkdir()
    noise_validation.mkdir()
    sf.write(noise_train / "train.wav", waveform[: 5 * 8_000], 8_000)
    sf.write(
        noise_validation / "validation.wav",
        waveform[5 * 8_000 : 10 * 8_000],
        8_000,
    )
    code_fingerprint = hashlib.sha256(
        "\n".join(
            f"{track_id}:{index:05d}" for index, track_id in enumerate(track_ids)
        ).encode()
    ).hexdigest()

    def make_store(name, starts, role, view_type):
        token_root = tmp_path / name
        token_root.mkdir()
        (token_root / "tokenizer_spec.json").write_text(
            json.dumps(
                {
                    "tokenizer": spec.to_dict(),
                    "fingerprint": spec.fingerprint,
                    "vocabulary": vocabulary.to_dict(),
                    "corpus_role": role,
                    "view_policy_fingerprint": f"{role}-fingerprint",
                    "track_ids": track_ids,
                    "code_mapping_fingerprint": code_fingerprint,
                }
            )
        )
        records = []
        parts = []
        offset = 0
        for track in range(4):
            for start in starts:
                tokens = np.array([int(start) % 1024, 1024 + int(start) % 1024], dtype=np.uint16)
                parts.append(tokens)
                records.append(
                    TokenRecord(
                        document_index=len(records),
                        track_id=f"track-{track}",
                        code=f"{track:05d}",
                        source_path=f"{track}.wav",
                        segment_start=float(start),
                        segment_duration=5.0,
                        status="ok",
                        token_offset=offset,
                        token_count=2,
                        frames=1,
                        view_type=view_type,
                        corpus_role=role,
                    )
                )
                offset += 2
        write_shard(
            token_root,
            0,
            records=records,
            tokens=np.concatenate(parts),
            tokenizer_spec=spec.to_dict(),
            tokenizer_fingerprint=spec.fingerprint,
            corpus_role=role,
        )
        return token_root

    canonical_starts = [0, 5, 10, 15, 20, 25]
    shifted_starts = [1, 2, 3, 4]
    heldout_starts = [2.5, 7.5]
    canonical_root = make_store(
        "canonical", canonical_starts, "canonical_training", "canonical"
    )
    shifted_root = make_store(
        "shifted", shifted_starts, "shifted_training", "shifted"
    )
    heldout_root = make_store(
        "heldout", heldout_starts, "heldout_evaluation", "heldout"
    )
    manifest = tmp_path / "training_tracks.json"
    manifest.write_text(json.dumps(track_ids))

    class FakeOnlineTokenizer:
        def __init__(self, *args, **kwargs):
            self.spec = spec

        def tokenize(self, waveform):
            return torch.tensor(
                [[3, 1027]], device=waveform.device, dtype=torch.long
            ).repeat(waveform.shape[0], 1)

    monkeypatch.setattr(
        "para_audio_id.audio_lm.training.MuQRVQTokenizer",
        FakeOnlineTokenizer,
    )
    def forced_schedule(step, *, max_steps):
        return NoiseConsistencySchedule(
            1.0,
            0.1,
            (0.0, 0.0, 0.0, 1.0),
            "easy",
        )

    monkeypatch.setattr(
        "para_audio_id.audio_lm.training.noise_consistency_schedule",
        forced_schedule,
    )

    class AlwaysOpenCurriculum(AdaptiveCurriculum):
        def __post_init__(self):
            super().__post_init__()
            self.gate_open = True
            self.gate_open_step = self.clean_steps
            self.regression_baseline = 1.0

    monkeypatch.setattr(
        "para_audio_id.audio_lm.training.AdaptiveCurriculum",
        AlwaysOpenCurriculum,
    )
    cfg = {
        "architecture": "audio_lm_v1",
        "tokenizer": {
            "model_name": "dummy",
            "revision": "resolved",
            "selected_codebooks": 2,
            "sample_rate": 24_000,
        },
        "model": {
            "architecture": "gpt2",
            "num_layers": 1,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "max_position_embeddings": 32,
            "resid_pdrop": 0.0,
            "embd_pdrop": 0.0,
            "attn_pdrop": 0.0,
            "tie_word_embeddings": True,
        },
        "data": {
            "audio_root": str(audio_root),
            "canonical_token_root": str(canonical_root),
            "shifted_training_token_root": str(shifted_root),
            "heldout_evaluation_token_root": str(heldout_root),
            "training_tracks_manifest": str(manifest),
            "view_mode": "paired",
            "canonical_starts": canonical_starts,
            "shifted_training_starts": shifted_starts,
            "shifted_evaluation_starts": heldout_starts,
            "segments_per_track": 6,
            "max_training_tracks": 4,
            "segment_duration": 5.0,
            "background_noise": {
                "training_root": str(noise_train),
                "validation_root": str(noise_validation),
                "preflight_examples_per_view": 0,
            },
        },
        "train": {
            "seed": 7,
            "deterministic": True,
            "deterministic_warn_only": True,
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "run_id": "smoke",
            "max_steps": 8,
            "tracks_per_microbatch": 4,
            "segments_per_track": 2,
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "evaluation_interval": 2,
            "checkpoint_interval": 2,
            "id_digit_weight": 20.0,
            "gradient_clip_norm": 1.0,
            "curriculum": {
                "protocol": "noise_consistency_curriculum_v1",
                "loss_protocol": "tc5_family_weighted_consistency_v2",
                "gate_threshold": 0.5,
                "gate_max_extra_steps": 0,
                "regression_drop": 0.05,
                "recovery_probes": 2,
                "recovery_timeout_steps": 1,
            },
            "wandb": {"enabled": False},
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
            "accumulate_grad_batches": 1,
            "log_every_n_steps": 1,
        },
    }
    train(cfg)
    last = tmp_path / "checkpoints" / "smoke" / "last.ckpt"
    assert last.exists()
    checkpoint = torch.load(last, map_location="cpu", weights_only=False)
    assert checkpoint["global_step"] == 8
    assert checkpoint["training_protocol"] == "noise_consistency_curriculum_v1"
    assert checkpoint["loss_protocol"] == "tc5_family_weighted_consistency_v2"
    assert checkpoint["adaptive_curriculum_state"]["gate_open"]
    assert "snr_epoch_counts" in checkpoint
    train(cfg, checkpoint=last)

    invalid_checkpoint = tmp_path / "invalid-prefixed-tc6.ckpt"
    checkpoint.pop("loss_protocol")
    torch.save(checkpoint, invalid_checkpoint)
    with pytest.raises(
        ValueError,
        match="invalid pre-fix tc6 loss protocol",
    ):
        train(cfg, checkpoint=invalid_checkpoint)
