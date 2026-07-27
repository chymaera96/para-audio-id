import hashlib
import json

import numpy as np

from para_audio_id.audio_lm.token_store import TokenRecord, write_shard
from para_audio_id.audio_lm.tokenizer import TokenizerSpec
from para_audio_id.audio_lm.training import train
from para_audio_id.audio_lm.vocabulary import AudioLMVocabulary


def test_one_step_training_smoke(tmp_path):
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
                        source_path=f"{track}.mp3",
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
    cfg = {
        "architecture": "audio_lm_v1",
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
        },
        "train": {
            "seed": 7,
            "deterministic": True,
            "deterministic_warn_only": True,
            "log_dir": str(tmp_path / "logs"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "run_id": "smoke",
            "max_steps": 3,
            "tracks_per_microbatch": 4,
            "segments_per_track": 2,
            "learning_rate": 3e-4,
            "betas": [0.9, 0.95],
            "weight_decay": 0.0,
            "warmup_steps": 1,
            "evaluation_interval": 2,
            "id_digit_weight": 20.0,
            "gradient_clip_norm": 1.0,
            "wandb": {"enabled": False},
        },
        "evaluation": {
            "probe_tracks": 2,
            "probe_batch_size": 2,
            "generation_batch_size": 2,
            "beam_width": 10,
            "beam_every_n_steps": 100,
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
    cfg["train"]["max_steps"] = 4
    train(cfg, checkpoint=last)
