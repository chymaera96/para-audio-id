from __future__ import annotations

from functools import partial
import hashlib
import json
import math
from pathlib import Path
import random
import time

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..audio import load_audio
from ..config import save_config
from .checkpoint import ARCHITECTURE
from .curriculum import AdaptiveCurriculum, effective_cosine_multiplier
from .dataset import (
    CachedPositionDataset,
    NoiseEvaluationCollator,
    OnlinePairCollator,
    PairedAudioTokenDataset,
    PairedViewBatchSampler,
    collate_causal_documents,
)
from .generation import (
    batched_beam_generate,
    prompts_from_audio_tokens,
)
from .losses import causal_losses_by_view, noise_consistency_losses
from .model import AudioCausalLM
from .noise import (
    BackgroundNoiseAssets,
    SNR_BINS,
    deterministic_consistency_noise_parameters,
    mix_background_noise,
    noise_consistency_schedule,
    stable_uint64,
)
from .token_store import TokenStoreIndex
from .tokenizer import MuQRVQTokenizer, TokenizerSpec
from .tokenization import load_training_track_ids
from .vocabulary import AudioLMVocabulary

TRAINING_PROTOCOL = "noise_consistency_curriculum_v1"
LOSS_PROTOCOL = "tc5_family_weighted_consistency_v2"
MONITOR_PROTOCOL = "compact_beam_monitor_v2"
TRAIN_METRICS = {
    "clean_audio_loss",
    "digit_loss",
    "consistency_loss",
    "same_track_cosine",
    "different_track_cosine",
    "teacher_forced_exact_accuracy",
}
VALIDATION_METRICS = {
    "audio_loss",
    "id_loss",
    "teacher_forced_exact_accuracy",
}
AUGMENTATION_METRICS = {
    "scheduled_probability",
    "scheduled_consistency_weight",
    "realized_noisy_fraction",
    "mean_snr_db",
    "online_tokenization_seconds",
}


class ResumableDataLoader(DataLoader):
    """DataLoader that restores its delivered-batch position within a pass."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batches_yielded = 0
        self._resume_batches = 0

    def __iter__(self):
        iterator = super().__iter__()
        skip = self._resume_batches
        self._resume_batches = 0
        self._batches_yielded = 0
        for _ in range(skip):
            next(iterator)
            self._batches_yielded += 1
        for batch in iterator:
            self._batches_yielded += 1
            yield batch

    def state_dict(self) -> dict:
        return {"batches_yielded": self._batches_yielded}

    def load_state_dict(self, state_dict: dict) -> None:
        batches = int(state_dict["batches_yielded"])
        if not 0 <= batches <= len(self):
            raise ValueError(f"Invalid resumed DataLoader batch position {batches}")
        self._resume_batches = batches


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def replace_secondary_rows_with_noisy_tokens(
    batch: dict,
    tokens: torch.Tensor,
    pairs: list[int],
    *,
    id_token_id: int,
) -> dict:
    if tokens.shape[0] != len(pairs):
        raise ValueError("Noisy token rows do not match selected pairs")
    tensor_fields = (
        "input_ids",
        "attention_mask",
        "audio_target_mask",
        "id_target_mask",
        "boundary_target_mask",
        "document_index",
        "segment_start",
        "segment_duration",
    )
    result = {
        **batch,
        **{field: batch[field].clone() for field in tensor_fields},
        "source_path": list(batch["source_path"]),
        "track_id": list(batch["track_id"]),
        "code": list(batch["code"]),
        "view_type": list(batch["view_type"]),
    }
    result["is_noisy"] = torch.zeros(
        len(result["input_ids"]),
        device=result["input_ids"].device,
        dtype=torch.bool,
    )
    for token_row, pair in zip(tokens, pairs, strict=True):
        anchor_row = pair * 2
        secondary_row = anchor_row + 1
        if secondary_row >= len(result["input_ids"]):
            raise ValueError(f"Invalid pair index {pair}")
        if result["track_id"][anchor_row] != result["track_id"][secondary_row]:
            raise ValueError("Noisy pair rows have different track identities")
        if result["code"][anchor_row] != result["code"][secondary_row]:
            raise ValueError("Noisy pair rows have different identifier codes")
        positions = (result["input_ids"][anchor_row] == id_token_id).nonzero()
        if len(positions) != 1:
            raise RuntimeError("Clean anchor has an invalid ID boundary")
        id_column = int(positions[0])
        if len(token_row) != id_column - 1:
            raise RuntimeError("Online and cached audio-token lengths do not match")
        for field in tensor_fields:
            result[field][secondary_row] = result[field][anchor_row]
        for field in ("source_path", "track_id", "code"):
            result[field][secondary_row] = result[field][anchor_row]
        result["input_ids"][secondary_row, 1:id_column] = token_row
        result["view_type"][secondary_row] = "noisy"
        result["is_noisy"][secondary_row] = True
    return result


def prefetched_waveforms_for_step(
    batch: dict, *, global_step: int, batch_idx: int
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    if (
        int(batch["planned_optimizer_step"]) != int(global_step)
        or int(batch["planned_batch_idx"]) != int(batch_idx)
    ):
        return {}
    return {
        pair: (batch["anchor_waveforms"][row], batch["noise_waveforms"][row])
        for row, pair in enumerate(batch["loaded_pair_indices"])
    }


class AudioLMDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str | None = None) -> None:
        if hasattr(self, "dataset"):
            return
        data_cfg = self.cfg["data"]
        spec_payload = json.loads(
            (
                Path(data_cfg["canonical_token_root"]) / "tokenizer_spec.json"
            ).read_text()
        )
        self.tokenizer_spec = TokenizerSpec(**spec_payload["tokenizer"])
        if self.tokenizer_spec.fingerprint != spec_payload["fingerprint"]:
            raise ValueError("Token-store tokenizer specification fingerprint is invalid")
        self.vocabulary = AudioLMVocabulary.from_dict(spec_payload["vocabulary"])
        self.vocabulary.validate()
        canonical_store = TokenStoreIndex(
            data_cfg["canonical_token_root"],
            tokenizer_fingerprint=self.tokenizer_spec.fingerprint,
            corpus_role="canonical_training",
        )
        self.training_track_ids = load_training_track_ids(
            data_cfg["training_tracks_manifest"],
            expected_count=int(data_cfg["max_training_tracks"]),
        )
        view_mode = data_cfg["view_mode"]
        shifted_store = (
            TokenStoreIndex(
                data_cfg["shifted_training_token_root"],
                tokenizer_fingerprint=self.tokenizer_spec.fingerprint,
                corpus_role="shifted_training",
            )
            if view_mode == "paired"
            else None
        )
        self.dataset = PairedAudioTokenDataset(
            canonical_store,
            shifted_store,
            track_ids=self.training_track_ids,
            canonical_starts=[float(value) for value in data_cfg["canonical_starts"]],
            shifted_starts=[
                float(value) for value in data_cfg["shifted_training_starts"]
            ],
            view_mode=view_mode,
        )
        if not self.dataset.track_ids:
            raise RuntimeError("No tracks have a complete canonical token set")
        evaluation_store = TokenStoreIndex(
            data_cfg["heldout_evaluation_token_root"],
            tokenizer_fingerprint=self.tokenizer_spec.fingerprint,
            corpus_role="heldout_evaluation",
        )
        self.evaluation_dataset = CachedPositionDataset(
            canonical_store,
            evaluation_store,
            track_ids=self.training_track_ids,
            shifted_store=shifted_store,
        )
        noise_cfg = data_cfg["background_noise"]
        self.noise_assets = BackgroundNoiseAssets(
            noise_cfg["training_root"],
            noise_cfg["validation_root"],
            sample_rate=int(self.tokenizer_spec.sample_rate),
            samples=round(
                float(data_cfg["segment_duration"])
                * int(self.tokenizer_spec.sample_rate)
            ),
        )
        longest = max(record.token_count for record in self.dataset.records) + 8
        maximum = int(self.cfg["model"]["max_position_embeddings"])
        if longest > maximum:
            raise ValueError(
                f"Cached causal document length {longest} exceeds model context {maximum}"
            )
        mapping = {record.track_id: record.code for record in self.dataset.records}
        encoded_mapping = "\n".join(
            f"{track_id}:{mapping[track_id]}" for track_id in sorted(mapping)
        ).encode()
        self.code_mapping_fingerprint = hashlib.sha256(encoded_mapping).hexdigest()
        role_roots = [data_cfg["heldout_evaluation_token_root"]]
        if shifted_store is not None:
            role_roots.append(data_cfg["shifted_training_token_root"])
        for root in role_roots:
            payload = json.loads((Path(root) / "tokenizer_spec.json").read_text())
            if payload.get("track_ids") != self.training_track_ids:
                raise ValueError(f"View-token store {root} uses a different 10K cohort")
            if (
                payload.get("code_mapping_fingerprint")
                != self.code_mapping_fingerprint
            ):
                raise ValueError(f"View-token store {root} uses different track codes")
        corpus_payload = {
            "training_protocol": TRAINING_PROTOCOL,
            "view_mode": view_mode,
            "track_ids": self.training_track_ids,
            "canonical_starts": data_cfg["canonical_starts"],
            "shifted_training_starts": data_cfg["shifted_training_starts"],
            "shifted_evaluation_starts": data_cfg["shifted_evaluation_starts"],
            "canonical_tokenizer": self.tokenizer_spec.fingerprint,
            "shifted_policy": json.loads(
                (
                    Path(data_cfg["shifted_training_token_root"])
                    / "tokenizer_spec.json"
                ).read_text()
            ).get("view_policy_fingerprint")
            if shifted_store is not None
            else None,
            "background_noise": self.noise_assets.manifest(),
        }
        self.training_corpus_fingerprint = hashlib.sha256(
            json.dumps(corpus_payload, sort_keys=True).encode()
        ).hexdigest()
        probe_count = min(
            int(self.cfg["evaluation"]["monitor_tracks"]),
            len(self.dataset.track_ids),
        )
        generator = np.random.default_rng(int(self.cfg["train"]["seed"]) + 919)
        selected = generator.choice(self.dataset.track_ids, size=probe_count, replace=False)
        self.probe_track_ids = [str(track_id) for track_id in selected]
        self.probe_indices = []
        self.monitor_recipes = []
        view_starts = {
            "canonical": [float(value) for value in data_cfg["canonical_starts"]],
            "shifted": [
                float(value) for value in data_cfg["shifted_training_starts"]
            ],
            "heldout": [
                float(value) for value in data_cfg["shifted_evaluation_starts"]
            ],
        }
        for track_offset, track_id in enumerate(self.probe_track_ids):
            for view_offset, (view_type, starts) in enumerate(view_starts.items()):
                position = starts[
                    (track_offset + int(self.cfg["train"]["seed"]) + view_offset)
                    % len(starts)
                ]
                index = self.evaluation_dataset.indices_for(
                    [track_id], view_type, position
                )[0]
                self.probe_indices.append(index)
                self.monitor_recipes.append(
                    {
                        "track_id": track_id,
                        "view_type": view_type,
                        "start": position,
                    }
                )

    def _common_loader_args(self) -> dict:
        workers = int(self.cfg["dataloader"]["num_workers"])
        return {
            "num_workers": workers,
            "pin_memory": True,
            "persistent_workers": workers > 0
            and bool(self.cfg["dataloader"]["persistent_workers"]),
            "prefetch_factor": int(self.cfg["dataloader"]["prefetch_factor"])
            if workers > 0
            else None,
            "worker_init_fn": seed_worker,
        }

    def _collate(self):
        return partial(
            collate_causal_documents,
            vocabulary=self.vocabulary,
            max_positions=int(self.cfg["model"]["max_position_embeddings"]),
        )

    def _training_collate(self):
        return OnlinePairCollator(
            vocabulary=self.vocabulary,
            max_positions=int(self.cfg["model"]["max_position_embeddings"]),
            audio_root=self.cfg["data"]["audio_root"],
            sample_rate=int(self.tokenizer_spec.sample_rate),
            segment_duration=float(self.cfg["data"]["segment_duration"]),
            noise_assets=self.noise_assets,
            seed=int(self.cfg["train"]["seed"]),
            nominal_max_steps=int(self.cfg["train"]["max_steps"]),
        )

    def train_dataloader(self) -> DataLoader:
        sampler = PairedViewBatchSampler(
            self.dataset,
            tracks_per_microbatch=int(self.cfg["train"]["tracks_per_microbatch"]),
            seed=int(self.cfg["train"]["seed"]),
            catalogue_pass=int(self.trainer.current_epoch),
            world_size=int(self.trainer.world_size),
            rank=int(self.trainer.global_rank),
            batch_count_multiple=int(
                self.cfg["trainer"]["accumulate_grad_batches"]
            ),
        )
        self.batch_sampler = sampler
        self.train_loader = ResumableDataLoader(
            self.dataset,
            batch_sampler=sampler,
            collate_fn=self._training_collate(),
            **self._common_loader_args(),
        )
        if hasattr(self, "_pending_loader_state"):
            self.train_loader.load_state_dict(self._pending_loader_state)
            del self._pending_loader_state
        return self.train_loader

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.evaluation_dataset, self.probe_indices),
            batch_size=int(self.cfg["evaluation"]["probe_batch_size"]),
            shuffle=False,
            collate_fn=self._collate(),
            **self._common_loader_args(),
        )

    def generation_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.evaluation_dataset, self.probe_indices),
            batch_size=int(self.cfg["evaluation"]["generation_batch_size"]),
            shuffle=False,
            collate_fn=NoiseEvaluationCollator(
                vocabulary=self.vocabulary,
                max_positions=int(
                    self.cfg["model"]["max_position_embeddings"]
                ),
                audio_root=self.cfg["data"]["audio_root"],
                sample_rate=int(self.tokenizer_spec.sample_rate),
                segment_duration=float(self.cfg["data"]["segment_duration"]),
                noise_assets=self.noise_assets,
                seed=int(self.cfg["train"]["seed"]) + 1771,
            ),
            **self._common_loader_args(),
        )

    def set_catalogue_pass(self, catalogue_pass: int) -> None:
        if hasattr(self, "batch_sampler"):
            self.batch_sampler.set_epoch(catalogue_pass)

    def state_dict(self) -> dict:
        return {
            "train_loader": (
                self.train_loader.state_dict()
                if hasattr(self, "train_loader")
                else {"batches_yielded": 0}
            )
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._pending_loader_state = state_dict.get(
            "train_loader", {"batches_yielded": 0}
        )


class AudioLMModule(pl.LightningModule):
    def __init__(
        self,
        cfg: dict,
        vocabulary: AudioLMVocabulary,
        tokenizer_spec: TokenizerSpec,
        code_mapping_fingerprint: str,
        validation_probe: list[str],
        training_track_ids: list[str],
        training_corpus_fingerprint: str,
        monitor_recipes: list[dict],
        noise_manifest: dict,
    ):
        super().__init__()
        self.cfg = cfg
        self.vocabulary = vocabulary
        self.tokenizer_spec = tokenizer_spec
        self.code_mapping_fingerprint = code_mapping_fingerprint
        self.validation_probe = validation_probe
        self.training_track_ids = training_track_ids
        self.training_corpus_fingerprint = training_corpus_fingerprint
        self.monitor_recipes = monitor_recipes
        self.noise_manifest = noise_manifest
        self.model = AudioCausalLM(cfg, vocabulary)
        curriculum_cfg = cfg["train"]["curriculum"]
        self.curriculum = AdaptiveCurriculum(
            nominal_max_steps=int(cfg["train"]["max_steps"]),
            gate_threshold=float(curriculum_cfg["gate_threshold"]),
            gate_max_extra_steps=int(curriculum_cfg["gate_max_extra_steps"]),
            regression_drop=float(curriculum_cfg["regression_drop"]),
            recovery_probes=int(curriculum_cfg["recovery_probes"]),
            recovery_timeout_steps=int(
                curriculum_cfg["recovery_timeout_steps"]
            ),
        )
        self.online_tokenizer = None
        self._last_probe_step = -1
        self.documents_consumed = 0
        self.tokens_consumed = 0
        self.started_at = time.perf_counter()
        self.session_tokens_start = 0
        self.snr_epoch_counts = {name: 0 for name, _, _ in SNR_BINS}
        self.snr_epoch_counts["exact_zero"] = 0
        self.snr_epoch_counts["noisy_documents"] = 0
        self.snr_epoch_counts["skipped_documents"] = 0
        self.save_hyperparameters(cfg)

    def _step(self, batch: dict, prefix: str) -> torch.Tensor:
        batch_size = int(batch["input_ids"].shape[0])
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss, overall_metrics, _ = causal_losses_by_view(
            logits,
            batch["input_ids"],
            batch["audio_target_mask"],
            batch["id_target_mask"],
            batch["boundary_target_mask"],
            batch.get("pair_role", batch["view_type"]),
            view_mode="paired_roles" if "pair_role" in batch else "canonical_only",
            id_digit_weight=float(self.cfg["train"]["id_digit_weight"]),
        )
        for name, value in overall_metrics.items():
            if name not in VALIDATION_METRICS:
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name in {"audio_loss", "teacher_forced_exact_accuracy"},
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=prefix == "train",
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if prefix == "train":
            self.documents_consumed += int(batch["input_ids"].shape[0]) * int(
                self.trainer.world_size
            )
            self.tokens_consumed += int(batch["attention_mask"].sum()) * int(
                self.trainer.world_size
            )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        batch, augmentation = self._apply_online_noise(batch, batch_idx)
        for name, value in augmentation.items():
            if name not in AUGMENTATION_METRICS:
                continue
            self.log(
                f"augmentation/{name}",
                value,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        logits, final_hidden_states = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            return_final_hidden_state=True,
        )
        schedule = noise_consistency_schedule(
            self.curriculum.effective_step(int(self.global_step)),
            max_steps=int(self.cfg["train"]["max_steps"]),
        )
        consistency_weight = (
            schedule.consistency_weight
            * self.curriculum.consistency_multiplier
        )
        loss, metrics = noise_consistency_losses(
            logits,
            final_hidden_states,
            batch["input_ids"],
            batch["audio_target_mask"],
            batch["id_target_mask"],
            batch["boundary_target_mask"],
            batch["is_noisy"],
            batch["track_id"],
            id_digit_weight=float(self.cfg["train"]["id_digit_weight"]),
            consistency_weight=consistency_weight,
        )
        batch_size = int(batch["input_ids"].shape[0])
        for name, value in metrics.items():
            if name not in TRAIN_METRICS:
                continue
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=False,
                prog_bar=name
                in {"clean_audio_loss", "teacher_forced_exact_accuracy"},
                sync_dist=True,
                batch_size=batch_size,
            )
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.documents_consumed += batch_size * int(self.trainer.world_size)
        self.tokens_consumed += int(batch["attention_mask"].sum()) * int(
            self.trainer.world_size
        )
        return loss

    def on_train_batch_end(
        self, outputs, batch: dict, batch_idx: int
    ) -> None:
        ended_at = time.perf_counter()
        if hasattr(self, "_previous_batch_end"):
            self.log(
                "throughput/total_batch_seconds",
                ended_at - self._previous_batch_end,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )
        self._previous_batch_end = ended_at

    def _apply_online_noise(self, batch: dict, batch_idx: int) -> tuple[dict, dict]:
        effective_step = self.curriculum.effective_step(int(self.global_step))
        schedule = noise_consistency_schedule(
            effective_step,
            max_steps=int(self.cfg["train"]["max_steps"]),
        )
        pair_count = len(batch["augmentation_keys"])
        selected, snrs, snr_bins = deterministic_consistency_noise_parameters(
            batch["augmentation_keys"],
            schedule=schedule,
            seed=int(self.cfg["train"]["seed"]),
            step=int(self.global_step),
            batch_idx=batch_idx,
        )
        realized_pairs: list[int] = []
        skipped_pairs: list[int] = []
        tokenizer_seconds = 0.0
        proposed_pairs = [index for index, keep in enumerate(selected) if keep]
        if proposed_pairs:
            loaded = prefetched_waveforms_for_step(
                batch,
                global_step=int(self.global_step),
                batch_idx=batch_idx,
            )
            missing_pairs = [pair for pair in proposed_pairs if pair not in loaded]
            for pair in missing_pairs:
                anchor_row = pair * 2
                audio = load_audio(
                    Path(self.cfg["data"]["audio_root"])
                    / batch["source_path"][anchor_row],
                    sample_rate=int(self.tokenizer_spec.sample_rate),
                    start=float(batch["segment_start"][anchor_row]),
                    duration=float(batch["segment_duration"][anchor_row]),
                    pad=True,
                )
                key = batch["augmentation_keys"][pair]
                noise_key = stable_uint64(
                    int(self.cfg["train"]["seed"]),
                    int(self.global_step),
                    int(batch_idx),
                    pair,
                    key,
                    "training-noise",
                )
                noise = self.trainer.datamodule.noise_assets.load_training(
                    noise_key
                )
                loaded[pair] = (
                    torch.from_numpy(audio).to(self.device),
                    torch.from_numpy(noise).to(self.device),
                )
            anchors = torch.stack(
                [loaded[pair][0] for pair in proposed_pairs]
            ).to(self.device)
            noises = torch.stack(
                [loaded[pair][1] for pair in proposed_pairs]
            ).to(self.device)
            selected_snrs = torch.tensor(
                [snr for snr, keep in zip(snrs, selected, strict=True) if keep],
                device=self.device,
                dtype=torch.float32,
            )
            mixed, valid = mix_background_noise(anchors, noises, selected_snrs)
            if valid.any():
                if self.online_tokenizer is None:
                    raise RuntimeError("Online MuQ tokenizer has not been initialized")
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                token_started = time.perf_counter()
                tokens = self.online_tokenizer.tokenize(mixed[valid])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                tokenizer_seconds = time.perf_counter() - token_started
                realized_pairs = [
                    pair
                    for pair, keep in zip(
                        proposed_pairs, valid.tolist(), strict=True
                    )
                    if keep
                ]
                if tokens.shape[0] != len(realized_pairs):
                    raise RuntimeError(
                        "Online token batch does not match selected pairs"
                    )
                batch = replace_secondary_rows_with_noisy_tokens(
                    batch,
                    tokens,
                    realized_pairs,
                    id_token_id=self.vocabulary.id_token_id,
                )
            skipped_pairs = [
                pair
                for pair, keep in zip(
                    proposed_pairs, valid.tolist(), strict=True
                )
                if not keep
            ]
        if "is_noisy" not in batch:
            batch = {
                **batch,
                "is_noisy": torch.zeros(
                    len(batch["input_ids"]),
                    device=batch["input_ids"].device,
                    dtype=torch.bool,
                ),
            }
        for pair in realized_pairs:
            bin_name = snr_bins[pair]
            if bin_name is not None:
                self.snr_epoch_counts[bin_name] += 1
            if snrs[pair] == 0.0:
                self.snr_epoch_counts["exact_zero"] += 1
        self.snr_epoch_counts["noisy_documents"] += len(realized_pairs)
        self.snr_epoch_counts["skipped_documents"] += len(skipped_pairs)
        metrics = {
            "scheduled_probability": float(schedule.probability),
            "scheduled_consistency_weight": float(
                schedule.consistency_weight
                * self.curriculum.consistency_multiplier
            ),
            "realized_noisy_fraction": len(realized_pairs) / max(1, pair_count),
            "effective_step": float(effective_step),
            "consistency_multiplier": float(
                self.curriculum.consistency_multiplier
            ),
            "gate_open": float(self.curriculum.gate_open),
            "recovery_active": float(self.curriculum.recovery_active),
            "skipped_noisy_documents": float(len(skipped_pairs)),
            "mean_snr_db": (
                sum(snrs[pair] for pair in realized_pairs) / len(realized_pairs)
                if realized_pairs
                else 0.0
            ),
            "online_tokenization_seconds": tokenizer_seconds,
        }
        probabilities = schedule.snr_bin_probabilities or (0.0, 0.0, 0.0, 0.0)
        for (name, _, _), probability in zip(
            SNR_BINS, probabilities, strict=True
        ):
            metrics[f"scheduled_snr_probability_{name}"] = float(probability)
        return batch, metrics

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "validation")

    def on_train_epoch_start(self) -> None:
        self.trainer.datamodule.set_catalogue_pass(int(self.current_epoch))

    def on_fit_start(self) -> None:
        if int(self.trainer.world_size) != 1:
            raise ValueError(
                "noise_consistency_curriculum_v1 currently requires one GPU"
            )
        self.started_at = time.perf_counter()
        self.session_tokens_start = self.tokens_consumed
        tokenizer_cfg = self.cfg["tokenizer"]
        self.online_tokenizer = MuQRVQTokenizer(
            tokenizer_cfg["model_name"],
            revision=tokenizer_cfg.get("revision", "main"),
            selected_codebooks=int(tokenizer_cfg["selected_codebooks"]),
            sample_rate=int(tokenizer_cfg["sample_rate"]),
            device=self.device,
            lightweight=True,
        )
        if self.online_tokenizer.spec.fingerprint != self.tokenizer_spec.fingerprint:
            raise ValueError("Online tokenizer does not match cached token shards")
        self._verify_cached_token_equivalence()
        if (
            int(self.global_step) == 0
            and self.cfg["evaluation"].get("online_monitor_enabled", True)
        ):
            self._generation_probe()
        if self.curriculum.completed(int(self.global_step)):
            self.trainer.should_stop = True

    def _verify_cached_token_equivalence(self) -> None:
        count = int(
            self.cfg["data"]["background_noise"].get(
                "preflight_examples_per_view", 2
            )
        )
        if count < 1:
            return
        dataset = self.trainer.datamodule.dataset
        examples = []
        for track_id in dataset.track_ids[:count]:
            for view_type in ("canonical", "shifted"):
                indices = dataset.view_indices(track_id, view_type)
                examples.append(dataset[indices[0]])
        waveforms = []
        for example in examples:
            audio = load_audio(
                Path(self.cfg["data"]["audio_root"]) / example["source_path"],
                sample_rate=int(self.tokenizer_spec.sample_rate),
                start=float(example["segment_start"]),
                duration=float(example["segment_duration"]),
                pad=True,
            )
            waveforms.append(audio)
        online = self.online_tokenizer.tokenize(
            torch.from_numpy(np.stack(waveforms)).to(self.device)
        ).cpu()
        for example, tokens in zip(examples, online, strict=True):
            if not torch.equal(example["audio_tokens"].long(), tokens.long()):
                raise RuntimeError(
                    "Online clean tokenization does not reproduce cached tokens for "
                    f"{example['track_id']} at {example['segment_start']:g}s"
                )

    def on_train_epoch_end(self) -> None:
        for name in self.snr_epoch_counts:
            self.snr_epoch_counts[name] = 0
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        self.log(
            "throughput/tokens_per_second",
            float((self.tokens_consumed - self.session_tokens_start) / elapsed),
            sync_dist=False,
        )
        if torch.cuda.is_available():
            self.log(
                "system/peak_gpu_memory_bytes",
                float(torch.cuda.max_memory_allocated(self.device)),
                sync_dist=False,
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_validation_epoch_end(self) -> None:
        self._generation_probe()

    def on_train_end(self) -> None:
        if self._last_probe_step != int(self.global_step):
            self._generation_probe()

    def _generation_probe(self) -> None:
        if not self.cfg["evaluation"].get("online_monitor_enabled", True):
            self._last_probe_step = int(self.global_step)
            if self.curriculum.completed(int(self.global_step)):
                self.trainer.should_stop = True
            return
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        if self.online_tokenizer is None:
            raise RuntimeError("Online MuQ tokenizer has not been initialized")
        was_training = self.model.training
        self.model.eval()
        clean_rows = []
        noisy_rows = []
        shifted_teacher_forced_correct = 0
        shifted_teacher_forced_count = 0
        snr_values = [
            float(value) for value in self.cfg["evaluation"]["noise_snr_db"]
        ]
        tokenization_seconds = 0.0
        with torch.inference_mode():
            loader = tqdm(
                self.trainer.datamodule.generation_dataloader(),
                desc=f"clean/noisy monitor step {int(self.global_step)}",
                disable=not self.trainer.is_global_zero,
            )
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                clean_logits = self.model(input_ids, attention_mask)
                id_columns = (input_ids == self.vocabulary.id_token_id).nonzero()
                unique_columns = id_columns[:, 1].unique()
                if len(unique_columns) != 1:
                    raise RuntimeError("Generation batch has inconsistent ID boundaries")
                id_column = int(unique_columns[0])
                prompts = input_ids[:, : id_column + 1]
                predictions = clean_logits[:, :-1].argmax(dim=-1)
                targets_tensor = input_ids[:, 1:]
                id_mask = batch["id_target_mask"].to(self.device)
                exact = ((predictions == targets_tensor) | ~id_mask).all(dim=1)
                shifted_rows = torch.tensor(
                    [view == "shifted" for view in batch["view_type"]],
                    device=self.device,
                    dtype=torch.bool,
                )
                shifted_teacher_forced_correct += int(exact[shifted_rows].sum())
                shifted_teacher_forced_count += int(shifted_rows.sum())
                rankings = batched_beam_generate(
                    self.model,
                    prompts,
                    self.vocabulary,
                    width=int(self.cfg["evaluation"]["beam_width"]),
                )
                clean_rows.extend(
                    {
                        "target": target,
                        "view_type": view_type,
                        "beam": ranking,
                    }
                    for target, view_type, ranking in zip(
                        batch["code"],
                        batch["view_type"],
                        rankings,
                        strict=True,
                    )
                )
                clean_waveforms = batch["clean_waveforms"].to(self.device)
                noise_waveforms = batch["noise_waveforms"].to(self.device)
                batch_size = clean_waveforms.shape[0]
                repeated_clean = clean_waveforms.repeat(len(snr_values), 1)
                repeated_noise = noise_waveforms.repeat(len(snr_values), 1)
                repeated_snr = torch.tensor(
                    snr_values,
                    device=self.device,
                    dtype=torch.float32,
                ).repeat_interleave(batch_size)
                mixed, valid = mix_background_noise(
                    repeated_clean, repeated_noise, repeated_snr
                )
                if not valid.any():
                    continue
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                token_started = time.perf_counter()
                noisy_tokens = self.online_tokenizer.tokenize(mixed[valid])
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                tokenization_seconds += time.perf_counter() - token_started
                noisy_prompts = prompts_from_audio_tokens(
                    noisy_tokens, self.vocabulary
                )
                noisy_rankings = []
                for offset in range(0, len(noisy_prompts), batch_size):
                    prompt_batch = noisy_prompts[offset : offset + batch_size]
                    noisy_rankings.extend(
                        batched_beam_generate(
                            self.model,
                            prompt_batch,
                            self.vocabulary,
                            width=int(self.cfg["evaluation"]["beam_width"]),
                        )
                    )
                valid_rows = valid.cpu().tolist()
                targets = [
                    value
                    for value, keep in zip(
                        batch["code"] * len(snr_values),
                        valid_rows,
                        strict=True,
                    )
                    if keep
                ]
                views = [
                    value
                    for value, keep in zip(
                        batch["view_type"] * len(snr_values),
                        valid_rows,
                        strict=True,
                    )
                    if keep
                ]
                valid_snrs = repeated_snr[valid].tolist()
                noisy_rows.extend(
                    {
                        "target": target,
                        "view_type": view_type,
                        "snr_db": snr,
                        "beam": ranking,
                    }
                    for (
                        target,
                        view_type,
                        snr,
                        ranking,
                    ) in zip(
                        targets,
                        views,
                        valid_snrs,
                        noisy_rankings,
                        strict=True,
                    )
                )
        if was_training:
            self.model.train()

        def beam_top1(rows: list[dict]) -> float:
            if not rows:
                return float("nan")
            return sum(
                bool(row["beam"]) and row["beam"][0].code == row["target"]
                for row in rows
            ) / len(rows)

        metrics: dict[str, float] = {}
        for view_type in ("canonical", "shifted", "heldout"):
            selected = [
                row for row in clean_rows if row["view_type"] == view_type
            ]
            metrics[f"probe/clean/{view_type}/beam_top1"] = beam_top1(selected)
            noisy_for_view = [
                row for row in noisy_rows if row["view_type"] == view_type
            ]
            metrics[f"probe/noise/{view_type}/beam_top1"] = beam_top1(
                noisy_for_view
            )
        for snr in snr_values:
            noisy = [
                row for row in noisy_rows if math.isclose(row["snr_db"], snr)
            ]
            metrics[f"probe/noise/snr_{snr:g}/beam_top1"] = beam_top1(noisy)
        metrics["probe/noise/aggregate/beam_top1"] = beam_top1(noisy_rows)
        metrics["probe/noise/online_tokenization_seconds"] = tokenization_seconds
        shifted_teacher_forced_exact = (
            shifted_teacher_forced_correct / shifted_teacher_forced_count
            if shifted_teacher_forced_count
            else float("nan")
        )
        shifted_clean = [
            row for row in clean_rows if row["view_type"] == "shifted"
        ]
        shifted_beam_top1 = beam_top1(shifted_clean)
        metrics[
            "probe/clean/shifted/teacher_forced_exact_accuracy"
        ] = shifted_teacher_forced_exact
        current_schedule = noise_consistency_schedule(
            self.curriculum.effective_step(int(self.global_step)),
            max_steps=int(self.cfg["train"]["max_steps"]),
        )
        decision = self.curriculum.observe_probe(
            global_step=int(self.global_step),
            shifted_teacher_forced_exact=shifted_teacher_forced_exact,
            shifted_beam_top1=shifted_beam_top1,
            consistency_is_active=(
                current_schedule.consistency_weight
                * self.curriculum.consistency_multiplier
                > 0
            ),
        )
        metrics["curriculum/effective_step"] = float(
            self.curriculum.effective_step(int(self.global_step))
        )
        metrics["curriculum/gate_open"] = float(self.curriculum.gate_open)
        metrics["curriculum/recovery_active"] = float(
            self.curriculum.recovery_active
        )
        metrics["curriculum/consistency_multiplier"] = float(
            self.curriculum.consistency_multiplier
        )
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=self.global_step)
        self._last_probe_step = int(self.global_step)
        if decision.failure is not None:
            self._save_diagnostic_and_fail(decision.failure)
        if self.curriculum.completed(int(self.global_step)):
            self.trainer.should_stop = True

    def _save_diagnostic_and_fail(self, reason: str) -> None:
        path = checkpoint_dir(self.cfg) / f"diagnostic-step-{int(self.global_step)}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.trainer.save_checkpoint(path)
        raise RuntimeError(
            f"tc6 adaptive curriculum stopped: {reason}. "
            f"Diagnostic checkpoint: {path}"
        )

    def on_before_optimizer_step(self, optimizer) -> None:
        if self.global_step % int(self.cfg["trainer"]["log_every_n_steps"]):
            return
        gradients = [
            parameter.grad.detach().norm(2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        norm = torch.stack(gradients).norm(2) if gradients else torch.tensor(0.0)
        self.log("train/gradient_norm", norm, sync_dist=False)
        self.log(
            "train/learning_rate",
            float(optimizer.param_groups[0]["lr"]),
            sync_dist=False,
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        numpy_state = np.random.get_state()
        checkpoint.update(
            {
                "architecture": ARCHITECTURE,
                "tokenizer_spec": self.tokenizer_spec.to_dict(),
                "tokenizer_fingerprint": self.tokenizer_spec.fingerprint,
                "vocabulary": self.vocabulary.to_dict(),
                "model_config": self.cfg["model"],
                "code_mapping_fingerprint": self.code_mapping_fingerprint,
                "validation_probe": self.validation_probe,
                "training_track_ids": self.training_track_ids,
                "training_corpus_fingerprint": self.training_corpus_fingerprint,
                "training_protocol": TRAINING_PROTOCOL,
                "loss_protocol": LOSS_PROTOCOL,
                "monitor_protocol": MONITOR_PROTOCOL,
                "curriculum_config": self.cfg["train"]["curriculum"],
                "adaptive_curriculum_state": self.curriculum.state_dict(),
                "snr_epoch_counts": self.snr_epoch_counts,
                "monitor_recipes": self.monitor_recipes,
                "background_noise_manifest": self.noise_manifest,
                "view_policy": {
                    "view_mode": self.cfg["data"]["view_mode"],
                    "canonical_starts": self.cfg["data"]["canonical_starts"],
                    "shifted_training_starts": self.cfg["data"][
                        "shifted_training_starts"
                    ],
                    "shifted_evaluation_starts": self.cfg["data"][
                        "shifted_evaluation_starts"
                    ],
                },
                "documents_consumed": self.documents_consumed,
                "tokens_consumed": self.tokens_consumed,
                "last_probe_step": self._last_probe_step,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": {
                    "bit_generator": numpy_state[0],
                    "state": numpy_state[1].tolist(),
                    "position": numpy_state[2],
                    "has_gauss": numpy_state[3],
                    "cached_gaussian": numpy_state[4],
                },
                "torch_rng_state": torch.get_rng_state(),
            }
        )
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("Cannot resume from a non-audio_lm_v1 checkpoint")
        if checkpoint.get("tokenizer_fingerprint") != self.tokenizer_spec.fingerprint:
            raise ValueError("Resume checkpoint tokenizer fingerprint does not match")
        if checkpoint.get("vocabulary") != self.vocabulary.to_dict():
            raise ValueError("Resume checkpoint vocabulary does not match")
        if checkpoint.get("code_mapping_fingerprint") != self.code_mapping_fingerprint:
            raise ValueError("Resume checkpoint code mapping does not match")
        if checkpoint.get("validation_probe") != self.validation_probe:
            raise ValueError("Resume checkpoint validation probe does not match")
        if checkpoint.get("training_track_ids") != self.training_track_ids:
            raise ValueError("Resume checkpoint training-track subset does not match")
        if (
            checkpoint.get("training_protocol")
            != TRAINING_PROTOCOL
        ):
            raise ValueError("Resume checkpoint uses a different training protocol")
        if checkpoint.get("loss_protocol") != LOSS_PROTOCOL:
            raise ValueError(
                "Resume checkpoint uses the invalid pre-fix tc6 loss protocol; "
                "restart tc6 from scratch"
            )
        adaptive_state = checkpoint["adaptive_curriculum_state"]
        if (
            checkpoint.get("monitor_protocol") != MONITOR_PROTOCOL
            and adaptive_state["gate_open"]
        ):
            raise ValueError(
                "Resume checkpoint opened its adaptive gate with the old greedy "
                "monitor; its regression baseline is incompatible with the "
                "beam-only monitor"
            )
        if checkpoint.get("curriculum_config") != self.cfg["train"]["curriculum"]:
            raise ValueError("Resume checkpoint uses a different curriculum")
        if checkpoint.get("monitor_recipes") != self.monitor_recipes:
            raise ValueError("Resume checkpoint monitoring recipes do not match")
        if checkpoint.get("background_noise_manifest") != self.noise_manifest:
            raise ValueError("Resume checkpoint background-noise assets do not match")
        if (
            checkpoint.get("training_corpus_fingerprint")
            != self.training_corpus_fingerprint
        ):
            raise ValueError(
                "Resume checkpoint paired-view corpus fingerprint does not match"
            )
        self.documents_consumed = int(checkpoint.get("documents_consumed", 0))
        self.tokens_consumed = int(checkpoint.get("tokens_consumed", 0))
        self._last_probe_step = int(checkpoint.get("last_probe_step", -1))
        self.curriculum.load_state_dict(adaptive_state)
        self.snr_epoch_counts = {
            key: int(value)
            for key, value in checkpoint.get(
                "snr_epoch_counts", self.snr_epoch_counts
            ).items()
        }
        random.setstate(checkpoint["python_rng_state"])
        numpy_state = checkpoint["numpy_rng_state"]
        np.random.set_state(
            (
                numpy_state["bit_generator"],
                np.asarray(numpy_state["state"], dtype=np.uint32),
                numpy_state["position"],
                numpy_state["has_gauss"],
                numpy_state["cached_gaussian"],
            )
        )
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

    def configure_optimizers(self):
        train_cfg = self.cfg["train"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            betas=tuple(float(value) for value in train_cfg["betas"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        max_steps = int(train_cfg["max_steps"])
        warmup_steps = int(train_cfg["warmup_steps"])
        if not 0 < warmup_steps < max_steps:
            raise ValueError("warmup_steps must be between zero and max_steps")

        def schedule(step: int) -> float:
            effective_step = self.curriculum.effective_step(step)
            return effective_cosine_multiplier(
                effective_step,
                max_steps=max_steps,
                warmup_steps=warmup_steps,
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def output_dir(cfg: dict) -> Path:
    base = Path(cfg["train"]["log_dir"])
    return base / cfg["train"]["run_id"] if cfg["train"].get("run_id") else base


def checkpoint_dir(cfg: dict) -> Path:
    base = Path(cfg["train"]["checkpoint_dir"])
    return base / cfg["train"]["run_id"] if cfg["train"].get("run_id") else base


def build_logger(cfg: dict, directory: Path):
    wandb = cfg["train"]["wandb"]
    if not wandb.get("enabled", False):
        return False
    run_id = cfg["train"].get("run_id")
    return WandbLogger(
        project=wandb.get("project", "para-audio-id"),
        entity=wandb.get("entity"),
        name=wandb.get("name") or run_id,
        version=run_id,
        resume="allow" if run_id else None,
        save_dir=str(directory),
        mode=wandb.get("mode", "offline"),
        config=cfg,
    )


def train(cfg: dict, *, checkpoint: str | Path | None = None) -> None:
    if cfg.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Configuration architecture must be {ARCHITECTURE}")
    if cfg["train"]["curriculum"].get("protocol") != TRAINING_PROTOCOL:
        raise ValueError(f"Curriculum protocol must be {TRAINING_PROTOCOL}")
    if cfg["train"]["curriculum"].get("loss_protocol") != LOSS_PROTOCOL:
        raise ValueError(f"Loss protocol must be {LOSS_PROTOCOL}")
    seed = int(cfg["train"]["seed"])
    pl.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(
        bool(cfg["train"]["deterministic"]), warn_only=bool(cfg["train"]["deterministic_warn_only"])
    )
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
    directory = output_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    save_config(cfg, directory / "config.yaml")
    datamodule = AudioLMDataModule(cfg)
    datamodule.setup("fit")
    (directory / "training_tracks.json").write_text(
        json.dumps(datamodule.dataset.track_ids, indent=2) + "\n"
    )
    (directory / "monitor_recipes.json").write_text(
        json.dumps(datamodule.monitor_recipes, indent=2, sort_keys=True) + "\n"
    )
    (directory / "background_noise_manifest.json").write_text(
        json.dumps(datamodule.noise_assets.manifest(), indent=2, sort_keys=True)
        + "\n"
    )
    module = AudioLMModule(
        cfg,
        datamodule.vocabulary,
        datamodule.tokenizer_spec,
        datamodule.code_mapping_fingerprint,
        datamodule.probe_track_ids,
        datamodule.dataset.track_ids,
        datamodule.training_corpus_fingerprint,
        datamodule.monitor_recipes,
        datamodule.noise_assets.manifest(),
    )
    logger = build_logger(cfg, directory)
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir(cfg),
            filename="step-{step}",
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=int(cfg["train"]["checkpoint_interval"]),
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    ]
    accumulation = int(cfg["trainer"]["accumulate_grad_batches"])
    display_sampler = PairedViewBatchSampler(
        datamodule.dataset,
        tracks_per_microbatch=int(cfg["train"]["tracks_per_microbatch"]),
        seed=seed,
        batch_count_multiple=accumulation,
    )
    optimizer_steps_per_pass = len(display_sampler) // accumulation
    display_max_epochs = math.ceil(
        module.curriculum.hard_max_steps / optimizer_steps_per_pass
    )
    evaluation_batches = int(cfg["train"]["evaluation_interval"]) * accumulation
    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=cfg["trainer"]["strategy"],
        max_epochs=display_max_epochs,
        max_steps=module.curriculum.hard_max_steps,
        precision=cfg["trainer"]["precision"],
        accumulate_grad_batches=accumulation,
        gradient_clip_val=float(cfg["train"]["gradient_clip_norm"]),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=directory,
        log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
        val_check_interval=evaluation_batches,
        check_val_every_n_epoch=None,
        deterministic=bool(cfg["train"]["deterministic"]),
        use_distributed_sampler=False,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=checkpoint)
