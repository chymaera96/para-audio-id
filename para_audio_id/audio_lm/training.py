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
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from ..config import save_config
from .checkpoint import ARCHITECTURE
from .dataset import (
    CachedPositionDataset,
    PairedAudioTokenDataset,
    PairedViewBatchSampler,
    collate_causal_documents,
)
from .generation import (
    batched_beam_generate,
    batched_greedy_generate,
)
from .losses import causal_losses_by_view
from .model import AudioCausalLM
from .token_store import TokenStoreIndex
from .tokenizer import TokenizerSpec
from .tokenization import load_training_track_ids
from .vocabulary import AudioLMVocabulary


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
        }
        self.training_corpus_fingerprint = hashlib.sha256(
            json.dumps(corpus_payload, sort_keys=True).encode()
        ).hexdigest()
        probe_count = min(int(self.cfg["evaluation"]["probe_tracks"]), len(self.dataset.track_ids))
        generator = np.random.default_rng(int(self.cfg["train"]["seed"]) + 919)
        selected = generator.choice(self.dataset.track_ids, size=probe_count, replace=False)
        self.probe_track_ids = [str(track_id) for track_id in selected]
        self.probe_indices = []
        for track_id in self.probe_track_ids:
            self.probe_indices.append(
                self.dataset.view_indices(track_id, "canonical")[0]
            )
            if view_mode == "paired":
                self.probe_indices.append(
                    self.dataset.view_indices(track_id, "shifted")[0]
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
        return ResumableDataLoader(
            self.dataset,
            batch_sampler=sampler,
            collate_fn=self._collate(),
            **self._common_loader_args(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            Subset(self.dataset, self.probe_indices),
            batch_size=int(self.cfg["evaluation"]["probe_batch_size"]),
            shuffle=False,
            collate_fn=self._collate(),
            **self._common_loader_args(),
        )

    def generation_dataloader(
        self, *, canonical_start: float, heldout_start: float
    ) -> DataLoader:
        indices = self.evaluation_dataset.indices_for(
            self.probe_track_ids, "canonical", canonical_start
        ) + self.evaluation_dataset.indices_for(
            self.probe_track_ids, "heldout", heldout_start
        )
        return DataLoader(
            Subset(self.evaluation_dataset, indices),
            batch_size=int(self.cfg["evaluation"]["generation_batch_size"]),
            shuffle=False,
            collate_fn=self._collate(),
            **self._common_loader_args(),
        )

    def set_catalogue_pass(self, catalogue_pass: int) -> None:
        if hasattr(self, "batch_sampler"):
            self.batch_sampler.set_epoch(catalogue_pass)


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
    ):
        super().__init__()
        self.cfg = cfg
        self.vocabulary = vocabulary
        self.tokenizer_spec = tokenizer_spec
        self.code_mapping_fingerprint = code_mapping_fingerprint
        self.validation_probe = validation_probe
        self.training_track_ids = training_track_ids
        self.training_corpus_fingerprint = training_corpus_fingerprint
        self.model = AudioCausalLM(cfg, vocabulary)
        self.documents_consumed = 0
        self.tokens_consumed = 0
        self.started_at = time.perf_counter()
        self.session_documents_start = 0
        self.session_tokens_start = 0
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
            batch["view_type"],
            view_mode=self.cfg["data"]["view_mode"],
            id_digit_weight=float(self.cfg["train"]["id_digit_weight"]),
        )
        for name, value in overall_metrics.items():
            if name == "loss":
                continue
            self.log(
                f"{prefix}/{name}",
                value,
                on_step=prefix == "train",
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
        self.log(
            "train/distinct_tracks",
            float(len(set(batch["track_id"])) * int(self.trainer.world_size)),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "validation")

    def on_train_epoch_start(self) -> None:
        self.trainer.datamodule.set_catalogue_pass(int(self.current_epoch))
        self.log("progress/catalogue_pass", float(self.current_epoch), sync_dist=False)

    def on_fit_start(self) -> None:
        self.started_at = time.perf_counter()
        self.session_documents_start = self.documents_consumed
        self.session_tokens_start = self.tokens_consumed
        if self.trainer.is_global_zero and self.logger is not None:
            self.logger.log_metrics(
                {
                    "model/parameters": float(
                        sum(parameter.numel() for parameter in self.model.parameters())
                    ),
                    "data/complete_tracks": float(
                        len(self.trainer.datamodule.dataset.track_ids)
                    ),
                    "data/view_mode": self.cfg["data"]["view_mode"],
                    "train/id_digit_weight": float(
                        self.cfg["train"]["id_digit_weight"]
                    ),
                },
                step=self.global_step,
            )

    def on_train_epoch_end(self) -> None:
        self.log(
            "progress/documents_consumed",
            float(self.documents_consumed),
            sync_dist=False,
        )
        self.log("progress/tokens_consumed", float(self.tokens_consumed), sync_dist=False)
        elapsed = max(time.perf_counter() - self.started_at, 1e-6)
        self.log(
            "throughput/documents_per_second",
            float((self.documents_consumed - self.session_documents_start) / elapsed),
            sync_dist=False,
        )
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

    def _generation_probe(self) -> None:
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        interval = int(self.cfg["train"]["evaluation_interval"])
        event = max(0, int(self.global_step) // interval - 1)
        canonical_starts = [
            float(value) for value in self.cfg["data"]["canonical_starts"]
        ]
        heldout_starts = [
            float(value) for value in self.cfg["data"]["shifted_evaluation_starts"]
        ]
        canonical_start = canonical_starts[event % len(canonical_starts)]
        heldout_start = heldout_starts[event % len(heldout_starts)]
        beam_interval = int(self.cfg["evaluation"]["beam_every_n_steps"])
        run_beam = (
            int(self.global_step) % beam_interval == 0
            or int(self.global_step) >= int(self.cfg["train"]["max_steps"])
        )
        was_training = self.model.training
        self.model.eval()
        rows = []
        with torch.inference_mode():
            loader = self.trainer.datamodule.generation_dataloader(
                canonical_start=canonical_start,
                heldout_start=heldout_start,
            )
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                id_columns = (input_ids == self.vocabulary.id_token_id).nonzero()
                unique_columns = id_columns[:, 1].unique()
                if len(unique_columns) != 1:
                    raise RuntimeError("Generation batch has inconsistent ID boundaries")
                prompts = input_ids[:, : int(unique_columns[0]) + 1]
                greedy = batched_greedy_generate(
                    self.model, prompts, self.vocabulary
                )
                rankings = (
                    batched_beam_generate(
                        self.model,
                        prompts,
                        self.vocabulary,
                        width=int(self.cfg["evaluation"]["beam_width"]),
                    )
                    if run_beam
                    else [[] for _ in greedy]
                )
                rows.extend(
                    {
                        "target": target,
                        "view_type": view_type,
                        "start": float(start),
                        "greedy": result,
                        "beam": ranking,
                    }
                    for target, view_type, start, result, ranking in zip(
                        batch["code"],
                        batch["view_type"],
                        batch["segment_start"],
                        greedy,
                        rankings,
                        strict=True,
                    )
                )
        if was_training:
            self.model.train()
        metrics = {}
        for view_type in ("canonical", "heldout"):
            selected = [row for row in rows if row["view_type"] == view_type]
            start = selected[0]["start"]
            prefix = f"probe/{view_type}/start_{start:g}"
            count = len(selected)
            metrics[f"{prefix}/greedy_top1"] = sum(
                row["greedy"].code == row["target"] for row in selected
            ) / count
            metrics[f"{prefix}/invalid_code_rate"] = sum(
                not row["greedy"].ended_with_eos for row in selected
            ) / count
            if run_beam:
                reciprocal_rank = 0.0
                for width in (1, 5, 10):
                    hits = 0
                    for row in selected:
                        codes = [result.code for result in row["beam"]]
                        hits += int(row["target"] in codes[:width])
                        if width == 10 and row["target"] in codes:
                            reciprocal_rank += 1 / (codes.index(row["target"]) + 1)
                    metrics[f"{prefix}/beam_top{width}"] = hits / count
                metrics[f"{prefix}/beam_mrr"] = reciprocal_rank / count
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=self.global_step)

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
            checkpoint.get("training_corpus_fingerprint")
            != self.training_corpus_fingerprint
        ):
            raise ValueError(
                "Resume checkpoint paired-view corpus fingerprint does not match"
            )
        self.documents_consumed = int(checkpoint.get("documents_consumed", 0))
        self.tokens_consumed = int(checkpoint.get("tokens_consumed", 0))
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
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

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
    module = AudioLMModule(
        cfg,
        datamodule.vocabulary,
        datamodule.tokenizer_spec,
        datamodule.code_mapping_fingerprint,
        datamodule.probe_track_ids,
        datamodule.dataset.track_ids,
        datamodule.training_corpus_fingerprint,
    )
    logger = build_logger(cfg, directory)
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir(cfg),
            filename="step-{step}",
            save_last=True,
            save_top_k=-1,
            every_n_train_steps=int(cfg["train"]["evaluation_interval"]),
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        )
    ]
    if logger:
        callbacks.append(LearningRateMonitor("step"))
    accumulation = int(cfg["trainer"]["accumulate_grad_batches"])
    display_sampler = PairedViewBatchSampler(
        datamodule.dataset,
        tracks_per_microbatch=int(cfg["train"]["tracks_per_microbatch"]),
        seed=seed,
        batch_count_multiple=accumulation,
    )
    optimizer_steps_per_pass = len(display_sampler) // accumulation
    display_max_epochs = math.ceil(
        int(cfg["train"]["max_steps"]) / optimizer_steps_per_pass
    )
    evaluation_batches = int(cfg["train"]["evaluation_interval"]) * accumulation
    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=cfg["trainer"]["strategy"],
        max_epochs=display_max_epochs,
        max_steps=int(cfg["train"]["max_steps"]),
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
