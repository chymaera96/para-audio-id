from __future__ import annotations

import math
import json
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from .catalogue import load_catalogue
from .config import save_config
from .data import (
    SEGMENT_POLICY_VERSION,
    CatalogueSegmentDataset,
    IdentityGroupedBatchSampler,
)
from .model import ParametricAudioIdentifier


def phase_two_due(completed_exposures: int, phase1_exposures: int) -> bool:
    if phase1_exposures < 0:
        raise ValueError("phase1_exposures cannot be negative")
    return completed_exposures >= phase1_exposures


def seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        dataset = getattr(info.dataset, "dataset", info.dataset)
        dataset.rng = np.random.default_rng(info.seed % (2**32))
        if dataset.augmenter is not None:
            dataset.augmenter.rng = np.random.default_rng(
                (info.seed + 1) % (2**32)
            )


class CatalogueDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is None:
            self.train_set = CatalogueSegmentDataset(self.cfg, training=True)
            self.val_set = CatalogueSegmentDataset(self.cfg, training=False)
            count = min(
                int(self.cfg["evaluation"]["probe_tracks"]),
                len(self.val_set.valid_record_indices()),
            )
            records = np.asarray(self.val_set.valid_record_indices(), dtype=np.int64)
            np.random.default_rng(int(self.cfg["train"]["seed"]) + 4242).shuffle(records)
            self.probe_indices = [
                self.val_set.segments_by_record[int(record)][
                    len(self.val_set.segments_by_record[int(record)]) // 2
                ]
                for record in records[:count]
            ]

    def _loader(self, dataset, *, batch_size: int) -> DataLoader:
        loader_cfg = self.cfg["dataloader"]
        workers = int(loader_cfg["num_workers"])
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=int(loader_cfg["prefetch_factor"]) if workers > 0 else None,
            worker_init_fn=seed_worker,
        )

    def train_dataloader(self) -> DataLoader:
        loader_cfg = self.cfg["dataloader"]
        workers = int(loader_cfg["num_workers"])
        sampler = IdentityGroupedBatchSampler(
            self.train_set,
            songs_per_batch=int(self.cfg["train"]["songs_per_batch"]),
            views_per_song=int(self.cfg["train"]["views_per_song"]),
            seed=int(self.cfg["train"]["seed"]),
            world_size=self.trainer.world_size,
            rank=self.trainer.global_rank,
            exposure=self.trainer.current_epoch,
        )
        self.train_batch_sampler = sampler
        return DataLoader(
            self.train_set,
            batch_sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0 and bool(loader_cfg["persistent_workers"]),
            prefetch_factor=int(loader_cfg["prefetch_factor"]) if workers > 0 else None,
            worker_init_fn=seed_worker,
        )

    def val_dataloader(self) -> DataLoader:
        return self.probe_dataloader()

    def probe_dataloader(self) -> DataLoader:
        return self._loader(
            Subset(self.val_set, self.probe_indices),
            batch_size=int(self.cfg["evaluation"].get("probe_batch_size", 16)),
        )

    def set_exposure(self, exposure: int) -> None:
        if hasattr(self, "train_batch_sampler"):
            self.train_batch_sampler.set_epoch(exposure)

    def probe_metadata(self) -> list[dict]:
        return [
            {
                "track_id": self.val_set.records[
                    self.val_set.segments[index].record_index
                ].track_id,
                "start": self.val_set.segments[index].start,
            }
            for index in self.probe_indices
        ]


class ParametricIdentifierModule(pl.LightningModule):
    def __init__(self, cfg: dict, model: ParametricAudioIdentifier | None = None):
        super().__init__()
        self.cfg = cfg
        self.network = model or ParametricAudioIdentifier(cfg)
        self.network.muq.freeze_all()
        self.phase_two = False
        self.phase1_exposures = int(cfg["train"]["phase1_exposures"])
        self.save_hyperparameters(cfg)

    def forward(self, audio: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.network(audio, targets)

    def _step(self, batch: dict, prefix: str) -> torch.Tensor:
        logits = self(batch["audio"], batch["target"])
        loss = F.cross_entropy(logits.flatten(0, 1), batch["target"].flatten())
        predictions = logits.argmax(dim=-1)
        digit_accuracy = (predictions == batch["target"]).float().mean()
        exact_accuracy = (predictions == batch["target"]).all(dim=1).float().mean()
        self.log(f"{prefix}/loss", loss, on_step=prefix == "train", on_epoch=True, sync_dist=True)
        self.log(
            f"{prefix}/teacher_forced_digit_accuracy",
            digit_accuracy,
            on_step=prefix == "train",
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/teacher_forced_exact_accuracy",
            exact_accuracy,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        global_identities = torch.tensor(
            float(len(set(batch["track_id"]))), device=self.device
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(global_identities, op=torch.distributed.ReduceOp.SUM)
        self.log(
            "batch/global_distinct_identities",
            global_identities,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "batch/views_per_identity",
            float(self.cfg["train"]["views_per_song"]),
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        applied = [json.loads(value) for value in batch["augmentation"]]
        clean = sum(not value for value in applied) / len(applied)
        self.log("augmentation/clean_fraction", clean, on_step=True, sync_dist=True)
        for name in ("background", "room_ir", "microphone_ir"):
            key = "background_snr_db" if name == "background" else name
            fraction = sum(key in value for value in applied) / len(applied)
            self.log(f"augmentation/{name}_fraction", fraction, on_step=True, sync_dist=True)
        combined = sum(len(value) > 1 for value in applied) / len(applied)
        self.log("augmentation/combined_fraction", combined, on_step=True, sync_dist=True)
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "validation")

    def on_fit_start(self) -> None:
        if phase_two_due(self.current_epoch, self.phase1_exposures):
            self._enter_phase_two()
        self._log_parameter_counts()

    def on_train_epoch_start(self) -> None:
        self.trainer.datamodule.set_exposure(self.current_epoch)
        if not self.phase_two and phase_two_due(
            self.current_epoch, self.phase1_exposures
        ):
            self._enter_phase_two()
            self._run_autoregressive_probe(include_beam=True)
        self.log("exposure/index", float(self.current_epoch), sync_dist=False)
        self.log("model/muq_finetuning", float(self.phase_two), sync_dist=False)
        self.log(
            "model/muq_backbone_eval_mode",
            float(not self.network.muq.model.training),
            sync_dist=False,
        )
        self.log(
            "model/muq_upper_blocks_training",
            float(
                bool(self.network.muq._upper_blocks)
                and all(block.training for block in self.network.muq._upper_blocks)
            ),
            sync_dist=False,
        )

    def _enter_phase_two(self) -> None:
        if self.phase_two:
            return
        fraction = float(self.cfg["train"]["muq_unfreeze_fraction"])
        blocks = self.network.muq.unfreeze_upper_fraction(fraction)
        checkpointing = bool(self.cfg["train"].get("gradient_checkpointing", False))
        if checkpointing and hasattr(self.network.muq.model, "gradient_checkpointing_enable"):
            self.network.muq.model.gradient_checkpointing_enable()
        self.phase_two = True
        self.print(
            f"Entered phase 2 at exposure {self.current_epoch}, step {self.global_step}; "
            f"unfroze {len(blocks)} MuQ blocks"
        )
        self._log_parameter_counts()

    def _log_parameter_counts(self) -> None:
        total = sum(parameter.numel() for parameter in self.network.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.network.parameters() if parameter.requires_grad
        )
        # Lightning does not permit self.log() from on_fit_start. These are
        # one-time metadata metrics, so send them directly through the logger.
        if self.trainer.is_global_zero and self.logger is not None:
            self.logger.log_metrics(
                {
                    "model/total_parameters": float(total),
                    "model/trainable_parameters": float(trainable),
                },
                step=self.global_step,
            )

    def on_train_epoch_end(self) -> None:
        completed = int(self.current_epoch) + 1
        self.log("exposure/completed", float(completed), sync_dist=False)
        greedy_every = int(self.cfg["evaluation"]["greedy_every_n_exposures"])
        beam_every = int(self.cfg["evaluation"]["beam_every_n_exposures"])
        if completed % greedy_every == 0:
            self._run_autoregressive_probe(include_beam=completed % beam_every == 0)
        if torch.cuda.is_available():
            self.log(
                "system/peak_gpu_memory_bytes",
                float(torch.cuda.max_memory_allocated(self.device)),
                sync_dist=False,
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    def _run_autoregressive_probe(self, *, include_beam: bool) -> None:
        if not self.trainer.is_global_zero or self.trainer.sanity_checking:
            return
        was_training = self.network.training
        self.network.eval()
        total = 0
        greedy_correct = 0
        beam_hits = {1: 0, 5: 0, 10: 0}
        with torch.inference_mode():
            for batch in self.trainer.datamodule.probe_dataloader():
                audio = batch["audio"].to(self.device)
                codes = list(batch["code"])
                greedy = self.network.greedy_decode(audio)
                greedy_correct += sum(a == b for a, b in zip(codes, greedy, strict=True))
                if include_beam:
                    rankings = self.network.beam_decode(audio, width=10)
                    for target, ranking in zip(codes, rankings, strict=True):
                        predicted = [item.code for item in ranking]
                        for width in beam_hits:
                            beam_hits[width] += int(target in predicted[:width])
                total += len(codes)
        if was_training:
            self.network.train()
        metrics = {"probe/greedy_exact_accuracy": greedy_correct / max(1, total)}
        if include_beam:
            metrics.update(
                {
                    f"probe/beam_top{width}": hits / max(1, total)
                    for width, hits in beam_hits.items()
                }
            )
        if self.logger is not None:
            self.logger.log_metrics(metrics, step=self.global_step)

    @staticmethod
    def _gradient_norm(parameters) -> float:
        gradients = [
            parameter.grad.detach().norm(2)
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            return 0.0
        return float(torch.stack(gradients).norm(2))

    def on_before_optimizer_step(self, optimizer) -> None:
        interval = int(self.cfg["trainer"]["log_every_n_steps"])
        if self.global_step % interval:
            return
        groups = optimizer.param_groups
        self.log("learning_rate/decoder", float(groups[0]["lr"]), sync_dist=False)
        self.log("learning_rate/muq", float(groups[1]["lr"]), sync_dist=False)
        self.log(
            "gradient_norm/projection",
            self._gradient_norm(self.network.digit_decoder.projection.parameters()),
            sync_dist=False,
        )
        self.log(
            "gradient_norm/decoder",
            self._gradient_norm(self.network.digit_decoder.decoder.parameters()),
            sync_dist=False,
        )
        self.log(
            "gradient_norm/muq_trainable",
            self._gradient_norm(
                parameter
                for parameter in self.network.muq.parameters()
                if parameter.requires_grad
            ),
            sync_dist=False,
        )

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["catalogue"] = [
            record.__dict__ for record in load_catalogue(self.cfg["data"]["catalogue"])
        ]
        checkpoint["preprocessing"] = {
            "sample_rate": self.cfg["model"]["sample_rate"],
            "query_duration": self.cfg["data"]["query_duration"],
            "quantile_norm": self.cfg["model"]["quantile_norm"],
            "segment_stride": self.cfg["data"]["segment_stride"],
            "segment_policy_version": SEGMENT_POLICY_VERSION,
        }
        checkpoint["phase_two"] = self.phase_two
        checkpoint["completed_exposures"] = int(self.current_epoch) + 1
        checkpoint["validation_probe"] = self.trainer.datamodule.probe_metadata()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        if checkpoint.get("phase_two", False):
            self.network.muq.unfreeze_upper_fraction(
                float(self.cfg["train"]["muq_unfreeze_fraction"])
            )
            if (
                self.cfg["train"].get("gradient_checkpointing", False)
                and hasattr(self.network.muq.model, "gradient_checkpointing_enable")
            ):
                self.network.muq.model.gradient_checkpointing_enable()
            self.phase_two = True

    def configure_optimizers(self):
        train_cfg = self.cfg["train"]
        encoder_params = list(self.network.muq.parameters())
        decoder_params = list(self.network.digit_decoder.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": decoder_params, "lr": float(train_cfg["decoder_lr"])},
                {"params": encoder_params, "lr": float(train_cfg["muq_lr"])},
            ],
            weight_decay=float(train_cfg["weight_decay"]),
        )
        max_steps = int(train_cfg["max_steps"])
        warmup = int(train_cfg["warmup_steps"])

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, max_steps - warmup)
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
    base = Path(cfg["train"].get("checkpoint_dir") or output_dir(cfg) / "checkpoints")
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
    pl.seed_everything(int(cfg["train"]["seed"]), workers=True)
    directory = output_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    save_config(cfg, directory / "config.yaml")
    logger = build_logger(cfg, directory)
    callbacks: list[pl.Callback] = [
        ModelCheckpoint(
            dirpath=checkpoint_dir(cfg),
            filename="step-{step}",
            save_last=True,
            save_top_k=1,
            every_n_epochs=int(cfg["trainer"]["checkpoint_every_n_epochs"]),
            save_on_train_epoch_end=True,
        )
    ]
    if logger:
        callbacks.append(LearningRateMonitor("step"))
    devices = cfg["trainer"]["devices"]
    strategy = cfg["trainer"]["strategy"]
    plugins = [LightningEnvironment()] if int(devices) == 1 else []
    trainer = pl.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=devices,
        strategy=strategy,
        plugins=plugins,
        max_steps=int(cfg["train"]["max_steps"]),
        precision=cfg["trainer"]["precision"],
        accumulate_grad_batches=int(cfg["trainer"]["accumulate_grad_batches"]),
        gradient_clip_val=float(cfg["train"]["gradient_clip_val"]),
        logger=logger,
        callbacks=callbacks,
        default_root_dir=directory,
        log_every_n_steps=int(cfg["trainer"]["log_every_n_steps"]),
        val_check_interval=int(cfg["trainer"]["val_check_interval"]),
        limit_val_batches=int(cfg["trainer"]["limit_val_batches"]),
        use_distributed_sampler=True,
    )
    trainer.fit(
        ParametricIdentifierModule(cfg),
        datamodule=CatalogueDataModule(cfg),
        ckpt_path=checkpoint,
    )
