from __future__ import annotations

import math
from pathlib import Path

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.fabric.plugins.environments import LightningEnvironment
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from .catalogue import load_catalogue
from .config import save_config
from .data import CatalogueCropDataset
from .model import ParametricAudioIdentifier


def seed_worker(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = __import__("numpy").random.default_rng(info.seed % (2**32))
        if info.dataset.augmenter is not None:
            info.dataset.augmenter.rng = __import__("numpy").random.default_rng(
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
            self.train_set = CatalogueCropDataset(self.cfg, training=True)
            self.val_set = CatalogueCropDataset(self.cfg, training=False)

    def _loader(self, dataset, *, shuffle: bool) -> DataLoader:
        loader_cfg = self.cfg["dataloader"]
        workers = int(loader_cfg["num_workers"])
        return DataLoader(
            dataset,
            batch_size=int(self.cfg["train"]["batch_size"]),
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0 and bool(loader_cfg["persistent_workers"]),
            prefetch_factor=int(loader_cfg["prefetch_factor"]) if workers > 0 else None,
            worker_init_fn=seed_worker,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)


class ParametricIdentifierModule(pl.LightningModule):
    def __init__(self, cfg: dict, model: ParametricAudioIdentifier | None = None):
        super().__init__()
        self.cfg = cfg
        self.network = model or ParametricAudioIdentifier(cfg)
        self.network.muq.freeze_all()
        self.phase_two = False
        self.transition_step = max(
            1, round(int(cfg["train"]["max_steps"]) * float(cfg["train"]["phase1_fraction"]))
        )
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
            f"{prefix}/digit_accuracy",
            digit_accuracy,
            on_step=prefix == "train",
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            f"{prefix}/exact_accuracy",
            exact_accuracy,
            on_step=prefix == "train",
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "validation")

    def on_fit_start(self) -> None:
        if self.global_step >= self.transition_step:
            self._enter_phase_two()
        self._log_parameter_counts()

    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:
        if not self.phase_two and self.global_step >= self.transition_step:
            self._enter_phase_two()

    def _enter_phase_two(self) -> None:
        if self.phase_two:
            return
        fraction = float(self.cfg["train"]["muq_unfreeze_fraction"])
        blocks = self.network.muq.unfreeze_upper_fraction(fraction)
        checkpointing = bool(self.cfg["train"].get("gradient_checkpointing", False))
        if checkpointing and hasattr(self.network.muq.model, "gradient_checkpointing_enable"):
            self.network.muq.model.gradient_checkpointing_enable()
        self.phase_two = True
        self.print(f"Entered phase 2 at step {self.global_step}; unfroze {len(blocks)} MuQ blocks")
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
        if torch.cuda.is_available():
            self.log(
                "system/peak_gpu_memory_bytes",
                float(torch.cuda.max_memory_allocated(self.device)),
                sync_dist=False,
            )
            torch.cuda.reset_peak_memory_stats(self.device)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["phase_two"] = self.phase_two
        checkpoint["catalogue"] = [
            record.__dict__ for record in load_catalogue(self.cfg["data"]["catalogue"])
        ]
        checkpoint["preprocessing"] = {
            "sample_rate": self.cfg["model"]["sample_rate"],
            "query_duration": self.cfg["data"]["query_duration"],
            "quantile_norm": self.cfg["model"]["quantile_norm"],
        }

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


def build_logger(cfg: dict, directory: Path):
    wandb = cfg["train"]["wandb"]
    if not wandb.get("enabled", False):
        return False
    return WandbLogger(
        project=wandb.get("project", "para-audio-id"),
        entity=wandb.get("entity"),
        name=wandb.get("name") or cfg["train"].get("run_id"),
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
            dirpath=directory / "checkpoints",
            filename="step-{step}",
            save_last=True,
            save_top_k=1,
            every_n_train_steps=int(cfg["trainer"]["checkpoint_every_n_steps"]),
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
