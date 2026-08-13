from __future__ import annotations

import argparse
from pathlib import Path

from para_audio_id.audio_lm.training import checkpoint_dir, train
from para_audio_id.config import load_config, with_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the discrete-audio causal LM.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--decoder", choices=("small", "medium"))
    parser.add_argument("--schedule", choices=("noise", "noise-rir"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-path", type=Path)
    args = parser.parse_args()
    base_cfg = load_config(args.config)
    run_cfg = with_overrides(
        base_cfg,
        run_id=args.run_id,
        wandb_online=args.wandb_online,
        devices=args.devices,
    )
    checkpoint = args.ckpt_path
    if args.resume and checkpoint is None:
        checkpoint = checkpoint_dir(run_cfg) / "last.ckpt"
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cfg = with_overrides(
        base_cfg,
        run_id=args.run_id,
        wandb_online=args.wandb_online,
        devices=args.devices,
        decoder=args.decoder,
        schedule=args.schedule,
        checkpoint=checkpoint,
    )
    train(cfg, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
