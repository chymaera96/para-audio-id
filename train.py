from __future__ import annotations

import argparse
from pathlib import Path

from para_audio_id.config import load_config, with_overrides
from para_audio_id.training import checkpoint_dir, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the parametric audio identifier.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--devices",
        type=int,
        default=None,
        help="Number of GPUs. Overrides trainer.devices and uses Lightning's automatic strategy.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ckpt-path", type=Path)
    args = parser.parse_args()
    cfg = with_overrides(
        load_config(args.config),
        run_id=args.run_id,
        wandb_online=args.wandb_online,
        devices=args.devices,
    )
    checkpoint = args.ckpt_path
    if args.resume and checkpoint is None:
        checkpoint = checkpoint_dir(cfg) / "last.ckpt"
    if checkpoint is not None and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    train(cfg, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
