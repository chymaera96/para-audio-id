#!/usr/bin/env python3
"""Freeze a completed tokenizer snapshot and prepare a 25K PSI configuration."""
import argparse
from pathlib import Path

import torch

from para_audio_id.config import load_config, save_config
from para_audio_id.audio_lm.fsq import FrozenFSQTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--output", type=Path, default=Path("configs/fsq_psi.yaml"))
    p.add_argument("--base-config", type=Path, default=Path("configs/fma_large.yaml"))
    p.add_argument("--export-only", action="store_true",
                   help="export the frozen tokenizer without generating a PSI config")
    a = p.parse_args()
    source = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    frozen_path = a.checkpoint.parent / f"tokenizer-step-{source['step']}.ckpt"
    if not frozen_path.exists():
        temporary = frozen_path.with_suffix(".tmp")
        torch.save({"format": source["format"], "config": source["config"],
                    "model": source["model"], "step": source["step"],
                    "split": source["split"], "asset_manifests": source["asset_manifests"]}, temporary)
        temporary.replace(frozen_path)
    frozen = FrozenFSQTokenizer(frozen_path, device="cpu")
    if frozen.model.cfg != source["config"]["model"]:
        raise ValueError("Existing frozen snapshot has different feature/model settings")
    # Verify an existing frozen export corresponds to this source snapshot.
    if any(not torch.equal(source["model"][k], v) for k, v in frozen.model.state_dict().items()):
        raise ValueError("Existing frozen snapshot has different weights")
    print(f"Frozen tokenizer: {frozen_path}")
    print(f"Tokenizer SHA256: {frozen.spec.revision}")
    if a.export_only:
        return
    cfg = load_config(a.base_config)
    cfg["tokenizer"] = dict(kind="fsq", checkpoint=str(frozen_path.resolve()),
                             device="cuda", sample_rate=24000,
                             selected_codebooks=frozen.selected_codebooks)
    cfg["data"]["catalogue"] = source["config"]["data"]["catalogue"]
    cfg["data"]["training_tracks_manifest"] = "/data/home/acw723/para-audio-id/data/training_tracks_25k.json"
    cfg["train"]["run_id"] = "fsq-psi-25k-small-v1"
    save_config(cfg, a.output)
    print(f"PSI configuration: {a.output}")
    print(f"python train.py {a.output} --run-id fsq-psi-25k-small-v1 --wandb-online")


if __name__ == "__main__":
    main()
