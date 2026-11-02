#!/usr/bin/env python3
"""Train or resume the CNN + FSQ clean-target tokenizer pilot."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from para_audio_id.config import load_config, save_config
from para_audio_id.audio_lm.fsq import CNNFSQ, reconstruction_loss
from para_audio_id.audio_lm.fsq_data import CONDITIONS, PairedCrops, assets, source_split


def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def monitor(model, loader, device, output, step):
    model.eval()
    stats = {name: dict(count=0, reconstruction=0., token_agreement=0.,
                        coordinate_agreement=0., coordinate_distance=0.) for name in CONDITIONS}
    q = model.quantizer
    histograms = {name: torch.zeros(q.groups, q.codebook_size, dtype=torch.long, device=device)
                  for name in (*CONDITIONS, "clean_target")}
    example = None
    with torch.inference_mode():
        for batch in loader:
            clean = model.features(batch["clean"].to(device))
            degraded = model.features(batch["degraded"].to(device))
            rec_c, coord_c, ids_c = model(clean)
            rec_d, coord_d, ids_d = model(degraded)
            for idx, name in enumerate(CONDITIONS):
                mask = batch["condition"].to(device) == idx
                n = int(mask.sum())
                if not n:
                    continue
                s = stats[name]
                s["count"] += n
                s["reconstruction"] += float(F.l1_loss(rec_d[mask], clean[mask])) * n
                s["token_agreement"] += float((ids_c[mask] == ids_d[mask]).float().mean()) * n
                s["coordinate_agreement"] += float((coord_c[mask] == coord_d[mask]).float().mean()) * n
                s["coordinate_distance"] += float((coord_c[mask] - coord_d[mask]).abs().mean()) * n
                for g in range(q.groups):
                    histograms[name][g] += torch.bincount(ids_d[mask, g].flatten(), minlength=q.codebook_size)
                    if idx == 0:
                        histograms["clean_target"][g] += torch.bincount(ids_c[mask, g].flatten(), minlength=q.codebook_size)
            if example is None:
                example = {"clean_mel": clean[:8].cpu(), "degraded_mel": degraded[:8].cpu(),
                           "reconstructed_clean": rec_c[:8].cpu(),
                           "reconstructed_degraded": rec_d[:8].cpu(),
                           "clean_ids": ids_c[:8].cpu(), "degraded_ids": ids_d[:8].cpu(),
                           "conditions": batch["condition"][:8], "track_ids": batch["track_id"][:8]}
    metrics = {}
    for name, s in stats.items():
        metrics[f"monitor/{name}/queries"] = s["count"]
        for key, total in s.items():
            if key != "count":
                metrics[f"monitor/{name}/{key}"] = total / max(1, s["count"])
    for name, counts in histograms.items():
        p = counts.float() / counts.sum(-1, keepdim=True).clamp_min(1)
        entropy = -(p * p.clamp_min(1e-20).log()).sum(-1)
        metrics[f"monitor/{name}/effective_tokens"] = float(entropy.exp().mean())
        metrics[f"monitor/{name}/used_tokens"] = float((counts > 0).float().sum(-1).mean())
    atomic_save({"step": step, "counts": {k: v.cpu() for k, v in histograms.items()},
                 "examples": example}, output / "monitor-latest.pt")
    model.train()
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb-online", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sanity-steps", type=int, default=0,
                        help="short run with its own run ID, no W&B, and at most 16 monitor tracks")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="stop and save after this many additional updates (keeps the full LR schedule)")
    args = parser.parse_args()
    if args.stop_after < 0 or args.sanity_steps < 0:
        parser.error("Step limits must be non-negative")
    cfg = load_config(args.config)
    if args.run_id:
        cfg["train"]["run_id"] = args.run_id
    t = cfg["train"]
    torch.set_num_threads(t["cpu_threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    split = source_split(cfg)
    print(f"Source split: {len(split['training'])} training / {len(split['monitor'])} monitoring", flush=True)
    print(f"Monitoring crop preparation: {len(split['monitor_preparation_audit'])} audited retries/replacements", flush=True)
    if args.prepare_only:
        return
    if args.sanity_steps:
        if args.resume or args.wandb_online:
            parser.error("--sanity-steps uses a separate fresh run without W&B")
        t["run_id"] += "-sanity"
    output = Path(t["output_root"]) / t["run_id"]
    last = output / "last.ckpt"
    if last.exists() and not args.resume:
        raise FileExistsError(f"Run already has a checkpoint: use --resume or a new --run-id ({last})")
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = CNNFSQ(cfg["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    step = 0
    wandb_id = None
    if args.resume:
        payload = torch.load(last, map_location="cpu", weights_only=False)
        if payload["config"] != cfg:
            raise ValueError("Resume configuration differs from checkpoint")
        if payload["split"] != split:
            raise ValueError("Resume source split differs")
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        step = payload["step"]
        wandb_id = payload.get("wandb_id")
    save_config(cfg, output / "config.yaml")
    noise, rir = assets(cfg)
    asset_manifests = {"noise": noise.manifest(), "rir": rir.manifest()}
    if args.resume and payload["asset_manifests"] != asset_manifests:
        raise ValueError("Augmentation assets changed since checkpoint")
    (output / "assets.json").write_text(json.dumps(asset_manifests, indent=2))
    monitor_records = split["monitor"][:16] if args.sanity_steps else split["monitor"]
    monitoring = PairedCrops(cfg, monitor_records, noise, rir, monitor=True,
                             monitor_keys=split["monitor_crop_keys"][:len(monitor_records)])
    kwargs = dict(batch_size=t["batch_size"], num_workers=t["workers"], pin_memory=device.type == "cuda")
    if t["workers"]:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    training = PairedCrops(cfg, split["training"], noise, rir,
                           start_index=step * t["batch_size"])
    loader = DataLoader(training, **kwargs)
    monitor_loader = DataLoader(monitoring, **kwargs)
    # Cache fixed draws in host memory: ~768 MB for 500 tracks x 4 conditions.
    print("Preparing fixed monitoring crops...", flush=True)
    monitor_batches = list(monitor_loader)
    del monitor_loader
    tracker = None
    if args.wandb_online:
        import wandb
        id_path = output / "wandb-id.txt"
        if id_path.exists():
            existing_id = id_path.read_text().strip()
            if wandb_id and existing_id != wandb_id:
                raise ValueError("Saved W&B ID differs from the checkpoint")
            wandb_id = existing_id
        tracker = wandb.init(project=t["wandb_project"], name=t["run_id"],
                             id=wandb_id, resume="must" if wandb_id else None, config=cfg)
        wandb_id = tracker.id
        id_path.write_text(wandb_id + "\n")
    enabled = device.type == "cuda" and t["precision"] == "bf16"
    start = time.perf_counter()
    start_step = step
    end_step = min(t["max_steps"], step + args.sanity_steps) if args.sanity_steps else t["max_steps"]
    if args.stop_after:
        end_step = min(end_step, step + args.stop_after)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}; "
          f"{model.frame_rate:g} frames/s; {model.quantizer.groups} group(s); "
          f"{model.quantizer.codebook_size} symbols/group", flush=True)

    def log(metrics):
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"step": step, **metrics}) + "\n")
        if tracker:
            tracker.log(metrics, step=step)

    def save():
        atomic_save({"format": "cnn_fsq_tokenizer_v1", "config": cfg, "step": step,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "split": split, "asset_manifests": asset_manifests,
                     "wandb_id": wandb_id}, last)

    model.train()
    for batch in loader:
        if step >= end_step:
            break
        update_start = time.perf_counter()
        if step < t["warmup_steps"]:
            lr = t["learning_rate"] * (step + 1) / t["warmup_steps"]
        else:
            progress = (step - t["warmup_steps"]) / max(1, t["max_steps"] - t["warmup_steps"])
            lr = t["minimum_learning_rate"] + .5 * (t["learning_rate"] - t["minimum_learning_rate"]) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        clean = model.features(batch["clean"].to(device, non_blocking=True))
        degraded = model.features(batch["degraded"].to(device, non_blocking=True))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled):
            reconstructed, coordinates, _ = model(torch.cat((clean, degraded)))
            rec_c, rec_d = reconstructed.chunk(2)
            q_c, q_d = coordinates.chunk(2)
            rec_loss = reconstruction_loss(clean, rec_c, rec_d)
            consistency = (q_c - q_d).abs().mean()
            loss = rec_loss + t["consistency_weight"] * consistency
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % t["log_interval"] == 0 or step == end_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics = {"train/loss": float(loss.detach()), "train/reconstruction": float(rec_loss.detach()),
                       "train/consistency": float(consistency.detach()), "train/lr": lr,
                       "train/gradient_norm": float(grad),
                       "train/replaced_crops": int(batch["replacements"].sum()),
                       "timing/seconds_per_update": (time.perf_counter() - start) / (step - start_step),
                       "timing/compute_seconds": time.perf_counter() - update_start}
            log(metrics)
            print(f"step {step}/{end_step}: loss={metrics['train/loss']:.4f}; "
                  f"{metrics['timing/seconds_per_update']:.3f} s/update", flush=True)
        if step % t["monitor_interval"] == 0 or step == end_step:
            monitor_start = time.perf_counter()
            metrics = monitor(model, monitor_batches, device, output, step)
            metrics["timing/monitor_seconds"] = time.perf_counter() - monitor_start
            log(metrics)
            if tracker:
                snapshot = torch.load(output / "monitor-latest.pt", map_location="cpu", weights_only=False)
                examples = snapshot["examples"]
                images = []
                for i, condition in enumerate(examples["conditions"][:4]):
                    mel_panel = torch.cat([examples[key][i] for key in (
                        "clean_mel", "degraded_mel", "reconstructed_clean", "reconstructed_degraded")], dim=-1)
                    images.append(wandb.Image(mel_panel.numpy(), caption=(
                        f"{CONDITIONS[int(condition)]}: clean target | input | clean reconstruction | degraded reconstruction")))
                tracker.log({"monitor/reconstructions": images}, step=step)
            print(f"Monitor: {json.dumps(metrics)}", flush=True)
        if step % t["checkpoint_interval"] == 0 or step == end_step:
            save()
            if not args.sanity_steps:
                atomic_save(torch.load(last, map_location="cpu", weights_only=False),
                            output / f"step-{step}.ckpt")
    if tracker:
        tracker.finish()
    print(f"Saved {last}", flush=True)


if __name__ == "__main__":
    main()
