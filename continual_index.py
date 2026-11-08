"""Prepare and run controlled tc21 catalogue additions without modifying train.py."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from para_audio_id.catalogue import CatalogueRecord
from para_audio_id.audio_lm.checkpoint import load_audio_lm
from para_audio_id.audio_lm.continual import (
    PROTOCOL, AdditionBatchSampler, AdditionCollator, asset_loaders, digest,
    evaluate_tokens, file_digest, full_schedule, prepare, write_json,
)
from para_audio_id.audio_lm.dataset import collate_causal_documents
from para_audio_id.audio_lm.evaluation import _checkpoint_tokenizer
from para_audio_id.audio_lm.losses import degraded_causal_base_losses
from para_audio_id.audio_lm.random_crops import OnlineTrackDataset
from para_audio_id.audio_lm.continual_replay import (
    CachedAdditionCollator, ReplayCache, prepare_replay, preservation_loss,
)


def pilot_data(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Unknown pilot protocol")
    for name, key in [("validation.pt", "validation_sha256"), ("baseline.json", "baseline_sha256")]:
        if file_digest(directory / name) != manifest[key]:
            raise ValueError(f"Prepared {name} has changed")
    cache = torch.load(directory / "validation.pt", map_location="cpu", weights_only=False)
    baseline = json.loads((directory / "baseline.json").read_text())["metrics"]
    if cache["tokenizer_fingerprint"] != manifest["tokenizer_fingerprint"]:
        raise ValueError("Validation tokenizer differs")
    return manifest, cache, baseline


def write_checkpoint(path, model, vocabulary, cfg, source, state, optimizer, scaler):
    # Ordinary model loading works; experiment evaluation uses this runner's
    # immutable old/new query pack rather than pretending to be a 25K profile.
    keys = ["architecture", "tokenizer_spec", "tokenizer_fingerprint", "model_config",
            "query_spec", "batch_spec"]
    payload = {k: deepcopy(source[k]) for k in keys}
    payload.update(vocabulary=vocabulary.to_dict(), hyper_parameters=cfg,
        state_dict={"model." + k: v for k, v in model.state_dict().items()},
        training_track_ids=state["catalogue_ids"], validation_probe=state["probe_ids"],
        code_mapping_fingerprint=state["mapping_sha256"],
        global_step=state["source_global_step"] + state["update_step"],
        schedule_config=cfg["train"]["schedule"],
        resolved_training_profile={"variant": PROTOCOL, "database_size": len(state["catalogue_ids"]),
            "source_profile": source["resolved_training_profile"].get(
                "source_profile", source["resolved_training_profile"])},
        continual_state=state, optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict(),
        rng_state={"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
                   "numpy": np.random.get_state(), "python": random.getstate()})
    payload["hyper_parameters"] = deepcopy(cfg)
    payload["hyper_parameters"]["resolved_training_profile"] = payload["resolved_training_profile"]
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    started = time.perf_counter()
    manifest, cache, baseline = pilot_data(args.pilot)
    if file_digest(manifest["source_checkpoint"]) != manifest["source_sha256"]:
        raise ValueError("The source checkpoint changed; do not overwrite a pilot's source")
    run_id = args.run_id or f"tc21-add1k-{args.arm.lower()}"
    directory = Path(args.output_root) / run_id
    if directory.exists() and not args.resume:
        raise FileExistsError(f"Existing run: {directory}; use --resume or another run ID")
    if args.resume and not (directory / "last.ckpt").exists():
        raise FileNotFoundError(directory / "last.ckpt")
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "last.ckpt" if args.resume else manifest["source_checkpoint"]
    model, vocabulary, cfg, source = load_audio_lm(checkpoint, "cpu")
    cfg = deepcopy(cfg)
    seed = int(manifest["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    model.to(device)
    tokenizer = _checkpoint_tokenizer(source, device)
    old = [CatalogueRecord(**r) for r in manifest["old"]]
    new = [CatalogueRecord(**r) for r in manifest["new"]]
    buffer = set(manifest["replay_ids"])
    replay = [] if args.arm == "A" else old if args.arm in "BDE" else [r for r in old if r.track_id in buffer]
    fixed = None
    if args.arm in "DE":
        fixed = ReplayCache(args.replay_cache or Path(args.pilot) / "replay-cache-v1", manifest)
        replay = [r for r in old if r.track_id in fixed.pairs]
    all_records = old + new
    mapping = __import__("hashlib").sha256(
        "\n".join(f"{r.track_id}:{r.code}" for r in all_records).encode()).hexdigest()
    lr = args.learning_rate
    if lr is None:
        lr = (source["continual_state"]["learning_rate"] if "continual_state" in source else
              float(source["optimizer_states"][0]["param_groups"][0]["lr"]))
    contract = dict(protocol=PROTOCOL, arm=args.arm, pilot_sha256=digest(manifest),
                    learning_rate=lr, steps=args.steps, microbatch=40, accumulation=2,
                    optimizer_initialization="fresh_adamw_moments", seed=seed,
                    precision=str(dtype), source_sha256=manifest["source_sha256"],
                    old_tracks=len(old), new_tracks=len(new), replay_tracks=len(replay),
                    augmentation=full_schedule(source))
    if fixed is not None:
        contract.update(replay_cache_sha256=fixed.sha256,
            replay_protocol=fixed.manifest["protocol"],
            preservation_weight=args.preservation_weight if args.arm == "E" else 0.0,
            preservation="teacher_correct_digits_eos_grouped_kl_v1" if args.arm == "E" else "none",
            replay_uncovered_ids=fixed.manifest["uncovered_ids"])
    contract_sha = digest(contract)
    state = dict(contract_sha256=contract_sha, contract=contract, update_step=0,
                 source_global_step=manifest["source_global_step"], learning_rate=lr,
                 catalogue_ids=[r.track_id for r in all_records], mapping_sha256=mapping,
                 probe_ids=sorted({r["track_id"] for r in cache["rows"]}),
                 new_draws=0, old_draws=0, elapsed_seconds=0.0, seen_new=[],
                 arm=args.arm, pilot_sha256=digest(manifest))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(.9, .95), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=1024)
    if args.resume:
        saved = source["continual_state"]
        if saved["contract_sha256"] != contract_sha:
            raise ValueError("Resume arm, data, optimizer, precision or update budget differs")
        state = saved
        optimizer.load_state_dict(source["optimizer_state"])
        scaler.load_state_dict(source["scaler_state"])
        rng = source["rng_state"]
        torch.set_rng_state(rng["torch"])
        torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    cfg["data"].update(database_size=len(all_records), max_training_tracks=len(all_records))
    cfg["data"]["training_tracks_manifest"] = str(Path(args.pilot).resolve() / "manifest.json")
    cfg["trainer"]["precision"] = "bf16-mixed" if dtype == torch.bfloat16 else "16-mixed"
    cfg["train"].update(run_id=run_id, max_steps=args.steps, learning_rate=lr,
                         schedule=contract["augmentation"], warmup_steps=0,
                         learning_rate_schedule={"policy": "continual_constant_v1"})
    write_json(directory / "experiment.json", contract)
    noise, room = asset_loaders(cfg, tokenizer.sample_rate)
    if noise.manifest() != manifest["noise_manifest"] or room.manifest() != manifest["rir_manifest"]:
        raise ValueError("Degradation assets changed since preparation")
    reserved = manifest["reserved_starts"]
    dataset = OnlineTrackDataset(new + replay)
    sampler = AdditionBatchSampler(len(new), len(replay), arm=args.arm, steps=args.steps,
                                   start=state["update_step"], seed=seed)
    collator_class = CachedAdditionCollator if fixed is not None else AdditionCollator
    collator = collator_class(new, replay, **({"cache": fixed} if fixed is not None else {}),
        audio_root=cfg["data"]["audio_root"],
        noise_assets=noise, rir_assets=room, schedule=contract["augmentation"],
        sample_rate=tokenizer.sample_rate, crop_duration=2.0, past_context_duration=2.0,
        seed=seed, reserved_starts=reserved)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator,
        num_workers=args.workers, multiprocessing_context="spawn" if args.workers else None,
        persistent_workers=bool(args.workers), prefetch_factor=2 if args.workers else None,
        pin_memory=True)
    logger = None
    if args.wandb_online:
        import wandb
        logger = wandb.init(project="para-audio-id", id=run_id, name=run_id,
            group="tc21-add1k-replay-v1" if fixed is not None else "tc21-add1k-v1",
            job_type="continual-index", mode="online",
            resume="must" if args.resume else "never", dir=str(directory), config=contract,
            settings=wandb.Settings(_disable_stats=True))
        logger.define_metric("update_step")
        logger.define_metric("*", step_metric="update_step")
    previous_elapsed = state["elapsed_seconds"]
    new_set = {r.track_id for r in new}
    old_codes = {r.code for r in old}
    new_codes = {r.code for r in new}
    seen = set(state["seen_new"])

    def validation(step):
        before = time.perf_counter()
        values, details = evaluate_tokens(model, vocabulary, cache, device)
        values["val/old_max_drop_pp"] = max(100 * (baseline[k] - v)
            for k, v in values.items() if k.startswith("val/old/"))
        summary = {"update_step": step, **values, "validation_seconds": time.perf_counter() - before}
        for group, opposite in [("old", new_codes), ("new", old_codes)]:
            subset = [r for r in details if r["group"] == group]
            summary[f"{group}_to_other_group"] = sum(r["prediction"] in opposite for r in subset) / len(subset)
        with (directory / "validation.jsonl").open("a") as f:
            f.write(json.dumps(summary) + "\n")
        write_json(directory / f"validation-{step}.json", {"summary": summary, "rows": details})
        if logger:
            logger.log({"update_step": step, **values})
        print("VALIDATION", json.dumps(summary), flush=True)
        model.train()

    if not args.resume:
        validation(0)
    iterator = iter(loader)
    model.train()
    stop = min(args.steps, state["update_step"] + args.stop_after) if args.stop_after else args.steps
    for step in range(state["update_step"] + 1, stop + 1):
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        sums, categories, failures, replacements = Counter(), Counter(), [], []
        for _ in range(2):
            batch = next(iterator)
            metadata = batch["metadata"]
            wave = batch["waveforms"].to(device, non_blocking=True)
            tokens = tokenizer.tokenize(wave)
            if fixed is not None:
                tokens = torch.cat([tokens.long(), batch["cached_tokens"].to(device)], dim=0)
            documents = collate_causal_documents([
                {"audio_tokens": token, "code": row["code"], "track_id": row["track_id"],
                 "document_index": -1, "view_type": "noisy" if row["is_noisy"] else "random_clean"}
                for token, row in zip(tokens, metadata, strict=True)], vocabulary, 512)
            documents = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in documents.items()}
            with torch.autocast("cuda", dtype=dtype):
                logits = model(documents["input_ids"], documents["attention_mask"])
                loss, metrics = degraded_causal_base_losses(logits, documents["input_ids"],
                    documents["audio_target_mask"], documents["id_target_mask"],
                    documents["boundary_target_mask"],
                    torch.tensor([r["is_noisy"] for r in metadata], device=device),
                    id_digit_weight=float(cfg["train"]["id_digit_weight"]))
                if args.arm == "E":
                    fresh_count = len(wave)
                    teacher = torch.cat([torch.zeros(fresh_count, 6, 12, device=device),
                                         batch["teacher"].to(device)], 0)
                    eligible = torch.cat([torch.zeros(fresh_count, device=device, dtype=torch.bool),
                                          batch["eligible"].to(device)], 0)
                    kd = preservation_loss(logits, documents, vocabulary, teacher, eligible,
                                           id_weight=float(cfg["train"]["id_digit_weight"]))
                    # Use the same target-family normalization as the base loss:
                    # preserve five ID digits plus EOS, not the ID boundary token.
                    coefficient = metrics["digit_family_coefficient"] + .5 * metrics["boundary_family_coefficient"]
                    loss = loss + args.preservation_weight * coefficient * kd
                    sums["preservation_loss"] += float(kd.detach()) / 2
                    sums["preservation_eligible_fraction"] += float(eligible.float().mean()) / 2
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at update {step}")
            scaler.scale(loss / 2).backward()
            sums.update({"loss": float(loss.detach()) / 2,
                         "audio_loss": float(metrics["clean_audio_loss"].detach()) / 2,
                         "id_loss": float(metrics["digit_loss"].detach()) / 2})
            categories.update(batch["categories"])
            failures.extend(batch["failures"])
            replacements.extend(batch["replacements"])
            for row in metadata[::2]:
                if row["track_id"] in new_set:
                    state["new_draws"] += 1
                    seen.add(row["track_id"])
                else:
                    state["old_draws"] += 1
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        seconds = time.perf_counter() - step_started
        state["update_step"] = step
        if step % 25 == 0 or step == 1 or failures or replacements:
            row = dict(update_step=step, **sums, step_seconds=seconds,
                       gradient_norm=float(norm), categories=dict(categories), failures=failures,
                       replacements=replacements, new_draws=state["new_draws"], old_draws=state["old_draws"],
                       new_exposures=state["new_draws"] / len(new), unique_new_seen=len(seen),
                       elapsed_seconds=previous_elapsed + time.perf_counter() - started)
            with (directory / "training.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            if logger and (step % 25 == 0 or step == 1):
                logger.log({"update_step": step, "train/loss": sums["loss"],
                    "train/audio_loss": sums["audio_loss"], "train/id_loss": sums["id_loss"],
                    "cost/new_exposures": row["new_exposures"],
                    "cost/gpu_hours": row["elapsed_seconds"] / 3600,
                    "cost/seconds_per_update": seconds})
                if args.arm == "E":
                    logger.log({"update_step": step,
                        "train/preservation_loss": sums["preservation_loss"],
                        "train/preservation_eligible_fraction": sums["preservation_eligible_fraction"]})
            print(f"UPDATE {step}/{args.steps}: loss={sums['loss']:.4f}, {seconds:.3f}s, "
                  f"new draws={state['new_draws']}, old draws={state['old_draws']}", flush=True)
        milestone = step in (1000, 2500, 5000, 10000, stop)
        if step % args.eval_interval == 0 or milestone:
            validation(step)
            state.update(elapsed_seconds=previous_elapsed + time.perf_counter() - started,
                         seen_new=sorted(seen))
            write_checkpoint(directory / "last.ckpt", model, vocabulary, cfg, source, state, optimizer, scaler)
            if milestone:
                # One physical snapshot; last.ckpt remains independently replaceable on later updates.
                import os
                snapshot = directory / f"step-{step}.ckpt"
                if not snapshot.exists():
                    os.link(directory / "last.ckpt", snapshot)
    if logger:
        logger.summary["completed_updates"] = state["update_step"]
        logger.summary["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        logger.finish()
    status = "COMPLETED" if state["update_step"] == args.steps else "STOPPED"
    print(f"{status} arm {args.arm}: {state['update_step']} updates; "
          f"elapsed {state['elapsed_seconds'] / 3600:.3f} hours", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("source")
    prep.add_argument("output")
    prep.add_argument("--new-tracks", type=int, default=1000)
    prep.add_argument("--buffer-tracks", type=int, default=1000)
    prep.add_argument("--monitor-tracks", type=int, default=100)
    fit = sub.add_parser("train")
    fit.add_argument("pilot")
    replay_prep = sub.add_parser("prepare-replay")
    replay_prep.add_argument("pilot")
    replay_prep.add_argument("output")
    replay_prep.add_argument("--workers", type=int, default=8)
    replay_prep.add_argument("--pairs-per-track", type=int, default=4)
    fit.add_argument("--arm", choices=list("ABCDE"), required=True)
    fit.add_argument("--replay-cache")
    fit.add_argument("--preservation-weight", type=float, default=1.0)
    fit.add_argument("--output-root", default="/gpfs/scratch/acw723/para-audio-id/audio-lm-checkpoints")
    fit.add_argument("--run-id")
    fit.add_argument("--steps", type=int, default=10000)
    fit.add_argument("--stop-after", type=int, help="save after N more updates, preserving the total budget")
    fit.add_argument("--eval-interval", type=int, default=1000)
    fit.add_argument("--workers", type=int, default=8)
    fit.add_argument("--learning-rate", type=float)
    fit.add_argument("--resume", action="store_true")
    fit.add_argument("--wandb-online", action="store_true")
    check = sub.add_parser("evaluate")
    check.add_argument("pilot")
    check.add_argument("checkpoint")
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.command == "prepare":
        prepare(args.source, args.output, new_count=args.new_tracks,
                buffer_count=args.buffer_tracks, monitor_count=args.monitor_tracks)
    elif args.command == "prepare-replay":
        if args.pairs_per_track < 1:
            p.error("Positive pair count required")
        pilot_data(args.pilot)
        prepare_replay(args.pilot, args.output, workers=args.workers,
                       pairs_per_track=args.pairs_per_track)
    elif args.command == "train":
        if args.steps < 1 or args.eval_interval < 1 or (args.stop_after is not None and args.stop_after < 1):
            p.error("Positive update counts are required")
        if not np.isfinite(args.preservation_weight) or args.preservation_weight <= 0:
            p.error("Preservation weight must be finite and positive")
        train(args)
    else:
        manifest, cache, _ = pilot_data(args.pilot)
        model, vocabulary, _, source = load_audio_lm(args.checkpoint, "cpu")
        if source["tokenizer_fingerprint"] != manifest["tokenizer_fingerprint"]:
            raise ValueError("Tokenizer mismatch")
        model.cuda()
        metrics, details = evaluate_tokens(model, vocabulary, cache, "cuda")
        print(json.dumps(metrics, indent=2))
        write_json(Path(args.checkpoint).with_suffix(".validation.json"),
                   {"metrics": metrics, "rows": details})


if __name__ == "__main__":
    main()
