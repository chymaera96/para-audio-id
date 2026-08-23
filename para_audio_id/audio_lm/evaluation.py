from __future__ import annotations

from contextlib import nullcontext
import csv
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..audio import BadFileRegistry, load_audio
from ..catalogue import load_catalogue
from .checkpoint import load_audio_lm
from .dataset import CachedPositionDataset, collate_causal_documents
from .generation import (
    batched_beam_generate,
    batched_greedy_generate,
    batched_joint_beam_generate,
    beam_generate,
    greedy_generate,
    prompt_from_audio_tokens,
    prompts_from_audio_tokens,
)
from .losses import causal_audio_id_losses
from .noise import BackgroundNoiseAssets, mix_background_noise, stable_uint64
from .profiles import historical_checkpoint_profile
from .random_crops import RandomEvaluationCollator, RandomEvaluationDataset
from .rir import RoomImpulseResponseAssets, convolve_full_wet
from .tokenizer import MuQRVQTokenizer
from .token_store import TokenStoreIndex


def select_checkpoint_cohort(
    checkpoint: dict,
    *,
    cohort: str,
    expected_tracks: int | None = None,
    max_tracks: int | None = None,
    sample_tracks: int | None = None,
    sample_seed: int = 1337,
) -> list[str]:
    key = {"probe": "validation_probe", "training": "training_track_ids"}.get(cohort)
    if key is None:
        raise ValueError(f"Unknown evaluation cohort {cohort!r}")
    if key not in checkpoint:
        raise ValueError(f"Checkpoint does not contain the {cohort} cohort")
    track_ids = list(checkpoint[key])
    if len(track_ids) != len(set(track_ids)):
        raise ValueError(f"Checkpoint {cohort} cohort contains duplicate track IDs")
    if expected_tracks is not None and len(track_ids) != expected_tracks:
        raise ValueError(
            f"Expected exactly {expected_tracks} {cohort} tracks, checkpoint has "
            f"{len(track_ids)}"
        )
    if max_tracks is not None and sample_tracks is not None:
        raise ValueError("max_tracks and sample_tracks are mutually exclusive")
    if sample_tracks is not None:
        if not 1 <= sample_tracks <= len(track_ids):
            raise ValueError(
                f"sample_tracks must be between 1 and {len(track_ids)}"
            )
        track_ids = random.Random(sample_seed).sample(track_ids, sample_tracks)
    elif max_tracks is not None:
        if max_tracks < 1:
            raise ValueError("max_tracks must be positive")
        track_ids = track_ids[:max_tracks]
    return track_ids


def _checkpoint_tokenizer(checkpoint: dict, device: str) -> MuQRVQTokenizer:
    spec = checkpoint["tokenizer_spec"]
    tokenizer = MuQRVQTokenizer(
        spec["model_name"],
        revision=spec["revision"],
        selected_codebooks=int(spec["selected_codebooks"]),
        sample_rate=int(spec["sample_rate"]),
        device=device,
        lightweight=True,
    )
    if tokenizer.spec.fingerprint != checkpoint["tokenizer_fingerprint"]:
        raise ValueError("Loaded MuQ tokenizer does not match the checkpoint")
    return tokenizer


def _evaluate_tc6_monitor_manifest(
    model,
    vocabulary,
    cfg: dict,
    checkpoint: dict,
    *,
    output: str | Path,
    device: str,
    beam_width: int | None,
) -> dict:
    manifest = checkpoint["monitor_recipes"]
    tokenizer = _checkpoint_tokenizer(checkpoint, device)
    noise_cfg = cfg["data"].get("background_noise")
    assets = (
        BackgroundNoiseAssets(
            noise_cfg["training_root"],
            noise_cfg["validation_root"],
            sample_rate=tokenizer.sample_rate,
            samples=round(
                tokenizer.sample_rate * float(cfg["data"]["segment_duration"])
            ),
        )
        if noise_cfg is not None
        else None
    )
    rir_cfg = cfg["data"].get("room_ir")
    rir_assets = (
        RoomImpulseResponseAssets(
            rir_cfg["training_root"],
            rir_cfg["validation_root"],
            sample_rate=tokenizer.sample_rate,
        )
        if rir_cfg is not None
        else None
    )
    loader = DataLoader(
        RandomEvaluationDataset(manifest),
        batch_size=int(cfg["evaluation"]["generation_batch_size"]),
        shuffle=False,
        collate_fn=RandomEvaluationCollator(
            audio_root=cfg["data"]["audio_root"],
            noise_assets=assets,
            rir_assets=rir_assets,
            sample_rate=tokenizer.sample_rate,
            past_context_duration=(
                float(rir_cfg["past_context_duration"]) if rir_cfg else 0.0
            ),
            seed=int(cfg["train"]["seed"]) + 1771,
        ),
    )
    snrs = (
        [float(value) for value in cfg["evaluation"]["noise_snr_db"]]
        if assets is not None
        else []
    )
    rows = []
    skipped_queries = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="tc6-compatible evaluation"):
            skipped_queries.extend(batch["skipped"])
            clean = batch["clean_waveforms"].to(device)
            if not len(clean):
                continue
            variants = [("clean", None, clean, list(range(len(clean))))]
            room = batch["rir_waveforms"].to(device)
            if rir_assets is not None:
                variants.append(("rir", None, room, list(range(len(room)))))
            for snr in snrs:
                noise = batch["noise_waveforms"].to(device)
                requested = torch.full((len(clean),), snr, device=device)
                mixed, valid = mix_background_noise(clean, noise, requested)
                valid_indices = valid.nonzero(as_tuple=False).flatten().tolist()
                for index in (~valid).nonzero(as_tuple=False).flatten().tolist():
                    skipped_queries.append(
                        {
                            "track_id": batch["track_id"][index],
                            "code": batch["code"][index],
                            "view_type": batch["view_type"][index],
                            "start": batch["start"][index],
                            "snr_db": snr,
                            "error": "noise mixing produced invalid audio",
                        }
                    )
                if valid_indices:
                    variants.append(("noise", snr, mixed[valid], valid_indices))
                if rir_assets is None:
                    continue
                contexts = torch.from_numpy(
                    np.stack([item[0] for item in batch["noise_rir_inputs"]])
                ).to(device)
                context_noise = torch.from_numpy(
                    np.stack([item[1] for item in batch["noise_rir_inputs"]])
                ).to(device)
                mixed_context, context_valid = mix_background_noise(
                    contexts,
                    context_noise,
                    torch.full((len(contexts),), snr, device=device),
                )
                past_samples = round(
                    float(rir_cfg["past_context_duration"]) * tokenizer.sample_rate
                )
                output_samples = round(
                    float(cfg["data"]["segment_duration"]) * tokenizer.sample_rate
                )
                combined = []
                combined_indices = []
                for index in context_valid.nonzero(as_tuple=False).flatten().tolist():
                    try:
                        combined.append(
                            convolve_full_wet(
                                mixed_context[index].cpu().numpy(),
                                batch["noise_rir_inputs"][index][2],
                                past_context_samples=past_samples,
                                output_samples=output_samples,
                            )
                        )
                        combined_indices.append(index)
                    except Exception as exc:
                        skipped_queries.append(
                            {
                                "track_id": batch["track_id"][index],
                                "code": batch["code"][index],
                                "view_type": batch["view_type"][index],
                                "start": batch["start"][index],
                                "snr_db": snr,
                                "condition": "noise_rir",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                if combined:
                    variants.append(
                        (
                            "noise_rir",
                            snr,
                            torch.from_numpy(np.stack(combined)).to(device),
                            combined_indices,
                        )
                    )
            for condition, snr, waveforms, indices in variants:
                tokens = tokenizer.tokenize(waveforms)
                prompts = torch.stack(
                    [
                        prompt_from_audio_tokens(token, vocabulary)
                        for token in tokens
                    ]
                )
                greedy = batched_greedy_generate(model, prompts, vocabulary)
                beams = (
                    batched_beam_generate(
                        model, prompts, vocabulary, width=beam_width
                    )
                    if beam_width is not None
                    else [[] for _ in greedy]
                )
                rows.extend(
                    {
                        "track_id": track_id,
                        "code": code,
                        "view_type": view_type,
                        "start": start,
                        "condition": condition,
                        "snr_db": snr,
                        "rir_path": (
                            batch["rir_path"][index]
                            if condition in {"rir", "noise_rir"}
                            else None
                        ),
                        "greedy": result.code,
                        "greedy_ended_with_eos": result.ended_with_eos,
                        "beam": [
                            {
                                "code": candidate.code,
                                "log_probability": candidate.log_probability,
                                "ended_with_eos": candidate.ended_with_eos,
                            }
                            for candidate in ranking
                        ],
                    }
                    for index, track_id, code, view_type, start, result, ranking in zip(
                        indices,
                        [batch["track_id"][index] for index in indices],
                        [batch["code"][index] for index in indices],
                        [batch["view_type"][index] for index in indices],
                        [batch["start"][index] for index in indices],
                        greedy,
                        beams,
                        strict=True,
                    )
                )
    clean_rows = [row for row in rows if row["condition"] == "clean"]
    noisy_rows = [row for row in rows if row["condition"] == "noise"]
    rir_rows = [row for row in rows if row["condition"] == "rir"]
    noise_rir_rows = [row for row in rows if row["condition"] == "noise_rir"]
    metrics = {
        "cohort": "tc11_fixed_probe",
        "selected_tracks": len({row["track_id"] for row in manifest}),
        "generation_protocol": "five_autoregressive_digits_then_eos",
        "clean": _generation_metrics(clean_rows),
        "noise": _generation_metrics(noisy_rows),
        "rir": _generation_metrics(rir_rows),
        "noise_rir": _generation_metrics(noise_rir_rows),
        "by_view": {
            view_type: {
                "clean": _generation_metrics(
                    [
                        row
                        for row in clean_rows
                        if row["view_type"] == view_type
                    ]
                ),
                "noise": _generation_metrics(
                    [
                        row
                        for row in noisy_rows
                        if row["view_type"] == view_type
                    ]
                ),
            }
            for view_type in ("canonical", "shifted", "heldout")
        },
        "rir_by_view": {
            view_type: _generation_metrics(
                [row for row in rir_rows if row["view_type"] == view_type]
            )
            for view_type in ("canonical", "shifted", "heldout")
        },
        "noise_rir_by_view": {
            view_type: _generation_metrics(
                [
                    row
                    for row in noise_rir_rows
                    if row["view_type"] == view_type
                ]
            )
            for view_type in ("canonical", "shifted", "heldout")
        },
        "by_snr": {
            f"{snr:g}": _generation_metrics(
                [
                    row
                    for row in noisy_rows
                    if math.isclose(float(row["snr_db"]), snr)
                ]
            )
            for snr in snrs
        },
        "noise_rir_by_snr": {
            f"{snr:g}": _generation_metrics(
                [
                    row
                    for row in noise_rir_rows
                    if math.isclose(float(row["snr_db"]), snr)
                ]
            )
            for snr in snrs
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "queries": rows,
                "skipped_queries": skipped_queries,
            },
            indent=2,
        )
        + "\n"
    )
    return metrics


def _generation_metrics(rows: list[dict]) -> dict:
    count = len(rows)
    if not count:
        return {"queries": 0}
    metrics = {
        "queries": count,
        "greedy_top1": sum(row["greedy"] == row["code"] for row in rows) / count,
        "invalid_code_rate": sum(not row["greedy_ended_with_eos"] for row in rows)
        / count,
    }
    if rows[0]["beam"]:
        reciprocal_rank = 0.0
        for width in (1, 5, 10):
            hits = 0
            for row in rows:
                codes = [result["code"] for result in row["beam"]]
                hits += int(row["code"] in codes[:width])
                if width == 10 and row["code"] in codes:
                    reciprocal_rank += 1 / (codes.index(row["code"]) + 1)
            metrics[f"beam_top{width}"] = hits / count
        metrics["beam_mrr"] = reciprocal_rank / count
    return metrics


def _evaluate_cached_positions(
    model,
    vocabulary,
    cfg: dict,
    checkpoint: dict,
    *,
    output: str | Path,
    cohort: str,
    expected_tracks: int | None,
    max_tracks: int | None,
    sample_tracks: int | None,
    sample_seed: int,
    device: str,
    beam_width: int | None,
) -> dict:
    track_ids = select_checkpoint_cohort(
        checkpoint,
        cohort=cohort,
        expected_tracks=expected_tracks,
        max_tracks=max_tracks,
        sample_tracks=sample_tracks,
        sample_seed=sample_seed,
    )
    data_cfg = cfg["data"]
    fingerprint = checkpoint["tokenizer_fingerprint"]
    canonical = TokenStoreIndex(
        data_cfg["canonical_token_root"],
        tokenizer_fingerprint=fingerprint,
        corpus_role="canonical_training",
    )
    heldout = TokenStoreIndex(
        data_cfg["heldout_evaluation_token_root"],
        tokenizer_fingerprint=fingerprint,
        corpus_role="heldout_evaluation",
    )
    dataset = CachedPositionDataset(
        canonical, heldout, track_ids=track_ids
    )
    rows = []
    started = time.perf_counter()
    policies = (
        ("canonical", [float(value) for value in data_cfg["canonical_starts"]]),
        (
            "heldout",
            [float(value) for value in data_cfg["shifted_evaluation_starts"]],
        ),
    )
    batch_size = int(cfg["evaluation"]["generation_batch_size"])
    batches_per_position = math.ceil(len(track_ids) / batch_size)
    total_positions = sum(len(starts) for _, starts in policies)
    progress = tqdm(
        total=total_positions * batches_per_position,
        desc=f"{cohort} evaluation",
        unit="batch",
    )
    for view_type, starts in policies:
        for start in starts:
            progress.set_postfix(view=view_type, start=f"{start:g}")
            indices = dataset.indices_for(track_ids, view_type, start)
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda examples: collate_causal_documents(
                    examples,
                    vocabulary,
                    int(cfg["model"]["max_position_embeddings"]),
                ),
            )
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                columns = (input_ids == vocabulary.id_token_id).nonzero()[:, 1].unique()
                if len(columns) != 1:
                    raise RuntimeError("Cached evaluation prompts have unequal lengths")
                prompts = input_ids[:, : int(columns[0]) + 1]
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if str(device).startswith("cuda")
                    else nullcontext()
                )
                with autocast:
                    greedy = batched_greedy_generate(model, prompts, vocabulary)
                    beams = (
                        batched_beam_generate(
                            model, prompts, vocabulary, width=beam_width
                        )
                        if beam_width is not None
                        else [[] for _ in greedy]
                    )
                rows.extend(
                    {
                        "track_id": track_id,
                        "code": code,
                        "view_type": view_type,
                        "start": start,
                        "greedy": greedy_result.code,
                        "greedy_ended_with_eos": greedy_result.ended_with_eos,
                        "beam": [
                            {
                                "code": result.code,
                                "log_probability": result.log_probability,
                                "ended_with_eos": result.ended_with_eos,
                            }
                            for result in ranking
                        ],
                    }
                    for track_id, code, greedy_result, ranking in zip(
                        batch["track_id"],
                        batch["code"],
                        greedy,
                        beams,
                        strict=True,
                    )
                )
                progress.update()
    progress.close()
    by_start = {
        f"{view_type}:{start:g}": _generation_metrics(
            [
                row
                for row in rows
                if row["view_type"] == view_type and row["start"] == start
            ]
        )
        for view_type, starts in policies
        for start in starts
    }
    metrics = {
        "cohort": cohort,
        "selected_tracks": len(track_ids),
        "sample_seed": sample_seed if sample_tracks is not None else None,
        "generation_protocol": "five_autoregressive_digits_then_eos",
        "canonical": _generation_metrics(
            [row for row in rows if row["view_type"] == "canonical"]
        ),
        "heldout": _generation_metrics(
            [row for row in rows if row["view_type"] == "heldout"]
        ),
        "by_start": by_start,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "queries": rows}, indent=2) + "\n")
    return metrics


JOINT_BEAM_PROTOCOL = "paper_joint_beam_inference_v1"


def joint_window_starts(
    query_samples: int, window_samples: int, hop_samples: int
) -> list[int]:
    if query_samples < window_samples:
        raise ValueError("Query must be at least as long as one model window")
    if window_samples < 1 or hop_samples < 1:
        raise ValueError("Window and hop sizes must be positive")
    final_start = query_samples - window_samples
    starts = list(range(0, final_start + 1, hop_samples))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _fingerprint_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_file_fingerprint(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _joint_output_paths(output: str | Path) -> tuple[Path, Path, Path, Path]:
    summary = Path(output)
    return (
        summary,
        summary.with_suffix(".csv"),
        summary.with_suffix(".queries.jsonl"),
        summary.with_suffix(".manifest.json"),
    )


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _valid_waveform(waveform: np.ndarray, expected_samples: int) -> None:
    if len(waveform) != expected_samples:
        raise ValueError(
            f"Decoded {len(waveform)} samples, expected {expected_samples}"
        )
    if not np.isfinite(waveform).all():
        raise ValueError("Decoded query contains non-finite samples")
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms <= 1e-8:
        raise ValueError("Decoded query is silent")


def _load_query_context(
    path: Path,
    *,
    sample_rate: int,
    start_sample: int,
    query_samples: int,
    past_samples: int,
) -> np.ndarray:
    available_past = min(start_sample, past_samples)
    decoded_samples = available_past + query_samples
    decoded = load_audio(
        path,
        sample_rate=sample_rate,
        start=(start_sample - available_past) / sample_rate,
        duration=decoded_samples / sample_rate,
        pad=False,
    )
    _valid_waveform(decoded[-query_samples:], query_samples)
    if len(decoded) != decoded_samples:
        raise ValueError(
            f"Decoded context has {len(decoded)} samples, expected {decoded_samples}"
        )
    if available_past < past_samples:
        decoded = np.pad(decoded, (past_samples - available_past, 0))
    expected = past_samples + query_samples
    if len(decoded) != expected:
        raise RuntimeError("Past-context padding produced an invalid length")
    return np.asarray(decoded, dtype=np.float32)


def _joint_manifest_configuration(
    *,
    checkpoint_fingerprint: str,
    checkpoint: dict,
    rir_manifest: dict | None,
    cohort: str,
    expected_tracks: int | None,
    sample_tracks: int,
    sample_seed: int,
    recipe_seed: int,
    query_lengths: tuple[float, ...],
    conditions: tuple[str, ...],
    beam_width: int,
    sample_rate: int,
    window_seconds: float,
    past_context_seconds: float,
) -> dict:
    return {
        "protocol": JOINT_BEAM_PROTOCOL,
        "checkpoint_sha256": checkpoint_fingerprint,
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "tokenizer_fingerprint": checkpoint["tokenizer_fingerprint"],
        "training_corpus_fingerprint": checkpoint.get(
            "training_corpus_fingerprint"
        ),
        "room_ir_validation_fingerprint": (
            rir_manifest["validation_fingerprint"] if rir_manifest else None
        ),
        "cohort": cohort,
        "expected_tracks": expected_tracks,
        "sample_tracks": sample_tracks,
        "sample_seed": sample_seed,
        "recipe_seed": recipe_seed,
        "query_lengths": list(query_lengths),
        "conditions": list(conditions),
        "beam_width": beam_width,
        "sample_rate": sample_rate,
        "window_seconds": window_seconds,
        "hop_seconds": window_seconds / 2,
        "past_context_seconds": past_context_seconds,
    }


def _load_or_create_joint_manifest(
    *,
    path: Path,
    configuration: dict,
    checkpoint: dict,
    cfg: dict,
    rir_assets: RoomImpulseResponseAssets | None,
) -> dict:
    if path.exists():
        manifest = json.loads(path.read_text())
        content = {key: value for key, value in manifest.items() if key != "fingerprint"}
        if manifest.get("fingerprint") != _fingerprint_json(content):
            raise ValueError("Existing joint-evaluation manifest fingerprint is invalid")
        if manifest.get("configuration") != configuration:
            raise ValueError(
                "Existing joint-evaluation manifest does not match this invocation"
            )
        if len(manifest.get("queries", [])) != configuration["sample_tracks"]:
            raise ValueError("Existing joint-evaluation manifest has the wrong size")
        return manifest

    track_ids = select_checkpoint_cohort(
        checkpoint,
        cohort=configuration["cohort"],
        expected_tracks=configuration["expected_tracks"],
    )
    random.Random(configuration["sample_seed"]).shuffle(track_ids)
    records = load_catalogue(cfg["data"]["catalogue"])
    by_track = {record.track_id: record for record in records}
    sample_rate = int(configuration["sample_rate"])
    maximum_seconds = max(configuration["query_lengths"])
    maximum_samples = round(maximum_seconds * sample_rate)
    audio_root = Path(cfg["data"]["audio_root"])
    recipes = []
    excluded = []
    for track_id in tqdm(track_ids, desc="building joint-evaluation manifest"):
        if len(recipes) == configuration["sample_tracks"]:
            break
        record = by_track.get(track_id)
        try:
            if record is None:
                raise ValueError("Track is missing from the current catalogue")
            source_samples = round(float(record.duration) * sample_rate)
            maximum_start = source_samples - maximum_samples
            if maximum_start < 0:
                raise ValueError("Track is shorter than the maximum query length")
            start_sample = int(
                stable_uint64(
                    configuration["recipe_seed"], track_id, "joint-query-start"
                )
                % (maximum_start + 1)
            )
            waveform = load_audio(
                audio_root / record.path,
                sample_rate=sample_rate,
                start=start_sample / sample_rate,
                duration=maximum_seconds,
                pad=False,
            )
            _valid_waveform(waveform, maximum_samples)
            rir_path = None
            if "rir" in configuration["conditions"]:
                if rir_assets is None:
                    raise RuntimeError("RIR evaluation requested without RIR assets")
                _, rir_path = rir_assets.load_validation(
                    stable_uint64(
                        configuration["recipe_seed"], track_id, "paper-room-ir"
                    )
                )
            recipes.append(
                {
                    "track_id": record.track_id,
                    "code": record.code,
                    "source_path": record.path,
                    "source_duration": float(record.duration),
                    "start_sample": start_sample,
                    "start_seconds": start_sample / sample_rate,
                    "rir_path": rir_path,
                }
            )
        except Exception as exc:
            excluded.append(
                {
                    "track_id": track_id,
                    "source_path": record.path if record is not None else None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if len(recipes) != configuration["sample_tracks"]:
        raise RuntimeError(
            f"Only {len(recipes)} valid tracks available for the requested "
            f"{configuration['sample_tracks']}-track evaluation"
        )
    content = {
        "configuration": configuration,
        "queries": recipes,
        "excluded_candidates": excluded,
    }
    manifest = {**content, "fingerprint": _fingerprint_json(content)}
    _atomic_write_json(path, manifest)
    return manifest


def _load_joint_rows(path: Path, *, fingerprint: str) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str, str], dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("protocol_fingerprint") != fingerprint:
                raise ValueError(
                    f"Joint-evaluation JSONL fingerprint mismatch on line {line_number}"
                )
            key = (
                str(row["track_id"]),
                f"{float(row['query_seconds']):g}",
                str(row["condition"]),
            )
            if key in rows:
                raise ValueError(f"Duplicate joint-evaluation result for {key}")
            rows[key] = row
    return rows


def _joint_metrics(rows: list[dict], *, selected_tracks: int) -> dict:
    if len(rows) != selected_tracks:
        raise ValueError(
            f"Expected {selected_tracks} completed queries, found {len(rows)}"
        )
    successful = [row for row in rows if row["status"] == "ok"]
    failed = [row for row in rows if row["status"] == "error"]
    result = {
        "selected_queries": selected_tracks,
        "evaluated_queries": len(successful),
        "failed_queries": len(failed),
        "elapsed_seconds": sum(float(row.get("latency_seconds", 0.0)) for row in rows),
    }
    elapsed = result["elapsed_seconds"]
    result["queries_per_second"] = len(successful) / elapsed if elapsed else 0.0
    ranks = [row["correct_rank"] for row in successful]
    for width in (1, 5, 10):
        result[f"beam_top{width}"] = sum(
            rank is not None and rank <= width for rank in ranks
        ) / selected_tracks
    result["beam_mrr"] = sum(
        0.0 if rank is None else 1.0 / rank for rank in ranks
    ) / selected_tracks
    return result


def _write_joint_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = (
        "condition",
        "query_seconds",
        "selected_queries",
        "evaluated_queries",
        "failed_queries",
        "beam_top1",
        "beam_top5",
        "beam_top10",
        "beam_mrr",
        "elapsed_seconds",
        "queries_per_second",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in rows)
    temporary.replace(path)


def _evaluate_joint_beam(
    checkpoint_path: str | Path,
    model,
    vocabulary,
    cfg: dict,
    checkpoint: dict,
    *,
    output: str | Path,
    cohort: str,
    expected_tracks: int | None,
    sample_tracks: int,
    sample_seed: int,
    recipe_seed: int,
    query_lengths: tuple[float, ...],
    conditions: tuple[str, ...],
    device: str,
    beam_width: int,
    rir_training_root: str | Path | None,
    rir_validation_root: str | Path | None,
) -> dict:
    profile = historical_checkpoint_profile(checkpoint)
    if beam_width != 10:
        raise ValueError("Paper-facing joint-beam evaluation requires beam_width=10")
    if cohort != "training":
        raise ValueError("Paper-facing joint-beam evaluation uses the training cohort")
    if sample_tracks < 1:
        raise ValueError("sample_tracks must be positive")
    if not query_lengths or any(length <= 0 for length in query_lengths):
        raise ValueError("query_lengths must contain positive durations")
    if len(set(query_lengths)) != len(query_lengths):
        raise ValueError("query_lengths must not contain duplicates")
    if not conditions or set(conditions) - {"clean", "rir"}:
        raise ValueError("Joint-beam conditions must be clean and/or rir")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must not contain duplicates")

    tokenizer = _checkpoint_tokenizer(checkpoint, device)
    window_seconds = float(cfg["data"]["segment_duration"])
    if not math.isclose(window_seconds, 2.0):
        raise ValueError("Paper-facing joint-beam evaluation requires 2-second windows")
    if any(length < window_seconds for length in query_lengths):
        raise ValueError("Every query length must be at least two seconds")
    rir_cfg = cfg["data"].get("room_ir", {})
    past_seconds = float(rir_cfg.get("past_context_duration", 2.0))
    rir_assets = None
    rir_manifest = None
    if "rir" in conditions:
        training_root = rir_training_root or rir_cfg.get(
            "training_root",
            "/gpfs/scratch/acw723/audio-degradation-data/degradation_24k/room_ir/train",
        )
        validation_root = rir_validation_root or rir_cfg.get(
            "validation_root",
            "/gpfs/scratch/acw723/audio-degradation-data/degradation_24k/room_ir/test",
        )
        rir_assets = RoomImpulseResponseAssets(
            training_root, validation_root, sample_rate=tokenizer.sample_rate
        )
        rir_manifest = rir_assets.manifest()
        saved_manifest = checkpoint.get("room_ir_manifest")
        if saved_manifest is not None and saved_manifest != rir_manifest:
            raise ValueError("Validation room-IR assets do not match the checkpoint")

    summary_path, csv_path, query_path, manifest_path = _joint_output_paths(output)
    checkpoint_fingerprint = _checkpoint_file_fingerprint(checkpoint_path)
    configuration = _joint_manifest_configuration(
        checkpoint_fingerprint=checkpoint_fingerprint,
        checkpoint=checkpoint,
        rir_manifest=rir_manifest,
        cohort=cohort,
        expected_tracks=expected_tracks,
        sample_tracks=sample_tracks,
        sample_seed=sample_seed,
        recipe_seed=recipe_seed,
        query_lengths=query_lengths,
        conditions=conditions,
        beam_width=beam_width,
        sample_rate=tokenizer.sample_rate,
        window_seconds=window_seconds,
        past_context_seconds=past_seconds,
    )
    manifest = _load_or_create_joint_manifest(
        path=manifest_path,
        configuration=configuration,
        checkpoint=checkpoint,
        cfg=cfg,
        rir_assets=rir_assets,
    )
    completed = _load_joint_rows(query_path, fingerprint=manifest["fingerprint"])
    total = len(manifest["queries"]) * len(query_lengths) * len(conditions)
    progress = tqdm(total=total, initial=len(completed), desc="joint-beam evaluation")
    query_path.parent.mkdir(parents=True, exist_ok=True)
    audio_root = Path(cfg["data"]["audio_root"])
    sample_rate = tokenizer.sample_rate
    window_samples = round(window_seconds * sample_rate)
    hop_samples = window_samples // 2
    past_samples = round(past_seconds * sample_rate)
    configured_batch = int(cfg["evaluation"]["generation_batch_size"])

    with query_path.open("a", encoding="utf-8") as output_handle:
        for query_seconds in query_lengths:
            query_samples = round(query_seconds * sample_rate)
            starts = joint_window_starts(query_samples, window_samples, hop_samples)
            query_batch_size = max(1, configured_batch // len(starts))
            for condition in conditions:
                pending = [
                    recipe
                    for recipe in manifest["queries"]
                    if (
                        recipe["track_id"],
                        f"{query_seconds:g}",
                        condition,
                    )
                    not in completed
                ]
                for batch_start in range(0, len(pending), query_batch_size):
                    batch_recipes = pending[batch_start : batch_start + query_batch_size]
                    batch_started = time.perf_counter()
                    prepared = []
                    for recipe in batch_recipes:
                        try:
                            if condition == "clean":
                                waveform = load_audio(
                                    audio_root / recipe["source_path"],
                                    sample_rate=sample_rate,
                                    start=recipe["start_sample"] / sample_rate,
                                    duration=query_seconds,
                                    pad=False,
                                )
                                _valid_waveform(waveform, query_samples)
                            else:
                                context = _load_query_context(
                                    audio_root / recipe["source_path"],
                                    sample_rate=sample_rate,
                                    start_sample=int(recipe["start_sample"]),
                                    query_samples=query_samples,
                                    past_samples=past_samples,
                                )
                                if rir_assets is None:
                                    raise RuntimeError("RIR assets are unavailable")
                                ir, rir_path = rir_assets.load_validation(
                                    stable_uint64(
                                        recipe_seed,
                                        recipe["track_id"],
                                        "paper-room-ir",
                                    )
                                )
                                if rir_path != recipe["rir_path"]:
                                    raise ValueError("Validation IR recipe changed")
                                waveform = convolve_full_wet(
                                    context,
                                    ir,
                                    past_context_samples=past_samples,
                                    output_samples=query_samples,
                                )
                            windows = np.stack(
                                [
                                    waveform[start : start + window_samples]
                                    for start in starts
                                ]
                            )
                            prepared.append((recipe, windows))
                        except Exception as exc:
                            row = {
                                "protocol_fingerprint": manifest["fingerprint"],
                                "status": "error",
                                "track_id": recipe["track_id"],
                                "code": recipe["code"],
                                "query_seconds": query_seconds,
                                "condition": condition,
                                "start_sample": recipe["start_sample"],
                                "rir_path": recipe["rir_path"] if condition == "rir" else None,
                                "window_starts": starts,
                                "latency_seconds": time.perf_counter() - batch_started,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                            output_handle.write(json.dumps(row, sort_keys=True) + "\n")
                            output_handle.flush()
                            completed[(recipe["track_id"], f"{query_seconds:g}", condition)] = row
                            progress.update()
                    if not prepared:
                        continue
                    flattened = torch.from_numpy(
                        np.concatenate([windows for _, windows in prepared], axis=0)
                    ).to(device)
                    audio_tokens = tokenizer.tokenize(flattened)
                    prompts = prompts_from_audio_tokens(audio_tokens, vocabulary).reshape(
                        len(prepared), len(starts), -1
                    )
                    autocast = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if str(device).startswith("cuda")
                        else nullcontext()
                    )
                    with autocast:
                        rankings = batched_joint_beam_generate(
                            model, prompts, vocabulary, width=beam_width
                        )
                    if str(device).startswith("cuda"):
                        torch.cuda.synchronize()
                    latency = (time.perf_counter() - batch_started) / len(prepared)
                    for (recipe, _), ranking in zip(prepared, rankings, strict=True):
                        codes = [candidate.code for candidate in ranking]
                        correct_rank = (
                            codes.index(recipe["code"]) + 1
                            if recipe["code"] in codes
                            else None
                        )
                        row = {
                            "protocol_fingerprint": manifest["fingerprint"],
                            "status": "ok",
                            "track_id": recipe["track_id"],
                            "code": recipe["code"],
                            "query_seconds": query_seconds,
                            "condition": condition,
                            "start_sample": recipe["start_sample"],
                            "rir_path": recipe["rir_path"] if condition == "rir" else None,
                            "window_starts": starts,
                            "window_count": len(starts),
                            "correct_rank": correct_rank,
                            "latency_seconds": latency,
                            "beam": [
                                {
                                    "code": candidate.code,
                                    "mean_log_probability": candidate.log_probability,
                                    "ended_with_eos": candidate.ended_with_eos,
                                }
                                for candidate in ranking
                            ],
                        }
                        output_handle.write(json.dumps(row, sort_keys=True) + "\n")
                        output_handle.flush()
                        completed[(recipe["track_id"], f"{query_seconds:g}", condition)] = row
                        progress.update()
    progress.close()

    metric_rows = []
    metrics = {}
    all_rows = list(completed.values())
    for condition in conditions:
        metrics[condition] = {}
        for query_seconds in query_lengths:
            selected = [
                row
                for row in all_rows
                if row["condition"] == condition
                and math.isclose(float(row["query_seconds"]), query_seconds)
            ]
            summary = _joint_metrics(
                selected, selected_tracks=len(manifest["queries"])
            )
            metrics[condition][f"{query_seconds:g}"] = summary
            metric_rows.append(
                {
                    "condition": condition,
                    "query_seconds": query_seconds,
                    **summary,
                }
            )
    payload = {
        "protocol": JOINT_BEAM_PROTOCOL,
        "manifest_fingerprint": manifest["fingerprint"],
        "configuration": configuration,
        "selected_tracks": len(manifest["queries"]),
        "checkpoint_profile": profile,
        "excluded_candidates": len(manifest["excluded_candidates"]),
        "metrics": metrics,
    }
    _atomic_write_json(summary_path, payload)
    _write_joint_csv(csv_path, metric_rows)
    return payload


def evaluate(
    checkpoint_path: str | Path,
    *,
    output: str | Path,
    protocol: str = "segment",
    cohort: str = "probe",
    expected_tracks: int | None = None,
    max_tracks: int | None = None,
    sample_tracks: int | None = None,
    sample_seed: int = 1337,
    device: str = "cuda",
    beam_width: int | None = 10,
    generation_only: bool = False,
    recipe_seed: int = 1337,
    query_lengths: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0),
    conditions: tuple[str, ...] = ("clean", "rir"),
    rir_training_root: str | Path | None = None,
    rir_validation_root: str | Path | None = None,
) -> dict:
    model, vocabulary, cfg, checkpoint = load_audio_lm(checkpoint_path, device)
    if protocol == "joint-beam":
        if max_tracks is not None:
            raise ValueError("joint-beam does not support max_tracks; use sample_tracks")
        if generation_only:
            raise ValueError("joint-beam always computes the complete beam metrics")
        return _evaluate_joint_beam(
            checkpoint_path,
            model,
            vocabulary,
            cfg,
            checkpoint,
            output=output,
            cohort=cohort,
            expected_tracks=(
                len(checkpoint["training_track_ids"])
                if expected_tracks is None
                else expected_tracks
            ),
            sample_tracks=1000 if sample_tracks is None else sample_tracks,
            sample_seed=sample_seed,
            recipe_seed=recipe_seed,
            query_lengths=tuple(float(value) for value in query_lengths),
            conditions=tuple(conditions),
            device=device,
            beam_width=10 if beam_width is None else beam_width,
            rir_training_root=rir_training_root,
            rir_validation_root=rir_validation_root,
        )
    if protocol != "segment":
        raise ValueError(f"Unknown evaluation protocol {protocol!r}")
    if checkpoint.get("training_protocol") in {
        "online_random_crop_noise_rir_consistency_25k_v1",
        "token_budget_matched_two_second_noise_consistency_v1",
        "online_random_crop_noise_consistency_v1",
        "online_random_crop_consistency_profile_v2",
        "online_random_crop_clean_capacity_v1",
        "online_random_crop_clean_capacity_eight_codebook_v2",
    }:
        return _evaluate_tc6_monitor_manifest(
            model,
            vocabulary,
            cfg,
            checkpoint,
            output=output,
            device=device,
            beam_width=beam_width,
        )
    if "view_mode" in cfg.get("data", {}):
        return _evaluate_cached_positions(
            model,
            vocabulary,
            cfg,
            checkpoint,
            output=output,
            cohort=cohort,
            expected_tracks=expected_tracks,
            max_tracks=max_tracks,
            sample_tracks=sample_tracks,
            sample_seed=sample_seed,
            device=device,
            beam_width=beam_width,
        )
    tokenizer = _checkpoint_tokenizer(checkpoint, device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    records = load_catalogue(cfg["data"]["catalogue"])
    by_track_id = {record.track_id: record for record in records}
    selected_track_ids = select_checkpoint_cohort(
        checkpoint,
        cohort=cohort,
        expected_tracks=expected_tracks,
        max_tracks=max_tracks,
        sample_tracks=sample_tracks,
        sample_seed=sample_seed,
    )
    positions = [float(value) for value in cfg["evaluation"]["shifted_starts"]]
    bad = BadFileRegistry(cfg["data"]["runtime_bad_files"])
    root = Path(cfg["data"]["audio_root"])
    duration = float(cfg["data"]["segment_duration"])
    if cohort == "training":
        canonical = [
            position
            for position in positions
            if math.isclose(position / duration, round(position / duration))
        ]
        if canonical:
            raise ValueError(
                "Training-cohort evaluation positions must differ from canonical "
                f"training starts; found {canonical}"
            )
    targets = []
    greedy_codes = []
    greedy_protocol_valid = []
    rankings = []
    combined_losses = []
    audio_losses = []
    id_losses = []
    boundary_eos_losses = []
    digit_correct = 0.0
    exact_correct = 0.0
    latency = 0.0
    rows = []
    for track_id in tqdm(selected_track_ids, desc=f"{cohort} tracks"):
        if track_id not in by_track_id:
            raise ValueError(
                f"Selected {cohort} track {track_id} is missing from the catalogue"
            )
        record = by_track_id[track_id]
        if bad.contains(record.path):
            continue
        for start in positions:
            try:
                waveform = load_audio(
                    root / record.path,
                    sample_rate=tokenizer.sample_rate,
                    start=start,
                    duration=duration,
                    pad=True,
                )
                audio_tokens = tokenizer.tokenize(
                    torch.from_numpy(waveform).unsqueeze(0)
                )[0].cpu()
            except Exception as exc:
                bad.add(record.path, exc)
                break
            started = time.perf_counter()
            with torch.inference_mode():
                if not generation_only:
                    example = {
                        "audio_tokens": audio_tokens,
                        "code": record.code,
                        "track_id": record.track_id,
                        "document_index": -1,
                    }
                    batch = collate_causal_documents(
                        [example],
                        vocabulary,
                        int(cfg["model"]["max_position_embeddings"]),
                    )
                    input_ids = batch["input_ids"].to(device)
                    logits = model(
                        input_ids, batch["attention_mask"].to(device)
                    )
                    losses = causal_audio_id_losses(
                        logits,
                        input_ids,
                        batch["audio_target_mask"].to(device),
                        batch["id_target_mask"].to(device),
                        batch["boundary_target_mask"].to(device),
                        id_digit_weight=float(cfg["train"]["id_digit_weight"]),
                    )
                prompt = prompt_from_audio_tokens(audio_tokens.to(device), vocabulary)
                greedy = greedy_generate(model, prompt, vocabulary)
                beam = (
                    beam_generate(model, prompt, vocabulary, width=beam_width)
                    if beam_width is not None
                    else []
                )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latency += time.perf_counter() - started
            targets.append(record.code)
            greedy_codes.append(greedy.code)
            greedy_protocol_valid.append(greedy.ended_with_eos)
            rankings.append(beam)
            if not generation_only:
                combined_losses.append(float(losses["loss"]))
                audio_losses.append(float(losses["audio_loss"]))
                id_losses.append(float(losses["id_loss"]))
                boundary_eos_losses.append(float(losses["boundary_eos_loss"]))
                digit_correct += float(losses["teacher_forced_digit_accuracy"])
                exact_correct += float(losses["teacher_forced_exact_accuracy"])
            rows.append(
                {
                    "track_id": record.track_id,
                    "code": record.code,
                    "path": record.path,
                    "start": start,
                    "greedy": greedy.code,
                    "greedy_ended_with_eos": greedy.ended_with_eos,
                    "beam": [
                        {
                            "code": result.code,
                            "log_probability": result.log_probability,
                            "ended_with_eos": result.ended_with_eos,
                        }
                        for result in beam
                    ],
                }
            )
    if not targets:
        raise RuntimeError("No evaluation queries were tokenized")
    count = len(targets)
    metrics = {
        "cohort": cohort,
        "selected_tracks": len(selected_track_ids),
        "sample_seed": sample_seed if sample_tracks is not None else None,
        "evaluated_tracks": len({row["track_id"] for row in rows}),
        "skipped_tracks": len(selected_track_ids)
        - len({row["track_id"] for row in rows}),
        "positions": positions,
        "queries": count,
        "generation_protocol": "five_autoregressive_digits_then_eos",
        "greedy_top1": sum(
            target == prediction
            for target, prediction in zip(targets, greedy_codes, strict=True)
        )
        / count,
        "invalid_code_rate": sum(not valid for valid in greedy_protocol_valid) / count,
        "mean_latency_seconds": latency / count,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_inference_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if str(device).startswith("cuda")
            else 0
        ),
    }
    if not generation_only:
        mean_audio_loss = sum(audio_losses) / count
        metrics.update(
            {
                "loss": sum(combined_losses) / count,
                "audio_loss": mean_audio_loss,
                "audio_perplexity": math.exp(min(20.0, mean_audio_loss)),
                "id_loss": sum(id_losses) / count,
                "boundary_eos_loss": sum(boundary_eos_losses) / count,
                "teacher_forced_digit_accuracy": digit_correct / count,
                "teacher_forced_exact_accuracy": exact_correct / count,
            }
        )
    if beam_width is not None:
        reciprocal_rank = 0.0
        for width in (1, 5, 10):
            hits = 0
            for target, ranking in zip(targets, rankings, strict=True):
                codes = [result.code for result in ranking]
                hits += int(target in codes[:width])
                if width == 10 and target in codes:
                    reciprocal_rank += 1 / (codes.index(target) + 1)
            metrics[f"beam_top{width}"] = hits / count
        metrics["beam_mrr"] = reciprocal_rank / count
    metrics["external_artifacts"] = [
        "audio_lm_checkpoint",
        "frozen_muq_tokenizer_weights",
        "query_audio",
    ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"metrics": metrics, "queries": rows}, indent=2) + "\n")
    return metrics
