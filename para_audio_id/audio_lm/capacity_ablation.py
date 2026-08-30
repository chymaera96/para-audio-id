from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..audio import load_audio
from ..catalogue import CatalogueRecord, load_catalogue
from ..config import load_config
from .checkpoint import load_audio_lm, validate_checkpoint_metadata
from .generation import batched_beam_generate, prompts_from_audio_tokens
from .noise import stable_uint64
from .profiles import (
    CAPACITY_TRAINING_PROTOCOL,
    SUPPORTED_DATABASE_SIZES,
    catalogue_fingerprint,
    cohort_manifest,
    decoder_profile,
    historical_checkpoint_profile,
    validate_cohort_manifest,
)
from .tokenizer import MuQRVQTokenizer


CAPACITY_ABLATION_PROTOCOL = "clean_capacity_common_2s_beam_mrr_v1"
CAPACITY_ABLATION_SEED = 1337
CAPACITY_ABLATION_TRACKS = 1_000
CAPACITY_ABLATION_SECONDS = 2.0
CAPACITY_ABLATION_SAMPLE_RATE = 24_000
CAPACITY_ABLATION_AUDIO_TOKENS = 400
CAPACITY_ABLATION_BEAM_WIDTH = 10


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _fingerprint(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capacity_run_name(database_size: int, decoder: str) -> str:
    if database_size not in SUPPORTED_DATABASE_SIZES:
        raise ValueError(f"Unsupported capacity database size: {database_size}")
    decoder_profile(decoder)
    return f"{database_size // 1_000}k-{decoder}-cb8"


def capacity_ablation_paths(
    cfg: dict, database_size: int, decoder: str
) -> tuple[Path, Path, Path]:
    run_name = capacity_run_name(database_size, decoder)
    checkpoint = Path(cfg["train"]["checkpoint_dir"]) / run_name / "last.ckpt"
    result_root = Path("capacity-ablation-results")
    result = result_root / f"{run_name.removesuffix('-cb8')}.json"
    manifest = result_root / "clean-2s-common-1k.manifest.json"
    return checkpoint, result, manifest


def _manifest_track_ids(path: Path, records: list[CatalogueRecord], size: int) -> list[str]:
    return validate_cohort_manifest(path, records, size)


def validate_nested_capacity_cohorts(
    cfg: dict, records: list[CatalogueRecord]
) -> tuple[dict[int, list[str]], dict[str, str]]:
    cohorts: dict[int, list[str]] = {}
    fingerprints: dict[str, str] = {}
    for size in SUPPORTED_DATABASE_SIZES:
        path = Path(cohort_manifest(size))
        track_ids = _manifest_track_ids(path, records, size)
        cohorts[size] = track_ids
        fingerprints[str(size)] = _file_sha256(path)
    for smaller, larger in zip(SUPPORTED_DATABASE_SIZES, SUPPORTED_DATABASE_SIZES[1:]):
        missing = set(cohorts[smaller]) - set(cohorts[larger])
        if missing:
            example = sorted(missing)[0]
            raise ValueError(
                "Capacity cohorts are not nested: "
                f"{smaller // 1_000}K is not a subset of {larger // 1_000}K "
                f"(for example, {example} is missing)"
            )
    return cohorts, fingerprints


def _validate_capacity_checkpoint(
    checkpoint: dict,
    *,
    database_size: int,
    decoder: str,
    expected_track_ids: list[str],
) -> dict:
    validate_checkpoint_metadata(checkpoint)
    profile = historical_checkpoint_profile(checkpoint)
    if profile.get("experiment") != "clean_capacity":
        raise ValueError("Checkpoint is not a clean-capacity experiment")
    if profile.get("schedule", {}).get("protocol") != CAPACITY_TRAINING_PROTOCOL:
        raise ValueError("Checkpoint is not an eight-codebook capacity checkpoint")
    if int(profile.get("database_size", -1)) != database_size:
        raise ValueError("Checkpoint database size does not match --database-size")
    if profile.get("decoder") != decoder_profile(decoder):
        raise ValueError("Checkpoint decoder does not match --decoder")
    query = profile.get("query", {})
    if int(query.get("selected_codebooks", -1)) != 8:
        raise ValueError("Capacity ablation requires an eight-codebook checkpoint")
    if not math.isclose(float(query.get("segment_duration_seconds", -1)), 2.0):
        raise ValueError("Capacity ablation requires a two-second checkpoint")
    spec = checkpoint["tokenizer_spec"]
    if int(spec.get("selected_codebooks", -1)) != 8:
        raise ValueError("Checkpoint tokenizer does not use eight codebooks")
    if list(checkpoint["training_track_ids"]) != expected_track_ids:
        raise ValueError(
            "Checkpoint training identities do not exactly match the requested manifest"
        )
    return profile


def _query_start_sample(record: CatalogueRecord, seed: int) -> int:
    maximum = int(math.floor(record.duration * CAPACITY_ABLATION_SAMPLE_RATE)) - round(
        CAPACITY_ABLATION_SECONDS * CAPACITY_ABLATION_SAMPLE_RATE
    )
    if maximum < 0:
        raise ValueError("track is shorter than two seconds")
    return int(stable_uint64(seed, record.track_id, "capacity-ablation-start") % (maximum + 1))


def _validate_waveform(waveform: np.ndarray) -> None:
    expected = round(CAPACITY_ABLATION_SECONDS * CAPACITY_ABLATION_SAMPLE_RATE)
    if waveform.shape != (expected,):
        raise ValueError(f"decoded {len(waveform)} samples instead of {expected}")
    if not np.isfinite(waveform).all():
        raise ValueError("decoded crop contains non-finite samples")
    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if rms <= 1e-8:
        raise ValueError("decoded crop is silent")


def _manifest_payload(
    *,
    queries: list[dict],
    exclusions: list[dict],
    cohort_fingerprints: dict[str, str],
    catalogue_digest: str,
    tokenizer_fingerprint: str,
) -> dict:
    payload = {
        "protocol": CAPACITY_ABLATION_PROTOCOL,
        "seed": CAPACITY_ABLATION_SEED,
        "sample_rate": CAPACITY_ABLATION_SAMPLE_RATE,
        "query_duration_seconds": CAPACITY_ABLATION_SECONDS,
        "selected_queries": CAPACITY_ABLATION_TRACKS,
        "source_cohort_size": 10_000,
        "cohort_fingerprints": cohort_fingerprints,
        "catalogue_fingerprint": catalogue_digest,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "queries": queries,
        "exclusions": exclusions,
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def _validate_query_manifest(
    payload: dict,
    *,
    records_by_id: dict[str, CatalogueRecord],
    common_track_ids: list[str],
    cohort_fingerprints: dict[str, str],
    catalogue_digest: str,
    tokenizer_fingerprint: str,
) -> dict:
    expected = {
        "protocol": CAPACITY_ABLATION_PROTOCOL,
        "seed": CAPACITY_ABLATION_SEED,
        "sample_rate": CAPACITY_ABLATION_SAMPLE_RATE,
        "query_duration_seconds": CAPACITY_ABLATION_SECONDS,
        "selected_queries": CAPACITY_ABLATION_TRACKS,
        "source_cohort_size": 10_000,
        "cohort_fingerprints": cohort_fingerprints,
        "catalogue_fingerprint": catalogue_digest,
        "tokenizer_fingerprint": tokenizer_fingerprint,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Existing common query manifest has incompatible {key}")
    supplied_fingerprint = payload.get("fingerprint")
    unsigned = {key: value for key, value in payload.items() if key != "fingerprint"}
    if supplied_fingerprint != _fingerprint(unsigned):
        raise ValueError("Existing common query manifest fingerprint is invalid")
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != CAPACITY_ABLATION_TRACKS:
        raise ValueError("Common query manifest must contain exactly 1,000 queries")
    query_ids = [row.get("track_id") for row in queries]
    if len(set(query_ids)) != CAPACITY_ABLATION_TRACKS:
        raise ValueError("Common query manifest contains duplicate track identities")
    common = set(common_track_ids)
    crop_samples = round(CAPACITY_ABLATION_SECONDS * CAPACITY_ABLATION_SAMPLE_RATE)
    for row in queries:
        track_id = row.get("track_id")
        if track_id not in common or track_id not in records_by_id:
            raise ValueError("Common query manifest contains an identity outside the 10K cohort")
        record = records_by_id[track_id]
        if row.get("code") != record.code or row.get("source_path") != record.path:
            raise ValueError("Common query manifest no longer matches the catalogue")
        start = int(row.get("start_sample", -1))
        maximum = int(math.floor(record.duration * CAPACITY_ABLATION_SAMPLE_RATE)) - crop_samples
        if start < 0 or start > maximum:
            raise ValueError("Common query manifest contains an out-of-range crop")
    return payload


def load_or_create_common_query_manifest(
    path: str | Path,
    *,
    audio_root: str | Path,
    records: list[CatalogueRecord],
    common_track_ids: list[str],
    cohort_fingerprints: dict[str, str],
    tokenizer_fingerprint: str,
) -> dict:
    destination = Path(path)
    records_by_id = {record.track_id: record for record in records}
    catalogue_digest = catalogue_fingerprint(records)
    if destination.exists():
        return _validate_query_manifest(
            json.loads(destination.read_text()),
            records_by_id=records_by_id,
            common_track_ids=common_track_ids,
            cohort_fingerprints=cohort_fingerprints,
            catalogue_digest=catalogue_digest,
            tokenizer_fingerprint=tokenizer_fingerprint,
        )

    order = np.random.default_rng(CAPACITY_ABLATION_SEED).permutation(common_track_ids)
    queries: list[dict] = []
    exclusions: list[dict] = []
    progress = tqdm(total=CAPACITY_ABLATION_TRACKS, desc="building common capacity queries")
    for candidate in order:
        record = records_by_id[str(candidate)]
        try:
            start_sample = _query_start_sample(record, CAPACITY_ABLATION_SEED)
            waveform = load_audio(
                Path(audio_root) / record.path,
                sample_rate=CAPACITY_ABLATION_SAMPLE_RATE,
                start=start_sample / CAPACITY_ABLATION_SAMPLE_RATE,
                duration=CAPACITY_ABLATION_SECONDS,
                pad=False,
            )
            _validate_waveform(waveform)
        except Exception as exc:
            exclusions.append(
                {
                    "track_id": record.track_id,
                    "code": record.code,
                    "source_path": record.path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        queries.append(
            {
                "track_id": record.track_id,
                "code": record.code,
                "source_path": record.path,
                "source_duration_seconds": record.duration,
                "start_sample": start_sample,
                "start_seconds": start_sample / CAPACITY_ABLATION_SAMPLE_RATE,
                "query_duration_seconds": CAPACITY_ABLATION_SECONDS,
            }
        )
        progress.update()
        if len(queries) == CAPACITY_ABLATION_TRACKS:
            break
    progress.close()
    if len(queries) != CAPACITY_ABLATION_TRACKS:
        raise RuntimeError(
            f"Only {len(queries)} usable tracks were found in the shared 10K cohort"
        )
    payload = _manifest_payload(
        queries=queries,
        exclusions=exclusions,
        cohort_fingerprints=cohort_fingerprints,
        catalogue_digest=catalogue_digest,
        tokenizer_fingerprint=tokenizer_fingerprint,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return payload


class CapacityQueryDataset(Dataset):
    def __init__(self, queries: list[dict], audio_root: str | Path):
        self.queries = list(queries)
        self.audio_root = Path(audio_root)

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, index: int) -> dict:
        row = dict(self.queries[index])
        try:
            waveform = load_audio(
                self.audio_root / row["source_path"],
                sample_rate=CAPACITY_ABLATION_SAMPLE_RATE,
                start=float(row["start_seconds"]),
                duration=CAPACITY_ABLATION_SECONDS,
                pad=False,
            )
            _validate_waveform(waveform)
            return {"query": row, "waveform": waveform, "error": None}
        except Exception as exc:
            return {
                "query": row,
                "waveform": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def _list_collate(rows: list[dict]) -> list[dict]:
    return rows


def reciprocal_rank(codes: list[str], target: str) -> float:
    try:
        return 1.0 / (codes.index(target) + 1)
    except ValueError:
        return 0.0


def aggregate_capacity_queries(rows: list[dict], denominator: int) -> dict:
    if denominator < 1 or len(rows) != denominator:
        raise ValueError("Capacity metrics require one result per selected query")
    successful = [row for row in rows if row["status"] == "ok"]
    return {
        "selected_queries": denominator,
        "evaluated_queries": len(successful),
        "failed_queries": denominator - len(successful),
        "beam_mrr": sum(float(row["reciprocal_rank"]) for row in rows) / denominator,
        "beam_top1": sum(row.get("correct_rank") == 1 for row in rows) / denominator,
        "beam_top5": sum(
            row.get("correct_rank") is not None and row["correct_rank"] <= 5
            for row in rows
        )
        / denominator,
        "beam_top10": sum(
            row.get("correct_rank") is not None and row["correct_rank"] <= 10
            for row in rows
        )
        / denominator,
    }


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


def evaluate_capacity_ablation(
    *, database_size: int, decoder: str, device: str = "cuda"
) -> dict:
    cfg = load_config(Path("configs/capacity.yaml"))
    checkpoint_path, result_path, manifest_path = capacity_ablation_paths(
        cfg, database_size, decoder
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing capacity checkpoint: {checkpoint_path}")

    records = load_catalogue(cfg["data"]["catalogue"])
    cohorts, cohort_fingerprints = validate_nested_capacity_cohorts(cfg, records)
    checkpoint_metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    profile = _validate_capacity_checkpoint(
        checkpoint_metadata,
        database_size=database_size,
        decoder=decoder,
        expected_track_ids=cohorts[database_size],
    )
    query_manifest = load_or_create_common_query_manifest(
        manifest_path,
        audio_root=cfg["data"]["audio_root"],
        records=records,
        common_track_ids=cohorts[10_000],
        cohort_fingerprints=cohort_fingerprints,
        tokenizer_fingerprint=checkpoint_metadata["tokenizer_fingerprint"],
    )
    del checkpoint_metadata

    model, vocabulary, _, checkpoint = load_audio_lm(checkpoint_path, device=device)
    if vocabulary.num_codebooks != 8:
        raise ValueError("Capacity ablation requires an eight-codebook vocabulary")
    tokenizer = _checkpoint_tokenizer(checkpoint, device)
    evaluation_cfg = cfg.get("evaluation", {})
    dataloader_cfg = cfg.get("dataloader", {})
    workers = int(dataloader_cfg.get("num_workers", 0))
    loader = DataLoader(
        CapacityQueryDataset(query_manifest["queries"], cfg["data"]["audio_root"]),
        batch_size=int(evaluation_cfg.get("generation_batch_size", 16)),
        shuffle=False,
        num_workers=workers,
        persistent_workers=bool(dataloader_cfg.get("persistent_workers", False))
        and workers > 0,
        prefetch_factor=(
            int(dataloader_cfg.get("prefetch_factor", 2)) if workers > 0 else None
        ),
        collate_fn=_list_collate,
    )

    rows: list[dict] = []
    started = time.perf_counter()
    progress = tqdm(total=CAPACITY_ABLATION_TRACKS, desc="capacity beam evaluation")
    for batch in loader:
        valid = [item for item in batch if item["error"] is None]
        for item in batch:
            if item["error"] is not None:
                query = item["query"]
                rows.append(
                    {
                        **query,
                        "status": "error",
                        "beam": [],
                        "correct_rank": None,
                        "reciprocal_rank": 0.0,
                        "error": item["error"],
                    }
                )
        if valid:
            batch_started = time.perf_counter()
            waveforms = torch.from_numpy(
                np.stack([item["waveform"] for item in valid])
            ).to(device)
            audio_tokens = tokenizer.tokenize(waveforms)
            if audio_tokens.shape != (len(valid), CAPACITY_ABLATION_AUDIO_TOKENS):
                raise ValueError(
                    "MuQ produced unexpected capacity token shape: "
                    f"{tuple(audio_tokens.shape)}"
                )
            prompts = prompts_from_audio_tokens(audio_tokens, vocabulary)
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if str(device).startswith("cuda")
                else nullcontext()
            )
            with autocast:
                rankings = batched_beam_generate(
                    model,
                    prompts,
                    vocabulary,
                    width=CAPACITY_ABLATION_BEAM_WIDTH,
                )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            latency = (time.perf_counter() - batch_started) / len(valid)
            for item, ranking in zip(valid, rankings, strict=True):
                query = item["query"]
                codes = [candidate.code for candidate in ranking]
                rank = codes.index(query["code"]) + 1 if query["code"] in codes else None
                rows.append(
                    {
                        **query,
                        "status": "ok",
                        "beam": [
                            {
                                "code": candidate.code,
                                "log_probability": candidate.log_probability,
                                "ended_with_eos": candidate.ended_with_eos,
                            }
                            for candidate in ranking
                        ],
                        "correct_rank": rank,
                        "reciprocal_rank": reciprocal_rank(codes, query["code"]),
                        "latency_seconds": latency,
                        "error": None,
                    }
                )
        progress.update(len(batch))
    progress.close()
    elapsed = time.perf_counter() - started
    order = {row["track_id"]: index for index, row in enumerate(query_manifest["queries"])}
    rows.sort(key=lambda row: order[row["track_id"]])
    metrics = aggregate_capacity_queries(rows, CAPACITY_ABLATION_TRACKS)
    metrics.update(
        {
            "elapsed_seconds": elapsed,
            "queries_per_second": CAPACITY_ABLATION_TRACKS / elapsed,
        }
    )
    result = {
        "protocol": CAPACITY_ABLATION_PROTOCOL,
        "database_size": database_size,
        "decoder": decoder,
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_profile": profile,
        "code_mapping_fingerprint": checkpoint.get("code_mapping_fingerprint"),
        "training_corpus_fingerprint": checkpoint.get("training_corpus_fingerprint"),
        "tokenizer_fingerprint": checkpoint["tokenizer_fingerprint"],
        "query_manifest": str(manifest_path),
        "query_manifest_fingerprint": query_manifest["fingerprint"],
        "metrics": metrics,
        "queries": rows,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(result_path)
    return {
        "result": str(result_path),
        "database_size": database_size,
        "decoder": decoder,
        **metrics,
    }
