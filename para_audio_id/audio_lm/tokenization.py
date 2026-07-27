from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from ..audio import BadFileRegistry, load_audio
from ..catalogue import CatalogueRecord, load_catalogue
from .token_store import TokenRecord, validate_shard, write_shard
from .tokenizer import MuQRVQTokenizer


@dataclass(frozen=True)
class CatalogueDocument:
    document_index: int
    record: CatalogueRecord
    start: float
    duration: float
    view_type: str = "canonical"
    corpus_role: str = "canonical_training"


def catalogue_documents(
    records: list[CatalogueRecord],
    *,
    segment_duration: float,
    segments_per_track: int,
) -> list[CatalogueDocument]:
    documents = []
    for record in records:
        for segment_index in range(segments_per_track):
            documents.append(
                CatalogueDocument(
                    document_index=len(documents),
                    record=record,
                    start=segment_index * segment_duration,
                    duration=segment_duration,
                )
            )
    return documents


def _load_document(
    document: CatalogueDocument,
    *,
    audio_root: Path,
    sample_rate: int,
) -> np.ndarray:
    audio = load_audio(
        audio_root / document.record.path,
        sample_rate=sample_rate,
        start=document.start,
        duration=document.duration,
        pad=True,
    )
    expected = round(document.duration * sample_rate)
    if len(audio) != expected or not np.isfinite(audio).all():
        raise ValueError("Decoded waveform has invalid length or samples")
    return audio


def _tokenize_loaded(
    tokenizer: MuQRVQTokenizer,
    loaded: list[tuple[CatalogueDocument, np.ndarray]],
) -> list[tuple[CatalogueDocument, np.ndarray, int]]:
    if not loaded:
        return []
    waveforms = torch.from_numpy(np.stack([audio for _, audio in loaded]))
    encoded = tokenizer.tokenize(waveforms).cpu().numpy()
    frames = encoded.shape[1] // tokenizer.selected_codebooks
    return [
        (document, np.asarray(tokens, dtype=np.uint16), frames)
        for (document, _), tokens in zip(loaded, encoded, strict=True)
    ]


def load_training_track_ids(path: str | Path, *, expected_count: int) -> list[str]:
    payload = json.loads(Path(path).read_text())
    track_ids = payload.get("track_ids") if isinstance(payload, dict) else payload
    if not isinstance(track_ids, list) or not all(
        isinstance(track_id, str) for track_id in track_ids
    ):
        raise ValueError("Training-track manifest must be a JSON list of track IDs")
    if len(track_ids) != expected_count or len(set(track_ids)) != expected_count:
        raise ValueError(
            f"Training-track manifest must contain {expected_count} unique IDs"
        )
    return track_ids


def view_policy_fingerprint(
    *,
    track_ids: list[str],
    starts: list[float],
    duration: float,
    corpus_role: str,
    tokenizer_fingerprint: str,
) -> str:
    payload = {
        "track_ids": track_ids,
        "starts": starts,
        "duration": duration,
        "corpus_role": corpus_role,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "policy_version": 1,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_view_document(
    document: CatalogueDocument,
    *,
    audio_root: Path,
    sample_rate: int,
) -> tuple[np.ndarray, int]:
    audio = _load_document(
        document,
        audio_root=audio_root,
        sample_rate=sample_rate,
    )
    available = max(0.0, document.record.duration - document.start)
    decoded_samples = min(
        round(document.duration * sample_rate),
        max(0, round(available * sample_rate)),
    )
    return audio, max(0, len(audio) - decoded_samples)


def _tokenize_view_store(
    *,
    tokenizer: MuQRVQTokenizer,
    root: Path,
    documents: list[CatalogueDocument],
    track_ids: list[str],
    starts: list[float],
    duration: float,
    corpus_role: str,
    view_type: str,
    audio_root: Path,
    bad: BadFileRegistry,
    shard_documents: int,
    extraction_batch_size: int,
) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    policy_fingerprint = view_policy_fingerprint(
        track_ids=track_ids,
        starts=starts,
        duration=duration,
        corpus_role=corpus_role,
        tokenizer_fingerprint=tokenizer.spec.fingerprint,
    )
    codes = {
        document.record.track_id: document.record.code for document in documents
    }
    code_fingerprint = hashlib.sha256(
        "\n".join(f"{track_id}:{codes[track_id]}" for track_id in track_ids).encode()
    ).hexdigest()
    spec_payload = {
        "tokenizer": tokenizer.spec.to_dict(),
        "fingerprint": tokenizer.spec.fingerprint,
        "vocabulary": tokenizer.vocabulary.to_dict(),
        "corpus_role": corpus_role,
        "view_type": view_type,
        "view_policy_fingerprint": policy_fingerprint,
        "starts": starts,
        "duration": duration,
        "track_ids": track_ids,
        "code_mapping_fingerprint": code_fingerprint,
    }
    spec_path = root / "tokenizer_spec.json"
    if spec_path.exists():
        if json.loads(spec_path.read_text()) != spec_payload:
            raise ValueError(f"Existing view-token root {root} is incompatible")
    else:
        temporary = spec_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(spec_payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(spec_path)

    shard_count = math.ceil(len(documents) / shard_documents)
    for shard_id in tqdm(range(shard_count), desc=f"{corpus_role} shards"):
        try:
            validate_shard(
                root,
                shard_id,
                tokenizer_fingerprint=tokenizer.spec.fingerprint,
                corpus_role=corpus_role,
            )
            continue
        except FileNotFoundError:
            pass
        shard_slice = documents[
            shard_id * shard_documents : (shard_id + 1) * shard_documents
        ]
        rows: list[TokenRecord] = []
        token_parts: list[np.ndarray] = []
        offset = 0
        for batch_start in range(0, len(shard_slice), extraction_batch_size):
            batch = shard_slice[batch_start : batch_start + extraction_batch_size]
            loaded: list[tuple[CatalogueDocument, np.ndarray]] = []
            padding: dict[int, int] = {}
            for document in batch:
                if bad.contains(document.record.path):
                    rows.append(
                        TokenRecord(
                            document_index=document.document_index,
                            track_id=document.record.track_id,
                            code=document.record.code,
                            source_path=document.record.path,
                            segment_start=document.start,
                            segment_duration=document.duration,
                            status="failed",
                            token_offset=offset,
                            token_count=0,
                            frames=0,
                            error="Previously registered bad source file",
                            view_type=view_type,
                            corpus_role=corpus_role,
                            source_duration=document.record.duration,
                        )
                    )
                    continue
                try:
                    audio, padded_samples = _load_view_document(
                        document,
                        audio_root=audio_root,
                        sample_rate=tokenizer.sample_rate,
                    )
                    loaded.append((document, audio))
                    padding[document.document_index] = padded_samples
                except Exception as exc:
                    bad.add(document.record.path, exc)
                    rows.append(
                        TokenRecord(
                            document_index=document.document_index,
                            track_id=document.record.track_id,
                            code=document.record.code,
                            source_path=document.record.path,
                            segment_start=document.start,
                            segment_duration=document.duration,
                            status="failed",
                            token_offset=offset,
                            token_count=0,
                            frames=0,
                            error=f"{type(exc).__name__}: {exc}",
                            view_type=view_type,
                            corpus_role=corpus_role,
                            source_duration=document.record.duration,
                        )
                    )
            try:
                encoded = _tokenize_loaded(tokenizer, loaded)
            except Exception:
                encoded = []
                for item in loaded:
                    try:
                        encoded.extend(_tokenize_loaded(tokenizer, [item]))
                    except Exception as exc:
                        document = item[0]
                        rows.append(
                            TokenRecord(
                                document_index=document.document_index,
                                track_id=document.record.track_id,
                                code=document.record.code,
                                source_path=document.record.path,
                                segment_start=document.start,
                                segment_duration=document.duration,
                                status="failed",
                                token_offset=offset,
                                token_count=0,
                                frames=0,
                                error=f"{type(exc).__name__}: {exc}",
                                view_type=view_type,
                                corpus_role=corpus_role,
                                source_duration=document.record.duration,
                                padded_samples=padding[document.document_index],
                            )
                        )
            for document, tokens, frames in encoded:
                token_parts.append(tokens)
                rows.append(
                    TokenRecord(
                        document_index=document.document_index,
                        track_id=document.record.track_id,
                        code=document.record.code,
                        source_path=document.record.path,
                        segment_start=document.start,
                        segment_duration=document.duration,
                        status="ok",
                        token_offset=offset,
                        token_count=len(tokens),
                        frames=frames,
                        view_type=view_type,
                        corpus_role=corpus_role,
                        source_duration=document.record.duration,
                        padded_samples=padding[document.document_index],
                    )
                )
                offset += len(tokens)
        rows.sort(key=lambda row: row.document_index)
        combined = (
            np.concatenate(token_parts).astype(np.uint16, copy=False)
            if token_parts
            else np.empty(0, dtype=np.uint16)
        )
        write_shard(
            root,
            shard_id,
            records=rows,
            tokens=combined,
            tokenizer_spec=tokenizer.spec.to_dict(),
            tokenizer_fingerprint=tokenizer.spec.fingerprint,
            corpus_role=corpus_role,
        )
    metadata = [
        validate_shard(
            root,
            shard_id,
            tokenizer_fingerprint=tokenizer.spec.fingerprint,
            corpus_role=corpus_role,
        )
        for shard_id in range(shard_count)
    ]
    report = {
        "architecture": "audio_lm_view_tokenization_report_v1",
        "corpus_role": corpus_role,
        "view_type": view_type,
        "view_policy_fingerprint": policy_fingerprint,
        "code_mapping_fingerprint": code_fingerprint,
        "tracks": len(track_ids),
        "starts": starts,
        "intended_documents": len(documents),
        "successful_documents": sum(item["successful_documents"] for item in metadata),
        "failed_documents": sum(item["failed_documents"] for item in metadata),
        "shards": shard_count,
    }
    report["complete"] = (
        report["successful_documents"] + report["failed_documents"]
        == report["intended_documents"]
    )
    (root / "tokenization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def tokenize_paired_views(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    tokenizer_cfg = cfg["tokenizer"]
    expected_tracks = int(data_cfg["max_training_tracks"])
    track_ids = load_training_track_ids(
        data_cfg["training_tracks_manifest"], expected_count=expected_tracks
    )
    records = load_catalogue(data_cfg["catalogue"])
    by_track = {record.track_id: record for record in records}
    missing = sorted(set(track_ids) - by_track.keys())
    if missing:
        raise ValueError(f"Training-track manifest contains missing IDs: {missing[:5]}")
    tokenizer = MuQRVQTokenizer(
        tokenizer_cfg["model_name"],
        revision=tokenizer_cfg.get("revision", "main"),
        selected_codebooks=int(tokenizer_cfg["selected_codebooks"]),
        sample_rate=int(tokenizer_cfg["sample_rate"]),
        device=tokenizer_cfg.get("device", "cuda"),
    )
    duration = float(data_cfg["segment_duration"])
    audio_root = Path(data_cfg["audio_root"])
    bad = BadFileRegistry(data_cfg["runtime_bad_files"])
    reports = {}
    policies = (
        (
            "shifted_training",
            "shifted",
            [float(value) for value in data_cfg["shifted_training_starts"]],
            Path(data_cfg["shifted_training_token_root"]),
        ),
        (
            "heldout_evaluation",
            "heldout",
            [float(value) for value in data_cfg["shifted_evaluation_starts"]],
            Path(data_cfg["heldout_evaluation_token_root"]),
        ),
    )
    for corpus_role, view_type, starts, root in policies:
        documents = [
            CatalogueDocument(
                document_index=index,
                record=by_track[track_id],
                start=start,
                duration=duration,
                view_type=view_type,
                corpus_role=corpus_role,
            )
            for index, (track_id, start) in enumerate(
                (track_id, start) for track_id in track_ids for start in starts
            )
        ]
        reports[corpus_role] = _tokenize_view_store(
            tokenizer=tokenizer,
            root=root,
            documents=documents,
            track_ids=track_ids,
            starts=starts,
            duration=duration,
            corpus_role=corpus_role,
            view_type=view_type,
            audio_root=audio_root,
            bad=bad,
            shard_documents=int(data_cfg["shard_documents"]),
            extraction_batch_size=int(data_cfg["extraction_batch_size"]),
        )
    failures = {
        role: report["failed_documents"]
        for role, report in reports.items()
        if report["failed_documents"]
    }
    if failures:
        raise RuntimeError(
            "Exact paired-view cohort preparation has failed documents: "
            f"{failures}. Inspect each tokenization_report.json."
        )
    return reports


def tokenize_catalogue(cfg: dict) -> dict:
    if cfg.get("architecture") != "audio_lm_v1":
        raise ValueError("Configuration architecture must be audio_lm_v1")
    tokenizer_cfg = cfg["tokenizer"]
    data_cfg = cfg["data"]
    tokenizer = MuQRVQTokenizer(
        tokenizer_cfg["model_name"],
        revision=tokenizer_cfg.get("revision", "main"),
        selected_codebooks=int(tokenizer_cfg["selected_codebooks"]),
        sample_rate=int(tokenizer_cfg["sample_rate"]),
        device=tokenizer_cfg.get("device", "cuda"),
    )
    token_root = Path(data_cfg["token_root"])
    token_root.mkdir(parents=True, exist_ok=True)
    spec_path = token_root / "tokenizer_spec.json"
    spec_payload = {
        "tokenizer": tokenizer.spec.to_dict(),
        "fingerprint": tokenizer.spec.fingerprint,
        "vocabulary": tokenizer.vocabulary.to_dict(),
    }
    if spec_path.exists():
        existing = json.loads(spec_path.read_text())
        if existing != spec_payload:
            raise ValueError("Existing token root uses an incompatible tokenizer")
    else:
        temporary = spec_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(spec_payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(spec_path)

    records = load_catalogue(data_cfg["catalogue"])
    documents = catalogue_documents(
        records,
        segment_duration=float(data_cfg["segment_duration"]),
        segments_per_track=int(data_cfg["segments_per_track"]),
    )
    shard_documents = int(data_cfg["shard_documents"])
    extraction_batch_size = int(data_cfg["extraction_batch_size"])
    audio_root = Path(data_cfg["audio_root"])
    bad = BadFileRegistry(data_cfg["runtime_bad_files"])
    shard_count = math.ceil(len(documents) / shard_documents)

    probe_report = None
    for document in documents:
        if bad.contains(document.record.path):
            continue
        try:
            waveform = _load_document(
                document,
                audio_root=audio_root,
                sample_rate=tokenizer.sample_rate,
            )
            probe_report = tokenizer.probe(torch.from_numpy(waveform).unsqueeze(0))
            break
        except Exception as exc:
            bad.add(document.record.path, exc)
    if probe_report is None:
        raise RuntimeError("No readable catalogue document is available for the MuQ probe")
    causal_length = probe_report["serialized_tokens_per_example"] + 8
    maximum = int(cfg["model"]["max_position_embeddings"])
    if causal_length > maximum:
        raise RuntimeError(
            f"Measured causal document length {causal_length} exceeds context {maximum}"
        )
    probe_report["causal_document_length"] = causal_length
    probe_report["max_position_embeddings"] = maximum
    (token_root / "probe_report.json").write_text(
        json.dumps(probe_report, indent=2, sort_keys=True) + "\n"
    )

    for shard_id in tqdm(range(shard_count), desc="token shards"):
        try:
            validate_shard(
                token_root,
                shard_id,
                tokenizer_fingerprint=tokenizer.spec.fingerprint,
            )
            continue
        except FileNotFoundError:
            pass
        shard_slice = documents[
            shard_id * shard_documents : (shard_id + 1) * shard_documents
        ]
        rows: list[TokenRecord] = []
        token_parts: list[np.ndarray] = []
        offset = 0
        for batch_start in range(0, len(shard_slice), extraction_batch_size):
            batch = shard_slice[batch_start : batch_start + extraction_batch_size]
            loaded = []
            for document in batch:
                if bad.contains(document.record.path):
                    rows.append(
                        TokenRecord(
                            document_index=document.document_index,
                            track_id=document.record.track_id,
                            code=document.record.code,
                            source_path=document.record.path,
                            segment_start=document.start,
                            segment_duration=document.duration,
                            status="failed",
                            token_offset=offset,
                            token_count=0,
                            frames=0,
                            error="Previously registered bad source file",
                        )
                    )
                    continue
                try:
                    loaded.append(
                        (
                            document,
                            _load_document(
                                document,
                                audio_root=audio_root,
                                sample_rate=tokenizer.sample_rate,
                            ),
                        )
                    )
                except Exception as exc:
                    bad.add(document.record.path, exc)
                    rows.append(
                        TokenRecord(
                            document_index=document.document_index,
                            track_id=document.record.track_id,
                            code=document.record.code,
                            source_path=document.record.path,
                            segment_start=document.start,
                            segment_duration=document.duration,
                            status="failed",
                            token_offset=offset,
                            token_count=0,
                            frames=0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            try:
                encoded = _tokenize_loaded(tokenizer, loaded)
            except Exception:
                encoded = []
                for item in loaded:
                    document = item[0]
                    try:
                        encoded.extend(_tokenize_loaded(tokenizer, [item]))
                    except Exception as exc:
                        rows.append(
                            TokenRecord(
                                document_index=document.document_index,
                                track_id=document.record.track_id,
                                code=document.record.code,
                                source_path=document.record.path,
                                segment_start=document.start,
                                segment_duration=document.duration,
                                status="failed",
                                token_offset=offset,
                                token_count=0,
                                frames=0,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
            for document, tokens, frames in encoded:
                token_parts.append(tokens)
                rows.append(
                    TokenRecord(
                        document_index=document.document_index,
                        track_id=document.record.track_id,
                        code=document.record.code,
                        source_path=document.record.path,
                        segment_start=document.start,
                        segment_duration=document.duration,
                        status="ok",
                        token_offset=offset,
                        token_count=len(tokens),
                        frames=frames,
                    )
                )
                offset += len(tokens)
        rows.sort(key=lambda row: row.document_index)
        combined = (
            np.concatenate(token_parts).astype(np.uint16, copy=False)
            if token_parts
            else np.empty(0, dtype=np.uint16)
        )
        write_shard(
            token_root,
            shard_id,
            records=rows,
            tokens=combined,
            tokenizer_spec=tokenizer.spec.to_dict(),
            tokenizer_fingerprint=tokenizer.spec.fingerprint,
        )

    metadata = [
        validate_shard(
            token_root, shard_id, tokenizer_fingerprint=tokenizer.spec.fingerprint
        )
        for shard_id in range(shard_count)
    ]
    report = {
        "architecture": "audio_lm_tokenization_report_v1",
        "tokenizer_fingerprint": tokenizer.spec.fingerprint,
        "catalogue_tracks": len(records),
        "intended_documents": len(documents),
        "successful_documents": sum(item["successful_documents"] for item in metadata),
        "failed_documents": sum(item["failed_documents"] for item in metadata),
        "tokens": sum(item["tokens"] for item in metadata),
        "shards": shard_count,
    }
    report["complete"] = (
        report["successful_documents"] + report["failed_documents"]
        == report["intended_documents"]
    )
    (token_root / "tokenization_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report
