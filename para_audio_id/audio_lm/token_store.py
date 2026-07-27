from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TokenRecord:
    document_index: int
    track_id: str
    code: str
    source_path: str
    segment_start: float
    segment_duration: float
    status: str
    token_offset: int
    token_count: int
    frames: int
    error: str | None = None
    view_type: str = "canonical"
    corpus_role: str = "canonical_training"
    source_duration: float | None = None
    padded_samples: int = 0


@dataclass(frozen=True)
class IndexedTokenRecord(TokenRecord):
    shard_id: int = 0
    token_file: str = ""


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_shard(
    root: str | Path,
    shard_id: int,
    *,
    records: list[TokenRecord],
    tokens: np.ndarray,
    tokenizer_spec: dict,
    tokenizer_fingerprint: str,
    corpus_role: str = "canonical_training",
) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{shard_id:06d}"
    token_path = root / f"{stem}.tokens.npy"
    index_path = root / f"{stem}.index.jsonl"
    metadata_path = root / f"{stem}.meta.json"
    if tokens.dtype != np.uint16 or tokens.ndim != 1:
        raise ValueError("Shard tokens must be a flat uint16 array")
    expected = sum(record.token_count for record in records if record.status == "ok")
    if expected != len(tokens):
        raise ValueError(f"Manifest accounts for {expected} tokens, received {len(tokens)}")

    temporary_tokens = token_path.with_suffix(token_path.suffix + ".tmp")
    with temporary_tokens.open("wb") as handle:
        np.save(handle, tokens, allow_pickle=False)
    os.replace(temporary_tokens, token_path)
    _atomic_jsonl(index_path, (asdict(record) for record in records))
    _atomic_json(
        metadata_path,
        {
            "architecture": "audio_lm_token_shard_v1",
            "complete": True,
            "shard_id": shard_id,
            "documents": len(records),
            "successful_documents": sum(record.status == "ok" for record in records),
            "failed_documents": sum(record.status != "ok" for record in records),
            "tokens": len(tokens),
            "token_dtype": "uint16",
            "tokenizer": tokenizer_spec,
            "tokenizer_fingerprint": tokenizer_fingerprint,
            "corpus_role": corpus_role,
        },
    )


def validate_shard(
    root: str | Path,
    shard_id: int,
    *,
    tokenizer_fingerprint: str,
    corpus_role: str | None = None,
) -> dict:
    root = Path(root)
    stem = f"shard-{shard_id:06d}"
    token_path = root / f"{stem}.tokens.npy"
    index_path = root / f"{stem}.index.jsonl"
    metadata_path = root / f"{stem}.meta.json"
    if not token_path.exists() or not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Incomplete token shard {shard_id}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("architecture") != "audio_lm_token_shard_v1" or not metadata.get(
        "complete"
    ):
        raise ValueError(f"Shard {shard_id} is not a complete audio-LM shard")
    if metadata.get("tokenizer_fingerprint") != tokenizer_fingerprint:
        raise ValueError(f"Shard {shard_id} tokenizer fingerprint does not match")
    actual_role = metadata.get("corpus_role", "canonical_training")
    if corpus_role is not None and actual_role != corpus_role:
        raise ValueError(
            f"Shard {shard_id} corpus role {actual_role!r} does not match "
            f"{corpus_role!r}"
        )
    tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    if tokens.dtype != np.uint16 or tokens.ndim != 1 or len(tokens) != metadata["tokens"]:
        raise ValueError(f"Shard {shard_id} token array is inconsistent with metadata")
    tokenizer = metadata.get("tokenizer", {})
    audio_vocabulary_size = int(tokenizer.get("selected_codebooks", 0)) * int(
        tokenizer.get("codebook_size", 0)
    )
    if audio_vocabulary_size and len(tokens) and int(tokens.max()) >= audio_vocabulary_size:
        raise ValueError(f"Shard {shard_id} contains an out-of-vocabulary audio token")
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line]
    if len(rows) != metadata["documents"]:
        raise ValueError(f"Shard {shard_id} index row count is inconsistent")
    for row in rows:
        row_role = row.get("corpus_role", "canonical_training")
        if row_role != actual_role:
            raise ValueError(
                f"Shard {shard_id} row corpus role {row_role!r} does not match "
                f"metadata role {actual_role!r}"
            )
        if row["status"] == "ok":
            end = row["token_offset"] + row["token_count"]
            if row["token_offset"] < 0 or end > len(tokens):
                raise ValueError(f"Shard {shard_id} contains an out-of-range token span")
    return metadata


class TokenStoreIndex:
    def __init__(
        self,
        root: str | Path,
        *,
        tokenizer_fingerprint: str,
        corpus_role: str | None = None,
    ):
        self.root = Path(root)
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.corpus_role = corpus_role
        self.records: list[IndexedTokenRecord] = []
        self._arrays: dict[int, np.ndarray] = {}
        metadata_paths = sorted(self.root.glob("shard-*.meta.json"))
        if not metadata_paths:
            raise FileNotFoundError(f"No token shards found under {self.root}")
        for metadata_path in metadata_paths:
            shard_id = int(metadata_path.name.split("-")[1].split(".")[0])
            metadata = validate_shard(
                self.root,
                shard_id,
                tokenizer_fingerprint=tokenizer_fingerprint,
                corpus_role=corpus_role,
            )
            actual_role = metadata.get("corpus_role", "canonical_training")
            if self.corpus_role is None:
                self.corpus_role = actual_role
            elif self.corpus_role != actual_role:
                raise ValueError("Token store mixes incompatible corpus roles")
            stem = f"shard-{shard_id:06d}"
            for line in (self.root / f"{stem}.index.jsonl").read_text().splitlines():
                if not line:
                    continue
                row = json.loads(line)
                if row["status"] == "ok":
                    self.records.append(
                        IndexedTokenRecord(
                            **row,
                            shard_id=shard_id,
                            token_file=f"{stem}.tokens.npy",
                        )
                    )
        self.records.sort(key=lambda record: record.document_index)
        if not self.records:
            raise RuntimeError("Token store contains no successful documents")

    def tokens(self, record: IndexedTokenRecord) -> np.ndarray:
        if record.shard_id not in self._arrays:
            self._arrays[record.shard_id] = np.load(
                self.root / record.token_file, mmap_mode="r", allow_pickle=False
            )
        array = self._arrays[record.shard_id]
        return np.asarray(
            array[record.token_offset : record.token_offset + record.token_count],
            dtype=np.int64,
        )
