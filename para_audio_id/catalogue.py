from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from .codes import assign_codes

AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


@dataclass(frozen=True)
class CatalogueRecord:
    path: str
    track_id: str
    code: str
    duration: float


def find_audio(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def probe_duration(path: Path) -> float:
    info = sf.info(path)
    if info.samplerate <= 0 or info.frames <= 0:
        raise ValueError("empty or invalid audio stream")
    return info.frames / info.samplerate


def stable_track_id(relative_path: str) -> str:
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:20]


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_catalogue(
    audio_root: str | Path,
    output: str | Path,
    *,
    selection_seed: int = 1337,
    code_seed: int = 1337,
    count: int = 100_000,
    bad_files: str | Path | None = None,
) -> list[CatalogueRecord]:
    if count != 100_000:
        raise ValueError("Initial implementation assigns the complete 100,000-code space")
    root = Path(audio_root).resolve()
    candidates = find_audio(root)
    rng = np.random.default_rng(selection_seed)
    order = rng.permutation(len(candidates))
    good: list[tuple[str, float]] = []
    rejected: list[dict] = []
    for index in tqdm(order, desc="validating catalogue audio"):
        path = candidates[int(index)]
        relative = path.relative_to(root).as_posix()
        try:
            duration = probe_duration(path)
            if duration < 5.0:
                raise ValueError("shorter than a 5-second query")
            good.append((relative, duration))
            if len(good) == count:
                break
        except Exception as exc:
            rejected.append({"path": relative, "error": f"{type(exc).__name__}: {exc}"})
    if len(good) < count:
        raise RuntimeError(f"Only {len(good):,} valid tracks found; need {count:,}")
    codes = assign_codes(count, code_seed)
    records = [
        CatalogueRecord(path=relative, track_id=stable_track_id(relative), code=code, duration=duration)
        for (relative, duration), code in zip(good, codes, strict=True)
    ]
    output_path = Path(output)
    write_jsonl([asdict(record) for record in records], output_path)
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "audio_root": str(root),
                "count": count,
                "selection_seed": selection_seed,
                "code_seed": code_seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if bad_files is not None:
        write_jsonl(rejected, Path(bad_files))
    return records


def load_catalogue(path: str | Path) -> list[CatalogueRecord]:
    with Path(path).open("r", encoding="utf-8") as handle:
        records = [CatalogueRecord(**json.loads(line)) for line in handle if line.strip()]
    codes = {record.code for record in records}
    if len(records) != len(codes):
        raise ValueError("Catalogue codes are not unique")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a validated 100k FMA catalogue.")
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bad-files", type=Path, default=None)
    parser.add_argument("--selection-seed", type=int, default=1337)
    parser.add_argument("--code-seed", type=int, default=1337)
    args = parser.parse_args()
    prepare_catalogue(
        args.audio_root,
        args.output,
        selection_seed=args.selection_seed,
        code_seed=args.code_seed,
        bad_files=args.bad_files,
    )


if __name__ == "__main__":
    main()
