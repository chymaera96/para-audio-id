from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
ROOM_TRAIN = {"meeting", "lecture", "stairway", "office"}
ROOM_TEST = {"aula_carolina", "booth"}
OPENAIR_ROOMS = {
    "elveden-hall-suffolk-england",
    "falkland-palace-royal-tennis-court",
    "gill-heads-mine",
    "hamilton-mausoleum",
    "innocent-railway-tunnel",
    "koli-national-park-summer",
    "koli-national-park-winter",
    "lady-chapel-st-albans-cathedral",
    "maes-howe",
    "r1-nuclear-reactor-hall",
    "saint-lawrence-church-molenbeek-wersbeek-belgium",
    "shrine-and-parish-church-all-saints-north-street-_",
    "spokane-womans-club",
    "sports-centre-university-york",
    "spring-lane-building-university-york",
    "st-andrews-church",
    "st-margarets-church-ncem-5-piece-band-spatial-measurements",
    "st-matthews-church-walsall",
    "st-patricks-church-patrington",
    "stairway-university-york",
    "terrys-factory-warehouse",
    "terrys-typing-room",
    "trollers-gill",
    "tyndall-bruce-monument",
    "usina-del-arte-symphony-hall",
    "waveguide-web-example-audio",
    "wheldrake-wood",
    "york-minster",
}
EXPECTED_COLLECTION_COUNTS = {
    "TUT2016": 1560,
    "MIT": 270,
    "OpenAIR": 143,
    "AachenAIR": 60,
    "SurreyMicrophoneIR": 708,
}
MICROPHONES = (
    "AKGC414",
    "AKGC451",
    "AKGD112",
    "AKGD12",
    "Coles4038",
    "DPA4006",
    "ElectroVoiceRE20",
    "NTiM2211",
    "NeumannU47FET",
    "NeumannU87Ai",
    "RodeK2",
    "RodeNT2A",
    "RodeNTG8",
    "RodeNTR",
    "RodeReporter",
    "RoyerR121",
    "SchoepsCMC5U",
    "SchoepsCMC6",
    "SchoepsCMIT5U",
    "SennheiserMD441",
    "ShureBeta52",
    "ShureSM57",
    "ShureSM58",
    "SonyC800",
    "SonyECM670",
)


@dataclass(frozen=True)
class Source:
    path: Path
    dataset: str
    split: str
    collection: str
    relative: Path
    channel: int | None = None


@dataclass(frozen=True)
class Conversion:
    dataset: str
    split: str
    collection: str
    source: str
    output: str
    source_sample_rate: int
    output_sample_rate: int
    source_channels: int
    selected_channel: int | None
    duration_seconds: float
    status: str


def audio_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def read_input(path: Path) -> tuple[np.ndarray, int]:
    if path.suffix.lower() == ".mat":
        contents = loadmat(path)
        if "h_air" not in contents:
            raise ValueError("Aachen MAT file has no h_air array")
        audio = np.asarray(contents["h_air"], dtype=np.float32).squeeze()
        if audio.ndim == 1:
            audio = audio[:, None]
        elif audio.ndim == 2 and audio.shape[0] <= 2 < audio.shape[1]:
            audio = audio.T
        if audio.ndim != 2:
            raise ValueError(f"Unexpected Aachen h_air shape: {audio.shape}")
        return audio, 48_000
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio, int(sample_rate)


def duration(path: Path) -> float:
    return float(sf.info(path).duration)


def collect_background(raw_root: Path) -> list[Source]:
    sources = []
    for path in audio_files(raw_root / "tut"):
        lower = path.as_posix().lower()
        if "development" in lower:
            split = "train"
        elif "evaluation" in lower:
            split = "test"
        else:
            continue
        sources.append(
            Source(path, "bg_noise", split, "TUT2016", Path("TUT2016") / path.name)
        )
    return sources


def duration_label(value: float, boundaries: tuple[float, ...]) -> str:
    return str(next((index for index, edge in enumerate(boundaries) if value < edge), len(boundaries)))


def merge_rare_numeric_labels(labels: list[str], minimum: int = 2) -> list[str]:
    """Merge undersized ordered duration bins into their nearest populated neighbor."""
    merged = list(labels)
    while True:
        counts = Counter(merged)
        rare = [label for label, count in counts.items() if count < minimum]
        if not rare:
            return merged
        label = min(rare, key=int)
        candidates = [candidate for candidate in counts if candidate != label]
        if not candidates:
            return merged
        replacement = min(candidates, key=lambda candidate: (abs(int(candidate) - int(label)), int(candidate)))
        merged = [replacement if value == label else value for value in merged]


def collect_mit(raw_root: Path) -> list[Source]:
    paths = audio_files(raw_root / "mit")
    labels = {
        str(path): duration_label(duration(path), (0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75))
        for path in paths
    }
    _, test_items = train_test_split(
        sorted(labels),
        test_size=0.2,
        random_state=27,
        stratify=[labels[item] for item in sorted(labels)],
    )
    test = set(test_items)
    return [
        Source(
            path,
            "room_ir",
            "test" if str(path) in test else "train",
            "MIT",
            Path("MIT") / path.name,
        )
        for path in paths
    ]


def openair_room(path: Path) -> str:
    parts = path.parts
    lowered = [part.lower() for part in parts]
    if "irs" in lowered:
        index = lowered.index("irs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return path.parent.name


def collect_openair(raw_root: Path) -> list[Source]:
    paths = []
    for path in audio_files(raw_root / "openair"):
        try:
            channels = sf.info(path).channels
        except (RuntimeError, sf.LibsndfileError):
            continue
        if channels in (1, 2):
            paths.append(path)
    rooms: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        room = openair_room(path)
        if room in OPENAIR_ROOMS:
            rooms[room].append(path)
    minimum = {room: min(duration(path) for path in values) for room, values in rooms.items()}
    labels = {
        room: duration_label(value, (3, 5, 5.5, 7, 16)) for room, value in minimum.items()
    }
    room_names = sorted(rooms)
    stratify_labels = merge_rare_numeric_labels([labels[room] for room in room_names])
    _, test_items = train_test_split(
        room_names,
        test_size=0.25,
        random_state=27,
        stratify=stratify_labels,
    )
    test_rooms = set(test_items)
    sources = []
    for room, values in rooms.items():
        split = "test" if room in test_rooms else "train"
        for path in values:
            channels = sf.info(path).channels
            for channel in range(channels):
                suffix = f"_ch{channel}" if channels > 1 else ""
                relative = Path("OpenAIR") / room / f"{path.stem}{suffix}.wav"
                sources.append(
                    Source(path, "room_ir", split, "OpenAIR", relative, channel)
                )
    return sources


def aachen_room(path: Path) -> str | None:
    name = path.name.lower()
    return next((room for room in ROOM_TRAIN | ROOM_TEST if room in name), None)


def collect_aachen(raw_root: Path) -> list[Source]:
    sources = []
    aachen_root = raw_root / "aachen"
    paths = audio_files(aachen_root)
    if not paths:
        paths = sorted(aachen_root.rglob("*.mat"))
    for path in paths:
        name = path.name.lower()
        room = aachen_room(path)
        if room is None or "binaural" not in name:
            continue
        if "dummy" in name or "kunstkopf" in name:
            continue
        split = "test" if room in ROOM_TEST else "train"
        channels = read_input(path)[0].shape[1]
        for channel in range(channels):
            suffix = f"_ch{channel}" if channels > 1 else ""
            relative = Path("AachenAIR") / room / f"{path.stem}{suffix}.wav"
            sources.append(
                Source(path, "room_ir", split, "AachenAIR", relative, channel)
            )
    return sources


def microphone_name(path: Path) -> str | None:
    joined = "/".join(path.parts)
    return next((name for name in MICROPHONES if name in joined), None)


def incident_angle(path: Path) -> int | None:
    matches = re.findall(r"(?:Deg|deg|angle[_-]?)(\d{1,3})|(\d{1,3})(?:Deg|deg)", path.name)
    if not matches:
        return None
    first = matches[-1]
    return int(first[0] or first[1])


def collect_microphones(raw_root: Path) -> list[Source]:
    candidates: list[tuple[Path, str]] = []
    for path in audio_files(raw_root / "microphone"):
        lower = path.as_posix().lower()
        if "normalised" not in lower and "normalized" not in lower:
            continue
        if "24bit" not in lower and "24-bit" not in lower:
            continue
        name = microphone_name(path)
        angle = incident_angle(path)
        if name is not None and angle is not None and angle % 60 == 0:
            candidates.append((path, name))
    mic_names = sorted({name for _, name in candidates})
    _, test_items = train_test_split(mic_names, test_size=0.2, random_state=27)
    test_names = set(test_items)
    return [
        Source(
            path,
            "microphone_ir",
            "test" if name in test_names else "train",
            "SurreyMicrophoneIR",
            Path("SurreyMicrophoneIR") / name / path.name,
        )
        for path, name in candidates
    ]


def collect_sources(raw_root: Path) -> list[Source]:
    collectors = (
        collect_background,
        collect_mit,
        collect_openair,
        collect_aachen,
        collect_microphones,
    )
    sources = [source for collector in collectors for source in collector(raw_root)]
    required = {
        ("bg_noise", "train"),
        ("bg_noise", "test"),
        ("room_ir", "train"),
        ("room_ir", "test"),
        ("microphone_ir", "train"),
        ("microphone_ir", "test"),
    }
    present = {(source.dataset, source.split) for source in sources}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"Source preparation produced empty required groups: {missing}")
    return sources


def valid_output(path: Path, sample_rate: int) -> bool:
    if not path.exists():
        return False
    try:
        info = sf.info(path)
    except (RuntimeError, sf.LibsndfileError):
        return False
    return info.samplerate == sample_rate and info.channels == 1 and info.frames > 0


def convert_one(task: tuple[Source, str, int, bool]) -> tuple[Conversion | None, dict | None]:
    source, output_string, sample_rate, force = task
    output = Path(output_string)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        input_audio, input_rate = read_input(source.path)
        source_channels = int(input_audio.shape[1])
        if valid_output(output, sample_rate) and not force:
            return (
                Conversion(
                    source.dataset,
                    source.split,
                    source.collection,
                    str(source.path),
                    str(output),
                    input_rate,
                    sample_rate,
                    source_channels,
                    source.channel,
                    float(sf.info(output).duration),
                    "existing",
                ),
                None,
            )
        audio, source_rate = input_audio, input_rate
        if audio.shape[0] == 0:
            raise ValueError("empty audio stream")
        if source.channel is None:
            mono = audio.mean(axis=1, dtype=np.float32)
        else:
            mono = audio[:, source.channel]
        if source_rate != sample_rate:
            mono = librosa.resample(
                mono,
                orig_sr=int(source_rate),
                target_sr=sample_rate,
                res_type="soxr_hq",
            )
        mono = np.asarray(mono, dtype=np.float32)
        if mono.size == 0 or not np.isfinite(mono).all():
            raise ValueError("invalid resampled audio")
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(temporary, mono, sample_rate, format="WAV", subtype="FLOAT")
        written = sf.info(temporary)
        if written.samplerate != sample_rate or written.channels != 1:
            raise RuntimeError("written output failed validation")
        temporary.replace(output)
        return (
            Conversion(
                source.dataset,
                source.split,
                source.collection,
                str(source.path),
                str(output),
                int(source_rate),
                sample_rate,
                int(audio.shape[1]),
                source.channel,
                float(len(mono) / sample_rate),
                "converted",
            ),
            None,
        )
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return (
            None,
            {
                "dataset": source.dataset,
                "split": source.split,
                "collection": source.collection,
                "source": str(source.path),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )


def write_jsonl(rows: list[dict], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    sample_rate: int = 24_000,
    workers: int = 8,
    force: bool = False,
    strict_counts: bool = False,
) -> tuple[list[Conversion], list[dict]]:
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root == raw_root or raw_root in output_root.parents:
        raise ValueError("output_root must be outside raw_root")
    output_root.mkdir(parents=True, exist_ok=True)
    sources = collect_sources(raw_root)
    actual_counts = {
        collection: sum(source.collection == collection for source in sources)
        for collection in EXPECTED_COLLECTION_COUNTS
    }
    mismatches = {
        collection: {"expected": expected, "actual": actual_counts[collection]}
        for collection, expected in EXPECTED_COLLECTION_COUNTS.items()
        if actual_counts[collection] != expected
    }
    if mismatches:
        message = (
            "Original-source selection differs from the processed neural-music-fp "
            f"release counts: {json.dumps(mismatches, sort_keys=True)}. "
            "This is expected when original stereo channels are retained separately "
            "or current provider layouts differ."
        )
        if strict_counts:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    tasks = [
        (
            source,
            str(output_root / source.dataset / source.split / source.relative),
            sample_rate,
            force,
        )
        for source in sources
    ]
    conversions: list[Conversion] = []
    failures: list[dict] = []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        results = executor.map(convert_one, tasks, chunksize=8)
        for conversion, failure in tqdm(results, total=len(tasks), desc="preparing degradation"):
            if conversion is not None:
                conversions.append(conversion)
            if failure is not None:
                failures.append(failure)
    write_jsonl([asdict(row) for row in conversions], output_root / "manifest.jsonl")
    write_jsonl(failures, output_root / "bad_files.jsonl")
    summary = {}
    for dataset in ("bg_noise", "room_ir", "microphone_ir"):
        summary[dataset] = {}
        for split in ("train", "test"):
            selected = [
                row for row in conversions if row.dataset == dataset and row.split == split
            ]
            summary[dataset][split] = {
                "files": len(selected),
                "converted": sum(row.status == "converted" for row in selected),
                "existing": sum(row.status == "existing" for row in selected),
                "failed": sum(
                    row["dataset"] == dataset and row["split"] == split for row in failures
                ),
                "collections": dict(
                    sorted(
                        (collection, sum(row.collection == collection for row in selected))
                        for collection in {row.collection for row in selected}
                    )
                ),
            }
    (output_root / "summary.json").write_text(
        json.dumps(
            {
                "raw_root": str(raw_root),
                "output_root": str(output_root),
                "sample_rate": sample_rate,
                "selection_seed": 27,
                "source_count_mismatches": mismatches,
                "datasets": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return conversions, failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile original degradation datasets into mono 24 kHz train/test trees."
    )
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--strict-counts",
        action="store_true",
        help="Fail instead of warning when source counts differ from the processed 8 kHz release.",
    )
    args = parser.parse_args()
    conversions, failures = prepare(
        args.raw_root,
        args.output_root,
        sample_rate=args.sample_rate,
        workers=args.workers,
        force=args.force,
        strict_counts=args.strict_counts,
    )
    converted = sum(row.status == "converted" for row in conversions)
    existing = sum(row.status == "existing" for row in conversions)
    print(f"converted={converted} existing={existing} failed={len(failures)}")
    if failures:
        raise SystemExit(
            f"{len(failures)} files failed; inspect {args.output_root / 'bad_files.jsonl'}"
        )


if __name__ == "__main__":
    main()
