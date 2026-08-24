from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from para_audio_id.catalogue import load_catalogue
from para_audio_id.config import load_config
from para_audio_id.audio_lm.profiles import catalogue_fingerprint, resolve_training_config


def prepare_training_cohort(cfg: dict, output: str | Path | None = None) -> dict:
    data_cfg = cfg["data"]
    records = load_catalogue(data_cfg["catalogue"])
    count = int(data_cfg["max_training_tracks"])
    if count > len(records):
        raise ValueError(f"Requested {count} tracks from a {len(records)}-track catalogue")
    ordered = sorted(records, key=lambda record: record.track_id)
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    indices = rng.choice(len(ordered), size=count, replace=False)
    selected = [ordered[int(index)] for index in indices]
    track_ids = [record.track_id for record in selected]
    if len(track_ids) != len(set(track_ids)) or len({record.code for record in selected}) != count:
        raise RuntimeError("Selected cohort does not have unique identities and codes")
    destination = Path(output or data_cfg["training_tracks_manifest"])
    payload = {
        "protocol": "fresh_seeded_catalogue_cohort_v1",
        "seed": int(cfg["train"]["seed"]),
        "database_size": count,
        "count": count,
        "catalogue_fingerprint": catalogue_fingerprint(records),
        "track_ids": track_ids,
        "code_mapping_fingerprint": hashlib.sha256(
            "\n".join(
                f"{record.track_id}:{record.code}" for record in selected
            ).encode()
        ).hexdigest(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the deterministic training-track cohort for audio-LM training."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--database-size", type=int, choices=(25_000, 100_000)
    )
    args = parser.parse_args()
    result = prepare_training_cohort(
        resolve_training_config(
            load_config(args.config), database_size=args.database_size
        ),
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
