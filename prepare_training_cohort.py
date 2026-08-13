from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from para_audio_id.catalogue import load_catalogue
from para_audio_id.config import load_config
from para_audio_id.audio_lm.profiles import (
    catalogue_fingerprint,
    resolve_capacity_config,
    resolve_training_config,
    validate_cohort_manifest,
)


def prepare_training_cohort(cfg: dict, output: str | Path | None = None) -> dict:
    data_cfg = cfg["data"]
    records = load_catalogue(data_cfg["catalogue"])
    count = int(data_cfg["max_training_tracks"])
    if count > len(records):
        raise ValueError(f"Requested {count} tracks from a {len(records)}-track catalogue")
    destination = Path(output or data_cfg["training_tracks_manifest"])
    if destination.exists():
        validate_cohort_manifest(destination, records, count)
        existing = json.loads(destination.read_text())
        if isinstance(existing, list):
            return {
                "protocol": "legacy_track_id_list_v1",
                "count": len(existing),
                "track_ids": existing,
                "reused_without_rewrite": True,
            }
        return existing
    ordered = sorted(records, key=lambda record: record.track_id)
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    indices = rng.choice(len(ordered), size=count, replace=False)
    selected = [ordered[int(index)] for index in indices]
    track_ids = [record.track_id for record in selected]
    if len(track_ids) != len(set(track_ids)) or len({record.code for record in selected}) != count:
        raise RuntimeError("Selected cohort does not have unique identities and codes")
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
        "--database-size",
        type=int,
        choices=(10_000, 25_000, 50_000, 100_000),
        help="Override data.database_size for capacity cohort preparation.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.database_size is not None:
        if "target_exposures" not in config.get("train", {}):
            raise ValueError("--database-size is only supported by capacity configs")
        config.setdefault("data", {})["database_size"] = args.database_size
    resolver = (
        resolve_capacity_config
        if "target_exposures" in config.get("train", {})
        else resolve_training_config
    )
    result = prepare_training_cohort(resolver(config), args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
