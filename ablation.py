from __future__ import annotations

import argparse
import json

from para_audio_id.audio_lm.capacity_ablation import evaluate_capacity_ablation
from para_audio_id.audio_lm.profiles import DECODER_PROFILES, SUPPORTED_DATABASE_SIZES


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate clean two-second MRR for an eight-codebook capacity run."
    )
    parser.add_argument(
        "--database-size", type=int, required=True, choices=SUPPORTED_DATABASE_SIZES
    )
    parser.add_argument(
        "--decoder", required=True, choices=tuple(DECODER_PROFILES)
    )
    args = parser.parse_args()
    summary = evaluate_capacity_ablation(
        database_size=args.database_size,
        decoder=args.decoder,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
