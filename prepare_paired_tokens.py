from __future__ import annotations

import argparse
import json
from pathlib import Path

from para_audio_id.audio_lm.tokenization import tokenize_paired_views
from para_audio_id.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare shifted-training and held-out token stores."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    reports = tokenize_paired_views(load_config(args.config))
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
