from __future__ import annotations

import argparse
import json
from pathlib import Path

from para_audio_id.audio_lm.tokenization import tokenize_catalogue
from para_audio_id.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize the FMA catalogue into MuQ RVQ shards.")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(tokenize_catalogue(load_config(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
