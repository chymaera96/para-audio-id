from __future__ import annotations

import argparse
import json
from pathlib import Path

from para_audio_id.audio_lm.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an audio causal LM checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-tracks", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()
    metrics = evaluate(
        args.checkpoint,
        output=args.output,
        max_tracks=args.max_tracks,
        device=args.device,
        beam_width=args.beam_width,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
