from __future__ import annotations

import argparse
import json
from pathlib import Path

from para_audio_id.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a parametric identifier checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--degraded", action="store_true")
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.checkpoint,
                output=args.output,
                degraded=args.degraded,
                max_queries=args.max_queries,
                device=args.device,
                beam_width=args.beam_width,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
