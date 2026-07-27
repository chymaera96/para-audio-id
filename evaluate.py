from __future__ import annotations

import argparse
import json
from pathlib import Path

from para_audio_id.audio_lm.evaluation import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an audio causal LM checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--cohort",
        choices=("probe", "training"),
        default="probe",
        help="Evaluate the fixed probe or every track embedded in the training subset.",
    )
    parser.add_argument(
        "--expected-tracks",
        type=int,
        help="Fail unless the selected checkpoint cohort has exactly this many tracks.",
    )
    parser.add_argument("--max-tracks", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help="Run only unconstrained five-digit autoregressive greedy evaluation.",
    )
    args = parser.parse_args()
    metrics = evaluate(
        args.checkpoint,
        output=args.output,
        cohort=args.cohort,
        expected_tracks=args.expected_tracks,
        max_tracks=args.max_tracks,
        device=args.device,
        beam_width=None if args.greedy_only else args.beam_width,
        generation_only=args.greedy_only,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
