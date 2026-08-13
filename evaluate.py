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
        "--protocol",
        choices=("segment", "joint-beam"),
        default="segment",
        help="Use legacy segment evaluation or paper-facing joint-query decoding.",
    )
    parser.add_argument(
        "--cohort",
        choices=("probe", "training"),
        default=None,
        help=(
            "Evaluate the fixed probe or training cohort. Defaults to training for "
            "joint-beam and probe for segment evaluation."
        ),
    )
    parser.add_argument(
        "--expected-tracks",
        type=int,
        help="Fail unless the selected checkpoint cohort has exactly this many tracks.",
    )
    parser.add_argument("--max-tracks", type=int)
    parser.add_argument(
        "--sample-tracks",
        type=int,
        help="Evaluate a seeded random subset of the selected cohort.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1337,
        help="Random-subset seed used with --sample-tracks (default: 1337).",
    )
    parser.add_argument(
        "--recipe-seed",
        type=int,
        default=1337,
        help="Seed for joint-query starts and held-out room IRs (default: 1337).",
    )
    parser.add_argument(
        "--query-lengths",
        type=float,
        nargs="+",
        default=(2.0, 3.0, 5.0, 10.0),
        help="Paper-facing query lengths in seconds.",
    )
    parser.add_argument(
        "--conditions",
        choices=("clean", "rir"),
        nargs="+",
        default=("clean", "rir"),
        help="Paper-facing waveform conditions.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument(
        "--rir-training-root",
        type=Path,
        help="Optional room-IR training root used to verify held-out separation.",
    )
    parser.add_argument(
        "--rir-validation-root",
        type=Path,
        help="Optional held-out room-IR root for joint-beam RIR evaluation.",
    )
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help="Run only unconstrained five-digit autoregressive greedy evaluation.",
    )
    args = parser.parse_args()
    metrics = evaluate(
        args.checkpoint,
        output=args.output,
        protocol=args.protocol,
        cohort=(
            args.cohort
            if args.cohort is not None
            else ("training" if args.protocol == "joint-beam" else "probe")
        ),
        expected_tracks=args.expected_tracks,
        max_tracks=args.max_tracks,
        sample_tracks=args.sample_tracks,
        sample_seed=args.sample_seed,
        device=args.device,
        beam_width=None if args.greedy_only else args.beam_width,
        generation_only=args.greedy_only,
        recipe_seed=args.recipe_seed,
        query_lengths=tuple(args.query_lengths),
        conditions=tuple(args.conditions),
        rir_training_root=args.rir_training_root,
        rir_validation_root=args.rir_validation_root,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
