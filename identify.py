from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from para_audio_id.audio import load_audio, quantile_normalize
from para_audio_id.checkpoint import load_network


def identify(checkpoint: Path, paths: list[Path], device: str = "cuda") -> list[dict]:
    network, cfg, _ = load_network(checkpoint, device)
    sample_rate = int(cfg["model"]["sample_rate"])
    duration = float(cfg["data"]["query_duration"])
    batch = torch.stack(
        [
            torch.from_numpy(
                quantile_normalize(
                    load_audio(
                        path, sample_rate=sample_rate, duration=duration, pad=True
                    ),
                    float(cfg["model"]["quantile_norm"]),
                )
            )
            for path in paths
        ]
    ).to(device)
    greedy = network.greedy_decode(batch)
    beams = network.beam_decode(batch, width=10)
    return [
        {
            "path": str(path),
            "greedy": greedy_code,
            "beam": [
                {"code": item.code, "log_probability": item.log_probability} for item in beam
            ],
        }
        for path, greedy_code, beam in zip(paths, greedy, beams, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify one or more audio queries.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(identify(args.checkpoint, args.audio, args.device), indent=2))


if __name__ == "__main__":
    main()
