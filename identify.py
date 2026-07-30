from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from para_audio_id.audio import load_audio
from para_audio_id.audio_lm.checkpoint import load_audio_lm
from para_audio_id.audio_lm.generation import (
    beam_generate,
    greedy_generate,
    prompt_from_audio_tokens,
)
from para_audio_id.audio_lm.tokenizer import MuQRVQTokenizer


def identify(
    checkpoint_path: str | Path,
    paths: list[Path],
    *,
    device: str = "cuda",
    beam_width: int = 10,
) -> list[dict]:
    model, vocabulary, cfg, checkpoint = load_audio_lm(checkpoint_path, device)
    spec = checkpoint["tokenizer_spec"]
    tokenizer = MuQRVQTokenizer(
        spec["model_name"],
        revision=spec["revision"],
        selected_codebooks=int(spec["selected_codebooks"]),
        sample_rate=int(spec["sample_rate"]),
        device=device,
        lightweight=True,
    )
    if tokenizer.spec.fingerprint != checkpoint["tokenizer_fingerprint"]:
        raise ValueError("Loaded MuQ tokenizer does not match the checkpoint")
    duration = float(cfg["data"]["segment_duration"])
    rows = []
    with torch.inference_mode():
        for path in paths:
            waveform = load_audio(
                path,
                sample_rate=tokenizer.sample_rate,
                duration=duration,
                pad=True,
            )
            audio_tokens = tokenizer.tokenize(torch.from_numpy(waveform).unsqueeze(0))[0]
            prompt = prompt_from_audio_tokens(audio_tokens, vocabulary)
            greedy = greedy_generate(model, prompt, vocabulary)
            beam = beam_generate(model, prompt, vocabulary, width=beam_width)
            rows.append(
                {
                    "path": str(path),
                    "greedy": {
                        "code": greedy.code,
                        "log_probability": greedy.log_probability,
                        "ended_with_eos": greedy.ended_with_eos,
                    },
                    "beam": [
                        {
                            "code": result.code,
                            "log_probability": result.log_probability,
                            "ended_with_eos": result.ended_with_eos,
                        }
                        for result in beam
                    ],
                    "external_artifacts": [
                        "audio_lm_checkpoint",
                        "frozen_muq_tokenizer_weights",
                        "query_audio",
                    ],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Identify audio with the causal audio LM.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--beam-width", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            identify(
                args.checkpoint,
                args.audio,
                device=args.device,
                beam_width=args.beam_width,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
