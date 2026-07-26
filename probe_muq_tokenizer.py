from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from para_audio_id.audio import load_audio
from para_audio_id.config import load_config
from para_audio_id.audio_lm.tokenizer import MuQRVQTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MuQ Mel-RVQ token access and layout.")
    parser.add_argument("config", type=Path)
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tokenizer_cfg = cfg["tokenizer"]
    tokenizer = MuQRVQTokenizer(
        tokenizer_cfg["model_name"],
        revision=tokenizer_cfg.get("revision", "main"),
        selected_codebooks=int(tokenizer_cfg["selected_codebooks"]),
        sample_rate=int(tokenizer_cfg["sample_rate"]),
        device=tokenizer_cfg.get("device", "cuda"),
    )
    duration = float(cfg["data"]["segment_duration"])
    waveform = load_audio(
        args.audio,
        sample_rate=tokenizer.sample_rate,
        duration=duration,
        pad=True,
    )
    report = tokenizer.probe(torch.from_numpy(waveform).unsqueeze(0))
    maximum = int(cfg["model"]["max_position_embeddings"])
    document_length = report["serialized_tokens_per_example"] + 8
    report["causal_document_length"] = document_length
    report["max_position_embeddings"] = maximum
    if document_length > maximum:
        raise RuntimeError(
            f"Measured document length {document_length} exceeds model context {maximum}"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
