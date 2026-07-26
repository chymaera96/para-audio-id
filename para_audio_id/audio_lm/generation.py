from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import AudioCausalLM
from .vocabulary import AudioLMVocabulary


@dataclass(frozen=True)
class GenerationResult:
    code: str
    log_probability: float
    ended_with_eos: bool


def prompt_from_audio_tokens(
    audio_tokens: torch.Tensor, vocabulary: AudioLMVocabulary
) -> torch.Tensor:
    if audio_tokens.ndim != 1:
        raise ValueError("A generation prompt requires one flat audio-token sequence")
    return torch.cat(
        (
            torch.tensor(
                [vocabulary.bos_token_id], device=audio_tokens.device, dtype=torch.long
            ),
            audio_tokens.long(),
            torch.tensor(
                [vocabulary.id_token_id], device=audio_tokens.device, dtype=torch.long
            ),
        )
    )


def digit_log_probabilities(
    model: AudioCausalLM, sequence: torch.Tensor, vocabulary: AudioLMVocabulary
) -> torch.Tensor:
    logits = model(sequence.unsqueeze(0))[:, -1, :]
    digit_logits = logits[:, vocabulary.digit_offset : vocabulary.digit_offset + 10]
    return digit_logits.log_softmax(dim=-1).squeeze(0)


def eos_log_probability(
    model: AudioCausalLM, sequence: torch.Tensor, vocabulary: AudioLMVocabulary
) -> torch.Tensor:
    logits = model(sequence.unsqueeze(0))[:, -1, :]
    return logits.log_softmax(dim=-1)[0, vocabulary.eos_token_id]


@torch.inference_mode()
def greedy_generate(
    model: AudioCausalLM, prompt: torch.Tensor, vocabulary: AudioLMVocabulary
) -> GenerationResult:
    sequence = prompt
    generated = []
    score = 0.0
    for _ in range(5):
        log_probs = digit_log_probabilities(model, sequence, vocabulary)
        raw_digit = int(log_probs.argmax())
        score += float(log_probs[raw_digit])
        token = vocabulary.digit_offset + raw_digit
        generated.append(token)
        sequence = torch.cat(
            (sequence, torch.tensor([token], device=sequence.device, dtype=torch.long))
        )
    eos_score = eos_log_probability(model, sequence, vocabulary)
    score += float(eos_score)
    sequence = torch.cat(
        (
            sequence,
            torch.tensor(
                [vocabulary.eos_token_id], device=sequence.device, dtype=torch.long
            ),
        )
    )
    return GenerationResult(
        vocabulary.decode_code(generated),
        score,
        ended_with_eos=int(sequence[-1]) == vocabulary.eos_token_id,
    )


@torch.inference_mode()
def beam_generate(
    model: AudioCausalLM,
    prompt: torch.Tensor,
    vocabulary: AudioLMVocabulary,
    *,
    width: int = 10,
) -> list[GenerationResult]:
    if width < 1 or width > 10:
        raise ValueError("Beam width must be between 1 and 10")
    beams: list[tuple[torch.Tensor, list[int], float]] = [(prompt, [], 0.0)]
    for _ in range(5):
        expanded = []
        for sequence, generated, score in beams:
            log_probs = digit_log_probabilities(model, sequence, vocabulary)
            values, digits = log_probs.topk(width)
            for value, raw_digit in zip(values, digits, strict=True):
                token = vocabulary.digit_offset + int(raw_digit)
                expanded.append(
                    (
                        torch.cat(
                            (
                                sequence,
                                torch.tensor(
                                    [token], device=sequence.device, dtype=torch.long
                                ),
                            )
                        ),
                        generated + [token],
                        score + float(value),
                    )
                )
        beams = sorted(expanded, key=lambda item: item[2], reverse=True)[:width]
    results = []
    for sequence, generated, score in beams:
        score += float(eos_log_probability(model, sequence, vocabulary))
        results.append(
            GenerationResult(
                vocabulary.decode_code(generated), score, ended_with_eos=True
            )
        )
    return sorted(results, key=lambda result: result.log_probability, reverse=True)
