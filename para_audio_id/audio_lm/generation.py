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


def prompts_from_audio_tokens(
    audio_tokens: torch.Tensor, vocabulary: AudioLMVocabulary
) -> torch.Tensor:
    if audio_tokens.ndim != 2:
        raise ValueError("Batched generation requires [batch, audio_tokens]")
    batch = audio_tokens.shape[0]
    bos = torch.full(
        (batch, 1),
        vocabulary.bos_token_id,
        device=audio_tokens.device,
        dtype=torch.long,
    )
    boundary = torch.full(
        (batch, 1),
        vocabulary.id_token_id,
        device=audio_tokens.device,
        dtype=torch.long,
    )
    return torch.cat((bos, audio_tokens.long(), boundary), dim=1)


@torch.inference_mode()
def batched_greedy_generate(
    model: AudioCausalLM,
    prompts: torch.Tensor,
    vocabulary: AudioLMVocabulary,
) -> list[GenerationResult]:
    if prompts.ndim != 2:
        raise ValueError("Batched prompts must have shape [batch, sequence]")
    generated = []
    scores = torch.zeros(prompts.shape[0], device=prompts.device)
    output = model.forward_with_cache(prompts)
    logits = output.logits[:, -1, :]
    past_key_values = output.past_key_values
    for _ in range(5):
        log_probs = logits[
            :, vocabulary.digit_offset : vocabulary.digit_offset + 10
        ].log_softmax(dim=-1)
        raw_digits = log_probs.argmax(dim=-1)
        scores += log_probs.gather(1, raw_digits[:, None]).squeeze(1)
        tokens = raw_digits + vocabulary.digit_offset
        generated.append(tokens)
        output = model.forward_with_cache(
            tokens[:, None],
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values
    eos_scores = logits.log_softmax(dim=-1)[
        :, vocabulary.eos_token_id
    ]
    scores += eos_scores
    digit_matrix = torch.stack(generated, dim=1).cpu()
    return [
        GenerationResult(
            vocabulary.decode_code(digits),
            float(score),
            ended_with_eos=True,
        )
        for digits, score in zip(digit_matrix, scores.cpu(), strict=True)
    ]


@torch.inference_mode()
def batched_beam_generate(
    model: AudioCausalLM,
    prompts: torch.Tensor,
    vocabulary: AudioLMVocabulary,
    *,
    width: int = 10,
) -> list[list[GenerationResult]]:
    if prompts.ndim != 2:
        raise ValueError("Batched prompts must have shape [batch, sequence]")
    if width < 1 or width > 10:
        raise ValueError("Beam width must be between 1 and 10")
    batch = prompts.shape[0]
    scores = torch.zeros((batch, 1), device=prompts.device)
    generated = torch.empty((batch, 1, 0), dtype=torch.long, device=prompts.device)
    beam_count = 1
    output = model.forward_with_cache(prompts)
    logits = output.logits[:, -1, :]
    past_key_values = output.past_key_values
    for _ in range(5):
        digit_log_probs = logits[
            :, vocabulary.digit_offset : vocabulary.digit_offset + 10
        ].log_softmax(dim=-1).reshape(batch, beam_count, 10)
        candidates = digit_log_probs + scores[:, :, None]
        next_scores, flat_indices = candidates.reshape(batch, -1).topk(width, dim=-1)
        parent = flat_indices // 10
        raw_digits = flat_indices % 10
        gather_generated = parent[:, :, None].expand(-1, -1, generated.shape[-1])
        generated = generated.gather(1, gather_generated)
        tokens = raw_digits + vocabulary.digit_offset
        generated = torch.cat((generated, tokens[:, :, None]), dim=-1)
        scores = next_scores
        parent_flat = (
            torch.arange(batch, device=prompts.device)[:, None] * beam_count + parent
        ).reshape(-1)
        past_key_values = model.reorder_cache(past_key_values, parent_flat)
        output = model.forward_with_cache(
            tokens.reshape(-1, 1),
            past_key_values=past_key_values,
        )
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values
        beam_count = width
    eos_scores = logits.log_softmax(dim=-1)[
        :, vocabulary.eos_token_id
    ].reshape(batch, width)
    scores += eos_scores
    order = scores.argsort(dim=-1, descending=True)
    generated = generated.gather(
        1, order[:, :, None].expand(-1, -1, generated.shape[-1])
    ).cpu()
    scores = scores.gather(1, order).cpu()
    return [
        [
            GenerationResult(
                vocabulary.decode_code(digits),
                float(score),
                ended_with_eos=True,
            )
            for digits, score in zip(batch_digits, batch_scores, strict=True)
        ]
        for batch_digits, batch_scores in zip(generated, scores, strict=True)
    ]


@torch.inference_mode()
def batched_joint_beam_generate(
    model: AudioCausalLM,
    prompts: torch.Tensor,
    vocabulary: AudioLMVocabulary,
    *,
    width: int = 10,
) -> list[list[GenerationResult]]:
    """Decode one shared identifier from multiple windows of each query.

    ``prompts`` has shape ``[queries, windows, sequence]``. At every identifier
    position, digit log-probabilities are averaged over a query's windows before
    its shared beam is expanded and pruned. This is deliberately different from
    decoding each window independently and voting over completed identifiers.
    """
    if prompts.ndim != 3:
        raise ValueError(
            "Joint-beam prompts must have shape [queries, windows, sequence]"
        )
    if prompts.shape[0] < 1 or prompts.shape[1] < 1:
        raise ValueError("Joint-beam decoding requires queries and windows")
    if width < 1 or width > 10:
        raise ValueError("Beam width must be between 1 and 10")

    query_count, window_count, _ = prompts.shape
    scores = torch.zeros((query_count, 1), device=prompts.device)
    generated = torch.empty(
        (query_count, 1, 0), dtype=torch.long, device=prompts.device
    )
    beam_count = 1
    output = model.forward_with_cache(prompts.flatten(0, 1))
    logits = output.logits[:, -1, :]
    past_key_values = output.past_key_values

    for _ in range(5):
        digit_log_probs = logits[
            :, vocabulary.digit_offset : vocabulary.digit_offset + 10
        ].log_softmax(dim=-1)
        digit_log_probs = digit_log_probs.reshape(
            query_count, beam_count, window_count, 10
        ).mean(dim=2)
        candidates = digit_log_probs + scores[:, :, None]
        next_scores, flat_indices = candidates.reshape(query_count, -1).topk(
            width, dim=-1
        )
        parent = flat_indices // 10
        raw_digits = flat_indices % 10
        generated = generated.gather(
            1, parent[:, :, None].expand(-1, -1, generated.shape[-1])
        )
        generated = torch.cat(
            (generated, (raw_digits + vocabulary.digit_offset)[:, :, None]), dim=-1
        )
        scores = next_scores

        query_indices = torch.arange(query_count, device=prompts.device)[:, None, None]
        window_indices = torch.arange(window_count, device=prompts.device)[None, None, :]
        cache_indices = (
            (query_indices * beam_count + parent[:, :, None]) * window_count
            + window_indices
        ).reshape(-1)
        past_key_values = model.reorder_cache(past_key_values, cache_indices)
        next_tokens = (raw_digits + vocabulary.digit_offset)[:, :, None].expand(
            -1, -1, window_count
        )
        output = model.forward_with_cache(
            next_tokens.reshape(-1, 1), past_key_values=past_key_values
        )
        logits = output.logits[:, -1, :]
        past_key_values = output.past_key_values
        beam_count = width

    eos_scores = logits.log_softmax(dim=-1)[:, vocabulary.eos_token_id]
    eos_scores = eos_scores.reshape(
        query_count, width, window_count
    ).mean(dim=2)
    scores += eos_scores
    order = scores.argsort(dim=-1, descending=True)
    generated = generated.gather(
        1, order[:, :, None].expand(-1, -1, generated.shape[-1])
    ).cpu()
    scores = scores.gather(1, order).cpu()
    return [
        [
            GenerationResult(
                vocabulary.decode_code(digits),
                float(score),
                ended_with_eos=True,
            )
            for digits, score in zip(batch_digits, batch_scores, strict=True)
        ]
        for batch_digits, batch_scores in zip(generated, scores, strict=True)
    ]


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
    return batched_greedy_generate(model, prompt.unsqueeze(0), vocabulary)[0]


@torch.inference_mode()
def beam_generate(
    model: AudioCausalLM,
    prompt: torch.Tensor,
    vocabulary: AudioLMVocabulary,
    *,
    width: int = 10,
) -> list[GenerationResult]:
    return batched_beam_generate(
        model, prompt.unsqueeze(0), vocabulary, width=width
    )[0]
