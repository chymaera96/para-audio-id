from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    shifted_logits: torch.Tensor, shifted_targets: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if shifted_logits.shape[:-1] != shifted_targets.shape or mask.shape != shifted_targets.shape:
        raise ValueError("Logits, targets, and loss mask shapes are inconsistent")
    selected = mask.bool()
    if not selected.any():
        raise ValueError("Loss mask selects no targets")
    return F.cross_entropy(shifted_logits[selected], shifted_targets[selected])


def causal_audio_id_losses(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_target_mask: torch.Tensor,
    id_target_mask: torch.Tensor,
    boundary_target_mask: torch.Tensor,
    *,
    id_digit_weight: float = 20.0,
) -> dict[str, torch.Tensor]:
    if id_digit_weight <= 0:
        raise ValueError("id_digit_weight must be positive")
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    audio_loss = masked_cross_entropy(shifted_logits, shifted_targets, audio_target_mask)
    id_loss = masked_cross_entropy(shifted_logits, shifted_targets, id_target_mask)
    boundary_loss = masked_cross_entropy(
        shifted_logits, shifted_targets, boundary_target_mask
    )
    token_losses = F.cross_entropy(
        shifted_logits.transpose(1, 2), shifted_targets, reduction="none"
    )
    weights = (
        audio_target_mask.to(token_losses.dtype)
        + float(id_digit_weight) * id_target_mask.to(token_losses.dtype)
        + boundary_target_mask.to(token_losses.dtype)
    )
    if ((audio_target_mask & id_target_mask).any()
        or (audio_target_mask & boundary_target_mask).any()
        or (id_target_mask & boundary_target_mask).any()):
        raise ValueError("Causal target masks must be disjoint")
    if not weights.any():
        raise ValueError("Combined loss mask selects no targets")
    predictions = shifted_logits.argmax(dim=-1)
    id_correct = (predictions == shifted_targets) & id_target_mask
    digit_accuracy = id_correct.sum() / id_target_mask.sum()
    exact_accuracy = (
        (predictions == shifted_targets) | ~id_target_mask
    ).all(dim=1).float().mean()
    return {
        "loss": (token_losses * weights).sum() / weights.sum(),
        "audio_loss": audio_loss,
        "audio_perplexity": audio_loss.detach().clamp(max=math.log(1e6)).exp(),
        "id_loss": id_loss,
        "boundary_eos_loss": boundary_loss,
        "teacher_forced_digit_accuracy": digit_accuracy,
        "teacher_forced_exact_accuracy": exact_accuracy,
    }
