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


def causal_losses_by_view(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_target_mask: torch.Tensor,
    id_target_mask: torch.Tensor,
    boundary_target_mask: torch.Tensor,
    view_types: list[str],
    *,
    view_mode: str,
    id_digit_weight: float = 20.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    overall = causal_audio_id_losses(
        logits,
        input_ids,
        audio_target_mask,
        id_target_mask,
        boundary_target_mask,
        id_digit_weight=id_digit_weight,
    )
    per_view = {}
    for view_type in sorted(set(view_types)):
        rows = torch.tensor(
            [value == view_type for value in view_types],
            device=logits.device,
            dtype=torch.bool,
        )
        per_view[view_type] = causal_audio_id_losses(
            logits[rows],
            input_ids[rows],
            audio_target_mask[rows],
            id_target_mask[rows],
            boundary_target_mask[rows],
            id_digit_weight=id_digit_weight,
        )
    if view_mode in {"paired", "paired_roles"}:
        expected = (
            {"canonical", "shifted"}
            if view_mode == "paired"
            else {"anchor", "secondary"}
        )
        if set(per_view) != expected:
            raise ValueError(f"{view_mode} loss requires {sorted(expected)} rows")
        left, right = sorted(expected)
        loss = 0.5 * (per_view[left]["loss"] + per_view[right]["loss"])
    elif view_mode == "canonical_only":
        loss = overall["loss"]
    else:
        raise ValueError(f"Unsupported view_mode {view_mode!r}")
    return loss, overall, per_view


def _optional_masked_cross_entropy(
    shifted_logits: torch.Tensor,
    shifted_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor | None:
    return (
        masked_cross_entropy(shifted_logits, shifted_targets, mask)
        if mask.any()
        else None
    )


def degraded_causal_base_losses(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    audio_target_mask: torch.Tensor,
    id_target_mask: torch.Tensor,
    boundary_target_mask: torch.Tensor,
    is_degraded: torch.Tensor,
    *,
    id_digit_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """tc14 base loss with degraded audio targets excluded."""
    if is_degraded.shape != (logits.shape[0],):
        raise ValueError("is_degraded must contain one value per document")
    if id_digit_weight <= 0:
        raise ValueError("id_digit_weight must be positive")
    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    clean_rows = ~is_degraded.bool()
    degraded_rows = is_degraded.bool()
    if not clean_rows.any():
        raise ValueError("The base loss requires at least one clean document")
    clean_mask = clean_rows[:, None]
    degraded_mask = degraded_rows[:, None]

    clean_audio_loss = masked_cross_entropy(
        shifted_logits, shifted_targets, audio_target_mask & clean_mask
    )
    digit_loss = masked_cross_entropy(shifted_logits, shifted_targets, id_target_mask)
    boundary_loss = masked_cross_entropy(
        shifted_logits, shifted_targets, boundary_target_mask
    )
    clean_digit_loss = masked_cross_entropy(
        shifted_logits, shifted_targets, id_target_mask & clean_mask
    )
    degraded_digit_loss = _optional_masked_cross_entropy(
        shifted_logits, shifted_targets, id_target_mask & degraded_mask
    )
    clean_boundary_loss = masked_cross_entropy(
        shifted_logits, shifted_targets, boundary_target_mask & clean_mask
    )
    degraded_boundary_loss = _optional_masked_cross_entropy(
        shifted_logits, shifted_targets, boundary_target_mask & degraded_mask
    )

    audio_counts = audio_target_mask.sum(dim=1)
    digit_counts = id_target_mask.sum(dim=1)
    boundary_counts = boundary_target_mask.sum(dim=1)
    if (
        audio_counts.unique().numel() != 1
        or digit_counts.unique().numel() != 1
        or boundary_counts.unique().numel() != 1
    ):
        raise ValueError("tc14 family weighting requires uniform target counts")
    audio_count = audio_counts[0].to(clean_audio_loss.dtype)
    digit_count = digit_counts[0].to(clean_audio_loss.dtype)
    boundary_count = boundary_counts[0].to(clean_audio_loss.dtype)
    family_weight = (
        audio_count + float(id_digit_weight) * digit_count + boundary_count
    )
    base_loss = (
        audio_count * clean_audio_loss
        + float(id_digit_weight) * digit_count * digit_loss
        + boundary_count * boundary_loss
    ) / family_weight

    predictions = shifted_logits.argmax(dim=-1)
    row_exact = ((predictions == shifted_targets) | ~id_target_mask).all(dim=1)
    digit_correct = (predictions == shifted_targets) & id_target_mask
    clean_exact = row_exact[clean_rows].float().mean()
    clean_digit_accuracy = (
        digit_correct[clean_rows].sum() / id_target_mask[clean_rows].sum()
    )
    degraded_exact = (
        row_exact[degraded_rows].float().mean()
        if degraded_rows.any()
        else torch.full_like(clean_exact, float("nan"))
    )
    degraded_digit_accuracy = (
        digit_correct[degraded_rows].sum() / id_target_mask[degraded_rows].sum()
        if degraded_rows.any()
        else torch.full_like(clean_digit_accuracy, float("nan"))
    )
    legacy = causal_audio_id_losses(
        logits,
        input_ids,
        audio_target_mask,
        id_target_mask,
        boundary_target_mask,
        id_digit_weight=id_digit_weight,
    )
    return base_loss, {
        "base_loss": base_loss,
        "clean_audio_loss": clean_audio_loss,
        "clean_audio_perplexity": clean_audio_loss.detach()
        .clamp(max=math.log(1e6))
        .exp(),
        "clean_digit_loss": clean_digit_loss,
        "degraded_digit_loss": (
            degraded_digit_loss
            if degraded_digit_loss is not None
            else torch.full_like(clean_digit_loss, float("nan"))
        ),
        "clean_boundary_loss": clean_boundary_loss,
        "degraded_boundary_loss": (
            degraded_boundary_loss
            if degraded_boundary_loss is not None
            else torch.full_like(clean_boundary_loss, float("nan"))
        ),
        "digit_loss": digit_loss,
        "boundary_loss": boundary_loss,
        "audio_family_coefficient": audio_count / family_weight,
        "digit_family_coefficient": float(id_digit_weight)
        * digit_count
        / family_weight,
        "boundary_family_coefficient": boundary_count / family_weight,
        "legacy_weighted_token_loss": legacy["loss"].detach(),
        "teacher_forced_digit_accuracy": legacy["teacher_forced_digit_accuracy"],
        "teacher_forced_exact_accuracy": legacy["teacher_forced_exact_accuracy"],
        "clean_teacher_forced_digit_accuracy": clean_digit_accuracy,
        "degraded_teacher_forced_digit_accuracy": degraded_digit_accuracy,
        "clean_teacher_forced_exact_accuracy": clean_exact,
        "degraded_teacher_forced_exact_accuracy": degraded_exact,
    }


def distillation_weight(
    step: int,
    *,
    maximum_weight: float,
    zero_until_step: int = 15_000,
    ramp_until_step: int = 30_000,
) -> float:
    """Resolve tc14's clean-teacher distillation weight."""
    if step < 0:
        raise ValueError("Global step cannot be negative")
    if not math.isfinite(maximum_weight) or maximum_weight < 0:
        raise ValueError("maximum_weight must be finite and non-negative")
    if not 0 <= zero_until_step < ramp_until_step:
        raise ValueError("Invalid distillation schedule boundaries")
    if step <= zero_until_step:
        return 0.0
    if step < ramp_until_step:
        progress = (step - zero_until_step) / (
            ramp_until_step - zero_until_step
        )
        return maximum_weight * progress
    return maximum_weight


def identifier_logit_distillation_loss(
    logits: torch.Tensor,
    id_target_mask: torch.Tensor,
    is_degraded: torch.Tensor,
    track_ids: list[str],
    digit_token_ids: list[int] | torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Distil clean next-digit distributions into same-track degraded rows."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    batch_size = logits.shape[0]
    shifted_logits = logits[:, :-1, :]
    if id_target_mask.shape != shifted_logits.shape[:2]:
        raise ValueError("Identifier target mask does not align with shifted logits")
    if is_degraded.shape != (batch_size,):
        raise ValueError("is_degraded must contain one value per document")
    if len(track_ids) != batch_size:
        raise ValueError("track_ids must contain one value per document")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not (id_target_mask.sum(dim=1) == 5).all():
        raise ValueError("Every document must contain exactly five digit targets")

    digit_ids = torch.as_tensor(
        digit_token_ids, device=logits.device, dtype=torch.long
    )
    if digit_ids.shape != (10,) or digit_ids.unique().numel() != 10:
        raise ValueError("digit_token_ids must contain ten unique token IDs")
    if (digit_ids < 0).any() or (digit_ids >= logits.shape[-1]).any():
        raise ValueError("digit_token_ids fall outside the model vocabulary")

    degraded_indices = is_degraded.bool().nonzero(as_tuple=False).flatten()
    if not len(degraded_indices):
        return logits.sum() * 0.0
    if (degraded_indices % 2 != 1).any():
        raise ValueError("Degraded rows must be the second row of their pair")
    clean_indices = degraded_indices - 1
    if is_degraded[clean_indices].any():
        raise ValueError("A degraded row must follow a clean anchor")
    for clean_index, degraded_index in zip(
        clean_indices.tolist(), degraded_indices.tolist(), strict=True
    ):
        if track_ids[clean_index] != track_ids[degraded_index]:
            raise ValueError("A degraded row and clean anchor must share a track")

    digit_logits = shifted_logits[id_target_mask.bool()].reshape(
        batch_size, 5, logits.shape[-1]
    )
    clean_logits = digit_logits[clean_indices].index_select(-1, digit_ids).detach()
    degraded_logits = digit_logits[degraded_indices].index_select(-1, digit_ids)
    teacher_probabilities = F.softmax(clean_logits / temperature, dim=-1)
    degraded_log_probabilities = F.log_softmax(
        degraded_logits / temperature, dim=-1
    )
    per_digit_kl = F.kl_div(
        degraded_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=-1)
    return temperature**2 * per_digit_kl.mean()
