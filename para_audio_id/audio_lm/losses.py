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


def _different_track_similarity(
    clean_states: torch.Tensor,
    noisy_states: torch.Tensor,
    clean_track_ids: list[str],
    noisy_track_ids: list[str],
) -> torch.Tensor:
    comparisons = []
    for noisy_state, noisy_track in zip(
        noisy_states, noisy_track_ids, strict=True
    ):
        index = next(
            (
                candidate
                for candidate, clean_track in enumerate(clean_track_ids)
                if clean_track != noisy_track
            ),
            None,
        )
        if index is not None:
            comparisons.append(
                F.cosine_similarity(
                    noisy_state.unsqueeze(0),
                    clean_states[index].detach().unsqueeze(0),
                )[0]
            )
    if not comparisons:
        return torch.full(
            (), float("nan"), device=clean_states.device, dtype=clean_states.dtype
        )
    return torch.stack(comparisons).mean()


def noise_consistency_losses(
    logits: torch.Tensor,
    final_hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    audio_target_mask: torch.Tensor,
    id_target_mask: torch.Tensor,
    boundary_target_mask: torch.Tensor,
    is_noisy: torch.Tensor,
    track_ids: list[str],
    *,
    id_digit_weight: float,
    consistency_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if is_noisy.shape != (logits.shape[0],):
        raise ValueError("is_noisy must contain one value per document")
    if final_hidden_states.shape[:2] != input_ids.shape:
        raise ValueError("Final hidden states do not match causal input shape")
    if len(track_ids) != logits.shape[0]:
        raise ValueError("track_ids must contain one value per document")
    if id_digit_weight <= 0 or consistency_weight < 0:
        raise ValueError("Loss weights are invalid")

    shifted_logits = logits[:, :-1, :]
    shifted_targets = input_ids[:, 1:]
    clean_rows = ~is_noisy.bool()
    clean_mask = clean_rows[:, None]
    noisy_mask = is_noisy.bool()[:, None]

    clean_audio_loss = masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        audio_target_mask & clean_mask,
    )
    clean_digit_loss = masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        id_target_mask & clean_mask,
    )
    noisy_digit_loss = _optional_masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        id_target_mask & noisy_mask,
    )
    clean_boundary_loss = masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        boundary_target_mask & clean_mask,
    )
    noisy_boundary_loss = _optional_masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        boundary_target_mask & noisy_mask,
    )

    digit_loss = masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        id_target_mask,
    )
    boundary_loss = masked_cross_entropy(
        shifted_logits,
        shifted_targets,
        boundary_target_mask,
    )

    if not (id_target_mask.sum(dim=1) == 5).all():
        raise ValueError("Every document must contain exactly five digit targets")
    id_columns = id_target_mask.long().argmax(dim=1)
    id_states = final_hidden_states[
        torch.arange(input_ids.shape[0], device=input_ids.device), id_columns
    ]

    noisy_indices = is_noisy.bool().nonzero(as_tuple=False).flatten()
    if len(noisy_indices):
        if (noisy_indices % 2 != 1).any():
            raise ValueError("Noisy rows must be the second row of their pair")
        clean_indices = noisy_indices - 1
        if is_noisy[clean_indices].any():
            raise ValueError("A noisy row must follow a clean anchor")
        clean_states = F.normalize(id_states[clean_indices], dim=-1)
        noisy_states = F.normalize(id_states[noisy_indices], dim=-1)
        same_similarity = F.cosine_similarity(
            clean_states.detach(), noisy_states, dim=-1
        ).mean()
        consistency_loss = 1.0 - same_similarity
        different_similarity = _different_track_similarity(
            F.normalize(id_states[clean_rows], dim=-1),
            noisy_states,
            [
                track_id
                for track_id, keep in zip(
                    track_ids, clean_rows.tolist(), strict=True
                )
                if keep
            ],
            [track_ids[index] for index in noisy_indices.tolist()],
        )
    else:
        consistency_loss = logits.sum() * 0.0
        same_similarity = torch.full(
            (), float("nan"), device=logits.device, dtype=logits.dtype
        )
        different_similarity = same_similarity.clone()

    audio_counts = audio_target_mask.sum(dim=1)
    digit_counts = id_target_mask.sum(dim=1)
    boundary_counts = boundary_target_mask.sum(dim=1)
    if (
        audio_counts.unique().numel() != 1
        or digit_counts.unique().numel() != 1
        or boundary_counts.unique().numel() != 1
    ):
        raise ValueError(
            "tc6 fixed family weighting requires uniform document target counts"
        )
    audio_count = audio_counts[0].to(clean_audio_loss.dtype)
    digit_count = digit_counts[0].to(clean_audio_loss.dtype)
    boundary_count = boundary_counts[0].to(clean_audio_loss.dtype)
    family_weight = (
        audio_count
        + float(id_digit_weight) * digit_count
        + boundary_count
    )
    base_loss = (
        audio_count * clean_audio_loss
        + float(id_digit_weight) * digit_count * digit_loss
        + boundary_count * boundary_loss
    ) / family_weight
    total = base_loss + float(consistency_weight) * consistency_loss
    legacy = causal_audio_id_losses(
        logits,
        input_ids,
        audio_target_mask,
        id_target_mask,
        boundary_target_mask,
        id_digit_weight=id_digit_weight,
    )
    return total, {
        "loss": total,
        "base_loss": base_loss,
        "clean_audio_loss": clean_audio_loss,
        "clean_audio_perplexity": clean_audio_loss.detach()
        .clamp(max=math.log(1e6))
        .exp(),
        "clean_digit_loss": clean_digit_loss,
        "noisy_digit_loss": (
            noisy_digit_loss
            if noisy_digit_loss is not None
            else torch.full_like(clean_digit_loss, float("nan"))
        ),
        "clean_boundary_loss": clean_boundary_loss,
        "noisy_boundary_loss": (
            noisy_boundary_loss
            if noisy_boundary_loss is not None
            else torch.full_like(clean_boundary_loss, float("nan"))
        ),
        "digit_loss": digit_loss,
        "boundary_loss": boundary_loss,
        "audio_family_coefficient": audio_count / family_weight,
        "digit_family_coefficient": (
            float(id_digit_weight) * digit_count / family_weight
        ),
        "boundary_family_coefficient": boundary_count / family_weight,
        "consistency_loss": consistency_loss,
        "same_track_cosine": same_similarity,
        "different_track_cosine": different_similarity,
        "legacy_weighted_token_loss": legacy["loss"].detach(),
        "teacher_forced_digit_accuracy": legacy[
            "teacher_forced_digit_accuracy"
        ],
        "teacher_forced_exact_accuracy": legacy[
            "teacher_forced_exact_accuracy"
        ],
    }
