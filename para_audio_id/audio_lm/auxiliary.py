from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def task_anchored_total_loss(
    base_loss: torch.Tensor,
    auxiliary_metrics: dict[str, torch.Tensor],
    *,
    summary_weight: float,
    predictive_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if summary_weight < 0 or predictive_weight < 0:
        raise ValueError("Task-anchored loss weights must be non-negative")
    summary_contribution = summary_weight * auxiliary_metrics["summary_loss"]
    predictive_contribution = (
        predictive_weight * auxiliary_metrics["predictive_loss"]
    )
    total = base_loss + summary_contribution + predictive_contribution
    return total, {
        "summary_contribution": summary_contribution,
        "predictive_contribution": predictive_contribution,
        "effective_predictive_weight": torch.tensor(
            predictive_weight, device=total.device, dtype=total.dtype
        ),
    }


class TaskAnchoredAuxiliary(nn.Module):
    """Training-only summary and asymmetric predictive heads for tc13."""

    def __init__(
        self,
        hidden_size: int,
        *,
        id_token_id: int,
        projector_hidden_size: int = 1024,
        projection_size: int = 256,
    ):
        super().__init__()
        self.id_token_id = int(id_token_id)
        self.summary_head = nn.Linear(hidden_size, 5 * 10)
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, projector_hidden_size, bias=False),
            nn.LayerNorm(projector_hidden_size),
            nn.GELU(),
            nn.Linear(projector_hidden_size, projection_size, bias=False),
        )
        self.predictor = nn.Sequential(
            nn.Linear(projection_size, projector_hidden_size, bias=False),
            nn.LayerNorm(projector_hidden_size),
            nn.GELU(),
            nn.Linear(projector_hidden_size, projection_size, bias=False),
        )

    def id_states(
        self,
        final_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        id_target_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.shape != final_hidden_states.shape[:2]:
            raise ValueError("Input IDs and final hidden states do not align")
        if final_hidden_states.shape[:2] != (
            id_target_mask.shape[0],
            id_target_mask.shape[1] + 1,
        ):
            raise ValueError("Final hidden states and identifier mask do not align")
        if not (id_target_mask.sum(dim=1) == 5).all():
            raise ValueError("Every document must contain exactly five digit targets")
        id_columns = id_target_mask.long().argmax(dim=1)
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        if not (input_ids[rows, id_columns] == self.id_token_id).all():
            raise ValueError("Identifier mask does not select the [ID] input state")
        return final_hidden_states[
            rows,
            id_columns,
        ]

    @staticmethod
    def _feature_std(projected: torch.Tensor) -> torch.Tensor:
        if projected.shape[0] < 2:
            return torch.full(
                (), float("nan"), device=projected.device, dtype=projected.dtype
            )
        return projected.std(dim=0, correction=0).mean()

    def forward(
        self,
        final_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        id_target_mask: torch.Tensor,
        identifier_digits: torch.Tensor,
        is_degraded: torch.Tensor,
        track_ids: list[str],
    ) -> dict[str, torch.Tensor]:
        batch_size = final_hidden_states.shape[0]
        if identifier_digits.shape != (batch_size, 5):
            raise ValueError("identifier_digits must have shape [batch, 5]")
        if identifier_digits.dtype != torch.long:
            raise ValueError("identifier_digits must be torch.long")
        if ((identifier_digits < 0) | (identifier_digits > 9)).any():
            raise ValueError("Identifier digit classes must be in [0, 9]")
        if is_degraded.shape != (batch_size,):
            raise ValueError("is_degraded must contain one value per document")
        if len(track_ids) != batch_size:
            raise ValueError("track_ids must contain one value per document")

        states = self.id_states(final_hidden_states, input_ids, id_target_mask)
        summary_logits = self.summary_head(states).view(batch_size, 5, 10)
        summary_loss = F.cross_entropy(
            summary_logits.reshape(-1, 10), identifier_digits.reshape(-1)
        )
        summary_predictions = summary_logits.argmax(dim=-1)
        summary_correct = summary_predictions == identifier_digits
        summary_digit_accuracy = summary_correct.float().mean()
        summary_exact = summary_correct.all(dim=1)
        clean_rows = ~is_degraded.bool()
        degraded_rows = is_degraded.bool()
        clean_summary_exact = summary_exact[clean_rows].float().mean()
        degraded_summary_exact = (
            summary_exact[degraded_rows].float().mean()
            if degraded_rows.any()
            else torch.full_like(clean_summary_exact, float("nan"))
        )

        degraded_indices = degraded_rows.nonzero(as_tuple=False).flatten()
        if len(degraded_indices):
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
                if not torch.equal(
                    identifier_digits[clean_index], identifier_digits[degraded_index]
                ):
                    raise ValueError("A degraded row and clean anchor must share a code")

            clean_projected = self.projector(states[clean_indices])
            clean_targets = F.normalize(clean_projected, dim=-1).detach()
            degraded_projected = self.projector(states[degraded_indices])
            degraded_predictions = F.normalize(
                self.predictor(degraded_projected), dim=-1
            )
            paired_cosine = (degraded_predictions * clean_targets).sum(dim=-1).mean()
            predictive_loss = 2.0 - 2.0 * paired_cosine
            clean_feature_std = self._feature_std(clean_projected)
            degraded_feature_std = self._feature_std(degraded_projected)

            if len(degraded_indices) >= 2:
                shuffled_targets = clean_targets.roll(shifts=1, dims=0)
                clean_tracks = [track_ids[index] for index in clean_indices.tolist()]
                shuffled_tracks = clean_tracks[-1:] + clean_tracks[:-1]
                degraded_tracks = [
                    track_ids[index] for index in degraded_indices.tolist()
                ]
                if any(
                    left == right
                    for left, right in zip(
                        degraded_tracks, shuffled_tracks, strict=True
                    )
                ):
                    raise ValueError("Shuffled targets must belong to different tracks")
                shuffled_cosine = (
                    degraded_predictions.detach() * shuffled_targets
                ).sum(dim=-1).mean()
                prediction_margin = paired_cosine.detach() - shuffled_cosine
            else:
                shuffled_cosine = torch.full_like(paired_cosine, float("nan"))
                prediction_margin = shuffled_cosine.clone()
        else:
            differentiable_zero = states.sum() * 0.0
            for parameter in self.projector.parameters():
                differentiable_zero = differentiable_zero + parameter.sum() * 0.0
            for parameter in self.predictor.parameters():
                differentiable_zero = differentiable_zero + parameter.sum() * 0.0
            predictive_loss = differentiable_zero
            unavailable = torch.full(
                (), float("nan"), device=states.device, dtype=states.dtype
            )
            paired_cosine = unavailable
            shuffled_cosine = unavailable.clone()
            prediction_margin = unavailable.clone()
            clean_feature_std = unavailable.clone()
            degraded_feature_std = unavailable.clone()

        return {
            "summary_loss": summary_loss,
            "summary_digit_accuracy": summary_digit_accuracy,
            "summary_exact_accuracy": summary_exact.float().mean(),
            "clean_summary_exact_accuracy": clean_summary_exact,
            "degraded_summary_exact_accuracy": degraded_summary_exact,
            "predictive_loss": predictive_loss,
            "paired_prediction_cosine": paired_cosine,
            "shuffled_prediction_cosine": shuffled_cosine,
            "prediction_cosine_margin": prediction_margin,
            "clean_projected_feature_std": clean_feature_std,
            "degraded_projected_feature_std": degraded_feature_std,
            "degraded_pair_count": torch.tensor(
                len(degraded_indices), device=states.device, dtype=torch.long
            ),
        }
