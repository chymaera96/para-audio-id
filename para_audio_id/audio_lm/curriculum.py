from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ProbeDecision:
    event: str | None = None
    failure: str | None = None


def effective_cosine_multiplier(
    effective_step: int, *, max_steps: int, warmup_steps: int
) -> float:
    if not 0 < warmup_steps < max_steps:
        raise ValueError("warmup_steps must be between zero and max_steps")
    if effective_step < 0:
        raise ValueError("effective_step cannot be negative")
    if effective_step < warmup_steps:
        return (effective_step + 1) / warmup_steps
    progress = (effective_step - warmup_steps) / max(
        1, max_steps - warmup_steps
    )
    return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


@dataclass
class AdaptiveCurriculum:
    nominal_max_steps: int
    gate_threshold: float = 0.5
    gate_max_extra_steps: int = 30_000
    regression_drop: float = 0.05
    recovery_probes: int = 2
    recovery_timeout_steps: int = 20_000
    gate_open: bool = False
    gate_open_step: int | None = None
    regression_baseline: float | None = None
    consistency_multiplier: float = 1.0
    recovery_active: bool = False
    recovery_start_step: int | None = None
    recovery_frozen_effective_step: int | None = None
    recovery_streak: int = 0
    completed_pause_steps: int = 0
    intervention_used: bool = False

    def __post_init__(self) -> None:
        if self.nominal_max_steps < 7:
            raise ValueError("nominal_max_steps is too small")
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("gate_threshold must be between zero and one")
        if self.gate_max_extra_steps < 0 or self.recovery_timeout_steps < 1:
            raise ValueError("Adaptive curriculum timeouts are invalid")
        if self.recovery_probes < 1:
            raise ValueError("recovery_probes must be positive")

    @property
    def clean_steps(self) -> int:
        return round(self.nominal_max_steps * 20_000 / 70_000)

    @property
    def hard_max_steps(self) -> int:
        return (
            self.nominal_max_steps
            + self.gate_max_extra_steps
            + self.recovery_timeout_steps
        )

    def effective_step(self, global_step: int) -> int:
        if global_step < 0:
            raise ValueError("global_step cannot be negative")
        if global_step < self.clean_steps:
            return global_step
        if not self.gate_open:
            return self.clean_steps
        if self.gate_open_step is None:
            raise RuntimeError("Open gate is missing its transition step")
        if self.recovery_active:
            if self.recovery_frozen_effective_step is None:
                raise RuntimeError("Active recovery is missing its frozen step")
            return self.recovery_frozen_effective_step
        curriculum_elapsed = (
            global_step - self.gate_open_step - self.completed_pause_steps
        )
        return min(
            self.nominal_max_steps,
            self.clean_steps + max(0, curriculum_elapsed),
        )

    def observe_probe(
        self,
        *,
        global_step: int,
        shifted_teacher_forced_exact: float,
        shifted_greedy_top1: float,
        consistency_is_active: bool,
    ) -> ProbeDecision:
        if not self.gate_open:
            if global_step < self.clean_steps:
                return ProbeDecision()
            if shifted_teacher_forced_exact >= self.gate_threshold:
                self.gate_open = True
                self.gate_open_step = global_step
                self.regression_baseline = shifted_greedy_top1
                return ProbeDecision(event="gate_opened")
            if global_step >= self.clean_steps + self.gate_max_extra_steps:
                return ProbeDecision(
                    failure=(
                        "clean shifted teacher-forced exact accuracy did not reach "
                        f"{self.gate_threshold:g} within the gate allowance"
                    )
                )
            return ProbeDecision(event="gate_waiting")

        if self.regression_baseline is None:
            raise RuntimeError("Open gate is missing its regression baseline")
        recovered = (
            shifted_greedy_top1
            >= self.regression_baseline - self.regression_drop
        )
        if self.recovery_active:
            if self.recovery_start_step is None:
                raise RuntimeError("Active recovery is missing its start step")
            self.recovery_streak = self.recovery_streak + 1 if recovered else 0
            if self.recovery_streak >= self.recovery_probes:
                self.completed_pause_steps += global_step - self.recovery_start_step
                self.recovery_active = False
                self.recovery_start_step = None
                self.recovery_frozen_effective_step = None
                self.recovery_streak = 0
                return ProbeDecision(event="recovery_completed")
            if global_step - self.recovery_start_step >= self.recovery_timeout_steps:
                return ProbeDecision(
                    failure="clean shifted Top-1 did not recover within the timeout"
                )
            return ProbeDecision(event="recovery_waiting")

        if consistency_is_active and not recovered:
            if self.intervention_used:
                return ProbeDecision(
                    failure="clean shifted Top-1 regressed a second time"
                )
            frozen_effective_step = self.effective_step(global_step)
            self.intervention_used = True
            self.consistency_multiplier *= 0.5
            self.recovery_active = True
            self.recovery_start_step = global_step
            self.recovery_frozen_effective_step = frozen_effective_step
            self.recovery_streak = 0
            return ProbeDecision(event="recovery_started")
        return ProbeDecision()

    def completed(self, global_step: int) -> bool:
        return self.effective_step(global_step) >= self.nominal_max_steps

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, payload: dict) -> None:
        if int(payload["nominal_max_steps"]) != self.nominal_max_steps:
            raise ValueError("Resume checkpoint uses a different nominal step count")
        immutable = (
            "gate_threshold",
            "gate_max_extra_steps",
            "regression_drop",
            "recovery_probes",
            "recovery_timeout_steps",
        )
        for key in immutable:
            if payload[key] != getattr(self, key):
                raise ValueError(f"Resume checkpoint changes curriculum setting {key}")
        for key in self.__dataclass_fields__:
            setattr(self, key, payload[key])
