import pytest

from para_audio_id.audio_lm.curriculum import (
    AdaptiveCurriculum,
    effective_cosine_multiplier,
)


def make_curriculum():
    return AdaptiveCurriculum(
        nominal_max_steps=70_000,
        gate_threshold=0.5,
        gate_max_extra_steps=30_000,
        regression_drop=0.05,
        recovery_probes=2,
        recovery_timeout_steps=20_000,
    )


def test_gate_pauses_effective_clock_opens_and_times_out():
    state = make_curriculum()
    assert state.effective_step(25_000) == 20_000
    waiting = state.observe_probe(
        global_step=20_000,
        shifted_teacher_forced_exact=0.49,
        shifted_greedy_top1=0.7,
        consistency_is_active=False,
    )
    assert waiting.event == "gate_waiting"
    opened = state.observe_probe(
        global_step=25_000,
        shifted_teacher_forced_exact=0.5,
        shifted_greedy_top1=0.8,
        consistency_is_active=False,
    )
    assert opened.event == "gate_opened"
    assert state.effective_step(25_000) == 20_000
    assert state.effective_step(30_000) == 25_000

    timed_out = make_curriculum().observe_probe(
        global_step=50_000,
        shifted_teacher_forced_exact=0.49,
        shifted_greedy_top1=0.7,
        consistency_is_active=False,
    )
    assert "gate allowance" in timed_out.failure


def test_recovery_freezes_clock_halves_weight_and_fails_on_recurrence():
    state = make_curriculum()
    state.observe_probe(
        global_step=20_000,
        shifted_teacher_forced_exact=0.5,
        shifted_greedy_top1=0.8,
        consistency_is_active=False,
    )
    started = state.observe_probe(
        global_step=25_000,
        shifted_teacher_forced_exact=0.8,
        shifted_greedy_top1=0.74,
        consistency_is_active=True,
    )
    assert started.event == "recovery_started"
    assert state.consistency_multiplier == 0.5
    frozen = state.effective_step(25_000)
    assert state.effective_step(27_500) == frozen
    first = state.observe_probe(
        global_step=27_500,
        shifted_teacher_forced_exact=0.8,
        shifted_greedy_top1=0.75,
        consistency_is_active=True,
    )
    assert first.event == "recovery_waiting"
    recovered = state.observe_probe(
        global_step=30_000,
        shifted_teacher_forced_exact=0.8,
        shifted_greedy_top1=0.76,
        consistency_is_active=True,
    )
    assert recovered.event == "recovery_completed"
    assert state.effective_step(30_000) == frozen
    assert state.effective_step(32_500) == frozen + 2_500
    recurrence = state.observe_probe(
        global_step=32_500,
        shifted_teacher_forced_exact=0.8,
        shifted_greedy_top1=0.70,
        consistency_is_active=True,
    )
    assert "second time" in recurrence.failure


def test_recovery_timeout_and_state_round_trip():
    state = make_curriculum()
    state.observe_probe(
        global_step=20_000,
        shifted_teacher_forced_exact=0.5,
        shifted_greedy_top1=0.8,
        consistency_is_active=False,
    )
    state.observe_probe(
        global_step=25_000,
        shifted_teacher_forced_exact=0.8,
        shifted_greedy_top1=0.7,
        consistency_is_active=True,
    )
    payload = state.state_dict()
    restored = make_curriculum()
    restored.load_state_dict(payload)
    assert restored.state_dict() == payload
    failure = restored.observe_probe(
        global_step=45_000,
        shifted_teacher_forced_exact=0.8,
        shifted_greedy_top1=0.7,
        consistency_is_active=True,
    )
    assert "recover within" in failure.failure


def test_checkpoint_state_rejects_changed_curriculum():
    state = make_curriculum()
    changed = AdaptiveCurriculum(
        nominal_max_steps=70_000,
        gate_threshold=0.6,
        gate_max_extra_steps=30_000,
        regression_drop=0.05,
        recovery_probes=2,
        recovery_timeout_steps=20_000,
    )
    with pytest.raises(ValueError, match="gate_threshold"):
        changed.load_state_dict(state.state_dict())


def test_lr_uses_frozen_effective_clock_and_finishes_at_zero():
    state = make_curriculum()
    state.observe_probe(
        global_step=20_000,
        shifted_teacher_forced_exact=0.4,
        shifted_greedy_top1=0.2,
        consistency_is_active=False,
    )
    first = effective_cosine_multiplier(
        state.effective_step(20_000), max_steps=70_000, warmup_steps=200
    )
    delayed = effective_cosine_multiplier(
        state.effective_step(40_000), max_steps=70_000, warmup_steps=200
    )
    assert delayed == first
    assert effective_cosine_multiplier(
        70_000, max_steps=70_000, warmup_steps=200
    ) == pytest.approx(0.0)
