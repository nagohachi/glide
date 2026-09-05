"""Tests for the two-phase projector-warmup LR schedule (pure lambda math)."""

from glide.trainers.length_sampler_trainer import projector_warmup_lambdas


def _lams():
    # P=100, T=1000, ratio = target/proj = 2e-5/1e-4 = 0.2, explicit W=100.
    return projector_warmup_lambdas(
        projector_only_steps=100, num_training_steps=1000,
        target_lr=2e-5, projector_lr=1e-4, full_warmup_steps=100,
    )


def test_phase1_projector_only():
    proj, rest = _lams()
    # rest is held at 0 throughout phase 1; projector ramps 0 -> 1 (=projector_lr).
    assert proj(0) == 0.0 and rest(0) == 0.0
    assert abs(proj(50) - 0.5) < 1e-9 and rest(50) == 0.0
    assert rest(99) == 0.0 and proj(99) > 0.9


def test_phase2_ramp_to_target_and_decay():
    proj, rest = _lams()
    ratio = 2e-5 / 1e-4  # 0.2
    # At phase-2 boundary both restart from 0.
    assert proj(100) == 0.0 and rest(100) == 0.0
    # Peak at P+W=200: both reach `ratio` (-> effective LR = ratio*projector_lr = target_lr).
    assert abs(proj(200) - ratio) < 1e-9
    assert abs(rest(200) - ratio) < 1e-9
    # Decays back toward 0 by T.
    assert proj(999) < ratio and rest(999) < ratio
    assert proj(1000) <= 1e-9 and rest(1000) <= 1e-9


def test_rest_never_updates_in_phase1():
    proj, rest = _lams()
    assert all(rest(s) == 0.0 for s in range(0, 100))


def test_default_full_warmup_uses_ratio():
    # full_warmup_steps=0 -> W = warmup_ratio*(T-P) = 0.1*900 = 90.
    proj, rest = projector_warmup_lambdas(
        projector_only_steps=100, num_training_steps=1000,
        target_lr=1e-5, projector_lr=1e-4, full_warmup_steps=0, warmup_ratio=0.1,
    )
    ratio = 1e-5 / 1e-4
    assert abs(proj(190) - ratio) < 1e-9  # peak at P + 90
