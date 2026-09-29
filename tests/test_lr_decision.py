"""The predeclared LR decision rule (scripts/lr_decision.py), on synthetic metrics."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "lr_decision", Path(__file__).resolve().parents[1] / "scripts" / "lr_decision.py")
lr_decision = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lr_decision)


def events(vals: dict[int, float], nan_train: bool = False) -> list[dict]:
    ev = [{"event": "val", "step": s, "val_loss": v, "val_ppl": 1.0, "val_tokens": 131072} for s, v in vals.items()]
    ev.append({"event": "train", "step": 1024, "train_loss": float("nan") if nan_train else 3.0, "grad_norm": 0.5})
    return ev


def vals(v512: float, v768: float, v1024: float, v0: float = 10.0) -> dict[int, float]:
    return {0: v0, 256: 7.0, 512: v512, 768: v768, 1024: v1024}


def decide(a_vals, b_vals, a_nan=False, b_nan=False):
    return lr_decision.decide(lr_decision.assess(events(a_vals, a_nan)), lr_decision.assess(events(b_vals, b_nan)))[0]


def test_both_stable_1e3_clearly_better_wins():
    assert decide(vals(6.0, 5.8, 5.7), vals(5.9, 5.7, 5.66)) == "1e-3"  # margin 0.07, final better


def test_both_stable_small_margin_defaults_to_6e4():
    assert decide(vals(6.0, 5.8, 5.7), vals(6.0, 5.79, 5.70)) == "6e-4"  # margin 0.005 < 0.02


def test_1e3_better_mean_but_worse_final_defaults_to_6e4():
    assert decide(vals(6.0, 5.8, 5.60), vals(5.9, 5.6, 5.65)) == "6e-4"  # margin 0.075 but val@1024 worse


def test_late_divergence_rejects_only_when_both_late_points_exceed():
    diverged = vals(5.0, 5.2, 5.3)  # both > 512 + 0.10
    one_bad = vals(5.0, 5.2, 4.9)  # only 768 exceeds
    assert lr_decision.assess(events(diverged))["stable"] is False
    assert lr_decision.assess(events(one_bad))["stable"] is True
    assert decide(vals(6.0, 5.8, 5.7), diverged) == "6e-4"
    assert decide(diverged, vals(6.0, 5.8, 5.7)) == "1e-3"  # only one passes: choose it


def test_non_finite_and_no_improvement_rejected():
    assert decide(vals(6.0, 5.8, 5.7), vals(5.0, 4.9, 4.8), b_nan=True) == "6e-4"
    no_gain = vals(10.5, 10.4, 10.4, v0=10.0)  # late_mean not below step 0
    assert lr_decision.assess(events(no_gain))["stable"] is False
    assert decide(no_gain, vals(10.5, 10.6, 10.7, v0=10.0)) is None  # neither passes: stop


def test_missing_validation_point_is_not_stable():
    partial = {0: 10.0, 256: 7.0, 512: 6.0}
    assert lr_decision.assess(events(partial))["stable"] is False
