"""Tests for src/codebase/metrics.py, in particular compute_opt_thres (GitHub issue #3)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "codebase"))

from metrics import compute_opt_thres  # noqa: E402


def test_import_regression():
    """Regression for issue #3: `from metrics import compute_opt_thres` raised ImportError."""
    import metrics

    assert callable(metrics.compute_opt_thres)


def test_returns_scalar_float_usable_as_threshold():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=200)
    y_pred = np.clip(y_true * 0.6 + rng.rand(200) * 0.4, 0, 1)
    th = compute_opt_thres(y_true, y_pred=y_pred)
    assert isinstance(th, float)
    # threshold must binarize predictions without error, as done at the call sites
    binarized = (y_pred >= th).astype(int)
    assert set(np.unique(binarized)).issubset({0, 1})


def test_perfectly_separable_scores():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    th = compute_opt_thres(y_true, y_pred)
    # threshold must separate the classes: all positives >= th, no negative >= th
    # except those allowed by the target FPR (0.15 of 4 negatives rounds to 0 or 1)
    assert (y_pred[y_true == 1] >= th).all()
    assert (y_pred[y_true == 0] >= th).sum() <= 1


def test_fpr_close_to_target():
    rng = np.random.RandomState(1)
    n = 5000
    y_true = rng.randint(0, 2, size=n)
    y_pred = np.clip(0.35 * y_true + rng.rand(n) * 0.75, 0, 1)
    target = 0.15
    th = compute_opt_thres(y_true, y_pred, target_fpr=target)
    neg = y_pred[y_true == 0]
    achieved_fpr = (neg >= th).mean()
    assert abs(achieved_fpr - target) < 0.02


def test_single_class_input_does_not_crash():
    y_true = np.zeros(10)
    y_pred = np.linspace(0, 1, 10)
    assert compute_opt_thres(y_true, y_pred) == 0.5
    y_true = np.ones(10)
    assert compute_opt_thres(y_true, y_pred) == 0.5


def test_threshold_is_finite_and_within_score_range():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0.2, 0.9, 0.1, 0.8, 0.7, 0.3])
    th = compute_opt_thres(y_true, y_pred)
    assert np.isfinite(th)
    assert y_pred.min() <= th <= y_pred.max()


def test_accepts_lists_and_pandas_like_inputs():
    th = compute_opt_thres([0, 1, 0, 1], y_pred=[0.1, 0.9, 0.2, 0.8])
    assert isinstance(th, float)


@pytest.mark.parametrize("target_fpr", [0.05, 0.15, 0.5])
def test_monotonic_in_target_fpr(target_fpr):
    """Higher allowed FPR must never raise the threshold."""
    rng = np.random.RandomState(2)
    y_true = rng.randint(0, 2, size=1000)
    y_pred = np.clip(0.4 * y_true + rng.rand(1000) * 0.7, 0, 1)
    th_low = compute_opt_thres(y_true, y_pred, target_fpr=0.01)
    th_here = compute_opt_thres(y_true, y_pred, target_fpr=target_fpr)
    assert th_here <= th_low + 1e-12
