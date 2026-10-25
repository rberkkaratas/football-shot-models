"""Scores, calibration, fold aggregation, reliability bins and thresholds (spec §4)."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, settings

from football_shot_models import metrics

Y4 = np.array([True, False, False, True])
P4 = np.array([0.8, 0.2, 0.7, 0.6])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _scores(**kw: float | int | None) -> metrics.Scores:
    base: dict[str, float | int | None] = {
        "n": 100, "goals": 10, "xg": 10.0, "log_loss": 0.3, "brier": 0.08, "auc": 0.7,
        "cal_intercept": 0.0, "cal_slope": 1.0,
    }  # fmt: skip
    return metrics.Scores(**{**base, **kw})  # type: ignore[arg-type]


def test_score_known_values() -> None:
    """Hand-computed: 4 shots, 2 goals; positives (0.8, 0.6) vs negatives (0.2, 0.7)."""
    s = metrics.score(Y4, P4)
    assert (s.n, s.goals) == (4, 2)
    assert s.xg == pytest.approx(2.3, abs=1e-12)
    assert s.xg_error == pytest.approx(0.15, abs=1e-12)  # |2.3 - 2| / 2, a fraction
    ll = -(math.log(0.8) + math.log(0.8) + math.log(0.3) + math.log(0.6)) / 4
    assert s.log_loss == pytest.approx(ll, abs=1e-12)
    assert s.brier == pytest.approx((0.04 + 0.04 + 0.49 + 0.16) / 4, abs=1e-12)
    assert s.auc == pytest.approx(0.75, abs=1e-12)  # 3 of 4 pairs ordered
    assert (s.cal_intercept, s.cal_slope) == metrics.calibration(Y4.astype(np.float64), P4)
    assert metrics.score(Y4, np.full(4, 0.5)).auc == 0.5
    # y need not be bool: 0 / 1 as int64 or float64 scores the same (spec §4)
    assert metrics.score(Y4.astype(np.int64), P4) == s
    assert metrics.score(Y4.astype(np.float64), P4) == s


@pytest.mark.parametrize(
    "call",
    [
        lambda: metrics.score(np.array([], dtype=bool), np.array([])),
        lambda: metrics.score(np.array([True, True]), np.array([0.2, 0.3])),
        lambda: metrics.score(Y4, np.array([0.8, math.nan, 0.7, 0.6])),
        lambda: metrics.score(Y4, np.array([0.8, 0.2, 1.5, 0.6])),
        lambda: metrics.score(Y4, P4[:3]),
        lambda: metrics.score(Y4.reshape(2, 2), P4.reshape(2, 2)),
        lambda: metrics.reliability(Y4, P4, bins=0),
        lambda: metrics.reliability(Y4, P4, bins=5),
        lambda: metrics.mean_scores(()),
    ],
    ids=[
        "score empty",
        "score one class",
        "score nan",
        "score p > 1",
        "score length mismatch",
        "score 2-D",
        "reliability bins 0",
        "reliability n < bins",
        "mean_scores empty",
    ],
)
def test_metrics_reject(call: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        call()


@settings(deadline=None, max_examples=60)
@given(
    n=st.integers(40, 300),
    a=st.floats(-1.5, 1.5),
    b=st.floats(-2.0, 2.0),
    k=st.floats(0.5, 2.0),
    seed=st.integers(0, 2**32 - 1),
)
def test_property_calibration_fixed_point(n: int, a: float, b: float, k: float, seed: int) -> None:
    """The MLE's own fitted values are calibrated: intercept 0 and slope 1; logit scaled by k
    gives slope 1 / k."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, n)
    y = (rng.uniform(size=n) < _sigmoid(a + b * x)).astype(np.float64)
    assume(0 < y.sum() < n)
    z = np.column_stack([np.ones(n), x])
    beta = metrics._irls(z, y, np.zeros(n))
    assume(beta is not None)
    eta = z @ beta
    assume(np.max(np.abs(eta)) < 20 and np.ptp(eta) > 1e-6)
    intercept, slope = metrics.calibration(y, _sigmoid(eta))
    assert intercept == pytest.approx(0.0, abs=1e-6)
    assert slope == pytest.approx(1.0, abs=1e-6)
    _, scaled = metrics.calibration(y, _sigmoid(k * eta))
    assert scaled == pytest.approx(1.0 / k, abs=1e-6)


@pytest.mark.parametrize("case", ["constant p", "separation"])
def test_calibration_not_estimable(case: str) -> None:
    if case == "constant p":
        y = np.array([1.0, 0.0, 0.0, 0.0])
        intercept, slope = metrics.calibration(y, np.full(4, 0.2))
        assert slope is None
        # calibration-in-the-large: logit(goal rate) - logit(p) = logit(0.25) - logit(0.2)
        assert intercept == pytest.approx(math.log(1 / 3) - math.log(1 / 4), abs=1e-9)
    else:
        y = np.array([False, False, True, True])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        intercept, slope = metrics.calibration(y.astype(np.float64), p)
        assert slope is None
        assert isinstance(intercept, float)
        assert metrics.score(y, p).cal_slope is None


def test_mean_scores() -> None:
    """Fold means, not medians: every metric is skewed, so a median gives another value."""
    folds = (
        _scores(n=10, goals=1, xg=1.5, log_loss=0.1, brier=0.05, auc=0.6, cal_intercept=0.1,
                cal_slope=0.7),
        _scores(n=20, goals=2, xg=2.5, log_loss=0.2, brier=0.06, auc=0.9, cal_intercept=0.2,
                cal_slope=0.8),
        _scores(n=30, goals=3, xg=3.0, log_loss=0.9, brier=0.13, auc=0.9, cal_intercept=0.6,
                cal_slope=1.5),
    )  # fmt: skip
    m = metrics.mean_scores(folds)
    assert (m.n, m.goals) == (60, 6)
    assert m.xg == pytest.approx(7.0)
    assert m.xg_error == pytest.approx(1 / 6)  # pooled, not the mean of 0.5, 0.25, 0.0
    assert m.log_loss == pytest.approx(0.4)
    assert m.brier == pytest.approx(0.08)
    assert m.auc == pytest.approx(0.8)
    assert m.cal_intercept == pytest.approx(0.3)
    assert m.cal_slope == pytest.approx(1.0)
    with_none = metrics.mean_scores((*folds[:2], replace(folds[2], cal_slope=None)))
    assert with_none.cal_slope is None
    assert with_none.cal_intercept == pytest.approx(0.3)
    assert metrics.fold_sd(folds, "log_loss") == pytest.approx(0.19**0.5)  # ddof=1, not 0.3559
    mean, worst = metrics.fold_xg_error(folds)  # fold errors 0.5, 0.25, 0.0
    assert (mean, worst) == (pytest.approx(0.25), 0.5)
    with pytest.raises(ValueError):
        metrics.fold_xg_error(())


@settings(deadline=None)
@given(
    st.integers(1, 10).flatmap(
        lambda bins: st.tuples(
            st.just(bins),
            st.lists(
                st.tuples(st.booleans(), st.sampled_from([0.0, 0.1, 0.5, 1.0]) | st.floats(0, 1)),
                min_size=bins,
                max_size=60,
            ),
        )
    )
)
def test_property_reliability_bins(case: tuple[int, list[tuple[bool, float]]]) -> None:
    bins, rows = case
    y = np.array([r[0] for r in rows], dtype=bool)
    p = np.array([r[1] for r in rows], dtype=np.float64)
    n = len(rows)
    table = metrics.reliability(y, p, bins)
    assert table.columns == ["bin", "shots", "mean_xg", "goal_rate"]
    assert table["bin"].to_list() == list(range(bins))
    shots = table["shots"].to_list()
    assert sum(shots) == n
    assert max(shots) - min(shots) <= 1
    # (ordinal rank - 1) * bins // n, ties broken by input order
    order = np.argsort(p, kind="stable")
    for b in range(bins):
        members = order[[i for i in range(n) if i * bins // n == b]]
        row = table.row(b, named=True)
        assert row["shots"] == len(members)
        assert row["mean_xg"] == pytest.approx(p[members].mean(), abs=1e-12)
        assert row["goal_rate"] == pytest.approx(y[members].mean(), abs=1e-12)


@pytest.mark.parametrize(
    ("scores", "passed"),
    [
        (_scores(log_loss=0.2, brier=0.05, auc=0.8, xg=101.0, goals=100), [True] * 4),
        (_scores(log_loss=0.3, brier=0.08, auc=0.7, xg=105.0, goals=100), [False] * 4),
        (_scores(log_loss=0.275, brier=0.077, auc=0.75, xg=103.0, goals=100), [True] * 4),
    ],
    ids=["pass", "miss", "equal passes"],
)
def test_check_thresholds(scores: metrics.Scores, passed: list[bool]) -> None:
    assert [(t.metric, t.op, t.value) for t in metrics.THRESHOLDS] == [
        ("log_loss", "<=", 0.275),
        ("brier", "<=", 0.077),
        ("auc", ">=", 0.75),
        ("xg_error", "<=", 0.03),
    ]
    checks = metrics.check_thresholds(scores)
    assert checks == metrics.check_thresholds(scores, metrics.THRESHOLDS)  # the default
    assert [c.threshold for c in checks] == list(metrics.THRESHOLDS)
    assert [c.measured for c in checks] == [
        scores.log_loss,
        scores.brier,
        scores.auc,
        scores.xg_error,
    ]
    assert [c.passed for c in checks] == passed


def test_check_thresholds_custom_targets() -> None:
    """MS3 passes its own targets; a misspelt metric or op fails when the target is made."""
    s = _scores(log_loss=0.26, auc=0.79)
    targets = (metrics.Threshold("log_loss", "<=", 0.258), metrics.Threshold("auc", ">=", 0.78))
    checks = metrics.check_thresholds(s, targets)
    assert [(c.threshold, c.measured, c.passed) for c in checks] == [
        (targets[0], 0.26, False),
        (targets[1], 0.79, True),
    ]
    assert metrics.check_thresholds(s, ()) == ()
    with pytest.raises(ValueError, match="unknown metric 'logloss'"):
        metrics.Threshold("logloss", "<=", 0.3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown op '<'"):
        metrics.Threshold("brier", "<", 0.3)  # type: ignore[arg-type]
    assert metrics.METRICS == ("log_loss", "brier", "auc", "xg_error")
