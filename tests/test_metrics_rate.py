"""The Beta interval of the penalty model and the signed Σ xG error (spec MS3.models §5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from football_shot_models import metrics


@pytest.mark.parametrize(
    ("goals", "n", "mean", "low", "high"),
    [
        (853, 1118, 0.762735, 0.737384, 0.787196),  # in-game pooled (spec_snippets.out)
        (242, 363, 0.666209, 0.617000, 0.713687),  # shoot-outs
        (1, 1, 0.75, 0.146746, 0.999614),
        (0, 10, 0.045455, 4.789043e-05, 0.217196),
    ],
)
def test_beta_interval_known(goals: int, n: int, mean: float, low: float, high: float) -> None:
    """§1.7, §5.1: Jeffreys Beta(1/2, 1/2): posterior mean (goals + 1/2) / (n + 1) and the 95%
    equal-tailed interval; ``rate`` is the raw conversion; numpy ints are accepted."""
    assert (metrics.JEFFREYS, metrics.LEVEL) == ((0.5, 0.5), 0.95)
    r = metrics.beta_interval(goals, n)
    assert (r.n, r.goals) == (n, goals)
    assert r.rate == goals / n
    assert r.mean == pytest.approx(mean, abs=1e-6)
    assert r.low == pytest.approx(low, abs=1e-6)
    assert r.high == pytest.approx(high, abs=1e-6)
    same = metrics.beta_interval(np.int64(goals), np.int64(n))
    assert same == r and type(same.n) is int and type(same.goals) is int
    uniform = metrics.beta_interval(goals, n, prior=(1.0, 1.0))
    assert uniform.mean == pytest.approx((goals + 1) / (n + 2), abs=1e-12)
    with pytest.raises(AttributeError):
        r.mean = 0.5  # type: ignore[misc]  # frozen


@pytest.mark.parametrize(
    ("goals", "n", "prior"),
    [
        (True, 3, (0.5, 0.5)),
        (1, True, (0.5, 0.5)),
        (1.0, 3, (0.5, 0.5)),
        (1, 3.0, (0.5, 0.5)),
        ("1", 3, (0.5, 0.5)),
        (2, 1, (0.5, 0.5)),
        (0, 0, (0.5, 0.5)),
        (-1, 3, (0.5, 0.5)),
        (1, 3, (0.0, 1.0)),
        (1, 3, (1.0, -1.0)),
        (1, 3, (math.nan, 1.0)),
        (1, 3, (1.0, math.inf)),
    ],
    ids=[
        "bool goals",
        "bool n",
        "float goals",
        "float n",
        "str goals",
        "goals > n",
        "n = 0",
        "negative goals",
        "zero prior",
        "negative prior",
        "NaN prior",
        "inf prior",
    ],  # fmt: skip
)
def test_beta_interval_rejects(goals: object, n: object, prior: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        metrics.beta_interval(goals, n, prior=prior)  # type: ignore[arg-type]


@given(st.integers(1, 5000).flatmap(lambda n: st.tuples(st.integers(0, n), st.just(n))))
def test_beta_interval_property(case: tuple[int, int]) -> None:
    """Property: 0 <= low < mean < high <= 1, and the Jeffreys mean lies between the raw rate
    and 1/2 (the prior pulls towards its mean)."""
    goals, n = case
    r = metrics.beta_interval(goals, n)
    assert 0.0 <= r.low < r.mean < r.high <= 1.0
    assert min(r.rate, 0.5) <= r.mean <= max(r.rate, 0.5)
    assert r.mean == pytest.approx((goals + 0.5) / (n + 1), rel=1e-12)


@pytest.mark.parametrize(
    ("xg", "goals", "signed"),
    [(975.0, 913, 62.0 / 913), (900.0, 913, -13.0 / 913), (913.0, 913, 0.0)],
)
def test_signed_xg_error(xg: float, goals: int, signed: float) -> None:
    """§5.2: ``xg_signed_error = (Σ xG - goals) / goals`` keeps the direction that
    ``xg_error`` drops (over-prediction on the hold-out is the GBM's open issue)."""
    s = metrics.Scores(9816, goals, xg, 0.25, 0.07, 0.8, 0.0, 1.0)
    assert s.xg_signed_error == pytest.approx(signed, abs=1e-15)
    assert s.xg_error == pytest.approx(abs(signed), abs=1e-15)
