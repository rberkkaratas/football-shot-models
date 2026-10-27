"""Expected calibration error and its noise floor (spec ``runs/MS5/xgot`` §7; test plan 5;
PM X3).

ECE is ``Σ_b n_b / n · |mean p_b − goal rate_b|`` over the equal-count bins of
``metrics.reliability`` (ties in input order), so the card's reliability table and its ECE read
the same bins. The values below are computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from football_shot_models import metrics


def test_ece_independent() -> None:
    """Three bins by hand; rows not in order; tied predictions split by input order (a pooled-
    ties ECE would give 0.05 on the second case); ``ECE_BINS`` is the default."""
    y = np.array([False, False, True, True, False, True])
    p = np.array([0.5, 0.1, 0.9, 0.4, 0.2, 0.8])
    # bins (0.1, 0.2) rate 0; (0.4, 0.5) rate 0.5; (0.8, 0.9) rate 1
    assert metrics.ece(y, p, bins=3) == pytest.approx((0.15 + 0.05 + 0.15) / 3, abs=1e-12)
    tied = np.full(4, 0.3)
    assert metrics.ece(np.array([1, 0, 0, 0]), tied, bins=2) == pytest.approx(0.25, abs=1e-12)
    assert metrics.ECE_BINS == 10
    rng = np.random.default_rng(3)
    q = rng.uniform(0.02, 0.9, 200)
    yy = rng.uniform(size=200) < q
    rel = metrics.reliability(yy, q, bins=10)
    by_hand = float(
        sum(n / 200 * abs(m - g) for _, n, m, g in rel.iter_rows())
    )  # the reliability table's own bins
    assert metrics.ece(yy, q) == pytest.approx(by_hand, abs=1e-12)
    assert metrics.ece(yy, q) == metrics.ece(yy, q, bins=metrics.ECE_BINS)
    assert metrics.ece(np.array([0, 1]), np.array([0.0, 1.0]), bins=2) == 0.0


def test_ece_floor_seeded() -> None:
    """The floor is the mean ECE of outcomes drawn from ``p`` itself: deterministic for a seed,
    0 when every ``p`` is 0 or 1, positive and small for a calibrated spread of ``p``."""
    p = np.random.default_rng(5).uniform(0.05, 0.6, 400)
    a = metrics.ece_floor(p)
    assert a == metrics.ece_floor(p, bins=metrics.ECE_BINS, reps=200, seed=metrics.SEED)
    assert metrics.ece_floor(p, seed=metrics.SEED + 1) != a
    assert metrics.ece_floor(p, reps=50) != a
    assert 0.0 < a < 0.1
    assert metrics.ece_floor(np.zeros(30)) == 0.0
    assert metrics.ece_floor(np.ones(30)) == 0.0
    small = metrics.ece_floor(np.full(40, 0.3), bins=10)
    large = metrics.ece_floor(np.full(4000, 0.3), bins=10)
    assert large < small  # the floor shrinks with the sample


@pytest.mark.parametrize(
    ("y", "p", "bins"),
    [
        ([0, 1, 1], [0.2, 0.5, 0.7], 0),  # bins < 1
        ([0, 1, 1], [0.2, 0.5, 0.7], 4),  # fewer shots than bins
        ([0, 1], [0.2, 0.5, 0.7], 2),  # lengths differ
        ([[0, 1]], [[0.2, 0.5]], 1),  # not 1-D
    ],
)
def test_ece_errors(y: list[int], p: list[float], bins: int) -> None:
    """The errors of ``reliability``."""
    with pytest.raises(ValueError):
        metrics.ece(np.array(y), np.array(p), bins=bins)
