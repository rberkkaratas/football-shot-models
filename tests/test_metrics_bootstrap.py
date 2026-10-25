"""The match-bootstrap CI on the log-loss gain (spec MS3.features §4.1): matches are resampled,
not shots, for fixed fold models; the gain is the plain per-shot mean."""

from __future__ import annotations

import random

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings

from football_shot_models import metrics

# 100 shots, alternating goal / miss; the new model is better on shots 20-99, worse on 0-19.
Y = np.array([1.0, 0.0] * 50)
P_REF = np.full(100, 0.3)
P_NEW = np.where(Y == 1, 0.5, 0.2)
P_NEW[:20] = np.where(Y[:20] == 1, 0.1, 0.6)


def _loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    out: np.ndarray = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    return out


def test_bootstrap_identical_models_zero() -> None:
    assert (metrics.N_BOOT, metrics.SEED) == (2000, 0)
    ci = metrics.bootstrap_gain(Y, P_REF, P_REF, np.arange(100) // 2)
    assert ci == metrics.GainCI(0.0, 0.0, 0.0, 2000, 50)


def test_bootstrap_groups_not_shots() -> None:
    """Near miss for the resampling unit: the same per-shot gains in 100 one-shot groups or in
    50 two-shot groups (matches) give the same gain but different, wider intervals."""
    shots = metrics.bootstrap_gain(Y, P_REF, P_NEW, np.arange(100))
    pairs = metrics.bootstrap_gain(Y, P_REF, P_NEW, np.arange(100) // 2)
    gain = float(np.mean(_loss(Y, P_REF) - _loss(Y, P_NEW)))
    assert shots.gain == pytest.approx(gain, rel=1e-12) and pairs.gain == shots.gain
    assert (shots.groups, pairs.groups) == (100, 50)
    assert (shots.low, shots.high) == pytest.approx((-0.0072, 0.1835), abs=1e-4)
    assert (pairs.low, pairs.high) == pytest.approx((-0.0462, 0.2070), abs=1e-4)
    assert pairs.high - pairs.low > shots.high - shots.low
    # match ids as polars gives them (strings) are groups like any other key
    ids = pl.Series([f"sb:m{i // 2:03d}" for i in range(100)]).to_numpy()
    by_id = metrics.bootstrap_gain(Y, P_REF, P_NEW, ids)
    assert (by_id.gain, by_id.groups) == (pairs.gain, 50)
    assert (by_id.low, by_id.high) == pytest.approx((pairs.low, pairs.high), abs=1e-12)


def test_bootstrap_gain_is_the_shot_mean() -> None:
    """``gain`` is the mean over shots, not over matches: a one-shot match and a three-shot
    match with different per-shot gains (near miss: the mean of the two match means)."""
    y = np.array([1.0, 0.0, 0.0, 0.0])
    p_ref = np.array([0.5, 0.5, 0.5, 0.5])
    p_new = np.array([0.9, 0.4, 0.4, 0.4])
    ci = metrics.bootstrap_gain(y, p_ref, p_new, np.array(["sb:m1", "sb:m2", "sb:m2", "sb:m2"]))
    d = _loss(y, p_ref) - _loss(y, p_new)
    assert ci.gain == pytest.approx(d.mean(), rel=1e-12)
    assert ci.gain != pytest.approx((d[0] + d[1:].mean()) / 2, rel=1e-3)
    # with two groups a resample is (2, 0), (1, 1) or (0, 2): the interval spans the two
    # matches' own gains, which bound every ratio of weighted sums
    assert (ci.low, ci.high) == pytest.approx(tuple(sorted((d[1:].mean(), d[0]))), rel=1e-12)


def test_bootstrap_deterministic_and_sign() -> None:
    rng = np.random.default_rng(7)
    y = (rng.uniform(size=300) < 0.3).astype(np.float64)
    p_ref = rng.uniform(0.05, 0.6, size=300)
    p_new = np.clip(p_ref + np.where(y == 1, 0.1, -0.05) + rng.normal(0, 0.05, 300), 0.01, 0.99)
    groups = np.arange(300) // 6
    a = metrics.bootstrap_gain(y, p_ref, p_new, groups)
    assert a == metrics.bootstrap_gain(y, p_ref, p_new, groups)  # seeded: same every time
    assert a.gain > 0 and a.low > 0  # the new model is better
    other = metrics.bootstrap_gain(y, p_ref, p_new, groups, seed=1)
    assert other.gain == a.gain and (other.low, other.high) != (a.low, a.high)
    small = metrics.bootstrap_gain(y.astype(bool), p_ref, p_new, groups, n_boot=1)
    assert (small.n_boot, small.groups, small.low) == (1, 50, small.high)


GOOD = (Y, P_REF, P_NEW, np.arange(100) // 2)
_NULL = pl.Series([f"sb:m{i // 2}" for i in range(99)] + [None]).to_numpy()
_NAN = (np.arange(100) // 2).astype(np.float64)
_NAN[3] = np.nan


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((Y * 2, *GOOD[1:]), {}),
        ((np.where(Y == 1, np.nan, 0.0), *GOOD[1:]), {}),
        (tuple(v.reshape(50, 2) for v in GOOD), {}),
        ((Y, P_REF[:-1], P_NEW, GOOD[3]), {}),
        ((Y, P_REF, P_NEW[:-1], GOOD[3]), {}),
        ((Y, np.where(Y == 1, np.nan, 0.3), P_NEW, GOOD[3]), {}),
        ((Y, P_REF, np.where(Y == 1, 1.2, 0.3), GOOD[3]), {}),
        ((Y, P_REF, np.where(Y == 1, -0.1, 0.3), GOOD[3]), {}),
        ((Y, P_REF, P_NEW, GOOD[3][:-1]), {}),
        ((Y, P_REF, P_NEW, _NULL), {}),
        ((Y, P_REF, P_NEW, _NAN), {}),
        ((Y, P_REF, P_NEW, np.zeros(100)), {}),
        ((np.array([]), np.array([]), np.array([]), np.array([])), {}),
        (GOOD, {"n_boot": 0}),
    ],
    ids=[
        "y=2", "y NaN", "2-D", "p_ref length", "p_new length", "p NaN", "p > 1", "p < 0",
        "groups length", "null group", "NaN group", "1 group", "empty", "n_boot 0",
    ],
)  # fmt: skip
def test_bootstrap_rejects(args: tuple[np.ndarray, ...], kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        metrics.bootstrap_gain(*args, **kwargs)


@settings(deadline=None, max_examples=40)
@given(
    data=st.lists(
        st.tuples(st.booleans(), st.floats(0.01, 0.99), st.floats(0.01, 0.99), st.integers(0, 5)),
        min_size=4,
        max_size=40,
    ),
    order=st.randoms(use_true_random=False),
)
def test_property_bootstrap_symmetry_and_shot_order(
    data: list[tuple[bool, float, float, int]], order: random.Random
) -> None:
    """Swapping the models negates the gain and mirrors the interval; shuffling the shots
    (with their match keys) changes nothing, since matches are what is resampled."""
    y = np.array([d[0] for d in data], dtype=np.float64)
    a = np.array([d[1] for d in data])
    b = np.array([d[2] for d in data])
    g = np.array([d[3] for d in data])
    if np.unique(g).size < 2:
        g[0] = 6
    ci = metrics.bootstrap_gain(y, a, b, g, n_boot=200)
    back = metrics.bootstrap_gain(y, b, a, g, n_boot=200)
    assert back.gain == pytest.approx(-ci.gain, abs=1e-12)
    assert (back.low, back.high) == pytest.approx((-ci.high, -ci.low), abs=1e-12)
    perm = list(range(len(data)))
    order.shuffle(perm)
    shuffled = metrics.bootstrap_gain(y[perm], a[perm], b[perm], g[perm], n_boot=200)
    assert shuffled.groups == ci.groups
    assert (shuffled.gain, shuffled.low, shuffled.high) == pytest.approx(
        (ci.gain, ci.low, ci.high), abs=1e-12
    )


def test_bootstrap_resample_gain_is_shot_weighted() -> None:
    """run3-ms3 (QA 13), MS3.features §4.1: each resample's gain is its weighted difference over
    its weighted shot count. Five matches of 1, 2, 5, 1 and 3 shots (near miss: the mean of the
    match means gives (-0.0544, 0.5951) here)."""
    groups = np.repeat(np.arange(5), [1, 2, 5, 1, 3])
    y = np.array([1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1], dtype=float)
    p_ref = np.full(12, 0.3)
    p_new = np.array([0.6, 0.2, 0.2, 0.4, 0.3, 0.25, 0.35, 0.2, 0.6, 0.1, 0.4, 0.5])
    ci = metrics.bootstrap_gain(y, p_ref, p_new, groups)
    assert ci.gain == pytest.approx(float(np.mean(_loss(y, p_ref) - _loss(y, p_new))), rel=1e-12)
    assert (ci.low, ci.high) == pytest.approx((-0.0680, 0.4829), abs=1e-4)


def test_bootstrap_hostile_settings() -> None:
    """run3-ms3 (QA 17): a bool ``n_boot`` is refused (not one resample), and groups mixing int
    and str keys are a named ValueError, not numpy's TypeError."""
    with pytest.raises(ValueError, match="n_boot must be an int >= 1, got True"):
        metrics.bootstrap_gain(Y, P_REF, P_NEW, np.arange(100) // 2, n_boot=True)
    mixed = np.array([i if i % 2 else f"sb:m{i}" for i in range(100)], dtype=object)
    with pytest.raises(ValueError, match="groups must be keys of one type"):
        metrics.bootstrap_gain(Y, P_REF, P_NEW, mixed)
