"""``metrics.by_group`` and ``metrics.group_weights`` (spec MS4.v1 §2.4): the per-competition
Σ xG error and calibration slope with match-bootstrap intervals.

The point fixture has three groups of unequal size (6 / 4 / 2 shots, two matches each), listed
out of key order (B, C, A), so a row order by first appearance is wrong; a size-weighted or
pooled statistic differs from the per-group one on every row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from football_shot_models import metrics

# B (4 shots: ws:3, ws:4), C (2 shots: ws:5, ws:6, no goal), A (6 shots: ws:1, ws:2)
KEYS = ["B"] * 4 + ["C"] * 2 + ["A"] * 6
Y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0])
P = np.array([0.6, 0.5, 0.3, 0.2, 0.2, 0.3, 0.5, 0.6, 0.1, 0.4, 0.3, 0.2])
G = np.array(["ws:3", "ws:3", "ws:4", "ws:4", "ws:5", "ws:6", *(["ws:1"] * 3), *(["ws:2"] * 3)])


def _by_group(**kw: Any) -> tuple[metrics.GroupRow, ...]:
    args: dict[str, Any] = {"min_shots": 4, "n_boot": 200} | kw
    return metrics.by_group(Y, P, pl.Series(KEYS), G, args.pop("min_shots"), **args)


def test_by_group_points() -> None:
    """§2.4: one row per key, sorted by key; shots, goals and Σ p per group; signed error
    (Σ p - goals) / goals per group (not pooled: pooled is (4.2 - 4) / 4 = +0.05, A's value, so
    B pins it); the slope is
    ``metrics.calibration`` on the group's own rows."""
    rows = _by_group()
    assert [r.key for r in rows] == ["A", "B", "C"]
    assert [(r.shots, r.goals) for r in rows] == [(6, 2), (4, 2), (2, 0)]
    assert [r.xg for r in rows] == pytest.approx([2.1, 1.6, 0.5])
    a, b, c = rows
    assert a.signed_error == pytest.approx(0.05) and b.signed_error == pytest.approx(-0.20)
    assert c.signed_error is None
    m = np.array(KEYS) == "A"
    assert a.cal_slope == pytest.approx(metrics.calibration(Y[m].astype(float), P[m])[1])
    assert isinstance(a, metrics.GroupRow)


def test_by_group_intervals() -> None:
    """§2.4: each interval is the 2.5-97.5 percentile over match resamples within the group:
    B's two matches give Σ xG error +0.10, -0.20 or -0.50, so its interval is (-0.50, +0.10);
    A has a resample of its goal-less match only (0 goals), so its error interval is ``None``;
    a 1-match group has no interval; on a fixture with many matches per group every interval
    exists and contains its point value."""
    a, b, _ = _by_group()
    assert b.error_ci == pytest.approx((-0.50, 0.10))
    assert a.signed_error is not None and a.error_ci is None
    one = metrics.by_group(Y[:4], P[:4], pl.Series(KEYS[:4]), np.array(["ws:3"] * 4), 1,
                           n_boot=50)  # fmt: skip
    assert one[0].signed_error is not None and one[0].error_ci is None and one[0].slope_ci is None
    rng = np.random.default_rng(3)
    sizes = {"Cup": 6, "League A": 14, "League B": 10}  # matches per group, 20 shots each
    keys, groups, p = [], [], []
    for key, n in sizes.items():
        for j in range(n):
            keys += [key] * 20
            groups += [f"ws:{key}-{j}"] * 20
            p += list(rng.uniform(0.03, 0.5, 20))
    pa = np.array(p)
    y = (rng.uniform(size=pa.size) < pa).astype(int)
    y[::20] = 1  # a goal in every match: no resample has 0 goals
    rows = metrics.by_group(y, pa, pl.Series(keys), np.array(groups), 100, n_boot=200)
    assert [r.key for r in rows] == sorted(sizes)
    for r in rows:
        assert r.signed_error is not None and r.cal_slope is not None, r.key
        assert r.error_ci is not None and r.slope_ci is not None, r.key
        assert r.error_ci[0] <= r.signed_error <= r.error_ci[1], r.key
        assert r.slope_ci[0] <= r.cal_slope <= r.slope_ci[1], r.key


def test_by_group_none_rules() -> None:
    """§2.4: under ``min_shots`` the slope and its interval are ``None`` (B has 4 shots: 5 is
    one too many, 4 is enough); 0 goals gives no error and no error interval (C)."""
    rows = {r.key: r for r in _by_group(min_shots=5)}
    assert rows["A"].cal_slope is not None  # 6 shots
    assert rows["B"].cal_slope is None and rows["B"].slope_ci is None
    assert rows["C"].signed_error is None and rows["C"].error_ci is None
    assert rows["C"].cal_slope is None and rows["C"].slope_ci is None
    assert rows["B"].error_ci is not None  # min_shots never touches the error
    at_four = {r.key: r for r in _by_group(min_shots=4)}
    m = np.array(KEYS) == "B"
    slope_b = metrics.calibration(Y[m].astype(float), P[m])[1]
    assert slope_b is not None and at_four["B"].cal_slope == pytest.approx(slope_b)


BAD: dict[str, dict[str, Any]] = {
    "list keys": {"keys": KEYS},
    "bare str keys": {"keys": "A"},
    "Int keys": {"keys": pl.Series([1] * 12)},
    "Categorical keys": {"keys": pl.Series(KEYS, dtype=pl.Categorical())},
    "null key": {"keys": pl.Series([*KEYS[:-1], None], dtype=pl.String())},
    "short p": {"p": P[:-1]},
    "short keys": {"keys": pl.Series(KEYS[:-1])},
    "short groups": {"groups": G[:-1]},
    "p above 1": {"p": np.array([*P[:-1], 1.2])},
    "p below 0": {"p": np.array([-0.1, *P[1:]])},
    "p NaN": {"p": np.array([*P[:-1], np.nan])},
    "y not 0/1": {"y": np.array([*Y[:-1], 2])},
    "empty": {"y": Y[:0], "p": P[:0], "keys": pl.Series([], dtype=pl.String()), "groups": G[:0]},
}


@pytest.mark.parametrize("case", list(BAD))
def test_by_group_refuses(case: str) -> None:
    """§2.4: each refusal is a ``ValueError``."""
    args: dict[str, Any] = {"y": Y, "p": P, "keys": pl.Series(KEYS), "groups": G} | BAD[case]
    with pytest.raises(ValueError):
        metrics.by_group(args["y"], args["p"], args["keys"], args["groups"], 4, n_boot=20)


def test_group_weights() -> None:
    """§2.4: the group index per shot and ``(n_boot, groups)`` multinomial weights, each row
    summing to the group count, the same for the same seed (``bootstrap_gain`` draws these);
    the refusals of ``bootstrap_gain`` move with it."""
    inv, w = metrics.group_weights(np.array(["ws:b", "ws:a", "ws:b", "ws:c"]), 7, 0)
    assert inv.tolist() == [1, 0, 1, 2]
    assert w.shape == (7, 3) and w.sum(axis=1).tolist() == [3] * 7
    assert np.array_equal(w, metrics.group_weights(np.array(["ws:x", "ws:y", "ws:z"]), 7, 0)[1])
    for groups, n_boot in (
        (np.array(["ws:a", None], dtype=object), 5),
        (np.array([1.0, np.nan]), 5),
        (np.array(["ws:a", 1], dtype=object), 5),
        (np.array(["ws:a", "ws:a"]), 5),
        (np.array([], dtype=str), 5),
        (np.array(["ws:a", "ws:b"]), 0),
        (np.array(["ws:a", "ws:b"]), True),
    ):
        with pytest.raises(ValueError):
            metrics.group_weights(groups, n_boot, 0)


@settings(deadline=None, max_examples=30)
@given(
    st.lists(
        st.tuples(st.sampled_from(["A", "B", "C", "D"]), st.integers(0, 5), st.booleans(),
                  st.floats(0.0, 1.0)),
        min_size=1, max_size=40,
    )
)  # fmt: skip
def test_by_group_totals(rows: list[tuple[str, int, bool, float]]) -> None:
    """Property: the rows partition the shots (shots, goals and Σ p add up to the totals) and
    the keys are the distinct keys, sorted."""
    keys = pl.Series([r[0] for r in rows], dtype=pl.String())
    groups = np.array([f"ws:{r[0]}{r[1]}" for r in rows])
    y = np.array([int(r[2]) for r in rows])
    p = np.array([r[3] for r in rows])
    out = metrics.by_group(y, p, keys, groups, 1000, n_boot=10)
    assert [r.key for r in out] == sorted(set(keys))
    assert sum(r.shots for r in out) == len(rows) and sum(r.goals for r in out) == y.sum()
    assert sum(r.xg for r in out) == pytest.approx(p.sum())
