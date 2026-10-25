"""M1.2 splines (spec MS3.features §3.2): fixed-knot restricted cubic spline terms of distance and
angle, registered as derived features so the leakage guard sees their inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from hypothesis import given, settings

from football_shot_models import features

SPLINE_NAMES = tuple(f"{c}_rcs{j}" for c in ("distance_m", "angle_rad") for j in (1, 2, 3))


def _rcs(values: list[float], knots: Sequence[float], j: int) -> list[float]:
    frame = pl.DataFrame({"v": values}, schema={"v": pl.Float64()})
    out: list[float] = frame.select(features.rcs_term(pl.col("v"), knots, j)).to_series().to_list()
    return out


def test_rcs_known_values() -> None:
    """Harrell's first term with the distance knots, at 0 / 6 / 30 / 40 m (spec §9)."""
    assert features.DISTANCE_KNOTS == (6.0, 11.5, 17.0, 22.5, 30.0)
    assert features.ANGLE_KNOTS == (0.2, 0.27, 0.35, 0.5, 1.0)
    values = _rcs([0.0, 6.0, 30.0, 40.0], features.DISTANCE_KNOTS, 0)
    assert values == pytest.approx([0.0, 0.0, 21.65625, 42.28125], rel=1e-12, abs=1e-12)
    # a list of knots is a Sequence too
    assert _rcs([40.0], list(features.DISTANCE_KNOTS), 0) == pytest.approx([42.28125], rel=1e-12)


def test_spline_features_registered() -> None:
    """``distance_m_rcs1..3`` and ``angle_rad_rcs1..3`` are rcs_term of their own column with
    their own knots (the angle terms must not use the distance knots), read per shot."""
    shots = xc.shots_frame(
        [
            xc.Shot("sb:s1", x=101.0, y=34.0),  # 4 m: below the first distance knot
            xc.Shot("sb:s2", x=94.0, y=30.0),
            xc.Shot("sb:s3", x=84.0, y=50.0),
            xc.Shot("sb:s4", x=70.0, y=20.0, goal=True),  # beyond the last distance knot
        ]
    )
    matrix, names = features.design_matrix(shots, list(SPLINE_NAMES))
    assert names == SPLINE_NAMES
    for k, name in enumerate(SPLINE_NAMES):
        column, j = name.rsplit("_rcs", 1)
        knots = features.DISTANCE_KNOTS if column == "distance_m" else features.ANGLE_KNOTS
        expected = shots.select(features.rcs_term(pl.col(column), knots, int(j) - 1)).to_series()
        np.testing.assert_array_equal(matrix[:, k], expected.to_numpy())
        assert features.DERIVED[name].sources == (("shots", column),)
    assert np.all(matrix >= 0.0)
    assert np.all(matrix[0, :3] == 0.0)  # 4 m is below 6 m: every distance term is 0


@settings(deadline=None, max_examples=200)
@given(
    start=st.floats(-10.0, 10.0),
    gaps=st.lists(st.floats(0.05, 10.0), min_size=2, max_size=5),
    j_frac=st.floats(0.0, 0.999),
    below=st.floats(0.0, 10.0),
    beyond=st.floats(0.0, 20.0),
    h=st.floats(0.01, 10.0),
)
def test_rcs_property_zero_then_linear(
    start: float, gaps: list[float], j_frac: float, below: float, beyond: float, h: float
) -> None:
    """Every term is 0 at or below the first knot, never negative, and linear beyond the last
    knot (second difference 0, up to rounding of the cubic terms)."""
    knots = [start]
    for g in gaps:
        knots.append(knots[-1] + g)
    k = len(knots)
    j = int(j_frac * (k - 2))  # 0 .. k - 3
    x0 = knots[-1] + beyond
    left, f0, f1, f2 = _rcs([knots[0] - below, x0, x0 + h, x0 + 2 * h], knots, j)
    assert left == 0.0
    scale = (x0 + 2 * h - knots[0]) ** 3 / (knots[-1] - knots[0]) ** 2
    assert f0 - 2 * f1 + f2 == pytest.approx(0.0, abs=1e-9 * scale + 1e-12)
    inside = np.linspace(knots[0], x0, 25).tolist()
    assert min(_rcs(inside, knots, j)) >= -1e-9 * scale


@pytest.mark.parametrize(
    ("knots", "j"),
    [
        ((1.0, 2.0), 0),  # fewer than 3 knots
        ((1.0, 1.0, 2.0), 0),  # not strictly increasing
        ((3.0, 2.0, 1.0), 0),
        ((1.0, math.nan, 3.0), 0),
        ((1.0, 2.0, math.inf), 0),
        ((-math.inf, 2.0, 3.0), 0),
        ((1.0, 2.0, 3.0), 1),  # j outside 0 .. k - 3
        ((1.0, 2.0, 3.0), -1),
        ((1.0, 2.0, 3.0, 4.0), 2),
    ],
    ids=["two", "tie", "decreasing", "nan", "inf", "-inf", "j=1", "j=-1", "j=k-2"],
)
def test_rcs_bad_knots(knots: tuple[float, ...], j: int) -> None:
    with pytest.raises(ValueError, match="knots"):
        features.rcs_term(pl.col("v"), knots, j)
