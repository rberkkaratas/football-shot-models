"""``design_matrix``: the only way to build a feature matrix (spec §3)."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
import xg_cases as xc

from football_shot_models import features

SHOTS = (
    xc.Shot("sb:s1", x=99.0, y=34.0, body_part="right_foot", goal=True, is_home=True),
    xc.Shot("sb:s2", x=90.0, y=30.0, body_part="head", is_home=False, period=2, minute=50),
    xc.Shot("sb:s3", x=95.0, y=40.0, body_part="left_foot", is_home=True),
)
POLYGON = pl.List(pl.Array(pl.Float64(), 2))


def test_design_matrix_columns() -> None:
    shots = xc.shots_frame(SHOTS)  # no body_part=other shot: that column must still exist
    x, names = features.design_matrix(shots, ["period", "is_home", "distance_m", "body_part"])
    assert names == (
        "period",
        "is_home",
        "distance_m",
        "body_part=left_foot",
        "body_part=head",
        "body_part=other",
    )
    assert x.dtype == np.float64
    assert x.shape == (3, 6)
    np.testing.assert_array_equal(x[:, 0], [1.0, 2.0, 1.0])  # Int8 -> float
    np.testing.assert_array_equal(x[:, 1], [1.0, 0.0, 1.0])  # Boolean -> 0 / 1
    np.testing.assert_array_equal(x[:, 2], shots["distance_m"].to_numpy())
    # right_foot is the reference category: no indicator of its own
    np.testing.assert_array_equal(x[:, 3:], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    x2, names2 = features.design_matrix(shots, ["angle_rad", "distance_m"])  # request order
    assert names2 == ("angle_rad", "distance_m")
    np.testing.assert_array_equal(x2[:, 0], shots["angle_rad"].to_numpy())


def _with(column: str, value: object) -> pl.DataFrame:
    """The fixture with one cell of ``column`` (row 2) replaced; other columns untouched."""
    shots = xc.shots_frame(SHOTS)
    values = shots[column].to_list()
    values[1] = value
    return shots.with_columns(pl.Series(column, values, dtype=shots.schema[column]))


def _polygons() -> pl.DataFrame:
    area = pl.Series("visible_area", [[[90.0, 30.0], [105.0, 30.0]]] * 3, dtype=POLYGON)
    return xc.shots_frame(SHOTS).with_columns(area)


@pytest.mark.parametrize(
    ("make", "feature", "needle"),
    [
        (lambda: _with("body_part", None), "body_part", r"'body_part' has 1 null value\(s\)"),
        (lambda: _with("distance_m", math.nan), "distance_m", r"^(?=.*distance_m)(?=.*\b1\b)"),
        (lambda: _with("angle_rad", -math.inf), "angle_rad", r"^(?=.*angle_rad)(?=.*\b1\b)"),
        (lambda: xc.shots_frame(SHOTS).drop("distance_m"), "distance_m", "distance_m"),
        (_polygons, "visible_area", "feature 'visible_area' has unsupported dtype"),
    ],
    ids=["null", "nan", "inf", "missing column", "list dtype"],
)
def test_design_matrix_rejects(make: Callable[[], pl.DataFrame], feature: str, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        features.design_matrix(make(), ["x", feature])


def test_design_matrix_no_features_shape() -> None:
    x, names = features.design_matrix(xc.shots_frame(SHOTS), [])
    assert x.shape == (3, 0)
    assert x.dtype == np.float64
    assert names == ()
