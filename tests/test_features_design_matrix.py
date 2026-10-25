"""``design_matrix``: the only way to build a feature matrix, from C3 or derived features (§3)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

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


def _keeper_depth(seen: list[dict[str, list[str]]]) -> features.DerivedFeature:
    """Metres from the opponent keeper to the goal line, per shot (a freeze-frame feature)."""

    def fn(tables: object) -> pl.Series:
        assert isinstance(tables, Mapping)
        seen.append({t: frame.columns for t, frame in tables.items()})
        keepers = tables["freeze_frames"].filter(pl.col("is_keeper"))
        depth = keepers.select("shot_id", depth=105.0 - pl.col("x"))
        shots = tables["shots"].join(depth, on="shot_id", how="left", maintain_order="left")
        return shots["depth"]

    sources = (("freeze_frames", "is_keeper"), ("freeze_frames", "x"), ("shots", "x"))
    return features.DerivedFeature("keeper_depth", sources, fn)


def test_derived_feature_reads_only_its_sources() -> None:
    seen: list[dict[str, list[str]]] = []
    depth = _keeper_depth(seen)
    shots = xc.shots_frame(SHOTS)
    frames = xc.frames_for(shots).with_columns(x=pl.Series([104.0, 103.0, 101.5]))
    x, names = features.design_matrix(
        shots, ["distance_m", "keeper_depth"], derived={"keeper_depth": depth}, freeze_frames=frames
    )
    assert names == ("distance_m", "keeper_depth")
    np.testing.assert_allclose(x[:, 1], [1.0, 2.0, 3.5])
    # fn got shot_id plus the declared columns, nothing else (no is_goal, no provider_xg)
    assert seen == [{"freeze_frames": ["shot_id", "is_keeper", "x"], "shots": ["shot_id", "x"]}]


@pytest.mark.parametrize(
    ("call", "needle"),
    [
        (lambda d: features.design_matrix(xc.shots_frame(SHOTS), ["keeper_depth"], derived=d),
         "reads freeze_frames"),
        (lambda d: features.check_xg_features(["keeper_depth"]), "not a column of the C3 shots"),
        (lambda d: features.check_xg_features(["depth"], {"depth": d["keeper_depth"]}),
         "registered as 'depth'"),
        (lambda d: features.DerivedFeature("distance_m", (("shots", "x"),), len), "C3 shot column"),
        (lambda d: features.DerivedFeature("none", (), len), "needs a tuple"),
        (lambda d: features.design_matrix(
            xc.shots_frame(SHOTS), ["short"],
            derived={"short": features.DerivedFeature(
                "short", (("shots", "x"),), lambda t: t["shots"]["x"].head(1))}),
         "must return a Series of 3"),
    ],
    ids=["no freeze_frames", "not registered", "key mismatch", "C3 name", "no sources",
         "wrong length"],
)  # fmt: skip
def test_derived_feature_rejects(call: Callable[[dict[str, Any]], object], needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        call({"keeper_depth": _keeper_depth([])})
