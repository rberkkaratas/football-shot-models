"""The M1.0 guard over the canonical matches table (MS3.matches spec §4.4).

Only unrestricted ``pre_shot`` matches columns (today ``gender``) may be features, as direct
names and as ``DerivedFeature`` sources. A derived feature sees match columns per shot: the
columns ``load_xg_data`` joined onto the shots, in shots order, never the one-row-per-match
table.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from football_core import schema

from football_shot_models import features

SHOT_NAMES = frozenset(schema.column_names("shots"))
MATCH_ROLE = {c.name: c.role for c in schema.MATCH_COLUMNS}
NOT_PRE_SHOT = tuple(name for name, role in MATCH_ROLE.items() if role != "pre_shot")


def _shots_with_gender() -> pl.DataFrame:
    """Three shots of two matches, not in shot_id order, with the joined gender column."""
    shots = xc.shots_frame(
        [
            xc.Shot("sb:s3", match_id="sb:m01", x=99.0),
            xc.Shot("sb:s1", match_id="sb:m02", x=90.0, goal=True),
            xc.Shot("sb:s2", match_id="sb:m01", x=94.0),
        ]
    )
    gender = pl.Series("gender", ["female", "male", "female"], dtype=schema.GENDER)
    return shots.with_columns(gender)


@pytest.mark.parametrize("column", NOT_PRE_SHOT)
def test_match_non_pre_shot_rejected(column: str) -> None:
    """Scores (outcome: they contain the shot's own result), has_360, ids, names and the
    version are refused as direct names and as derived sources, with their role."""
    role = MATCH_ROLE[column]
    text = f"column {column!r} has role {role!r}; xG features must have role 'pre_shot'"
    assert column not in features.XG_ALLOWED
    with pytest.raises(ValueError, match="pre_shot") as err:
        features.check_xg_features(["distance_m", column])
    assert text in str(err.value)
    if column not in SHOT_NAMES:  # shared names are found in shots first, with the same role
        assert str(err.value) == f"matches {text}"
    with pytest.raises(ValueError, match="pre_shot") as err:
        features.DerivedFeature("d", (("shots", "distance_m"), ("matches", column)), len)
    assert f"matches {text}" in str(err.value)


def test_gender_is_allowed() -> None:
    assert features.check_xg_features(["distance_m", "gender"]) == ("distance_m", "gender")
    shots = _shots_with_gender()
    matrix, names = features.design_matrix(shots, ["distance_m", "gender"])
    assert names == ("distance_m", "gender=female")  # male is the reference category
    np.testing.assert_array_equal(matrix[:, 1], [1.0, 0.0, 1.0])

    seen: list[dict[str, pl.DataFrame]] = []

    def women_far(t: Mapping[str, pl.DataFrame]) -> pl.Series:
        tables = dict(t)
        seen.append(tables)
        far = tables["shots"]["distance_m"] > 10
        return (tables["matches"]["gender"] == "female") & far

    sources = (("shots", "distance_m"), ("matches", "gender"))
    derived = {"women_far": features.DerivedFeature("women_far", sources, women_far)}
    matrix, names = features.design_matrix(shots, ["women_far"], derived=derived)
    assert names == ("women_far",)
    np.testing.assert_array_equal(matrix[:, 0], [0.0, 0.0, 1.0])  # s3 is at 6 m, s2 at 11 m
    (tables,) = seen
    assert tables["shots"].columns == ["shot_id", "distance_m"]
    # per shot, in shots order and with the shots height: not one row per match (2 rows)
    assert tables["matches"].columns == ["shot_id", "gender"]
    assert tables["matches"]["shot_id"].to_list() == ["sb:s3", "sb:s1", "sb:s2"]
    assert tables["matches"]["gender"].to_list() == ["female", "male", "female"]

    with pytest.raises(
        ValueError,
        match=r"^derived feature 'women_far': shots lacks match column\(s\) \['gender'\]; "
        r"load with data\.load_xg_data$",
    ):
        features.design_matrix(shots.drop("gender"), ["women_far"], derived=derived)


@pytest.mark.parametrize("name", ["gender", "home_score", "has_360", "season_name"])
def test_derived_name_not_a_matches_column(name: str) -> None:
    """A derived feature may not shadow a matches column (it would hide the joined column)."""
    with pytest.raises(ValueError) as err:
        features.DerivedFeature(name, (("shots", "x"),), len)
    assert repr(name) in str(err.value)
