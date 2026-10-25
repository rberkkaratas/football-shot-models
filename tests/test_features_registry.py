"""The derived-feature registry passes the M1.0 guard (spec MS3.features §3.1, §8): every source
is pre_shot, the keeper comes from the frame, and ``needs`` keeps the keeper 0.0 with its
indicator."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl
import pytest
import xg_cases as xc
from football_core import schema

from football_shot_models import baselines, features

ROLE = {
    (table, c.name): c.role
    for table, columns in (
        ("shots", schema.SHOT_COLUMNS),
        ("freeze_frames", schema.FREEZE_FRAME_COLUMNS),
        ("matches", schema.MATCH_COLUMNS),
    )
    for c in columns
}
SPLINES = tuple(f"{c}_rcs{j}" for c in ("distance_m", "angle_rad") for j in (1, 2, 3))
FRAME = (
    "opponents_in_cone", "teammates_in_cone", "gk_in_cone", "nearest_opponent_m", "gk_absent",
    "gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m",
)  # fmt: skip
KEEPER = ("gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m")
FRAME_SOURCES = {
    ("shots", "x"), ("shots", "y"), ("freeze_frames", "is_teammate"),
    ("freeze_frames", "is_keeper"), ("freeze_frames", "x"), ("freeze_frames", "y"),
}  # fmt: skip


def _variant_features(step: tuple[str, ...], kind: str, columns: tuple[str, ...]) -> list[str]:
    if kind == "drop":
        return [f for f in step if f not in columns]
    return [*step, *columns]


def test_derived_registry_passes_guard() -> None:
    """The acceptance criterion "every feature passes the M1.0 guard", for the registry and
    for every step and variant feature list."""
    from football_shot_models import ablation

    assert set(features.DERIVED) == {*SPLINES, "phase", "contact", *FRAME}
    assert len(features.DERIVED) == 16
    assert [f.name for f in features.FRAME_FEATURES] == list(FRAME)
    for f in features.FRAME_FEATURES:
        assert features.DERIVED[f.name] is f
        assert set(f.sources) == FRAME_SOURCES
    restricted = set(features.RESTRICTED_UNTIL)
    for name, feature in features.DERIVED.items():
        assert isinstance(feature, features.DerivedFeature)
        assert feature.name == name
        for table, column in feature.sources:
            assert ROLE[(table, column)] == "pre_shot", (name, table, column)
            # the keeper position is read from the frame only, never from shots.gk_*
            assert (table, column) not in {("shots", "gk_x"), ("shots", "gk_y")}
            if table == "shots" and column in restricted:
                assert name in features.RESTRICTED_READERS[column]
    assert features.DERIVED["phase"].sources == (("shots", "situation"), ("shots", "y"))
    assert features.check_xg_features(tuple(features.DERIVED)) == tuple(features.DERIVED)
    for spec in (*baselines.BASELINES, *ablation.STEPS):
        assert features.check_xg_features(spec.features) == spec.features
    steps = {s.name: s.features for s in ablation.STEPS}
    for v in ablation.VARIANTS:
        names = _variant_features(steps[v.step], v.kind, v.columns)
        assert features.check_xg_features(names) == tuple(names)


def test_needs_enforced() -> None:
    """A keeper distance of 0.0 means "no keeper" only next to ``gk_absent``: the guard refuses
    a keeper feature without it, naming the missing features in ``needs`` order."""
    for name in features.DERIVED:
        assert features.DERIVED[name].needs == (("gk_absent",) if name in KEEPER else ())
    for name in KEEPER:
        with pytest.raises(ValueError) as err:
            features.check_xg_features(["distance_m", name])
        assert f"derived feature {name!r} is defined only together with" in str(err.value)
        assert "gk_absent" in str(err.value)
        assert str(err.value).endswith("; add them to the features")
        both = [name, "gk_absent"]  # order within the list does not matter
        assert features.check_xg_features(both) == tuple(both)
    shots = xc.shots_frame([xc.Shot("sb:s1"), xc.Shot("sb:s2", goal=True)])
    with pytest.raises(ValueError, match="is defined only together with"):
        features.design_matrix(shots, ["gk_to_goal_m"], freeze_frames=xc.frames_for(shots))

    def fn(t: Mapping[schema.Table, pl.DataFrame]) -> pl.Series:
        raise AssertionError("not computed")

    src = (("shots", "distance_m"),)
    registry = {
        "a": features.DerivedFeature("a", src, fn),
        "b": features.DerivedFeature("b", src, fn),
        "c": features.DerivedFeature("c", src, fn, needs=("b", "a")),
    }
    with pytest.raises(ValueError) as err:
        features.check_xg_features(["c"], registry)
    message = str(err.value)
    assert message.startswith("derived feature 'c' is defined only together with")
    assert message.index("'b'") < message.index("'a'")
    with pytest.raises(ValueError) as err:
        features.check_xg_features(["a", "c"], registry)
    assert "'b'" in str(err.value) and "'a'" not in str(err.value).split("with", 1)[1]
    assert features.check_xg_features(["c", "a", "b"], registry) == ("c", "a", "b")


@pytest.mark.parametrize(
    "needs",
    [["gk_absent"], "gk_absent", (1,), ("gk_absent", None), ("self_ref",)],
    ids=["list", "bare str", "int", "None", "self"],
)
def test_needs_validated(needs: object) -> None:
    """``needs`` is a tuple of str naming other features; anything else fails when made."""
    with pytest.raises(ValueError):
        features.DerivedFeature(
            "self_ref",
            (("shots", "distance_m"),),
            lambda t: t["shots"]["distance_m"],
            needs=needs,
        )
