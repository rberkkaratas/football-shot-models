"""Placement features (spec ``runs/MS5/xgot`` §4.2; test plan 2).

The goal-mouth point is read through ``pitch.to_goal_frame`` (y positive towards the shooter's
right post), clipped to the frame plus the goal-mouth tolerance, then rounded half to even to
``PLACEMENT_STEP``. The side is the shot's side of the centre line, with a dead zone; ``far_side``
is +1 for the far post, -1 for the near post and 0 at the centre. Expected values below are
written by hand from those rules.
"""

from __future__ import annotations

import math

import polars as pl
import pytest
import xgot_cases as xg
from hypothesis import given, settings
from hypothesis import strategies as st

from football_shot_models import features

NAMES = ("mouth_y_abs", "mouth_z", "far_side", "mouth_far", "angle_x_mouth_y", "angle_x_mouth_z",
         "mouth_yz")  # fmt: skip
# goal_y, goal_z, shot y, angle -> mouth_y_abs, mouth_z, far_side
CASES = {
    "right post from y=20 is near": (30.34, 1.0, 20.0, 0.5, 3.7, 1.0, -1.0),
    "left post from y=20 is far": (37.66, 1.0, 20.0, 0.5, 3.7, 1.0, 1.0),
    "33.95 rounds to the centre": (33.95, 1.0, 20.0, 0.5, 0.0, 1.0, 0.0),
    "33.75 half to even": (33.75, 1.0, 50.0, 0.4, 0.2, 1.0, 1.0),
    "z 0.1499 down": (32.0, 0.1499, 50.0, 0.4, 2.0, 0.1, 1.0),
    "z 0.25 half to even": (36.0, 0.25, 50.0, 0.4, 2.0, 0.2, -1.0),
    "clip then round at the edge": (34.0 - 3.81, 2.59, 20.0, 0.3, 3.8, 2.6, -1.0),
    "beyond the edge clips": (29.0, 3.0, 50.0, 0.3, 3.8, 2.6, 1.0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_placement_values(case: str) -> None:
    """Each feature on hand-computed points (clip, round half to even, near / far post), and the
    spline terms are ``rcs_term`` of the rounded value at the spec's knots."""
    goal_y, goal_z, y, angle, y_abs, z, far = CASES[case]
    out = xg.placement(xg.mouth_frame([goal_y], [goal_z], [y], angle=[angle]), NAMES)
    expected = {
        "mouth_y_abs": y_abs, "mouth_z": z, "far_side": far, "mouth_far": far * y_abs,
        "angle_x_mouth_y": angle * y_abs, "angle_x_mouth_z": angle * z, "mouth_yz": y_abs * z,
    }  # fmt: skip
    for name, value in expected.items():
        assert out[name] == [pytest.approx(value, abs=1e-12)], (case, name)
    assert features.MOUTH_Y_KNOTS == (0.15, 0.9, 1.7, 2.5, 3.25)
    assert features.MOUTH_Z_KNOTS == (0.1, 0.3, 0.8, 1.25, 2.0)
    assert (features.PLACEMENT_STEP, features.CENTRE_DEAD_ZONE) == (0.1, 0.5)
    assert pytest.approx(3.81) == features.EDGE_Y and pytest.approx(2.59) == features.EDGE_Z
    splines = xg.placement(
        xg.mouth_frame([goal_y], [goal_z], [y], angle=[angle]),
        [f"mouth_y_abs_rcs{j}" for j in (1, 2, 3)] + [f"mouth_z_rcs{j}" for j in (1, 2, 3)],
    )
    for j in range(3):
        for col, knots, value in (("mouth_y_abs", features.MOUTH_Y_KNOTS, y_abs),
                                  ("mouth_z", features.MOUTH_Z_KNOTS, z)):  # fmt: skip
            ref = pl.DataFrame({"v": [value]}).select(features.rcs_term(pl.col("v"), knots, j))
            assert splines[f"{col}_rcs{j + 1}"] == [pytest.approx(ref.item(), abs=1e-9)]


def test_placement_rounds_grid() -> None:
    """A goal on StatsBomb's 0.1-yard grid and a projected save off it, in the same 0.1 m cell,
    give identical placement rows (the grid alone cannot separate them); a save one cell over
    differs (near miss)."""
    yard = 0.9144
    on_grid = 3 * 0.1 * yard  # 0.274 m from the centre, height 1.0 yd
    frame = xg.mouth_frame(
        [34.0 - on_grid, 33.69, 33.64], [1.0 * yard, 0.88, 0.88], [50.0, 50.0, 50.0]
    )
    out = xg.placement(frame, xg.placement_names())
    for name, values in out.items():
        assert values[0] == values[1], name
    assert out["mouth_y_abs"] == [pytest.approx(0.3), pytest.approx(0.3), pytest.approx(0.4)]
    assert out["mouth_z"][:2] == [pytest.approx(0.9), pytest.approx(0.9)]


def test_placement_off_target_raises() -> None:
    """Every placement feature declares ``on_target`` and refuses a row that is not on target,
    with the count; the all-on-target frame passes (near miss)."""
    frame = xg.mouth_frame([33.0, 35.0, 36.0], [1.0, 1.0, 1.0], [20.0, 20.0, 20.0])
    for name in xg.placement_names():
        assert len(xg.placement(frame, [name])[name]) == 3
    bad = xg.mouth_frame([33.0, 35.0, 36.0], [1.0, 1.0, 1.0], [20.0, 20.0, 20.0],
                         on_target=[True, False, False])  # fmt: skip
    for name in xg.placement_names():
        with pytest.raises(ValueError, match=r"\b2\b"):
            xg.placement(bad, [name])
        sources = {c for t, c in features.XGOT_DERIVED[name].sources if t == "shots"}
        assert {"goal_y", "goal_z", "on_target"} <= sources, name
        assert features.XGOT_DERIVED[name].reads == frozenset({"goal_y", "goal_z", "on_target"})


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    ("column", "feature"), [("goal_y", "mouth_y_abs"), ("goal_z", "mouth_z"), ("y", "far_side")]
)
def test_placement_non_finite_raises(column: str, feature: str, value: float | None) -> None:
    """A null, NaN or infinite ``goal_y``, ``goal_z`` or shot ``y`` is an error naming the
    column (no clip turns an infinity into the edge value)."""
    good = {"goal_y": [33.0, 35.0], "goal_z": [1.0, 1.0], "y": [20.0, 20.0]}
    assert len(xg.placement(xg.mouth_frame(**good), [feature])[feature]) == 2
    bad = {k: list(v) for k, v in good.items()}
    bad[column][1] = value
    with pytest.raises(ValueError, match=rf"\b{column}\b"):
        xg.placement(xg.mouth_frame(**bad), [feature])


@settings(deadline=None, max_examples=60)
@given(
    st.lists(
        st.tuples(
            st.floats(0.0, 68.0),
            st.floats(0.0, 4.0),
            st.floats(0.0, 68.0),
            st.floats(0.0, math.pi),
        ),  # fmt: skip
        min_size=1,
        max_size=12,
    )
)
def test_placement_property(rows: list[tuple[float, float, float, float]]) -> None:
    """For any on-target point: |y| <= 3.8, 0 <= z <= 2.6, both on the 0.1 grid; ``far_side`` in
    {-1, 0, 1} and ``mouth_far == far_side * mouth_y_abs``."""
    goal_y, goal_z, y, angle = (list(c) for c in zip(*rows, strict=True))
    out = xg.placement(xg.mouth_frame(goal_y, goal_z, y, angle=angle), NAMES)
    step = features.PLACEMENT_STEP
    for v in out["mouth_y_abs"]:
        assert 0.0 <= v <= 3.8 + 1e-9
        assert abs(v / step - round(v / step)) < 1e-9
    for v in out["mouth_z"]:
        assert 0.0 <= v <= 2.6 + 1e-9
        assert abs(v / step - round(v / step)) < 1e-9
    assert set(out["far_side"]) <= {-1.0, 0.0, 1.0}
    for f, a, m in zip(out["far_side"], out["mouth_y_abs"], out["mouth_far"], strict=True):
        assert m == pytest.approx(f * a, abs=1e-12)


@pytest.mark.parametrize(
    ("y", "far"),
    [(33.51, 0.0), (34.49, 0.0), (34.0, 0.0), (33.5, -1.0), (34.5, 1.0), (33.49, -1.0),
     (34.51, 1.0)],
)  # fmt: skip
def test_far_side_dead_zone(y: float, far: float) -> None:
    """A shot within ``CENTRE_DEAD_ZONE`` (strictly) of the centre line has no near or far post;
    at exactly 0.5 m it has one. The ball goes to the right post side (frame y +2.0): near for a
    shot from the right (y < 34), far from the left."""
    out = xg.placement(xg.mouth_frame([32.0], [1.0], [y]), ["far_side", "mouth_far"])
    assert out["far_side"] == [far]
    assert out["mouth_far"] == [pytest.approx(2.0 * far)]
