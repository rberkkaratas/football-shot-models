"""M1.3 freeze-frame and keeper features (spec MS3.features §3.4, §8): cone counts, nearest
outfield opponent and keeper geometry, each computed per shot from that shot's frame rows."""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from football_core import pitch, schema
from hypothesis import given, settings

from football_shot_models import features

FRAME = (
    "opponents_in_cone", "teammates_in_cone", "gk_in_cone", "nearest_opponent_m", "gk_absent",
    "gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m",
)  # fmt: skip
Player = tuple[bool, bool, float, float]  # (is_teammate, is_keeper, x, y)


def _shots(locations: dict[str, tuple[float, float]]) -> pl.DataFrame:
    """C3 shots in the given (not sorted) order; the first one is a goal (two classes)."""
    return xc.shots_frame(
        [xc.Shot(sid, x=x, y=y, goal=i == 0) for i, (sid, (x, y)) in enumerate(locations.items())]
    )


def _frames(players: dict[str, list[Player]]) -> pl.DataFrame:
    """C3 freeze-frame rows, built column by column, sorted by shot_id (not shots order)."""
    rows = [(sid, j, *p) for sid in sorted(players) for j, p in enumerate(players[sid])]
    return pl.DataFrame(
        {
            "shot_id": [r[0] for r in rows],
            "player_id": [f"sb:p{r[0][3:]}-{r[1]}" for r in rows],
            "is_teammate": [r[2] for r in rows],
            "is_keeper": [r[3] for r in rows],
            "x": [r[4] for r in rows],
            "y": [r[5] for r in rows],
        },
        schema=schema.FREEZE_FRAME_SCHEMA,
        strict=True,
    )


def _features(
    shots: pl.DataFrame, frames: pl.DataFrame, names: tuple[str, ...] = FRAME
) -> np.ndarray:
    matrix, design = features.design_matrix(shots, list(names), freeze_frames=frames)
    assert design == names
    return matrix


# a: central at 11 m. Opponent (100, 34) in the cone at 6 m; opponent (100, 37) just outside;
#    opponent (105, 36) on the goal line between the posts (an edge: inside); opponent keeper
#    (104, 34) in the cone; a teammate and the shooter's OWN keeper in the cone (teammates 2,
#    gk_in_cone from the opponent keeper only).
# b: one opponent 12 m away (capped at 10), no opponent keeper.
# c: tight angle; the keeper 34 m from goal centre (capped at 20).
# e: on the goal line (x = 105) outside the posts: the cone is empty, although the frame
#    players at x = 105 lie on the degenerate segment.
# f: the keeper (4 m) and a teammate (1 m) are nearer than the nearest outfield opponent (7 m).
LOCATIONS = {
    "sb:c": (103.0, 20.0),
    "sb:a": (94.0, 34.0),
    "sb:f": (100.0, 34.0),
    "sb:b": (99.5, 30.34),
    "sb:e": (105.0, 10.0),
}
PLAYERS: dict[str, list[Player]] = {
    "sb:a": [
        (False, False, 100.0, 34.0),
        (False, False, 100.0, 37.0),
        (False, True, 104.0, 34.0),
        (True, False, 98.0, 34.5),
        (True, True, 96.0, 34.0),
        (False, False, 105.0, 36.0),
    ],
    "sb:b": [(False, False, 99.5, 42.34)],
    "sb:c": [(False, True, 104.5, 0.0), (False, False, 90.0, 0.0)],
    "sb:e": [(False, False, 105.0, 50.0), (False, True, 105.0, 34.0)],
    "sb:f": [(False, True, 104.0, 34.0), (False, False, 100.0, 41.0), (True, False, 101.0, 34.0)],
}
EXPECTED = {  # in FRAME order
    "sb:a": [2.0, 2.0, 1.0, 6.0, 0.0, 1.0, 10.0, 0.0],
    "sb:b": [0.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 0.0],
    "sb:c": [0.0, 0.0, 0.0, 10.0, 0.0, 20.0, 20.0, 4.509390603792405],
    "sb:e": [0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0, 0.0],
    "sb:f": [0.0, 1.0, 1.0, 7.0, 0.0, 1.0, 4.0, 0.0],
}


def test_frame_features_fixture() -> None:
    assert (features.NEAREST_CAP_M, features.GK_CAP_M) == (10.0, 20.0)
    shots, frames = _shots(LOCATIONS), _frames(PLAYERS)
    matrix = _features(shots, frames)
    expected = np.array([EXPECTED[s] for s in LOCATIONS])  # shots order, not frame order
    np.testing.assert_allclose(matrix, expected, rtol=1e-12, atol=1e-12)
    for f in features.FRAME_FEATURES:  # each feature returns Float64 itself
        out = f.fn({"shots": shots.select("shot_id", "x", "y"), "freeze_frames": frames})
        assert out.dtype == pl.Float64()
        assert out.len() == shots.height


def test_frame_features_missing_frame_raises() -> None:
    """A shot without frame rows is an error for every frame feature (an Opta shot has none,
    so M1.3 specs fail closed on WhoScored), as is a frame with two opponent keepers."""
    shots = _shots(LOCATIONS)
    no_b = _frames({k: v for k, v in PLAYERS.items() if k != "sb:b"})
    empty = _frames({})
    two = dict(PLAYERS, **{"sb:a": [(False, True, 100.0, 34.0), *PLAYERS["sb:a"][1:]]})
    for f in features.FRAME_FEATURES:
        tables = {"shots": shots.select("shot_id", "x", "y")}
        with pytest.raises(ValueError, match=rf"^derived feature '{f.name}': 1 shot\(s\) have no"):
            f.fn({**tables, "freeze_frames": no_b})
        with pytest.raises(ValueError, match=rf"'{f.name}': 5 shot\(s\) have no frame rows"):
            f.fn({**tables, "freeze_frames": empty})
        with pytest.raises(ValueError, match=rf"'{f.name}': 1 frame\(s\) hold 2\+ opponent"):
            f.fn({**tables, "freeze_frames": _frames(two)})
    with pytest.raises(ValueError, match="no frame rows"):
        _features(shots, no_b)
    with pytest.raises(ValueError, match="reads freeze_frames; pass freeze_frames="):
        features.design_matrix(shots, ["gk_absent"])
    # no shots: an empty matrix, whatever the frames hold
    assert _features(shots.head(0), _frames(PLAYERS)).shape == (0, len(FRAME))


def test_keeper_features_geometry_and_absent() -> None:
    """The keeper is the opponent keeper's frame row (never shots.gk_x/gk_y); absent gives 0.0
    and gk_absent 1.0; distances are capped at 20 m; the offset is from the shot-angle bisector,
    so a keeper on the bisector at a tight angle is not "off line"."""
    d_r, d_l = np.hypot(5.0, pitch.RIGHT_POST_Y - 20.0), np.hypot(5.0, pitch.LEFT_POST_Y - 20.0)
    y_b = pitch.RIGHT_POST_Y + pitch.GOAL_WIDTH * d_r / (d_r + d_l)
    locations = {
        "sb:k1": (88.0, 34.0),  # keeper 35 m from goal (capped), 18 m from the shot (kept)
        "sb:k2": (100.0, 20.0),  # keeper on the bisector's goal-line point y_b
        "sb:k3": (100.0, 20.0),  # keeper at goal centre: off the bisector
        "sb:k4": (94.0, 34.0),  # keeper absent, an opponent at the goal centre
    }
    players: dict[str, list[Player]] = {
        "sb:k1": [(False, True, 70.0, 34.0)],
        "sb:k2": [(False, True, 105.0, y_b)],
        "sb:k3": [(False, True, 105.0, 34.0)],
        "sb:k4": [(False, False, 105.0, 34.0), (True, True, 90.0, 34.0)],
    }
    shots, frames = _shots(locations), _frames(players)
    keeper = ("gk_absent", "gk_in_cone", "gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m")
    m = _features(shots, frames, keeper)
    np.testing.assert_allclose(m[0], [0.0, 0.0, 20.0, 18.0, 0.0], atol=1e-12)
    assert m[1, 4] == pytest.approx(0.0, abs=1e-9)  # the goal-centre line would give 0.283
    assert m[2, 4] == pytest.approx(0.2992741726275366, rel=1e-9)
    assert m[1, 1] == m[2, 1] == 1.0
    np.testing.assert_array_equal(m[3], [1.0, 0.0, 0.0, 0.0, 0.0])  # own keeper ignored
    # shots.gk_x / gk_y elsewhere change nothing: the frame row is the source
    moved = shots.with_columns(gk_x=pl.lit(60.0), gk_y=pl.lit(10.0))
    np.testing.assert_array_equal(_features(moved, frames, keeper), m)
    # a shot on the goal line between the posts has no bisector: NaN, refused
    line = _shots({"sb:g1": (105.0, 34.0), "sb:g2": (94.0, 34.0)})
    line_frames = _frames({s: [(False, True, 104.0, 34.0)] for s in ("sb:g1", "sb:g2")})
    with pytest.raises(ValueError, match=r"feature 'gk_off_line_m' has 1 NaN or infinite"):
        _features(line, line_frames, ("gk_absent", "gk_off_line_m"))


_situations = st.sampled_from(
    [s for s in schema.SITUATIONS if s not in ("penalty", "direct_corner")]
)
_player = st.tuples(st.booleans(), st.booleans(), st.floats(60.0, 105.0), st.floats(0.0, 68.0))


@st.composite
def _population(draw: st.DrawFn) -> tuple[pl.DataFrame, pl.DataFrame, list[int]]:
    n = draw(st.integers(2, 6))
    locations = {
        f"sb:s{i}": (draw(st.floats(70.0, 104.0)), draw(st.floats(1.0, 67.0))) for i in range(n)
    }
    players: dict[str, list[Player]] = {}
    for sid in locations:
        ps = draw(st.lists(_player, min_size=1, max_size=4))
        seen = {True: False, False: False}  # one keeper per side (C3)
        fixed = []
        for mate, keeper, x, y in ps:
            fixed.append((mate, keeper and not seen[mate], x, y))
            seen[mate] |= keeper
        players[sid] = fixed
    shots = _shots(locations).with_columns(
        situation=pl.Series([draw(_situations) for _ in range(n)], dtype=schema.SITUATION),
        technique=pl.Series(
            [draw(st.sampled_from(schema.TECHNIQUES)) for _ in range(n)], dtype=schema.TECHNIQUE
        ),
    )
    subset = draw(
        st.permutations(range(n)).flatmap(lambda p: st.integers(1, n).map(lambda k: p[:k]))
    )
    return shots, _frames(players), list(subset)


@settings(deadline=None, max_examples=60)
@given(_population())
def test_derived_per_shot_purity(case: tuple[pl.DataFrame, pl.DataFrame, list[int]]) -> None:
    """Every registered feature of a shot depends on that shot alone (no fitted knots, no
    cross-shot statistics): a reordered subset of shots gets the same rows as in the full set,
    with the frames of the other shots still present."""
    shots, frames, subset = case
    names = list(features.DERIVED)
    full, design = features.design_matrix(shots, names, freeze_frames=frames)
    part, again = features.design_matrix(shots[subset], names, freeze_frames=frames)
    assert again == design
    np.testing.assert_array_equal(part, full[subset])


def test_nearest_opponent_without_outfield_opponent() -> None:
    """run3-ms3 (QA 14), §1.3: a frame with only the opponent keeper and teammates has no
    outfield marker, so ``nearest_opponent_m`` is the cap (10 m, the most open value the +1
    monotone sign expects), not 0."""
    shots = _shots({"sb:g": (94.0, 34.0), "sb:b": LOCATIONS["sb:b"]})
    frames = _frames(
        {
            "sb:g": [(False, True, 104.0, 34.0), (True, False, 96.0, 34.0)],
            "sb:b": PLAYERS["sb:b"],
        }
    )
    matrix = _features(shots, frames, ("nearest_opponent_m",))
    assert matrix[:, 0].tolist() == [features.NEAREST_CAP_M, 10.0]
    assert features.NEAREST_CAP_M == 10.0
