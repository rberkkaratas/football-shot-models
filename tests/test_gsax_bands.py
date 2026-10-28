"""The MS6 gsax result-audit fixes in ``gsax``: the rated q by placement band (issue 1), the
match out-of-fold diagnostic beside it, and the team-selected split by side (issue 2).

Frames are built column by column; every number is invented. The expected band counts and
gaps are recomputed here with numpy masks on the hand-placed rows, not through ``band_table``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl
import pytest

from football_shot_models import xgot


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


# three groups of unequal size, rows not in band or source order:
#   S: 100 swapped_fit rows at |y| 0, height 1.0 m, angle 0.45 rad, q 0.5, 80 goals (gap -30,
#      sd 5: outside 2 SD, exactly at the 100-shot minimum, so judged)
#   H: 99 holdout_fit rows at |y| 3.0 (the [3, 3.3) boundary), height 2.1 m, angle 1.1 rad,
#      q 0.3, 30 goals (one short of the minimum: not judged)
#   O: 1 holdout_fit row recorded 4.5 m from the centre (outside the frame), height 0.5 m
N_S, N_H = 100, 99


def _frames(kind_last: str = "open_play") -> tuple[pl.DataFrame, pl.DataFrame]:
    n = N_S + N_H + 1
    order = np.random.default_rng(7).permutation(n)
    src = np.array(["swapped_fit"] * N_S + ["holdout_fit"] * (N_H + 1))
    gy = np.array([34.0] * N_S + [31.0] * N_H + [34.0 - 4.5])
    gz = np.array([1.0] * N_S + [2.1] * N_H + [0.5])
    ang = np.array([0.45] * N_S + [1.1] * N_H + [0.2])
    q = np.array([0.5] * N_S + [0.3] * N_H + [0.6])
    goal = np.array([True] * 80 + [False] * 20 + [True] * 30 + [False] * 69 + [True])
    kind = ["open_play"] * (n - 1) + [kind_last]
    ids = [f"sb:s{i:04d}" for i in range(n)]
    rated = pl.DataFrame(
        {
            "shot_id": pl.Series([ids[i] for i in order], dtype=pl.String),
            "q": pl.Series(q[order], dtype=pl.Float64),
            "is_goal": pl.Series(goal[order], dtype=pl.Boolean),
            "kind": pl.Series([kind[i] for i in order], dtype=pl.String),
            "source": pl.Series(src[order], dtype=pl.String),
        }
    )
    geo = pl.DataFrame(
        {
            "shot_id": pl.Series(ids, dtype=pl.String),
            "goal_y": pl.Series(gy, dtype=pl.Float64),
            "goal_z": pl.Series(gz, dtype=pl.Float64),
            "on_target": pl.Series([True] * n, dtype=pl.Boolean),
            "angle_rad": pl.Series(ang, dtype=pl.Float64),
        }
    )
    return rated, geo


def _row(t: pl.DataFrame, feature: str, band: str, source: str) -> dict[str, Any]:
    rows = t.filter(
        (pl.col("feature") == feature) & (pl.col("band") == band) & (pl.col("source") == source)
    )
    assert rows.height == 1, (feature, band, source)
    return rows.row(0, named=True)


def test_band_table_counts_rule_and_outside() -> None:
    """Counts, Σ q − goals and SD per band and source (pooled = the sum of the sources); the
    100-shot minimum at its exact value (100 judged, 99 not); the 2-SD rule; the point outside
    the frame in its own ``|y|`` row (not in ``>= 3.55``) but in its height and angle bands;
    every band of every source listed, empty ones with 0 shots and not judged."""
    gsax = _gsax()
    rated, geo = _frames()
    t = gsax.band_table(rated, geo)
    assert t.columns == ["feature", "band", "source", "shots", "goals", "gap", "sd", "judged",
                         "within"]  # fmt: skip
    bands = {f: len(c) + 1 + (f == "mouth_y_abs") for f, c in gsax.BAND_CUTS.items()}
    assert t.height == 3 * sum(bands.values())  # all, holdout_fit, swapped_fit
    assert t["source"].unique(maintain_order=True).to_list() == [
        gsax.POOLED,
        "holdout_fit",
        "swapped_fit",
    ]
    s = _row(t, "mouth_y_abs", "[0, 1)", "swapped_fit")
    assert (s["shots"], s["goals"], s["judged"], s["within"]) == (100, 80, True, False)
    assert math.isclose(s["gap"], 100 * 0.5 - 80) and math.isclose(s["sd"], math.sqrt(25.0))
    h = _row(t, "mouth_y_abs", "[3, 3.3)", "holdout_fit")
    assert (h["shots"], h["goals"], h["judged"], h["within"]) == (99, 30, False, True)
    assert math.isclose(h["gap"], 99 * 0.3 - 30)
    out = _row(t, "mouth_y_abs", gsax.OUTSIDE, "holdout_fit")
    assert (out["shots"], out["goals"]) == (1, 1)
    assert _row(t, "mouth_y_abs", ">= 3.55", gsax.POOLED)["shots"] == 0
    assert _row(t, "mouth_z", "[0.3, 0.6)", gsax.POOLED)["shots"] == 1  # the outside point
    assert _row(t, "mouth_z", "[2, 2.3)", gsax.POOLED)["shots"] == N_H
    assert _row(t, "angle_rad", ">= 1", gsax.POOLED)["shots"] == N_H
    assert _row(t, "angle_rad", "[0.4, 0.5)", "swapped_fit")["shots"] == N_S
    pooled = t.filter(pl.col("source") == gsax.POOLED)
    parts = (
        t.filter(pl.col("source") != gsax.POOLED)
        .group_by("feature", "band")
        .agg(pl.col("shots", "goals", "gap").sum())
    )
    joined = pooled.join(parts, on=["feature", "band"], suffix="_sum")
    assert joined.height == pooled.height
    assert (joined["shots"] == joined["shots_sum"]).all()
    assert (joined["goals"] == joined["goals_sum"]).all()
    assert np.allclose(joined["gap"], joined["gap_sum"])
    for f in gsax.BAND_CUTS:
        assert pooled.filter(pl.col("feature") == f)["shots"].sum() == rated.height
    empty = t.filter(pl.col("shots") == 0)
    assert empty.height and not empty["judged"].any()


def test_band_table_minimum_read_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """The minimum is ``xgot.MIN_BAND_SHOTS`` read when called: at 99 the 99-shot band is
    judged, at 101 the 100-shot band is not."""
    gsax = _gsax()
    rated, geo = _frames()
    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 99)
    assert _row(gsax.band_table(rated, geo), "mouth_y_abs", "[3, 3.3)", "holdout_fit")["judged"]
    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 101)
    t = gsax.band_table(rated, geo)
    assert not _row(t, "mouth_y_abs", "[0, 1)", "swapped_fit")["judged"]


def test_band_table_errors() -> None:
    """A penalty row and a rated row without geometry raise ``ValueError``."""
    gsax = _gsax()
    rated, geo = _frames(kind_last="penalty")
    with pytest.raises(ValueError, match="not open play"):
        gsax.band_table(rated, geo)
    rated, geo = _frames()
    with pytest.raises(ValueError, match="1 rated row"):
        gsax.band_table(rated, geo.slice(1))


def test_placement_bands_and_oof_rated(gsax_fits: dict[str, Any]) -> None:
    """``oof_rated`` rates the xGOT card's match out-of-fold xGOT per row (same rows, same
    order), each non-small competition-season and kind summing to its goals;
    ``placement_bands`` is the season-out band table plus that diagnostic's gap and SD, the
    pooled rows covering every open-play rated row once per feature."""
    gsax = _gsax()
    for p, g in gsax_fits["fits"].items():
        alt = gsax.oof_rated(g)
        assert alt["shot_id"].to_list() == g.predictions["shot_id"].to_list(), p
        want = dict(g.xgot.predictions.select("shot_id", "xgot").iter_rows())
        assert np.allclose(alt["xgot"].to_numpy(), [want[s] for s in alt["shot_id"]]), p
        sums = (
            alt.filter(~pl.col("small") & pl.col("delta").is_finite())
            .group_by(*gsax.SEASON_KEY, "kind")
            .agg(s=(pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum())
        )
        assert sums.height and float(sums["s"].abs().max()) < 1e-6, p  # type: ignore[arg-type]
        t = gsax.placement_bands(g)
        assert t.columns[-2:] == ["gap_oof", "sd_oof"], p
        open_ = gsax.variant(g.rated, gsax.WITHOUT)
        pooled = t.filter(pl.col("source") == gsax.POOLED)
        for f in gsax.BAND_CUTS:
            part = pooled.filter(pl.col("feature") == f)
            assert part["shots"].sum() == open_.height, (p, f)
            gap = float((open_["q"] - open_["is_goal"].cast(pl.Float64)).sum())
            assert math.isclose(float(part["gap"].sum()), gap, abs_tol=1e-6), (p, f)
            alt_open = gsax.variant(alt, gsax.WITHOUT)
            gap_oof = float((alt_open["q"] - alt_open["is_goal"].cast(pl.Float64)).sum())
            assert math.isclose(float(part["gap_oof"].sum()), gap_oof, abs_tol=1e-6), (p, f)
        assert set(t["source"]) == {gsax.POOLED, *g.rated["source"].cast(pl.String)}, p


def test_team_selected_split() -> None:
    """Team-selected keeper-seasons by side at ``SELECTED_SIDE`` shots exactly (30 is the
    selected side, 29 an opponent); other families ignored; SD = √Σ sd²; no team-selected row
    gives an empty frame."""
    gsax = _gsax()
    assert gsax.SELECTED_SIDE == 30
    t = pl.DataFrame(
        {
            "family": pl.Series(
                [
                    gsax.TEAM_SELECTED,
                    "women",
                    gsax.TEAM_SELECTED,
                    gsax.TEAM_SELECTED,
                    gsax.TEAM_SELECTED,
                ],
                dtype=pl.String,
            ),  # fmt: skip
            "shots": pl.Series([29, 200, 30, 120, 4], dtype=pl.Int64),
            "gsax_raw": pl.Series([-1.0, 8.0, 1.5, 4.0, -0.5], dtype=pl.Float64),
            "gsax": pl.Series([-1.5, 9.0, 0.5, 3.0, -0.25], dtype=pl.Float64),
            "sd": pl.Series([2.0, 5.0, 3.0, 4.0, 1.0], dtype=pl.Float64),
        }
    )
    got = gsax.team_selected_split(t)
    # run6-ms6 (football 4): raw (before δ) beside rated
    assert got.columns == ["side", "keeper_seasons", "shots", "gsax_raw", "gsax", "sd"]
    rows = got.rows()
    assert [r[:3] for r in rows] == [("selected side", 2, 150), ("opponents", 2, 33)]
    assert math.isclose(rows[0][3], 5.5) and math.isclose(rows[1][3], -1.5)
    assert math.isclose(rows[0][4], 3.5) and math.isclose(rows[1][4], -1.75)
    assert math.isclose(rows[0][5], 5.0) and math.isclose(rows[1][5], math.sqrt(5.0))
    assert gsax.team_selected_split(t.filter(pl.col("family") == "women")).height == 0


def test_band_table_rule_between_two_and_three_sd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run6-ms6 (QA 8, architect A5): a judged band at 2.6 SD (100 shots, q 0.5, 37 goals: gap
    +13, SD 5) is outside the 2-SD rule; the rule is ``xgot.within`` read when called, so at
    ``BAND_SDS`` 3 the same band is within (one rule for the xGOT and GSAx band tables)."""
    gsax = _gsax()
    n = 100
    ids = [f"sb:r{i:03d}" for i in range(n)]
    rated = pl.DataFrame({
        "shot_id": ids, "q": [0.5] * n, "is_goal": [i < 37 for i in range(n)],
        "kind": ["open_play"] * n, "source": ["season_folds"] * n,
    })  # fmt: skip
    geo = pl.DataFrame({
        "shot_id": ids, "goal_y": [34.0] * n, "goal_z": [1.2] * n, "on_target": [True] * n,
        "angle_rad": [0.45] * n,
    })  # fmt: skip
    row = _row(gsax.band_table(rated, geo), "mouth_y_abs", "[0, 1)", "season_folds")
    assert (row["shots"], row["judged"]) == (100, True)
    assert math.isclose(row["gap"], 13.0) and math.isclose(row["sd"], 5.0)
    assert row["within"] is False and xgot.within(row["gap"], row["sd"]) is False
    monkeypatch.setattr(xgot, "BAND_SDS", 3.0)
    row = _row(gsax.band_table(rated, geo), "mouth_y_abs", "[0, 1)", "season_folds")
    assert row["within"] is True and xgot.within(13.0, 5.0) is True
