"""GSAx check row 3 after PR1 and the escalated band set (spec ``runs/PR1/keeper-fixes`` §3.2,
§3.3, §9; PM 2026-10-28 (a) "one penalty δ per (provider, source)"; PM E1 (a), PR1).

Row 3 is judged per rating group (open play: one δ per competition-season; penalties: one δ per
season-out source), every group at δ = 0 by rule (E2-small season, team-selected season or an
E2-small pool) excepted. The session ``gsax_census`` (``conftest``) has 2 team-selected seasons
(T18 also E2-small), 2 other small seasons (A20, F1), a fitted ``season_folds`` pool and an
E2-small ``holdout_fit`` pool.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import polars as pl
import pytest

ROW3 = 2  # 0-based index of check row 3


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _bump(rated: pl.DataFrame, mask: pl.Expr, by: float = 0.05) -> pl.DataFrame:
    """``rated`` with ``q`` raised by ``by`` on the first row matching ``mask``."""
    first = mask & (pl.int_range(pl.len()).over(mask) == 0)
    return rated.with_columns(q=pl.when(first).then(pl.col("q") + by).otherwise(pl.col("q")))


@pytest.mark.parametrize(
    ("where", "result"),
    [("open A18", "missed"), ("pool season_folds", "missed"), ("zero T19", "met"),
     ("small pool holdout_fit", "met"), ("small A20", "met")],
)  # fmt: skip
def test_row3_missed(gsax_census: dict[str, Any], where: str, result: str) -> None:
    """Row 3 can fail: one row's q off by 0.05 in a fitted group (A18's open play, the
    ``season_folds`` pool) -> ``missed``; the same in a group at δ = 0 by rule (team-selected
    T19, the E2-small ``holdout_fit`` pool, E2-small A20) -> still ``met``."""
    gsax = _gsax()
    g = gsax_census["g"]
    kind, season = pl.col("kind").cast(pl.String), pl.col("season_id")
    mask = {
        "open A18": (kind == "open_play") & (season == "sb:42"),
        "pool season_folds": (kind == "penalty") & (pl.col("group") == "season_folds"),
        "zero T19": (kind == "open_play") & (season == "sb:4"),
        "small pool holdout_fit": (kind == "penalty") & (pl.col("group") == "holdout_fit"),
        "small A20": (kind == "open_play") & (season == "sb:108"),
    }[where]
    assert g.rated.filter(mask).height  # precondition: the group exists
    assert g.rated.filter(mask)[0, "pool_small"] == (where == "small pool holdout_fit")
    bad = dataclasses.replace(g, rated=_bump(g.rated, mask))
    assert gsax.checks(bad)[ROW3].result == result, gsax.checks(bad)[ROW3]


def test_row3_counts(gsax_census: dict[str, Any]) -> None:
    """Row 3 is met on the census while its pooled kicks do not sum to 0 per competition-season
    (MS6's row 3, per season *and kind*, would miss); its target names one penalty δ per
    season-out source and the PM decision; its measured text counts the δ = 0 groups without
    double counting: 2 team-selected competition-seasons (1 also small: T18), 2 other small
    (A20, F1), 1 penalty pool at δ = 0 by E2 (``holdout_fit``); the counts come from
    ``rating_groups``."""
    gsax = _gsax()
    g = gsax_census["g"]
    row = g.checks[ROW3]
    assert row.result == "met", row
    pens = g.rated.filter(~gsax.AT_ZERO & (pl.col("kind").cast(pl.String) == "penalty"))
    per = pens.group_by("competition_id", "season_id").agg(
        (pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum().alias("gap")
    )
    assert per.height >= 2 and per["gap"].abs().max() > 1e-3  # MS6's per-season rule would miss
    assert "per rating group" in row.target and "season-out source" in row.target
    assert "PM 2026-10-28" in row.target
    want = (r"\b2 team-selected competition-seasons \(1 also small\), 2 other small, 1 penalty "
            r"pool\(s\) at δ = 0 by E2\b")  # fmt: skip
    assert re.search(want, row.measured), row.measured
    t = gsax.rating_groups(g.rated)
    open_ = t.filter(pl.col("kind") == "open_play")
    assert open_.filter(pl.col("zero_family")).height == 2  # recomputed from the groups
    assert open_.filter(pl.col("zero_family") & pl.col("small")).height == 1
    assert open_.filter(pl.col("small") & ~pl.col("zero_family")).height == 2
    assert t.filter(pl.col("pool_small")).height == 1


@pytest.mark.parametrize("column", ["delta", "small", "zero_family", "pool_small"])
def test_rating_groups_constant(gsax_census: dict[str, Any], column: str) -> None:
    """§3.2: ``delta`` and each flag must be constant within a (kind, group); one row changed
    raises ``ValueError`` (count named: 1 group)."""
    gsax = _gsax()
    rated = gsax_census["g"].rated
    mask = (pl.col("kind").cast(pl.String) == "open_play") & (pl.col("season_id") == "sb:42")
    first = mask & (pl.int_range(pl.len()).over(mask) == 0)
    new = pl.col(column) + 0.25 if column == "delta" else ~pl.col(column)
    bad = rated.with_columns(pl.when(first).then(new).otherwise(pl.col(column)).alias(column))
    with pytest.raises(ValueError, match=r"\b1\b"):
        gsax.rating_groups(bad)


def test_escalated_bands_constant() -> None:
    """PM E1 (a), PR1: ``ESCALATED_BANDS`` pins the WhoScored ``holdout_fit`` bands listed as the
    card's limitation (labels as ``band_table`` prints them); StatsBomb has none."""
    gsax = _gsax()
    assert dict(gsax.ESCALATED_BANDS) == {
        "statsbomb": frozenset(),
        "whoscored": frozenset({("mouth_y_abs", "[0, 1)", "holdout_fit"),
                                ("mouth_y_abs", "[3.3, 3.55)", "holdout_fit")}),
    }  # fmt: skip
    cuts = gsax.BAND_CUTS["mouth_y_abs"]
    assert (cuts[0], cuts[3], cuts[4]) == (1.0, 3.3, 3.55)  # the labels exist in band_table
