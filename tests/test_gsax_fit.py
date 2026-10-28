"""``gsax.fit``: the pure GSAx fit on an ``XgotFit`` (spec ``runs/MS6/gsax`` §4.7, §7 Rule 1).

The session fixture ``gsax_fits`` (``conftest``) runs ``gsax.fit`` once per provider on the
``xgot_fits`` fixture fits (``cache=None``) from an empty working directory, with stdout and
stderr captured and the tmp trees recorded before and after.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

PROVIDERS = ("statsbomb", "whoscored")
FIELDS = (
    "xgot", "predictions", "rated", "keepers", "leagues", "ci", "model", "career", "next_season",
    "next_career", "next_no_team", "tiers", "checks",
)  # fmt: skip
KEY = ("competition_id", "season_id")


def test_fit_is_pure(gsax_fits: dict[str, Any]) -> None:
    """``fit`` writes and prints nothing (no file in the tmp roots or the working directory, no
    cache entry); it returns a ``GsaxFit`` with every §4.7 field, keyed tables per variant and
    the models fitted by keeper-season and by career."""
    from football_shot_models import gsax

    assert gsax_fits["stdout"] == "" and gsax_fits["stderr"] == ""
    assert gsax_fits["before"] == gsax_fits["after"]
    for p in PROVIDERS:
        g = gsax_fits["fits"][p]
        assert isinstance(g, gsax.GsaxFit)
        for name in FIELDS:
            assert hasattr(g, name), name
        assert g.xgot is gsax_fits["xgot"][p] and g.xgot.cache is None
        assert set(g.keepers) == set(gsax.VARIANTS) == set(g.leagues)
        assert tuple(g.model.by) == gsax.KEEPER_SEASON and tuple(g.career.by) == gsax.CAREER
        assert [t.tier for t in g.tiers] == list(gsax.TIERS)
        assert len(g.checks) == 6
    assert gsax_fits["fits"]["whoscored"].next_no_team is None  # no team-selected season


@pytest.mark.parametrize("p", PROVIDERS)
def test_fit_checks_and_tables(gsax_fits: dict[str, Any], p: str) -> None:
    """On the fixture: checks 1-4 and 6 are ``met``, check 5 ``n/a`` (each keeper plays one
    match: no keeper in two seasons); the rated rows are the predictions rated with
    ``SMALL_LEAGUE``; per variant the keeper table's shots equal its rows and its ``gsax`` sums
    to 0 per comp-season, and the league table is ``league_table`` of the same rows;
    ``keeper_rows`` holds only "without penalties" keeper-seasons with >= ``TABLE_MIN_SHOTS``
    shots, sorted by ``gsax`` descending, with ``gsax_shrunk`` and ``gsax_with``."""
    from football_shot_models import gsax

    g = gsax_fits["fits"][p]
    assert [c.result for c in g.checks] == ["met"] * 4 + ["n/a", "met"], [
        (c.target, c.measured, c.result) for c in g.checks
    ]
    assert g.next_season.untestable is not None
    rated = gsax.rate(g.predictions, small=gsax.SMALL_LEAGUE)
    assert (
        rated.sort("shot_id")
        .select("shot_id", "q", "delta")
        .equals(g.rated.sort("shot_id").select("shot_id", "q", "delta"))
    )
    for v, kinds in gsax.VARIANTS.items():
        rows = g.rated.filter(pl.col("kind").cast(pl.String).is_in(list(kinds)))
        kt = g.keepers[v]
        assert int(kt["shots"].sum()) == rows.height
        sums = kt.group_by(*KEY).agg(pl.col("gsax").sum())
        assert (sums["gsax"].abs() <= gsax.SUM_TOL).all(), v
        assert g.leagues[v].sort(*KEY).equals(gsax.league_table(rows).sort(*KEY))
    table = gsax.keeper_rows(g)
    assert {"gsax_shrunk", "gsax_with"} <= set(table.columns)
    assert table.height == 0 or (
        table["shots"].min() >= gsax.TABLE_MIN_SHOTS
        and table["gsax"].to_list() == sorted(table["gsax"].to_list(), reverse=True)
    )
