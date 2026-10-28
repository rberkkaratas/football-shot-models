"""``gsax.fit``: the pure GSAx fit on an ``XgotFit`` (spec ``runs/MS6/gsax`` §4.7, §7 Rule 1).

The session fixture ``gsax_fits`` (``conftest``) runs ``gsax.fit`` once per provider on the
``xgot_fits`` fixture fits (``cache=None``) from an empty working directory, with stdout and
stderr captured and the tmp trees recorded before and after.
"""

from __future__ import annotations

import dataclasses
import math
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


def _open(g: Any) -> pl.DataFrame:
    from football_shot_models import gsax

    return gsax.variant(g.rated, gsax.WITHOUT)


@pytest.mark.parametrize("p", PROVIDERS)
def test_fit_wiring(gsax_fits: dict[str, Any], p: str) -> None:
    """Run6-ms6 (QA 2): ``fit`` rates the season-out rows with ``SMALL_LEAGUE`` and feeds the
    rated "without penalties" rows (q, not raw xGOT) to the CI counts, both keeper models, the
    next-season tests and the reliability tiers; its outputs equal those functions applied to
    ``variant(g.rated, WITHOUT)``. Precondition: δ is not 0 everywhere, so q differs from
    xGOT."""
    from football_shot_models import gsax, metrics

    g, calls = gsax_fits["fits"][p], gsax_fits["gsax_calls"][p]
    open_ = _open(g)
    assert open_["delta"].abs().max() > 0  # precondition: q != xgot on some row
    assert (open_["q"] != open_["xgot"]).any()
    rates = [a for n, a in calls if n == "rate"]
    assert len(rates) == 1 and rates[0]["small"] is gsax.SMALL_LEAGUE
    assert rates[0]["pred"].equals(g.predictions)
    key = ("shot_id", "q")
    want = open_.select(key).sort("shot_id")
    fed = [(n, a) for n, a in calls if n != "rate"]
    assert {n for n, _ in fed} == {"ci_counts", "fit_keeper_model", "next_season", "reliability"}
    for name, args in fed:
        assert args["rated"].select(key).sort("shot_id").equals(want), name
    assert [a["by"] for n, a in fed if n == "fit_keeper_model"] == [gsax.KEEPER_SEASON, gsax.CAREER]
    assert [a["by"] for n, a in fed if n == "next_season"][:2] == [gsax.KEEPER_SEASON, gsax.CAREER]
    assert g.model.estimates.equals(gsax.fit_keeper_model(open_, by=gsax.KEEPER_SEASON).estimates)
    assert g.career.estimates.equals(gsax.fit_keeper_model(open_, by=gsax.CAREER).estimates)
    assert g.tiers == gsax.reliability(open_)
    assert g.ci == gsax.ci_counts(open_, seed=metrics.SEED)
    assert g.next_season == gsax.next_season(open_, by=gsax.KEEPER_SEASON)
    assert g.next_career == gsax.next_season(open_, by=gsax.CAREER)


def _keepers(g: Any, column: str, value: float) -> Any:
    """``g`` with one keeper-season's ``column`` in every variant's table set to ``value``."""
    first = pl.int_range(pl.len()) == 0
    keepers = {
        v: t.with_columns(
            pl.when(first).then(pl.lit(value)).otherwise(pl.col(column)).alias(column)
        )
        for v, t in g.keepers.items()
    }
    return dataclasses.replace(g, keepers=keepers, checks=())


@pytest.mark.parametrize("p", PROVIDERS)
def test_checks_missed_branches(gsax_fits: dict[str, Any], p: str) -> None:
    """Run6-ms6 (QA 1, 7): each check that can fail is driven to ``missed``. Check 1: the rated
    rows read the match out-of-fold xGOT (which saw the season), or the rating reads other rows
    than the season-out ones; check 2: one keeper-season's raw or rated GSAx off its league sum;
    check 4: a CI on a one-match keeper-season (null exactly then) or a NaN SD. Check 6 is a
    reporting row; check 5 cites both groupings."""
    from football_shot_models import gsax

    g = gsax_fits["fits"][p]
    assert [c.result for i, c in enumerate(g.checks) if i != 4] == ["met"] * 5  # precondition
    oof = g.predictions.join(
        g.xgot.predictions.select("shot_id", oof="xgot"), on="shot_id", how="left",
        maintain_order="left",
    ).with_columns(xgot=pl.col("oof")).drop("oof")  # fmt: skip
    assert not oof["xgot"].equals(g.predictions["xgot"])  # precondition: season-out differs
    leak = dataclasses.replace(g, predictions=oof, rated=gsax.rate(oof, small=gsax.SMALL_LEAGUE))
    assert gsax.checks(leak)[0].result == "missed"
    other = dataclasses.replace(g, rated=g.rated.with_columns(xgot=pl.lit(0.5)))
    assert gsax.checks(other)[0].result == "missed"
    for column in ("gsax_raw", "gsax"):
        assert gsax.checks(_keepers(g, column, 99.0))[1].result == "missed", column
    assert (g.keepers[gsax.WITHOUT]["matches"] == 1).all()  # fixture: one match each, CI null
    assert gsax.checks(_keepers(g, "ci_low", -1.0))[3].result == "missed"  # a one-match CI
    assert gsax.checks(_keepers(g, "sd", math.nan))[3].result == "missed"
    five, six = g.checks[gsax.NEXT_SEASON_CHECK - 1], g.checks[5]
    assert "(1 | keeper)" in five.target and "; career " in five.measured
    assert "reporting check" in six.target
