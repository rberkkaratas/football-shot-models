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
from conftest import FIT_SPIED, GSAX_SPLITS

PROVIDERS = ("statsbomb", "whoscored")
FIELDS = (
    "xgot", "predictions", "rated", "keepers", "leagues", "ci", "model", "career", "next_season",
    "next_career", "next_no_team", "tiers", "checks", "model_no_team", "saw_test", "line", "noz",
    "line_seasons",
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
    rated = gsax.rate(g.predictions, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
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
        if v == gsax.WITHOUT:  # PR1: penalties sum to 0 per source pool, not per season
            assert (sums["gsax"].abs() <= gsax.SUM_TOL).all(), v
        fitted = rows.filter(~gsax.AT_ZERO)  # an E2-small pool sits at δ = 0
        gap = (fitted["q"] - fitted["is_goal"].cast(pl.Float64)).sum()
        assert abs(float(gap)) <= gsax.SUM_TOL, v
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
    """Run6-ms6 (QA 2): ``fit`` rates the season-out rows with ``SMALL_LEAGUE`` and
    ``ZERO_FAMILIES`` (PR1) and feeds the
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
    assert all(a["small"] is gsax.SMALL_LEAGUE for a in rates)
    assert all(a["zero_families"] is gsax.ZERO_FAMILIES for a in rates)
    headline = [a for a in rates if a["pred"].equals(g.predictions)]
    assert len(headline) == 1  # others (PM E2 (c): the z-free rows) rate open play only
    assert all(set(a["pred"]["kind"].cast(pl.String).unique().to_list()) == {"open_play"}
               for a in rates if a is not headline[0])  # fmt: skip
    key = ("shot_id", "q")
    want = open_.select(key).sort("shot_id")
    fed = [(n, a) for n, a in calls if n not in ("rate", *FIT_SPIED)]
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
    leak = dataclasses.replace(
        g,
        predictions=oof,
        rated=gsax.rate(oof, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES),
    )
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


@pytest.mark.parametrize("p", PROVIDERS)
def test_fit_feeds_sensitivities(
    gsax_fits: dict[str, Any], p: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration census (v8) of ``fit``'s own steps, beyond ``assemble``: StatsBomb (in
    ``LINE_PROVIDERS`` and ``NOZ_PROVIDERS``) feeds ``line_rated`` the season-out rows,
    ``stops(f)`` and ``f.coarse`` once, with saves on both sides of ``LINE_X``, and
    ``noz_rated`` the z-free season-out rows once (open play, xGOT not the headline's, from a
    ``-no-mouth_z`` refit); ``g.line`` / ``g.noz`` are their keeper tables. WhoScored reaches
    neither (both None). Both feed ``saw_test`` ``season_out.trained_seasons(f)``."""
    from football_shot_models import evaluation, gsax, season_out

    monkeypatch.setattr(evaluation, "N_SPLITS", GSAX_SPLITS[p])  # as the fixture's fit
    g, f, calls = gsax_fits["fits"][p], gsax_fits["xgot"][p], gsax_fits["gsax_calls"][p]
    by = {n: [a for m, a in calls if m == n] for n in FIT_SPIED}
    assert len(by["saw_test"]) == 1
    assert by["saw_test"][0]["trained"].equals(season_out.trained_seasons(f))
    if p == "whoscored":
        assert by["line_rated"] == by["noz_rated"] == [] and g.line is None and g.noz is None
        return
    (line,), (noz,) = by["line_rated"], by["noz_rated"]
    stops = gsax.stops(f)
    assert stops is not None and line["stops"].equals(stops)
    assert line["pred"].equals(g.predictions) and line["coarse"].equals(f.coarse)
    saves = g.predictions.filter((pl.col("kind") == "open_play") & ~pl.col("is_goal"))
    x = saves.join(stops, on="shot_id")["end_x"]
    assert (x < gsax.LINE_X).any() and (x >= gsax.LINE_X).any()  # both branches of the rule
    assert g.line is not None and g.line.equals(
        gsax.keeper_table(gsax.line_rated(g.predictions, stops, coarse=f.coarse))
    )
    assert g.line["shots"].sum() < gsax.variant(g.rated, gsax.WITHOUT).height  # far saves out
    rows = noz["pred"]
    assert set(rows["kind"].cast(pl.String).unique().to_list()) == {"open_play"}
    head = g.predictions.filter(pl.col("kind") == "open_play").select("shot_id", head="xgot")
    j = rows.join(head, on="shot_id", how="full", coalesce=True)
    assert j.height == rows.height == head.height and (j["xgot"] != j["head"]).any()
    names = [c["spec"].name for c in gsax_fits["calls"][p]]
    assert any(n.endswith("-no-mouth_z") for n in names), names
    assert g.noz is not None and g.noz.equals(gsax.keeper_table(gsax.noz_rated(rows)))


# ---- PR1 keeper-fixes §3.5, §3.7, §9: assemble on the census --------------------------------


def test_census_reached(gsax_census: dict[str, Any]) -> None:
    """v8 fixture census: ``gsax_census`` reaches every branch the wiring feeds, each at least
    once: kept and left-out pairs; a team-selected season, an E2-small one, one both; a fitted
    and an E2-small penalty pool; the leave-out NaN, ``AT_ZERO``, ``loo_small`` and fitted
    branches; stops on both sides of ``LINE_X`` and one on it; a seen and an unseen pair; a
    keeper-season >= ``TABLE_MIN_SHOTS``; the line table, ``model_no_team`` and
    ``next_no_team``; checks 1-4 and 6 met."""
    from football_shot_models import gsax

    g = gsax_census["g"]
    ns = g.next_season
    assert ns.pairs >= 1 and ns.left_out >= 1 and ns.untestable is None
    t = gsax.rating_groups(g.rated)
    op, pen = t.filter(pl.col("kind") == "open_play"), t.filter(pl.col("kind") == "penalty")
    assert op.filter(pl.col("zero_family") & ~pl.col("small")).height >= 1
    assert op.filter(pl.col("small") & ~pl.col("zero_family")).height >= 1
    assert op.filter(pl.col("small") & pl.col("zero_family")).height >= 1
    assert pen.filter(pl.col("pool_small")).height >= 1
    assert pen.filter(~pl.col("pool_small") & ~pl.col("small") & ~pl.col("zero_family")).height
    assert ns.loo_missing >= 1 and ns.loo_small >= 1 and ns.gain_loo is not None
    test_seasons = {(c, n) for _, c, _, n in ns.kept_pairs}
    flagged = g.rated.filter(pl.col("zero_family")).select(KEY).unique().rows()
    assert test_seasons & set(flagged)  # an AT_ZERO test season (T19)
    x = gsax_census["stops"]["end_x"]
    assert (x < gsax.LINE_X).any() and (x > gsax.LINE_X).any() and (x == gsax.LINE_X).any()
    seen, kept, left = g.saw_test
    assert 1 <= seen < kept and (kept, left) == (ns.pairs, ns.left_out)
    assert gsax.keeper_rows(g).height >= 1
    assert g.line is not None and g.model_no_team is not None and g.next_no_team is not None
    assert [c.result for i, c in enumerate(g.checks) if i != 4] == ["met"] * 5, g.checks


def test_saw_test_field(gsax_census: dict[str, Any], gsax_fits: dict[str, Any]) -> None:
    """§3.5: ``GsaxFit.saw_test`` = (S, P, L): ``saw_test(g.next_season, trained)``,
    ``next_season.pairs`` and ``.left_out`` (keeper-season grouping); census (1, 5, 1). On the
    WhoScored fixture S = 0: its test season is held out of every fold."""
    import gsax_cases as gc

    from football_shot_models import gsax

    g = gsax_census["g"]
    assert g.saw_test == (gc.CENSUS_SEEN, gc.CENSUS_KEPT, gc.CENSUS_LEFT_OUT)
    assert g.saw_test[0] == gsax.saw_test(g.next_season, gsax_census["trained"])
    ws = gsax_fits["fits"]["whoscored"]
    assert ws.saw_test == (0, ws.next_season.pairs, ws.next_season.left_out)


def _spied_assemble(c: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Any]]:
    """``assemble`` on the census inputs with pass-through spies on ``next_season`` and
    ``fit_keeper_model``; (name, ``rated`` argument, ``by``) per call, in order."""
    from football_shot_models import gsax

    seen: list[tuple[str, Any]] = []
    for name in ("next_season", "fit_keeper_model"):
        real = getattr(gsax, name)

        def spy(rated: pl.DataFrame, *, by: Any, _n: str = name, _f: Any = real) -> Any:
            seen.append((_n, (rated, tuple(by))))
            return _f(rated, by=by)

        monkeypatch.setattr(gsax, name, spy)
    gsax.assemble(c["f"], c["pred"], stops=c["stops"], trained=c["trained"])
    return seen


def test_next_no_team_reads_flagged_rows(
    gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.7 (spy): the "without team-selected" next-season test and keeper model read the rated
    open-play rows minus the team-selected seasons' rows (q as rated, δ = 0 there), never the
    raw predictions; the with-team test reads all of them. Precondition: the census has
    team-selected rows and a team-selected pair (sb:g31 T18 -> T19)."""
    from football_shot_models import gsax

    g = gsax_census["g"]
    open_ = gsax.variant(g.rated, gsax.WITHOUT)
    team = pl.col("family").cast(pl.String) == gsax.TEAM_SELECTED
    no_team = open_.filter(~team)
    assert open_.filter(team).height and g.next_no_team.pairs < g.next_season.pairs
    calls = _spied_assemble(gsax_census, monkeypatch)
    key = ("shot_id", "q")
    no_team_key = no_team.select(key).sort("shot_id")
    ns_calls = [a for n, a in calls if n == "next_season" and a[1] == gsax.KEEPER_SEASON]
    assert [r.select(key).sort("shot_id").equals(no_team_key) for r, _ in ns_calls].count(True) == 1
    assert any(r.select(key).sort("shot_id").equals(open_.select(key).sort("shot_id"))
               for r, _ in ns_calls)  # fmt: skip
    km = [a for n, a in calls if n == "fit_keeper_model" and a[1] == gsax.KEEPER_SEASON]
    assert any(r.select(key).sort("shot_id").equals(no_team_key) for r, _ in km)
    assert g.next_no_team == gsax.next_season(no_team, by=gsax.KEEPER_SEASON)


def test_model_no_team(gsax_census: dict[str, Any], gsax_fits: dict[str, Any]) -> None:
    """§3.7 / Football F1: ``model_no_team`` is ``fit_keeper_model`` on the open-play rows
    without team-selected seasons (τ̂ printed beside the headline τ̂); its groups hold no
    team-selected season; None when no row is team-selected (the fixture fits)."""
    from football_shot_models import gsax

    g = gsax_census["g"]
    open_ = gsax.variant(g.rated, gsax.WITHOUT)
    no_team = open_.filter(pl.col("family").cast(pl.String) != gsax.TEAM_SELECTED)
    want = gsax.fit_keeper_model(no_team, by=gsax.KEEPER_SEASON)
    m = g.model_no_team
    assert m is not None and m.tau == pytest.approx(want.tau, abs=1e-12)
    assert m.estimates.equals(want.estimates)
    team_seasons = set(open_.filter(pl.col("family").cast(pl.String) == gsax.TEAM_SELECTED)
                       .select(KEY).unique().rows())  # fmt: skip
    assert team_seasons and not set(m.estimates.select(KEY).rows()) & team_seasons
    assert m.estimates.height < g.model.estimates.height  # precondition: rows were dropped
    for p in PROVIDERS:
        assert gsax_fits["fits"][p].model_no_team is None
