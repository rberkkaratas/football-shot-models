"""Next-season test of the keeper model: ``season_start``, ``next_season`` and check 5 (spec
``runs/MS6/gsax`` §4.5, §4.7).

A pair is one keeper in one competition in two seasons whose start years are one apart. Per
test start year, the keeper model is fitted on every row with an earlier start; raw GSAx is the
keeper's own no-pooling shift; the three predictors (none, raw, shrunken) are scored by log loss
on the paired keepers' next-season rows, with match-bootstrap gains. The ``_loo`` gains replace
the test season's δ with one fitted on its rows not faced by tested keepers.

The fixture (``_pairs_frame``): sb:g01 A18 -> A19 -> A20 (2 pairs); sb:g04 A18 -> A19 and B18 ->
B19 (a pair per competition); sb:g02 concedes 0 in A18 and sb:g07 all goals in A19 (both pairs
left out); sb:g03 plays one season, sb:g05 skips a season (A18, A20), sb:g06 changes
competition (A18, B19): none paired. Season ids do not sort in season order.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import gsax_cases as gc
import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from football_shot_models import metrics

KEEPER_SEASON = ("gk_id", "competition_id", "season_id")
PAIRS = (  # (gk, rated season, test season)
    ("sb:g01", "A18", "A19"), ("sb:g01", "A19", "A20"), ("sb:g04", "A18", "A19"),
    ("sb:g04", "B18", "B19"),
)  # fmt: skip


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _pairs_frame(*, b19_unpaired: bool = True, a19_matches: int = 3) -> pl.DataFrame:
    b = [f"sb:b{i}" for i in range(1, a19_matches + 1)]
    parts = [
        gc.block("sb:g01", "A18", goals=gc.pattern(24, 6), xgot=gc.spread(24, 0.1, 0.6),
                 matches=["sb:a1", "sb:a2", "sb:a3"]),
        gc.block("sb:g01", "A19", goals=gc.pattern(24, 4), xgot=gc.spread(24, 0.1, 0.6), matches=b),
        gc.block("sb:g01", "A20", goals=gc.pattern(24, 5), xgot=gc.spread(24, 0.1, 0.6),
                 matches=["sb:c1", "sb:c2"]),
        gc.block("sb:g02", "A18", goals=[False] * 12, xgot=gc.spread(12, 0.1, 0.5),
                 matches=["sb:a4", "sb:a5"]),
        gc.block("sb:g02", "A19", goals=gc.pattern(12, 3), xgot=gc.spread(12, 0.1, 0.5),
                 matches=["sb:b4", "sb:b5"]),
        gc.block("sb:g03", "A18", goals=gc.pattern(20, 3), xgot=gc.spread(20, 0.2, 0.6),
                 matches=["sb:a6", "sb:a7"]),
        gc.block("sb:g04", "A18", goals=gc.pattern(20, 4), xgot=gc.spread(20, 0.1, 0.5),
                 matches=["sb:a8", "sb:a9"]),
        gc.block("sb:g04", "A19", goals=gc.pattern(20, 2), xgot=gc.spread(20, 0.1, 0.5),
                 matches=["sb:b6", "sb:b7"]),
        gc.block("sb:g04", "B18", goals=gc.pattern(16, 4), xgot=gc.spread(16, 0.2, 0.7),
                 matches=["sb:d1", "sb:d2"]),
        gc.block("sb:g04", "B19", goals=gc.pattern(16, 3), xgot=gc.spread(16, 0.2, 0.7),
                 matches=["sb:e1", "sb:e2"]),
        gc.block("sb:g05", "A18", goals=gc.pattern(20, 5), xgot=gc.spread(20, 0.1, 0.5),
                 matches=["sb:a10"]),
        gc.block("sb:g05", "A20", goals=gc.pattern(20, 3), xgot=gc.spread(20, 0.1, 0.5),
                 matches=["sb:c3", "sb:c4"]),
        gc.block("sb:g06", "A18", goals=gc.pattern(16, 4), xgot=gc.spread(16, 0.2, 0.6),
                 matches=["sb:a11"]),
        gc.block("sb:g07", "A19", goals=[True] * 6, xgot=gc.spread(6, 0.2, 0.6),
                 matches=["sb:b8"]),
        gc.block("sb:g07", "A20", goals=gc.pattern(12, 3), xgot=gc.spread(12, 0.2, 0.6),
                 matches=["sb:c5"]),
        gc.block("sb:g08", "A19", goals=gc.pattern(20, 3), xgot=gc.spread(20, 0.2, 0.7),
                 matches=["sb:b9", "sb:b10"]),
        gc.block("sb:g09", "B18", goals=gc.pattern(16, 5), xgot=gc.spread(16, 0.2, 0.7),
                 matches=["sb:d3"]),
    ]  # fmt: skip
    if b19_unpaired:
        b19 = gc.block("sb:g06", "B19", goals=gc.pattern(16, 2), xgot=gc.spread(16, 0.2, 0.6),
                       matches=["sb:e3", "sb:e4"])  # fmt: skip
        parts.append(b19)
    f = gc.blocks(*parts)
    return f.sort("shot_id")


def _key(season: str) -> tuple[str, str]:
    return gc.SEASONS[season][0], gc.SEASONS[season][1]


def _rows(rated: pl.DataFrame, gk: str | None, season: str) -> pl.DataFrame:
    comp, sid = _key(season)
    out = rated.filter((pl.col("competition_id") == comp) & (pl.col("season_id") == sid))
    return out if gk is None else out.filter(pl.col("gk_id") == gk)


def _loss(y: np.ndarray, p: np.ndarray) -> float:
    q = np.clip(p, metrics.EPS, 1 - metrics.EPS)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log1p(-q)))


def _expected(rated: pl.DataFrame) -> dict[str, Any]:
    """The next-season test written out: rolling fits per test start year, raw shifts, the three
    predictors, and the leave-out δ per test comp-season (0 in a ``small`` one, PM E2 (b); a
    test season with no unpaired row or a non-finite δ leaves the leave-out gains)."""
    gsax = _gsax()
    start = {s: gc._start(s) for s in gc.SEASONS}
    models: dict[int, Any] = {}
    y, m, p = (
        [],
        [],
        {k: [] for k in ("none", "raw", "shrunk", "none_loo", "raw_loo", "shrunk_loo")},
    )
    tested = {(gk, t1) for gk, _, t1 in PAIRS}
    for gk, t0, t1 in PAIRS:
        year = start[t0]
        if year not in models:
            starts = rated["season_name"].str.slice(0, 4).cast(pl.Int64)
            models[year] = gsax.fit_keeper_model(rated.filter(starts <= year),
                                                 by=gsax.KEEPER_SEASON)  # fmt: skip
        est = models[year].estimates
        comp, sid = _key(t0)
        u_s = est.filter((pl.col("gk_id") == gk) & (pl.col("competition_id") == comp)
                         & (pl.col("season_id") == sid))["u"].item()  # fmt: skip
        own = _rows(rated, gk, t0)
        u_r = gsax.league_shift(own["q"].to_numpy(), own["is_goal"].to_numpy().astype(float))
        test = _rows(rated, gk, t1)
        others = _rows(rated, None, t1).filter(
            ~pl.col("gk_id").is_in([g for g, s in tested if s == t1])
        )
        if bool(_rows(rated, None, t1)["small"].any()):
            d_loo = 0.0
        elif others.height:
            d_loo = gsax.league_shift(others["xgot"].to_numpy(),
                                      others["is_goal"].to_numpy().astype(float))  # fmt: skip
        else:
            d_loo = math.nan
        d_loo = d_loo if math.isfinite(d_loo) else math.nan
        o, o_loo = gc.logit(test["q"].to_numpy()), gc.logit(test["xgot"].to_numpy()) + d_loo
        y.append(test["is_goal"].to_numpy().astype(float))
        m.append(test["match_id"].to_numpy())
        for suffix, base in (("", o), ("_loo", o_loo)):
            p["none" + suffix].append(expit(base))
            p["raw" + suffix].append(expit(base + u_r))
            p["shrunk" + suffix].append(expit(base + u_s))
    Y, M = np.concatenate(y), np.concatenate(m)
    P = {k: np.concatenate(v) for k, v in p.items()}
    ok = np.isfinite(P["none_loo"])
    return {
        "y": Y, "m": M, "p": P, "loo_shots": int((~ok).sum()),
        "log_loss": tuple(_loss(Y, P[k]) for k in ("none", "raw", "shrunk")),
        "gain": metrics.bootstrap_gain(Y, P["raw"], P["shrunk"], M),
        "gain_none": metrics.bootstrap_gain(Y, P["none"], P["shrunk"], M),
        "gain_loo": metrics.bootstrap_gain(Y[ok], P["raw_loo"][ok], P["shrunk_loo"][ok], M[ok]),
        "gain_none_loo": metrics.bootstrap_gain(Y[ok], P["none_loo"][ok], P["shrunk_loo"][ok],
                                                M[ok]),
    }  # fmt: skip


def _same(a: Any, b: metrics.GainCI) -> None:
    assert a is not None
    assert (a.gain, a.low, a.high) == (
        pytest.approx(b.gain, abs=1e-9), pytest.approx(b.low, abs=1e-9),
        pytest.approx(b.high, abs=1e-9),
    )  # fmt: skip
    assert a.groups == b.groups


def test_season_start() -> None:
    """``"2015/2016"`` -> 2015, ``"2018"`` -> 2018; anything else raises with the value, a
    trailing newline, a second year that is not the first + 1 and non-ASCII digits included
    (run6-ms6 QA 11)."""
    gsax = _gsax()
    assert gsax.season_start("2015/2016") == 2015
    assert gsax.season_start("2018") == 2018
    for bad in ("2015/2016\n", "2018\n", "15/16", "2015/16", "2015-2016", " 2018", "", "2018/",
                "2016/2015", "2015/2017", "２０１８", "2015/２０１６"):  # fmt: skip
        with pytest.raises(ValueError) as e:
            gsax.season_start(bad)
        assert repr(bad) in str(e.value) or bad.strip() in str(e.value), bad


def test_pairs_and_left_out() -> None:
    """4 pairs (a keeper paired per competition, twice in a row), 2 left out (0 goals, all
    goals in the rated season); one-season, gap-year and cross-competition keepers unpaired;
    test shots and matches are the paired keepers' next-season rows."""
    gsax = _gsax()
    rated = gsax.rate(_pairs_frame(), small=None)
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    assert ns.untestable is None
    assert (ns.pairs, ns.left_out) == (4, 2)
    tests = pl.concat([_rows(rated, gk, t1) for gk, _, t1 in PAIRS])
    assert ns.test_shots == tests.height == 84
    assert ns.test_matches == tests["match_id"].n_unique() == 9


def test_next_season_log_losses() -> None:
    """The three log losses and the two gains, recomputed from the rolling fits (``by`` =
    ``KEEPER_SEASON``), the raw shifts and ``metrics.bootstrap_gain``."""
    gsax = _gsax()
    rated = gsax.rate(_pairs_frame(), small=None)
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    want = _expected(rated)
    assert len(ns.log_loss) == 3
    for got, exp in zip(ns.log_loss, want["log_loss"], strict=True):
        assert got == pytest.approx(exp, abs=1e-9)
    assert want["log_loss"][1] != pytest.approx(want["log_loss"][2], abs=1e-6)  # precondition
    _same(ns.gain, want["gain"])
    _same(ns.gain_none, want["gain_none"])


def test_leave_out_offsets() -> None:
    """``gain_loo`` / ``gain_none_loo`` use δ_loo, the shift of each test season's rows not faced
    by tested keepers (sb:g07's A20 rows count: his pair was left out), applied to logit xgot in
    place of δ. A test season with no unpaired row (B19 without sb:g06) leaves the leave-out
    gains (run6-ms6 football 5): they are scored on the other test seasons' rows, and its test
    shots and pairs are counted."""
    gsax = _gsax()
    rated = gsax.rate(_pairs_frame(), small=None)
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    want = _expected(rated)
    _same(ns.gain_loo, want["gain_loo"])
    _same(ns.gain_none_loo, want["gain_none_loo"])
    assert (ns.loo_missing, ns.loo_shots, ns.loo_pairs) == (0, 0, 0)
    assert abs(want["gain_loo"].gain - want["gain"].gain) > 1e-6  # precondition: δ_loo matters
    alone = gsax.rate(_pairs_frame(b19_unpaired=False), small=None)
    ns2 = gsax.next_season(alone, by=gsax.KEEPER_SEASON)
    assert ns2.gain is not None and ns2.untestable is None
    want2 = _expected(alone)
    b19 = _rows(alone, "sb:g04", "B19").height
    assert want2["loo_shots"] == b19 > 0  # precondition: B19 is the only season left out
    assert (ns2.loo_missing, ns2.loo_shots, ns2.loo_pairs) == (1, b19, 1)
    _same(ns2.gain_loo, want2["gain_loo"])
    _same(ns2.gain_none_loo, want2["gain_none_loo"])
    assert ns2.gain_loo is not None and ns2.gain_loo.groups < ns2.test_matches


def test_leave_out_small_season() -> None:
    """PM E2 (b) in the leave-out columns (run6-ms6 architect A6): a test season flagged
    ``small`` is rated at δ = 0 there too, never at a δ fitted on its unpaired keepers' rows
    (B19: sb:g04 paired, sb:g06 unpaired, 2 keepers and 32 shots); without sb:g06 it is still
    scored (no season left out)."""
    gsax = _gsax()
    rated = gsax.rate(_pairs_frame(), small=gsax.SMALL_LEAGUE)
    b19 = _rows(rated, None, "B19")
    assert b19["small"].all() and not _rows(rated, None, "A19")["small"].any()  # precondition
    g06 = b19.filter(pl.col("gk_id") == "sb:g06")
    d06 = gsax.league_shift(g06["xgot"].to_numpy(), g06["is_goal"].to_numpy().astype(float))
    assert abs(d06) > 0.05  # precondition: the unpaired keeper's own δ is not 0
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    want = _expected(rated)
    _same(ns.gain_loo, want["gain_loo"])
    _same(ns.gain_none_loo, want["gain_none_loo"])
    alone = gsax.next_season(
        gsax.rate(_pairs_frame(b19_unpaired=False), small=gsax.SMALL_LEAGUE), by=gsax.KEEPER_SEASON
    )
    assert (alone.loo_missing, alone.loo_shots) == (0, 0) and alone.gain_loo is not None


def test_untestable_counts() -> None:
    """No pair, or only left-out pairs: "no pair with a finite raw estimate"; every test row in
    one match: "fewer than 2 test matches"; the counts are kept and nothing raises."""
    gsax = _gsax()
    full = _pairs_frame()
    only_a18 = gsax.rate(full.filter(pl.col("season_id") == "sb:42"), small=None)
    ns = gsax.next_season(only_a18, by=gsax.KEEPER_SEASON)
    assert ns.untestable is not None and "no pair with a finite raw estimate" in ns.untestable
    assert (ns.pairs, ns.left_out, ns.test_shots) == (0, 0, 0)
    assert ns.gain is None and ns.gain_none is None
    g02 = gsax.rate(full.filter(pl.col("gk_id").is_in(["sb:g02", "sb:g08", "sb:g03"])),
                    small=None)  # fmt: skip
    ns = gsax.next_season(g02, by=gsax.KEEPER_SEASON)
    assert ns.untestable is not None and "no pair with a finite raw estimate" in ns.untestable
    assert (ns.pairs, ns.left_out) == (0, 1)
    one = _pairs_frame(a19_matches=1).filter(
        pl.col("season_id").is_in(["sb:42", "sb:90"]) & (pl.col("gk_id") != "sb:g04")
    )
    ns = gsax.next_season(gsax.rate(one, small=None), by=gsax.KEEPER_SEASON)
    assert ns.untestable is not None and "fewer than 2 test matches" in ns.untestable
    assert (ns.pairs, ns.test_matches) == (1, 1)


@pytest.mark.parametrize(
    ("low", "untestable", "result"),
    [(1e-9, None, "met"), (0.0, None, "missed"), (-1e-9, None, "missed"),
     (None, "no pair with a finite raw estimate (0 pairs)", "n/a")],
)  # fmt: skip
def test_check5_met_missed_na(
    gsax_fits: dict[str, Any], low: float | None, untestable: str | None, result: str
) -> None:
    """Check 5 (shrunken beats raw on the next season) is ``met`` iff ``gain.low > 0``, ``n/a``
    when ``untestable``, else ``missed``; the other rows do not read ``next_season``."""
    gsax = _gsax()
    g = gsax_fits["fits"]["statsbomb"]
    gain = None if low is None else metrics.GainCI(0.01, low, 0.02, metrics.N_BOOT, 12)
    ns = dataclasses.replace(g.next_season, gain=gain, untestable=untestable)
    checks = gsax.checks(dataclasses.replace(g, next_season=ns))
    assert len(checks) == 6
    assert checks[4].result == result
    if untestable is not None:
        assert untestable in checks[4].measured
    assert [c.result for i, c in enumerate(checks) if i != 4] == [
        c.result for i, c in enumerate(g.checks) if i != 4
    ]
