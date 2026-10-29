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


def _expected(
    rated: pl.DataFrame,
    pairs: tuple[tuple[str, str, str], ...] = PAIRS,
    d_loo: dict[str, float] | None = None,
) -> dict[str, Any]:
    """The next-season test written out: rolling fits per test start year, raw shifts, the three
    predictors, and the leave-out δ per test comp-season (0 in a ``small`` one, PM E2 (b); a
    test season with no unpaired row or a non-finite δ leaves the leave-out gains); ``d_loo``
    gives the leave-out δ per test season instead (PR1 §3.4 order, worked out by the test)."""
    gsax = _gsax()
    start = {s: gc._start(s) for s in gc.SEASONS}
    models: dict[int, Any] = {}
    y, m, p = (
        [],
        [],
        {k: [] for k in ("none", "raw", "shrunk", "none_loo", "raw_loo", "shrunk_loo")},
    )
    tested = {(gk, t1) for gk, _, t1 in pairs}
    for gk, t0, t1 in pairs:
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
        season = _rows(rated, None, t1)
        flagged = bool(season["small"].any()) or (
            "zero_family" in season.columns and bool(season["zero_family"].any())
        )
        rule = gsax.SMALL_LEAGUE  # PR1 §3.4: E2 on the remainder, read when called
        small_rest = rule is not None and (
            others["gk_id"].n_unique() <= rule.max_keepers or others.height < rule.min_shots
        )
        if d_loo is not None:
            d = d_loo[t1]
        elif others.height == 0:  # 1. empty remainder
            d = math.nan
        elif flagged or small_rest:  # 2. the test season at δ = 0; 3. the remainder fails E2
            d = 0.0
        else:  # 4. its league shift
            d = gsax.league_shift(others["xgot"].to_numpy(),
                                  others["is_goal"].to_numpy().astype(float))  # fmt: skip
        d = d if math.isfinite(d) else math.nan
        o, o_loo = gc.logit(test["q"].to_numpy()), gc.logit(test["xgot"].to_numpy()) + d
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
    rated = gsax.rate(_pairs_frame(), small=None, zero_families=())
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
    rated = gsax.rate(_pairs_frame(), small=None, zero_families=())
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    want = _expected(rated)
    assert len(ns.log_loss) == 3
    for got, exp in zip(ns.log_loss, want["log_loss"], strict=True):
        assert got == pytest.approx(exp, abs=1e-9)
    assert want["log_loss"][1] != pytest.approx(want["log_loss"][2], abs=1e-6)  # precondition
    _same(ns.gain, want["gain"])
    _same(ns.gain_none, want["gain_none"])


def test_leave_out_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """``gain_loo`` / ``gain_none_loo`` use δ_loo, the shift of each test season's rows not faced
    by tested keepers (sb:g07's A20 rows count: his pair was left out), applied to logit xgot in
    place of δ (with ``SMALL_LEAGUE`` None, read when called: PR1 §3.4 step 3 would otherwise
    set the small remainders to 0). A test season with no unpaired row (B19 without sb:g06)
    leaves the leave-out
    gains (run6-ms6 football 5): they are scored on the other test seasons' rows, and its test
    shots and pairs are counted."""
    gsax = _gsax()
    monkeypatch.setattr(gsax, "SMALL_LEAGUE", None)  # no E2 minimum here: every δ_loo fitted
    rated = gsax.rate(_pairs_frame(), small=None, zero_families=())
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    want = _expected(rated)
    _same(ns.gain_loo, want["gain_loo"])
    _same(ns.gain_none_loo, want["gain_none_loo"])
    assert (ns.loo_missing, ns.loo_shots, ns.loo_pairs) == (0, 0, 0)
    assert abs(want["gain_loo"].gain - want["gain"].gain) > 1e-6  # precondition: δ_loo matters
    alone = gsax.rate(_pairs_frame(b19_unpaired=False), small=None, zero_families=())
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
    (B19: sb:g04 paired, sb:g06 unpaired, 2 keepers and 32 shots). Amended in PR1 (§3.4: an
    empty remainder is decided first): without sb:g06, B19 is left out of the leave-out pair
    (NaN, ``loo_missing``), not rated at 0."""
    gsax = _gsax()
    rated = gsax.rate(_pairs_frame(), small=gsax.SMALL_LEAGUE, zero_families=())
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
        gsax.rate(_pairs_frame(b19_unpaired=False), small=gsax.SMALL_LEAGUE, zero_families=()),
        by=gsax.KEEPER_SEASON,
    )
    b19 = _rows(_pairs_frame(b19_unpaired=False), "sb:g04", "B19").height
    # A19's remainder (3 keepers, 38 rows) and A20's (2 keepers) fail E2: δ_loo 0, loo_small
    assert (alone.loo_missing, alone.loo_shots, alone.loo_small) == (1, b19, 2)
    assert ns.loo_small == 2
    assert alone.gain_loo is not None


def test_untestable_counts() -> None:
    """No pair, or only left-out pairs: "no pair with a finite raw estimate"; every test row in
    one match: "fewer than 2 test matches"; the counts are kept and nothing raises."""
    gsax = _gsax()
    full = _pairs_frame()
    only_a18 = gsax.rate(full.filter(pl.col("season_id") == "sb:42"), small=None, zero_families=())
    ns = gsax.next_season(only_a18, by=gsax.KEEPER_SEASON)
    assert ns.untestable is not None and "no pair with a finite raw estimate" in ns.untestable
    assert (ns.pairs, ns.left_out, ns.test_shots) == (0, 0, 0)
    assert ns.gain is None and ns.gain_none is None
    g02 = gsax.rate(full.filter(pl.col("gk_id").is_in(["sb:g02", "sb:g08", "sb:g03"])),
                    small=None, zero_families=())  # fmt: skip
    ns = gsax.next_season(g02, by=gsax.KEEPER_SEASON)
    assert ns.untestable is not None and "no pair with a finite raw estimate" in ns.untestable
    assert (ns.pairs, ns.left_out) == (0, 1)
    one = _pairs_frame(a19_matches=1).filter(
        pl.col("season_id").is_in(["sb:42", "sb:90"]) & (pl.col("gk_id") != "sb:g04")
    )
    ns = gsax.next_season(gsax.rate(one, small=None, zero_families=()), by=gsax.KEEPER_SEASON)
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


# ---- PR1 keeper-fixes §3.4 (leave-out order), §3.5 (E4 count) on the census ----------------

CENSUS_PAIRS = (  # (gk, rated season, test season), the kept pairs of gsax_cases.census_frame
    ("sb:g01", "A18", "A19"), ("sb:g01", "A19", "A20"), ("sb:g21", "C18", "C19"),
    ("sb:g22", "C18", "C19"), ("sb:g31", "T18", "T19"),
)  # fmt: skip


def _census_open() -> pl.DataFrame:
    gsax = _gsax()
    rated = gsax.rate(gc.census_frame(), small=gsax.SMALL_LEAGUE,
                      zero_families=gsax.ZERO_FAMILIES)  # fmt: skip
    return gsax.variant(rated, gsax.WITHOUT)


def _remainder(rated: pl.DataFrame, season: str) -> pl.DataFrame:
    tested = [gk for gk, _, t1 in CENSUS_PAIRS if t1 == season]
    return _rows(rated, None, season).filter(~pl.col("gk_id").is_in(tested))


def test_loo_order() -> None:
    """§3.4, one test season per branch, in order: A20's remainder is empty -> NaN and
    ``loo_missing`` (decided before its E2-small flag); T19 is team-selected (``AT_ZERO`` by its
    family alone) -> 0, though its remainder (3 keepers, 60 rows) would be fitted; C19's
    remainder is 2 keepers -> 0 and ``loo_small`` 1 (C19 itself is not small); A19's remainder
    (4 keepers, 110 rows) -> its ``league_shift``. The leave-out gains are recomputed with
    those δ; ``loo_shots`` / ``loo_pairs`` count A20's test rows and pair."""
    gsax = _gsax()
    rated = _census_open()
    flags = {s: _rows(rated, None, s).select(pl.col("small").any(), pl.col("zero_family").any())
             .row(0) for s in ("A19", "A20", "C19", "T19")}  # fmt: skip
    assert flags == {"A19": (False, False), "A20": (True, False), "C19": (False, False),
                     "T19": (False, True)}  # fmt: skip
    assert _remainder(rated, "A20").height == 0
    shift = {s: gsax.league_shift(r["xgot"].to_numpy(), r["is_goal"].to_numpy().astype(float))
             for s in ("A19", "C19", "T19") if (r := _remainder(rated, s)).height}  # fmt: skip
    assert _remainder(rated, "C19")["gk_id"].n_unique() == 2
    t19 = _remainder(rated, "T19")
    assert t19["gk_id"].n_unique() == 3 and t19.height >= gsax.SMALL_LEAGUE.min_shots
    assert abs(shift["C19"]) > 0.05 and abs(shift["T19"]) > 0.05  # preconditions: 0 matters
    ns = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    assert (ns.pairs, ns.left_out) == (len(CENSUS_PAIRS), 1)
    assert (ns.loo_missing, ns.loo_small) == (1, 1)
    a20 = _rows(rated, "sb:g01", "A20").height
    assert (ns.loo_shots, ns.loo_pairs) == (a20, 1) and a20 > 0
    want = _expected(rated, CENSUS_PAIRS,
                     {"A19": shift["A19"], "A20": math.nan, "C19": 0.0, "T19": 0.0})  # fmt: skip
    _same(ns.gain, want["gain"])
    _same(ns.gain_loo, want["gain_loo"])
    _same(ns.gain_none_loo, want["gain_none_loo"])


def test_kept_pairs() -> None:
    """§3.5: ``kept_pairs`` holds (``gk_id``, ``competition_id``, ``season_id``,
    ``next_season_id``) of the pairs the test scores, ``len == pairs``; the left-out pair (sb:g02
    A18 -> A19, 0 goals in A18) is not in it."""
    gsax = _gsax()
    ns = gsax.next_season(_census_open(), by=gsax.KEEPER_SEASON)
    want = {(gk, *_key(t0), _key(t1)[1]) for gk, t0, t1 in CENSUS_PAIRS}
    assert set(ns.kept_pairs) == want and len(ns.kept_pairs) == ns.pairs == len(want)
    assert ("sb:g02", *_key("A18"), _key("A19")[1]) not in set(ns.kept_pairs)
    assert ns.left_out == 1


def test_saw_test() -> None:
    """§3.5 / PM E4: ``saw_test(ns, trained)`` counts the kept pairs whose test season is among
    the training seasons of their rated season. Census: g01 A18 -> A19 is seen; g01 A19 -> A20
    is not (A19's list holds (sb:16, sb:108), A20's season id in another competition); the
    left-out pair g02 A18 -> A19 is not counted. A rated season of a kept pair missing from
    ``trained`` raises (fails closed, count named); no kept pair -> 0."""
    gsax = _gsax()
    ns = gsax.next_season(_census_open(), by=gsax.KEEPER_SEASON)
    trained = gc.census_trained()
    assert gsax.saw_test(ns, trained) == gc.CENSUS_SEEN == 1
    assert (ns.pairs, ns.left_out) == (gc.CENSUS_KEPT, gc.CENSUS_LEFT_OUT)
    no_t18 = trained.filter(pl.col("season_id") != _key("T18")[1])
    with pytest.raises(ValueError, match=r"\b1\b"):
        gsax.saw_test(ns, no_t18)
    no_c18 = trained.filter(~((pl.col("competition_id") == _key("C18")[0])
                              & (pl.col("season_id") == _key("C18")[1])))  # fmt: skip
    with pytest.raises(ValueError):  # the two C18 -> C19 pairs' rated season
        gsax.saw_test(ns, no_c18)
    none = dataclasses.replace(ns, kept_pairs=(), pairs=0)
    assert gsax.saw_test(none, trained.clear()) == 0
