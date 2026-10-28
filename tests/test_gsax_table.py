"""Keeper and league tables, the match-bootstrap CI and the null CI counts (spec
``runs/MS6/gsax`` §4.3, §6).

The hand-worked fixture ``gsax_cases.TABLE_ROWS`` has constant xGOT within each season and
kind, so the rated ``q`` is the season's goal rate and every number below is worked by hand:
A18 open play q = 5/12, B18 open play q = 3/8, B18 penalties q = 2/3, A18 penalties (all
scored) q = goal.
"""

from __future__ import annotations

import math
from typing import Any

import gsax_cases as gc
import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings

from football_shot_models import data, metrics

KEY = ("competition_id", "season_id")
A18, B18 = ("sb:9", "sb:42"), ("sb:72", "sb:30")
KEEPER_COLUMNS = (
    "gk_id", "competition_id", "season_id", "competition_name", "season_name", "family",
    "shots", "goals", "matches", "penalties", "open_goal", "xgot", "gsax_raw", "gsax",
    "open_goal_gsax", "sd", "ci_low", "ci_high",
)  # fmt: skip
LEAGUE_COLUMNS = (
    "competition_id", "season_id", "competition_name", "season_name", "family", "shots", "goals",
    "keepers", "xgot", "gsax_raw", "delta", "delta_penalty", "small",
)  # fmt: skip


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _variant(rated: pl.DataFrame, v: str) -> pl.DataFrame:
    kinds = list(_gsax().VARIANTS[v])
    return rated.filter(pl.col("kind").cast(pl.String).is_in(kinds))


def _ci(rows: pl.DataFrame) -> tuple[float, float] | None:
    """The 90 % match-bootstrap CI of Σ (q − goal), written out from ``group_weights``."""
    m = rows["match_id"].to_numpy()
    if np.unique(m).size < 2:
        return None
    v = rows["q"].to_numpy() - rows["is_goal"].to_numpy().astype(np.float64)
    inv, w = metrics.group_weights(m, metrics.N_BOOT, metrics.SEED)
    sums = w @ np.bincount(inv, weights=v, minlength=w.shape[1])
    lo, hi = np.quantile(sums, [0.05, 0.95])
    return float(lo), float(hi)


def _by_key(table: pl.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (r["gk_id"], r["competition_id"], r["season_id"]): r for r in table.iter_rows(named=True)
    }


# (gk, season) -> shots, goals, matches, penalties, open_goal, xgot, gsax_raw, gsax,
# open_goal_gsax, Σ q(1 − q)
q_a, q_b, q_p = 5 / 12, 3 / 8, 2 / 3
WITHOUT = {
    ("sb:g12", *B18): (6, 1, 2, 0, None, 1.5, 0.5, 6 * q_b - 1, None, 6 * q_b * (1 - q_b)),
    ("sb:g44", *B18): (2, 2, 1, 0, None, 0.5, -1.5, 2 * q_b - 2, None, 2 * q_b * (1 - q_b)),
    ("sb:g07", *A18): (3, 0, 1, 0, 0, 0.9, 0.9, 3 * q_a, 0.0, 3 * q_a * (1 - q_a)),
    ("sb:g12", *A18): (4, 3, 2, 0, 1, 1.2, -1.8, 4 * q_a - 3, q_a - 1, 4 * q_a * (1 - q_a)),
    ("sb:g30", *A18): (5, 2, 2, 0, 1, 1.5, -0.5, 5 * q_a - 2, q_a - 1, 5 * q_a * (1 - q_a)),
}
WITH = {
    ("sb:g12", *B18): (7, 2, 2, 1, None, 2.25, 0.25, 6 * q_b - 1 + q_p - 1, None,
                       6 * q_b * (1 - q_b) + q_p * (1 - q_p)),
    ("sb:g44", *B18): (4, 3, 1, 2, None, 2.0, -1.0, 2 * q_b - 2 + 2 * q_p - 1, None,
                       2 * q_b * (1 - q_b) + 2 * q_p * (1 - q_p)),
    ("sb:g07", *A18): WITHOUT[("sb:g07", *A18)],
    ("sb:g12", *A18): (5, 4, 2, 1, 1, 2.0, -2.0, 4 * q_a - 3, q_a - 1, 4 * q_a * (1 - q_a)),
    ("sb:g30", *A18): (7, 4, 2, 2, 1, 3.1, -0.9, 5 * q_a - 2, q_a - 1, 5 * q_a * (1 - q_a)),
}  # fmt: skip
NAMES = ("shots", "goals", "matches", "penalties", "open_goal", "xgot", "gsax_raw", "gsax",
         "open_goal_gsax")  # fmt: skip


def test_keeper_table_values() -> None:
    """Hand-worked, both variants: one row per keeper-season sorted by (competition_id,
    season_id, gk_id) whatever the row order; ``gsax_raw`` = Σ xgot − goals, ``gsax`` = Σ q −
    goals, ``sd`` = √Σ q(1−q); ``penalties`` 0 on every "without" row; ``open_goal`` and
    ``open_goal_gsax`` counted on the open-goal rows (0 when none), null for a keeper-season
    whose ``open_goal`` is all null (B18). League table: the league sums, δ per kind."""
    gsax = _gsax()
    rated = gsax.rate(gc.table_frame(), small=None)
    for v, expected in (("without penalties", WITHOUT), ("with penalties", WITH)):
        table = gsax.keeper_table(_variant(rated, v))
        assert set(KEEPER_COLUMNS) <= set(table.columns), table.columns
        for c in ("shots", "goals", "matches", "penalties", "open_goal"):
            assert table.schema[c] == pl.Int64, c
        for c in ("xgot", "gsax_raw", "gsax", "open_goal_gsax", "sd", "ci_low", "ci_high"):
            assert table.schema[c] == pl.Float64, c
        assert table.schema["family"] == data.FAMILY
        assert [tuple(r) for r in table.select(*KEY, "gk_id").rows()] == [
            (*B18, "sb:g12"), (*B18, "sb:g44"), (*A18, "sb:g07"), (*A18, "sb:g12"),
            (*A18, "sb:g30"),
        ]  # fmt: skip
        rows = _by_key(table)
        assert set(rows) == set(expected), v
        for k, exp in expected.items():
            r = rows[k]
            for name, want in zip(NAMES, exp[:-1], strict=True):
                if want is None:
                    assert r[name] is None, (v, k, name)
                else:
                    assert r[name] == pytest.approx(want, abs=1e-9), (v, k, name)
            assert r["sd"] == pytest.approx(math.sqrt(exp[-1]), abs=1e-9), (v, k)
            season = "2018/2019" if k[1:] == A18 else "2018"
            assert (r["competition_name"], r["season_name"]) == (
                "Alpha League" if k[1:] == A18 else "Bravo Cup",
                season,
            )
        league = gsax.league_table(_variant(rated, v)).sort(*KEY)
        assert set(LEAGUE_COLUMNS) <= set(league.columns), league.columns
        assert league.schema["keepers"] == pl.Int64 and league.schema["small"] == pl.Boolean
        lt = {tuple(r[:2]): r for r in league.select(*LEAGUE_COLUMNS).iter_rows()}
        for key in (A18, B18):
            sub = [e for k, e in expected.items() if k[1:] == key]
            row = lt[key]
            assert row[5] == sum(e[0] for e in sub) and row[6] == sum(e[1] for e in sub)
            assert row[7] == len(sub)
            assert row[8] == pytest.approx(sum(e[5] for e in sub), abs=1e-9)
            assert row[9] == pytest.approx(sum(e[6] for e in sub), abs=1e-9)
            assert row[12] is False
        logit = lambda p: math.log(p / (1 - p))  # noqa: E731
        assert lt[A18][10] == pytest.approx(logit(q_a) - logit(0.3), abs=1e-9)
        assert lt[B18][10] == pytest.approx(logit(q_b) - logit(0.25), abs=1e-9)
        if v == "without penalties":
            assert lt[A18][11] is None and lt[B18][11] is None
        else:
            assert lt[A18][11] == math.inf
            assert lt[B18][11] == pytest.approx(logit(q_p) - logit(0.75), abs=1e-9)


def test_ci_is_match_bootstrap() -> None:
    """``ci_low`` / ``ci_high`` are the 5th / 95th percentiles of Σ (q − goal) over the
    ``group_weights(match_id, N_BOOT, SEED)`` resamples of the keeper-season's matches, rating
    held fixed; a 1-match keeper-season has a null CI. ``CI_LEVEL`` is 0.90."""
    gsax = _gsax()
    assert gsax.CI_LEVEL == 0.90
    rated = gsax.rate(gc.table_frame(), small=None)
    for v in gsax.VARIANTS:
        rows = _variant(rated, v)
        table = gsax.keeper_table(rows)
        nulls = 0
        for r in table.iter_rows(named=True):
            mine = rows.filter(
                (pl.col("gk_id") == r["gk_id"]) & (pl.col("competition_id") == r["competition_id"])
                & (pl.col("season_id") == r["season_id"])
            )  # fmt: skip
            want = _ci(mine)
            if want is None:
                nulls += 1
                assert r["ci_low"] is None and r["ci_high"] is None, r
            else:
                assert (r["ci_low"], r["ci_high"]) == (
                    pytest.approx(want[0], abs=1e-12), pytest.approx(want[1], abs=1e-12)
                ), r  # fmt: skip
                assert want[0] < want[1]  # precondition: a non-degenerate interval
        assert nulls == 2  # sb:g44 (B18, m08) and sb:g07 (A18, m03)


@settings(max_examples=25, deadline=None)
@given(st.data())
def test_totals_sum_to_league(draw: st.DataObject) -> None:
    """Property, both variants: per comp-season the keepers' ``gsax_raw`` sum to the league's,
    their ``gsax`` to 0 (within ``SUM_TOL``), their shots and goals to the league's and to the
    rows; the table does not depend on row order."""
    gsax = _gsax()
    n = draw.draw(st.integers(min_value=6, max_value=60))
    seasons = draw.draw(st.lists(st.sampled_from(["A18", "B18", "C18", "A19"]), min_size=n,
                                 max_size=n))  # fmt: skip
    gks = draw.draw(st.lists(st.sampled_from(["sb:g1", "sb:g2", "sb:g3", "sb:g4"]), min_size=n,
                             max_size=n))  # fmt: skip
    matches = draw.draw(st.lists(st.sampled_from(["sb:m1", "sb:m2", "sb:m3"]), min_size=n,
                                 max_size=n))  # fmt: skip
    goals = draw.draw(st.lists(st.booleans(), min_size=n, max_size=n))
    p = draw.draw(st.lists(st.floats(min_value=0.01, max_value=0.99), min_size=n, max_size=n))
    kinds = draw.draw(st.lists(st.sampled_from(["open_play", "open_play", "penalty"]),
                               min_size=n, max_size=n))  # fmt: skip
    f = gc.frame(gk=gks, season=seasons, match=matches, goal=goals, xgot=p, kind=kinds)
    rated = gsax.rate(f, small=None)
    for v in gsax.VARIANTS:
        rows = _variant(rated, v)
        kt, lt = gsax.keeper_table(rows), gsax.league_table(rows)
        assert int(kt["shots"].sum()) == rows.height == int(lt["shots"].sum())
        assert int(kt["goals"].sum()) == int(rows["is_goal"].sum()) == int(lt["goals"].sum())
        sums = kt.group_by(*KEY).agg(pl.col("gsax_raw").sum(), pl.col("gsax").sum())
        joined = sums.join(lt.select(*KEY, league="gsax_raw"), on=list(KEY), how="full")
        assert joined.height == lt.height
        for r in joined.iter_rows(named=True):
            assert abs(r["gsax_raw"] - r["league"]) <= gsax.SUM_TOL, r
            assert abs(r["gsax"]) <= gsax.SUM_TOL, r
        perm = draw.draw(st.permutations(list(range(rows.height))))
        shuffled = gsax.keeper_table(rows[perm] if perm else rows)
        assert shuffled.select(*KEY, "gk_id", "shots", "goals").equals(
            kt.select(*KEY, "gk_id", "shots", "goals")
        )
        assert np.allclose(shuffled["gsax"].to_numpy(), kt["gsax"].to_numpy(), atol=1e-12)


def test_null_gk_id_raises() -> None:
    """A null ``gk_id`` raises in ``rate``, ``keeper_table`` and ``league_table`` (count named),
    never filtered."""
    gsax = _gsax()
    f = gc.table_frame()
    bad = f.with_columns(
        gk_id=pl.when(pl.int_range(pl.len()).is_in([0, 5])).then(None).otherwise(pl.col("gk_id"))
    )
    with pytest.raises(ValueError, match=r"\b2\b"):
        gsax.rate(bad, small=None)
    rated = gsax.rate(f, small=None)
    rated_bad = rated.with_columns(gk_id=bad["gk_id"])
    for fn in (gsax.keeper_table, gsax.league_table):
        with pytest.raises(ValueError, match=r"\b2\b"):
            fn(rated_bad)


def test_empty() -> None:
    """An empty frame: ``rate``, ``keeper_table`` and ``league_table`` return empty frames with
    their schema; ``reliability`` returns the three tiers with 0 keepers."""
    gsax = _gsax()
    empty = gc.table_frame().clear()
    rated = gsax.rate(empty, small=None)
    assert rated.height == 0 and {"delta", "q", "small"} <= set(rated.columns)
    assert rated.schema["q"] == pl.Float64 and rated.schema["small"] == pl.Boolean
    kt, lt = gsax.keeper_table(rated), gsax.league_table(rated)
    assert kt.height == 0 and set(KEEPER_COLUMNS) <= set(kt.columns)
    assert lt.height == 0 and set(LEAGUE_COLUMNS) <= set(lt.columns)
    assert kt.schema["gsax"] == pl.Float64 and kt.schema["shots"] == pl.Int64
    tiers = gsax.reliability(rated)
    assert [t.tier for t in tiers] == list(gsax.TIERS) and all(t.keepers == 0 for t in tiers)


class _Rng:
    """A fake ``numpy.random.Generator`` for one seed: its uniform draws are ``u``, so a
    Bernoulli(q) redraw is ``u < q`` whichever method draws it."""

    def __init__(self, u: np.ndarray) -> None:
        self.u = u

    def _take(self, size: Any) -> np.ndarray:
        n = int(np.prod(size)) if size is not None else 1
        assert n == self.u.size, f"one draw per rated row expected, got size {size}"
        return self.u.copy()

    def random(self, size: Any = None, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._take(size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: Any = None) -> np.ndarray:
        assert (low, high) == (0.0, 1.0)
        return self._take(size)

    def binomial(self, n: Any, p: Any, size: Any = None) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        assert np.all(np.asarray(n) == 1)
        return (self._take(p.size if size is None else size) < p).astype(np.int64)


def _ci_frame() -> pl.DataFrame:
    """6 keepers: 4 with >= 100 shots over 4-5 matches (two conceding far fewer than q, one
    far more, one level), 2 with 30 shots over 3 matches; one season, xGOT varied."""
    parts = [
        gc.block("sb:k1", "A18", goals=gc.pattern(120, 8), xgot=gc.spread(120, 0.2, 0.6),
                 matches=["sb:m1", "sb:m2", "sb:m3", "sb:m4"]),
        gc.block("sb:k2", "A18", goals=gc.pattern(110, 7), xgot=gc.spread(110, 0.2, 0.6),
                 matches=["sb:m5", "sb:m6", "sb:m7", "sb:m8", "sb:m9"]),
        gc.block("sb:k3", "A18", goals=gc.pattern(100, 2), xgot=gc.spread(100, 0.2, 0.5),
                 matches=["sb:n1", "sb:n2", "sb:n3", "sb:n4"]),
        gc.block("sb:k4", "A18", goals=gc.pattern(105, 3), xgot=gc.spread(105, 0.2, 0.5),
                 matches=["sb:n5", "sb:n6", "sb:n7", "sb:n8"]),
        gc.block("sb:k5", "A18", goals=gc.pattern(30, 2), xgot=gc.spread(30, 0.1, 0.3),
                 matches=["sb:o1", "sb:o2", "sb:o3"]),
        gc.block("sb:k6", "A18", goals=gc.pattern(30, 9), xgot=gc.spread(30, 0.4, 0.7),
                 matches=["sb:o4", "sb:o5", "sb:o6"]),
    ]  # fmt: skip
    return gc.blocks(*parts)


def _counts(rated: pl.DataFrame, min_shots: int) -> tuple[int, int, int, int]:
    """(table rows, of them CI excluding 0, >= 2-match rows, of them CI excluding 0)."""
    out = [0, 0, 0, 0]
    for gk in rated["gk_id"].unique().to_list():
        rows = rated.filter(pl.col("gk_id") == gk)
        ci = _ci(rows)
        excl = ci is not None and (ci[0] > 0 or ci[1] < 0)
        if rows.height >= min_shots:
            out[0] += 1
            out[1] += excl
        if rows["match_id"].n_unique() >= 2:
            out[2] += 1
            out[3] += excl
    return out[0], out[1], out[2], out[3]


def test_ci_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ci_counts(rated, *, seed)``: observed counts from the keeper table; null counts after
    one redraw of the goals as Bernoulli(q) from ``default_rng(seed)`` (a fake here, whose
    uniform draws are fixed), rating held fixed. Both recomputed by hand."""
    gsax = _gsax()
    rated = gsax.rate(_ci_frame(), small=None)
    q = rated["q"].to_numpy()
    u = np.array([(i * 0.6180339887) % 1.0 for i in range(rated.height)])
    seed = 987_654
    real = np.random.default_rng

    def fake(s: Any = None, *args: Any, **kwargs: Any) -> Any:
        return _Rng(u) if s == seed else real(s, *args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", fake)
    if hasattr(gsax, "default_rng"):  # imported by name
        monkeypatch.setattr(gsax, "default_rng", fake)
    got = gsax.ci_counts(rated, seed=seed)
    obs = _counts(rated, gsax.TABLE_MIN_SHOTS)
    null = _counts(rated.with_columns(is_goal=pl.Series(u < q)), gsax.TABLE_MIN_SHOTS)
    assert (got.table_rows, got.table_observed, got.multi_rows, got.multi_observed) == obs
    assert (got.table_rows, got.table_null, got.multi_rows, got.multi_null) == null
    assert obs[0] == 4 and obs[2] == 6 and obs[1] >= 2  # precondition: the fixture separates
    assert null[1] != obs[1] or null[3] != obs[3]  # precondition: null differs from observed
