"""League-sum rating: ``gsax.league_shift`` and ``gsax.rate`` (spec ``runs/MS6/gsax`` §4.2, §6).

GSAx is rated against the season's league sum: one logit shift δ per (competition-season, kind)
makes the rated xGOT ``q`` sum to the goals, so the xGOT level gap is not charged to keepers.
"With penalties" is open play at δ_open plus penalties at δ_pen, so the variants differ by
exactly the keeper's penalty GSAx. PM ruling E2 (b): ``SMALL_LEAGUE = SmallLeague(2, 50)`` rates
comp-seasons with at most 2 keepers or fewer than 50 open-play shots at δ = 0 and flags them.
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
from scipy.special import expit

KEY = ("competition_id", "season_id")


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _sum_by(rated: pl.DataFrame, by: tuple[str, ...]) -> dict[tuple[Any, ...], float]:
    out: dict[tuple[Any, ...], float] = {}
    for row in rated.select(*by, "q", "is_goal").iter_rows():
        k = tuple(str(v) for v in row[: len(by)])
        out[k] = out.get(k, 0.0) + row[-2] - float(row[-1])
    return out


def test_league_shift_values() -> None:
    """p (0.2, 0.2) with goals (1, 0) -> δ = log 4; no goal -> −inf; every row a goal -> +inf;
    1 goal in 10⁴ rows at the logit clip (δ ≈ −43.7, outside ±40) and its mirror bracket; an
    empty input raises."""
    gsax = _gsax()
    d = gsax.league_shift(np.array([0.2, 0.2]), np.array([1.0, 0.0]))
    assert d == pytest.approx(math.log(4.0), abs=1e-10)
    assert gsax.league_shift(np.array([0.3, 0.6]), np.array([0.0, 0.0])) == -math.inf
    assert gsax.league_shift(np.array([0.3, 0.6]), np.array([1.0, 1.0])) == math.inf
    n = 10_000
    one = np.zeros(n)
    one[17] = 1.0
    d = gsax.league_shift(np.ones(n), one)
    assert d < -40.0
    assert float(expit(gc.logit(np.ones(n)) + d).sum()) == pytest.approx(1.0, abs=1e-8)
    d = gsax.league_shift(np.zeros(n), 1.0 - one)
    assert d > 40.0
    assert float(expit(gc.logit(np.zeros(n)) + d).sum()) == pytest.approx(n - 1.0, abs=1e-8)
    with pytest.raises(ValueError):
        gsax.league_shift(np.array([]), np.array([]))


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(min_value=0.0, max_value=1.0), min_size=2, max_size=40),
    st.data(),
)
def test_league_shift_property(p: list[float], draw: st.DataObject) -> None:
    """Property: with 0 < goals < rows, Σ expit(logit p + δ) = goals (p at 0 and 1 included)."""
    goals = draw.draw(st.lists(st.booleans(), min_size=len(p), max_size=len(p)))
    y = np.array(goals, dtype=np.float64)
    if y.sum() in (0.0, float(y.size)):
        y[0] = 1.0 - y[0]
    if y.sum() in (0.0, float(y.size)):
        return
    d = _gsax().league_shift(np.array(p), y)
    assert math.isfinite(d)
    assert float(expit(gc.logit(np.array(p)) + d).sum()) == pytest.approx(y.sum(), abs=1e-8)


def _sums_frame() -> pl.DataFrame:
    """4 comp-seasons of 30, 18, 12, 21 open-play rows (A18 and A19 share a competition, B18
    and C18 a season id; their goal rates differ), penalties in A18 and B18; varied xGOT."""
    parts = [
        gc.block("sb:g01", "A18", goals=gc.pattern(30, 3), xgot=gc.spread(30, 0.05, 0.4),
                 matches=["sb:a1", "sb:a2", "sb:a3"]),
        gc.block("sb:g02", "B18", goals=gc.pattern(18, 2), xgot=gc.spread(18, 0.1, 0.6),
                 matches=["sb:b1", "sb:b2"]),
        gc.block("sb:g03", "C18", goals=gc.pattern(12, 6), xgot=gc.spread(12, 0.2, 0.8),
                 matches=["sb:c1", "sb:c2"]),
        gc.block("sb:g01", "A19", goals=gc.pattern(21, 4, 1), xgot=gc.spread(21, 0.02, 0.97),
                 matches=["sb:d1", "sb:d2", "sb:d3"]),
        gc.block("sb:g01", "A18", goals=[True, False, True, True], xgot=[0.7, 0.8, 0.75, 0.9],
                 matches=["sb:a1", "sb:a2"], kind="penalty"),
        gc.block("sb:g02", "B18", goals=[True, False, False], xgot=[0.78, 0.7, 0.8],
                 matches=["sb:b1"], kind="penalty"),
    ]  # fmt: skip
    f = gc.blocks(*parts)
    return f.sort("shot_id")  # shot ids run backwards: rows leave key order


def test_rate_sums_to_zero() -> None:
    """Per (comp-season, kind), Σ (q − goal) is 0 within ``SUM_TOL``; ``delta`` is one value
    per group and ``q = expit(logit xgot + delta)`` row by row; seasons sharing a competition
    or a season id are separate groups; the penalty δ is not the open-play δ."""
    gsax = _gsax()
    f = _sums_frame()
    rated = gsax.rate(f, small=None)
    assert rated.height == f.height
    assert rated.schema["delta"] == pl.Float64 and rated.schema["q"] == pl.Float64
    assert rated.schema["small"] == pl.Boolean
    sums = _sum_by(rated, (*KEY, "kind"))
    assert len(sums) == 6
    for k, s in sums.items():
        assert abs(s) <= gsax.SUM_TOL, k
    for a, b in ((("sb:72", "sb:30"), ("sb:16", "sb:30")), (("sb:9", "sb:42"), ("sb:9", "sb:90"))):
        # precondition: one δ over the near-miss pair would leave each part off zero
        rows = [f.filter((pl.col("competition_id") == k[0]) & (pl.col("season_id") == k[1])
                         & (pl.col("kind") == "open_play")) for k in (a, b)]  # fmt: skip
        both = pl.concat(rows)
        d = gsax.league_shift(both["xgot"].to_numpy(), both["is_goal"].to_numpy().astype(float))
        part = expit(gc.logit(rows[0]["xgot"].to_numpy()) + d).sum() - rows[0]["is_goal"].sum()
        assert abs(float(part)) > 0.1, (a, b)
    deltas = rated.group_by(*KEY, "kind").agg(pl.col("delta").n_unique().alias("n"),
                                              pl.col("delta").first().alias("d"))  # fmt: skip
    assert (deltas["n"] == 1).all()
    a18 = deltas.filter(pl.col("season_id") == "sb:42")
    d_open = a18.filter(pl.col("kind") == "open_play")["d"].item()
    d_pen = a18.filter(pl.col("kind") == "penalty")["d"].item()
    assert abs(d_open - d_pen) > 0.1
    q = expit(gc.logit(rated["xgot"].to_numpy()) + rated["delta"].to_numpy())
    assert np.allclose(rated["q"].to_numpy(), q, rtol=0.0, atol=1e-12)


def test_rate_variants_additive() -> None:
    """``VARIANTS`` filters the rated rows: a keeper with no penalty has equal GSAx in both
    variants; GSAx(with) − GSAx(without) is his penalty rows' Σ (q − goal); a season whose kicks
    were all scored (δ = +inf, q = goal) gives every keeper 0 penalty GSAx."""
    gsax = _gsax()
    assert gsax.RATED_KINDS == ("open_play", "penalty")
    assert dict(gsax.VARIANTS) == {
        "without penalties": ("open_play",), "with penalties": ("open_play", "penalty"),
    }  # fmt: skip
    rated = gsax.rate(gc.table_frame(), small=None)
    tables = {
        v: gsax.keeper_table(rated.filter(pl.col("kind").cast(pl.String).is_in(list(kinds))))
        for v, kinds in gsax.VARIANTS.items()
    }
    key = ("gk_id", *KEY)
    gs = {v: {r[:3]: r[3] for r in t.select(*key, "gsax").iter_rows()} for v, t in tables.items()}
    pen = _sum_by(rated.filter(pl.col("kind") == "penalty"), key)
    assert set(gs["with penalties"]) == set(gs["without penalties"])
    for k, without in gs["without penalties"].items():
        diff = gs["with penalties"][k] - without
        assert diff == pytest.approx(pen.get(k, 0.0), abs=1e-9), k
    assert gs["with penalties"][("sb:g07", "sb:9", "sb:42")] == pytest.approx(
        gs["without penalties"][("sb:g07", "sb:9", "sb:42")], abs=1e-12
    )
    a18_pen = rated.filter((pl.col("season_id") == "sb:42") & (pl.col("kind") == "penalty"))
    assert (a18_pen["delta"] == math.inf).all()
    assert a18_pen["q"].to_list() == [1.0] * a18_pen.height  # q = goal: all scored
    for gk in ("sb:g30", "sb:g12"):
        assert pen[(gk, "sb:9", "sb:42")] == 0.0
    assert pen[("sb:g44", "sb:72", "sb:30")] == pytest.approx(1 / 3, abs=1e-9)
    assert pen[("sb:g12", "sb:72", "sb:30")] == pytest.approx(-1 / 3, abs=1e-9)


def _small_frame() -> pl.DataFrame:
    """F1: a 2-keeper final, 1 match, 20 open-play shots + 2 kicks. A18: 3 keepers and exactly
    50 open-play shots (not small). B18: 3 keepers and 49 open-play shots plus 3 kicks (small:
    kicks do not count). C18: 4 keepers, 60 shots."""
    parts = [
        gc.block("sb:gf1", "F1", goals=gc.pattern(10, 10), xgot=0.3, matches=["sb:f1"]),
        gc.block("sb:gf2", "F1", goals=gc.pattern(10, 10), xgot=0.3, matches=["sb:f1"]),
        gc.block("sb:gf1", "F1", goals=[True, True], xgot=0.7, matches=["sb:f1"], kind="penalty"),
        *(gc.block(f"sb:ga{i}", "A18", goals=gc.pattern(n, 3, i), xgot=gc.spread(n, 0.1, 0.6),
                   matches=[f"sb:a{i}", f"sb:a{i + 3}"])
          for i, n in enumerate((17, 17, 16))),
        *(gc.block(f"sb:gb{i}", "B18", goals=gc.pattern(n, 4, i), xgot=gc.spread(n, 0.1, 0.6),
                   matches=[f"sb:b{i}", f"sb:b{i + 3}"])
          for i, n in enumerate((17, 16, 16))),
        gc.block("sb:gb0", "B18", goals=[True, False, True], xgot=0.75, matches=["sb:b0"],
                 kind="penalty"),
        *(gc.block(f"sb:gc{i}", "C18", goals=gc.pattern(15, 2, i), xgot=gc.spread(15, 0.1, 0.6),
                   matches=[f"sb:c{i}"])
          for i in range(4)),
    ]  # fmt: skip
    return gc.blocks(*parts)


def test_rate_small_league() -> None:
    """``small=None`` flags nothing and fits every δ. ``SmallLeague(2, 50)`` (``SMALL_LEAGUE``,
    PM E2 (b)) rates the 2-keeper final and the 49-shot season at δ = 0 in both kinds (q =
    xgot) and flags them; 3 keepers with exactly 50 shots, and 4 keepers, are rated."""
    gsax = _gsax()
    assert gsax.SmallLeague(2, 50) == gsax.SMALL_LEAGUE
    f = _small_frame()
    free = gsax.rate(f, small=None)
    assert not free["small"].any()
    f1_open = free.filter((pl.col("season_id") == "sb:43") & (pl.col("kind") == "open_play"))
    assert f1_open["delta"][0] == pytest.approx(
        float(gc.logit(np.array([0.1]))[0] - gc.logit(np.array([0.3]))[0]), abs=1e-9
    )
    rated = gsax.rate(f, small=gsax.SmallLeague(2, 50))
    per_key = {
        r[:2]: r[2:] for r in rated.group_by(*KEY).agg(
            pl.col("small").all().alias("a"), pl.col("small").any().alias("b"),
            (pl.col("delta") == 0.0).all().alias("c"),
        ).iter_rows()
    }  # fmt: skip
    assert per_key[("sb:55", "sb:43")] == (True, True, True)  # F1: 2 keepers
    assert per_key[("sb:72", "sb:30")] == (True, True, True)  # B18: 49 open-play shots
    assert per_key[("sb:9", "sb:42")] == (False, False, False)  # A18: 3 keepers, 50 shots
    assert per_key[("sb:16", "sb:30")] == (False, False, False)  # C18
    small = rated.filter(pl.col("small"))
    assert np.allclose(small["q"].to_numpy(), small["xgot"].to_numpy(), rtol=0.0, atol=1e-12)
    assert set(small["kind"].cast(pl.String).to_list()) == {"open_play", "penalty"}
    for k, s in _sum_by(rated.filter(~pl.col("small")), (*KEY, "kind")).items():
        assert abs(s) <= gsax.SUM_TOL, k


def _bad(f: pl.DataFrame, col: str, value: Any) -> pl.DataFrame:
    """``f`` with ``value`` in ``col`` on rows 3, 7 and 11 (three bad rows)."""
    hit = pl.int_range(pl.len()).is_in([3, 7, 11])
    return f.with_columns(
        pl.when(hit).then(pl.lit(value, dtype=f.schema[col])).otherwise(pl.col(col)).alias(col)
    )


@pytest.mark.parametrize(
    ("col", "value"),
    [
        ("is_goal", None), ("xgot", None), ("xgot", 1.5), ("xgot", -0.01), ("xgot", math.nan),
        ("gk_id", None), ("match_id", None), ("competition_id", None), ("season_id", None),
        ("competition_name", None), ("season_name", None), ("kind", "shootout"),
    ],
)  # fmt: skip
def test_rate_errors(col: str, value: Any) -> None:
    """A null ``gk_id``, ``xgot``, ``is_goal``, ``match_id``, key or name, a NaN or
    out-of-[0, 1] ``xgot`` and a kind outside ``RATED_KINDS`` raise ``ValueError`` naming the
    count (3 rows), never filtered."""
    f = _bad(gc.table_frame(), col, value)
    with pytest.raises(ValueError, match=r"\b3\b"):
        _gsax().rate(f, small=None)


def test_rate_small_two_keepers_many_shots() -> None:
    """PM E2 (b), the keeper clause on its own (run6-ms6 QA 4): a 2-keeper competition-season
    with 60 open-play shots (at least 50) is flagged small and rated at δ = 0 in both kinds; a
    rule allowing 1 keeper rates it, so the flag comes from the keeper count."""
    gsax = _gsax()
    f = gc.blocks(
        gc.block("sb:gf1", "F1", goals=gc.pattern(30, 5), xgot=0.3, matches=["sb:f1", "sb:f2"]),
        gc.block("sb:gf2", "F1", goals=gc.pattern(30, 3), xgot=0.3, matches=["sb:f1", "sb:f2"]),
        gc.block("sb:gf1", "F1", goals=[True, False], xgot=0.7, matches=["sb:f1"], kind="penalty"),
    )
    open_play = f.filter(pl.col("kind") == "open_play")
    assert open_play.height >= gsax.SMALL_LEAGUE.min_shots  # precondition: the shots clause is off
    assert open_play["gk_id"].n_unique() == gsax.SMALL_LEAGUE.max_keepers == 2
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE)
    assert rated["small"].all() and (rated["delta"] == 0.0).all()
    assert set(rated["kind"].cast(pl.String).to_list()) == {"open_play", "penalty"}
    assert np.allclose(rated["q"].to_numpy(), rated["xgot"].to_numpy(), rtol=0.0, atol=1e-12)
    one = gsax.rate(f, small=gsax.SmallLeague(1, gsax.SMALL_LEAGUE.min_shots))
    assert not one["small"].any() and (one["delta"] != 0.0).all()


def test_guards_on_hostile_primitives() -> None:
    """Run6-ms6 QA 11: ``league_shift`` refuses a goal outside {0, 1} (2, −1, 0.5, NaN) with its
    own message; ``keeper_modes`` refuses τ that is 0, negative, NaN or infinite."""
    gsax = _gsax()
    for goals in ([2.0, 0.0], [-1.0, 1.0], [0.5, 0.0], [math.nan, 1.0]):
        with pytest.raises(ValueError, match="goal is not 0 / 1"):
            gsax.league_shift(np.array([0.3, 0.6]), np.array(goals))
    off, y, grp = np.zeros(4), np.array([1.0, 0.0, 1.0, 0.0]), np.array([0, 0, 1, 1])
    for tau in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="tau"):
            gsax.keeper_modes(off, y, grp, tau=tau)
    assert gsax.keeper_modes(off, y, grp, tau=1.0)[0].shape == (2,)
