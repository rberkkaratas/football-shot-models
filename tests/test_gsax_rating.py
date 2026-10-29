"""League-sum rating: ``gsax.league_shift``, ``gsax.rate``, ``gsax.too_small`` and
``gsax.rating_groups`` (spec ``runs/MS6/gsax`` §4.2, §6; ``runs/PR1/keeper-fixes`` §3.1, §3.2).

GSAx is rated against the league sum: one logit shift δ per rating group makes the rated xGOT
``q`` sum to the goals, so the xGOT level gap is not charged to keepers. Since PR1 (PM
2026-10-28 (a)) the open-play group is the competition-season and the penalty group the
season-out ``source`` (one penalty δ per (provider, source)). "With penalties" is open play at
δ_open plus penalties at δ_pen, so the variants differ by exactly the keeper's penalty GSAx. PM
E2 (b): ``SMALL_LEAGUE = SmallLeague(2, 50)`` rates comp-seasons with at most 2 keepers or fewer
than 50 open-play shots at δ = 0 and flags them; the minimum holds for every grouping and kind
(a penalty pool too, ``pool_small``); team-selected comp-seasons (``ZERO_FAMILIES``) are rated at
δ = 0 in both kinds and flagged ``zero_family`` (PM 2026-10-28).
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

from football_shot_models import data

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
    """Per rating group, Σ (q − goal) is 0 within ``SUM_TOL`` (PR1 §3.1): open play per
    comp-season, penalties per season-out source (one pool here: A18 and B18 kicks, both
    ``season_folds``), so each season's kicks alone need not sum to 0; ``delta`` is one value
    per group and ``q = expit(logit xgot + delta)`` row by row; seasons sharing a competition
    or a season id are separate open-play groups; the penalty δ is not the open-play δ."""
    gsax = _gsax()
    f = _sums_frame()
    rated = gsax.rate(f, small=None, zero_families=())
    assert rated.height == f.height
    assert rated.schema["delta"] == pl.Float64 and rated.schema["q"] == pl.Float64
    assert rated.schema["small"] == pl.Boolean
    open_ = rated.filter(pl.col("kind") == "open_play")
    sums = _sum_by(open_, KEY)
    assert len(sums) == 4
    for k, s in sums.items():
        assert abs(s) <= gsax.SUM_TOL, k
    pens = rated.filter(pl.col("kind") == "penalty")
    assert abs(sum(_sum_by(pens, ("source",)).values())) <= gsax.SUM_TOL
    per_season = _sum_by(pens, KEY)
    assert len(per_season) == 2 and all(abs(v) > 0.01 for v in per_season.values())  # pooled
    for a, b in ((("sb:72", "sb:30"), ("sb:16", "sb:30")), (("sb:9", "sb:42"), ("sb:9", "sb:90"))):
        # precondition: one δ over the near-miss pair would leave each part off zero
        rows = [f.filter((pl.col("competition_id") == k[0]) & (pl.col("season_id") == k[1])
                         & (pl.col("kind") == "open_play")) for k in (a, b)]  # fmt: skip
        both = pl.concat(rows)
        d = gsax.league_shift(both["xgot"].to_numpy(), both["is_goal"].to_numpy().astype(float))
        part = expit(gc.logit(rows[0]["xgot"].to_numpy()) + d).sum() - rows[0]["is_goal"].sum()
        assert abs(float(part)) > 0.1, (a, b)
    deltas = rated.group_by("kind", "group").agg(pl.col("delta").n_unique().alias("n"),
                                                 pl.col("delta").first().alias("d"))  # fmt: skip
    assert (deltas["n"] == 1).all()
    assert pens["delta"].n_unique() == 1 and set(pens["group"].to_list()) == {"season_folds"}
    a18 = rated.filter(pl.col("season_id") == "sb:42")
    d_open = a18.filter(pl.col("kind") == "open_play")["delta"][0]
    assert abs(d_open - pens["delta"][0]) > 0.1
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
    rated = gsax.rate(gc.table_frame(), small=None, zero_families=())
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
    free = gsax.rate(f, small=None, zero_families=())
    assert not free["small"].any()
    f1_open = free.filter((pl.col("season_id") == "sb:43") & (pl.col("kind") == "open_play"))
    assert f1_open["delta"][0] == pytest.approx(
        float(gc.logit(np.array([0.1]))[0] - gc.logit(np.array([0.3]))[0]), abs=1e-9
    )
    rated = gsax.rate(f, small=gsax.SmallLeague(2, 50), zero_families=())
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
        ("family", None), ("source", None), ("competition_id", "sb:9 x"), ("season_id", "sb: 42"),
    ],
)  # fmt: skip
def test_rate_errors(col: str, value: Any) -> None:
    """A null ``gk_id``, ``xgot``, ``is_goal``, ``match_id``, key or name, a NaN or
    out-of-[0, 1] ``xgot`` and a kind outside ``RATED_KINDS`` raise ``ValueError`` naming the
    count (3 rows), never filtered; PR1 §3.1 / §7: a null ``family`` or ``source`` and an id
    containing a space (the ``group`` label must be unique) too."""
    f = _bad(gc.table_frame(), col, value)
    with pytest.raises(ValueError, match=r"\b3\b"):
        _gsax().rate(f, small=None, zero_families=())


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
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=())
    assert rated["small"].all() and (rated["delta"] == 0.0).all()
    assert set(rated["kind"].cast(pl.String).to_list()) == {"open_play", "penalty"}
    assert np.allclose(rated["q"].to_numpy(), rated["xgot"].to_numpy(), rtol=0.0, atol=1e-12)
    one = gsax.rate(f, small=gsax.SmallLeague(1, gsax.SMALL_LEAGUE.min_shots), zero_families=())
    assert not one["small"].any()
    assert (one.filter(pl.col("kind") == "open_play")["delta"] != 0.0).all()
    kicks = one.filter(pl.col("kind") == "penalty")  # PR1: 2 kicks, 1 keeper: an E2-small pool
    assert kicks["pool_small"].all() and (kicks["delta"] == 0.0).all()


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


# ---- PR1 keeper-fixes §3.1, §3.2: rating groups, pools, team-selected, E2 everywhere ----------

OUT_COLUMNS = ("group", "delta", "q", "small", "zero_family", "pool_small")


def _season(season: str, keepers: int, shots: int, *, tag: str = "", every: int = 3,
            lo: float = 0.1, hi: float = 0.6) -> list[dict[str, list[Any]]]:  # fmt: skip
    """``keepers`` keepers of ``season`` with ``shots`` open-play shots in all, split as evenly
    as possible (unequal blocks), 2 matches each."""
    sizes = [shots // keepers + (i < shots % keepers) for i in range(keepers)]
    return [
        gc.block(f"sb:g{tag}{season.lower()}{i}", season, goals=gc.pattern(n, every, i),
                 xgot=gc.spread(n, lo, hi), matches=[f"sb:m{season}{i}a", f"sb:m{season}{i}b"])
        for i, n in enumerate(sizes)
    ]  # fmt: skip


def _kicks(season: str, keepers: int, kicks: int, *, every: int = 4) -> list[dict[str, list[Any]]]:
    sizes = [kicks // keepers + (i < kicks % keepers) for i in range(keepers)]
    return [
        gc.block(f"sb:k{season.lower()}{i}", season, goals=[not v for v in gc.pattern(n, every, i)],
                 xgot=gc.spread(n, 0.68, 0.84), matches=[f"sb:m{season}{i}a"], kind="penalty")
        for i, n in enumerate(sizes)
    ]  # fmt: skip


def _framed(parts: list[dict[str, list[Any]]], family: dict[str, str] | None = None,
            source: dict[str, str] | None = None) -> pl.DataFrame:  # fmt: skip
    seasons = [x for part in parts for x in part["season"]]
    return gc.blocks(
        *parts,
        family=[(family or {}).get(x, gc.LEAGUE) for x in seasons],
        source=[(source or {}).get(x, "season_folds") for x in seasons],
    )


def _label(season: str) -> str:
    return f"{gc.SEASONS[season][0]} {gc.SEASONS[season][1]}"


def _shift(rows: pl.DataFrame) -> float:
    return float(_gsax().league_shift(rows["xgot"].to_numpy(),
                                      rows["is_goal"].to_numpy().astype(float)))  # fmt: skip


def test_rate_columns_and_empty() -> None:
    """§3.1 output: ``pred``'s columns in order, then ``group`` (String, none null), ``delta``,
    ``q`` (Float64), ``small``, ``zero_family``, ``pool_small`` (Boolean); empty in, empty out
    with those columns."""
    gsax = _gsax()
    f = gc.table_frame()
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    assert rated.columns == [*f.columns, *OUT_COLUMNS]
    want = (pl.String, pl.Float64, pl.Float64, pl.Boolean, pl.Boolean, pl.Boolean)
    assert [rated.schema[c] for c in OUT_COLUMNS] == list(want)
    assert rated["group"].null_count() == 0
    empty = gsax.rate(f.clear(), small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    assert empty.height == 0 and empty.columns == rated.columns
    assert [empty.schema[c] for c in OUT_COLUMNS] == list(want)
    assert gsax.ZERO_FAMILIES == (gsax.TEAM_SELECTED,) and gsax.TEAM_SELECTED in data.FAMILIES


def test_penalty_pool_per_source() -> None:
    """PM 2026-10-28 (a): one penalty δ per source. Kicks of 3 ``season_folds`` seasons and 3
    ``holdout_fit`` seasons (unequal pools, rows out of key order): each source has one δ, the
    ``league_shift`` of its own kicks (recomputed here), Σ (q − goal) = 0 per pool and not per
    season; penalty rows carry the source as ``group``, open-play rows "<competition_id>
    <season_id>"; every open-play season keeps its own δ."""
    gsax = _gsax()
    folds, held = ("A18", "B18", "C18"), ("A19", "B19", "A20")
    parts = [p for s in (*folds, *held) for p in _season(s, 3, 60)]
    parts += [p for s, n in zip((*folds, *held), (22, 17, 19, 15, 21, 18), strict=True)
              for p in _kicks(s, 3, n)]  # fmt: skip
    f = _framed(parts, source=dict.fromkeys(held, "holdout_fit")).sort("shot_id")
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    pens = rated.filter(pl.col("kind") == "penalty")
    assert not rated["pool_small"].any() and not rated["small"].any()
    for src in ("season_folds", "holdout_fit"):
        mine = pens.filter(pl.col("source") == src)
        assert mine["delta"].n_unique() == 1 and set(mine["group"].to_list()) == {src}
        assert mine["delta"][0] == pytest.approx(_shift(mine), abs=1e-9)
        assert abs(float((mine["q"] - mine["is_goal"].cast(pl.Float64)).sum())) <= gsax.SUM_TOL
    per_season = _sum_by(pens, KEY)
    assert len(per_season) == 6 and max(abs(v) for v in per_season.values()) > 0.05
    assert pens.filter(pl.col("source") == "season_folds")["delta"][0] != pytest.approx(
        pens.filter(pl.col("source") == "holdout_fit")["delta"][0], abs=1e-6
    )  # precondition: the two pools differ
    open_ = rated.filter(pl.col("kind") == "open_play")
    labels = open_.select("group", *KEY).unique()
    assert all(g == f"{c} {s}" for g, c, s in labels.rows()) and labels.height == 6
    for k, v in _sum_by(open_, KEY).items():
        assert abs(v) <= gsax.SUM_TOL, k


def test_zero_season_out_of_pool() -> None:
    """§3.1 rule 1 and Decisions: the kicks of a small or team-selected season are rated at δ = 0
    and stay out of their source's pool (its δ is the shift of the other seasons' kicks only);
    their ``group`` is the season label and ``pool_small`` is False on them."""
    gsax = _gsax()
    parts = [*_season("A18", 3, 60), *_season("B18", 4, 60), *_season("T18", 3, 60),
             *_season("F1", 2, 20), *_kicks("A18", 3, 30), *_kicks("B18", 3, 30),
             *_kicks("T18", 2, 6, every=2), *_kicks("F1", 2, 4, every=2)]  # fmt: skip
    f = _framed(parts, family={"T18": gc.TEAM})
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    pens = rated.filter(pl.col("kind") == "penalty")
    zero = pens.filter(pl.col("season_id").is_in(["sb:3", "sb:43"]))
    pool = pens.filter(~pl.col("season_id").is_in(["sb:3", "sb:43"]))
    assert zero.height == 10 and pool.height == 60  # precondition
    assert (zero["delta"] == 0.0).all() and np.allclose(zero["q"], zero["xgot"], atol=1e-12)
    assert not zero["pool_small"].any()
    assert set(zero["group"].to_list()) == {_label("T18"), _label("F1")}
    assert pool["delta"][0] == pytest.approx(_shift(pool), abs=1e-9)
    assert abs(_shift(pens) - _shift(pool)) > 0.01  # precondition: pooling them would differ
    assert zero.filter(pl.col("season_id") == "sb:3")["zero_family"].all()
    assert zero.filter(pl.col("season_id") == "sb:43")["small"].all()


@pytest.mark.parametrize(
    ("keepers", "kicks", "pool_small"), [(2, 60, True), (3, 49, True), (3, 50, False)]
)
def test_penalty_pool_small(keepers: int, kicks: int, pool_small: bool) -> None:
    """E2's minimum on the penalty grouping (PM 2026-10-28 (a)): a pool with 2 keepers and >= 50
    kicks, or 3 keepers and 49 kicks, is rated at δ = 0 with ``pool_small``; 3 keepers and
    exactly 50 kicks is fitted. ``small`` (the season's open-play flag) stays False on those
    rows and the open-play δ is fitted."""
    gsax = _gsax()
    sizes = [kicks // 3 + (i < kicks % 3) for i in range(3)]  # 3 seasons, unequal shares
    parts = [p for s in ("A18", "B18", "C18") for p in _season(s, 3, 60)]
    gks = [f"sb:k{i % keepers}" for i in range(3)]
    for s, n, gk in zip(("A18", "B18", "C18"), sizes, gks, strict=True):
        b = _kicks(s, 1, n)[0]
        parts.append({**b, "gk": [gk] * n})
    f = _framed(parts)
    pens = f.filter(pl.col("kind") == "penalty")
    assert (pens.height, pens["gk_id"].n_unique()) == (kicks, keepers)  # precondition
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    r = rated.filter(pl.col("kind") == "penalty")
    assert (r["pool_small"] == pool_small).all() and not r["small"].any()
    assert not rated.filter(pl.col("kind") == "open_play")["pool_small"].any()
    if pool_small:
        assert (r["delta"] == 0.0).all() and np.allclose(r["q"], r["xgot"], atol=1e-12)
    else:
        assert r["delta"][0] == pytest.approx(_shift(r), abs=1e-9) and r["delta"][0] != 0.0
    assert (rated.filter(pl.col("kind") == "open_play")["delta"] != 0.0).all()


@pytest.mark.parametrize(
    ("units", "rows", "small", "want"),
    [
        (3, 50, "rule", False), (2, 50, "rule", True), (3, 49, "rule", True),
        (2, 49, "rule", True), (100, 10_000, "rule", False), (3, 0, "rule", True),
        (0, 0, "rule", True), (2, 10, None, False), (0, 0, None, False),
    ],
)  # fmt: skip
def test_too_small_table(units: int, rows: int, small: str | None, want: bool) -> None:
    """``too_small(units, rows, small)``: the one expression of E2 (``units <=
    small.max_keepers`` or ``rows < small.min_shots``), each clause failing alone and both at
    their exact boundary; ``rows = 0`` is small; ``small=None`` is never small."""
    gsax = _gsax()
    rule = gsax.SmallLeague(2, 50) if small else None
    assert gsax.too_small(units, rows, rule) is want


def test_zero_family() -> None:
    """PM 2026-10-28: a team-selected comp-season (3 keepers, 60 shots: not E2-small) is rated at
    δ = 0 in both kinds and flagged ``zero_family`` (``small`` False); q = xgot. With
    ``zero_families=()`` the same season is fitted (δ != 0): the flag comes from the family."""
    gsax = _gsax()
    parts = [*_season("T18", 3, 60, lo=0.05, hi=0.35), *_season("A18", 3, 60),
             *_kicks("T18", 3, 9), *_kicks("A18", 3, 60)]  # fmt: skip
    f = _framed(parts, family={"T18": gc.TEAM})
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    t18 = rated.filter(pl.col("season_id") == "sb:3")
    assert t18["zero_family"].all() and not t18["small"].any() and not t18["pool_small"].any()
    assert (t18["delta"] == 0.0).all() and np.allclose(t18["q"], t18["xgot"], atol=1e-12)
    assert set(t18["kind"].cast(pl.String).to_list()) == {"open_play", "penalty"}
    assert set(t18["group"].to_list()) == {_label("T18")}
    assert not rated.filter(pl.col("season_id") != "sb:3")["zero_family"].any()
    free = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=())
    assert not free["zero_family"].any()
    assert (free.filter(pl.col("season_id") == "sb:3")["delta"] != 0.0).all()


def test_zero_families_bad_value() -> None:
    """§3.1: ``zero_families`` is a sequence of ``data.FAMILIES`` values: "team selected" (no
    hyphen) raises ``ValueError`` naming it and the allowed values; a bare ``str`` raises
    ``TypeError`` (it would iterate as letters)."""
    gsax = _gsax()
    f = gc.table_frame()
    with pytest.raises(ValueError, match="team selected") as e:
        gsax.rate(f, small=None, zero_families=("team selected",))
    assert "team-selected" in str(e.value) and "women" in str(e.value)
    with pytest.raises(TypeError):
        gsax.rate(f, small=None, zero_families="team-selected")


_SEASON_POOL = ("A18", "A19", "B18", "T18")


@settings(max_examples=40, deadline=None)
@given(st.data())
def test_rate_property(draw: st.DataObject) -> None:
    """Property: for any frame, (1) δ is constant within each (kind, group); (2) every row at
    δ = 0 by rule (``AT_ZERO``) has q = xgot; (3) every other group with a finite δ sums to its
    goals, Σ (q − goal) within ``SUM_TOL``; (4) a penalty row's ``group`` is its source unless
    its season is small or team-selected; (5) the flags are per season (``small``,
    ``zero_family``) or per pool (``pool_small``)."""
    gsax = _gsax()
    n = draw.draw(st.integers(min_value=4, max_value=50))
    pick = lambda xs: draw.draw(st.lists(st.sampled_from(xs), min_size=n, max_size=n))  # noqa: E731
    seasons = pick(list(_SEASON_POOL))
    f = gc.frame(
        gk=pick(["sb:g1", "sb:g2", "sb:g3", "sb:g4"]), season=seasons,
        match=pick(["sb:m1", "sb:m2", "sb:m3"]),
        goal=draw.draw(st.lists(st.booleans(), min_size=n, max_size=n)),
        xgot=draw.draw(st.lists(st.floats(0.01, 0.99), min_size=n, max_size=n)),
        kind=pick(["open_play", "open_play", "penalty"]),
        family=[gc.TEAM if s == "T18" else gc.LEAGUE for s in seasons],
        source=["holdout_fit" if s == "B18" else "season_folds" for s in seasons],
    )  # fmt: skip
    rule = draw.draw(st.sampled_from([None, gsax.SmallLeague(1, 3), gsax.SmallLeague(2, 5)]))
    rated = gsax.rate(f, small=rule, zero_families=gsax.ZERO_FAMILIES)
    assert rated.height == f.height
    per = rated.group_by("kind", "group").agg(
        pl.col("delta").n_unique().alias("n"), pl.col("delta").first().alias("d"),
        pl.col("pool_small").n_unique().alias("p"),
        (pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum().alias("gap"),
        (pl.col("small") | pl.col("zero_family") | pl.col("pool_small")).all().alias("z"),
    )  # fmt: skip
    assert (per["n"] == 1).all() and (per["p"] == 1).all()
    for kind, group, _, d, _, gap, z in per.iter_rows():
        if not z and math.isfinite(d):
            assert abs(gap) <= gsax.SUM_TOL, (kind, group)
    zero = rated.filter(gsax.AT_ZERO)
    assert (zero["delta"] == 0.0).all() and np.allclose(zero["q"], zero["xgot"], atol=1e-12)
    assert (rated.filter(pl.col("season_id") == "sb:3")["zero_family"]).all()
    flags = rated.group_by(KEY).agg(pl.col("small").n_unique().alias("s"),
                                    pl.col("zero_family").n_unique().alias("z"))  # fmt: skip
    assert (flags["s"] == 1).all() and (flags["z"] == 1).all()
    pens = rated.filter((pl.col("kind") == "penalty") & ~pl.col("small") & ~pl.col("zero_family"))
    assert (pens["group"] == pens["source"].cast(pl.String)).all()
    assert not rated.filter(pl.col("kind") == "open_play")["pool_small"].any()


def test_rating_groups_values() -> None:
    """§3.2: one row per (kind, group) of ``rate`` output, sorted by kind then group; ``rows``,
    ``keepers``, ``goals`` and ``gap`` = Σ (q − goal) recomputed here, ``delta`` and the three
    flags copied; Int64 / Float64 / Boolean / String dtypes."""
    gsax = _gsax()
    parts = [*_season("A18", 3, 60), *_season("T18", 3, 60), *_season("F1", 2, 20),
             *_season("H15", 3, 60), *_kicks("A18", 3, 60), *_kicks("T18", 2, 6),
             *_kicks("H15", 2, 8)]  # fmt: skip
    f = _framed(parts, family={"T18": gc.TEAM}, source={"H15": "holdout_fit"}).sort("shot_id")
    rated = gsax.rate(f, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    t = gsax.rating_groups(rated)
    cols = ["kind", "group", "rows", "keepers", "goals", "delta", "small", "zero_family",
            "pool_small", "gap"]  # fmt: skip
    assert sorted(t.columns) == sorted(cols)
    types = {"kind": pl.String, "group": pl.String, "rows": pl.Int64, "keepers": pl.Int64,
             "goals": pl.Int64, "delta": pl.Float64, "gap": pl.Float64}  # fmt: skip
    assert all(t.schema[c] == d for c, d in types.items())
    assert all(t.schema[c] == pl.Boolean for c in ("small", "zero_family", "pool_small"))
    assert t.select("kind", "group").rows() == sorted(t.select("kind", "group").rows())
    r = rated.with_columns(k=pl.col("kind").cast(pl.String))
    want = {}
    for (k, grp), part in r.group_by("k", "group"):
        want[(k, grp)] = (part.height, part["gk_id"].n_unique(), int(part["is_goal"].sum()),
                          float((part["q"] - part["is_goal"].cast(pl.Float64)).sum()),
                          part["delta"][0], part["small"][0], part["zero_family"][0],
                          part["pool_small"][0])  # fmt: skip
    assert t.height == len(want) == 7  # 4 open-play seasons, the 2 pools, T18's kicks at 0
    for row in t.iter_rows(named=True):
        w = want[(row["kind"], row["group"])]
        assert (row["rows"], row["keepers"], row["goals"]) == w[:3], row
        assert row["gap"] == pytest.approx(w[3], abs=1e-12) and row["delta"] == w[4], row
        assert (row["small"], row["zero_family"], row["pool_small"]) == w[5:], row
    pool = t.filter((pl.col("kind") == "penalty") & (pl.col("group") == "holdout_fit"))
    assert pool["pool_small"].item() and pool["delta"].item() == 0.0  # 2 keepers
    empty = gsax.rating_groups(rated.clear())
    assert empty.height == 0 and sorted(empty.columns) == sorted(cols)
