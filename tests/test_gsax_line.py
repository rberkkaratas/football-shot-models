"""The goal_z sensitivity of GSAx (spec ``runs/PR1/keeper-fixes`` §3.6; PM after MS6 "GSAx on saves
stopped within 1 yd of the line beside the headline"; PM E2 (c), PR1: the z-free refit beside it).

A StatsBomb projected save's ``goal_z`` is where the keeper met the ball, so only the saves
stopped within ``LINE_YD`` of the line (``end_x >= LINE_X``) keep a z near the line. The line
sensitivity re-rates every goal and those saves only (δ and E2 smallness refitted on the subset);
the z-free sensitivity rates a season-out refit without ``mouth_z``. Both are summaries beside
the headline, never a per-keeper column on the card.

Unit tests use hand-built frames (``gsax_cases``); ``line_summary`` / ``shift_summary`` use the
session ``gsax_census`` (``conftest``), whose stops sit on both sides of ``LINE_X``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import gsax_cases as gc
import numpy as np
import polars as pl
import pytest
from scipy import stats

SEASON_KEY = ("competition_id", "season_id")
KEEPER_SEASON = ("gk_id", *SEASON_KEY)


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def test_line_constants() -> None:
    """``LINE_YD`` 1.0; ``LINE_X`` is StatsBomb raw x 120 − 1 yd on the fine grid, a float
    (104.0375 m, the snapshot ``end_x`` of raw 119.0); ``LINE_PROVIDERS`` is StatsBomb only."""
    from football_core import pitch
    from football_core.io import statsbomb

    gsax = _gsax()
    assert gsax.LINE_YD == 1.0
    want = pitch.to_canonical(119.0, 40.0, "statsbomb", cell_offset=statsbomb.FINE_OFFSET)[0]
    assert type(gsax.LINE_X) is float and float(want) == gsax.LINE_X
    assert pytest.approx(104.0375, abs=1e-9) == gsax.LINE_X
    assert frozenset({"statsbomb"}) == gsax.LINE_PROVIDERS


def _line_frame() -> tuple[pl.DataFrame, pl.DataFrame]:
    """A18: 4 keepers, 64 open-play shots; B18: 3 keepers, 54; plus 6 kicks (no stop: penalties
    are not in the sensitivity). Saves alternate far / near by row with one save exactly on
    ``LINE_X`` and one a float below it; every goal is stopped at 100.0 (far: still kept)."""
    gsax = _gsax()
    parts = [
        *(gc.block(f"sb:g1{i}", "A18", goals=gc.pattern(16, 3, i), xgot=gc.spread(16, 0.1, 0.6),
                   matches=[f"sb:a{i}", f"sb:a{i + 4}"]) for i in range(4)),
        *(gc.block(f"sb:g2{i}", "B18", goals=gc.pattern(18, 4, i), xgot=gc.spread(18, 0.1, 0.5),
                   matches=[f"sb:b{i}", f"sb:b{i + 3}"]) for i in range(3)),
        gc.block("sb:g10", "A18", goals=[True, False, True], xgot=0.75, matches=["sb:a0"],
                 kind="penalty"),
        gc.block("sb:g20", "B18", goals=[True, True, False], xgot=0.75, matches=["sb:b0"],
                 kind="penalty"),
    ]  # fmt: skip
    pred = gc.blocks(*parts)
    open_ = pred.filter(pl.col("kind") == "open_play").sort("shot_id")
    below = float(np.nextafter(gsax.LINE_X, 0.0))
    end_x, saves = [], 0
    for goal in open_["is_goal"].to_list():
        if goal:
            end_x.append(100.0)
            continue
        end_x.append([gsax.LINE_X, below, 104.6, 99.0, 104.2][saves] if saves < 5
                     else (100.0 if saves % 3 == 0 else 104.4))  # fmt: skip
        saves += 1
    stops = pl.DataFrame({"shot_id": open_["shot_id"], "end_x": pl.Series(end_x, dtype=pl.Float64)})
    return pred, stops


def test_line_threshold() -> None:
    """A save is kept iff ``end_x >= LINE_X`` (the save exactly on ``LINE_X`` in float kept, the
    one ``nextafter`` below dropped); every goal is kept at any ``end_x``; penalties are not in
    the output. The kept rows are re-rated by ``rate(..., small=SMALL_LEAGUE,
    zero_families=ZERO_FAMILIES)`` on the subset (δ refitted: it differs from the headline's).
    An empty ``pred`` gives an empty frame."""
    gsax = _gsax()
    pred, stops = _line_frame()
    below = float(np.nextafter(gsax.LINE_X, 0.0))
    assert below < gsax.LINE_X and stops.filter(pl.col("end_x") == below).height == 1
    assert stops.filter(pl.col("end_x") == gsax.LINE_X).height == 1  # preconditions
    out = gsax.line_rated(pred, stops, coarse=pl.Series([], dtype=pl.String))
    j = pred.join(stops, on="shot_id", how="left")
    keep = j.filter(
        (pl.col("kind") == "open_play") & (pl.col("is_goal") | (pl.col("end_x") >= gsax.LINE_X))
    )
    assert set(out["shot_id"].to_list()) == set(keep["shot_id"].to_list())
    on_line = stops.filter(pl.col("end_x") == gsax.LINE_X)["shot_id"].item()
    off_line = stops.filter(pl.col("end_x") == below)["shot_id"].item()
    assert on_line in set(out["shot_id"]) and off_line not in set(out["shot_id"])
    goals = j.filter((pl.col("kind") == "open_play") & pl.col("is_goal"))
    assert set(goals["shot_id"]) <= set(out["shot_id"]) and (goals["end_x"] < gsax.LINE_X).all()
    assert set(out["kind"].cast(pl.String).to_list()) == {"open_play"}
    want = gsax.rate(pred.filter(pl.col("shot_id").is_in(keep["shot_id"].implode())),
                     small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)  # fmt: skip
    cols = ("shot_id", "delta", "q")
    assert out.select(cols).sort("shot_id").equals(want.select(cols).sort("shot_id"))
    full = gsax.rate(pred, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    d_full = full.filter((pl.col("kind") == "open_play") & (pl.col("season_id") == "sb:42"))
    d_out = out.filter(pl.col("season_id") == "sb:42")
    assert abs(d_full["delta"][0] - d_out["delta"][0]) > 0.01  # the subset's own δ
    empty = gsax.line_rated(pred.clear(), stops, coarse=pl.Series([], dtype=pl.String))
    assert empty.height == 0  # §7: empty in, empty out


@pytest.mark.parametrize("case", ["null end_x", "NaN end_x", "missing stop", "coarse match"])
def test_line_rated_errors(case: str) -> None:
    """§3.6 / §7: a save with a null or NaN ``end_x``, an open-play row without a stop, and a row
    of a ``coarse`` match raise ``ValueError`` naming the count (2 rows each), never dropped."""
    gsax = _gsax()
    pred, stops = _line_frame()
    saves = pred.filter((pl.col("kind") == "open_play") & ~pl.col("is_goal")).sort("shot_id")
    two = saves["shot_id"].head(2)
    coarse = pl.Series([], dtype=pl.String)
    hit = pl.col("shot_id").is_in(two.implode())
    if case in ("null end_x", "NaN end_x"):
        value = None if case == "null end_x" else math.nan
        stops = stops.with_columns(
            end_x=pl.when(hit).then(pl.lit(value, dtype=pl.Float64)).otherwise(pl.col("end_x"))
        )
    elif case == "missing stop":
        stops = stops.filter(~hit)
    else:  # two saves moved to a match of their own, on the coarse grid
        pred = pred.with_columns(
            match_id=pl.when(hit).then(pl.lit("sb:coarse")).otherwise(pl.col("match_id"))
        )
        coarse = pl.Series(["sb:coarse"])
    with pytest.raises(ValueError, match=r"\b2\b"):
        gsax.line_rated(pred, stops, coarse=coarse)


def test_stops(xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """``stops(f)``: ``shot_id`` (unique) and ``end_x`` (Float64) of every open-play row of
    ``f.inp.on_dev`` and ``on_held``; None for WhoScored, and for StatsBomb once it leaves
    ``LINE_PROVIDERS`` (read when called)."""
    gsax = _gsax()
    f = xgot_fits["fits"]["statsbomb"]
    out = gsax.stops(f)
    assert out is not None and out.columns == ["shot_id", "end_x"]
    assert out.schema["end_x"] == pl.Float64 and out["shot_id"].n_unique() == out.height
    ids: set[str] = set()
    for rows in (f.inp.on_dev, f.inp.on_held):
        if "kind" in rows.columns:
            rows = rows.filter(pl.col("kind").cast(pl.String) == "open_play")
        ids |= set(rows["shot_id"].to_list())
    assert set(out["shot_id"].to_list()) == ids and out.height > 0
    assert gsax.stops(xgot_fits["fits"]["whoscored"]) is None
    monkeypatch.setattr(gsax, "LINE_PROVIDERS", frozenset())
    assert gsax.stops(f) is None


# ---- summaries on the census -----------------------------------------------------------------


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).statistic)


def _far_counts(c: dict[str, Any]) -> pl.DataFrame:
    """Per keeper-season: open-play shots, saves and far saves (``end_x < LINE_X``)."""
    gsax = _gsax()
    rows = c["pred"].filter(pl.col("kind") == "open_play").join(c["stops"], on="shot_id")
    far = ~pl.col("is_goal") & (pl.col("end_x") < gsax.LINE_X)
    return rows.group_by(KEEPER_SEASON).agg(
        shots=pl.len(), saves=(~pl.col("is_goal")).sum(), far=far.sum()
    )


def _fitted(c: dict[str, Any]) -> pl.DataFrame:
    """The competition-seasons with a δ fitted on the line subset, recomputed from the census's
    line rows (not ``g.line_seasons``): neither ``small`` nor ``zero_family`` there."""
    gsax = _gsax()
    rows = gsax.line_rated(c["pred"], c["stops"], coarse=pl.Series([], dtype=pl.String))
    return rows.filter(~pl.col("small") & ~pl.col("zero_family")).select(SEASON_KEY).unique()


def test_line_summary(gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """``line_summary(g)`` recomputed: over the keeper-seasons with >= ``TABLE_MIN_SHOTS``
    headline shots (lowered to 20, read when called) in competition-seasons with a δ fitted on
    the line subset (audit A10), the shift ``gsax_line − gsax``: median
    and max |shift|, the 90th percentile between them; ``rank_r`` the Spearman of ``gsax`` and
    ``gsax_line``; ``far_share_r`` the Spearman of the shift and the keeper-season's far saves
    per shot (the probe's share, spec §1); ``far`` the open-play saves stopped before
    ``LINE_X``. ``keeper_rows`` carries ``gsax_line`` = ``g.line``'s GSAx of the same
    keeper-season."""
    gsax = _gsax()
    g = gsax_census["g"]
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 20)
    rows = gsax.keeper_rows(g)
    line = g.line.select(*KEEPER_SEASON, want="gsax")
    j = rows.join(line, on=KEEPER_SEASON, how="left")
    assert j["gsax_line"].equals(j["want"], check_names=False)
    every = j.drop_nulls("gsax_line").join(_far_counts(gsax_census), on=KEEPER_SEASON, how="left")
    j = every.join(_fitted(gsax_census), on=SEASON_KEY, how="semi")
    assert j.height >= 8 and every.height - j.height >= 2  # preconditions: both sides populated
    shift = (j["gsax_line"] - j["gsax"]).to_numpy()
    s = gsax.line_summary(g)
    assert s is not None and s.keeper_seasons == j.height
    assert s.at_zero == every.height - j.height
    assert s.median_abs == pytest.approx(float(np.median(np.abs(shift))), abs=1e-9)
    assert s.max_abs == pytest.approx(float(np.abs(shift).max()), abs=1e-9)
    assert s.median_abs <= s.p90_abs <= s.max_abs and s.mean_sd > 0
    assert s.rank_r == pytest.approx(_spearman(j["gsax"].to_numpy(), j["gsax_line"].to_numpy()),
                                     abs=1e-9)  # fmt: skip
    share = (j["far"] / j["shots"]).to_numpy()
    assert len(set(share.tolist())) > 3  # precondition: the share varies by keeper
    assert s.far_share_r == pytest.approx(_spearman(shift, share), abs=1e-9)
    far = _far_counts(gsax_census)
    assert s.far == int(far["far"].sum()) > 0
    assert s.saves in (int(far["saves"].sum()), int(far["saves"].sum() - far["far"].sum()))


def test_line_summary_none_branches(
    gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.6 / §7: ``g.line`` None -> None; one keeper-season (only sb:g01 A18 reaches 100 shots)
    -> both Spearmans None (polars' NaN mapped); a shift constant across keeper-seasons ->
    ``far_share_r`` None while ``rank_r`` is 1; no keeper-season -> every float None."""
    gsax = _gsax()
    g = gsax_census["g"]
    assert gsax.line_summary(dataclasses.replace(g, line=None)) is None
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 100)
    one = gsax.line_summary(g)
    assert one is not None and one.keeper_seasons == 1
    assert one.rank_r is None and one.far_share_r is None
    assert one.median_abs == pytest.approx(one.max_abs, abs=1e-12)
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 20)
    shifted = g.keepers[gsax.WITHOUT].with_columns(gsax=pl.col("gsax") + 0.5)
    const = gsax.line_summary(dataclasses.replace(g, line=shifted))
    assert const is not None and const.far_share_r is None
    assert const.rank_r == pytest.approx(1.0, abs=1e-12)
    assert const.median_abs == pytest.approx(0.5, abs=1e-9)
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 10_000)
    none = gsax.line_summary(g)
    assert none is not None and none.keeper_seasons == 0
    fields = (none.median_abs, none.p90_abs, none.max_abs, none.mean_sd, none.rank_r,
              none.far_share_r)  # fmt: skip
    assert fields == (None,) * 6


def test_keeper_rows_seam_columns(
    gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chart seam (§3.6, §12): ``keeper_rows`` carries ``gsax_line`` (null for a keeper-season
    with no row in ``g.line``, all null when ``g.line`` is None) and ``zero_family`` (the
    season's flag, for a team-selected filter without a refit)."""
    gsax = _gsax()
    g = gsax_census["g"]
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 20)
    rows = gsax.keeper_rows(g)
    assert {"gsax_line", "zero_family"} <= set(rows.columns)
    team = rows.filter(pl.col("family").cast(pl.String) == gsax.TEAM_SELECTED)
    assert team.height and team["zero_family"].all()  # precondition: sb:g31 T19
    assert not rows.filter(pl.col("family").cast(pl.String) != gsax.TEAM_SELECTED)[
        "zero_family"
    ].any()
    first = rows.select(KEEPER_SEASON).row(0)
    hit = pl.all_horizontal(pl.col(c) == v for c, v in zip(KEEPER_SEASON, first, strict=True))
    fewer = gsax.keeper_rows(dataclasses.replace(g, line=g.line.filter(~hit)))
    assert fewer.filter(hit)["gsax_line"].is_null().all()
    assert (
        fewer.filter(~hit)["gsax_line"].null_count() == rows.filter(~hit)["gsax_line"].null_count()
    )
    assert gsax.keeper_rows(dataclasses.replace(g, line=None))["gsax_line"].is_null().all()


# ---- PM E2 (c): the z-free sensitivity --------------------------------------------------------


def test_noz_rated(monkeypatch: pytest.MonkeyPatch) -> None:
    """``noz_rated(pred)`` rates the z-free season-out rows with the §3.1 rule:
    ``rate(pred, small=SMALL_LEAGUE, zero_families=ZERO_FAMILIES)``, constants read when called
    (a patched ``SMALL_LEAGUE`` changes it)."""
    gsax = _gsax()
    pred = gc.census_frame().filter(pl.col("kind") == "open_play")
    want = gsax.rate(pred, small=gsax.SMALL_LEAGUE, zero_families=gsax.ZERO_FAMILIES)
    assert gsax.noz_rated(pred).equals(want)
    other = gsax.SmallLeague(3, 50)
    monkeypatch.setattr(gsax, "SMALL_LEAGUE", other)
    got = gsax.noz_rated(pred)
    assert got.equals(gsax.rate(pred, small=other, zero_families=gsax.ZERO_FAMILIES))
    assert not got["small"].equals(want["small"])  # precondition: the rule matters here


def test_shift_summary_noz(gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """PM E2 (c): with ``g.noz`` (the keeper table of ``noz_rated`` rows), ``keeper_rows`` carries
    ``gsax_noz`` and ``shift_summary(g, "gsax_noz")`` gives the keeper-level shift beside the
    headline: median and max |shift| and the Spearman with the headline, recomputed here on the
    line summary's population (audit A10: δ fitted on the line subset); without ``g.noz`` it is
    None; an unknown column raises ``ValueError``."""
    gsax = _gsax()
    g = gsax_census["g"]
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 20)
    assert gsax.shift_summary(dataclasses.replace(g, noz=None), "gsax_noz") is None
    open_ = gsax_census["pred"].filter(pl.col("kind") == "open_play")
    moved = open_.with_columns(xgot=(pl.col("xgot") * (0.8 + 0.4 * (pl.int_range(pl.len()) % 3)
                                                        / 2)).clip(0.01, 0.99))  # fmt: skip
    noz = gsax.keeper_table(gsax.noz_rated(moved))
    g2 = dataclasses.replace(g, noz=noz)
    rows = gsax.keeper_rows(g2)
    j = rows.join(noz.select(*KEEPER_SEASON, want="gsax"), on=KEEPER_SEASON, how="left")
    assert j["gsax_noz"].equals(j["want"], check_names=False) and j.height >= 10
    j = j.join(_fitted(gsax_census), on=SEASON_KEY, how="semi")
    shift = (j["gsax_noz"] - j["gsax"]).to_numpy()
    s = gsax.shift_summary(g2, "gsax_noz")
    line = gsax.line_summary(g2)
    assert s is not None and line is not None and s.keeper_seasons == j.height
    assert (s.keeper_seasons, s.at_zero) == (line.keeper_seasons, line.at_zero)  # one population
    assert s.median_abs == pytest.approx(float(np.median(np.abs(shift))), abs=1e-9)
    assert s.max_abs == pytest.approx(float(np.abs(shift).max()), abs=1e-9)
    assert s.rank_r == pytest.approx(_spearman(j["gsax"].to_numpy(), j["gsax_noz"].to_numpy()),
                                     abs=1e-9)  # fmt: skip
    with pytest.raises(ValueError, match="gsax_nope"):
        gsax.shift_summary(g2, "gsax_nope")


@pytest.mark.parametrize("column", ["gsax_line", "gsax_noz"])
def test_shift_summary_leaves_out_delta_zero_seasons(
    gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch, column: str
) -> None:
    """Audit A10: a keeper-season in a competition-season at δ = 0 on the line subset (sb:g31
    in team-selected sb:43 sb:4; sb:16 sb:30, E2-small only on the subset) is left out of both
    summaries and only counted (``at_zero``, its median |shift| beside it): moving its
    sensitivity GSAx by 1,000 goals changes no judged number, while moving a fitted
    keeper-season's (sb:g01 sb:9 sb:42) changes the max. ``line_seasons`` is
    ``season_flags`` of the line rows."""
    gsax = _gsax()
    g = gsax_census["g"]
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 20)
    open_ = gsax_census["pred"].filter(pl.col("kind") == "open_play")
    g = dataclasses.replace(g, noz=gsax.keeper_table(gsax.noz_rated(open_)))
    line_rows = gsax.line_rated(gsax_census["pred"], gsax_census["stops"],
                                coarse=pl.Series([], dtype=pl.String))  # fmt: skip
    assert g.line_seasons.equals(gsax.season_flags(line_rows))
    flags = line_rows.group_by(SEASON_KEY).agg(pl.col("small", "zero_family").first())
    team = flags.filter(pl.col("zero_family") & ~pl.col("small"))
    assert ("sb:43", "sb:4") in team.select(SEASON_KEY).rows()  # precondition: not small there
    assert flags.filter((pl.col("competition_id") == "sb:16") & (pl.col("season_id") == "sb:30")
                        & pl.col("small") & ~pl.col("zero_family")).height == 1  # fmt: skip
    base = gsax.shift_summary(g, column)
    assert base is not None and base.at_zero >= 2

    def moved(gk: str, comp: str, season: str) -> Any:
        hit = ((pl.col("gk_id") == gk) & (pl.col("competition_id") == comp)
               & (pl.col("season_id") == season))  # fmt: skip
        attr = gsax.SHIFTS[column]
        table = getattr(g, attr)
        assert table.filter(hit).height == 1  # precondition: the keeper-season is in the table
        bumped = table.with_columns(gsax=pl.when(hit).then(pl.col("gsax") + 1000.0)
                                    .otherwise(pl.col("gsax")))  # fmt: skip
        return gsax.shift_summary(dataclasses.replace(g, **{attr: bumped}), column)

    for key in (("sb:g31", "sb:43", "sb:4"), ("sb:g21", "sb:16", "sb:30")):
        s = moved(*key)
        assert (s.keeper_seasons, s.median_abs, s.p90_abs, s.max_abs, s.rank_r) == (
            base.keeper_seasons,
            base.median_abs,
            base.p90_abs,
            base.max_abs,
            base.rank_r,
        )
        assert s.at_zero == base.at_zero
    fitted = moved("sb:g01", "sb:9", "sb:42")
    assert fitted.max_abs > 900 and fitted.keeper_seasons == base.keeper_seasons
    rows = gsax.keeper_rows(g).drop_nulls(column)
    out = rows.join(_fitted(gsax_census), on=SEASON_KEY, how="anti")
    d = (out[column] - out["gsax"]).abs().to_numpy()
    assert base.at_zero == out.height and base.at_zero_median_abs == pytest.approx(
        float(np.median(d)), abs=1e-9
    )
