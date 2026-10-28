"""Ratify tests for the MS6 gsax build decisions (``runs/MS6/gsax/build.md`` R1-R21) that had no
dedicated test: one test per decision, each failing if the decision is reverted.

Synthetic frames only (``gsax_cases``, ``xgot_cases``); the card tests reuse the session
``xgot_fits`` / ``gsax_fits`` fixtures and change one field with ``dataclasses.replace``.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import gsax_cases as gc
import numpy as np
import polars as pl
import pytest
import xgot_cases as xg
from scipy import optimize
from scipy.special import expit

from football_shot_models import gsax, gsax_cards, metrics, xgot, xgot_cards

KEY = ("competition_id", "season_id")


# ---- R1: coarse-grid matches leave whole; the xG input fit keeps them -----------------------


def test_coarse_matches_leave_whole(
    xgot_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 (PM E1 (a)): with one StatsBomb match on the coarse grid, its open-play shots and its
    kicks leave the predictions, the penalty xGOT and ``inp``; its on-target C3 rows are
    ``coarse_grid`` in the reconciliation and in ``grid``; the xG input fit still sees them
    (its out-of-fold xG equals the fit without a coarse match)."""
    from football_core.io import snapshot

    base = xgot_fits["fits"]["statsbomb"]
    pred = base.predictions
    both = set(base.kicks.dev["match_id"]) & set(
        pred.filter((pl.col("kind") == "open_play") & (pl.col("split") == "dev"))["match_id"]
    )
    assert both  # precondition: a dev match with open-play shots and an in-game kick
    c3 = snapshot.load_snapshot(xgot_fits["manifests"]["statsbomb"], tables=("shots",))["shots"]
    mid, f = None, None
    for cand in sorted(both):
        monkeypatch.setattr(xgot, "coarse_matches", lambda _c3, m=cand: pl.Series([m]))
        monkeypatch.setattr(
            xgot, "PROVIDERS", xg.patched_providers(xgot, xgot_fits["manifests"], monkeypatch)
        )
        monkeypatch.chdir(tmp_path)
        try:
            f = xgot.fit("statsbomb", cache=None)
        except ValueError:  # a fold left without a test row: try the next match
            continue
        mid = cand
        break
    assert f is not None and mid is not None
    hit = pl.col("match_id") == mid
    assert base.predictions.filter(hit).height and not f.predictions.filter(hit).height
    for frame in (f.inp.dev, f.inp.held, f.inp.on_dev, f.inp.on_held, f.kicks.dev, f.kicks.held,
                  f.kicks.shootouts):  # fmt: skip
        assert frame.filter(hit).height == 0
    assert base.inp.on_dev.filter(hit).height  # precondition: the match was in the fit
    on = c3.filter(pl.col("on_target") & hit)
    rows = {r: (n, g) for r, n, g in f.recon.rows}
    assert f.recon.rows[0][0] == "coarse_grid"
    assert rows["coarse_grid"] == (on.height, int(on["is_goal"].sum()))
    assert f.grid.height == 1 and f.grid.row(0)[2:] == (1, on.height, int(on["is_goal"].sum()))
    assert np.array_equal(f.xg_input.oof, base.xg_input.oof)  # the xG input fit is unchanged


# ---- R3: an unknown grid is kept --------------------------------------------------------------


def test_unknown_grid_match_kept() -> None:
    """R3: ``match_grids`` gives null fidelity for a match with no usable location; such a match
    is not coarse, so ``coarse_matches`` keeps it in the fit; WhoScored rows are ignored."""
    from football_core import goalmouth

    c3 = pl.DataFrame({
        "provider": ["statsbomb", "statsbomb", "statsbomb", "statsbomb", "whoscored"],
        "match_id": ["sb:3", "sb:1", "sb:2", "sb:1", "ws:9"],
        "is_own_goal": [False] * 5,
        "raw_qualifiers": ["{}", '{"location":[100.0,40.0]}', '{"location":[100.4,40.0]}',
                           '{"location":[101.0,39.0]}', '{"location":[100.0,40.0]}'],
    })  # fmt: skip
    assert goalmouth.match_grids(c3).rows() == [("sb:1", "coarse"), ("sb:2", "fine"),
                                                ("sb:3", None)]  # fmt: skip
    assert xgot.coarse_matches(c3).to_list() == ["sb:1"]


# ---- R5: league_shift is polished past brentq ---------------------------------------------------


def test_league_shift_polished() -> None:
    """R5: ``league_shift`` polishes brentq with Newton steps: on 5,000 rows where brentq alone
    leaves Σ q − goals above 1e-11, the returned δ leaves it at rounding level."""
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.random(5000) < 0.9 * p).astype(np.float64)
    lp = gc.logit(p)

    def gap(d: float) -> float:
        return float(expit(lp + d).sum()) - float(y.sum())

    raw = optimize.brentq(gap, -gsax.BRACKET, gsax.BRACKET, xtol=1e-12)
    assert abs(gap(raw)) > 1e-11  # precondition: brentq alone is not enough here
    assert abs(gap(gsax.league_shift(p, y))) <= 1e-11


# ---- R6: a kicks-only competition-season is small ------------------------------------------


def test_kicks_only_season_is_small() -> None:
    """R6: with a small-league rule, a competition-season with penalties but no open-play row is
    small (δ 0, q = xgot); without the rule it is rated to its league sum."""
    parts = [gc.block(f"sb:g0{i}", "A18", goals=gc.pattern(20, 4), xgot=gc.spread(20, 0.1, 0.6),
                      matches=["sb:a1", "sb:a2"]) for i in range(3)]  # fmt: skip
    parts.append(gc.block("sb:g09", "B18", goals=[True, False, True, True],
                          xgot=[0.7, 0.8, 0.75, 0.9], matches=["sb:b1", "sb:b2"],
                          kind="penalty"))  # fmt: skip
    f = gc.blocks(*parts)
    b18 = pl.col("season_id") == "sb:30"
    rated = gsax.rate(f, small=gsax.SmallLeague(2, 50))
    assert rated.filter(b18)["small"].all() and (rated.filter(b18)["delta"] == 0.0).all()
    assert np.allclose(rated.filter(b18)["q"].to_numpy(), rated.filter(b18)["xgot"].to_numpy())
    assert not rated.filter(~b18)["small"].any()  # A18: 3 keepers, 60 shots
    free = gsax.rate(f, small=None).filter(b18)
    assert not free["small"].any() and free["delta"][0] != 0.0
    assert abs(float((free["q"] - free["is_goal"].cast(pl.Float64)).sum())) <= gsax.SUM_TOL


# ---- R7, R8, R10: next-season pairs, loo_missing, bare str ----------------------------------


def _next_frame(*, a20_unpaired_goals: bool = True, b19_unpaired: bool = True) -> pl.DataFrame:
    """g01 A18 -> A19 -> A20; g04 B18 -> B19; g10 A18 (goals) -> A19 (no goal) -> A20; g11 A18
    and A19 (no goal) -> A20; g05 and g06 unpaired in A20 and B19; g08 unpaired in A19. Under
    ``KEEPER_SEASON`` the A20 rows of g05, g10 and g11 are the unpaired ones (g10's and g11's
    pairs are left out)."""
    m = iter(f"sb:x{i}" for i in range(100))
    blk = lambda gk, s, goals, n=16: gc.block(  # noqa: E731
        gk,
        s,
        goals=goals if isinstance(goals, list) else gc.pattern(n, goals),
        xgot=gc.spread(n, 0.1, 0.6),
        matches=[next(m), next(m)],
    )
    a20: Any = 4 if a20_unpaired_goals else [False] * 16  # g05, g10, g11: unpaired in A20
    parts = [
        blk("sb:g01", "A18", 4), blk("sb:g01", "A19", 3), blk("sb:g01", "A20", 5),
        blk("sb:g04", "B18", 4), blk("sb:g04", "B19", 3),
        blk("sb:g10", "A18", 3), blk("sb:g10", "A19", [False] * 16), blk("sb:g10", "A20", a20),
        blk("sb:g11", "A18", [False] * 16), blk("sb:g11", "A19", [False] * 16),
        blk("sb:g11", "A20", a20), blk("sb:g08", "A19", 3), blk("sb:g05", "A20", a20),
    ]  # fmt: skip
    if b19_unpaired:
        parts.append(blk("sb:g06", "B19", 2))
    return gc.blocks(*parts).sort("shot_id")


def test_next_season_left_out_by_grouping() -> None:
    """R7: a pair is left out when its raw estimate in its ``by`` group is infinite: for
    ``KEEPER_SEASON`` the rated season's rows (g10 A19 -> A20, g11 both pairs: 3 out), for
    ``CAREER`` the keeper's rows with start <= t (g10's A18 goals keep his A19 -> A20 pair;
    g11 has none: 2 out). R10: a bare str for ``by`` raises ``TypeError``."""
    rated = gsax.rate(_next_frame(), small=None)
    season = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    career = gsax.next_season(rated, by=gsax.CAREER)
    assert (season.pairs + season.left_out, season.left_out) == (7, 3)
    assert (career.pairs + career.left_out, career.left_out) == (7, 2)
    with pytest.raises(TypeError):
        gsax.next_season(rated, by="gk_id")


def test_next_season_loo_missing() -> None:
    """R8: ``loo_missing`` counts the test seasons with no unpaired row (B19 without g06) or a
    non-finite δ without the paired keepers (A20 whose unpaired keepers concede 0); 0 when
    every test season has one, and then the leave-out gains exist."""
    full = gsax.next_season(gsax.rate(_next_frame(), small=None), by=gsax.KEEPER_SEASON)
    assert full.loo_missing == 0 and full.gain_loo is not None
    one = _next_frame(b19_unpaired=False)
    ns = gsax.next_season(gsax.rate(one, small=None), by=gsax.KEEPER_SEASON)
    assert ns.untestable is None and ns.gain is not None
    assert ns.loo_missing == 1 and ns.gain_loo is None and ns.gain_none_loo is None
    two = _next_frame(b19_unpaired=False, a20_unpaired_goals=False)
    ns = gsax.next_season(gsax.rate(two, small=None), by=gsax.KEEPER_SEASON)
    assert ns.loo_missing == 2 and ns.gain_loo is None


# ---- R9: the Spearman-Brown fields are SplitHalf ---------------------------------------------


def test_tier_sb_are_split_half() -> None:
    """R9: ``Tier.first_sb`` / ``all_shots_sb`` are ``SplitHalf`` with the base's keys and
    r / low / high stepped up (an empty tier included)."""
    parts = [gc.block(f"sb:k{i}", "A18", goals=gc.pattern(100, 3 + i % 4, i),
                      xgot=gc.spread(100, 0.1, 0.6, step=7 + 2 * i),
                      matches=[f"sb:m{i}{j}" for j in range(5)]) for i in range(6)]  # fmt: skip
    tiers = gsax.reliability(gc.rated(gc.blocks(*parts)))
    assert tiers[0].keepers == 6 and tiers[1].keepers == 0  # precondition: one full, one empty
    for t in tiers:
        for sb, base in ((t.first_sb, t.first), (t.all_shots_sb, t.all_shots)):
            assert isinstance(sb, metrics.SplitHalf) and sb.keys == base.keys
            want = tuple(metrics.spearman_brown(x) for x in (base.r, base.low, base.high))
            assert (sb.r, sb.low, sb.high) == want
    assert tiers[0].first_sb.r is not None


# ---- R12: the StatsBomb raw default --------------------------------------------------------


def test_statsbomb_raw_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """R12: ``gsax_cards.STATSBOMB_RAW`` is ``statsbomb.DEFAULT_CACHE`` (this repo's cache, never
    the sibling path) and is ``main``'s ``--statsbomb-raw`` default; the Makefile sets
    ``STATSBOMB_RAW ?=`` and exports it."""
    from football_core.io import statsbomb

    assert gsax_cards.STATSBOMB_RAW == statsbomb.DEFAULT_CACHE
    seen: list[Any] = []

    def fake(provider: str, *, docs_dir: Any, cache: Any, raw: Any) -> Any:
        seen.append(raw)
        raise ValueError("stop")

    monkeypatch.setattr(gsax_cards, "run", fake)
    assert gsax_cards.main(["--provider", "statsbomb", "--no-cache"]) == 1
    assert seen == [statsbomb.DEFAULT_CACHE]
    make = (Path(__file__).parents[1] / "Makefile").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("STATSBOMB_RAW ?=") for line in make)
    assert any(line.split("#")[0].strip() == "export STATSBOMB_RAW" for line in make)


# ---- R13, R14: run_from ----------------------------------------------------------------------


def test_run_from_reads_checkout_and_chart_path(
    gsax_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13: ``run_from`` reads the lineups from ``<raw>/<OPEN_DATA_COMMIT>`` and never calls
    ``statsbomb.sync``. R14: the chart goes to ``CHART_PATH`` read when called; the result is a
    ``GsaxRun``, a ``GsaxFit`` with the cards, SVG and chart path; the card still names
    ``CHART_FILE`` (R20)."""
    import matplotlib.image as mpimg
    from football_core.io import statsbomb

    g, f = gsax_fits["fits"]["statsbomb"], gsax_fits["xgot"]["statsbomb"]
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 5)
    logo = tmp_path / "logo.png"
    mpimg.imsave(logo, np.ones((4, 8, 4)))
    seen: list[Path] = []

    def names(checkout: Path, match_ids: Any, *, player_ids: list[str]) -> pl.DataFrame:
        seen.append(Path(checkout))
        ids = sorted(set(player_ids))
        return pl.DataFrame({"player_id": ids, "player_name": [f"Keeper {i}" for i in ids],
                             "names": pl.Series([1] * len(ids), dtype=pl.Int64())})  # fmt: skip

    def no_sync(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("run_from must not sync")

    monkeypatch.setattr(gsax, "fit", lambda _f: g)
    monkeypatch.setattr(statsbomb, "sync", no_sync)
    monkeypatch.setattr(statsbomb, "player_names", names)
    monkeypatch.setattr(statsbomb, "logo_file", lambda *a, **k: logo)
    chart = tmp_path / "elsewhere" / "k.svg"
    monkeypatch.setattr(gsax_cards, "CHART_PATH", chart)
    raw = tmp_path / "raw"
    run = gsax_cards.run_from(f, docs_dir=tmp_path / "docs", raw=raw)
    assert seen == [raw / statsbomb.OPEN_DATA_COMMIT]
    assert isinstance(run, gsax_cards.GsaxRun) and isinstance(run, gsax.GsaxFit)
    assert run.chart_path == chart and chart.read_text(encoding="utf-8") == run.svg
    assert sorted(p.name for p in (tmp_path / "docs").iterdir()) == ["gk-model.md", "gsax.md"]
    assert (
        f"`{gsax_cards.CHART_FILE}`" in run.cards["gsax"] and "elsewhere" not in run.cards["gsax"]
    )


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_write_refuses_tracked_chart_path(tmp_path: Path) -> None:
    """R14: ``write`` refuses a chart path git would track (nothing written there) and writes
    one that the work tree ignores."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / ".gitignore").write_text("data/\n", encoding="utf-8")
    tracked = repo / "charts" / "k.svg"
    with pytest.raises(ValueError, match="would not be git-ignored"):
        gsax_cards.write({}, "<svg/>", docs_dir=tmp_path / "docs", chart_path=tracked)
    assert not tracked.exists()
    ignored = repo / "data" / "charts" / "k.svg"
    out = gsax_cards.write({}, "<svg/>", docs_dir=tmp_path / "docs", chart_path=ignored)
    assert out == ignored and ignored.read_text(encoding="utf-8") == "<svg/>"


# ---- R16, R18: coarse-grid table and the ruled-out reconciliation row on the cards ----------


def _coarse(f: Any) -> Any:
    """``f`` with 2 coarse-grid comp-seasons (5 matches, 31 shots, 9 goals) in ``grid`` and in
    the reconciliation's ``coarse_grid`` row (C3 grown by the same)."""
    grid = pl.DataFrame(
        {
            "competition_name": ["Fixture Cup", "Alpha League"],
            "season_name": ["2018", "2018/2019"],
            "matches": [3, 2],
            "shots": [19, 12],
            "goals": [5, 4],
        },
        schema_overrides={k: pl.Int64 for k in ("matches", "shots", "goals")},
    )
    rows = tuple(("coarse_grid", 31, 9) if r == "coarse_grid" else (r, n, g)
                 for r, n, g in f.recon.rows)  # fmt: skip
    recon = dataclasses.replace(f.recon, rows=rows, c3_shots=f.recon.c3_shots + 31,
                                c3_goals=f.recon.c3_goals + 9)  # fmt: skip
    return dataclasses.replace(f, grid=grid, recon=recon)


def test_grid_and_ruled_rows_on_cards(gsax_fits: dict[str, Any]) -> None:
    """R16: the coarse-grid table (each comp-season row and the total) is on all three StatsBomb
    xGOT cards and the StatsBomb GSAx card. R18: the headline card's reconciliation shows the
    "Left out by a PM ruling" row with the ruled counts, absent when nothing is ruled out."""
    f = gsax_fits["xgot"]["statsbomb"]
    assert f.grid.height == 0 and f.recon.ruled == (0, 0)  # precondition: no coarse fixture row
    lines = ("| Fixture Cup | 2018 | 3 | 19 | 5 |", "| Alpha League | 2018/2019 | 2 | 12 | 4 |",
             "| **Total** | | 5 | 31 | 9 |")  # fmt: skip
    ruled = "| Left out by a PM ruling (coarse-grid matches) | 31 | 9 |"
    plain = xgot_cards.render(f)
    assert len(plain) == 3 and not any("Coarse-grid matches left out" in c for c in plain.values())
    assert not any("Left out by a PM ruling" in c for c in plain.values())
    f2 = _coarse(f)
    written = xgot_cards.render(f2)
    for name, card in written.items():
        assert all(line in card for line in lines), name
    assert sum(ruled in c for c in written.values()) == 1
    assert ruled in written[f2.head.spec.name]
    g = dataclasses.replace(gsax_fits["fits"]["statsbomb"], xgot=f2)
    table = gsax.keeper_rows(g)
    names = pl.DataFrame({"player_id": table["gk_id"], "player_name": table["gk_id"],
                          "names": pl.Series([1] * table.height, dtype=pl.Int64())})  # fmt: skip
    card = gsax_cards.render(g, names=names)["gsax"]
    assert all(line in card for line in lines)
    assert "- **Coarse-grid matches:** 31 on-target shots" in card
    assert "Coarse-grid" not in gsax_cards.render(gsax_fits["fits"]["statsbomb"],
                                                  names=names)["gsax"]  # fmt: skip


# ---- R19, R20, R21: check printing, chart file name, leave-out note ---------------------------


def test_check_rows_print_floor(gsax_fits: dict[str, Any]) -> None:
    """R19: check rows 2-3 print a sum below ``PRINT_FLOOR`` as "< 1e-12" (thread-order noise),
    and one at or above it in scientific notation."""
    assert gsax.PRINT_FLOOR == 1e-12
    for p in ("statsbomb", "whoscored"):
        c = gsax_fits["fits"][p].checks
        assert c[1].measured == "max abs. difference < 1e-12", p
        assert c[2].measured.startswith("max < 1e-12; "), p
    assert gsax._tiny(math.nextafter(1e-12, 0.0)) == "< 1e-12"
    assert gsax._tiny(1e-12) == "1.0e-12" and gsax._tiny(3.4e-10) == "3.4e-10"


def test_model_card_loo_note(gsax_fits: dict[str, Any]) -> None:
    """R21: the gk-model card prints why a row's leave-out columns are n/a, with
    ``loo_missing``, once per such row; no note when every row has them."""
    g = gsax_fits["fits"]["whoscored"]
    note = "Leave-out columns of *"
    plain = dataclasses.replace(
        g,
        next_season=dataclasses.replace(g.next_season, loo_missing=0),
        next_career=dataclasses.replace(g.next_career, loo_missing=0),
    )
    assert note not in gsax_cards.render(plain, names=None)["ws-gk-model"]
    two = dataclasses.replace(plain, next_career=dataclasses.replace(g.next_career, loo_missing=2))
    card = gsax_cards.render(two, names=None)["ws-gk-model"]
    assert card.count(note) == 1
    assert ("Leave-out columns of *Career effect (the literal formula)*: n/a, because 2 test "
            "season(s) have no shot faced by an unpaired keeper") in card  # fmt: skip
