"""Ratify tests for the MS6 gsax build decisions (``runs/MS6/gsax/build.md`` R1-R21) that had no
dedicated test: one test per decision, each failing if the decision is reverted.

Synthetic frames only (``gsax_cases``, ``xgot_cases``); the card tests reuse the session
``xgot_fits`` / ``gsax_fits`` fixtures and change one field with ``dataclasses.replace``.
"""

from __future__ import annotations

import dataclasses
import math
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
    rated = gsax.rate(f, small=gsax.SmallLeague(2, 50), zero_families=())
    assert rated.filter(b18)["small"].all() and (rated.filter(b18)["delta"] == 0.0).all()
    assert np.allclose(rated.filter(b18)["q"].to_numpy(), rated.filter(b18)["xgot"].to_numpy())
    assert not rated.filter(~b18)["small"].any()  # A18: 3 keepers, 60 shots
    free = gsax.rate(f, small=None, zero_families=()).filter(b18)
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
    rated = gsax.rate(_next_frame(), small=None, zero_families=())
    season = gsax.next_season(rated, by=gsax.KEEPER_SEASON)
    career = gsax.next_season(rated, by=gsax.CAREER)
    assert (season.pairs + season.left_out, season.left_out) == (7, 3)
    assert (career.pairs + career.left_out, career.left_out) == (7, 2)
    with pytest.raises(TypeError):
        gsax.next_season(rated, by="gk_id")


def test_next_season_loo_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """R8: ``loo_missing`` counts the test seasons with no unpaired row (B19 without g06) or a
    non-finite δ without the paired keepers (A20 whose unpaired keepers concede 0); 0 when
    every test season has one. Amended in run6-ms6 (football 5): such a season leaves the
    leave-out gains, which are scored on the other test seasons' rows (``loo_shots`` /
    ``loo_pairs`` count what left), and are None only when fewer than 2 test matches remain.
    Amended in PR1 (§3.4): a remainder failing E2 is rated at 0 (``loo_small``) before its
    shift is tried, so the non-finite branch is shown with ``SMALL_LEAGUE`` None."""
    full = gsax.next_season(
        gsax.rate(_next_frame(), small=None, zero_families=()), by=gsax.KEEPER_SEASON
    )
    assert full.loo_missing == 0 and full.gain_loo is not None
    assert (full.loo_shots, full.loo_pairs) == (0, 0)
    one = _next_frame(b19_unpaired=False)
    ns = gsax.next_season(gsax.rate(one, small=None, zero_families=()), by=gsax.KEEPER_SEASON)
    assert ns.untestable is None and ns.gain is not None
    assert ns.loo_missing == 1 and (ns.loo_shots, ns.loo_pairs) == (16, 1)  # g04 B19
    assert ns.gain_loo is not None and ns.gain_none_loo is not None
    assert ns.gain_loo.groups == full.gain_loo.groups - 2  # B19's 2 matches left out
    two = _next_frame(b19_unpaired=False, a20_unpaired_goals=False)
    rated_two = gsax.rate(two, small=None, zero_families=())
    # PR1 §3.4: under SMALL_LEAGUE the remainders of A19 (g08, g11: 2 keepers) and A20 (g05,
    # g10, g11: 3 keepers, 48 rows) fail E2 -> δ_loo 0; B19 (empty) is the one NaN
    ns = gsax.next_season(rated_two, by=gsax.KEEPER_SEASON)
    assert (ns.loo_missing, ns.loo_small) == (1, 2)
    monkeypatch.setattr(gsax, "SMALL_LEAGUE", None)  # step 4: a non-finite δ -> NaN
    ns = gsax.next_season(rated_two, by=gsax.KEEPER_SEASON)
    a20 = ns.test_shots - ns.loo_shots
    assert ns.loo_missing == 2 and ns.loo_shots > 16 and a20 > 0
    assert ns.gain_loo is not None and ns.gain_loo.groups < ns.test_matches


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


def test_statsbomb_raw_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """R12: ``gsax_cards.STATSBOMB_RAW`` is ``statsbomb.DEFAULT_CACHE`` (this repo's cache, never
    the sibling path) and is ``main``'s ``--statsbomb-raw`` default; the Makefile sets
    ``STATSBOMB_RAW ?=`` and exports it. Amended in run6-ms6 (architect A9): ``main`` refuses a
    StatsBomb run before any fit when the cache has no ``<commit>/tree.json``, so the default
    reaches ``run`` from a working directory whose default cache holds one."""
    from football_core.io import statsbomb

    assert gsax_cards.STATSBOMB_RAW == statsbomb.DEFAULT_CACHE
    seen: list[Any] = []

    def fake(provider: str, *, docs_dir: Any, cache: Any, raw: Any) -> Any:
        seen.append(raw)
        raise ValueError("stop")

    monkeypatch.setattr(gsax_cards, "run", fake)
    monkeypatch.chdir(tmp_path)
    assert gsax_cards.main(["--provider", "statsbomb", "--no-cache"]) == 1
    assert seen == []  # no tree.json in the default cache: refused before run
    tree = tmp_path / statsbomb.DEFAULT_CACHE / statsbomb.OPEN_DATA_COMMIT / "tree.json"
    tree.parent.mkdir(parents=True)
    tree.write_text("{}", encoding="utf-8")
    assert gsax_cards.main(["--provider", "statsbomb", "--no-cache"]) == 1
    assert seen == [statsbomb.DEFAULT_CACHE]
    assert gsax_cards.main(["--provider", "whoscored", "--no-cache"]) == 1  # no tree needed
    assert len(seen) == 2
    make = (Path(__file__).parents[1] / "Makefile").read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("STATSBOMB_RAW ?=") for line in make)
    assert any(line.split("#")[0].strip() == "export STATSBOMB_RAW" for line in make)


# ---- R13, R14: run_from ----------------------------------------------------------------------


def test_run_from_reads_checkout_and_chart_path(
    gsax_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R13: ``run_from`` reads the lineups from ``<raw>/<OPEN_DATA_COMMIT>`` and never calls
    ``statsbomb.sync``. R14 (amended, CH1 keeper-model-charts §3.3): the charts go to
    ``CHART_DIR`` read when called; the result is a ``GsaxRun``, a ``GsaxFit`` with the cards
    and ``charts`` (file name → path); the card names ``CHART_FOLDER`` (never the moved folder)
    and carries the SHA-256 of each chart's bytes (run6-ms6 architect A2: the committed card
    checks the git-ignored charts)."""
    import hashlib

    import matplotlib.image as mpimg
    from football_core.io import statsbomb
    from test_gsax_cards import at_table_min

    f = gsax_fits["xgot"]["statsbomb"]
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 5)
    g = at_table_min(gsax_fits["fits"]["statsbomb"])
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
    folder = tmp_path / "elsewhere"
    monkeypatch.setattr(gsax_cards, "CHART_DIR", folder)
    raw = tmp_path / "raw"
    run = gsax_cards.run_from(f, docs_dir=tmp_path / "docs", raw=raw)
    assert seen == [raw / statsbomb.OPEN_DATA_COMMIT]
    assert isinstance(run, gsax_cards.GsaxRun) and isinstance(run, gsax.GsaxFit)
    assert {Path(p).parent for p in run.charts.values()} == {folder} and run.charts
    assert sorted(p.name for p in (tmp_path / "docs").iterdir()) == ["gk-model.md", "gsax.md"]
    card = run.cards["gsax"]
    assert f"`{gsax_cards.CHART_FOLDER}/`" in card and "elsewhere" not in card
    for name, path in run.charts.items():
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert f"`{name}` (SHA-256 `{digest}`)" in card
    assert (tmp_path / "docs" / "gsax.md").read_text(encoding="utf-8") == card


def test_write_cards_only(tmp_path: Path) -> None:
    """R14 (amended, CH1 §3.3): ``write`` writes the cards only (``docs_dir/<name>.md``, LF);
    the chart guard is ``write_charts``' (``test_write_charts_all_or_nothing``)."""
    gsax_cards.write({"gsax": "# x\n", "gk-model": "# y\n"}, docs_dir=tmp_path / "docs")
    assert sorted(p.name for p in (tmp_path / "docs").iterdir()) == ["gk-model.md", "gsax.md"]
    assert (tmp_path / "docs" / "gsax.md").read_bytes() == b"# x\n"


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


# ---- PR1 keeper-fixes build.md ratify items without a QA test (added at integration) ---------


def test_pr1_r4_two_families_raise() -> None:
    """PR1 R4: a competition-season with two ``family`` values makes ``rate`` raise (its
    ``zero_family`` flag would be ambiguous); one family per season rates."""
    f = gc.blocks(
        gc.block("sb:g01", "A18", goals=[True, False], xgot=0.3, matches=["sb:m1"]),
        gc.block("sb:g02", "A18", goals=[False, True], xgot=0.3, matches=["sb:m2"]),
    )
    assert gsax.rate(f, small=None, zero_families=gsax.ZERO_FAMILIES).height == 4  # precondition
    mixed = f.with_columns(family=pl.Series([gc.LEAGUE] * 3 + [gsax.TEAM_SELECTED],
                                            dtype=f.schema["family"]))  # fmt: skip
    with pytest.raises(ValueError, match="1 competition-season.*more than one family"):
        gsax.rate(mixed, small=None, zero_families=gsax.ZERO_FAMILIES)


def test_pr1_r5_model_no_team_none_when_all_team_selected(gsax_census: dict[str, Any]) -> None:
    """PR1 R5: with every open-play row team-selected, ``model_no_team`` is None (no keeper
    model on an empty frame) and the without-team test is untestable, not an error."""
    c = gsax_census
    team = pl.lit(gsax.TEAM_SELECTED).cast(c["pred"].schema["family"])
    g = gsax.assemble(c["f"], c["pred"].with_columns(family=team), stops=c["stops"],
                      trained=c["trained"])  # fmt: skip
    assert g.rated["zero_family"].all()  # precondition
    assert g.model_no_team is None
    assert g.next_no_team is not None and g.next_no_team.untestable is not None


def test_pr1_r6_next_season_worse_than_xgot_alone(gsax_fits: dict[str, Any]) -> None:
    """PR1 R6: a gain over xGOT alone whose interval lies below 0 reads "predict next season
    worse than xGOT alone", not "not distinguishable"; an interval holding 0 reads the latter."""
    base = gsax_fits["fits"]["statsbomb"].next_season
    ci = lambda lo, hi: metrics.GainCI((lo + hi) / 2, lo, hi, metrics.N_BOOT, 40)  # noqa: E731
    say = lambda none: gsax_cards.next_season_sentence(  # noqa: E731
        dataclasses.replace(base, untestable=None, gain=ci(0.01, 0.03), gain_none=none)
    )
    worse, alone = "predict next season worse than xGOT alone", "not distinguishable from xGOT"
    assert worse in say(ci(-0.02, -0.001)) and alone not in say(ci(-0.02, -0.001))
    assert alone in say(ci(-0.02, 0.0)) and worse not in say(ci(-0.02, 0.0))


def test_pr1_r9_stops_without_end_x(gsax_fits: dict[str, Any]) -> None:
    """PR1 R9: ``stops(f)`` raises ``ValueError`` naming ``end_x`` when the xGOT rows have no
    such column (not a polars error)."""
    f = gsax_fits["xgot"]["statsbomb"]
    assert gsax.stops(f) is not None  # precondition
    inp = dataclasses.replace(f.inp, on_held=f.inp.on_held.drop("end_x"))
    with pytest.raises(ValueError, match="no end_x column"):
        gsax.stops(dataclasses.replace(f, inp=inp))


# ---- PR1 keeper-fixes: the remaining build.md ratify items (QA, engineer-decision pass) ------


def _census_names(table: pl.DataFrame) -> pl.DataFrame:
    ids = sorted(table["gk_id"].to_list())
    return pl.DataFrame({"player_id": ids, "player_name": [f"Fixture Keeper {i:02d}"
                                                           for i in range(len(ids))],
                         "names": pl.Series([1] * len(ids), dtype=pl.Int64())})  # fmt: skip


def _with_noz(c: dict[str, Any]) -> Any:
    """The census fit with a z-free table whose xGOT differs from the headline's."""
    open_ = c["pred"].filter(pl.col("kind") == "open_play")
    moved = open_.with_columns(xgot=(pl.col("xgot") * 0.9).clip(0.01, 0.99))
    return dataclasses.replace(c["g"], noz=gsax.keeper_table(gsax.noz_rated(moved)))


def test_pr1_r1_noz_gate_apart_from_line_gate(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR1 R1 (§3.6, E2 (c)): ``fit`` runs one z-free season-out (``without=NOZ_FEATURES``) for
    a provider in ``NOZ_PROVIDERS``; both constants are read when called and gated apart from
    ``LINE_PROVIDERS``: the line gate off keeps the z-free refit, the z-free gate off keeps the
    line stops."""
    from football_shot_models import season_out

    f = gsax_fits["xgot"]["statsbomb"]
    seen: list[tuple[str, ...]] = []

    def fake(_f: Any, *, without: Any = ()) -> str:
        seen.append(tuple(without))
        return f"pred {tuple(without)}"

    monkeypatch.setattr(season_out, "season_out", fake)
    monkeypatch.setattr(season_out, "trained_seasons", lambda _f: "trained")
    monkeypatch.setattr(gsax, "assemble", lambda _f, pred, **kw: {"pred": pred, **kw})

    def run(**patch: Any) -> tuple[dict[str, Any], list[tuple[str, ...]]]:
        seen.clear()
        with pytest.MonkeyPatch.context() as mp:
            for k, v in patch.items():
                mp.setattr(gsax, k, v)
            return gsax.fit(f), list(seen)

    out, calls = run()
    assert calls == [(), ("mouth_z",)] and out["pred"] == "pred ()"
    assert out["noz"] == "pred ('mouth_z',)" and out["stops"] is not None
    out, calls = run(LINE_PROVIDERS=frozenset())
    assert out["stops"] is None and out["noz"] == "pred ('mouth_z',)"
    out, calls = run(NOZ_PROVIDERS=frozenset())
    assert calls == [()] and out["noz"] is None and out["stops"] is not None
    out, calls = run(NOZ_FEATURES=("angle_rad",))
    assert calls == [(), ("angle_rad",)]


def test_pr1_r2_noz_refit_reads_the_same_xg_input(gsax_fits: dict[str, Any]) -> None:
    """PR1 R2: the z-free refit fits the xG input under the headline season-out's own spec, on
    the same rows and folds (one cache identity: a hit in ``make gsax``; without a cache, as in
    the fixture and ``reproduce-cold``, StatsBomb fits it twice), then the headline once as
    ``-season-out-no-mouth_z``, and no penalty xGOT (the season-out one is fitted once)."""
    f = gsax_fits["xgot"]["statsbomb"]
    calls = gsax_fits["calls"]["statsbomb"]
    names = [c["spec"].name for c in calls]
    xg_name = f"{f.xg_input.spec.name}-season-out"
    assert names.count(xg_name) == 2, names
    a, b = (c for c in calls if c["spec"].name == xg_name)
    assert a["spec"] == b["spec"]
    for side in ("dev", "held"):
        assert a[side]["shot_id"].equals(b[side]["shot_id"])
    fa, fb = (np.concatenate([np.asarray(t) for pair in c["folds"] for t in pair])
              for c in (a, b))  # fmt: skip
    assert np.array_equal(fa, fb)
    assert [n for n in names if n.endswith("-no-mouth_z")] == [
        f"{f.head.spec.name}-season-out-no-mouth_z"
    ]
    assert names.count(f"{f.penalty.spec.name}-season-out") == 1
    assert not [n for n in names if n.startswith(f.penalty.spec.name) and "-no-" in n]


def test_pr1_r3_shift_summary_needs_the_line_sensitivity(gsax_census: dict[str, Any]) -> None:
    """PR1 R3: ``shift_summary(g, "gsax_noz")`` is None when ``g.line`` or ``g.line_seasons`` is
    None even with ``g.noz`` set (the far-save counts and the population come from the line)."""
    g = _with_noz(gsax_census)
    assert gsax.shift_summary(g, "gsax_noz") is not None  # precondition
    assert gsax.shift_summary(dataclasses.replace(g, line=None), "gsax_noz") is None
    assert gsax.shift_summary(dataclasses.replace(g, line_seasons=None), "gsax_noz") is None


def test_pr1_r7_row3_delta_zero_sum_both_kinds(gsax_census: dict[str, Any]) -> None:
    """PR1 R7 (§3.3): row 3's "Σ (q − goal) at δ = 0" sums open play and kicks of the δ = 0
    competition-seasons (team-selected; other small), recomputed here from the rated rows; the
    kicks move the team-selected sum (precondition)."""
    g = gsax_census["g"]
    gap = (pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum()
    team, other = pl.col("zero_family"), pl.col("small") & ~pl.col("zero_family")
    op = pl.col("kind").cast(pl.String) == "open_play"
    pen = pl.col("kind").cast(pl.String) == "penalty"
    r = g.rated
    assert r.filter(team & pen).height > 0  # precondition: kicks in a team-selected season
    bump = pl.when(team & pen).then(pl.col("q") + 0.25).otherwise(pl.col("q"))
    r2 = r.with_columns(q=bump)
    row = gsax.checks(dataclasses.replace(g, rated=r2))[2]
    t_all, t_op = (float(r2.filter(team & m).select(gap).item()) for m in (pl.lit(True), op))
    o_all = float(r2.filter(other).select(gap).item())
    assert f"{t_all:+.1f}" != f"{t_op:+.1f}"  # precondition: the kicks matter at print precision
    assert f"Σ (q − goal) at δ = 0: team-selected {t_all:+.1f}, other small {o_all:+.1f}" in (
        row.measured
    ), row.measured


def test_pr1_r8_penalty_zero_line_and_loo_small_always(
    gsax_census: dict[str, Any], gsax_fits: dict[str, Any]
) -> None:
    """PR1 R8 (§4): the rating lines end with "- penalty at δ = 0: K kicks of competition-seasons
    at δ = 0; P source pool(s) failing E2's minimum (N kicks)" (census: 7 kicks in 3 seasons;
    1 pool, ``holdout_fit``); the keeper-model card's leave-out δ line is printed with 0 / 0
    too."""
    g = gsax_census["g"]
    pens = g.rated.filter(pl.col("kind").cast(pl.String) == "penalty")
    k = pens.filter(pl.col("small") | pl.col("zero_family")).height
    pool = pens.filter(pl.col("pool_small"))
    assert (k, pool["group"].n_unique()) == (7, 1) and pool.height > 0  # census
    card = gsax_cards.render(g, names=_census_names(gsax.keeper_rows(g)))["gsax"]
    want = (f"- penalty at δ = 0: {k:,} kicks of competition-seasons at δ = 0; 1 source pool(s) "
            f"failing E2's minimum ({pool.height:,} kicks)")  # fmt: skip
    assert want in card.splitlines(), [x for x in card.splitlines() if "penalty at" in x]
    ws = gsax_fits["fits"]["whoscored"]
    zero = {n: dataclasses.replace(getattr(ws, n), loo_missing=0, loo_small=0)
            for n in ("next_season", "next_career")}  # fmt: skip
    model = gsax_cards.render(dataclasses.replace(ws, **zero), names=None)["ws-gk-model"]
    line = next(x for x in model.splitlines() if x.startswith("Leave-out δ per test season"))
    assert line.count("0 left out, 0 at δ = 0") == 2, line


def test_pr1_r10_keeper_rows_column_order(gsax_census: dict[str, Any]) -> None:
    """PR1 R10: ``keeper_rows`` ends ``gsax_shrunk, gsax_with, gsax_line, gsax_noz,
    zero_family`` (the chart seam's column order)."""
    cols = gsax.keeper_rows(_with_noz(gsax_census)).columns
    assert cols[-5:] == ["gsax_shrunk", "gsax_with", "gsax_line", "gsax_noz", "zero_family"]


def test_pr1_r11_chart_note(
    gsax_census: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR1 R11 (amended, CH1 keeper-model-charts §3.2): with a shown team-selected row the
    keeper chart flags it and its footer note reads "* Only the team's own matches in the data:
    rated against the model, not the league" (``keepers.FLAG``); without one there is no such
    note."""
    import charts_cases as cc
    import matplotlib.image as mpimg
    from football_core import viz

    logo = tmp_path / "logo.png"
    mpimg.imsave(logo, np.full((10, 60, 4), 0.5))
    from test_gsax_cards import at_table_min

    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 40)  # 4 rows, all shown: 2 team-selected
    g = at_table_min(gsax_census["g"])
    table = gsax.keeper_rows(g)
    team = pl.col("family").cast(pl.String) == gsax.TEAM_SELECTED
    assert 0 < table.filter(team).height < table.height <= 6  # precondition
    from football_shot_charts import keepers

    # the chart's flag (run8-ch1 item 17: "†" read as "+"); the card table keeps gsax_cards.FLAG
    note = (f"{keepers.FLAG} Only the team's own matches in the data: rated against the model, "
            "not the league")  # fmt: skip
    charts = gsax_cards.keeper_charts(g, names=_census_names(table), logo=logo)
    for c in charts.values():
        text = " ".join(t.get_text() for t in cc.by_gid(c, viz.NOTE_GID))
        assert note in cc.collapse(text)
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 120)  # sb:g01 A18 alone: not team-selected
    g = at_table_min(g)
    plain = gsax.keeper_rows(g)
    assert plain.height == 1 and not plain.filter(team).height  # precondition
    for c in gsax_cards.keeper_charts(g, names=_census_names(plain), logo=logo).values():
        assert "team-selected season" not in cc.all_text(c)


def test_pr1_population_season_without_line_rows_is_at_zero(
    gsax_census: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Audit A10 population: a keeper-season whose competition-season has no row in
    ``g.line_seasons`` counts as ``at_zero`` (left out, counted), not judged and not dropped;
    with no season at δ = 0, ``at_zero`` is 0 and ``at_zero_median_abs`` None. The card's count
    line names the rule ("team-selected, or E2-small there")."""
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", 20)
    g = _with_noz(gsax_census)
    base = gsax.shift_summary(g, "gsax_noz")
    hit = (pl.col("competition_id") == "sb:9") & (pl.col("season_id") == "sb:42")
    assert base is not None and g.line_seasons is not None
    assert g.line_seasons.filter(hit & ~pl.col("at_zero")).height == 1  # precondition: fitted
    n = gsax.keeper_rows(g).filter(hit).drop_nulls("gsax_noz").height
    assert n > 0  # precondition
    gone = gsax.shift_summary(dataclasses.replace(g, line_seasons=g.line_seasons.filter(~hit)),
                              "gsax_noz")  # fmt: skip
    assert gone is not None
    assert (gone.keeper_seasons, gone.at_zero) == (base.keeper_seasons - n, base.at_zero + n)
    none = dataclasses.replace(g, line_seasons=g.line_seasons.with_columns(at_zero=pl.lit(False)))
    for column in gsax.SHIFTS:
        s = gsax.shift_summary(none, column)
        assert s is not None and s.at_zero == 0 and s.at_zero_median_abs is None
    card = gsax_cards.render(g, names=_census_names(gsax.keeper_rows(g)))["gsax"]
    assert "δ = 0 on the subset (team-selected, or E2-small there; median |shift|" in card
