"""GSAx and keeper-model cards, the README block, the StatsBomb keeper chart and Rule 1 (spec
``runs/MS6/gsax`` §5, §7; PM rulings E3 (b): the chart is git-ignored and linked by its make
target).

Cards render from the session ``gsax_fits`` (``conftest``). Each fixture keeper plays one match
(about 10 shots), so the StatsBomb table tests lower ``gsax.TABLE_MIN_SHOTS`` (read when
``keeper_rows`` is called) to get table rows. WhoScored cards carry counts and metrics only: no
fixture id (``xgot_cases.ws_sentinels()``) may appear in them or in the README block.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xgot_cases as xg

from football_shot_models import metrics

TABLE_MIN = 5


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _cards() -> Any:
    from football_shot_models import gsax_cards

    return gsax_cards


def _names(table: pl.DataFrame, conflicts: int = 2) -> pl.DataFrame:
    ids = sorted(table["gk_id"].to_list())
    return pl.DataFrame(
        {
            "player_id": ids,
            "player_name": [f"Fixture Keeper {chr(65 + i % 26)}{i:02d}" for i in range(len(ids))],
            "names": pl.Series([2 if i < conflicts else 1 for i in range(len(ids))],
                               dtype=pl.Int64()),
        }
    )  # fmt: skip


@pytest.fixture
def sb(gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The StatsBomb fit with ``TABLE_MIN_SHOTS`` lowered, its table rows and a names frame."""
    gsax = _gsax()
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", TABLE_MIN)
    g = gsax_fits["fits"]["statsbomb"]
    table = gsax.keeper_rows(g)
    assert table.height >= 3  # precondition
    return {"g": g, "table": table, "names": _names(table)}


@pytest.fixture
def logo(tmp_path: Path) -> Path:
    """A synthetic two-colour PNG (never the StatsBomb logo)."""
    import matplotlib.image as mpimg

    img = np.zeros((10, 60, 4))
    img[:, :30] = (0.8, 0.1, 0.1, 1.0)
    img[:, 30:] = (0.1, 0.1, 0.8, 1.0)
    path = tmp_path / "logo.png"
    mpimg.imsave(path, img)
    return path


def test_render_statsbomb_table_and_chart_link(sb: dict[str, Any]) -> None:
    """``gsax`` and ``gk-model``: the use-and-limits sentence; the keeper table (name, family,
    open goals, shrunken GSAx, GSAx with penalties...) in ``keeper_rows`` order; the count of
    resolved name conflicts; the chart linked through ``make gsax``; one stability sentence per
    tier; the next-season sentence on the model card."""
    cards = _cards()
    assert dict(cards.CARD_PREFIX) == {"statsbomb": "", "whoscored": "ws-"}
    out = cards.render(sb["g"], names=sb["names"])
    assert set(out) == {"gsax", "gk-model"}
    card = out["gsax"]
    limits = "a record of shots faced, rated against the league, not a skill rating"
    assert limits in card.lower()  # a sentence may start with a capital
    assert "make gsax" in card
    names = dict(zip(sb["names"]["player_id"], sb["names"]["player_name"], strict=True))
    where = [card.find(names[gk]) for gk in sb["table"]["gk_id"].to_list()]
    assert -1 not in where and where == sorted(where)  # in keeper_rows order (GSAx descending)
    header = next(
        x for x in card.splitlines() if x.startswith("|") and "amily" in x and "hrunk" in x
    ).lower()
    for word in ("family", "open goal", "shrunk", "with penalties", "sd", "ci"):
        assert word in header, word
    for k in (2, 3):  # the count of names > 1 among the table keepers, not a constant
        text = cards.render(sb["g"], names=_names(sb["table"], conflicts=k))["gsax"]
        lines = [x for x in text.splitlines() if re.search("conflict|spelling", x, re.I)]
        assert any(re.search(rf"\b{k}\b", x) for x in lines), (k, lines)
    for tier in sb["g"].tiers:
        assert cards.stability_sentence(tier) in card, tier.tier
    assert cards.next_season_sentence(sb["g"].next_season) in out["gk-model"]
    assert "StatsBomb" in card


def test_render_whoscored_has_no_keeper(gsax_fits: dict[str, Any], sb: dict[str, Any]) -> None:
    """WhoScored: ``ws-gsax`` and ``ws-gk-model``, no fixture id in them or in the README block
    (both providers); names given for WhoScored raise; the block prints no StatsBomb keeper id
    or name either."""
    cards = _cards()
    g = gsax_fits["fits"]["whoscored"]
    out = cards.render(g, names=None)
    assert set(out) == {"ws-gsax", "ws-gk-model"}
    block = cards.summary_block([sb["g"], g])
    sentinels = xg.ws_sentinels()
    for name, text in (*out.items(), ("block", block)):
        assert re.search(r"ws:\d", text) is None, name
        found = [s for s in sentinels if s in text]
        assert found == [], (name, found[:3])
    assert "sb:g" not in block
    assert not any(n in block for n in sb["names"]["player_name"].to_list())
    with pytest.raises(ValueError):
        cards.render(g, names=sb["names"])


def test_render_statsbomb_needs_names(sb: dict[str, Any]) -> None:
    """StatsBomb with ``names=None``, or names missing a table keeper, raises ``ValueError``."""
    cards = _cards()
    with pytest.raises(ValueError):
        cards.render(sb["g"], names=None)
    missing = sb["names"].filter(pl.col("player_id") != sb["table"]["gk_id"][0])
    with pytest.raises(ValueError):
        cards.render(sb["g"], names=missing)


def test_chart_logo_and_provider(sb: dict[str, Any], logo: Path, tmp_path: Path) -> None:
    """``chart`` returns the same SVG bytes twice, carrying the logo (``viz.LOGO_GID``); a ``ws:``
    row, a keeper missing from ``names`` and a missing logo raise; 0 rows still render."""
    from football_core import viz

    cards = _cards()
    svg = cards.chart(sb["table"], names=sb["names"], logo=logo)
    assert isinstance(svg, str) and "<svg" in svg and viz.LOGO_GID in svg
    assert cards.chart(sb["table"], names=sb["names"], logo=logo) == svg
    ws = sb["table"].with_columns(
        gk_id=pl.when(pl.int_range(pl.len()) == 1).then(pl.lit("ws:860101"))
        .otherwise(pl.col("gk_id"))
    )  # fmt: skip
    with pytest.raises(ValueError):
        cards.chart(ws, names=sb["names"], logo=logo)
    gone = sb["names"].filter(pl.col("player_id") != sb["table"]["gk_id"][1])
    with pytest.raises(ValueError):
        cards.chart(sb["table"], names=gone, logo=logo)
    with pytest.raises((FileNotFoundError, ValueError)):
        cards.chart(sb["table"], names=sb["names"], logo=tmp_path / "no-logo.png")
    empty = cards.chart(sb["table"].clear(), names=sb["names"], logo=logo)
    assert "<svg" in empty and viz.LOGO_GID in empty and empty != svg


def _like(proto: Any, r: float | None, low: float | None, high: float | None) -> Any:
    """A Spearman-Brown triple in the type ``Tier`` uses (a ``SplitHalf`` or a 3-tuple)."""
    if hasattr(proto, "r"):
        return dataclasses.replace(proto, r=r, low=low, high=high)
    return (r, low, high)


def _tier(gsax_fits: dict[str, Any], tier: int, keepers: int, first: tuple[Any, ...],
          all_shots: tuple[Any, ...]) -> Any:  # fmt: skip
    """A ``Tier`` with the given split-half r / low / high for ``first`` and ``all_shots``; the
    ``_sb`` fields are their Spearman-Brown step-ups."""
    proto = gsax_fits["fits"]["statsbomb"].tiers[0]
    sb = lambda t: tuple(metrics.spearman_brown(x) for x in t)  # noqa: E731
    k = keepers if first[0] is not None else min(keepers, 3)  # SplitHalf.keys: keys used
    return dataclasses.replace(
        proto, tier=tier, keepers=keepers,
        first=metrics.SplitHalf(k, *first), first_sb=_like(proto.first_sb, *sb(first)),
        all_shots=metrics.SplitHalf(k, *all_shots),
        all_shots_sb=_like(proto.all_shots_sb, *sb(all_shots)),
    )  # fmt: skip


def test_stability_sentence(gsax_fits: dict[str, Any]) -> None:
    """Five branches: empty tier; r None; fewer than ``MIN_TIER_KEEPERS`` (9 keepers with both
    CI lows > 0 included, and 29 at the boundary); 30 keepers with both Spearman-Brown CI lows
    > 0; 30 keepers with only ``first_sb`` low > 0 (or a low exactly 0) -> noise. The positive
    sentence appears in no other branch."""
    cards = _cards()
    none = (None, None, None)
    good, weak = (0.6, 0.3, 0.8), (0.4, -0.05, 0.7)
    s = lambda *a: cards.stability_sentence(_tier(gsax_fits, *a)).lower()  # noqa: E731
    # run6-ms6 (architect nit): the tiers pool each keeper's shots across seasons
    repeat = ("keeper records are repeatable beyond chance at 100 shots; this measures "
              "the keeper and the defence in front of him, not the keeper alone")  # fmt: skip
    noise = "not distinguishable from noise at 100 shots"
    assert "no keeper reaches 500 shots (empty tier)" in s(500, 0, none, none)
    assert "3 keepers reach 250 shots: too few to estimate" in s(250, 3, none, none)
    assert "9 keepers reach 250 shots: too few to judge" in s(250, 9, good, good)
    assert "29 keepers reach 100 shots: too few to judge" in s(100, 29, good, good)
    assert repeat in s(100, 30, good, good)
    assert noise in s(100, 30, good, weak)
    assert noise in s(100, 30, weak, good)
    zero = (0.5, 0.0, 0.8)  # Spearman-Brown keeps 0 at 0: "> 0" is strict
    assert noise in s(100, 30, good, zero)
    outputs = [s(500, 0, none, none), s(250, 3, none, none), s(250, 9, good, good),
               s(100, 30, good, weak), s(100, 30, good, zero)]  # fmt: skip
    assert not any("repeatable" in x for x in outputs)


def test_next_season_sentence(gsax_fits: dict[str, Any]) -> None:
    """Untestable -> the reason with its counts; beats raw (``gain.low > 0``) with and without a
    gain over xGOT alone; not better than raw (``gain.low`` 0 included)."""
    cards = _cards()
    base = gsax_fits["fits"]["statsbomb"].next_season
    gain = lambda low: metrics.GainCI(0.02, low, 0.04, metrics.N_BOOT, 40)  # noqa: E731
    ns = lambda **kw: dataclasses.replace(base, **kw)  # noqa: E731
    say = lambda **kw: cards.next_season_sentence(ns(**kw)).lower()  # noqa: E731
    reason = "fewer than 2 test matches (1 match, 3 pairs)"
    assert reason in say(untestable=reason, gain=None, gain_none=None)
    beats = ("shrunken estimates predict next season better than raw GSAx, because raw GSAx "
             "overfits small samples").lower()  # fmt: skip
    alone = "and predict next season no better than xGOT alone (the league average)".lower()
    not_better = "shrunken estimates do not predict next season better than raw GSAx".lower()
    text = say(untestable=None, gain=gain(0.01), gain_none=gain(0.001))
    # run6-ms6 (football 3): a gain is an improvement, never "add ... log loss"
    assert beats in text and "and improve log loss by 0.02000 over xgot alone" in text
    assert alone not in text and "add" not in text
    text = say(untestable=None, gain=gain(0.01), gain_none=gain(0.0))
    assert beats in text and alone in text
    for low in (0.0, -0.01):
        text = say(untestable=None, gain=gain(low), gain_none=gain(-0.001))
        assert not_better in text and beats not in text


def test_render_pure(gsax_fits: dict[str, Any], sb: dict[str, Any], tmp_path: Path,
                     monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """``render`` and ``summary_block`` give the same text twice and write nothing."""
    cards = _cards()
    monkeypatch.chdir(tmp_path)
    g_ws = gsax_fits["fits"]["whoscored"]
    assert cards.render(sb["g"], names=sb["names"]) == cards.render(sb["g"], names=sb["names"])
    assert cards.render(g_ws, names=None) == cards.render(g_ws, names=None)
    assert cards.summary_block([sb["g"], g_ws]) == cards.summary_block([sb["g"], g_ws])
    assert list(tmp_path.iterdir()) == []


def test_main_errors(
    gsax_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    """A bad provider exits non-zero; a missing manifest and a missing ``--statsbomb-raw``
    exit 1 with ``fsm: error:``; never a Traceback; no card is written when the names cannot be
    read. The raw case reuses the fixture fit and blocks the network (a missing checkout must
    never trigger a download in a test)."""
    from football_core.io import statsbomb

    from football_shot_models import evaluation, xgot

    cards = _cards()
    monkeypatch.setenv("FSM_CACHE", "off")
    try:
        code = cards.main(["--provider", "nope"])
    except SystemExit as e:
        code = e.code
    assert code not in (0, None)
    assert "Traceback" not in capsys.readouterr().err
    missing = {p: tmp_path / "manifests" / f"{p}.json" for p in ("statsbomb", "whoscored")}
    with monkeypatch.context() as mp:
        mp.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, missing, mp))
        args = ["--provider", "whoscored", "--no-cache", "--docs-dir", str(tmp_path / "d")]
        assert cards.main(args) == 1
    err = capsys.readouterr().err
    assert err.startswith("fsm: error: ") and "Traceback" not in err

    def offline(url: str) -> bytes:
        raise OSError(f"network blocked in tests: {url}")

    f = gsax_fits["xgot"]["statsbomb"]
    monkeypatch.setattr(statsbomb, "_http_get", offline)
    monkeypatch.setattr(xgot, "fit", lambda provider, *, cache: f)
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    raw = tmp_path / "no-raw"
    args = ["--provider", "statsbomb", "--no-cache", "--docs-dir", str(tmp_path / "docs"),
            "--statsbomb-raw", str(raw)]  # fmt: skip
    assert cards.main(args) == 1
    err = capsys.readouterr().err
    assert err.startswith("fsm: error: ") and "Traceback" not in err
    assert not raw.exists()  # nothing created at the missing checkout
    assert not (tmp_path / "docs").exists() or list((tmp_path / "docs").iterdir()) == []


# ---- result-audit fixes (MS6 gsax): band table, team-selected flag, shots per δ ----------------


def test_cards_band_section(gsax_fits: dict[str, Any], sb: dict[str, Any],
                            monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """Both GSAx cards carry the band table of the season-out q (one row per
    ``placement_bands`` row) and its limitation with the count of failing judged bands; only
    the WhoScored card states the swapped fit's rows, share and training rows (counted here);
    with the minimum at 0 every failing band is named in the limitation."""
    from football_shot_models import xgot

    gsax, cards = _gsax(), _cards()
    for p, names in (("statsbomb", sb["names"]), ("whoscored", None)):
        g = gsax_fits["fits"][p]
        card = cards.render(g, names=names)[f"{cards.CARD_PREFIX[p]}gsax"]
        lines = card.splitlines()
        head = lines.index("### Calibration of the rated q by placement band")
        start = next(i for i in range(head, len(lines)) if lines[i].startswith("| Feature |"))
        body = list(itertools.takewhile(lambda x: x.startswith("|"), lines[start + 2 :]))
        t = gsax.placement_bands(g)
        assert len(body) == t.height, p
        bad = t.filter(pl.col("judged") & ~pl.col("within")).height
        judged = int(t["judged"].sum())
        assert f"Limitation: {bad:,} of {judged:,} judged bands" in card, p
        src = g.rated["source"].cast(pl.String)
        swap, held = int((src == "swapped_fit").sum()), int((src == "holdout_fit").sum())
        if p == "whoscored":
            assert swap > 0
            assert f"`swapped_fit` rows ({swap:,}, {swap / g.rated.height:.0%}" in card
            assert f"test season's {held:,} rated rows" in card
        else:
            assert swap == 0 and "Swapped fit:" not in card
    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 0)
    g = gsax_fits["fits"]["whoscored"]
    t = gsax.placement_bands(g).filter(pl.col("judged") & ~pl.col("within"))
    assert t.height  # precondition: the fixture has a failing band once every band is judged
    card = cards.render(g, names=None)["ws-gsax"]
    for r in t.iter_rows(named=True):
        assert f"{r['feature']} {r['band']} ({r['source']}): {r['shots']:,} shots" in card


def test_team_selected_flag(sb: dict[str, Any], logo: Path) -> None:
    """Team-selected keeper-seasons are flagged († after the family) in the StatsBomb table and
    the chart labels, with the limitation line giving the flagged count and the split by side
    (``team_selected_split``); without team-selected rows there is no flag and no line."""
    gsax, cards = _gsax(), _cards()
    g = sb["g"]
    assert not (g.keepers[gsax.WITHOUT]["family"] == gsax.TEAM_SELECTED).any()  # precondition
    plain = cards.render(g, names=sb["names"])["gsax"]
    assert cards.FLAG not in plain and "Limitation (" not in plain
    season = sb["table"].select(gsax.SEASON_KEY).row(0)
    hit = (pl.col("competition_id") == season[0]) & (pl.col("season_id") == season[1])
    fam = pl.when(hit).then(pl.lit(gsax.TEAM_SELECTED)).otherwise(pl.col("family"))
    keepers = {k: v.with_columns(family=fam.cast(v.schema["family"]))
               for k, v in g.keepers.items()}  # fmt: skip
    g2 = dataclasses.replace(g, keepers=keepers)
    table = gsax.keeper_rows(g2)
    flagged = table.filter(pl.col("family") == gsax.TEAM_SELECTED)
    assert 0 < flagged.height < table.height  # precondition
    card = cards.render(g2, names=sb["names"])["gsax"]
    names = dict(zip(sb["names"]["player_id"], sb["names"]["player_name"], strict=True))
    rows = {names[r["gk_id"]]: r["family"] for r in table.iter_rows(named=True)}
    for line in card.splitlines():
        hits = [n for n in rows if f"| {n} |" in line]
        if hits:
            team = rows[hits[0]] == gsax.TEAM_SELECTED
            assert (f"{gsax.TEAM_SELECTED} {cards.FLAG} |" in line) == team, line
    split = gsax.team_selected_split(g2.keepers[gsax.WITHOUT])
    assert split.height
    assert f"{flagged.height:,} of the {table.height:,} keeper-seasons" in card
    for r in split.iter_rows(named=True):
        assert f"{r['side']}: {r['keeper_seasons']:,} keeper-seasons, {r['shots']:,} shots" in card
    svg = cards.chart(table, names=sb["names"], logo=logo)
    assert cards.FLAG in svg
    assert cards.FLAG not in cards.chart(sb["table"], names=sb["names"], logo=logo)


def test_rating_lines_shots_per_delta(gsax_fits: dict[str, Any], sb: dict[str, Any]) -> None:
    """Each kind's rating line states the shots behind its δ (min and median over the
    competition-seasons that are not small), counted here from the rated rows."""
    gsax, cards = _gsax(), _cards()
    for p, names in (("statsbomb", sb["names"]), ("whoscored", None)):
        g = gsax_fits["fits"][p]
        card = cards.render(g, names=names)[f"{cards.CARD_PREFIX[p]}gsax"]
        for kind in gsax.RATED_KINDS:
            n = (
                g.rated.filter((pl.col("kind").cast(pl.String) == kind) & ~pl.col("small"))
                .group_by(*gsax.SEASON_KEY).len()["len"].to_numpy()
            )  # fmt: skip
            line = next(
                x for x in card.splitlines() if x.startswith(f"- {kind.replace('_', ' ')}:")
            )
            want = f"min {n.min():,}, median {float(np.median(n)):,.1f}" if n.size else "none"
            assert line.endswith(f"shots per δ: {want}"), (p, kind, line)


def test_distribution_with_and_without_penalties(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release (gate, criterion "reported with and without penalties" on both providers): the
    WhoScored GSAx card's distribution prints both variants over the same keeper-seasons (those
    with at least ``TABLE_MIN_SHOTS`` shots without penalties); each row's quantiles are
    recomputed here as Σ (q − goal) per keeper-season straight from the rated rows."""
    g = gsax_fits["fits"]["whoscored"]
    gs = _gsax()
    rated = g.rated.with_columns(d=pl.col("q") - pl.col("is_goal").cast(pl.Float64))
    open_play = rated.filter(pl.col("kind").cast(pl.String) == "open_play")
    shots = open_play.group_by(gs.KEEPER_SEASON).len("n")
    least = int(shots["n"].median())  # a minimum that keeps some keeper-seasons and drops others
    kept = shots.filter(pl.col("n") >= least).select(gs.KEEPER_SEASON)
    assert 0 < kept.height < shots.height
    monkeypatch.setattr(gs, "TABLE_MIN_SHOTS", least)
    card = _cards().render(g, names=None)["ws-gsax"]
    assert f"{kept.height:,} keeper-seasons; no row per keeper" in card
    assert "| GSAx | Min | 10% | 25% | Median | 75% | 90% | Max |" in card
    sums: dict[str, pl.Series] = {}
    for name, frame in (("without penalties", open_play), ("with penalties", rated)):
        per = kept.join(
            frame.group_by(gs.KEEPER_SEASON).agg(pl.col("d").sum()), on=gs.KEEPER_SEASON
        )
        vals = [per["d"].quantile(q, "nearest") for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)]
        row = f"| {name} | " + " | ".join(f"{v:+.1f}" for v in vals) + " |"
        assert row in card, (name, row)
        sums[name] = per.sort(gs.KEEPER_SEASON)["d"]
    # precondition for the mutant "both rows read one variant": the kept keepers faced penalties
    assert not sums["with penalties"].equals(sums["without penalties"])


def test_next_season_sentence_without_team_selected(gsax_fits: dict[str, Any]) -> None:
    """Run6-ms6 (architect A4, football 3; the small-margin rule): with the result without
    team-selected competition-seasons, the sentence adds its pairs, both gains and whether each
    interval is above 0, each from its own branch (low exactly 0 is not above); untestable gives
    its reason; the gk-model card and the README block carry it."""
    gsax, cards = _gsax(), _cards()
    g = gsax_fits["fits"]["statsbomb"]
    gain = lambda low: metrics.GainCI(0.02, low, 0.04, metrics.N_BOOT, 40)  # noqa: E731
    ns = dataclasses.replace(g.next_season, untestable=None, gain=gain(0.01),
                             gain_none=gain(0.001))  # fmt: skip
    base = cards.next_season_sentence(ns)
    no_team = dataclasses.replace(ns, pairs=22, gain=gain(0.002), gain_none=gain(-0.0005))
    text = cards.next_season_sentence(ns, no_team)
    assert text.startswith(base + " ")
    assert (f"Without {gsax.TEAM_SELECTED} competition-seasons (22 pairs) the gain over raw is "
            f"{gsax.gain_text(no_team.gain)}, its interval above 0, and over xGOT alone "
            f"{gsax.gain_text(no_team.gain_none)}, its interval not above 0.") in text  # fmt: skip
    zero = cards.next_season_sentence(ns, dataclasses.replace(no_team, gain=gain(0.0)))
    assert "the gain over raw is +0.02000 [+0.00000, +0.04000], its interval not above 0" in zero
    why = cards.next_season_sentence(ns, dataclasses.replace(no_team, untestable="no pair (0)"))
    assert (
        why == f"{base} Without {gsax.TEAM_SELECTED} competition-seasons: untestable (no pair (0))."
    )
    ws = gsax_fits["fits"]["whoscored"]
    g2 = dataclasses.replace(ws, next_season=ns, next_no_team=no_team)
    assert text in cards.render(g2, names=None)["ws-gk-model"]
    assert text in cards.summary_block([g2])


def test_gsax_with_penalties_column(sb: dict[str, Any]) -> None:
    """Run6-ms6 (QA 3; criterion "reported with and without penalties", StatsBomb half): the
    table's ``gsax_with`` is the "with penalties" keeper table's GSAx of the same keeper-season,
    recomputed as Σ (q − goal) over its open-play and penalty rows, and the card's last cell
    prints it. Precondition: a table keeper faced a penalty, so the two columns differ."""
    gsax, cards = _gsax(), _cards()
    g, table = sb["g"], sb["table"]
    rows = g.rated.with_columns(d=pl.col("q") - pl.col("is_goal").cast(pl.Float64))
    want = rows.group_by(gsax.KEEPER_SEASON).agg(pl.col("d").sum().alias("want"))
    got = table.join(want, on=gsax.KEEPER_SEASON, how="left")
    assert np.allclose(got["gsax_with"].to_numpy(), got["want"].to_numpy(), rtol=0, atol=1e-9)
    assert not np.allclose(got["gsax_with"].to_numpy(), got["gsax"].to_numpy())  # precondition
    keepers = gsax.keeper_table(gsax.variant(g.rated, gsax.WITH))
    joined = table.join(keepers.select(*gsax.KEEPER_SEASON, k="gsax"), on=gsax.KEEPER_SEASON,
                        how="left", maintain_order="left")  # fmt: skip
    assert joined["k"].equals(table["gsax_with"], check_names=False)
    card = cards.render(g, names=sb["names"])["gsax"]
    names = dict(zip(sb["names"]["player_id"], sb["names"]["player_name"], strict=True))
    for r in table.iter_rows(named=True):
        line = next(x for x in card.splitlines() if f"| {names[r['gk_id']]} |" in x)
        assert line.endswith(f"| {r['gsax_with']:+.1f} |"), line


def test_keeper_rows_minimum_inclusive(sb: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Run6-ms6 (QA 8): ``keeper_rows`` keeps a keeper-season with exactly ``TABLE_MIN_SHOTS``
    shots (``>=``, not ``>``)."""
    gsax = _gsax()
    g = sb["g"]
    shots = g.keepers[gsax.WITHOUT]["shots"]
    edge = int(shots.max())
    monkeypatch.setattr(gsax, "TABLE_MIN_SHOTS", edge)
    table = gsax.keeper_rows(g)
    assert table.height == int((shots == edge).sum()) >= 1
    assert (table["shots"] == edge).all()


def test_cards_text_from_constants(
    gsax_fits: dict[str, Any], sb: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run6-ms6 (architect A8, nit): the stability intervals print ``metrics.LEVEL`` (read when
    called); the gk-model card types no timing and cites no run folder; it prints check row
    ``NEXT_SEASON_CHECK``."""
    gsax, cards = _gsax(), _cards()
    g = gsax_fits["fits"]["whoscored"]
    monkeypatch.setattr(metrics, "LEVEL", 0.8)
    out = cards.render(g, names=None)
    assert (
        "First T shots: r [80% CI]" in out["ws-gsax"]
        and "80% intervals by Fisher z" in (out["ws-gsax"])
    )
    assert "95% CI" not in out["ws-gsax"]
    model = out["ws-gk-model"]
    assert "run folder" not in model and "second" not in model
    five = g.checks[gsax.NEXT_SEASON_CHECK - 1]
    assert f"| {five.target} | {five.measured} | {five.result} |" in model
    assert "next season" in five.target


def test_team_lines_raw_and_share(sb: dict[str, Any]) -> None:
    """Run6-ms6 (football 4): the team-selected limitation prints raw (before δ) beside rated
    per side and the selected side's share of the team-selected shots, computed from
    ``team_selected_split``; it names both channels (no finisher term; δ absorbs the selected
    keeper's record) instead of a credit for weaker finishers."""
    gsax, cards = _gsax(), _cards()
    g = sb["g"]
    season = sb["table"].select(gsax.SEASON_KEY).row(0)
    hit = (pl.col("competition_id") == season[0]) & (pl.col("season_id") == season[1])
    fam = pl.when(hit).then(pl.lit(gsax.TEAM_SELECTED)).otherwise(pl.col("family"))
    keepers = {k: v.with_columns(family=fam.cast(v.schema["family"])) for k, v in g.keepers.items()}
    g2 = dataclasses.replace(g, keepers=keepers)
    card = cards.render(g2, names=sb["names"])["gsax"]
    split = gsax.team_selected_split(g2.keepers[gsax.WITHOUT])
    assert split.height
    from football_shot_models.cards import signed

    for r in split.iter_rows(named=True):
        assert (f"{r['side']}: {r['keeper_seasons']:,} keeper-seasons, {r['shots']:,} shots, GSAx "
                f"raw {signed(r['gsax_raw'])}, rated {signed(r['gsax'])}") in card  # fmt: skip
    sel = split.filter(pl.col("side") == "selected side")["shots"].sum()
    share = sel / split["shots"].sum()
    assert f"face {share:.0%} of the team-selected rated shots" in card
    assert "weaker finishers" not in card and "absorbs part of their own record" in card
