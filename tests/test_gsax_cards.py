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
from collections.abc import Mapping
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


def at_table_min(g: Any) -> Any:
    """``g`` with its first reliability tier moved to the (patched) ``TABLE_MIN_SHOTS``, so the
    keeper chart's caveat tier exists when a test lowers the threshold for the fixture."""
    gsax = _gsax()
    first = dataclasses.replace(g.tiers[0], tier=gsax.TABLE_MIN_SHOTS)
    return dataclasses.replace(g, tiers=(first, *g.tiers[1:]))


def _keeper_fit(gsax_fits: dict[str, Any], provider: str = "statsbomb") -> Any:
    """A ``keeper_cases`` fit (12 keeper-seasons of 150 shots, 2 team-selected) on the
    provider's session fit; WhoScored ids are ``ws:``."""
    import keeper_cases as kc

    specs = kc.TWELVE if provider == "statsbomb" else [
        (f"ws:86{s[0][4:]}", *s[1:]) for s in kc.TWELVE]  # fmt: skip
    return kc.fit_from(gsax_fits["fits"][provider], kc.rated(specs))


def test_keeper_charts_themes_logo_deterministic(
    gsax_fits: dict[str, Any], logo: Path, tmp_path: Path
) -> None:
    """``keeper_charts``: one canvas per ``viz.THEMES`` named ``gsax-keepers-{theme}.png``
    (``-no-team`` before the theme in drop mode); StatsBomb named, with the logo and the caveat
    of the ``TABLE_MIN_SHOTS`` tier as subtitle; WhoScored unnamed, no logo; the same PNG bytes
    twice (deterministic)."""
    import charts_cases as cc
    import keeper_cases as kc
    from football_core import viz

    gsax, cards = _gsax(), _cards()
    g = _keeper_fit(gsax_fits)
    names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    charts = cards.keeper_charts(g, names=names, logo=logo)
    assert sorted(charts) == sorted(f"gsax-keepers-{t}.png" for t in viz.THEMES)
    tier = next(t for t in g.tiers if t.tier == gsax.TABLE_MIN_SHOTS)
    for name, c in charts.items():
        assert c.theme.name in name and cc.by_gid(c, viz.LOGO_GID)
        assert viz.check(c) == []
        assert cc.collapse(" ".join(t.get_text() for t in cc.by_gid(c, "subtitle"))) == (
            cc.collapse(cards.caveat(tier))
        )
        assert kc.name_of(names, "sb:g71") in cc.all_text(c)
        # integration census: the † number reaches the footer from the card's row
        gap = cards.team_gap_100(g)
        assert gap is not None and gap > 0  # precondition: a selected-side row
        note = cc.collapse(" ".join(t.get_text() for t in cc.by_gid(c, viz.NOTE_GID)))
        assert f"sit {gap:.1f} per 100 above it" in note
    # integration census: a "noise" verdict at TABLE_MIN_SHOTS (the real-data branch) reaches
    # the subtitle through keeper_charts, chosen over an "empty" tier below the threshold
    noise = _tier(gsax_fits, gsax.TABLE_MIN_SHOTS, 111, (0.06, -0.12, 0.25), (0.2, 0.05, 0.4))
    low = dataclasses.replace(g.tiers[0], tier=gsax.TABLE_MIN_SHOTS - 1, keepers=0)
    gn = dataclasses.replace(g, tiers=(low, noise, *g.tiers[1:]))
    assert cards.stability_branch(noise) == "noise"  # precondition
    for c in cards.keeper_charts(gn, names=names, logo=logo).values():
        subtitle = cc.collapse(" ".join(t.get_text() for t in cc.by_gid(c, "subtitle")))
        assert subtitle == cc.collapse(cards.caveat(noise))
    again = cards.keeper_charts(g, names=names, logo=logo)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for name in charts:
        a = viz.save(charts[name], tmp_path / "a" / name).read_bytes()
        b = viz.save(again[name], tmp_path / "b" / name).read_bytes()
        assert a == b, name
    drop = cards.keeper_charts(g, names=names, logo=logo, team_selected="drop")
    assert sorted(drop) == sorted(f"gsax-keepers-no-team-{t}.png" for t in viz.THEMES)
    ws = cards.keeper_charts(_keeper_fit(gsax_fits, "whoscored"), names=None, logo=None)
    for c in ws.values():
        assert not cc.by_gid(c, viz.LOGO_GID) and viz.check(c) == []
        assert viz.ATTRIBUTION["whoscored"] in cc.all_text(c)
        assert re.search(r"ws:|\b86\d{2,}\b", cc.all_text(c)) is None


def test_keeper_charts_refusals(gsax_fits: dict[str, Any], logo: Path, tmp_path: Path) -> None:
    """``ValueError``: names for a WhoScored fit (Rule 1), a keeper missing from ``names``;
    a missing logo: ``FileNotFoundError`` before anything is drawn."""
    import keeper_cases as kc

    gsax, cards = _gsax(), _cards()
    g = _keeper_fit(gsax_fits)
    names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    ws = _keeper_fit(gsax_fits, "whoscored")
    with pytest.raises(ValueError):
        cards.keeper_charts(ws, names=kc.names(gsax.keeper_rows(ws)["gk_id"].to_list()),
                            logo=logo)  # fmt: skip
    with pytest.raises(ValueError):
        cards.keeper_charts(g, names=names.filter(pl.col("player_id") != "sb:g02"), logo=logo)
    with pytest.raises(FileNotFoundError):
        cards.keeper_charts(g, names=names, logo=tmp_path / "no-logo.png")


def test_keeper_charts_provider_names(gsax_fits: dict[str, Any], logo: Path) -> None:
    """Ratify R11: ``keeper_charts`` itself refuses a StatsBomb fit without names and a WhoScored
    fit with names (the provider is the fit's, never inferred from ``names``)."""
    import keeper_cases as kc

    gsax, cards = _gsax(), _cards()
    g = _keeper_fit(gsax_fits)
    ws = _keeper_fit(gsax_fits, "whoscored")
    assert g.xgot.provider == "statsbomb" and ws.xgot.provider == "whoscored"  # precondition
    with pytest.raises(ValueError, match="keeper_charts: statsbomb charts need names"):
        cards.keeper_charts(g, names=None, logo=logo)
    ws_names = kc.names(gsax.keeper_rows(ws)["gk_id"].to_list())
    with pytest.raises(ValueError, match="keeper_charts: whoscored charts take no names"):
        cards.keeper_charts(ws, names=ws_names, logo=None)


def _fmt_r(s: Any) -> str:
    """``+0.11 [−0.21, +0.40]`` (U+2212, 2 dp); "n/a" for None."""
    if s.r is None:
        return "n/a"
    return f"{s.r:+.2f} [{s.low:+.2f}, {s.high:+.2f}]".replace("-", "−")


def test_caveat_branches(gsax_fits: dict[str, Any]) -> None:
    """``caveat``: one sentence per ``stability_branch``, printing the statistic the rule read
    (for noise / repeatable the binding one of ``first_sb`` and ``all_shots_sb``: the smaller
    ``low``, with its scope); counts with a comma; ``stability_sentence``'s text is unchanged."""
    cards = _cards()
    none = (None, None, None)
    cases = [
        (_tier(gsax_fits, 500, 0, none, none), "empty",
         "No keeper reaches 500 shots: keeper reliability cannot be shown.",
         "no keeper reaches 500 shots (empty tier)"),
        (_tier(gsax_fits, 250, 3, none, none), "estimate",
         "3 keepers reach 250 shots: too few to estimate keeper reliability.",
         "3 keepers reach 250 shots: too few to estimate"),
        (_tier(gsax_fits, 250, 9, (0.83, 0.05, 0.98), (0.73, -0.21, 0.97)), "few",
         "9 keepers reach 250 shots: too few to judge (split-half r +0.83 [+0.05, +0.98]).",
         "9 keepers reach 250 shots: too few to judge (r +0.83 [+0.05, +0.98])"),
    ]  # fmt: skip
    for tier, branch, caveat, old in cases:
        assert cards.stability_branch(tier) == branch
        assert cards.caveat(tier) == caveat
        assert cards.stability_sentence(tier) == old
    first = _tier(gsax_fits, 100, 1111, (0.06, -0.12, 0.25), (0.19, 0.01, 0.37))
    assert first.first_sb.low < first.all_shots_sb.low  # precondition: first_sb binds
    assert cards.stability_branch(first) == "noise"
    assert cards.caveat(first) == (
        f"Not distinguishable from noise at 100 shots: stepped-up split-half r "
        f"{_fmt_r(first.first_sb)} on the first 100 shots of 1,111 keepers."
    )
    assert cards.stability_sentence(first) == "not distinguishable from noise at 100 shots"
    alls = _tier(gsax_fits, 100, 111, (0.3, 0.05, 0.5), (0.2, -0.1, 0.4))
    assert alls.all_shots_sb.low < alls.first_sb.low  # precondition: all_shots_sb binds
    assert cards.caveat(alls).endswith(
        f"split-half r {_fmt_r(alls.all_shots_sb)} on all shots of 111 keepers."
    )
    assert "−" in _fmt_r(alls.all_shots_sb)  # precondition: a negative low prints U+2212
    rep = _tier(gsax_fits, 250, 1111, (0.6, 0.3, 0.8), (0.5, 0.2, 0.7))
    assert cards.stability_branch(rep) == "repeatable"
    assert cards.caveat(rep) == (
        "Keeper and defence repeatable at 250 shots: stepped-up "
        f"split-half r {_fmt_r(rep.all_shots_sb)} on all shots of 1,111 keepers."
    )
    assert cards.stability_sentence(rep) == (
        "keeper records are repeatable beyond chance at 250 shots; this measures the keeper "
        "and the defence in front of him, not the keeper alone"
    )


def _chart_sentence(top_n: int, charts: Mapping[str, str] | None) -> str:
    head = (
        f"The chart shows the highest and lowest {top_n} keeper-seasons by shrunken GSAx per "
        "100 shots with 90% keeper-model intervals; `make gsax` writes them to `data/charts/` "
        "(git-ignored: they embed the StatsBomb logo, PM E3)"
    )
    if not charts:
        return head + "."
    parts = " and ".join(f"`{n}` (SHA-256 `{s}`)" for n, s in sorted(charts.items()))
    return f"{head} as {parts}, so `make reproduce` checks the charts through this card."


def test_card_chart_sentence(
    sb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.4: the keeper-table paragraph is the unchanged names sentence plus the new chart
    sentence (``TOP_N``, ``CI_LEVEL``, ``CHART_FOLDER``, file names sorted with their SHA-256);
    "." without charts; ``CHART_FOLDER`` is printed when ``CHART_DIR`` is moved; no other line
    mentions an SVG."""
    from football_shot_charts import keepers

    cards = _cards()
    assert cards.CHART_FOLDER == "data/charts"
    assert cards.CHART_NAME == "gsax-keepers-{theme}.png"
    monkeypatch.setattr(cards, "CHART_DIR", tmp_path / "elsewhere")
    digests = {"gsax-keepers-light.png": "a" * 64, "gsax-keepers-dark.png": "b" * 64}
    n = sb["table"].height
    k = int((sb["names"].join(sb["table"].select(player_id="gk_id"), on="player_id",
                              how="semi")["names"] > 1).sum())  # fmt: skip
    lead = (
        f"{n:,} keeper-seasons. Names are the lineup display names (the nickname when StatsBomb "
        f"gives one); {k:,} table keeper(s) appear under two spellings, resolved to the latest "
        "match. "
    )
    for charts in (digests, None):
        kw = {} if charts is None else {"charts": charts}
        card = cards.render(sb["g"], names=sb["names"], **kw)["gsax"]
        para = next(x for x in card.splitlines() if x.startswith(f"{n:,} keeper-seasons. Names"))
        assert para == lead + _chart_sentence(keepers.TOP_N, charts)
        assert "elsewhere" not in card and ".svg" not in card


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo (cwd) ignoring ``data/``."""
    import subprocess

    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("data/\n")
    monkeypatch.chdir(root)
    return root


def test_write_charts_all_or_nothing(gsax_fits: dict[str, Any], logo: Path, repo: Path) -> None:
    """``write_charts`` guards every chart before saving any: a chart failing ``viz.check``, a
    tracked folder or a WhoScored canvas in the mapping → ``ValueError`` and no file at all; a
    clean mapping writes every file under the ignored folder and returns the paths."""
    import keeper_cases as kc
    from football_core import viz

    gsax, cards = _gsax(), _cards()
    g = _keeper_fit(gsax_fits)
    names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())

    def fresh() -> dict[str, Any]:
        return dict(cards.keeper_charts(g, names=names, logo=logo))

    bad = fresh()
    c = bad["gsax-keepers-dark.png"]
    for word in ("Overlapping", "Underneath"):
        c.fig.text(0.5, 0.5, word, fontsize=14)
    ws = fresh()
    ws["ws-gsax-keepers-light.png"] = next(iter(cards.keeper_charts(
        _keeper_fit(gsax_fits, "whoscored"), names=None, logo=None).values()))  # fmt: skip
    for charts, folder in ((bad, repo / "data" / "charts"), (fresh(), repo / "charts"),
                           (ws, repo / "data" / "charts")):  # fmt: skip
        with pytest.raises(ValueError):
            cards.write_charts(charts, chart_dir=folder)
        assert [p for p in repo.rglob("*.png")] == []
    out = cards.write_charts(fresh(), chart_dir=repo / "data" / "charts")
    assert sorted(Path(p).name for p in out) == sorted(f"gsax-keepers-{t}.png" for t in viz.THEMES)
    assert all(Path(p).is_file() for p in out)


def test_run_from_digests(
    gsax_fits: dict[str, Any],
    sb: dict[str, Any],
    logo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_from`` (StatsBomb): the keeper charts go to ``CHART_DIR`` read when called, each
    file's SHA-256 (of its bytes) is on the card next to its name, ``GsaxRun.charts`` maps the
    names to the paths; WhoScored: no chart, ``charts`` empty."""
    import hashlib

    from football_core.io import statsbomb

    gsax, cards = _gsax(), _cards()
    g = at_table_min(sb["g"])

    def names(checkout: Path, match_ids: Any, *, player_ids: list[str]) -> pl.DataFrame:
        return _names(pl.DataFrame({"gk_id": sorted(set(player_ids))}))

    monkeypatch.setattr(gsax, "fit", lambda f: g if f.provider == "statsbomb"
                        else gsax_fits["fits"]["whoscored"])  # fmt: skip
    monkeypatch.setattr(statsbomb, "player_names", names)
    monkeypatch.setattr(statsbomb, "logo_file", lambda *a, **k: logo)
    monkeypatch.setattr(cards, "CHART_DIR", tmp_path / "charts")
    run = cards.run_from(gsax_fits["xgot"]["statsbomb"], docs_dir=tmp_path / "docs",
                         raw=tmp_path / "raw")  # fmt: skip
    assert sorted(run.charts) == ["gsax-keepers-dark.png", "gsax-keepers-light.png"]
    card = (tmp_path / "docs" / "gsax.md").read_text(encoding="utf-8")
    assert card == run.cards["gsax"]
    for name, path in run.charts.items():
        assert Path(path) == tmp_path / "charts" / name and Path(path).is_file()
        sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert f"`{name}` (SHA-256 `{sha}`)" in card
    ws = cards.run_from(gsax_fits["xgot"]["whoscored"], docs_dir=tmp_path / "ws-docs",
                        raw=tmp_path / "raw")  # fmt: skip
    assert dict(ws.charts) == {}
    assert sorted(p.name for p in (tmp_path / "charts").iterdir()) == sorted(run.charts)


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
    gain over xGOT alone ("not distinguishable from xGOT alone" when that interval holds 0, PR1
    §4); not better than raw (``gain.low`` 0 included)."""
    cards = _cards()
    base = gsax_fits["fits"]["statsbomb"].next_season
    gain = lambda low: metrics.GainCI(0.02, low, 0.04, metrics.N_BOOT, 40)  # noqa: E731
    ns = lambda **kw: dataclasses.replace(base, **kw)  # noqa: E731
    say = lambda **kw: cards.next_season_sentence(ns(**kw)).lower()  # noqa: E731
    reason = "fewer than 2 test matches (1 match, 3 pairs)"
    assert reason in say(untestable=reason, gain=None, gain_none=None)
    beats = ("shrunken estimates predict next season better than raw GSAx, because raw GSAx "
             "overfits small samples").lower()  # fmt: skip
    alone = "not distinguishable from xgot alone"  # PR1 §4 (was "no better than")
    not_better = "shrunken estimates do not predict next season better than raw GSAx".lower()
    text = say(untestable=None, gain=gain(0.01), gain_none=gain(0.001))
    # run6-ms6 (football 3): a gain is an improvement, never "add ... log loss"
    assert beats in text and "and improve log loss by 0.02000 over xgot alone" in text
    assert alone not in text and "add" not in text
    text = say(untestable=None, gain=gain(0.01), gain_none=gain(0.0))
    assert beats in text and alone in text and "no better than" not in text
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
    ``placement_bands`` row) and its limitation with the count of failing judged bands; no card
    mentions a swapped fit (PR1: deleted); with the minimum at 0 every failing band is named in
    the limitation."""

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
        assert "swapped_fit" not in card and "Swapped fit" not in card, p
    monkeypatch.setattr(metrics, "MIN_JUDGED_SHOTS", 0)
    g = gsax_fits["fits"]["whoscored"]
    t = gsax.placement_bands(g).filter(pl.col("judged") & ~pl.col("within"))
    assert t.height  # precondition: the fixture has a failing band once every band is judged
    card = cards.render(g, names=None)["ws-gsax"]
    for r in t.iter_rows(named=True):
        assert f"{r['feature']} {r['band']} ({r['source']}): {r['shots']:,} shots" in card


def test_team_selected_flag(sb: dict[str, Any]) -> None:
    """Team-selected keeper-seasons are flagged († after the family) in the StatsBomb table;
    without team-selected rows there is no flag. (The split by side moved to
    ``test_team_lines_count_only``, PR1; the chart's flag to ``test_charts_keepers``, CH1.)"""
    gsax, cards = _gsax(), _cards()
    g = sb["g"]
    assert not (g.keepers[gsax.WITHOUT]["family"] == gsax.TEAM_SELECTED).any()  # precondition
    plain = cards.render(g, names=sb["names"])["gsax"]
    assert cards.FLAG not in plain
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


# ---- PR1 keeper-fixes §4: render-only changes ------------------------------------------------


def _census_cards(c: dict[str, Any], g: Any = None) -> Mapping[str, str]:
    """The StatsBomb cards of the census fit (or of ``g``), names for its table keepers."""
    gsax, cards = _gsax(), _cards()
    g = c["g"] if g is None else g
    return cards.render(g, names=_names(gsax.keeper_rows(g), conflicts=0))


def _section(card: str, heading: str) -> str:
    lines = card.splitlines()
    start = lines.index(heading)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("#")), len(lines))
    return "\n".join(lines[start:end])


def test_changelog_v2(gsax_census: dict[str, Any], gsax_fits: dict[str, Any]) -> None:
    """§4: ``CHANGELOG`` is a tuple of the two versions, rendered one bullet each on all four
    cards; the version is v2 (one constant: no card says v1 in its name line)."""
    cards = _cards()
    assert cards.CHANGELOG == (
        "v1 (MS6, P3.1-P3.3): first version.",
        "v2 (PR1): WhoScored season-out from competition-season folds; one penalty δ per "
        "season-out source; team-selected competition-seasons at δ = 0 (flagged); goal_z "
        "sensitivity (StatsBomb).",
    )
    out = {**_census_cards(gsax_census), **cards.render(gsax_fits["fits"]["whoscored"], names=None)}
    assert set(out) == {"gsax", "gk-model", "ws-gsax", "ws-gk-model"}
    for name, card in out.items():
        for entry in cards.CHANGELOG:
            assert f"- {entry}" in card, name
        line = next(x for x in card.splitlines() if x.startswith("- **Name / version:**"))
        assert line.endswith(f"` v{len(cards.CHANGELOG)}"), (name, line)


def test_population_team_selected_line(gsax_census: dict[str, Any]) -> None:
    """§4 ``_population``: "Team-selected competition-seasons: N rated at δ = 0 and flagged (PM
    2026-10-28), K also E2-small, with M keeper-seasons"; census N 2 (T18, T19), K 1 (T18), M 6,
    recomputed here from the flags."""
    gsax = _gsax()
    g = gsax_census["g"]
    flags = g.rated.group_by(gsax.SEASON_KEY).agg(pl.col("zero_family").any(),
                                                  pl.col("small").any())  # fmt: skip
    team = flags.filter(pl.col("zero_family"))
    n, k = team.height, team.filter(pl.col("small")).height
    m = g.keepers[gsax.WITHOUT].join(team.select(gsax.SEASON_KEY), on=gsax.SEASON_KEY,
                                     how="semi").height  # fmt: skip
    assert (n, k, m) == (2, 1, 6)
    card = _census_cards(gsax_census)["gsax"].replace("**", "")  # markdown bold aside
    assert (f"Team-selected competition-seasons: {n:,} rated at δ = 0 and flagged (PM "
            f"2026-10-28), {k:,} also E2-small, with {m:,} keeper-seasons") in card  # fmt: skip


def test_team_lines_count_only(gsax_census: dict[str, Any]) -> None:
    """§4 ``_team_lines``: the split by side over the team-selected seasons not E2-small (T19
    only), each side's GSAx with its SD and shots (recomputed with ``team_selected_split`` on
    those seasons' keeper-seasons) and the seasons' Σ (q − goal), then "δ = 0 removes the
    league-sum coupling, not the gap between the sides."; no sentence attributes the gap (no
    finishing credit, no δ absorbing a record)."""
    from football_shot_models.cards import signed

    gsax = _gsax()
    g = gsax_census["g"]
    groups = gsax.rating_groups(g.rated).filter(
        (pl.col("kind") == "open_play") & pl.col("zero_family") & ~pl.col("small")
    )
    assert groups.height == 1  # precondition: T19
    seasons = g.rated.filter(pl.col("zero_family") & ~pl.col("small")).select(gsax.SEASON_KEY)
    table = g.keepers[gsax.WITHOUT].join(seasons.unique(), on=gsax.SEASON_KEY, how="semi")
    split = gsax.team_selected_split(table)
    assert split.height == 2  # precondition: both sides
    card = _census_cards(gsax_census)["gsax"]
    coupling = "δ = 0 removes the league-sum coupling, not the gap between the sides."
    line = next(x for x in card.splitlines() if coupling in x)
    for r in split.iter_rows(named=True):
        assert f"{r['shots']:,} shots" in line, r
        assert f"{signed(r['gsax'])} (SD {r['sd']:.1f})" in line, r
    assert signed(float(groups["gap"].sum())) in line
    low = card.lower()
    assert "absorbs" not in low and "weaker finishers" not in low and "finisher" not in low


def test_rating_lines_pools(gsax_census: dict[str, Any]) -> None:
    """§4 ``_rating_lines``: one open-play line over the fitted competition-seasons (5 in the
    census) and, per fitted penalty pool, its source with its kicks and keepers (from
    ``rating_groups``; groups at δ = 0 are only counted)."""
    gsax = _gsax()
    g = gsax_census["g"]
    t = gsax.rating_groups(g.rated)
    fitted = t.filter((pl.col("kind") == "open_play") & ~pl.col("small") & ~pl.col("zero_family"))
    assert fitted.height == 5
    card = _census_cards(gsax_census)["gsax"]
    lines = card.splitlines()
    op = next(x for x in lines if x.startswith("- open play:"))
    assert f"over {fitted.height:,} competition-seasons" in op
    pools = t.filter((pl.col("kind") == "penalty") & ~gsax.AT_ZERO)
    assert pools["group"].to_list() == ["season_folds"]  # holdout_fit is an E2-small pool
    for r in pools.iter_rows(named=True):
        hit = [x for x in lines if f"on {r['rows']:,} kicks ({r['keepers']:,} keepers" in x]
        assert hit and r["group"] in hit[0], r


def _segments(card: str, bad: pl.DataFrame) -> list[str]:
    """Per failing band, the card text from its label to the next label or the line's end."""
    labels = [f"{r['feature']} {r['band']} ({r['source']})" for r in bad.iter_rows(named=True)]
    line = next(x for x in card.splitlines() if labels[0] in x)
    starts = [line.index(lab) for lab in labels]
    ends = [min([s for s in starts if s > a], default=len(line)) for a in starts]
    return [line[a:b] for a, b in zip(starts, ends, strict=True)]


def test_band_lines_failing(gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """§4 / PM E1 (a): each failing judged band is printed with feature, band, source, shots,
    goals, Σ q − goals, SD and z (= gap / SD), and marked escalated ("not escalated" is not a
    mark) iff it is in
    ``ESCALATED_BANDS[provider]`` (read when called: the first failing band pinned, then none);
    no adjective ("noise")."""
    from types import MappingProxyType

    from football_shot_models.cards import signed

    gsax, cards = _gsax(), _cards()
    monkeypatch.setattr(metrics, "MIN_JUDGED_SHOTS", 0)
    g = gsax_fits["fits"]["whoscored"]
    bad = gsax.placement_bands(g).filter(pl.col("judged") & ~pl.col("within"))
    assert bad.height >= 1  # precondition: a failing band once every band is judged
    first = bad.row(0, named=True)
    for pinned in ({(first["feature"], first["band"], first["source"])}, set()):
        escalated = MappingProxyType({"statsbomb": frozenset(), "whoscored": frozenset(pinned)})
        monkeypatch.setattr(gsax, "ESCALATED_BANDS", escalated)
        card = cards.render(g, names=None)["ws-gsax"]
        segs = _segments(card, bad)
        for i, (r, seg) in enumerate(zip(bad.iter_rows(named=True), segs, strict=True)):
            z = r["gap"] / r["sd"]
            for part in (f"{r['shots']:,}", f"{r['goals']:,}", signed(r["gap"]),
                         f"{r['sd']:.1f}", f"z {z:+.2f}".replace("-", "−")):  # fmt: skip
                assert part in seg, (part, seg)
            marked = "escalated" in seg.lower() and "not escalated" not in seg.lower()
            assert marked == (i == 0 and bool(pinned)), seg
        assert "noise" not in "".join(segs).lower()
        assert "escalated to the PM" not in "".join(segs)  # E1 (a) is ruled (run7)
        assert not re.search(r"z -\d", "".join(segs))  # true minus signs (run7 architect 7)


def test_disclosure_branches(gsax_fits: dict[str, Any]) -> None:
    """§4 gk-model disclosure (2) from ``g.saw_test`` (S, P, L): S > 0 -> "For S of the P pairs
    the test scores, the rated season's xGOT comes from fits that trained on the test season"
    and "L more pairs are left out"; S = 0 -> "No pair's rated-season xGOT was fitted on its
    test season (P pairs scored, L left out)."."""
    cards = _cards()
    g = gsax_fits["fits"]["whoscored"]
    seen = cards.render(dataclasses.replace(g, saw_test=(3, 7, 2)), names=None)["ws-gk-model"]
    assert ("For 3 of the 7 pairs the test scores, the rated season's xGOT comes from fits that "
            "trained on the test season") in seen  # fmt: skip
    assert "2 more pairs are left out" in seen and "No pair's rated-season xGOT" not in seen
    none = cards.render(dataclasses.replace(g, saw_test=(0, 7, 2)), names=None)["ws-gk-model"]
    assert ("No pair's rated-season xGOT was fitted on its test season (7 pairs scored, 2 left "
            "out).") in none  # fmt: skip
    assert "For 0 of" not in none


def test_tau_without_team_selected(gsax_census: dict[str, Any], gsax_fits: dict[str, Any]) -> None:
    """§4 / Football F1: the keeper-model card prints τ̂ without team-selected seasons beside τ̂
    (``g.model_no_team``); omitted when it is None."""
    g = gsax_census["g"]
    card = _census_cards(gsax_census)["gk-model"]
    line = next(x for x in card.splitlines() if "without team-selected" in x.lower()
                and f"{g.model_no_team.tau:.4f}" in x)  # fmt: skip
    assert line
    ws = _cards().render(gsax_fits["fits"]["whoscored"], names=None)["ws-gk-model"]
    assert gsax_fits["fits"]["whoscored"].model_no_team is None
    assert not any("τ̂" in x and "without team-selected" in x.lower() for x in ws.splitlines())


def test_line_paragraph_statsbomb_only(
    gsax_census: dict[str, Any], gsax_fits: dict[str, Any]
) -> None:
    """§4 / §8: the StatsBomb GSAx card has a "### goal_z sensitivity" summary from
    ``line_summary`` (keeper-seasons, the saves left out, what the shift tracks, every goal kept,
    open play only; the keeper-seasons at δ = 0 on the subset on their own count line, audit A10)
    with no keeper name in it and no per-keeper column in the table; no
    WhoScored card has one. With ``g.noz`` (PM E2 (c)) the section also reports the z-free
    shift (it names ``mouth_z``)."""
    gsax = _gsax()
    g = gsax_census["g"]
    s = gsax.line_summary(g)
    assert s is not None
    card = _census_cards(gsax_census)["gsax"]
    sec = _section(card, "### goal_z sensitivity")
    assert f"{s.far:,}" in sec and f"{s.keeper_seasons:,}" in sec
    assert "tracks the share of saves stopped off the line" in sec
    assert f"Left out of both summaries: {s.at_zero:,} keeper-seasons" in sec  # audit A10
    assert "cannot re-level a subset selected on the outcome" in sec
    assert "every goal" in sec.lower() and "open play" in sec.lower()
    names = _names(gsax.keeper_rows(g), conflicts=0)["player_name"].to_list()
    assert names and not any(n in sec for n in names)
    header = next(x for x in card.splitlines() if x.startswith("|") and "hrunk" in x).lower()
    assert "line" not in header and "yd" not in header and "noz" not in header
    for p in ("whoscored",):
        ws = _cards().render(gsax_fits["fits"][p], names=None)["ws-gsax"]
        assert "### goal_z sensitivity" not in ws and "gsax_line" not in ws  # CHANGELOG aside
    open_ = gsax_census["pred"].filter(pl.col("kind") == "open_play")
    noz = gsax.keeper_table(gsax.noz_rated(open_.with_columns(xgot=pl.col("xgot") * 0.9)))
    g2 = dataclasses.replace(g, noz=noz)
    sec2 = _section(_census_cards(gsax_census, g2)["gsax"], "### goal_z sensitivity")
    assert "mouth_z" in sec2 and sec2 != sec
    with pytest.MonkeyPatch.context() as mp:  # audit A10: δ = 0 keeper-seasons counted apart
        mp.setattr(gsax, "TABLE_MIN_SHOTS", 20)
        s20 = gsax.line_summary(g)
        assert s20 is not None and s20.at_zero > 0 and s20.at_zero_median_abs is not None
        sec3 = _section(_census_cards(gsax_census)["gsax"], "### goal_z sensitivity")
        assert f"Over the {s20.keeper_seasons:,} keeper-seasons" in sec3
        assert (f"Left out of both summaries: {s20.at_zero:,} keeper-seasons" in sec3
                and f"median |shift| {s20.at_zero_median_abs:.2f}" in sec3)  # fmt: skip


def test_team_lines_unchanged(gsax_census: dict[str, Any], gsax_fits: dict[str, Any]) -> None:
    """CH1 §3.3: ``team_split`` is the frame ``_team_lines`` printed before the extraction
    (``team_selected_split`` of the team-selected, not E2-small seasons' keeper-seasons,
    recomputed here); ``team_gap_100`` is 100 · GSAx / shots of its "selected side" row, None
    without one; ``_team_lines`` prints that frame's numbers."""
    from football_shot_models.cards import signed

    gsax, cards = _gsax(), _cards()
    g = gsax_census["g"]
    seasons = g.rated.filter(pl.col("zero_family") & ~pl.col("small")).select(gsax.SEASON_KEY)
    table = g.keepers[gsax.WITHOUT].join(seasons.unique(), on=gsax.SEASON_KEY, how="semi")
    want = gsax.team_selected_split(table)
    got = cards.team_split(g)
    assert got.equals(want)
    sel = want.filter(pl.col("side") == "selected side").row(0, named=True)
    assert cards.team_gap_100(g) == pytest.approx(100 * sel["gsax"] / sel["shots"], abs=1e-12)
    line = next(x for x in cards._team_lines(g) if "δ = 0 removes" in x)
    for r in got.iter_rows(named=True):
        assert f"{r['shots']:,} shots, GSAx {signed(r['gsax'])} (SD {r['sd']:.1f})" in line
    assert cards.team_gap_100(gsax_fits["fits"]["statsbomb"]) is None
    assert cards._team_lines(gsax_fits["fits"]["statsbomb"]) == []


def test_keeper_names_wiring(
    sb: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratify R12: ``keeper_names(g, raw)`` reads ``<raw>/<OPEN_DATA_COMMIT>`` with the fit's
    match ids in date order and the table keepers' ids (``keeper_rows``), and returns what the
    reader returns."""
    from football_core.io import statsbomb

    gsax, cards = _gsax(), _cards()
    g = sb["g"]
    seen: list[tuple[Path, list[str], list[str]]] = []
    out = pl.DataFrame({"player_id": ["sb:x"], "player_name": ["X"], "names": [1]})

    def reader(checkout: Path, ids: Any, *, player_ids: Any) -> pl.DataFrame:
        seen.append((Path(checkout), list(ids), list(player_ids)))
        return out

    monkeypatch.setattr(statsbomb, "player_names", reader)
    assert cards.keeper_names(g, tmp_path / "raw") is out
    ((checkout, ids, players),) = seen
    assert checkout == tmp_path / "raw" / statsbomb.OPEN_DATA_COMMIT
    want = g.xgot.xg.matches.sort("match_date", "match_id")["match_id"].to_list()
    assert ids == want and want != sorted(want, reverse=True)
    assert sorted(players) == sorted(gsax.keeper_rows(g)["gk_id"].to_list()) and players
