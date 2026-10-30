"""The StatsBomb keeper (GSAx) chart: rows, interval, selection, titles, strip, footer and
layout (spec ``runs/CH1/keeper-model-charts`` §3.1, §3.2, §8).

Fits are the session's synthetic ``gsax_fits`` fits with keeper rows built by
``keeper_cases`` (a fixed τ, so the test controls which intervals clear 0). The interval is
checked by an independent route: the posterior mode by root-finding the score, the curvature
by a finite difference, the bounds mapped through Σ q − Σ expit(logit q + u) by hand.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any

import charts_cases as cc
import gsax_cases as gc
import keeper_cases as kc
import numpy as np
import polars as pl
import pytest
from football_core import viz
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq
from scipy.special import expit

from football_shot_models import gsax

FLAG = "†"
MINUS = "−"
CAVEAT = (
    "Not distinguishable from noise at 100 shots: stepped-up split-half r +0.32 [+0.01, +0.54] "
    "on all shots of 111 keepers."
)
MODEL = "xGOT: own model, fitted without the rated season"


def _keepers() -> Any:
    from football_shot_charts import keepers

    return keepers


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("keeper-logo") / "logo.png")


@pytest.fixture
def sb_base(gsax_fits: dict[str, Any]) -> Any:
    return gsax_fits["fits"]["statsbomb"]


@pytest.fixture
def ws_base(gsax_fits: dict[str, Any]) -> Any:
    return gsax_fits["fits"]["whoscored"]


def _fit(base: Any, specs: Sequence[tuple[str, str, int, int]], **kw: Any) -> Any:
    return kc.fit_from(base, kc.rated(specs), **kw)


def _view(g: Any, *, named: bool = True, team_gap_100: float | None = 4.07, **kw: Any) -> Any:
    ids = gsax.keeper_rows(g)["gk_id"].to_list()
    return _keepers().keeper_view(
        g, names=kc.names(ids) if named else None, caveat=CAVEAT, flag=FLAG,
        team_gap_100=team_gap_100, **kw,
    )  # fmt: skip


def _chart(view: Any, logo: Path, *, theme: str | viz.Theme = "light", named: bool = True) -> Any:
    footer = viz.Footer(("statsbomb",) if named else ("whoscored",))
    return _keepers().keeper_chart(view, theme=theme, footer=footer,
                                   logo=logo if named else None)  # fmt: skip


def _gtext(c: viz.Canvas, gid: str) -> str:
    return cc.collapse(" ".join(t.get_text() for t in cc.by_gid(c, gid) if hasattr(t, "get_text")))


def _interval(rows: pl.DataFrame, tau: float, level: float) -> tuple[float, float, float]:
    """(shrunk, low, high) in goals of one keeper-season's open-play rows, independently."""
    q = rows["q"].to_numpy()
    y = rows["is_goal"].cast(pl.Float64).to_numpy()
    lq = np.log(q / (1 - q))

    def score(u: float) -> float:
        return float(np.sum(y - expit(lq + u)) - u / tau**2)

    u = brentq(score, -30.0, 30.0, xtol=1e-14)
    e = 1e-5
    curv = -(score(u + e) - score(u - e)) / (2 * e)
    s = curv**-0.5
    z = NormalDist().inv_cdf((1 + level) / 2)
    total = float(q.sum())
    return (total - float(expit(lq + u).sum()), total - float(expit(lq + u + z * s).sum()),
            total - float(expit(lq + u - z * s).sum()))  # fmt: skip


# ---- rows and interval ------------------------------------------------------------------------


def test_shrunk_interval_independent(sb_base: Any) -> None:
    """``shrunk_interval`` equals the independent route per keeper-season (penalty rows in
    ``rated`` are left out by ``variant(…, WITHOUT)``); the view's rows are per 100 shots, the
    tick is the *rated* GSAx (Σ q − goals, not Σ xgot − goals) and ``clear`` is ``low > 0 or
    high < 0``."""
    keepers = _keepers()
    rows = kc.rated(kc.TWELVE, penalties=[("sb:g71", "A18", 6, 5), ("sb:g02", "B18", 4, 1)])
    g = kc.fit_from(sb_base, rows)
    out = keepers.shrunk_interval(g.model, g.rated)
    assert {*gsax.KEEPER_SEASON, "gsax_shrunk", "low", "high"} <= set(out.columns)
    assert out.height == len(kc.TWELVE)
    open_ = gsax.variant(rows, gsax.WITHOUT)
    view = _view(g)
    by_key = {tuple(r[k] for k in gsax.KEEPER_SEASON): r for r in out.iter_rows(named=True)}
    table = gsax.keeper_rows(g)
    for key, part in open_.group_by(gsax.KEEPER_SEASON):
        shrunk, low, high = _interval(part, kc.TAU, gsax.CI_LEVEL)
        r = by_key[tuple(key)]
        assert r["gsax_shrunk"] == pytest.approx(shrunk, abs=1e-8)
        assert r["low"] == pytest.approx(low, rel=1e-6, abs=1e-6)
        assert r["high"] == pytest.approx(high, rel=1e-6, abs=1e-6)
        assert low < shrunk < high
    raw = {r["gk_id"]: (r["gsax"], r["gsax_raw"], r["shots"]) for r in table.iter_rows(named=True)}
    names = kc.names(table["gk_id"].to_list())
    assert view.rows.height == len(kc.TWELVE)
    for r in view.rows.iter_rows(named=True):
        gk = next(k for k in raw if kc.name_of(names, k) in r["label"])
        gsax_rated, gsax_raw, shots = raw[gk]
        assert gsax_rated != pytest.approx(gsax_raw)  # precondition: δ ≠ 0 separates them
        assert r["shots"] == shots
        assert r["unshrunk"] == pytest.approx(100 * gsax_rated / shots, abs=1e-9)
        key = next(k for k in by_key if k[0] == gk)
        assert r["shrunk"] == pytest.approx(100 * by_key[key]["gsax_shrunk"] / shots, abs=1e-9)
        assert r["low"] == pytest.approx(100 * by_key[key]["low"] / shots, abs=1e-9)
        assert r["high"] == pytest.approx(100 * by_key[key]["high"] / shots, abs=1e-9)
        assert r["clear"] == (r["low"] > 0 or r["high"] < 0)
    assert view.rows.schema["rank"] == pl.Int64 and view.rows["rank"].to_list() == list(
        range(1, 13)
    )
    assert view.rows["shrunk"].to_list() == sorted(view.rows["shrunk"].to_list(), reverse=True)
    assert view.rows.null_count().sum_horizontal().item() == 0


def test_reconstruction_guard(sb_base: Any) -> None:
    """``estimates.gsax_shrunk`` tampered by 1e-6 goals on one row: ``ValueError``
    (``RECON_TOL`` 1e-9); a model grouped by ``CAREER``: ``ValueError``."""
    keepers = _keepers()
    g = _fit(sb_base, kc.TWELVE[:4])
    keepers.shrunk_interval(g.model, g.rated)  # the untampered model passes
    est = g.model.estimates.with_columns(
        gsax_shrunk=pl.when(pl.int_range(pl.len()) == 2).then(pl.col("gsax_shrunk") + 1e-6)
        .otherwise(pl.col("gsax_shrunk"))
    )  # fmt: skip
    with pytest.raises(ValueError):
        keepers.shrunk_interval(dataclasses.replace(g.model, estimates=est), g.rated)
    career = kc.model_at(gsax.variant(g.rated, gsax.WITHOUT), kc.TAU, by=gsax.CAREER)
    with pytest.raises(ValueError):
        keepers.shrunk_interval(career, g.rated)


def test_rows_threshold_per_100(sb_base: Any) -> None:
    """At ``TABLE_MIN_SHOTS`` (100): a 99-shot keeper-season is left out, a 100-shot one is in;
    ``n`` counts the rows of the view, ``min_shots`` prints the threshold."""
    assert gsax.TABLE_MIN_SHOTS == 100
    g = _fit(sb_base, [("sb:g05", "A18", 99, 20), ("sb:g06", "A18", 100, 30),
                       ("sb:g07", "B18", 150, 40)])  # fmt: skip
    view = _view(g)
    assert view.n == 2 and view.min_shots == 100
    assert sorted(view.rows["shots"].to_list()) == [100, 150]
    assert not any("g05" in label for label in view.rows["label"].to_list())


def test_shots_agree(sb_base: Any) -> None:
    """``keeper_rows.shots`` ≠ the model's ``estimates.shots`` on a row: ``ValueError`` (two
    inputs stating one fact)."""
    g = _fit(sb_base, kc.TWELVE[:4])
    est = g.model.estimates.with_columns(
        shots=pl.when(pl.int_range(pl.len()) == 1).then(pl.col("shots") + 1)
        .otherwise(pl.col("shots"))
    )  # fmt: skip
    bad = dataclasses.replace(g, model=dataclasses.replace(g.model, estimates=est))
    with pytest.raises(ValueError):
        _view(bad)


def _xs(c: viz.Canvas, gid: str) -> list[float]:
    """Data x of each ``gid`` artist: a marker's offset, a line's first x."""
    out: list[float] = []
    for a in cc.by_gid(c, gid):
        if hasattr(a, "get_offsets") and len(a.get_offsets()):
            out += [float(v) for v in np.asarray(a.get_offsets())[:, 0]]
        elif hasattr(a, "get_xdata"):
            out.append(float(np.asarray(a.get_xdata(), dtype=float)[0]))
    return sorted(out)


def _bands(c: viz.Canvas, gid: str = "band-row") -> list[tuple[float, float]]:
    return sorted((float(a.get_x()), float(a.get_x() + a.get_width())) for a in cc.by_gid(c, gid))


def _shown(view: Any) -> pl.DataFrame:
    n = _keepers().TOP_N
    r = view.rows
    return r if r.height <= 2 * n else pl.concat([r.head(n), r.tail(n)])


def _num(s: str) -> float:
    return float(s.replace(MINUS, "-").replace("+", ""))


def test_marks_drawn(sb_base: Any, logo: Path) -> None:
    """Per shown row: the dot at the shrunk value, the tick at the unshrunk value, the band from
    low to high (data coordinates of the track); the ``mark-zero`` guide at 0 on the track (QA
    pin: it may be drawn in segments around the full-width divider); the scale row's
    labels are ``nice_ticks`` of ``scale_of`` over the shown low, high and unshrunk values; the
    track's x range is that scale."""
    from football_shot_charts import rows as rw

    view = _view(_fit(sb_base, kc.TWELVE))
    c = _chart(view, logo)
    shown = _shown(view)
    assert shown.height == 6  # precondition: TOP_N = 3 at both ends
    assert _xs(c, "mark-dot") == pytest.approx(sorted(shown["shrunk"].to_list()), abs=1e-9)
    assert _xs(c, "mark-tick") == pytest.approx(sorted(shown["unshrunk"].to_list()), abs=1e-9)
    want = sorted(zip(shown["low"].to_list(), shown["high"].to_list(), strict=True))
    got = _bands(c)
    assert len(got) == len(want)
    for (a, b), (lo, hi) in zip(got, want, strict=True):
        assert a == pytest.approx(lo, abs=1e-9) and b == pytest.approx(hi, abs=1e-9)
    zero = cc.by_gid(c, "mark-zero")  # one guide at 0 (it may break around the divider)
    assert zero and all(set(np.asarray(z.get_xdata(), dtype=float)) == {0.0} for z in zero)
    scale = rw.scale_of([*shown["low"], *shown["high"], *shown["unshrunk"]])
    assert {id(z.axes) for z in zero} == {id(a.axes) for a in cc.by_gid(c, "mark-dot")}
    assert zero[0].axes.get_xlim() == pytest.approx(scale)
    labels = [t.get_text() for t in cc.by_gid(c, "scale-label")]
    assert sorted(_num(x) for x in labels) == pytest.approx(sorted(rw.nice_ticks(scale)))


# ---- selection and divider ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("idx", "divider"),
    [
        ((0, 1, 2, 3, 4), None),  # n = 2N − 1: all rows
        ((0, 1, 2, 3, 4, 5), None),  # n = 2N: all rows, no divider
        ((0, 1, 2, 4, 5, 6, 7), "1 keeper-season in between, none differs clearly"),
        ((0, 1, 2, 3, 4, 5, 6), "1 keeper-season in between, 1 differs clearly"),
        (tuple(range(12)), "6 keeper-seasons in between, {j} {verb} clearly"),
    ],
)
def test_selection_and_divider(
    sb_base: Any, logo: Path, idx: tuple[int, ...], divider: str | None
) -> None:
    """The highest and lowest ``TOP_N`` by shrunk per 100 are shown (ties by key: sb:g33 before
    its identical sb:g55), the rest summarised by one divider counting the hidden rows and the
    hidden clear rows (singular forms computed)."""
    keepers = _keepers()
    assert keepers.TOP_N == 3
    specs = [kc.TWELVE[i] for i in idx]
    g = _fit(sb_base, specs)
    view = _view(g)
    c = _chart(view, logo)
    text = cc.all_text(c)
    names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    shown = _shown(view)
    hidden = view.rows.filter(~pl.col("rank").is_in(shown["rank"].implode()))
    for r in view.rows.iter_rows(named=True):
        gk = next(s[0] for s in specs if kc.name_of(names, s[0]) in r["label"])
        visible = re.search(re.escape(kc.name_of(names, gk)) + r"(?!\d)", text) is not None
        assert visible == (r["rank"] in shown["rank"].to_list()), (gk, r["rank"])
    if 3 in idx:  # the tie: identical rows, the smaller key ranks first
        ranks = {kc.name_of(names, gk): None for gk in ("sb:g33", "sb:g55")}
        for r in view.rows.iter_rows(named=True):
            for name in ranks:
                if name in r["label"]:
                    ranks[name] = r["rank"]
        a, b = ranks.values()
        assert a is not None and b is not None and a + 1 == b
        tied = view.rows.filter(pl.col("rank").is_in([a, b]))
        assert tied["shrunk"][0] == tied["shrunk"][1]
    if divider is None:
        assert hidden.height == 0 and "in between" not in text
        return
    j = int(hidden["clear"].sum())
    if "{j}" in divider:
        assert j == 1  # precondition: sb:g55 is the one hidden clear row
        divider = divider.format(j=j, verb="differs" if j == 1 else "differ")
    else:
        assert j == (1 if "1 differs" in divider else 0)  # precondition
    assert divider in text
    assert text.count("in between") == 1


# ---- team-selected rows ---------------------------------------------------------------------


def _note(c: viz.Canvas) -> str:
    return _gtext(c, viz.NOTE_GID)


@pytest.mark.parametrize(
    ("gap", "tail"),
    [(4.07, "; the selected sides' keepers sit 4.1 per 100 above it"),
     (-2.04, "; the selected sides' keepers sit 2.0 per 100 below it"),
     (None, "")],
)  # fmt: skip
def test_flag_note_number(sb_base: Any, logo: Path, gap: float | None, tail: str) -> None:
    """A shown team-selected row is marked ``†`` and the footer note reads the computed
    sentence with ``team_gap_100`` (its sign picks above / below; None ends the note after
    "not its league"); no shown team-selected row (sb:g61 hidden): no note."""
    g = _fit(sb_base, kc.TWELVE)
    c = _chart(_view(g, team_gap_100=gap), logo)
    note = _note(c)
    lead = f"{FLAG} team-selected season: rated against the model, not its league"
    assert note == lead + tail
    names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    assert f"{kc.name_of(names, 'sb:g12')} {FLAG}" in cc.all_text(c)
    hidden_only = [s for s in kc.TWELVE if s[0] != "sb:g12"]
    view = _view(_fit(sb_base, hidden_only), team_gap_100=gap)
    assert view.rows.filter(pl.col("team_selected"))["rank"].to_list() == [6]  # precondition
    c2 = _chart(view, logo)
    assert _note(c2) == "" and FLAG not in _gtext(c2, viz.NOTE_GID)
    assert "As a group" not in cc.all_text(c2) and "as a group" not in cc.all_text(c2)


def test_drop_mode(sb_base: Any, logo: Path) -> None:
    """``team_selected="drop"``: team-selected rows removed before rating, the rest rated with
    ``g.model_no_team`` (its intervals and spread), ``left_out`` counted and printed in the
    footer note; no ``model_no_team``: ``ValueError``."""
    keepers = _keepers()
    g = _fit(sb_base, kc.TWELVE, tau_no_team=0.6)
    view = _view(g, team_selected="drop")
    assert view.mode == "drop" and view.left_out == 2 and view.n == 10
    assert not view.rows["team_selected"].any() and FLAG not in "".join(view.rows["label"])
    assert view.spread_100 == pytest.approx(g.model_no_team.per_100)
    open_ = gsax.variant(g.rated, gsax.WITHOUT)
    part = open_.filter(pl.col("gk_id") == "sb:g71")
    _, low, high = _interval(part, 0.6, gsax.CI_LEVEL)
    names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    row = view.rows.filter(pl.col("label") == kc.name_of(names, "sb:g71"))
    assert row.height == 1  # precondition
    assert row["low"][0] == pytest.approx(100 * low / kc.SHOTS, rel=1e-6)
    assert row["high"][0] == pytest.approx(100 * high / kc.SHOTS, rel=1e-6)
    c = _chart(view, logo)
    assert _note(c) == "2 team-selected keeper-seasons left out"
    assert "model estimate, SD per 100, without †" in cc.all_text(c)
    with pytest.raises(ValueError):
        keepers.keeper_view(
            dataclasses.replace(g, model_no_team=None),
            names=None,
            caveat=CAVEAT,
            flag=FLAG,
            team_gap_100=None,
            team_selected="drop",
        )


def test_title_without_flag(sb_base: Any) -> None:
    """The only clear row is team-selected: the title adds "(0 without †)"; with a clear row
    that is not team-selected as well, "(1 without †)"; no clear team-selected row: no
    parenthesis."""
    keepers = _keepers()
    near = [
        (f"sb:g9{i}", s, kc.SHOTS, 40 + i % 3)
        for i, s in enumerate(("A18", "A19", "B18", "C18", "C19"))
    ]
    only = _view(_fit(sb_base, [("sb:g12", "T18", kc.SHOTS, 18), *near]))
    assert only.k == 1 and only.k_no_team == 0  # precondition
    assert keepers.title(only) == (
        f"After shrinkage, 1 of 6 keeper-seasons differs clearly from average (0 without {FLAG})"
    )
    both = _view(
        _fit(sb_base, [("sb:g12", "T18", kc.SHOTS, 18), ("sb:g02", "B19", kc.SHOTS, 62), *near])
    )
    assert both.k == 2 and both.k_no_team == 1  # precondition
    assert keepers.title(both).endswith(f"differ clearly from average (1 without {FLAG})")
    plain = _view(_fit(sb_base, [("sb:g02", "B19", kc.SHOTS, 62), *near]))
    assert keepers.title(plain) == (
        "After shrinkage, 1 of 6 keeper-seasons differs clearly from average"
    )


def test_strip_spread_and_width(sb_base: Any, logo: Path) -> None:
    """Strip: ``Keeper spread`` = ``g.model.per_100`` (1 dp) with the no-team spread in its note
    ("n/a" without a no-team model; "at its bound" when τ̂ is); ``Median interval`` = the median
    of high − low per 100 over *all* rows (not only those shown), note "90% width, per 100
    shots"."""
    g = _fit(sb_base, kc.TWELVE, tau_no_team=0.6)
    view = _view(g)
    assert view.spread_100 == pytest.approx(g.model.per_100)
    assert view.spread_100_no_team == pytest.approx(g.model_no_team.per_100)
    widths = (view.rows["high"] - view.rows["low"]).to_numpy()
    assert view.median_width_100 == pytest.approx(float(np.median(widths)), abs=1e-12)
    shown = _shown(view)
    assert float(np.median((shown["high"] - shown["low"]).to_numpy())) != pytest.approx(
        view.median_width_100
    )  # precondition: the shown rows alone give another median
    text = cc.all_text(_chart(view, logo))
    assert f"{g.model.per_100:.1f}" in text and "Keeper spread" in text
    assert f"model estimate, SD per 100 · {g.model_no_team.per_100:.1f} without {FLAG}" in text
    assert f"{view.median_width_100:.1f}" in text and "Median interval" in text
    assert "90% width, per 100 shots" in text
    none = _view(dataclasses.replace(g, model_no_team=None))
    assert none.spread_100_no_team is None
    assert f"model estimate, SD per 100 · n/a without {FLAG}" in cc.all_text(_chart(none, logo))
    bound = _view(dataclasses.replace(g, model=dataclasses.replace(g.model, at_bound=True)))
    assert bound.at_bound
    assert "model estimate at its bound" in cc.all_text(_chart(bound, logo))


# ---- title and words --------------------------------------------------------------------------


def test_title_counts_all_rows(sb_base: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """k = 0 (τ tiny: nothing clears), k = n, and k counting a clear row outside the shown 2N;
    an interval with ``low == 0`` exactly is not clear."""
    keepers = _keepers()
    none = _view(_fit(sb_base, kc.TWELVE, tau=1e-3))
    assert none.k == 0
    assert keepers.title(none) == (
        "After shrinkage, none of 12 keeper-seasons differs clearly from average"
    )
    extremes = [(f"sb:g{10 + i}", s, kc.SHOTS, g) for i, (s, g) in
                enumerate((("A18", 10), ("B18", 12), ("C18", 70), ("A19", 72)))]  # fmt: skip
    every = _view(_fit(sb_base, extremes))
    assert every.k == every.n == 4
    assert (
        keepers.title(every) == "After shrinkage, 4 of 4 keeper-seasons differ clearly from average"
    )
    twelve = _view(_fit(sb_base, kc.TWELVE))
    shown = _shown(twelve)
    k_shown = int(shown["clear"].sum())
    assert twelve.k == int(twelve.rows["clear"].sum()) > k_shown  # sb:g55 is hidden and clear
    assert f" {twelve.k} of 12 keeper-seasons" in keepers.title(twelve)
    # a band touching 0: not clear
    real = keepers.shrunk_interval

    def touching(model: Any, rated: pl.DataFrame) -> pl.DataFrame:
        out = real(model, rated)
        first = pl.int_range(pl.len()) == 0
        return out.with_columns(
            low=pl.when(first).then(0.0).otherwise(pl.col("low")),
            high=pl.when(first).then(1.0).otherwise(pl.col("high")),
        )

    monkeypatch.setattr(keepers, "shrunk_interval", touching)
    edge = _view(_fit(sb_base, extremes))
    zero = edge.rows.filter(pl.col("low") == 0.0)
    assert zero.height == 1 and not zero["clear"][0]
    assert edge.k == 3


def test_no_skill_words(sb_base: Any, ws_base: Any, logo: Path) -> None:
    """No title, subtitle, caption, divider, label or note says best, worst, skill, elite,
    clinical, lucky or perform (named and unnamed, flag and drop)."""
    g = _fit(sb_base, kc.TWELVE, tau_no_team=0.6)
    ws = _fit(ws_base, [(f"ws:86{s[0][4:]}", *s[1:]) for s in kc.TWELVE])
    for c in (_chart(_view(g), logo), _chart(_view(g, team_selected="drop"), logo),
              _chart(_view(ws, named=False), logo, named=False)):  # fmt: skip
        text = cc.all_text(c)
        assert re.search(kc.BANNED, text, re.IGNORECASE) is None, text


# ---- refusals ---------------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["ws-named", "missing-name", "career", "no-row", "null-shrunk"])
def test_keeper_refusals(sb_base: Any, ws_base: Any, case: str) -> None:
    """``ValueError``: a named view with a non-``sb:`` keeper (Rule 1), a keeper missing from
    ``names``, a model grouped by ``CAREER``, no row at the threshold, a null shrunk value (a
    keeper-season missing from the model)."""
    keepers = _keepers()
    if case == "ws-named":
        g = _fit(ws_base, [(f"ws:86{s[0][4:]}", *s[1:]) for s in kc.TWELVE[:4]])
        names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    else:
        g = _fit(sb_base, kc.TWELVE[:4])
        names = kc.names(gsax.keeper_rows(g)["gk_id"].to_list())
    if case == "missing-name":
        names = names.filter(pl.col("player_id") != "sb:g33")
    elif case == "career":
        open_ = gsax.variant(g.rated, gsax.WITHOUT)
        g = dataclasses.replace(g, model=kc.model_at(open_, kc.TAU, by=gsax.CAREER))
    elif case == "no-row":
        g = _fit(sb_base, [("sb:g05", "A18", 99, 20)])
    elif case == "null-shrunk":
        est = g.model.estimates.filter(pl.col("gk_id") != "sb:g12")
        g = dataclasses.replace(g, model=dataclasses.replace(g.model, estimates=est))
    with pytest.raises(ValueError):
        keepers.keeper_view(g, names=names, caveat=CAVEAT, flag=FLAG, team_gap_100=None)


# ---- layout -----------------------------------------------------------------------------------


def _caveats() -> list[str]:
    """Each caveat branch at its longest form (TIERS 500, counts with a comma, U+2212 signs)."""
    r = f"{MINUS}0.11 [{MINUS}0.21, {MINUS}0.40]"
    return [
        "No keeper reaches 500 shots: keeper reliability cannot be shown.",
        "1,111 keepers reach 500 shots: too few to estimate keeper reliability.",
        f"29 keepers reach 500 shots: too few to judge (split-half r {r}).",
        f"Not distinguishable from noise at 500 shots: stepped-up split-half r {r} on the first "
        "500 shots of 1,111 keepers.",
        "Keeper and defence repeatable at 500 shots: stepped-up "
        "split-half r +0.11 [+0.01, +0.40] on the first 500 shots of 1,111 keepers.",
    ]


@pytest.mark.parametrize("family", ["Inter", "fallback"])
@pytest.mark.parametrize("branch", range(5))
def test_caveat_layout(sb_base: Any, logo: Path, family: str, branch: int) -> None:
    """Each caveat branch at its longest fits the subtitle (≤ 2 lines) in both fonts, with the
    2-line worst title, and the chart passes ``viz.check``."""
    g = _fit(sb_base, kc.TWELVE)
    view = dataclasses.replace(_view(g), caveat=_caveats()[branch])
    theme = viz.THEMES["light"] if family == "Inter" else cc.fallback("light")
    c = _chart(view, logo, theme=theme)
    assert viz.check(c) == []
    assert cc.collapse(_caveats()[branch]) in cc.collapse(_gtext(c, "subtitle"))


def _worst(base: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """126 keeper-seasons: 6 clear above 0 (2 of them team-selected, one named with the
    34-char name), 6 clear below 0 in the 33-char competition-season, 114 near average."""
    monkeypatch.setitem(gc.SEASONS, "W18", kc.COMP_33)
    specs = [("sb:g001", "T18", 150, 15), ("sb:g002", "T19", 150, 16)]
    specs += [(f"sb:g{3 + i:03d}", "A18", 150, 17 + i) for i in range(4)]
    specs += [(f"sb:g{7 + i:03d}", "W18", 150, 66 + i) for i in range(6)]
    seasons = ("A19", "A20", "B18", "B19", "C18", "C19")
    specs += [(f"sb:g{13 + i:03d}", seasons[i % 6], 100, 27) for i in range(114)]
    return kc.fit_from(base, kc.rated(specs))


def test_keeper_worst_case_fits(sb_base: Any, logo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Worst-case lengths (a3): a 2-line title with "(10 without †)", the longest caveat, a
    flagged 34-char name at the top, ``#126 · `` + the 33-char competition-season at the bottom
    (unnamed), the flagged footer note, ``TOP_N`` rows at each end; Inter and the fallback
    font; every chart passes ``viz.check``."""
    keepers = _keepers()
    g = _worst(sb_base, monkeypatch)
    ids = gsax.keeper_rows(g)["gk_id"].to_list()
    names = kc.names(ids, long=kc.NAME_34)
    for named in (True, False):
        view = keepers.keeper_view(
            g, names=names if named else None, caveat=_caveats()[3], flag=FLAG,
            team_gap_100=-10.8,
        )  # fmt: skip
        assert view.n == 126 and view.k >= 10 and view.k_no_team == view.k - 2  # precondition
        title = keepers.title(view)
        assert title.endswith(f"({view.k_no_team} without {FLAG})")
        for theme in (viz.THEMES["light"], cc.fallback("dark")):
            c = _chart(view, logo, theme=theme, named=named)
            assert viz.check(c) == [], (named, theme.family)
            text = cc.all_text(c)
            if named:
                assert f"{kc.NAME_34} {FLAG}" in text
            else:
                assert f"#126 · {kc.COMP_33[2]} {kc.COMP_33[3]}" in text


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("family", ["Inter", "fallback"])
@pytest.mark.parametrize("named", [True, False])
def test_keeper_chart_checks(
    sb_base: Any, ws_base: Any, logo: Path, theme: str, family: str, named: bool
) -> None:
    """Both themes and fonts, named (StatsBomb, logo) and unnamed (WhoScored, no logo):
    ``viz.check`` passes; the logo artist only when named; unnamed labels are rank-prefixed and
    hold no id; the title is ``title(view)``, the subtitle the caveat, the model credit
    printed."""
    keepers = _keepers()
    if named:
        g = _fit(sb_base, kc.TWELVE)
    else:
        g = _fit(ws_base, [(f"ws:86{s[0][4:]}", *s[1:]) for s in kc.TWELVE])
    view = _view(g, named=named)
    t = viz.THEMES[theme] if family == "Inter" else cc.fallback(theme)
    c = _chart(view, logo, theme=t, named=named)
    assert viz.check(c) == []
    assert bool(cc.by_gid(c, viz.LOGO_GID)) == named
    assert cc.collapse(_gtext(c, "title")) == cc.collapse(keepers.title(view))
    assert cc.collapse(_gtext(c, "subtitle")) == cc.collapse(CAVEAT)
    assert MODEL in _gtext(c, viz.TEXT_GID)
    text = cc.all_text(c)
    heading = "GSAx per 100 on-target shots · 100+ shots, penalties excluded"
    assert heading in text
    for item in ("After shrinkage", "Before shrinkage", "90% interval"):
        assert item in text
    if not named:
        assert re.search(r"ws:|\b86\d{2,}\b", text) is None
        for r in _shown(view).iter_rows(named=True):
            assert r["label"].startswith(f"#{r['rank']} · ")
            assert r["sublabel"] == f"{r['shots']} shots"
            assert r["label"] in text
    else:
        for r in _shown(view).iter_rows(named=True):
            assert r["sublabel"] in text


@settings(
    max_examples=8, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(goals=st.lists(st.integers(min_value=5, max_value=80), min_size=1, max_size=8))
def test_view_rows_property(sb_base: Any, goals: list[int]) -> None:
    """For any goal counts: ranks 1..n in shrunk order, low ≤ shrunk ≤ high, ``clear`` iff the
    band leaves 0, k = Σ clear, ``n`` = rows."""
    seasons = ("A18", "A19", "A20", "B18", "B19", "C18", "C19", "F1")
    specs = [(f"sb:g{i:02d}", seasons[i], 120, gl) for i, gl in enumerate(goals)]
    view = _view(_fit(sb_base, specs))
    r = view.rows
    assert view.n == r.height == len(goals)
    assert r["rank"].to_list() == list(range(1, len(goals) + 1))
    assert all(a >= b for a, b in zip(r["shrunk"], r["shrunk"][1:], strict=False))
    assert ((r["low"] <= r["shrunk"]) & (r["shrunk"] <= r["high"])).all()
    assert r["clear"].to_list() == [
        lo > 0 or hi < 0 for lo, hi in zip(r["low"], r["high"], strict=True)
    ]
    assert view.k == int(r["clear"].sum())
    assert math.isfinite(view.median_width_100)


def test_zero_guide_per_row(sb_base: Any, logo: Path) -> None:
    """Ratify R3: the ``"zero"`` guide is one segment per shown row, and no segment runs under
    the divider's text or a scale label (display px, both themes)."""
    view = _view(_fit(sb_base, kc.TWELVE))
    assert _shown(view).height == 6 and view.n > 6  # precondition: 6 rows and a divider
    for theme in ("light", "dark"):
        c = _chart(view, logo, theme=theme)
        r = cc.renderer(c)
        zero = cc.by_gid(c, "mark-zero")
        assert len(zero) == 6
        lo, hi = zero[0].axes.get_xlim()
        assert lo < 0 < hi  # precondition: 0 inside the scale
        divider = [t for t in cc.texts(c) if "in between" in t.get_text()]
        labels = cc.by_gid(c, "scale-label")
        assert divider and labels  # precondition: both drawn
        for t in [*divider, *labels]:
            tb = t.get_window_extent(r)
            for z in zero:
                ys = z.get_transform().transform(np.asarray(z.get_xydata()))[:, 1]
                assert min(ys.max(), tb.y1) - max(ys.min(), tb.y0) <= 0.5, t.get_text()


def test_flag_field_carried(sb_base: Any) -> None:
    """Ratify R7: the view's ``flag`` is the mark everywhere it is printed (title, strip note,
    footer note, the team-selected row's label), never a typed †."""
    keepers = _keepers()
    near = [
        (f"sb:g9{i}", s, kc.SHOTS, 40 + i % 3)
        for i, s in enumerate(("A18", "A19", "B18", "C18", "C19"))
    ]
    g = _fit(sb_base, [("sb:g12", "T18", kc.SHOTS, 18), *near])
    ids = gsax.keeper_rows(g)["gk_id"].to_list()
    view = keepers.keeper_view(g, names=kc.names(ids), caveat=CAVEAT, flag="‡", team_gap_100=4.07)
    assert view.k == 1 and view.k_no_team == 0  # precondition: the clear row is team-selected
    texts = [keepers.title(view), *(s.note for s in keepers.strip(view)),
             keepers.footer_note(view) or "", *view.rows["label"].to_list()]  # fmt: skip
    assert texts[0].endswith("(0 without ‡)")
    assert any("without ‡" in x for x in texts[1:3])
    assert (texts[3]).startswith("‡ team-selected season")
    assert sum(x.endswith(" ‡") for x in texts[4:]) == 1
    assert not any(FLAG in x for x in texts)
