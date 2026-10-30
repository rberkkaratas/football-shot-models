"""Model-card visuals drawn from plain inputs: the calibration diagram and the model
comparison (spec ``runs/CH1/keeper-model-charts`` §4.1, §8: ``test_charts_diagnostics``).

Views are hand-built (``keeper_cases.model_view`` / ``xgot_view``): bins frames, gain rows and
strip extras with invented numbers; nothing reads a fit or the cache.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import charts_cases as cc
import keeper_cases as kc
import numpy as np
import pytest
from football_core import viz
from test_cache_render_only import _imports

REPO = Path(__file__).resolve().parents[1]
MINUS = "−"
SB_FOOTER = viz.Footer(("statsbomb",), model="xG: own model, not trained on Test League 2015/2016")
WS_FOOTER = viz.Footer(("whoscored",), model="xG: own model, not trained on the test season")
PERCENTS = {"1%", "2%", "5%", "10%", "20%", "50%", "80%", "90%", "95%", "99%"}


def _dg() -> Any:
    from football_shot_charts import diagnostics

    return diagnostics


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("diag-logo") / "logo.png")


def _cal(view: Any, logo: Path | None, theme: str | viz.Theme = "light") -> viz.Canvas:
    footer = SB_FOOTER if logo is not None else WS_FOOTER
    return _dg().calibration_chart(view, theme=theme, footer=footer, logo=logo)


def _cmp(view: Any, logo: Path | None, theme: str | viz.Theme = "light") -> viz.Canvas:
    dg = _dg()
    footer = SB_FOOTER if logo is not None else WS_FOOTER
    size = dg.COMPARISON_SIZES.get(view.quantity, "portrait")
    return dg.comparison_chart(view, theme=theme, footer=footer, logo=logo, size=size)


def _gtext(c: viz.Canvas, gid: str) -> str:
    return cc.collapse(" ".join(t.get_text() for t in cc.by_gid(c, gid) if hasattr(t, "get_text")))


def _stats(c: viz.Canvas) -> list[str]:
    return [cc.collapse(t.get_text()) for gid in ("stat-value", "stat-label", "stat-note")
            for t in cc.by_gid(c, gid)]  # fmt: skip


def _n_marks(c: viz.Canvas, gid: str) -> int:
    n = 0
    for a in cc.by_gid(c, gid):
        n += len(a.get_offsets()) if hasattr(a, "get_offsets") else 1
    return n


def _norm(s: str) -> str:
    return s.replace(MINUS, "-")


# ---- calibration --------------------------------------------------------------------------


def test_calibration_title_and_strip(logo: Path) -> None:
    """Title k per split = bins whose band holds the mean prediction; the subtitle names the
    unit, bins and level; the strip prints both counts, the CV slope and the split's goals − xG
    with its SD noted by the split's name; axis titles name the outcome."""
    view = kc.model_view()
    c = _cal(view, logo)
    assert _gtext(c, "title") == (
        "xG inside the band in 10 of 10 bins in cross-validation, 9 of 10 on the hold-out"
    )
    assert _gtext(c, "subtitle") == (
        "Out-of-fold and hold-out predictions of shots in 10 equal-count bins · bands: 95% "
        "match bootstrap of the observed rate"
    )
    stats = _stats(c)
    assert stats.count("Shots") == 2 and "Calibration slope" in stats
    assert any(s in stats for s in ("89,895", "89895")) and any(
        s in stats for s in ("9,816", "9816")
    )
    assert "1.01" in stats and "CV fold mean" in stats and "Cross-validation" in stats
    assert "Goals − xG" in stats and "−50.0" in stats and "SD 26.4 · Hold-out" in stats
    text = cc.all_text(c)
    assert "Predicted goal rate" in text and "Observed goal rate" in text
    ws = _cal(kc.xgot_view(), None)
    assert _gtext(ws, "title") == (
        "xGOT inside the band in 10 of 10 bins in cross-validation, 10 of 10 on the test season"
    )
    stats = _stats(ws)
    assert stats.count("On-target shots") == 2 and "n/a" in stats  # slope None
    assert "Goals − xGOT" in stats and "+12.5" in stats and "SD 18.0 · Test season" in stats
    assert "test-season predictions of on-target shots" in _gtext(ws, "subtitle")


@pytest.mark.parametrize("family", ["Inter", "fallback"])
@pytest.mark.parametrize("which", ["xG", "xGOT"])
def test_calibration_ticks_real_ranges(logo: Path, family: str, which: str) -> None:
    """The real bin ranges (xG [0.006, 0.442], xGOT [0.003, 0.905]) on logit axes: tick labels
    from ``TICKS`` thinned so none overlap (``viz.check``), at least 3 per chart; the diagonal,
    10 CV rings and 10 split dots are drawn."""
    if which == "xG":
        view = kc.model_view(
            held_bins=kc.bins(
                kc.XG_MEANS,
                [m * 1.02 for m in kc.XG_MEANS],
                [m * 0.9 for m in kc.XG_MEANS],
                [min(m * 1.035, 0.442) for m in kc.XG_MEANS],
            )
        )
    else:
        view = kc.xgot_view()
    theme = viz.THEMES["light"] if family == "Inter" else cc.fallback("light")
    c = _cal(view, logo, theme)
    assert viz.check(c) == []
    labels = [t.get_text() for t in cc.texts(c) if re.fullmatch(r"\d+%", t.get_text())]
    assert set(labels) <= PERCENTS and len(set(labels)) >= 3
    assert cc.by_gid(c, "mark-diagonal")
    assert _n_marks(c, "mark-cv") == 10 and _n_marks(c, "mark-dot") == 10


def test_outside_bin_label(logo: Path) -> None:
    """A bin outside its band gets a halo (``mark-outside``) and the computed label "{mean_p:.0%}
    predicted, {rate:.0%} observed"; bins inside get neither."""
    view = kc.model_view()
    held = view.held_bins.row(9, named=True)
    c = _cal(view, logo)
    assert _n_marks(c, "mark-outside") == 1
    label = f"{held['mean_p']:.0%} predicted, {held['rate']:.0%} observed"
    assert label == "43% predicted, 53% observed"  # precondition: the fixture's top bin
    assert label in cc.all_text(c)
    inside = _cal(kc.model_view(held_bins=kc.band_bins(kc.XG_MEANS)), logo)
    assert _n_marks(inside, "mark-outside") == 0 and "predicted," not in cc.all_text(inside)


def test_clipped_rate_triangle_and_note(logo: Path) -> None:
    """A clipped bin (observed rate 0) is drawn as an open triangle (``mark-clipped``) and the
    footer note says how it is drawn (``P_FLOOR``); no clipped bin, no triangle and no note."""
    means = list(kc.XG_MEANS)
    rate = [0.001, *[m * 1.02 for m in means[1:]]]
    low = [0.001, *[m * 0.9 for m in means[1:]]]
    high = [0.02, *[m * 1.12 for m in means[1:]]]
    held = kc.bins(means, rate, low, high, clipped=[True] + [False] * 9)
    c = _cal(kc.model_view(held_bins=held), logo)
    assert _n_marks(c, "mark-clipped") == 1
    note = "Open triangle: an observed rate of 0 or 100%, drawn at 0.1% / 99.9%"
    assert note in _gtext(c, viz.NOTE_GID)
    plain = _cal(kc.model_view(), logo)
    assert _n_marks(plain, "mark-clipped") == 0 and "Open triangle" not in cc.all_text(plain)


def test_context_lengths(logo: Path) -> None:
    """The key line carries both contexts at their longest (a 33-char competition-season) and
    still passes the checks in both fonts."""
    view = kc.xgot_view(contexts=("126 competition-seasons", "Abcdef Ghijk Lmnop Qrst 2018/2019"))
    for theme in (viz.THEMES["dark"], cc.fallback("dark")):
        c = _cal(view, logo, theme)
        assert viz.check(c) == []
        text = cc.all_text(c)
        assert "Cross-validation · 126 competition-seasons" in text
        assert "Test season · Abcdef Ghijk Lmnop Qrst 2018/2019" in text


# ---- comparison -------------------------------------------------------------------------------


def test_comparison_title_counts(logo: Path) -> None:
    """a, b = references whose CV / split interval has ``low > 0``; ``low == 0`` is not above;
    CV and the split are counted apart; one reference is singular."""
    dg = _dg()
    view = kc.model_view()
    assert _gtext(_cmp(view, logo), "title") == (
        "Clearly lower log loss than 3 of 3 references in cross-validation and 1 of 3 on the "
        "hold-out"
    )
    rows = (dataclasses.replace(view.rows[0], cv=(0.0010, 0.0, 0.0020)), *view.rows[1:])
    edge = kc.model_view(rows=rows)
    assert _gtext(_cmp(edge, logo), "title").startswith(
        "Clearly lower log loss than 2 of 3 references in cross-validation"
    )
    one = kc.xgot_view()
    assert _gtext(_cmp(one, None), "title") == (
        "Clearly lower log loss than 1 of 1 reference in cross-validation and 1 of 1 on the "
        "test season"
    )
    assert dict(dg.COMPARISON_SIZES) == {"xG": "portrait", "xGOT": "square"}


def test_comparison_scale_and_marks(logo: Path) -> None:
    """Two panels (CV, then the split), each with one row per reference: the dot at the gain,
    the band from low to high; one shared scale (``rows.scale_of`` over every low and high) on
    both tracks; a zero line; values signed to 4 dp; headings, key, caption."""
    from football_shot_charts import rows as rw

    view = kc.model_view()
    c = _cmp(view, logo)
    assert viz.check(c) == []
    cv = [r.cv for r in view.rows]
    held = [r.held for r in view.rows]
    dots = sorted(
        float(o[0]) for a in cc.by_gid(c, "mark-dot") for o in np.asarray(a.get_offsets())
    )
    assert dots == pytest.approx(sorted([g for g, _, _ in cv + held]), abs=1e-12)
    bands = sorted((float(a.get_x()), float(a.get_x() + a.get_width()))
                   for a in cc.by_gid(c, "band-row"))  # fmt: skip
    want = sorted((lo, hi) for _, lo, hi in cv + held)
    assert [v for b in bands for v in b] == pytest.approx([v for w in want for v in w], abs=1e-12)
    scale = rw.scale_of([v for _, lo, hi in cv + held for v in (lo, hi)])
    axes = {id(a.axes): a.axes for a in cc.by_gid(c, "mark-dot")}
    assert len(axes) == 2
    for ax in axes.values():
        assert ax.get_xlim() == pytest.approx(scale)
    zero = cc.by_gid(c, "mark-zero")
    assert {id(z.axes) for z in zero} == set(axes)  # a zero guide on each panel's track
    assert all(set(np.asarray(z.get_xdata(), dtype=float)) == {0.0} for z in zero)
    text = _norm(cc.all_text(c))
    for g, _, _ in cv + held:
        assert f"{g:+.4f}" in text
    assert text.count(": log-loss gain over each reference") == 2
    assert "Hold-out: log-loss gain over each reference" in text
    assert "Right of 0: lower log loss than the reference on the same shots." in text
    assert "Gain" in text and "95% interval" in text
    for r in view.rows:
        assert r.role in text and r.title in text
    assert "Log loss" in text and "0.2641" in text and "CV · Hold-out 0.2702" in text


def test_no_beats_or_matches(logo: Path) -> None:
    """Neither chart says "beats" or "matches" (the comparison is a gain with an interval)."""
    for view in (kc.model_view(), kc.xgot_view()):
        for c in (_cal(view, logo), _cmp(view, logo)):
            assert re.search(r"\bbeats?\b|\bmatches\b", cc.all_text(c), re.IGNORECASE) is None


def test_key_level_patched(logo: Path) -> None:
    """The interval level and the resample count are the view's (read, never typed): level 0.8
    and 1,234 resamples print "80% interval", "80% intervals from 1,234 match resamples" and
    "bands: 80% match bootstrap"."""
    view = kc.model_view(level=0.8, n_boot=1234)
    cmp_text = cc.all_text(_cmp(view, logo))
    assert "80% interval" in cmp_text and "95%" not in cmp_text
    assert "80% intervals from 1,234 match resamples of fixed fits" in cmp_text
    assert "bands: 80% match bootstrap of the observed rate" in cc.all_text(_cal(view, logo))


def test_xgot_extras_in_strip() -> None:
    """xGOT: the square comparison draws the one reference on the headline's scale; the
    xG-only and base-rate gains are strip stats ("Gain over …", signed 3 dp, the split's value
    in the note), not rows."""
    view = kc.xgot_view()
    c = _cmp(view, None)
    assert c.size == "square" and viz.check(c) == []
    stats = _stats(c)
    assert "Gain over xG only" in stats and "+0.108" in stats and "CV · Test season +0.113" in stats
    assert "Gain over base rate" in stats and "+0.202" in stats
    assert "CV · Test season +0.210" in stats
    assert "Log loss" in stats and "0.4030" in stats and "CV · Test season 0.3997" in stats
    assert len(stats) == 9  # 3 stats × value, label, note
    assert _n_marks(c, "mark-dot") == 2  # one row per panel


# ---- seams and checks -------------------------------------------------------------------------


def test_drawing_imports() -> None:
    """``rows`` and ``diagnostics`` import nothing from ``football_shot_models`` (the lift to
    core is a file move); nothing imports ``render``."""
    pkg = REPO / "src" / "football_shot_charts"
    for name in ("rows", "diagnostics"):
        source = (pkg / f"{name}.py").read_text(encoding="utf-8")
        mods = _imports(source, f"football_shot_charts.{name}", False)
        assert [m for m in mods if m.split(".")[0] == "football_shot_models"] == [], name
    for path in sorted(pkg.glob("*.py")):
        mods = _imports(path.read_text(encoding="utf-8"), f"football_shot_charts.{path.stem}",
                        path.name == "__init__.py")  # fmt: skip
        assert "football_shot_charts.render" not in mods, path.name


def test_pass_model_renders(logo: Path) -> None:
    """The drawing takes outcome and unit words: an xP view (completion, passes) prints
    "Completions − xP", "Predicted completion rate" and the unit; nothing says goal or shot."""
    view = kc.model_view(quantity="xP", outcome="completion", unit="passes")
    cal, cmp_ = _cal(view, logo), _cmp(view, logo)
    for c in (cal, cmp_):
        assert viz.check(c) == []
    stats = _stats(cal)
    assert "Completions − xP" in stats and stats.count("Passes") == 2
    text = cc.all_text(cal)
    assert "Predicted completion rate" in text and "Observed completion rate" in text
    assert "xP inside the band" in text
    assert "on the same passes" in cc.all_text(cmp_)
    for c in (cal, cmp_):
        body = " ".join(t.get_text() for t in cc.texts(c) if t.get_gid() not in
                        (viz.TEXT_GID, "prepared-by"))  # fmt: skip
        assert re.search(r"\bgoals?\b|\bshots?\b", body, re.IGNORECASE) is None, body


@pytest.mark.parametrize("chart", ["calibration", "comparison"])
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("provider", ["statsbomb", "whoscored"])
@pytest.mark.parametrize("family", ["Inter", "fallback"])
def test_diagnostic_charts_pass_checks(
    logo: Path, chart: str, theme: str, provider: str, family: str
) -> None:
    """Both charts × both themes × both footers × both fonts pass ``viz.check`` with the longest
    reference titles; the logo artist only on StatsBomb charts."""
    view = kc.model_view() if provider == "statsbomb" else kc.xgot_view()
    t = viz.THEMES[theme] if family == "Inter" else cc.fallback(theme)
    lg = logo if provider == "statsbomb" else None
    c = _cal(view, lg, t) if chart == "calibration" else _cmp(view, lg, t)
    assert viz.check(c) == []
    assert bool(cc.by_gid(c, viz.LOGO_GID)) == (provider == "statsbomb")


def test_calibration_marks_inside_axes(logo: Path) -> None:
    """Ratify R14: the logit axis range covers every bin's (drawn) rate, not only mean, low and
    high: a top bin observed at 70 % with a band [40 %, 45 %] still has its dot inside the plot,
    and so does every ring and dot."""
    means = list(kc.XG_MEANS)
    rate = [m * 1.02 for m in means[:-1]] + [0.70]
    low = [m * 0.9 for m in means[:-1]] + [0.40]
    high = [m * 1.12 for m in means[:-1]] + [0.45]
    c = _cal(kc.model_view(held_bins=kc.bins(means, rate, low, high)), logo)
    r = cc.renderer(c)
    for gid in ("mark-dot", "mark-cv"):
        for a in cc.by_gid(c, gid):
            box = a.axes.get_window_extent(r)
            pts = a.get_offset_transform().transform(np.asarray(a.get_offsets()))
            for x, y in pts:
                assert box.x0 - 0.5 <= x <= box.x1 + 0.5 and box.y0 - 0.5 <= y <= box.y1 + 0.5
    assert viz.check(c) == []


# ---- build decisions (ratify R15-R17) ---------------------------------------------------------


def test_label_box_order(logo: Path) -> None:
    """Ratify R15: an outside bin's label goes right of the marker, else left, below, above:
    the first candidate inside the plot and clear of every taken box."""
    dg = _dg()
    c = _cal(kc.model_view(), logo)
    a = dg._Axes(0.0, 0.0, 2000.0, -5.0, 0.0)
    text = "43% predicted, 53% observed"
    at = (1000.0, 1000.0)
    taken: list[Any] = []
    want = [("left", "center"), ("right", "center"), ("center", "top"), ("center", "bottom")]
    for ha, va in want:
        box, got_ha, got_va = dg._label_box(c, text, at, a, taken)
        assert (got_ha, got_va) == (ha, va)
        assert box[0] >= 0 and box[1] >= 0 and box[2] <= 2000 and box[3] <= 2000
        if ha == "left":
            assert box[0] > at[0]
        elif ha == "right":
            assert box[2] < at[0]
        elif va == "top":
            assert box[3] < at[1]
        else:
            assert box[1] > at[1]
        taken.append(box)
    # at the right edge, right is outside the plot: left wins with nothing taken
    _, ha, _ = dg._label_box(c, text, (1995.0, 1000.0), a, [])
    assert ha == "right"


def test_diagonal_gap_under_label() -> None:
    """Ratify R15: the dashed diagonal is short segments and leaves out every dash under a label
    box (padded by ``LABEL_GAP_PX / 2``); the other dashes are drawn."""
    from matplotlib.figure import Figure

    dg = _dg()
    a = dg._Axes(0.0, 0.0, 400.0, -5.0, 0.0)
    hole = (180.0, 190.0, 260.0, 210.0)  # a label box across the diagonal
    pad = dg.LABEL_GAP_PX / 2
    for holes, cut in (([], False), ([hole], True)):
        ax = Figure().add_subplot()
        dg._diagonal(ax, a, "#000000", holes)
        segs = [np.asarray(ln.get_xydata()) for ln in ax.lines if ln.get_gid() == "mark-diagonal"]
        assert len(segs) > 10
        hit = [
            s for s in segs
            if min(s[:, 0].max(), hole[2] + pad) > max(s[:, 0].min(), hole[0] - pad)
            and min(s[:, 1].max(), hole[3] + pad) > max(s[:, 1].min(), hole[1] - pad)
        ]  # fmt: skip
        assert (not hit) is cut


def _path(marker: str) -> np.ndarray:
    from matplotlib.markers import MarkerStyle

    m = MarkerStyle(marker)
    return np.asarray(m.get_path().transformed(m.get_transform()).vertices)


def test_calibration_mark_styles(logo: Path) -> None:
    """Ratify R16: halo a hollow ink ring of ``DOT_PX + 8``; a clipped rate an open ``^`` (rate 1)
    or ``v`` (rate 0) in the series colour (accent for the split, neutral for CV); the CV band a
    neutral ``CV_LW`` line; the split band ``t.band`` with ``t.mark`` caps."""
    from matplotlib.colors import to_rgba

    from football_shot_charts import rows as rw

    dg = _dg()
    assert dg.HALO_PX == rw.DOT_PX + 8 and dg.CV_LW == 1.5
    means = list(kc.XG_MEANS)
    rate = [0.001, *[m * 1.02 for m in means[1:]]]
    low = [0.001, *[m * 0.9 for m in means[1:]]]
    high = [0.02, *[m * 1.12 for m in means[1:]]]
    held = kc.bins(means, rate, low, high, clipped=[True] + [False] * 9)
    cv_means = [*means[:-1], 0.6]
    cv = kc.bins(cv_means, [m * 1.02 for m in cv_means[:-1]] + [0.999],
                 [m * 0.9 for m in cv_means[:-1]] + [0.55],
                 [m * 1.12 for m in cv_means[:-1]] + [0.9995],
                 clipped=[False] * 9 + [True])  # fmt: skip
    view = kc.model_view()  # one held bin outside its band: one halo
    for theme in ("light", "dark"):
        t = viz.THEMES[theme]
        c = _cal(kc.model_view(held_bins=held, cv_bins=cv), logo, theme)
        clipped = cc.by_gid(c, "mark-clipped")
        assert len(clipped) == 2
        got = {}
        for coll in clipped:
            edge = tuple(np.asarray(coll.get_edgecolor())[0])
            assert np.asarray(coll.get_facecolor()).size == 0 or coll.get_facecolor()[0][3] == 0
            got[edge] = np.asarray(coll.get_paths()[0].vertices)
        assert set(got) == {to_rgba(t.accent), to_rgba(t.neutral)}
        assert np.allclose(got[to_rgba(t.accent)], _path("v"))  # held rate 0
        assert np.allclose(got[to_rgba(t.neutral)], _path("^"))  # CV rate 1
        for ln in cc.by_gid(c, "mark-cv-band"):
            assert ln.get_linewidth() == dg.CV_LW and to_rgba(ln.get_color()) == to_rgba(t.neutral)
        bands = cc.by_gid(c, "band-split")
        assert bands and all(b.get_facecolor() == to_rgba(t.band) for b in bands)
        caps = cc.by_gid(c, "mark-band")
        assert len(caps) == 2 * len(bands)
        assert all(to_rgba(ln.get_color()) == to_rgba(t.mark) for ln in caps)
        halo_c = _cal(view, logo, theme)
        (halo,) = cc.by_gid(halo_c, "mark-outside")
        assert to_rgba(np.asarray(halo.get_edgecolor())[0]) == to_rgba(t.ink)
        face = np.asarray(halo.get_facecolor())
        assert face.size == 0 or face[0][3] == 0
        assert np.asarray(halo.get_sizes())[0] == pytest.approx((dg.HALO_PX * 72 / viz.DPI) ** 2)
        assert np.allclose(halo.get_paths()[0].vertices, _path("o"))


def test_comparison_caption_once(logo: Path) -> None:
    """Ratify R17: the comparison caption (it holds for both panels) is printed once, in the
    last (split) panel, for xG (portrait) and xGOT."""
    for view in (kc.model_view(), kc.xgot_view()):
        c = _cmp(view, logo)
        caps = [t for t in cc.texts(c) if t.get_text().startswith("Right of 0")]
        assert len(caps) == 1
        heading = [t for t in cc.texts(c) if ": log-loss gain over each reference" in t.get_text()]
        assert len(heading) == 2  # precondition: two panels
        last = next(h for h in heading if h.get_text().startswith(view.split[0]))
        assert caps[0].axes is last.axes


# ---- run8-ch1 assessment fixes -----------------------------------------------------------------


@pytest.mark.parametrize("family", ["inter", "fallback"])
def test_level_shift_through_every_bin_is_named(logo: Path, family: str) -> None:
    """Item 8: every bin inside its band but the split's total outside the band rule
    (``held_level_ok`` False, as the WhoScored xGOT test season: +77.7 goals, SD 26.9): the
    title says "but not in total" and the strip note "outside 2 SD"; inside the rule, neither."""
    theme: str | viz.Theme = "light" if family == "inter" else cc.fallback("light")
    view = kc.xgot_view(held_gap=77.7, held_sd=26.9, held_level_ok=False, band_sds=2.0)
    assert bool(view.held_bins["inside"].all()) and bool(view.cv_bins["inside"].all())
    c = _cal(view, None, theme)
    assert viz.check(c) == []
    assert _gtext(c, "title") == (
        "xGOT inside the band in 10 of 10 bins in CV, 10 of 10 on the test season, but not in total"
    )
    assert "+77.7" in _stats(c) and "SD 26.9, outside 2 SD" in _stats(c)
    ok = _cal(dataclasses.replace(view, held_level_ok=True), None, theme)
    assert not _gtext(ok, "title").endswith("not in total")
    assert not any("outside" in s for s in _stats(ok))


def test_calibration_key_names_the_cv_band(logo: Path) -> None:
    """Item 19: the key has a neutral line glyph labelled "Cross-validation band", drawn as the
    CV bands are (``CV_LW``, ``neutral``), in both themes."""
    from matplotlib.colors import to_hex

    for theme in viz.THEMES:
        c = _cal(kc.model_view(), logo, theme)
        assert "Cross-validation band" in [t.get_text() for t in cc.by_gid(c, "panel-key")]
        lines = [a for a in cc.by_gid(c, "mark-panel-key") if hasattr(a, "get_xdata")]
        verticals = [a for a in lines if len(set(a.get_xdata())) == 1]
        assert len(verticals) == 1, verticals
        (key,) = verticals
        assert to_hex(key.get_color()) == viz.THEMES[theme].neutral
        assert key.get_linewidth() == _dg().CV_LW
        assert viz.check(c) == []
