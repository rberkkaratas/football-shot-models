"""The shared row panel, scale and ticks, and the one pre-write guard (spec
``runs/CH1/keeper-model-charts`` §2, §6, §8: ``test_charts_rows``)."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

import charts_cases as cc
import pytest
from football_core import viz
from hypothesis import assume, given, settings
from hypothesis import strategies as st


def _rows() -> Any:
    from football_shot_charts import rows

    return rows


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("rows-logo") / "logo.png")


def _canvas(theme: str | viz.Theme = "light", provider: str = "whoscored", logo: Path | None = None
            ) -> viz.Canvas:  # fmt: skip
    return viz.canvas(theme, "portrait", title="A synthetic title", subtitle="A subtitle",
                      footer=viz.Footer((provider,)), logo=logo)  # fmt: skip


def _body(c: viz.Canvas) -> tuple[float, float]:
    w, h = viz.SIZES[c.size]
    return c.body[2] * w, c.body[3] * h


def _row(label: str = "Alpha keeper", value: str = "+1.2", **kw: Any) -> Any:
    base = {"label": label, "sublabel": "Alpha League 2018/2019", "value": value,
            "point": 1.2, "low": -0.5, "high": 2.9, "tick": 2.0}  # fmt: skip
    return _rows().Row(**{**base, **kw})


def _panel(rows: tuple[Any, ...], **kw: Any) -> Any:
    base = {"heading": "A heading", "key": (("dot", "Point"), ("band", "90% interval")),
            "rows": rows, "caption": "Right of 0: a sentence.", "scale_row": True}  # fmt: skip
    return _rows().Panel(**{**base, **kw})


def _draw(c: viz.Canvas, panel: Any, values: list[float]) -> Any:
    rw = _rows()
    bw, bh = _body(c)
    m = rw.place(c, panel, (0.0, 0.0, bw, bh))
    rw.draw_panel(c, (0.0, bh - m.height, bw, m.height), m, scale=rw.scale_of(values),
                  gid="panel-rows")  # fmt: skip
    return m


# ---- measure and place ---------------------------------------------------------------------


def test_wide_rows_and_no_layout() -> None:
    """Wide (label | track | value) when the widest label + widest value + ``MIN_TRACK_PX`` + 2
    gaps fit, stacked below that, ``None`` when a label + value pair, the heading or the
    one-line key does not fit; stacked rows are taller; one arrangement per panel."""
    rw = _rows()
    c = _canvas()
    panel = _panel((_row(), _row(label="Bravo keeper", value="−0.4")))
    label_w = max(c.measure("Alpha keeper", "label")[0], c.measure("Alpha League 2018/2019",
                                                                    "caption")[0])  # fmt: skip
    value_w = c.measure("+1.2", "label")[0]
    wide = rw.measure(c, panel, 900.0)
    assert wide is not None and wide.wide
    narrow = label_w + value_w + 2 * 16 + rw.MIN_TRACK_PX / 2
    stacked = rw.measure(c, panel, narrow)
    assert stacked is not None and not stacked.wide
    assert stacked.height > wide.height
    assert rw.measure(c, panel, label_w - 10) is None
    long_heading = _panel(panel.rows, heading="A heading " * 30)
    assert rw.measure(c, long_heading, 900.0) is None
    long_key = _panel(panel.rows, key=(("dot", "A key label that is long " * 4),
                                       ("band", "Another long key label " * 4)))  # fmt: skip
    assert rw.measure(c, long_key, 900.0) is None


def test_height_no_layout() -> None:
    """``place`` raises ``NoLayout`` naming the height (the rect's px) when the rows are taller
    than the rect, and naming the width when they do not fit it; ``NoLayout`` is a
    ``ValueError``."""
    rw = _rows()
    c = _canvas()
    panel = _panel(tuple(_row(label=f"Keeper {i:02d}") for i in range(30)))
    bw, bh = _body(c)
    assert issubclass(rw.NoLayout, ValueError)
    with pytest.raises(
        rw.NoLayout, match=rf"(tall|height).*\b{bh:.0f}\b|\b{bh:.0f}\b.*(tall|height)"
    ):
        rw.place(c, panel, (0.0, 0.0, bw, bh))
    with pytest.raises(rw.NoLayout, match=r"wide|width|\b40\b"):
        rw.place(c, _panel((_row(),)), (0.0, 0.0, 40.0, bh))
    m = rw.place(c, _panel((_row(),)), (0.0, 0.0, bw, bh))
    assert 0 < m.height <= bh


def test_divider_full_width(logo: Path) -> None:
    """A ``Divider`` spans the full inner width from the label column: a caption wider than the
    track column (but not than the panel) is drawn on one row, left-aligned with the labels;
    one that needs more than 2 lines makes ``measure`` return ``None``."""
    rw = _rows()
    c = _canvas(provider="statsbomb", logo=logo)
    text = "118 keeper-seasons in between, none differs clearly from an average keeper"
    panel = _panel((_row(), rw.Divider(text=text), _row(label="Bravo keeper")))
    m = _draw(c, panel, [-0.5, 2.9, 2.0])
    assert m.wide
    r = cc.renderer(c)
    div = [t for t in cc.texts(c) if cc.collapse(t.get_text()) == text]
    label = [t for t in cc.texts(c) if t.get_text() == "Alpha keeper"]
    assert len(div) == 1 and len(label) == 1
    assert div[0].get_window_extent(r).x0 == pytest.approx(label[0].get_window_extent(r).x0,
                                                           abs=1.0)  # fmt: skip
    assert "\n" not in div[0].get_text()
    assert viz.check(c) == []
    huge = _panel((_row(), rw.Divider(text="word " * 120)))
    assert rw.measure(c, huge, 900.0) is None


def test_hostile_labels(logo: Path) -> None:
    """Quotes, a ``$…$`` pair and an apostrophe in data texts (labels, sublabels, divider) are
    drawn verbatim (``parse_math=False``)."""
    rw = _rows()
    c = _canvas(provider="statsbomb", logo=logo)
    label, sub, div = 'N\'Golo "Kanté"', "Cup $5$ and $10$ 2018", "3 rows in $between$"
    _draw(c, _panel((_row(label=label, sublabel=sub), rw.Divider(text=div))), [-0.5, 2.9, 2.0])
    found = {cc.collapse(t.get_text()): t for t in cc.texts(c)}
    for s in (label, sub, div):
        assert s in found, s
        assert found[s].get_parse_math() is False
    assert viz.check(c) == []


# ---- scale and ticks --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], (-0.15, 3.15)),  # 0 joins the span [0, 3], padded 5 % each side
        ([-2.0, -1.0], (-2.1, 0.1)),
        ([-1.0, 3.0], (-1.2, 3.2)),
        ([0.0, 0.0], (-1.0, 1.0)),  # span 0
    ],
)
def test_scale_of(values: list[float], expected: tuple[float, float]) -> None:
    """[min(v, 0), max(v, 0)] padded by ``SCALE_PAD`` of the span; span 0 → ±1; NaN or ±inf →
    ``ValueError``."""
    rw = _rows()
    assert rw.SCALE_PAD == 0.05
    assert rw.scale_of(values) == pytest.approx(expected)
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            rw.scale_of([1.0, bad])


def _feasible(lo: float, hi: float) -> bool:
    """Some (1, 2, 5) × 10^k step puts 3-5 multiples inside [lo, hi]."""
    for k in range(-15, 15):
        for m in (1, 2, 5):
            s = m * 10.0**k
            if 3 <= math.floor(hi / s) - math.ceil(lo / s) + 1 <= 5:
                return True
    return False


VALUES = st.floats(min_value=-1e6, max_value=1e6).filter(lambda v: v == 0 or abs(v) > 1e-6)


@settings(max_examples=200, deadline=None)
@given(st.lists(VALUES, min_size=1, max_size=12))
def test_scale_and_ticks_property(values: list[float]) -> None:
    """``scale_of`` contains every value and 0; where a (1, 2, 5) × 10^k step can give 3-5 ticks
    (``_feasible``; QA found ranges where none can, see ``test_nice_ticks_gap``),
    ``nice_ticks`` gives 3-5 ascending, evenly spaced ticks inside the scale on such a step, 0
    among them."""
    rw = _rows()
    lo, hi = rw.scale_of(values)
    assert lo < 0 < hi
    assert lo <= min(values) and hi >= max(values)
    span = max(max(values), 0.0) - min(min(values), 0.0)
    if span > 0:
        assert hi - lo == pytest.approx(span * (1 + 2 * rw.SCALE_PAD), rel=1e-9)
    assume(_feasible(lo, hi))
    ticks = rw.nice_ticks((lo, hi))
    assert 3 <= len(ticks) <= 5
    assert all(lo <= t <= hi for t in ticks) and list(ticks) == sorted(ticks)
    assert any(t == 0 for t in ticks)
    step = ticks[1] - ticks[0]
    assert all(
        b - a == pytest.approx(step, rel=1e-9) for a, b in zip(ticks, ticks[1:], strict=False)
    )
    mantissa = step / 10 ** math.floor(math.log10(step))
    assert min(abs(mantissa - m) for m in (1.0, 2.0, 5.0, 10.0)) < 1e-6


def test_nice_ticks_gap() -> None:
    """Ratify R6 (QA pin, tests.md): (−0.651, 0.480) has no (1, 2, 5) × 10^k step with 3-5
    ticks inside (0.2 gives 6, 0.5 gives 2), which the spec's rule does not cover;
    ``nice_ticks`` falls back to 2-5 evenly spaced ticks inside the scale with 0 among them, and
    does not raise."""
    rw = _rows()
    assert not _feasible(-0.651, 0.480)  # precondition
    ticks = rw.nice_ticks((-0.651, 0.480))
    assert 2 <= len(ticks) <= 5 and 0 in ticks
    assert all(-0.651 <= t <= 0.480 for t in ticks)


def test_nice_ticks_and_thin() -> None:
    """Known cases: (−0.15, 3.15) → 0, 1, 2, 3; (−12, 31) → −10 … 30 by 10. ``thin`` keeps a
    label when it clears the last kept one by ``viz.KEY_SPACING_PX`` (ascending indices)."""
    rw = _rows()
    assert rw.nice_ticks((-0.15, 3.15)) == pytest.approx((0.0, 1.0, 2.0, 3.0))
    assert rw.nice_ticks((-12.0, 31.0)) == pytest.approx((-10.0, 0.0, 10.0, 20.0, 30.0))
    c = _canvas()
    labels = ("1%", "2%", "5%", "10%")
    assert rw.thin(c, labels, (100.0, 101.0, 400.0, 700.0)) == (0, 2, 3)
    assert rw.thin(c, labels, (100.0, 300.0, 500.0, 700.0)) == (0, 1, 2, 3)
    w = c.measure("10%", "caption")[0]
    near = (100.0, 100.0 + w + viz.KEY_SPACING_PX / 4, 100.0 + 3 * (w + 2 * viz.KEY_SPACING_PX))
    assert rw.thin(c, ("10%", "10%", "10%"), near) == (0, 2)


# ---- the write guard --------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo (cwd) ignoring ``data/`` and ``outputs/``, like the real one."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("data/\noutputs/\n")
    monkeypatch.chdir(root)
    return root


def test_guard_write(repo: Path, logo: Path) -> None:
    """``guard_write`` writes nothing and raises ``ValueError`` naming the path for: a path git
    would track; a WhoScored chart without ``private``; a private chart outside
    ``PRIVATE_DIR``; a chart with ``viz.check`` problems (named). A clean StatsBomb chart under
    an ignored path passes and still writes nothing."""
    rw = _rows()
    sb = _canvas(provider="statsbomb", logo=logo)
    ws = _canvas(provider="whoscored")
    tracked = repo / "charts" / "x.png"
    with pytest.raises(ValueError, match="x.png"):
        rw.guard_write(sb, tracked, private=False)
    with pytest.raises(ValueError):
        rw.guard_write(ws, Path("outputs/charts/ws.png"), private=False)
    with pytest.raises(ValueError):
        rw.guard_write(ws, Path("outputs/charts/ws.png"), private=True)
    bad = _canvas(provider="statsbomb", logo=logo)
    for word in ("Overlapping", "Underneath"):
        bad.fig.text(0.5, 0.5, word, fontsize=14)
    with pytest.raises(ValueError, match="Overlapping|Underneath"):
        rw.guard_write(bad, Path("outputs/charts/bad.png"), private=False)
    ok = rw.guard_write(sb, Path("outputs/charts/ok.png"), private=False)
    assert Path(ok).name == "ok.png"
    assert rw.guard_write(ws, Path(rw.PRIVATE_DIR) / "ws.png", private=True)
    assert [p for p in repo.rglob("*.png")] == []


def test_row_keyword_only() -> None:
    """Ratify R1: ``Row`` takes keywords only (a defaulted field sits before ``point`` in the
    spec's order); ``low``, ``high`` and ``tick`` default to None (a row without a band)."""
    rw = _rows()
    r = rw.Row(label="A", sublabel="B", value="+1.0", point=1.0)
    assert (r.low, r.high, r.tick, r.subvalue) == (None, None, None, "")
    with pytest.raises(TypeError):
        rw.Row("A", "B", "+1.0", "", 1.0)  # type: ignore[misc]


def test_thin_y_axis() -> None:
    """Ratify R4: ``thin(..., axis="y")`` spaces labels by their heights: two labels closer than
    a label height plus ``viz.KEY_SPACING_PX`` keep the first, a far one is kept; the same
    positions on x (widths) keep a different set when the label is wider than tall."""
    rw = _rows()
    c = _canvas()
    w, h = c.measure("10%", "caption")
    assert w > h  # precondition: a wide label
    gap = h + viz.KEY_SPACING_PX + 1
    ys = (100.0, 100.0 + gap, 100.0 + 2 * gap)
    assert rw.thin(c, ("10%",) * 3, ys, axis="y") == (0, 1, 2)
    assert rw.thin(c, ("10%",) * 3, ys, axis="x") != (0, 1, 2)
    assert rw.thin(c, ("10%",) * 3, (100.0, 101.0, 100.0 + gap), axis="y") == (0, 2)
