"""The row panel every CH1 chart shares: one mark grammar, one layout engine, one write guard.

A panel is a heading, a one-line key of the marks it draws, rows, optional dividers, an
optional scale row and a caption. Each :class:`Row` is a label block (label over sublabel), a
track and a value block; on the track the grammar is fixed across charts (package 1's A1):

- **dot** (accent): the headline value, drawn last;
- **tick** (ink): the reference it is read against;
- **band with caps**: the interval, under both.

Rows go **wide** (label | track | value) when the widest label and value leave the track
``MIN_TRACK_PX``, else **stacked** (the track under the text); one arrangement per panel, so
rows line up. :func:`measure` returns ``None`` when a part cannot fit the width (a layout
search reads that), :func:`place` raises :class:`NoLayout` naming the width or the height.

:func:`write_all` is the one way a chart file is written (``render`` and
``gsax_cards.write_charts`` both call it): every chart passes :func:`guard_write` before any is
written. The guard refuses a chart drawn in another font than Inter, a WhoScored chart that is
not private and under ``PRIVATE_DIR``, a path git would track, and any ``viz.check`` problem.

This module reads nothing from ``football_shot_models``: its inputs are plain values, so it can
move into ``football_core.viz`` with the passing models as a file move.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
from football_core import viz
from football_core.io import snapshot
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.text import Text

StrPath = viz.StrPath
Rect = tuple[float, float, float, float]  # body px: left, bottom, width, height

PRIVATE_DIR: Final = "data/charts/private"
PANEL_PAD_PX: Final = 16
MIN_TRACK_PX: Final = 160
INNER_GAP_PX: Final = 16  # label block | track | value block
ROW_GAP_PX: Final = 8
TRACK_GAP_PX: Final = 4  # value blocks to the track below them (stacked rows)
TRACK_PX: Final = 20  # the track's height: the tall tick
BAND_PX: Final = 14
KEY_SLOT_PX: Final[Mapping[str, float]] = MappingProxyType(
    {"dot": 12.0, "tick": 12.0, "band": 24.0, "ring": 12.0, "line": 12.0}
)  # glyph widths in a key
KEY_TEXT_GAP_PX: Final = 8
KEY_GID: Final = "mark-panel-key"
KEY_ITEM_GAP_PX: Final = 16
TICK_LW: Final = 2.0
CAP_LW: Final = 1.5
ZERO_LW: Final = 1.0
RING_LW: Final = 1.5
DOT_PX: Final = 10.0
SCALE_PAD: Final = 0.05  # of the span, each side
DIVIDER_LINES: Final = 2
HEADING_LINES: Final = 2  # a long heading wraps (the fallback font is wider) rather than fail
NICE_STEPS: Final = (1.0, 2.0, 5.0)


class NoLayout(ValueError):
    """No arrangement fits the space given (the message names the width or the height)."""


# --- Values -----------------------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Row:
    """One panel row: the label block, the value block and the marks on its track.

    ``point`` is the dot, ``tick`` the reference tick (none when ``None``), ``low`` / ``high``
    the band with its caps (none unless both are set). A value block with ``subvalue == ""``
    is one line, centred on the row.
    """

    label: str
    sublabel: str
    value: str
    subvalue: str = ""
    point: float
    low: float | None = None
    high: float | None = None
    tick: float | None = None


@dataclass(frozen=True)
class Divider:
    """A full-width line of text between rows (what the rows above and below leave out)."""

    text: str


@dataclass(frozen=True)
class Gids:
    """The gids of the row marks (a chart's tests find its marks by them)."""

    band: str = "band-row"
    cap: str = "mark-band"
    tick: str = "mark-tick"
    dot: str = "mark-dot"


@dataclass(frozen=True)
class Panel:
    """What a panel holds; :func:`measure` lays it out.

    ``key`` is ``(glyph, label)`` pairs, glyph one of ``KEY_SLOT_PX``; ``caption`` paragraphs
    are split on ``"\\n"``, each at most ``caption_lines`` lines. ``guide`` ``"zero"`` draws one
    line at 0 under the marks, ``"rail"`` a line along each row's track (package 1).
    """

    heading: str
    key: tuple[tuple[str, str], ...]
    rows: tuple[Row | Divider, ...]
    caption: str = ""
    caption_lines: int = 2
    guide: Literal["zero", "rail"] = "zero"
    scale_row: bool = False
    gids: Gids = Gids()


@dataclass(frozen=True)
class Measured:
    """A panel laid out at an inner width: ``blocks`` per item (a row's two text-line heights,
    a divider's wrapped text and height), heights in px."""

    panel: Panel
    inner: float
    heading: str
    wide: bool
    label_w: float
    value_w: float
    head_h: float
    key_h: float
    blocks: tuple[tuple[float, float] | tuple[str, float], ...]
    scale_h: float
    caption: str
    height: float


# --- Words and numbers ------------------------------------------------------------------------


def signed(v: float, digits: int = 1) -> str:
    """``+1.8``, ``−3.0`` (U+2212) or ``0.0``: a signed value whose rounding to 0 has no sign.

    Examples
    --------
    >>> signed(1.84), signed(-3.04), signed(-0.04), signed(0.00971, 4)
    ('+1.8', '−3.0', '0.0', '+0.0097')
    """
    r = round(v, digits)
    return f"{0.0:.{digits}f}" if r == 0 else f"{r:+.{digits}f}".replace("-", "−")


def scale_of(values: Sequence[float]) -> tuple[float, float]:
    """``[min(v, 0), max(v, 0)]`` padded by ``SCALE_PAD`` of the span; a span of 0 gives ±1.

    Raises
    ------
    ValueError
        A NaN or infinite value, or no value.

    Examples
    --------
    >>> scale_of([2.0, 4.0]), scale_of([0.0])
    ((-0.2, 4.2), (-1.0, 1.0))
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError(f"scale_of needs finite values, got {list(values)!r}")
    lo, hi = min(float(v.min()), 0.0), max(float(v.max()), 0.0)
    span = hi - lo
    if span == 0:
        return (-1.0, 1.0)
    return (lo - SCALE_PAD * span, hi + SCALE_PAD * span)


def nice_ticks(scale: tuple[float, float]) -> tuple[float, ...]:
    """Ticks at a (1, 2, 5) × 10^k step inside ``scale``: the smallest step giving 3 to 5
    ticks (0 included when inside, as every multiple of the step is).

    Examples
    --------
    >>> nice_ticks((-0.4, 3.15))
    (0.0, 1.0, 2.0, 3.0)
    >>> nice_ticks((-1.0, 1.0))
    (-1.0, -0.5, 0.0, 0.5, 1.0)
    """
    lo, hi = scale
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise ValueError(f"nice_ticks needs a finite scale with low < high, got {scale!r}")
    k0 = math.floor(math.log10(hi - lo)) - 2
    found: list[tuple[float, int, int]] = []
    for step in (m * 10.0**k for k in range(k0, k0 + 5) for m in NICE_STEPS):
        first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
        found.append((step, first, last))
    # the smallest step with 3-5 ticks, else the smallest with 2-5 (a span the steps jump over)
    for least in (3, 2):
        for step, first, last in found:
            if least <= last - first + 1 <= 5:
                return tuple(round(i * step, 12) + 0.0 for i in range(first, last + 1))
    raise ValueError(f"no (1, 2, 5) step gives 2-5 ticks in {scale!r}")


def tick_label(v: float, ticks: Sequence[float]) -> str:
    """A tick's label: as many decimals as the step needs, signed (U+2212), ``0`` bare.

    Examples
    --------
    >>> [tick_label(t, (-0.5, 0.0, 0.5)) for t in (-0.5, 0.0, 0.5)]
    ['−0.5', '0', '+0.5']
    >>> tick_label(0.25, (0.0, 0.25, 0.5)), tick_label(20.0, (0.0, 20.0, 40.0))
    ('+0.25', '+20')
    """
    step = min((b - a for a, b in zip(ticks, ticks[1:], strict=False)), default=1.0)
    digits = next((d for d in range(13) if abs(round(step, d) - step) <= 1e-9 * abs(step)), 12)
    return "0" if round(v, digits) == 0 else signed(v, digits)


def thin(
    c: viz.Canvas,
    labels: Sequence[str],
    xs_px: Sequence[float],
    *,
    axis: Literal["x", "y"] = "x",
) -> tuple[int, ...]:
    """Indices of the labels to keep, ascending: a label centred at ``xs_px`` (caption style)
    is kept when its near edge clears the last kept one's far edge by ``viz.KEY_SPACING_PX``
    (``axis="y"``: heights along a vertical axis).

    Examples
    --------
    >>> c = viz.canvas("light", "square", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> thin(c, ["1%", "2%", "5%"], [0.0, 10.0, 200.0])
    (0, 2)
    """
    keep: list[int] = []
    edge = -math.inf
    i_size = 0 if axis == "x" else 1
    for i in sorted(range(len(labels)), key=lambda j: xs_px[j]):
        half = c.measure(labels[i], "caption")[i_size] / 2
        if xs_px[i] - half >= edge + viz.KEY_SPACING_PX or not keep:
            keep.append(i)
            edge = xs_px[i] + half
    return tuple(sorted(keep))


# --- Layout -----------------------------------------------------------------------------------


def _size(c: viz.Canvas, text: str, style: str) -> tuple[float, float]:
    """Measured size as drawn with ``parse_math=False``: a ``$`` pair would otherwise be
    measured as mathtext, so such a text is measured as a drawn, then removed, artist."""
    if "$" not in text:
        return c.measure(text, style)
    t = c.text(0.0, 0.0, text, style, parse_math=False)
    box = t.get_window_extent(c.fig.canvas.get_renderer())  # type: ignore[attr-defined]
    t.remove()
    return float(box.width), float(box.height)


def wrap(c: viz.Canvas, text: str, style: str, width: float) -> list[str]:
    """Greedy word wrap by measured width (a word wider than ``width`` stays one line).

    Examples
    --------
    >>> c = viz.canvas("light", "square", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> wrap(c, "one two three", "caption", 1000.0)
    ['one two three']
    """
    out: list[str] = []
    for word in text.split(" "):
        if out and _size(c, f"{out[-1]} {word}", style)[0] <= width:
            out[-1] = f"{out[-1]} {word}"
        else:
            out.append(word)
    return out


def key_width(c: viz.Canvas, items: Sequence[tuple[str, str]]) -> float:
    """The one-line key's width in px."""
    return sum(KEY_SLOT_PX[g] + KEY_TEXT_GAP_PX + c.measure(t, "caption")[0] for g, t in items) + (
        KEY_ITEM_GAP_PX * (len(items) - 1))  # fmt: skip


def _blocks(c: viz.Canvas, row: Row) -> tuple[float, float]:
    """A row's first and second text-line heights, each its tallest text (extents differ by
    glyphs); a one-line value block counts on the first line only if taller than both."""
    first, second = _size(c, row.label, "label")[1], _size(c, row.sublabel, "caption")[1]
    if row.subvalue:
        return (max(first, c.measure(row.value, "label")[1]),
                max(second, c.measure(row.subvalue, "caption")[1]))  # fmt: skip
    value = c.measure(row.value, "label")[1]
    return (first, second) if value <= first + second else (value - second, second)


def measure(c: viz.Canvas, panel: Panel, inner: float) -> Measured | None:
    """The panel laid out at an inner width (the panel's width less ``2 * PANEL_PAD_PX``), or
    ``None`` when the heading (at most ``HEADING_LINES`` lines), the key (one line), a label +
    value pair, a divider (at most ``DIVIDER_LINES`` lines) or a caption paragraph (at most
    ``caption_lines``) does not fit.

    Examples
    --------
    >>> c = viz.canvas("light", "portrait", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> p = Panel("Heading", (("dot", "Gain"),), (Row(label="A", sublabel="a", value="+1.0",
    ...           point=1.0),))
    >>> m = measure(c, p, 900.0); m is not None and m.wide
    True
    >>> measure(c, p, 20.0) is None
    True
    """
    rows = [r for r in panel.rows if isinstance(r, Row)]
    label_w = max((max(_size(c, r.label, "label")[0], _size(c, r.sublabel, "caption")[0])
                   for r in rows), default=0.0)  # fmt: skip
    value_w = max((max(c.measure(r.value, "label")[0],
                       c.measure(r.subvalue, "caption")[0] if r.subvalue else 0.0)
                   for r in rows), default=0.0)  # fmt: skip
    head_lines = wrap(c, panel.heading, "heading", inner)
    heading = "\n".join(head_lines)
    paras = [wrap(c, para, "caption", inner) for para in panel.caption.split("\n") if para]
    cap = [x for para in paras for x in para]
    dividers = {i: wrap(c, r.text, "caption", inner) for i, r in enumerate(panel.rows)
                if isinstance(r, Divider)}  # fmt: skip
    if (
        len(head_lines) > HEADING_LINES
        or any(c.measure(x, "heading")[0] > inner for x in head_lines)
        or label_w + INNER_GAP_PX + value_w > inner
        or key_width(c, panel.key) > inner
        or any(len(para) > panel.caption_lines for para in paras)
        or any(c.measure(x, "caption")[0] > inner for x in cap)
        or any(len(d) > DIVIDER_LINES or any(_size(c, x, "caption")[0] > inner for x in d)
               for d in dividers.values())
    ):  # fmt: skip
        return None
    wide = inner >= label_w + value_w + MIN_TRACK_PX + 2 * INNER_GAP_PX
    track = 0.0 if wide else TRACK_GAP_PX + TRACK_PX
    blocks: list[tuple[float, float] | tuple[str, float]] = []
    total = 0.0
    for i, r in enumerate(panel.rows):
        if isinstance(r, Divider):
            text = "\n".join(dividers[i])
            h = _size(c, text, "caption")[1]
            blocks.append((text, h))
            total += h + ROW_GAP_PX
        else:
            a, b = _blocks(c, r)
            blocks.append((a, b))
            total += a + b + track + ROW_GAP_PX
    head_h = c.measure(heading, "heading")[1]
    key_h = max(BAND_PX, DOT_PX, *(c.measure(t, "caption")[1] for _, t in panel.key))
    scale_h = c.measure("0123456789", "caption")[1] if panel.scale_row else 0.0
    joined = "\n".join(cap)
    height = (
        2 * PANEL_PAD_PX + head_h + ROW_GAP_PX + key_h + ROW_GAP_PX + total
        + (scale_h + ROW_GAP_PX if panel.scale_row else 0.0)
        + (c.measure(joined, "caption")[1] if joined else -ROW_GAP_PX)
    )  # fmt: skip
    return Measured(panel, inner, heading, wide, label_w, value_w, head_h, key_h, tuple(blocks),
                    scale_h, joined, height)  # fmt: skip


def place(c: viz.Canvas, panel: Panel, rect: Rect) -> Measured:
    """:func:`measure` at ``rect``'s inner width, refusing a panel taller than ``rect``.

    Raises
    ------
    NoLayout
        A part does not fit the width, or the panel is taller than ``rect`` (each named).

    Examples
    --------
    >>> c = viz.canvas("light", "portrait", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> p = Panel("Heading", (), (Row(label="A", sublabel="a", value="+1.0", point=1.0),))
    >>> place(c, p, (0.0, 0.0, 900.0, 20.0))
    Traceback (most recent call last):
    ...
    football_shot_charts.rows.NoLayout: the panel 'Heading' is ... px tall, over the 20 px given
    """
    m = measure(c, panel, rect[2] - 2 * PANEL_PAD_PX)
    if m is None:
        raise NoLayout(f"the panel {panel.heading!r} does not fit {rect[2]:.0f} px of width")
    if m.height > rect[3] + viz.EPS_PX:
        raise NoLayout(
            f"the panel {panel.heading!r} is {m.height:.0f} px tall, over the {rect[3]:.0f} px "
            "given"
        )
    return m


def fraction(c: viz.Canvas, rect: Rect) -> Rect:
    """A rect in body px (origin bottom-left of the body) as fractions of the body."""
    w, h = viz.SIZES[c.size]
    bw, bh = c.body[2] * w, c.body[3] * h
    return (rect[0] / bw, rect[1] / bh, rect[2] / bw, rect[3] / bh)


def body_px(c: viz.Canvas) -> tuple[float, float]:
    """The body's width and height in px."""
    w, h = viz.SIZES[c.size]
    return c.body[2] * w, c.body[3] * h


# --- Marks ------------------------------------------------------------------------------------


def band(ax: Axes, lo: float, hi: float, cy: float, t: viz.Theme, gids: tuple[str, str]) -> None:
    """A band ``BAND_PX`` tall from ``lo`` to ``hi`` with a cap at each end."""
    ax.add_patch(Rectangle((lo, cy - BAND_PX / 2), hi - lo, BAND_PX, fc=t.band, ec="none",
                           zorder=2, gid=gids[0]))  # fmt: skip
    for v in (lo, hi):
        ax.plot([v, v], [cy - BAND_PX / 2, cy + BAND_PX / 2], color=t.mark, lw=CAP_LW,
                solid_capstyle="butt", zorder=3, gid=gids[1])  # fmt: skip


def tick(ax: Axes, x: float, cy: float, length: float, t: viz.Theme, gid: str) -> None:
    """The reference tick (ink)."""
    ax.plot([x, x], [cy - length / 2, cy + length / 2], color=t.ink, lw=TICK_LW,
            solid_capstyle="butt", zorder=4, gid=gid)  # fmt: skip


def dot(ax: Axes, x: float, cy: float, t: viz.Theme, gid: str) -> None:
    """The headline dot (accent, with a background edge so it stands on any mark)."""
    ax.scatter([x], [cy], s=(DOT_PX * 72 / viz.DPI) ** 2, c=t.accent, edgecolors=t.background,
               linewidths=viz.MARK_LW, zorder=5, gid=gid, clip_on=False)  # fmt: skip


def ring(ax: Axes, x: float, cy: float, colour: str, gid: str, size_px: float = DOT_PX) -> None:
    """A hollow marker: the reference series (cross-validation) or a halo."""
    ax.scatter([x], [cy], s=(size_px * 72 / viz.DPI) ** 2, facecolors="none", edgecolors=colour,
               linewidths=RING_LW, zorder=5, gid=gid, clip_on=False)  # fmt: skip


def draw_key(
    c: viz.Canvas, ax: Axes, items: Sequence[tuple[str, str]], x: float, cy: float
) -> None:
    """The key's glyphs (as the rows draw them; gids ``band-key`` and ``KEY_GID``) and their
    labels, in one line from ``x``, centred on ``cy`` (``ax`` in px, y down or up). ``"line"``
    is a neutral vertical hairline (the calibration chart's cross-validation band)."""
    t = c.theme
    for glyph, label in items:
        slot = KEY_SLOT_PX[glyph]
        if glyph == "band":
            band(ax, x, x + slot, cy, t, ("band-key", KEY_GID))
        elif glyph == "tick":
            tick(ax, x + slot / 2, cy, BAND_PX, t, KEY_GID)
        elif glyph == "ring":
            ring(ax, x + slot / 2, cy, t.neutral, KEY_GID)
        elif glyph == "line":
            ax.plot([x + slot / 2] * 2, [cy - BAND_PX / 2, cy + BAND_PX / 2], color=t.neutral,
                    lw=CAP_LW, solid_capstyle="butt", zorder=3, gid=KEY_GID)  # fmt: skip
        else:
            dot(ax, x + slot / 2, cy, t, KEY_GID)
        x += slot + KEY_TEXT_GAP_PX
        c.text(x, cy, label, "caption", ax=ax, va="center", gid="panel-key")
        x += c.measure(label, "caption")[0] + KEY_ITEM_GAP_PX


def _marks(ax: Axes, row: Row, cy: float, t: viz.Theme, g: Gids, rail: tuple[float, float] | None
           ) -> None:  # fmt: skip
    if rail is not None:
        ax.plot(list(rail), [cy, cy], color=t.line, lw=1.0, zorder=1, gid="track-rail")
    if row.low is not None and row.high is not None:
        band(ax, row.low, row.high, cy, t, (g.band, g.cap))
    if row.tick is not None:
        tick(ax, row.tick, cy, TRACK_PX, t, g.tick)
    dot(ax, row.point, cy, t, g.dot)


def draw_panel(
    c: viz.Canvas, rect: Rect, m: Measured, *, scale: tuple[float, float], gid: str
) -> None:
    """Draw a measured panel in ``rect`` (body px): the panel axes (px from its top-left) and,
    over the track column, the track axes whose x is ``scale``.

    Examples
    --------
    >>> c = viz.canvas("light", "portrait", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> p = Panel("Gains", (("dot", "Gain"), ("band", "95% interval")), (Row(label="A",
    ...           sublabel="a", value="+1.0", point=1.0, low=0.5, high=1.5),), scale_row=True)
    >>> m = place(c, p, (0.0, 0.0, 900.0, 600.0))
    >>> draw_panel(c, (0.0, 600.0 - m.height, 900.0, m.height), m, scale=scale_of([0.5, 1.5]),
    ...            gid="gains")
    >>> viz.check(c)
    []
    """
    p, t = m.panel, c.theme
    ax = c.axes(fraction(c, rect), gid=gid, panel=True)
    pad, iw = PANEL_PAD_PX, m.inner
    ax.set_xlim(0.0, iw + 2 * pad)
    ax.set_ylim(m.height, 0.0)
    c.text(pad, pad, m.heading, "heading", ax=ax, va="top")
    x0, width = float(pad), iw
    if m.wide:
        x0 = pad + m.label_w + INNER_GAP_PX
        width = iw - m.label_w - m.value_w - 2 * INNER_GAP_PX
    track = c.axes(fraction(c, (rect[0] + x0, rect[1], width, rect[3])), gid="track")
    track.set_xlim(*scale)
    track.set_ylim(m.height, 0.0)
    y = pad + m.head_h + ROW_GAP_PX
    draw_key(c, ax, p.key, float(pad), y + m.key_h / 2)
    y += m.key_h + ROW_GAP_PX
    zero = p.guide == "zero" and scale[0] < 0 < scale[1]
    rail = scale if p.guide == "rail" else None
    for item, block in zip(p.rows, m.blocks, strict=True):
        if isinstance(item, Divider):
            text, h = block
            c.text(pad, y, str(text), "caption", ax=ax, va="top", parse_math=False)
            y += float(h) + ROW_GAP_PX
            continue
        first_h, second_h = float(block[0]), float(block[1])
        c.text(pad, y, item.label, "label", ax=ax, va="top", parse_math=False)
        c.text(pad, y + first_h, item.sublabel, "caption", ax=ax, va="top", parse_math=False)
        rows_h = first_h + second_h
        if item.subvalue:
            c.text(pad + iw, y, item.value, "label", ax=ax, va="top", ha="right")
            c.text(pad + iw, y + first_h, item.subvalue, "caption", ax=ax, va="top", ha="right")
        else:
            c.text(pad + iw, y + rows_h / 2, item.value, "label", ax=ax, va="center", ha="right")
        cy = y + rows_h / 2 if m.wide else y + rows_h + TRACK_GAP_PX + TRACK_PX / 2
        step = rows_h + (0.0 if m.wide else TRACK_GAP_PX + TRACK_PX)
        if zero:  # one segment per row: never under a divider's or a stacked row's text
            half = ROW_GAP_PX / 2
            span = (y - half, y + step + half) if m.wide else (cy - TRACK_PX / 2, cy + TRACK_PX / 2)
            track.plot([0.0, 0.0], list(span), color=t.ink, lw=ZERO_LW, zorder=1,
                       solid_capstyle="butt", gid="mark-zero")  # fmt: skip
        _marks(track, item, cy, t, p.gids, rail)
        y += step + ROW_GAP_PX
    if p.scale_row:
        ticks = nice_ticks(scale)
        labels = [tick_label(v, ticks) for v in ticks]
        xs = [x0 + (v - scale[0]) / (scale[1] - scale[0]) * width for v in ticks]
        for i in thin(c, labels, xs):
            c.text(xs[i], y, labels[i], "caption", ax=ax, va="top", ha="center",
                   gid="scale-label")  # fmt: skip
        y += m.scale_h + ROW_GAP_PX
    if m.caption:
        c.text(pad, y, m.caption, "caption", ax=ax, va="top")


# --- Write guard ------------------------------------------------------------------------------


def _private(path: Path) -> bool:
    root = os.path.normpath(os.path.abspath(PRIVATE_DIR)) + os.sep
    return os.path.normpath(os.path.abspath(path)).startswith(root)


def guard_write(c: viz.Canvas, path: StrPath, *, private: bool) -> Path:
    """Every check before a chart file is written; writes nothing.

    A WhoScored chart is recognised by ``c.providers`` (set by ``viz.canvas``) and, as a second
    check, by ``viz.ATTRIBUTION["whoscored"]`` in any text of the figure (axes texts included).

    Raises
    ------
    ValueError
        A chart not drawn in ``viz.FONT_FAMILY`` (the vendored Inter did not register, so the
        DejaVu fallback drew it); a WhoScored chart without ``private``; a private chart outside
        ``PRIVATE_DIR`` (``normpath(abspath)``, never resolved); a path git would track, or git
        not answering; a chart ``viz.check`` finds problems in (listed). Each message names the
        path.

    Examples
    --------
    >>> c = viz.canvas("light", "square", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> guard_write(c, "outputs/charts/ws.png", private=False)
    Traceback (most recent call last):
    ...
    ValueError: outputs/charts/ws.png: a WhoScored chart is private: pass private=True and a ...
    """
    path = Path(path)
    if viz.FAMILY != viz.FONT_FAMILY or c.theme.family != viz.FONT_FAMILY:
        files = ", ".join(viz.FONT_FILES.values())
        raise ValueError(
            f"{path}: drawn in {c.theme.family}, not {viz.FONT_FAMILY}: the vendored fonts "
            f"({files} in {viz.FONT_DIR}) did not register (see the import warning)"
        )
    texts = " ".join(" ".join(t.get_text().split()) for t in c.fig.findobj(Text))
    whoscored = "whoscored" in c.providers or viz.ATTRIBUTION["whoscored"] in texts
    if whoscored and not private:
        raise ValueError(
            f"{path}: a WhoScored chart is private: pass private=True and a path under "
            f"{PRIVATE_DIR}"
        )
    if private and not _private(path):
        raise ValueError(f"{path}: a private chart goes under {PRIVATE_DIR}")
    snapshot.check_ignored(Path(os.path.abspath(path)), what="chart")
    if problems := viz.check(c):
        raise ValueError(f"{path}: {len(problems)} chart check(s) failed: " + "; ".join(problems))
    return path


def write_all(charts: Mapping[str, viz.Canvas], out: StrPath, *, private: bool) -> tuple[Path, ...]:
    """Write every chart to ``out/<name>`` after every one passes :func:`guard_write`: nothing
    is written when one fails. Returns the paths in ``charts`` order.

    Raises
    ------
    ValueError
        From :func:`guard_write`.

    Examples
    --------
    >>> c = viz.canvas("light", "square", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> write_all({"ws.png": c}, "outputs/charts", private=False)
    Traceback (most recent call last):
    ...
    ValueError: outputs/charts/ws.png: a WhoScored chart is private: pass private=True and a ...
    """
    paths = [guard_write(c, Path(out) / name, private=private) for name, c in charts.items()]
    for path, c in zip(paths, charts.values(), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        viz.save(c, path)
    return tuple(paths)
