"""Model-card visuals: is the model calibrated, and is it better than simpler alternatives?

Two charts, drawn from plain values (a :class:`ModelView`) so the passing models can reuse
them for xP (``quantity="xP"``, ``outcome="completion"``); nothing here reads
``football_shot_models`` or types "goal" or "shot".

**Calibration** (:func:`calibration_chart`): one reliability diagram on equal-aspect *logit*
axes (on linear axes 7 of 10 xG bins crowd into the bottom fifth), both splits overlaid so a
hold-out miss shows against its cross-validation ring. Each bin is a marker at (mean
prediction, observed rate) with the match-bootstrap band of the observed rate; the title counts
the bins whose band holds the mean prediction and, since a level shift through every bin passes
every bin's band, adds "but not in total" when the split's outcomes − Σ prediction fails the
project's band rule (``held_level_ok``, judged by the caller). Draw order keeps every mark
visible: the dashed diagonal, the cross-validation bands (thin neutral lines, keyed) and hollow
rings (neutral), the split's bands with caps, its dots (accent) last. A bin outside its band
gets a halo and a computed label; an observed rate of 0 or 1 cannot sit on a logit axis, so it
is an open triangle at ``P_FLOOR``.

**Comparison** (:func:`comparison_chart`): the log-loss gain over each reference with its
interval against 0, cross-validation and the split in two panels on one scale (the row panel
of :mod:`~football_shot_charts.rows`). References far off the headline's scale go on the
strip instead (``Extra``), so the rows that matter are not flattened.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import polars as pl
from football_core import viz
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from football_shot_charts import rows

Gain = tuple[float, float, float]  # gain, low, high
Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in body px

TICKS: Final = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99)
LOGIT_PAD: Final = 0.25  # logit units around the drawn values
P_FLOOR: Final = 1e-3  # an observed rate of 0 or 1 is drawn at P_FLOOR or 1 - P_FLOOR
COMPARISON_SIZES: Final[Mapping[str, str]] = MappingProxyType({"xG": "portrait", "xGOT": "square"})
CALIBRATED_MISSES: Final = 1  # bins per split outside the band "well calibrated" allows
BIN_COLUMNS: Final = ("bin", "shots", "mean_p", "rate", "low", "high", "inside", "clipped")
MAX_EXTRAS: Final = 2  # a 4-stat comparison strip fails in DejaVu (a2)
AXIS_GAP_PX: Final = 8
HALO_PX: Final = rows.DOT_PX + 8
LABEL_GAP_PX: Final = 6
DASH_PX: Final = (8.0, 6.0)  # the diagonal's dash and gap: short artists, so its box is small
CV_LW: Final = rows.CAP_LW  # as the key's "line" glyph
GRID_LW: Final = 0.8
DIAGONAL_LW: Final = 1.2


@dataclass(frozen=True)
class GainRow:
    """One reference: its role (label), title (sublabel) and the head's gain over it."""

    role: str
    title: str
    cv: Gain
    held: Gain


@dataclass(frozen=True)
class Extra:
    """A gain printed on the strip, not drawn (a reference far off the headline's scale)."""

    label: str
    cv: float
    held: float


@dataclass(frozen=True, eq=False)
class ModelView:
    """What the two model-card charts draw, in plain values.

    Attributes
    ----------
    quantity, outcome, unit : str
        ``"xG"``, ``"goal"``, ``"shots"`` (or ``"xP"``, ``"completion"``, ``"passes"``).
    split : tuple of str
        The scored-once split as name, title form and adjective, e.g. ``("Hold-out", "the
        hold-out", "hold-out")``.
    contexts : tuple of str
        What cross-validation and the split cover (``"36 competition-seasons"``).
    cv_bins, held_bins : polars.DataFrame
        ``BIN_COLUMNS``: equal-count bins with the band of the observed rate.
    cv_n, held_n : int
    cv_slope : float or None
        The calibration slope, cross-validation fold mean.
    held_gap, held_sd : float
        Outcomes − Σ prediction on the split and its Poisson-binomial SD.
    held_level_ok : bool
        ``held_gap`` within the band rule (``metrics.within``, judged by the caller, so this
        module reads nothing from ``football_shot_models``).
    band_sds : float
        The band rule's width in SDs (``metrics.BAND_SDS``), printed when the level fails.
    cv_log_loss, held_log_loss : float
    rows : tuple of GainRow
        At least one reference drawn.
    extras : tuple of Extra
        At most ``MAX_EXTRAS`` gains on the strip.
    level : float
        The interval level of the bands and the gains.
    n_boot : int
        Match resamples behind every interval.
    notes : tuple of str
        Footer note parts of the comparison chart only: they qualify a reference row, and the
        calibration chart draws no reference.
    """

    quantity: str
    outcome: str
    unit: str
    split: tuple[str, str, str]
    contexts: tuple[str, str]
    cv_bins: pl.DataFrame
    held_bins: pl.DataFrame
    cv_n: int
    held_n: int
    cv_slope: float | None
    held_gap: float
    held_sd: float
    held_level_ok: bool
    band_sds: float
    cv_log_loss: float
    held_log_loss: float
    rows: tuple[GainRow, ...]
    extras: tuple[Extra, ...]
    level: float
    n_boot: int
    notes: tuple[str, ...] = ()

    @property
    def count_word(self) -> str:
        """``"Goals"`` (``outcome`` capitalised, plural)."""
        return f"{self.outcome.capitalize()}s"


def _check_view(view: ModelView) -> None:
    if not view.rows:
        raise ValueError("a model view needs at least one reference row")
    if len(view.extras) > MAX_EXTRAS:
        raise ValueError(f"{len(view.extras)} extras; at most {MAX_EXTRAS} fit the strip")
    for name, bins in (("cv_bins", view.cv_bins), ("held_bins", view.held_bins)):
        if missing := [col for col in BIN_COLUMNS if col not in bins.columns]:
            raise ValueError(f"{name} is missing column(s): {', '.join(missing)}")
        if bins.height == 0:
            raise ValueError(f"{name} is empty")


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _pct(p: float) -> str:
    return f"{p:.0%}" if p >= 0.01 else f"{p:.1%}"


def _notes(parts: Sequence[str]) -> str | None:
    return " · ".join(parts) or None


# --- Calibration ------------------------------------------------------------------------------


def calibration_title(view: ModelView) -> str:
    """Bins that agree (their band holds the mean prediction), per split. "Well calibrated"
    only when each split misses at most ``CALIBRATED_MISSES`` bins and the split's total
    passes the band rule (``held_level_ok``); when the total fails, ", but not in total" (the
    strip prints the gap, its range and "outside"), with "CV" so the title keeps 2 lines.

    Examples
    --------
    >>> calibration_title(_example_view())
    'xG is well calibrated: 3 of 3 bins agree in cross-validation, 2 of 3 on the hold-out'
    >>> calibration_title(dataclasses.replace(_example_view(), held_level_ok=False))
    'xG: 3 of 3 bins agree in CV, 2 of 3 on the hold-out, but not in total'
    """
    (k1, n1), (k2, n2) = ((int(b["inside"].sum()), b.height)
                          for b in (view.cv_bins, view.held_bins))  # fmt: skip
    counts = f"{k2} of {n2} on {view.split[1]}"
    if not view.held_level_ok:
        return f"{view.quantity}: {k1} of {n1} bins agree in CV, {counts}, but not in total"
    good = n1 - k1 <= CALIBRATED_MISSES and n2 - k2 <= CALIBRATED_MISSES
    head = f"{view.quantity} is well calibrated" if good else view.quantity
    return f"{head}: {k1} of {n1} bins agree in cross-validation, {counts}"


def calibration_strip(view: ModelView) -> tuple[viz.Stat, ...]:
    """Counts of both splits, the CV calibration slope and the split's outcomes − prediction
    with its normal range (± ``band_sds`` SD); when ``held_level_ok`` is False the note reads
    "outside" in place of "normal".

    Examples
    --------
    >>> [(s.value, s.label, s.note) for s in calibration_strip(_example_view())][-1]
    ('−3.2', 'Goals − xG', 'Hold-out, normal ±5.8')
    >>> calibration_strip(dataclasses.replace(_example_view(), held_level_ok=False))[-1].note
    'Hold-out, outside ±5.8'
    """
    unit = view.unit.capitalize()
    slope = "n/a" if view.cv_slope is None else f"{view.cv_slope:.2f}"
    name = view.split[0]
    where = "normal" if view.held_level_ok else "outside"
    return (
        viz.Stat(f"{view.cv_n:,}", unit, "Cross-validation"),
        viz.Stat(slope, "Calibration slope", "1.00 is perfect"),
        viz.Stat(f"{view.held_n:,}", unit, name),
        viz.Stat(rows.signed(view.held_gap), f"{view.count_word} − {view.quantity}",
                 f"{name.split()[0]}, {where} \u00b1{view.band_sds * view.held_sd:.1f}"),
    )  # fmt: skip


@dataclass(frozen=True)
class _Axes:
    """The logit plot square in body px."""

    x0: float
    y0: float
    side: float
    lo: float
    hi: float

    def at(self, p: float) -> float:
        return (_logit(p) - self.lo) / (self.hi - self.lo) * self.side

    def xy(self, p: float, rate: float) -> tuple[float, float]:
        return self.x0 + self.at(p), self.y0 + self.at(rate)


def _range(view: ModelView) -> tuple[float, float]:
    cols = ("mean_p", "rate", "low", "high")
    v = np.concatenate([b[c].to_numpy() for b in (view.cv_bins, view.held_bins) for c in cols])
    lo, hi = float(v.min()), float(v.max())
    if not (0 < lo <= hi < 1):
        raise ValueError(f"calibration values must lie in (0, 1) for a logit axis: {lo}, {hi}")
    return _logit(lo) - LOGIT_PAD, _logit(hi) + LOGIT_PAD


def _overlaps(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 0 and min(a[3], b[3]) - max(a[1], b[1]) > 0


def _label_box(
    c: viz.Canvas, text: str, at: tuple[float, float], a: _Axes, taken: list[Box]
) -> tuple[Box, str, str]:
    """Where an outside bin's label goes: right of the marker, else left, below or above,
    the first inside the plot and clear of every band, marker and earlier label (the diagonal
    leaves a gap under it)."""
    w, h = c.measure(text, "caption")
    mx, my = at
    r = HALO_PX / 2 + LABEL_GAP_PX
    candidates: list[tuple[Box, str, str]] = [
        ((mx + r, my - h / 2, mx + r + w, my + h / 2), "left", "center"),
        ((mx - r - w, my - h / 2, mx - r, my + h / 2), "right", "center"),
        ((mx - w / 2, my - r - h, mx + w / 2, my - r), "center", "top"),
        ((mx - w / 2, my + r, mx + w / 2, my + r + h), "center", "bottom"),
    ]
    plot: Box = (a.x0, a.y0, a.x0 + a.side, a.y0 + a.side)

    def inside(b: Box) -> bool:
        return b[0] >= plot[0] and b[1] >= plot[1] and b[2] <= plot[2] and b[3] <= plot[3]

    for box, ha, va in candidates:
        if inside(box) and not any(_overlaps(box, t) for t in taken):
            return box, ha, va
    return next((cand for cand in candidates if inside(cand[0])), candidates[0])


def _anchor(box: Box, ha: str, va: str) -> tuple[float, float]:
    x = {"left": box[0], "right": box[2], "center": (box[0] + box[2]) / 2}[ha]
    y = {"bottom": box[1], "top": box[3], "center": (box[1] + box[3]) / 2}[va]
    return x, y


def _scatter(
    ax: Axes, xy: tuple[float, float], marker: str, size_px: float, face: str, edge: str,
    gid: str, zorder: int,
) -> None:  # fmt: skip
    ax.scatter([xy[0]], [xy[1]], s=(size_px * 72 / viz.DPI) ** 2, marker=marker,
               facecolors=face, edgecolors=edge, linewidths=rows.RING_LW, zorder=zorder, gid=gid,
               clip_on=False)  # fmt: skip


def _diagonal(ax: Axes, a: _Axes, colour: str, holes: Sequence[Box]) -> None:
    """The dashed diagonal as short segments (so each mark's box is small), leaving out the
    dashes under a label (``holes``, padded by ``LABEL_GAP_PX / 2``)."""
    length = a.side * math.sqrt(2)
    dash, gap = DASH_PX
    pad = LABEL_GAP_PX / 2
    padded = [(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad) for b in holes]
    s = 0.0
    while s < length:
        e = min(s + dash, length)
        f0, f1 = s / length * a.side, e / length * a.side
        seg = (a.x0 + f0, a.y0 + f0, a.x0 + f1, a.y0 + f1)
        if not any(_overlaps(seg, h) for h in padded):
            ax.add_line(Line2D([seg[0], seg[2]], [seg[1], seg[3]], color=colour, lw=DIAGONAL_LW,
                               zorder=1, gid="mark-diagonal"))  # fmt: skip
        s = e + gap


def _grid(c: viz.Canvas, ax: Axes, a: _Axes, outcome: str) -> None:
    """Grid lines at the kept ticks, the tick labels and both axis titles."""
    t = c.theme
    ticks = [p for p in TICKS if a.lo <= _logit(p) <= a.hi]
    labels = [_pct(p) for p in ticks]
    pos = [a.at(p) for p in ticks]
    h_tick = c.measure("0123456789%", "caption")[1]
    for axis in ("x", "y"):
        for i in rows.thin(c, labels, pos, axis="x" if axis == "x" else "y"):
            v = pos[i]
            if axis == "x":
                ax.plot([a.x0 + v] * 2, [a.y0, a.y0 + a.side], color=t.line, lw=GRID_LW, zorder=0)
                c.text(a.x0 + v, a.y0 - AXIS_GAP_PX, labels[i], "caption", ax=ax, ha="center",
                       va="top", gid="tick-label")  # fmt: skip
            else:
                ax.plot([a.x0, a.x0 + a.side], [a.y0 + v] * 2, color=t.line, lw=GRID_LW, zorder=0)
                c.text(a.x0 - AXIS_GAP_PX, a.y0 + v, labels[i], "caption", ax=ax, ha="right",
                       va="center", gid="tick-label")  # fmt: skip
    ax.add_patch(Rectangle((a.x0, a.y0), a.side, a.side, fc="none", ec=t.line, lw=GRID_LW,
                           zorder=0, gid="plot-frame"))  # fmt: skip
    tick_w = max(c.measure(x, "caption")[0] for x in labels) if labels else 0.0
    c.text(a.x0 + a.side / 2, a.y0 - 2 * AXIS_GAP_PX - h_tick, f"Predicted {outcome} rate",
           "label", ax=ax, ha="center", va="top", gid="axis-title")  # fmt: skip
    c.text(a.x0 - 2 * AXIS_GAP_PX - tick_w, a.y0 + a.side / 2, f"Observed {outcome} rate",
           "label", ax=ax, ha="center", va="bottom", rotation=90, rotation_mode="anchor",
           gid="axis-title")  # fmt: skip


def _plot(c: viz.Canvas, ax: Axes, top: float, view: ModelView) -> _Axes:
    """The plot square under ``top``: centred with its tick labels and axis titles, as large as
    the body allows."""
    bw, _ = rows.body_px(c)
    lo, hi = _range(view)
    labels = [_pct(p) for p in TICKS if lo <= _logit(p) <= hi]
    tick_w = max((c.measure(x, "caption")[0] for x in labels), default=0.0)
    tick_h = c.measure("0123456789%", "caption")[1]
    title_h = c.measure("Observed", "label")[1]
    left = title_h + tick_w + 3 * AXIS_GAP_PX
    bottom = tick_h + title_h + 3 * AXIS_GAP_PX
    right = tick_w / 2
    side = min(bw - left - right, top - bottom)
    if side < rows.MIN_TRACK_PX:
        raise rows.NoLayout(f"the calibration plot would be {side:.0f} px across")
    x0 = (bw - left - side - right) / 2 + left
    return _Axes(x0, top - side, side, lo, hi)


def _bins(c: viz.Canvas, ax: Axes, a: _Axes, bins: pl.DataFrame, cv: bool) -> list[Box]:
    """One split's bins: bands, then markers (rings for CV, dots for the split); returns the
    boxes of what it drew, for the outside labels."""
    t = c.theme
    boxes: list[Box] = []
    for r in bins.iter_rows(named=True):
        x = a.x0 + a.at(r["mean_p"])
        y_lo, y_hi = a.y0 + a.at(r["low"]), a.y0 + a.at(r["high"])
        if cv:
            ax.plot([x, x], [y_lo, y_hi], color=t.neutral, lw=CV_LW, solid_capstyle="butt",
                    zorder=2, gid="mark-cv-band")  # fmt: skip
            boxes.append((x - CV_LW, y_lo, x + CV_LW, y_hi))
        else:
            half = rows.BAND_PX / 2
            ax.add_patch(Rectangle((x - half, y_lo), rows.BAND_PX, y_hi - y_lo, fc=t.band,
                                   ec="none", zorder=2, gid="band-split"))  # fmt: skip
            for y in (y_lo, y_hi):
                ax.plot([x - half, x + half], [y, y], color=t.mark, lw=rows.CAP_LW,
                        solid_capstyle="butt", zorder=3, gid="mark-band")  # fmt: skip
            boxes.append((x - half, y_lo, x + half, y_hi))
        xy = a.xy(r["mean_p"], r["rate"])
        colour = t.neutral if cv else t.accent
        if not r["inside"]:
            _scatter(ax, xy, "o", HALO_PX, "none", t.ink, "mark-outside", 4)
        if r["clipped"]:
            marker = "^" if r["rate"] > 0.5 else "v"
            _scatter(ax, xy, marker, rows.DOT_PX + 2, "none", colour, "mark-clipped", 6)
        elif cv:
            _scatter(ax, xy, "o", rows.DOT_PX, "none", colour, "mark-cv", 5)
        else:
            rows.dot(ax, xy[0], xy[1], t, "mark-dot")
        size = HALO_PX if not r["inside"] else rows.DOT_PX + 2
        boxes.append((xy[0] - size / 2, xy[1] - size / 2, xy[0] + size / 2, xy[1] + size / 2))
    return boxes


def calibration_chart(
    view: ModelView, *, theme: str | viz.Theme, footer: viz.Footer, logo: viz.StrPath | None
) -> viz.Canvas:
    """The reliability diagram of both splits (portrait).

    Raises
    ------
    ValueError
        No reference row, too many extras, a bins frame missing a column or empty, a value
        outside (0, 1); from ``viz.canvas``; ``rows.NoLayout`` when the key or the plot does
        not fit.

    Examples
    --------
    >>> c = calibration_chart(_example_view(), theme="light", footer=viz.Footer(("whoscored",)),
    ...                       logo=None)
    >>> viz.check(c)
    []
    """
    _check_view(view)
    name = view.split[0]
    clipped = any(bool(b["clipped"].any()) for b in (view.cv_bins, view.held_bins))
    clip = (f"Open triangle: an observed rate of 0 or 100%, drawn at {P_FLOOR:.1%} / "
            f"{1 - P_FLOOR:.1%}")  # fmt: skip
    note = _notes([clip] if clipped else [])
    c = viz.canvas(
        theme, "portrait", title=calibration_title(view),
        subtitle=f"{view.unit.capitalize()} in {view.cv_bins.height} equal-size bins · a bin "
        f"agrees when the {view.level:.0%} match-bootstrap band of its {view.outcome} rate holds "
        "the prediction",
        stats=calibration_strip(view), footer=dataclasses.replace(footer, note=note), logo=logo,
    )  # fmt: skip
    bw, bh = rows.body_px(c)
    ax = c.axes((0.0, 0.0, 1.0, 1.0), gid="calibration")
    ax.set_xlim(0.0, bw)
    ax.set_ylim(0.0, bh)
    band = f"{view.level:.0%} band"
    lines = [[("ring", f"Cross-validation · {view.contexts[0]}"), ("line", band)],
             [("dot", f"{name} · {view.contexts[1]}"), ("band", band)]]  # fmt: skip
    if any(rows.key_width(c, line) > bw for line in lines):
        raise rows.NoLayout(f"the calibration key does not fit {bw:.0f} px in 2 lines")
    key_h = max(rows.BAND_PX, rows.DOT_PX,
                *(c.measure(x, "caption")[1] for line in lines for _, x in line))  # fmt: skip
    y = bh
    for line in lines:
        rows.draw_key(c, ax, line, 0.0, y - key_h / 2)
        y -= key_h + viz.LINE_GAP_PX
    a = _plot(c, ax, y - viz.GAP_PX, view)
    _grid(c, ax, a, view.outcome)
    taken = _bins(c, ax, a, view.cv_bins, cv=True)
    taken += _bins(c, ax, a, view.held_bins, cv=False)
    labels: list[Box] = []
    for bins in (view.cv_bins, view.held_bins):
        for r in bins.filter(~pl.col("inside")).iter_rows(named=True):
            text = f"{r['mean_p']:.0%} predicted, {r['rate']:.0%} observed"
            box, ha, va = _label_box(c, text, a.xy(r["mean_p"], r["rate"]), a, taken)
            taken.append(box)
            labels.append(box)
            c.text(*_anchor(box, ha, va), text, "caption", ax=ax, ha=ha, va=va,
                   gid="outside-label")  # fmt: skip
    _diagonal(ax, a, c.theme.neutral, labels)
    return c


# --- Comparison -------------------------------------------------------------------------------


def _reference(role: str) -> str:
    """A row's role in a sentence: "the standard baseline"; a possessive ("StatsBomb's xG")
    stays as it is."""
    role = role.split(" (")[0]
    return role if "'" in role else "the " + role[0].lower() + role[1:]


def _and(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def comparison_titles(view: ModelView) -> tuple[str, str]:
    """The split's verdict on each reference, by its gain interval: "clearly better than" when
    it lies above 0, "clearly behind" below, "level with" across it (the panels show
    cross-validation beside it). The first title names the references; the second counts them,
    for when the names do not fit 2 lines. The split's name is its first word ("Hold-out").

    Examples
    --------
    >>> comparison_titles(_example_view())
    ('Hold-out: clearly better than the best regression, level with the standard baseline', 'Hold-out: clearly better than 1 of 2 references, level with 1')
    """  # noqa: E501
    groups: dict[str, list[str]] = {"clearly better than": [], "level with": [],
                                    "clearly behind": []}  # fmt: skip
    for r in view.rows:
        _, low, high = r.held
        verdict = ("clearly better than" if low > 0 else "clearly behind" if high < 0
                   else "level with")  # fmt: skip
        groups[verdict].append(_reference(r.role))
    split, k = view.split[0].split()[0], len(view.rows)
    named = [f"{verdict} {_and(names)}" for verdict, names in groups.items() if names]
    counted, first = [], True
    for verdict, names in groups.items():
        if names:
            of = f" of {k} reference{'' if k == 1 else 's'}" if first else ""
            counted.append(f"{verdict} {len(names)}{of}")
            first = False
    return f"{split}: {', '.join(named)}", f"{split}: {', '.join(counted)}"


def percent(gain: Gain, log_loss: float) -> Gain:
    """A log-loss gain (and its interval) as a percent of the reference's log loss, the
    reference being ``log_loss + gain`` (the model's plus the gain); one fixed denominator, so
    the interval scales with the point.

    Examples
    --------
    >>> [round(v, 2) for v in percent((0.01, 0.0, 0.02), 0.24)]
    [4.0, 0.0, 8.0]
    """
    ref = log_loss + gain[0]
    return (100 * gain[0] / ref, 100 * gain[1] / ref, 100 * gain[2] / ref)


def _signed_pct(v: float) -> str:
    return rows.signed(v) + "%"


def comparison_strip(view: ModelView) -> tuple[viz.Stat, ...]:
    """The head's log loss, then one stat per ``Extra`` (as :func:`percent`).

    Examples
    --------
    >>> [(s.value, s.note) for s in comparison_strip(_example_view())]
    [('0.2689', 'CV · Hold-out 0.2701'), ('+28.7%', 'CV · Hold-out +29.5%')]
    """
    name = view.split[0].split()[0]  # "Hold-out": the column's width
    return (
        viz.Stat(f"{view.cv_log_loss:.4f}", "Log loss", f"CV · {name} {view.held_log_loss:.4f}"),
        *(viz.Stat(_signed_pct(percent((e.cv, 0.0, 0.0), view.cv_log_loss)[0]),
                   f"Better than {e.label}",
                   f"CV · {name} {_signed_pct(percent((e.held, 0.0, 0.0), view.held_log_loss)[0])}")
          for e in view.extras),
    )  # fmt: skip


def _gain_panel(
    view: ModelView, heading: str, pick: Callable[[GainRow], Gain], *, last: bool
) -> rows.Panel:
    """One split's gain rows in percent (``pick`` returns them so); the caption (it holds for
    both panels) under the last only."""
    items = tuple(
        rows.Row(label=r.role, sublabel=r.title, value=_signed_pct(pick(r)[0]), point=pick(r)[0],
                 low=pick(r)[1], high=pick(r)[2])
        for r in view.rows
    )  # fmt: skip
    return rows.Panel(
        f"{heading}: % lower log loss than each reference",
        (("dot", "Improvement"), ("band", f"{view.level:.0%} interval")), items,
        caption=f"Right of 0: the {view.quantity} model predicts the same {view.unit} better "
        "(lower log loss) than the reference." if last else "",
        scale_row=True, unit="%",
    )  # fmt: skip


def comparison_chart(
    view: ModelView,
    *,
    theme: str | viz.Theme,
    footer: viz.Footer,
    logo: viz.StrPath | None,
    size: str,
) -> viz.Canvas:
    """The gain over each reference, cross-validation then the split, on one scale.

    Raises
    ------
    ValueError
        As :func:`calibration_chart`; ``rows.NoLayout`` when the panels do not fit the body.

    Examples
    --------
    >>> c = comparison_chart(_example_view(), theme="dark", footer=viz.Footer(("whoscored",)),
    ...                      logo=None, size="portrait")
    >>> viz.check(c)
    []
    """
    _check_view(view)
    named, counted = comparison_titles(view)
    kw = dict(
        subtitle=f"The {view.quantity} model against each reference, in % of the reference's log "
        f"loss · {view.level:.0%} intervals from {view.n_boot:,} match resamples",
        stats=comparison_strip(view), footer=dataclasses.replace(footer, note=_notes(view.notes)),
        logo=logo,
    )  # fmt: skip
    try:
        c = viz.canvas(theme, size, title=named, **kw)  # type: ignore[arg-type]
    except ValueError as e:
        if not str(e).startswith("title "):
            raise
        c = viz.canvas(theme, size, title=counted, **kw)  # type: ignore[arg-type]
    bw, bh = rows.body_px(c)
    cv = {r.role: percent(r.cv, view.cv_log_loss) for r in view.rows}
    held = {r.role: percent(r.held, view.held_log_loss) for r in view.rows}
    scale = rows.scale_of([v for g in (*cv.values(), *held.values()) for v in g])
    panels = (_gain_panel(view, "Cross-validation", lambda r: cv[r.role], last=False),
              _gain_panel(view, view.split[0], lambda r: held[r.role], last=True))  # fmt: skip
    measured = [rows.place(c, p, (0.0, 0.0, bw, bh)) for p in panels]
    need = sum(m.height for m in measured) + viz.GAP_PX * (len(measured) - 1)
    if need > bh + viz.EPS_PX:
        raise rows.NoLayout(f"the gain panels need {need:.0f} px, over the {bh:.0f} px body")
    y = bh
    for gid, m in zip(("gains-cv", "gains-split"), measured, strict=True):
        rows.draw_panel(c, (0.0, y - m.height, bw, m.height), m, scale=scale, gid=gid)
        y -= m.height + viz.GAP_PX
    return c


# --- Doctest fixture --------------------------------------------------------------------------


def _example_bins(rates: Sequence[float], inside: Sequence[bool]) -> pl.DataFrame:
    means = (0.03, 0.1, 0.35)
    return pl.DataFrame({
        "bin": [0, 1, 2], "shots": [300, 300, 300], "mean_p": list(means), "rate": list(rates),
        "low": [r * 0.6 for r in rates], "high": [min(0.99, r * 1.4) for r in rates],
        "inside": list(inside), "clipped": [False] * 3,
    }, schema_overrides={"bin": pl.Int64, "shots": pl.Int64})  # fmt: skip


def _example_view() -> ModelView:
    """A made-up view: three bins per split, one hold-out bin outside its band."""
    return ModelView(
        "xG", "goal", "shots", ("Hold-out", "the hold-out", "hold-out"),
        ("12 competition-seasons", "Test League 2015/2016"),
        _example_bins((0.032, 0.095, 0.36), (True, True, True)),
        _example_bins((0.031, 0.104, 0.25), (True, True, False)), 9000, 900, 1.02, -3.2, 2.9,
        True, 2.0, 0.2689, 0.2701,
        (GainRow("Best regression", "LR splines", (0.002, 0.001, 0.003),
                 (0.004, 0.0005, 0.007)),
         GainRow("Standard baseline (S0)", "LR distance + angle", (0.010, -0.001, 0.02),
                 (0.012, -0.002, 0.03))),
        (Extra("xG only", 0.108, 0.113),), 0.95, 2000,
    )  # fmt: skip
