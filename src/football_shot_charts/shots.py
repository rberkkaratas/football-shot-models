"""The team shot chart and the player shot profile: frames in, a themed ``viz.Canvas`` out.

Both charts answer *who shoots from where and how good the chances were*: a pitch cropped to
the shots (marker area proportional to xG, goals drawn above misses), a stat strip, and a
panel of goals against xG per zone (team) or per body part (player). No IO, no fit: the
frames carry the ``xg-gbm`` hold-out predictions in an ``xg`` column
(:func:`football_shot_charts.render.holdout_xg`).

**Titles are computed statements** (:func:`team_title`, :func:`player_title`), never typed.
A sentence about goals against xG is made only when the gap lies outside the model's own
noise band (``metrics.within`` on the Poisson-binomial SD); even then it says "more/fewer than
the model", never why, because the model's known misprice by zone (the PL 2015/16 hold-out
over-predicts within 6 m) can explain a gap as well as finishing can. Inside the band, the team
title is the box share of goals with the league's share beside it.

**The league band.** A team or player who shoots from where the model misprices inherits that
misprice. Each shot's xG is rescaled by the rest of the league's goals per xG in its zone
(``r_zone``, own rows left out). The verdict reads that band (``zone_bias``: outside the
model's band, inside the league's).

**One reference per row.** A panel row draws the goals dot against the model's xG tick; the
league-rate expectation is drawn as a band only (± ``metrics.BAND_SDS`` SD, rows of at least
``metrics.MIN_JUDGED_SHOTS`` shots), never as a second tick, and a one-line key of the drawn
marks sits under the heading (two ticks told apart by shade were unreadable on a phone). The
model's level on the rest of the league is a footer sentence after the model credit, not a stat
tile (readers took it for the team's own conversion).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import pitch, schema, viz
from matplotlib.axes import Axes

from football_shot_charts import rows
from football_shot_models import metrics

StrPath = viz.StrPath
Verdict = Literal["beyond", "zone_bias", "noise", "too_few"]

ZONES: Final = ("six_yard", "rest_of_box", "outside")
ZONE: Final = pl.Enum(ZONES)
ZONE_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    dict(zip(ZONES, ("Six-yard box", "Rest of the box", "Outside the box"), strict=True))
)
BODY_PART_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {"right_foot": "Right foot", "left_foot": "Left foot", "head": "Head", "other": "Other"}
)
NOT_RECORDED: Final = "Not recorded"
ALL_SHOTS: Final = "All shots"
_C3: Final = ("shot_id", "provider", "team_id", "player_id", "x", "y", "is_goal", "is_penalty",
               "period", "body_part")  # fmt: skip
COLUMNS: Final[Mapping[str, pl.DataType]] = MappingProxyType(
    {**{c: schema.SHOT_SCHEMA[c] for c in _C3}, "xg": pl.Float64()}
)
NULLABLE: Final = frozenset({"body_part"})
MIN_PITCH_PX: Final = 400
PANEL_STEP_PX: Final = 8
CAPTION_LINES: Final = 4  # the player's zone_bias sentence under the rows
# the row panel lives in ``rows`` (one engine for every chart)
GIDS: Final = rows.Gids(band="band-league", cap="mark-band", tick="mark-xg", dot="mark-goals-dot")
ZONE_HEADING: Final = "Goals and xG by zone"
BODY_PART_HEADING: Final = "Goals and xG by body part"


# --- Frames, lines and verdicts ---------------------------------------------------------------


def zone() -> pl.Expr:
    """The shot's zone from canonical ``x``/``y``: lines belong to the area (IFAB).

    Examples
    --------
    >>> pl.DataFrame({"x": [99.5, 88.5, 88.4], "y": [24.84, 13.84, 34.0]}).select(
    ...     zone()).to_series().to_list()
    ['six_yard', 'rest_of_box', 'outside']
    """
    x, y = pl.col("x"), pl.col("y")
    six = (x >= pitch.METRE_X[7]) & y.is_between(pitch.METRE_Y[2], pitch.METRE_Y[5])
    box = (x >= pitch.METRE_X[5]) & y.is_between(pitch.METRE_Y[1], pitch.METRE_Y[6])
    return (
        pl.when(six).then(pl.lit("six_yard")).when(box).then(pl.lit("rest_of_box"))
        .otherwise(pl.lit("outside")).cast(ZONE).alias("zone")
    )  # fmt: skip


def _validate(frame: pl.DataFrame, what: str) -> None:
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{what} is missing column(s): {', '.join(missing)}")
    for col, dtype in COLUMNS.items():
        if frame.schema[col] != dtype:
            raise ValueError(f"{what}: column {col!r} has dtype {frame.schema[col]}, not {dtype}")
        if col not in NULLABLE and (n := frame[col].null_count()):
            raise ValueError(f"{what}: column {col!r} has {n} null value(s)")
    for col in ("x", "y", "xg"):
        if n := int((~frame[col].is_finite()).sum()):
            raise ValueError(f"{what}: column {col!r} has {n} NaN or infinite value(s)")
    if n := int(((frame["xg"] < 0) | (frame["xg"] > 1)).sum()):
        raise ValueError(f"{what}: {n} xg value(s) outside [0, 1]")
    if n := frame.height - frame["shot_id"].n_unique():
        raise ValueError(f"{what}: {n} duplicate shot_id value(s)")
    if n := int((frame["is_penalty"] | (frame["period"] == 5)).sum()):
        raise ValueError(
            f"{what}: {n} penalty or shoot-out row(s); charts draw the xg-gbm population"
        )


def check_inputs(shots: pl.DataFrame, league: pl.DataFrame) -> None:
    """Refuse frames a chart cannot draw honestly (both checked by one rule set).

    Raises
    ------
    ValueError
        A missing column or wrong dtype (shot-table dtypes, ``xg`` Float64); a null outside
        ``body_part``; non-finite ``x``/``y``/``xg`` or ``xg`` outside [0, 1]; a duplicate
        ``shot_id``; a penalty or shoot-out row; an empty ``shots``; a ``shots`` id missing
        from ``league``; more than one provider.

    Examples
    --------
    >>> check_inputs(pl.DataFrame({"shot_id": ["sb:1"]}), pl.DataFrame())
    Traceback (most recent call last):
    ...
    ValueError: shots is missing column(s): provider, team_id, player_id, x, y, is_goal, ...
    """
    _validate(shots, "shots")
    _validate(league, "league")
    if shots.height == 0:
        raise ValueError("shots is empty")
    if n := shots.join(league, on="shot_id", how="anti").height:
        raise ValueError(f"{n} shots row(s) are not in league (league holds every kept row)")
    if (k := pl.concat([shots["provider"], league["provider"]]).n_unique()) > 1:
        raise ValueError(f"{k} providers in shots and league; a chart draws one")


def _rest(shots: pl.DataFrame, league: pl.DataFrame) -> pl.DataFrame:
    return league.join(shots.select("shot_id"), on="shot_id", how="anti")


def _adjusted(shots: pl.DataFrame, league: pl.DataFrame) -> pl.DataFrame:
    """Shots with ``zone``, ``r`` (the rest of the league's goals per xG in the zone, 1 when
    that xG is 0) and ``p_adj = clip(xg * r, 0, 1)``."""
    rates = (
        _rest(shots, league).with_columns(zone())
        .group_by("zone").agg(pl.col("is_goal").sum().alias("g"), pl.col("xg").sum().alias("x"))
        .select("zone", r=pl.when(pl.col("x") > 0).then(pl.col("g") / pl.col("x")).otherwise(1.0))
    )  # fmt: skip
    return (
        shots.with_columns(zone())
        .join(rates, on="zone", how="left", maintain_order="left")
        .with_columns(pl.col("r").fill_null(1.0))
        .with_columns(p_adj=(pl.col("xg") * pl.col("r")).clip(0.0, 1.0))
    )


@dataclass(frozen=True)
class Line:
    """Goals against xG for one group of shots.

    Attributes
    ----------
    label : str
    n, goals : int
    xg, sd : float
        Σ xG and its Poisson-binomial SD ``√Σ p(1 − p)``.
    adj_xg, sd_adj : float
        The same with each shot's xG at the rest of the league's rate in its zone.
    """

    label: str
    n: int
    goals: int
    xg: float
    sd: float
    adj_xg: float
    sd_adj: float

    @property
    def gap(self) -> float:
        """``goals − xg``: the sign every title, strip and test uses."""
        return self.goals - self.xg


def _line(label: str, f: pl.DataFrame) -> Line:
    y = f["is_goal"].to_numpy()
    p, q = f["xg"].to_numpy(), f["p_adj"].to_numpy()
    sd, sd_adj = metrics.poisson_binomial(y, p)[1], metrics.poisson_binomial(y, q)[1]
    return Line(label, f.height, int(y.sum()), float(p.sum()), sd, float(q.sum()), sd_adj)


def summary(shots: pl.DataFrame, league: pl.DataFrame) -> Line:
    """Every shot of ``shots`` as one :class:`Line` (label ``ALL_SHOTS``).

    Examples
    --------
    >>> s = _example()
    >>> line = summary(s, s)
    >>> (line.n, line.goals, round(line.xg, 2), round(line.gap, 2))
    (3, 1, 0.9, 0.1)
    """
    check_inputs(shots, league)
    return _line(ALL_SHOTS, _adjusted(shots, league))


def table(
    shots: pl.DataFrame, league: pl.DataFrame, by: Literal["zone", "body_part"]
) -> tuple[Line, ...]:
    """One :class:`Line` per zone (``ZONES`` order) or body part (``BODY_PARTS`` order, then
    ``NOT_RECORDED`` for a null); only groups with shots.

    Examples
    --------
    >>> s = _example()
    >>> [(t.label, t.n, t.goals) for t in table(s, s, "zone")]
    [('Six-yard box', 1, 1), ('Outside the box', 2, 0)]
    """
    check_inputs(shots, league)
    f = _adjusted(shots, league)
    if by == "zone":
        groups = [(ZONE_LABELS[z], f.filter(pl.col("zone") == z)) for z in ZONES]
    elif by == "body_part":
        groups = [
            (BODY_PART_LABELS[b], f.filter(pl.col("body_part") == b)) for b in schema.BODY_PARTS
        ]
        groups.append((NOT_RECORDED, f.filter(pl.col("body_part").is_null())))
    else:
        raise ValueError(f"by must be 'zone' or 'body_part', not {by!r}")
    return tuple(_line(label, g) for label, g in groups if g.height)


def league_rate(shots: pl.DataFrame, league: pl.DataFrame) -> float | None:
    """Goals per xG of the league rows not in ``shots``; ``None`` when their xG is 0.

    Examples
    --------
    >>> s = _example()
    >>> league_rate(s.head(1), s)
    0.0
    """
    check_inputs(shots, league)
    rest = _rest(shots, league)
    xg = float(rest["xg"].sum())
    return None if xg == 0 else int(rest["is_goal"].sum()) / xg


def verdict(line: Line) -> Verdict:
    """``too_few`` under ``metrics.MIN_JUDGED_SHOTS`` shots; ``noise`` when goals are within
    the model's band; ``zone_bias`` when within the league-rate band only; else ``beyond``.
    The constants are read when called.

    Examples
    --------
    >>> verdict(Line("x", 100, 20, 10.0, 3.0, 19.0, 3.0)), verdict(Line("x", 99, 20, 10.0, 3.0,
    ...     10.0, 3.0))
    ('zone_bias', 'too_few')
    """
    if line.n < metrics.MIN_JUDGED_SHOTS:
        return "too_few"
    if metrics.within(line.gap, line.sd):
        return "noise"
    if metrics.within(line.goals - line.adj_xg, line.sd_adj):
        return "zone_bias"
    return "beyond"


# --- Words ------------------------------------------------------------------------------------


def _n(count: int, noun: str, plural: str | None = None) -> str:
    return f"{count} {noun if count == 1 else plural or noun + 's'}"


def _beyond(line: Line) -> str:
    word = "more" if line.gap > 0 else "fewer"
    return (
        f"{_n(line.goals, 'non-penalty goal')}, {abs(line.gap):.1f} {word} than the model's "
        f"{line.xg:.1f} xG (SD {line.sd:.1f})"
    )


def team_title(shots: pl.DataFrame, league: pl.DataFrame) -> str:
    """The team chart's statement: goals against the model when outside its noise band, else
    the box share of goals with the rest of the league's share (and its rank when strictly
    the league's highest or lowest among teams with a goal).

    Examples
    --------
    >>> s = _example()
    >>> team_title(s.filter(pl.col("team_id") == "sb:1"), s)  # the rest scored no goal
    '1 of 1 non-penalty goals came from inside the box'
    """
    line = summary(shots, league)
    if verdict(line) == "beyond":
        return _beyond(line)
    if line.goals == 0:
        return f"{_n(line.n, 'non-penalty shot')}, {line.xg:.1f} xG and no goal"
    inside = pl.col("zone") != "outside"
    box = int(shots.with_columns(zone()).filter(inside)["is_goal"].sum())
    rest = _rest(shots, league).filter("is_goal").with_columns(zone())
    others = rest.group_by("team_id").agg(inside.mean().alias("share"))["share"].to_list()
    share, rank = box / line.goals, ""
    if others and share > max(others):
        rank = ", the league's highest share"
    elif others and share < min(others):
        rank = ", the league's lowest share"
    p = f" (rest of league: {rest.select(inside.mean()).item():.0%})" if rest.height else ""
    return f"{box} of {line.goals} non-penalty goals came from inside the box{rank}{p}"


def player_title(shots: pl.DataFrame, league: pl.DataFrame) -> str:
    """The player profile's statement, by :func:`verdict`: goals against the model when outside
    both bands, "in line with" the model or the league's zone rate otherwise, and "too few to
    judge" under ``metrics.MIN_JUDGED_SHOTS`` shots.

    Examples
    --------
    >>> s = _example()
    >>> player_title(s.head(2), s)
    '1 non-penalty goal from 0.5 xG on 2 shots: too few to judge'
    """
    line = summary(shots, league)
    head = f"{_n(line.goals, 'non-penalty goal')} from {line.xg:.1f} xG"
    match verdict(line):
        case "beyond":
            return _beyond(line)
        case "noise":
            return f"{head}: in line with the model (SD {line.sd:.1f})"
        case "zone_bias":
            return (
                f"{head}: in line with the league's rate from the same zones (SD {line.sd_adj:.1f})"
            )
        case _:
            return f"{head} on {_n(line.n, 'shot')}: too few to judge"


def stats(
    shots: pl.DataFrame, league: pl.DataFrame, *, per: Literal["xg", "goals"] = "xg"
) -> tuple[viz.Stat, ...]:
    """The stat strip: shots, goals, xG, goals − xG with its SD, then xG per shot (``per="xg"``,
    the team chart) or goals per shot (``per="goals"``, the player profile) at 2 dp.

    Examples
    --------
    >>> s = _example()
    >>> [(t.value, t.label, t.note) for t in stats(s.head(2), s)][3:]
    [('+0.5', 'Goals − xG', 'SD 0.5'), ('0.25', 'xG per shot', None)]
    >>> stats(s.head(2), s, per="goals")[-1].label
    'Goals per shot'
    """
    line = summary(shots, league)
    if per == "xg":
        last = viz.Stat(f"{line.xg / line.n:.2f}", "xG per shot")
    elif per == "goals":
        last = viz.Stat(f"{line.goals / line.n:.2f}", "Goals per shot")
    else:
        raise ValueError(f"per must be 'xg' or 'goals', not {per!r}")
    return (
        viz.Stat(str(line.n), "Shots"),
        viz.Stat(str(line.goals), "Goals"),
        viz.Stat(f"{line.xg:.1f}", "xG"),
        viz.Stat(rows.signed(line.gap), "Goals \u2212 xG", f"SD {line.sd:.1f}"),
        last,
    )


def model_level(
    shots: pl.DataFrame, league: pl.DataFrame, subject: str = "The model"
) -> str | None:
    """The footer sentence on the model's level over the rest of the league: "over-" or
    "under-predicts" only outside its noise band (``metrics.within``), else "is in line with";
    ``None`` when the rest has no xG. ``subject`` opens it.

    Examples
    --------
    >>> s = _example()
    >>> model_level(s.head(1), s)  # the rest: 0 goals from 0.45 xG, SD 0.6
    'The model is in line with the rest of the league: 0.00 goals per xG'
    >>> model_level(s, s) is None
    True
    """
    rate = league_rate(shots, league)
    if rate is None:
        return None
    rest = _rest(shots, league)
    gap, sd = metrics.poisson_binomial(rest["is_goal"].to_numpy(), rest["xg"].to_numpy())
    how = "is in line with" if metrics.within(gap, sd) else (
        "over-predicts" if gap > 0 else "under-predicts")  # fmt: skip
    return f"{subject} {how} the rest of the league: {rate:.2f} goals per xG"


def key(lines: Sequence[Line]) -> tuple[tuple[str, str], ...]:
    """The panel key, one ``(glyph, label)`` per mark the rows draw: the goals dot, the xG
    tick, and the league-rate band only when a row has at least ``metrics.MIN_JUDGED_SHOTS``
    shots (constants read when called).

    Examples
    --------
    >>> key([Line("x", 99, 1, 1.0, 1.0, 1.0, 1.0)])
    (('dot', 'Goals'), ('tick', 'xG'))
    >>> key([Line("x", 100, 1, 1.0, 1.0, 1.0, 1.0)])[-1]
    ('band', 'League rate ±2 SD')
    """
    items = [("dot", "Goals"), ("tick", "xG")]
    if any(ln.n >= metrics.MIN_JUDGED_SHOTS for ln in lines):
        items.append(("band", f"League rate ±{metrics.BAND_SDS:g} SD"))
    return tuple(items)


def caption(line: Line | None = None) -> str:
    """The sentence under the panel rows: with the player's ``zone_bias`` line, the shots'
    worth at the league's zone rates; otherwise none (``""``).

    Examples
    --------
    >>> caption(Line("x", 100, 20, 10.0, 3.0, 19.04, 3.0))
    "At the league's rate per zone these shots are worth 19.0 goals."
    >>> caption()
    ''
    """
    if line is None:
        return ""
    return f"At the league's rate per zone these shots are worth {line.adj_xg:.1f} goals."


def mispriced_zones(shots: pl.DataFrame, league: pl.DataFrame) -> tuple[str, ...]:
    """The ``ZONE_LABELS`` of the zones where the rest of the league's goals fall outside the
    model's band (``metrics.within`` on its Poisson-binomial SD), in ``ZONES`` order.

    Examples
    --------
    >>> s = _example()
    >>> mispriced_zones(s.head(1), s)  # the rest: 0 goals from 0.45 xG outside the box
    ()
    """
    check_inputs(shots, league)
    rest = _rest(shots, league).with_columns(zone())
    out = []
    for z in ZONES:
        r = rest.filter(pl.col("zone") == z)
        if not metrics.within(*metrics.poisson_binomial(r["is_goal"].to_numpy(),
                                                        r["xg"].to_numpy())):  # fmt: skip
            out.append(ZONE_LABELS[z])
    return tuple(out)


def unjudged_caption(lines: Sequence[Line], mispriced: Sequence[str]) -> str:
    """The zone panel's sentence on a row that draws no band (under ``metrics.MIN_JUDGED_SHOTS``
    shots) in a zone the model misprices on the rest of the league (:func:`mispriced_zones`):
    its shots at the league's rate there (``Line.adj_xg``), so the misprice (the PL 2015/16
    six-yard box) is not read as finishing; ``""`` when there is no such row.

    Examples
    --------
    >>> six = Line("Six-yard box", 48, 9, 15.3, 3.0, 13.62, 3.0)
    >>> unjudged_caption([six], ["Six-yard box"])
    "Not judged (under 100 shots), at the league's rate in the zone: Six-yard box 13.6 goals."
    >>> unjudged_caption([six], [])
    ''
    """
    least = metrics.MIN_JUDGED_SHOTS
    parts = [f"{ln.label} {ln.adj_xg:.1f} goals" for ln in lines
             if ln.n < least and ln.label in mispriced]  # fmt: skip
    if not parts:
        return ""
    return (f"Not judged (under {least} shots), at the league's rate in the zone: "
            f"{'; '.join(parts)}.")  # fmt: skip


# --- Layout -----------------------------------------------------------------------------------


def _panel(heading: str, lines: Sequence[Line], text: str) -> rows.Panel:
    """The goals-against-xG panel as rows: the goals dot, the xG tick and, on judged rows, the
    league-rate band (± ``metrics.BAND_SDS`` SD, floored at 0)."""
    least, b = metrics.MIN_JUDGED_SHOTS, metrics.BAND_SDS
    items = tuple(
        rows.Row(
            label=ln.label, sublabel=_n(ln.n, "shot") + (" · not judged" if ln.n < least else ""),
            value=_n(ln.goals, "goal"), subvalue=f"{ln.xg:.1f} xG", point=float(ln.goals),
            low=max(0.0, ln.adj_xg - b * ln.sd_adj) if ln.n >= least else None,
            high=ln.adj_xg + b * ln.sd_adj if ln.n >= least else None, tick=ln.xg,
        )
        for ln in lines
    )  # fmt: skip
    return rows.Panel(heading, key(lines), items, caption=text, caption_lines=CAPTION_LINES,
                      guide="rail", gids=GIDS)  # fmt: skip


def _scale(lines: Sequence[Line]) -> tuple[float, float]:
    """The track's x range in goals, ``[0, x_max]`` shared by the rows."""
    least, b = metrics.MIN_JUDGED_SHOTS, metrics.BAND_SDS
    top = max(max(ln.goals, ln.xg, ln.adj_xg + (b * ln.sd_adj if ln.n >= least else 0.0))
              for ln in lines) or 1.0  # fmt: skip
    return (0.0, top)


def _on_figure(c: viz.Canvas, rect: rows.Rect) -> rows.Rect:
    left, bottom, width, height = c.body
    f = rows.fraction(c, rect)
    return (left + f[0] * width, bottom + f[1] * height, f[2] * width, f[3] * height)


def _place_map(
    c: viz.Canvas, rect: rows.Rect, f: pl.DataFrame
) -> tuple[Axes, npt.NDArray[np.bool_], float] | None:
    """Pitch, crop and marker scale at the top of the map block, the legend's room under it;
    ``None`` (nothing left drawn) when the pitch axes would be under ``MIN_PITCH_PX``.

    A block taller than the crop needs would make ``crop_pitch`` draw deeper than the 35 m
    cap (into empty grass, past shots the note counts as not drawn): the pitch axes then
    shrink to the capped depth, never under ``MIN_PITCH_PX``, and the rest stays empty.
    """
    t, top = c.theme, rect[1] + rect[3]
    x, y, p = (f[k].to_numpy() for k in ("x", "y", "xg"))
    floor = max(pitch.PITCH_LENGTH - viz.CROP_MAX_DEPTH, float(x.min()) - viz.CROP_PAD)
    ax = c.axes(rows.fraction(c, rect), gid="pitch")
    viz.draw_half_pitch(ax, colour=t.line, box_colour=t.box_line)
    reserve = 0.0
    for _ in range(4):
        height = rect[3] - reserve
        if min(rect[2], height) < MIN_PITCH_PX:
            ax.remove()
            return None
        ax.set_position(_on_figure(c, (rect[0], top - height, rect[2], height)))
        mask = viz.crop_pitch(ax, x, y)
        lo, hi = ax.get_ylim()
        if lo < floor - 1e-9:
            height = max(float(MIN_PITCH_PX), height * (hi - floor) / (hi - lo))
            ax.set_position(_on_figure(c, (rect[0], top - height, rect[2], height)))
            mask = viz.crop_pitch(ax, x, y)
        area = viz.ink_scale(float(p[mask].sum()), ax)
        try:
            need = c.legend_height(viz.shot_keys(area), c.legend_width(ax))
        except ValueError:  # the keys need a third row under this map: the layout fails
            break
        if height + need <= rect[3]:
            return ax, mask, area
        reserve = need
    ax.remove()
    return None


def _layouts(c: viz.Canvas, panel: rows.Panel) -> list[tuple[rows.Measured, rows.Rect, rows.Rect]]:
    """Where the panel may go, in order: under the map (portrait only) when the map keeps
    ``MIN_PITCH_PX`` of height, then beside it at the narrowest inner width (``PANEL_STEP_PX``
    steps) at which the panel fits the body. Rects are body px ``(left, bottom, w, h)``."""
    w, h = viz.SIZES[c.size]
    bw, bh = c.body[2] * w, c.body[3] * h
    out: list[tuple[rows.Measured, rows.Rect, rows.Rect]] = []
    if c.size == "portrait":
        under = rows.measure(c, panel, bw - 2 * rows.PANEL_PAD_PX)
        if under is not None and bh - under.height - viz.GAP_PX >= MIN_PITCH_PX:
            top = under.height + viz.GAP_PX
            out.append((under, (0.0, 0.0, bw, under.height), (0.0, top, bw, bh - top)))
    inner = float(rows.MIN_TRACK_PX)
    while inner <= bw - MIN_PITCH_PX - viz.GAP_PX - 2 * rows.PANEL_PAD_PX:
        side = rows.measure(c, panel, inner)
        if side is not None and side.height <= bh:
            pw = inner + 2 * rows.PANEL_PAD_PX
            map_rect = (0.0, 0.0, bw - pw - viz.GAP_PX, bh)
            out.append((side, (bw - pw, bh - side.height, pw, side.height), map_rect))
            break
        inner += PANEL_STEP_PX
    return out


def _arrange(
    c: viz.Canvas, panel: rows.Panel, shots: pl.DataFrame
) -> tuple[rows.Measured, rows.Rect, tuple[Axes, npt.NDArray[np.bool_], float]]:
    """The first layout whose map keeps its pitch, with the map placed."""
    for measured, panel_rect, map_rect in _layouts(c, panel):
        if (placed := _place_map(c, map_rect, shots)) is not None:
            return measured, panel_rect, placed
    raise rows.NoLayout(f"no layout fits the {c.size} canvas (pitch >= {MIN_PITCH_PX} px)")


def _note(excluded: Sequence[str], hidden: pl.DataFrame | None) -> str | None:
    parts = list(excluded)
    if hidden is not None and hidden.height:
        goals, xg = int(hidden["is_goal"].sum()), float(hidden["xg"].sum())
        parts.append(
            f"{_n(hidden.height, 'shot')} outside the drawn area ({_n(goals, 'goal')}, {xg:.1f} xG)"
        )
    return "Not shown: " + " · ".join(parts) if parts else None


def _with_level(footer: viz.Footer, shots: pl.DataFrame, league: pl.DataFrame) -> viz.Footer:
    """The footer with :func:`model_level` after the model credit: the footer's sentence about
    the model (the not-shown note is capped at 2 lines by core and can fill them). Its subject
    is always "The model": on a team chart "It" could be read as the team."""
    level = model_level(shots, league)
    model = ". ".join(x for x in (footer.model, level) if x)
    return dataclasses.replace(footer, model=model or None)


def _chart(
    shots: pl.DataFrame,
    *,
    title: str,
    subtitle: str,
    strip: Sequence[viz.Stat],
    heading: str,
    lines: Sequence[Line],
    text: str,
    theme: str | viz.Theme,
    size: str,
    footer: viz.Footer,
    excluded: Sequence[str],
    logo: StrPath | None,
) -> viz.Canvas:
    """The canvas both charts share.

    The footer note counts the shots the crop leaves out, and the crop depends on the body
    the note leaves: lay out, crop, and redo with the counted note until the note states what
    the map drew (a second pass at most in practice; a third that still differs raises).
    When no layout fits with the stat strip (a square profile with five body-part rows), the
    chart is laid out without it: the title already states goals, xG and the SD.
    """
    if isinstance(excluded, str):
        raise TypeError("excluded must be a sequence of note parts, not a str")
    try:
        return _settle(shots, title, subtitle, strip, heading, lines, text, theme, size, footer,
                       excluded, logo)  # fmt: skip
    except rows.NoLayout:
        if not strip:
            raise
    return _settle(shots, title, subtitle, (), heading, lines, text, theme, size, footer,
                   excluded, logo)  # fmt: skip


def _settle(
    shots: pl.DataFrame,
    title: str,
    subtitle: str,
    strip: Sequence[viz.Stat],
    heading: str,
    lines: Sequence[Line],
    text: str,
    theme: str | viz.Theme,
    size: str,
    footer: viz.Footer,
    excluded: Sequence[str],
    logo: StrPath | None,
) -> viz.Canvas:
    note = _note(excluded, None)
    for _ in range(3):
        c = viz.canvas(theme, size, title=title, subtitle=subtitle, stats=strip, logo=logo,
                       footer=dataclasses.replace(footer, note=note))  # fmt: skip
        measured, panel_rect, (ax, mask, area) = _arrange(c, _panel(heading, lines, text), shots)
        counted = _note(excluded, shots.filter(pl.Series(~mask)))
        if counted == note:
            x, y, p, g = (shots[k].to_numpy() for k in ("x", "y", "xg", "is_goal"))
            viz.draw_shots(ax, x[mask], y[mask], p[mask], g[mask], c.theme, area_per_xg=area)
            c.legend(viz.shot_keys(area), ax=ax)
            gid = "zones" if heading == ZONE_HEADING else "body-parts"
            rows.draw_panel(c, panel_rect, measured, scale=_scale(lines), gid=gid)
            return c
        note = counted
    raise ValueError("the footer note and the crop did not settle in 3 passes")


def team_chart(
    shots: pl.DataFrame,
    league: pl.DataFrame,
    *,
    team: str,
    context: str,
    theme: str | viz.Theme,
    size: str,
    footer: viz.Footer,
    excluded: Sequence[str] = (),
    logo: StrPath | None = None,
) -> viz.Canvas:
    """The team shot chart: statement title, stat strip, cropped shot map, zone panel.

    Parameters
    ----------
    shots : polars.DataFrame
        One team's kept shots (``COLUMNS``), ``xg`` the model of record's hold-out xG.
    league : polars.DataFrame
        Every kept row of that competition-season (``shots`` included).
    team, context : str
        Display name and ``"<competition> <season>"``: the subtitle.
    theme, size : str
        Keys of ``viz.THEMES`` (or a Theme) and ``viz.SIZES``.
    footer : viz.Footer
        Providers and model; the note is the chart's (``excluded`` parts, then the shots
        outside the drawn area).
    excluded : sequence of str
        Note parts for the rows not in ``shots`` (``render.excluded_note``).
    logo : str or path, optional
        The StatsBomb logo (``viz.DEFAULT_LOGO`` when omitted).

    Raises
    ------
    ValueError
        From :func:`check_inputs`; shots of more than one team; no layout fits.

    Examples
    --------
    >>> s = _example()
    >>> c = team_chart(s.filter(pl.col("team_id") == "sb:1"), s, team="Team", context="L 2015/2016",
    ...                theme="light", size="square", footer=viz.Footer(("whoscored",)))
    >>> viz.check(c)
    []
    """
    check_inputs(shots, league)
    if (k := shots["team_id"].n_unique()) > 1:
        raise ValueError(f"shots hold {k} teams; a team chart draws one")
    lines = table(shots, league, "zone")
    return _chart(
        shots, title=team_title(shots, league), subtitle=f"{team} · {context}",
        strip=stats(shots, league, per="xg"), heading=ZONE_HEADING, lines=lines,
        text=unjudged_caption(lines, mispriced_zones(shots, league)),
        theme=theme, size=size, footer=_with_level(footer, shots, league), excluded=excluded,
        logo=logo,
    )  # fmt: skip


def player_chart(
    shots: pl.DataFrame,
    league: pl.DataFrame,
    *,
    player: str,
    team: str,
    context: str,
    theme: str | viz.Theme,
    size: str,
    footer: viz.Footer,
    excluded: Sequence[str] = (),
    logo: StrPath | None = None,
) -> viz.Canvas:
    """The player shot profile: as :func:`team_chart`, with the body-part panel and the
    player's verdict title; subtitle ``"{player} · {team} · {context}"``.

    Raises
    ------
    ValueError
        From :func:`check_inputs`; shots of more than one (``player_id``, ``team_id``); no
        layout fits.

    Examples
    --------
    >>> s = _example()
    >>> c = player_chart(s.head(2), s, player="P", team="T", context="L 2015/2016",
    ...                  theme="dark", size="portrait", footer=viz.Footer(("whoscored",)))
    >>> viz.check(c)
    []
    """
    check_inputs(shots, league)
    if (k := shots.select("player_id", "team_id").unique().height) > 1:
        raise ValueError(f"shots hold {k} (player_id, team_id) pairs; a profile draws one")
    line = summary(shots, league)
    return _chart(
        shots, title=player_title(shots, league), subtitle=f"{player} · {team} · {context}",
        strip=stats(shots, league, per="goals"), heading=BODY_PART_HEADING,
        lines=table(shots, league, "body_part"),
        text=caption(line if verdict(line) == "zone_bias" else None), theme=theme,
        size=size, footer=_with_level(footer, shots, league), excluded=excluded, logo=logo,
    )  # fmt: skip


def _example() -> pl.DataFrame:
    """Three made-up shots for the doctests (no provider row)."""
    return pl.DataFrame(
        {
            "shot_id": ["sb:1", "sb:2", "sb:3"], "provider": ["whoscored"] * 3,
            "team_id": ["sb:1", "sb:1", "sb:2"], "player_id": ["sb:7", "sb:7", "sb:8"],
            "x": [101.0, 80.0, 75.0], "y": [34.0, 30.0, 40.0], "is_goal": [True, False, False],
            "is_penalty": [False] * 3, "period": [1, 2, 2],
            "body_part": ["right_foot", None, "head"], "xg": [0.45, 0.05, 0.4],
        },
        schema=dict(COLUMNS),
    )  # fmt: skip
