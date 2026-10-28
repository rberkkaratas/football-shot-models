"""The M1.0 leakage guard: xG features come only from C3 columns whose role is ``pre_shot``.

An xG model must only see what was known when the ball was struck. The C3 schema
(``football_core.schema``) gives every shot and freeze-frame column a role, so the allow-list
is derived from it rather than written by hand: a column added to C3 later is never a feature
unless C3 calls it ``pre_shot``. Outcome, post-shot (goal-mouth point, deflection), restricted
(big chance) and lineage columns (provider xG, raw payloads) are refused with their role in
the message. Match-level columns count too: the matches table's ``pre_shot`` columns (today
``gender``, fixed before kick-off) are allowed, while its scores contain the shot's own result
and are refused as ``outcome``.

Three additions sit on top of the roles:

- ``RESTRICTED_UNTIL`` refuses a ``pre_shot`` column that is known to be partly recorded on
  outcome, with the reason. ``situation`` is one: StatsBomb logs direct corners and wide direct
  free kicks as shots mostly when they threaten goal (C4 F5). ``technique`` is another: its
  ``lob`` level is read from the ball's flight, after the strike.
- ``RESTRICTED_READERS`` names the one registered derived feature that may read each restricted
  column, with the bad levels handled: ``phase`` (direct corners are not in the population, wide
  direct free kicks join ``free_kick``) and ``contact`` (lob merged into normal). Only the
  objects in :data:`DERIVED` qualify, so a custom feature cannot borrow the name.
- A :class:`DerivedFeature` (a spline term, a freeze-frame count) names the C3 columns it
  reads. Each source must pass the same check, and :func:`design_matrix` hands the feature's
  function only those columns (plus ``shot_id``, the join key), so it cannot read others. Its
  ``needs`` names features it is only defined together with: a keeper distance of 0.0 means
  "no keeper" only next to ``gk_absent``.

**Ablation only.** A ``restricted`` column in :data:`ABLATION_ONLY` (Opta's big chance) may
enter a labelled ablation and never a candidate model. A model spec carries a *guard*, the
function that checks its feature names (default :func:`check_xg_features`, which refuses it);
an ablation passes :func:`check_ablation_features` instead, and :func:`uses_ablation_only` is
the one rule that keeps such a spec out of the model choice.

The registry :data:`DERIVED` holds the M1.2 and M1.3 features, and ``assist`` (MS4). Every
one is computed per shot from that shot's own row and freeze frame, never from other shots:
the spline knots are fixed constants (rounded, label-free quantiles), so a fold fit cannot
leak through them, and the freeze-frame features are local to the shot (shooting cone,
nearest opponent, keeper), in canonical metres (``football_core.pitch``). The frame is
recorded at the strike, so it is pre-shot.

:func:`design_matrix` is the only way the models turn a frame into numbers, and it calls the
guard first. It never fills a missing value: a null, NaN or infinity is an error, because a
fill value would be a modelling decision taken silently.

**xGOT (MS5).** The leakage rule is target-specific. xG keeps :func:`check_xg_features`. An
xGOT model (P(goal | on-target shot)) may read the goal-mouth point, but only through the
placement features of :data:`PLACEMENT`, each of which declares the post-shot columns it reads
(``DerivedFeature.reads``, a subset of :data:`XGOT_DECLARED`) and refuses a shot that is not on
target. :func:`check_xgot_features` is the xGOT guard: it accepts those objects and the
cross-fitted xG input :data:`XG_INPUT`, refuses the raw columns and every keeper-position
feature (:data:`KEEPER_POSITION`: xGOT is keeper-neutral, so the keeper's positioning stays in
the goals-saved measure), and sends every other name through :func:`check_xg_features`.
Placement is read through ``pitch.to_goal_frame``, clipped to the frame plus the goal-mouth
tolerance (a point recorded beyond it is flagged by ``mouth_out``, :func:`clipped`) and rounded
to :data:`PLACEMENT_STEP`. Rounding puts a goal and a save that fall in one 0.1 m cell on the
same values, but it does not remove a provider's recording grid: StatsBomb goals sit on its
0.1 yd (0.0914 m) lattice while its projected saves are continuous, so a 0.1 m cell that holds
two lattice values gets about twice the goal mass of its neighbours (a beat of about 1.07 m;
the census is on the methodology page). Opta records goals and saves on one grid.

A ball whose centre is beyond :data:`INNER_Y` from the goal centre touches the post, so an
on-target shot out there is mostly a save: ``mouth_y_in`` (the distance up to ``INNER_Y``) is
the part of the width a model may hold monotone, and ``mouth_y_past`` / ``mouth_out`` are left
free.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import goalmouth, pitch, schema

FloatArray = npt.NDArray[np.float64]
Source = tuple[schema.Table, str]
Tables = Mapping[schema.Table, pl.DataFrame]
DeriveFn = Callable[[Tables], pl.Series]

RESTRICTED_UNTIL: Final[Mapping[str, str]] = MappingProxyType(
    {
        "situation": "F5 (direct_corner / direct_free_kick shots are logged on outcome); the "
        "derived feature 'phase' reads it with those levels handled",
        "technique": "its 'lob' level is split off (lob is labelled from the ball's flight, "
        "post-shot); the derived feature 'contact' reads it with lob merged into normal",
    }
)
RESTRICTED_READERS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {"situation": frozenset({"phase"}), "technique": frozenset({"contact"})}
)
XG_ALLOWED: Final[frozenset[str]] = frozenset(
    schema.column_names("shots", roles="pre_shot")
    + schema.column_names("matches", roles="pre_shot")
) - frozenset(RESTRICTED_UNTIL)
_TABLES: Final[tuple[schema.Table, ...]] = ("shots", "freeze_frames", "matches")
_ROLES: Final[Mapping[Source, str]] = MappingProxyType(
    {("shots", c.name): c.role for c in schema.SHOT_COLUMNS}
    | {("freeze_frames", c.name): c.role for c in schema.FREEZE_FRAME_COLUMNS}
    | {("matches", c.name): c.role for c in schema.MATCH_COLUMNS}
)
# The non-pre_shot columns an xGOT derived feature may declare in ``reads``, with the reason.
XGOT_DECLARED: Final[Mapping[str, str]] = MappingProxyType(
    {
        "goal_y": "placement, read through pitch.to_goal_frame",
        "goal_z": "placement, read through pitch.to_goal_frame",
        "on_target": "population check only",
    }
)
XG_INPUT: Final[str] = "xg_logit"  # the cross-fitted xG input of an xGOT model, by name
# Where the keeper stood: refused by the xGOT guard (PM E3; open_goal is the annotator's
# "keeper out of position"). gk_x / gk_y are named so that a later column cannot slip in.
KEEPER_POSITION: Final[frozenset[str]] = frozenset(
    {"gk_x", "gk_y", "gk_in_cone", "gk_absent", "gk_to_goal_m", "gk_to_shot_m",
     "gk_off_line_m", "open_goal"}
)  # fmt: skip


def _check_source(
    table: str, column: str, reader: str | None = None, reads: frozenset[str] = frozenset()
) -> None:
    """The guard for one C3 column: ``pre_shot`` (or declared in ``reads``), and not restricted
    unless ``reader`` may."""
    known = next((t for t in _TABLES if t == table), None)
    if known is None:
        raise ValueError(f"unknown table {table!r}; expected one of {_TABLES}")
    key: Source = (known, column)
    if key not in _ROLES:
        raise ValueError(f"{column!r} is not a column of the C3 {table} table")
    if table == "shots" and column in reads:
        return
    role = _ROLES[key]
    prefix = "" if table == "shots" else f"{table} "
    if role != "pre_shot":
        raise ValueError(
            f"{prefix}column {column!r} has role {role!r}; xG features must have role 'pre_shot'"
        )
    readers = RESTRICTED_READERS.get(column, frozenset())
    if table == "shots" and column in RESTRICTED_UNTIL and reader not in readers:
        raise ValueError(
            f"column {column!r} must not be an xG feature until {RESTRICTED_UNTIL[column]}"
        )


@dataclass(frozen=True)
class DerivedFeature:
    """A feature computed from C3 columns, checked like one.

    Attributes
    ----------
    name : str
        The feature name; not a column of the C3 shots or matches table.
    sources : tuple of (table, column)
        Every C3 column ``fn`` reads; each must be ``pre_shot`` (and not restricted) in
        ``"shots"``, ``"freeze_frames"`` or ``"matches"``.
    fn : callable
        ``fn(tables) -> polars.Series``, one value per shot in shots order. ``fn`` receives
        one frame per source table: ``t["shots"]`` and ``t["matches"]`` are per-shot
        (``shot_id`` plus the declared columns, in shots order); ``t["freeze_frames"]`` is per
        player row.
    needs : tuple of str, default ()
        Other feature names that must be in the same feature list (checked by
        :func:`check_xg_features`); never the feature's own name.
    reads : frozenset of str, default frozenset()
        Shots columns of a non-``pre_shot`` role that ``fn`` may read, a subset of
        :data:`XGOT_DECLARED`. A feature with ``reads`` is refused by :func:`check_xg_features`;
        :func:`check_xgot_features` accepts only the :data:`XGOT_DERIVED` object of its name.

    Raises
    ------
    ValueError
        A bad name, no sources, a source that fails the guard (the message names its role), a
        bad ``needs``, or ``reads`` outside :data:`XGOT_DECLARED`.

    Examples
    --------
    >>> DerivedFeature("end_dist", (("shots", "end_x"),), lambda t: t["shots"]["end_x"])
    Traceback (most recent call last):
    ...
    ValueError: column 'end_x' has role 'outcome'; xG features must have role 'pre_shot'
    """

    name: str
    sources: tuple[Source, ...]
    fn: DeriveFn
    needs: tuple[str, ...] = ()
    reads: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        taken = ("shots", self.name) in _ROLES or ("matches", self.name) in _ROLES
        if not isinstance(self.name, str) or not self.name or taken:
            raise ValueError(
                f"derived feature name {self.name!r} is empty or a C3 shot column (or a matches "
                "column)"
            )
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError(f"derived feature {self.name!r} needs a tuple of (table, column)")
        if not isinstance(self.reads, frozenset) or not self.reads <= set(XGOT_DECLARED):
            raise ValueError(
                f"derived feature {self.name!r}: reads must be a frozenset within "
                f"{sorted(XGOT_DECLARED)}, got {self.reads!r}"
            )
        for table, column in self.sources:
            _check_source(table, column, reader=self.name, reads=self.reads)
        named = isinstance(self.needs, tuple) and all(isinstance(n, str) for n in self.needs)
        if not named or self.name in self.needs:
            raise ValueError(
                f"derived feature {self.name!r}: needs must be a tuple of other feature names, "
                f"got {self.needs!r}"
            )


# ---- M1.2 splines ---------------------------------------------------------------------------

DISTANCE_KNOTS: Final[tuple[float, ...]] = (6.0, 11.5, 17.0, 22.5, 30.0)  # metres
ANGLE_KNOTS: Final[tuple[float, ...]] = (0.2, 0.27, 0.35, 0.5, 1.0)  # radians


def rcs_term(x: pl.Expr, knots: Sequence[float], j: int) -> pl.Expr:
    r"""The j-th nonlinear term of a restricted cubic spline (Harrell), as an expression.

    With knots $t_1 < \dots < t_k$, $a = (t_k - t_j) / (t_k - t_{k-1})$ and
    $b = (t_{k-1} - t_j) / (t_k - t_{k-1})$, the term is

    $$
    \frac{(x - t_j)_+^3 - a\,(x - t_{k-1})_+^3 + b\,(x - t_k)_+^3}{(t_k - t_1)^2}.
    $$

    It is 0 below $t_j$ and linear beyond the last knot, so together with ``x`` itself the
    ``k - 2`` terms give a curve that is cubic between the knots and straight in the tails,
    where shots are few. Dividing by $(t_k - t_1)^2$ keeps the terms in the units of ``x``.

    Parameters
    ----------
    x : polars.Expr
        The input (metres or radians).
    knots : sequence of float
        At least three finite, strictly increasing knots.
    j : int
        Term index, ``0 <= j <= len(knots) - 3``.

    Returns
    -------
    polars.Expr

    Raises
    ------
    ValueError
        Fewer than three knots, a NaN or infinite knot, knots not strictly increasing, or ``j``
        out of range.

    Examples
    --------
    >>> d = pl.DataFrame({"d": [0.0, 6.0, 30.0, 40.0]})
    >>> d.select(rcs_term(pl.col("d"), DISTANCE_KNOTS, 0)).to_series().to_list()
    [0.0, 0.0, 21.65625, 42.28125]
    """
    t, k = tuple(knots), len(knots)
    increasing = all(math.isfinite(v) for v in t) and all(a < b for a, b in pairwise(t))
    if k < 3 or not increasing or not 0 <= j <= k - 3:
        raise ValueError(f"need >= 3 finite increasing knots and 0 <= j <= {k - 3}, got {t}, j={j}")

    def p3(v: float) -> pl.Expr:
        return (x - v).clip(lower_bound=0.0) ** 3

    a = (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
    b = (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
    # times the reciprocal, not divided: polars divides a 1-row column by a literal exactly
    # but a longer one via the reciprocal, so a shot's value would depend on the frame's length
    scale = 1.0 / (t[k - 1] - t[0]) ** 2
    return (p3(t[j]) - a * p3(t[k - 2]) + b * p3(t[k - 1])) * scale


def _spline(column: str, knots: tuple[float, ...], j: int) -> DerivedFeature:
    return DerivedFeature(
        f"{column}_rcs{j + 1}",
        (("shots", column),),
        lambda t: t["shots"].select(rcs_term(pl.col(column), knots, j)).to_series(),
    )


# ---- M1.2 context: phase and contact ---------------------------------------------------------

PHASES: Final = ("open_play", "fast_break", "corner", "free_kick", "throw_in", "direct_free_kick")
PHASE: Final = pl.Enum(PHASES)
BOX_RIGHT_Y, BOX_LEFT_Y = pitch.METRE_Y[1], pitch.METRE_Y[-2]  # 13.84, 54.16
# "wide of the box": the one predicate phase and the card's wide-DFK line both read
WIDE: Final[pl.Expr] = (pl.col("y") < BOX_RIGHT_Y) | (pl.col("y") > BOX_LEFT_Y)
CONTACTS: Final = tuple(c for c in schema.TECHNIQUES if c != "lob")
CONTACT: Final = pl.Enum(CONTACTS)


def _phase(t: Tables) -> pl.Series:
    sit = pl.col("situation")
    expr = (
        pl.when(sit.is_in(["open_play", "kick_off"]))
        .then(pl.lit("open_play"))
        .when(sit == "fast_break")
        .then(pl.lit("fast_break"))
        .when(sit == "from_corner")
        .then(pl.lit("corner"))
        .when((sit == "from_free_kick") | ((sit == "direct_free_kick") & WIDE))
        .then(pl.lit("free_kick"))
        .when(sit == "from_throw_in")
        .then(pl.lit("throw_in"))
        .when(sit == "direct_free_kick")
        .then(pl.lit("direct_free_kick"))
        .otherwise(None)  # penalty, direct_corner: excluded, so design_matrix raises on them
        .cast(PHASE)
    )
    return t["shots"].select(expr).to_series()


# The WhoScored archive has no pull-back qualifier, so a cutback level would be a constant-0
# indicator there.
ASSISTS: Final = tuple(a for a in schema.ASSIST_TYPES if a != "cutback")
ASSIST: Final = pl.Enum(ASSISTS)


def _assist(t: Tables) -> pl.Series:
    """``assist``, the WhoScored / Opta encoding of ``assist_type``: its levels without
    ``cutback``, which Opta cannot record. A StatsBomb spec reads ``assist_type`` itself."""
    col = t["shots"]["assist_type"].cast(pl.String)
    if bad := int(col.is_in(["cutback"]).sum()):
        raise ValueError(
            f"derived feature 'assist' is the WhoScored / Opta encoding of assist_type: {bad} "
            "cutback value(s), which the WhoScored archive cannot record (no pull-back "
            "qualifier, C5 census); a StatsBomb spec reads assist_type itself"
        )
    return col.cast(ASSIST)


def _contact(t: Tables) -> pl.Series:
    tech = pl.col("technique").cast(pl.String)
    lob_as_normal = pl.when(tech == "lob").then(pl.lit("normal")).otherwise(tech)
    return t["shots"].select(lob_as_normal.cast(CONTACT)).to_series()


# ---- M1.3 freeze-frame and keeper features ---------------------------------------------------

NEAREST_CAP_M: Final[float] = 10.0
GK_CAP_M: Final[float] = 20.0
_FRAME_SOURCES: Final[tuple[tuple[schema.Table, str], ...]] = (
    ("shots", "x"),
    ("shots", "y"),
    ("freeze_frames", "is_teammate"),
    ("freeze_frames", "is_keeper"),
    ("freeze_frames", "x"),
    ("freeze_frames", "y"),
)
_GX, _RY, _LY = pitch.PITCH_LENGTH, pitch.RIGHT_POST_Y, pitch.LEFT_POST_Y
_PX, _PY, _SX, _SY = pl.col("x"), pl.col("y"), pl.col("sx"), pl.col("sy")


def _hypot(dx: pl.Expr, dy: pl.Expr) -> pl.Expr:
    return (dx**2 + dy**2).sqrt()


def _cross(ax: pl.Expr, ay: pl.Expr, bx: pl.Expr, by: pl.Expr) -> pl.Expr:
    return ax * by - ay * bx


def _frame_rows(t: Tables) -> pl.DataFrame:
    """Frame rows of the given shots, with the shooting-cone test and the distance to the shot."""
    gx, ry, ly = pl.lit(_GX), pl.lit(_RY), pl.lit(_LY)
    d1 = _cross(gx - _SX, ry - _SY, _PX - _SX, _PY - _SY)  # shot -> right post
    d2 = _cross(pl.lit(0.0), ly - ry, _PX - gx, _PY - ry)  # right post -> left post
    d3 = _cross(_SX - gx, _SY - ly, _PX - gx, _PY - ly)  # left post -> shot
    same_side = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    shots = t["shots"].select("shot_id", sx="x", sy="y")
    return (
        t["freeze_frames"]
        .join(shots, on="shot_id", how="inner")
        .with_columns(
            cone=same_side & (gx > _SX),  # a shot on the goal line has no cone
            dist=_hypot(_PX - _SX, _PY - _SY),
            opponent=~pl.col("is_teammate") & ~pl.col("is_keeper"),
            keeper=~pl.col("is_teammate") & pl.col("is_keeper"),
        )
    )


def _frame(name: str, expr: pl.Expr, needs: tuple[str, ...] = ()) -> DerivedFeature:
    def fn(t: Tables) -> pl.Series:
        agg = (
            _frame_rows(t)
            .group_by("shot_id")
            .agg(rows=pl.len(), keepers=pl.col("keeper").sum(), value=expr)
        )
        out = t["shots"].select("shot_id")
        out = out.join(agg, on="shot_id", how="left", maintain_order="left")
        if missing := out["rows"].null_count():
            raise ValueError(f"derived feature {name!r}: {missing} shot(s) have no frame rows")
        if extra := int((out["keepers"] > 1).sum()):
            raise ValueError(f"derived feature {name!r}: {extra} frame(s) hold 2+ opponent keepers")
        return out["value"].cast(pl.Float64)

    return DerivedFeature(name, _FRAME_SOURCES, fn, needs=needs)


def _keeper(name: str, expr: pl.Expr) -> DerivedFeature:
    """The opponent keeper's ``expr``, capped at GK_CAP_M; 0.0 without one (gk_absent)."""
    value = expr.filter(pl.col("keeper")).first().fill_null(0.0).clip(upper_bound=GK_CAP_M)
    return _frame(name, value, needs=("gk_absent",))


# Angle-bisector theorem: the bisector of the shot's angle meets the goal line at y_b, which
# splits the goal mouth in the ratio of the shot's distances to the two posts.
_D_R, _D_L = _hypot(_GX - _SX, _RY - _SY), _hypot(_GX - _SX, _LY - _SY)
_BX, _BY = pl.lit(_GX), _RY + pitch.GOAL_WIDTH * _D_R / (_D_R + _D_L)
FRAME_FEATURES: Final[tuple[DerivedFeature, ...]] = (
    _frame("opponents_in_cone", (pl.col("cone") & pl.col("opponent")).sum()),
    _frame("teammates_in_cone", (pl.col("cone") & pl.col("is_teammate")).sum()),
    _frame("gk_in_cone", (pl.col("cone") & pl.col("keeper")).any()),
    _frame(
        "nearest_opponent_m",
        pl.col("dist")
        .filter(pl.col("opponent"))
        .min()
        .fill_null(NEAREST_CAP_M)
        .clip(upper_bound=NEAREST_CAP_M),
    ),  # fmt: skip
    _frame("gk_absent", ~pl.col("keeper").any()),
    _keeper("gk_to_goal_m", _hypot(_PX - _GX, _PY - pitch.GOAL_CENTRE_Y)),
    _keeper("gk_to_shot_m", pl.col("dist")),
    _keeper(
        "gk_off_line_m",
        _cross(_BX - _SX, _BY - _SY, _PX - _SX, _PY - _SY).abs() / _hypot(_BX - _SX, _BY - _SY),
    ),
)

DERIVED: Final[Mapping[str, DerivedFeature]] = MappingProxyType(
    {
        f.name: f
        for f in (
            *(_spline("distance_m", DISTANCE_KNOTS, j) for j in range(len(DISTANCE_KNOTS) - 2)),
            *(_spline("angle_rad", ANGLE_KNOTS, j) for j in range(len(ANGLE_KNOTS) - 2)),
            DerivedFeature("phase", (("shots", "situation"), ("shots", "y")), _phase),
            DerivedFeature("contact", (("shots", "technique"),), _contact),
            DerivedFeature("assist", (("shots", "assist_type"),), _assist),
            *FRAME_FEATURES,
        )
    }
)

# ---- MS5 xGOT placement: where the ball crossed the goal line ------------------------------

PLACEMENT_STEP: Final[float] = 0.1  # metres; both providers, before any feature reads the frame
_PER_METRE: Final[float] = round(1 / PLACEMENT_STEP)  # k / 10, not k * 0.1: 3.7, not 3.70...02
EDGE_Y: Final[float] = pitch.GOAL_WIDTH / 2 + goalmouth.GOAL_TOLERANCE  # 3.81
EDGE_Z: Final[float] = pitch.GOAL_HEIGHT + goalmouth.GOAL_TOLERANCE  # 2.59
CENTRE_DEAD_ZONE: Final[float] = 0.5  # metres: a shot this close to the centre line has no side
BALL_RADIUS: Final[float] = 0.11  # metres, a size-5 ball
INNER_Y: Final[float] = round(pitch.GOAL_WIDTH / 2 - BALL_RADIUS, 2)  # 3.55: beyond, it hits a post
# rounded 5 / 27.5 / 50 / 72.5 / 95 % quantiles of on-target |y| and z, both providers
MOUTH_Y_KNOTS: Final[tuple[float, ...]] = (0.15, 0.9, 1.7, 2.5, 3.25)
MOUTH_Z_KNOTS: Final[tuple[float, ...]] = (0.1, 0.3, 0.8, 1.25, 2.0)
_MOUTH: Final[frozenset[str]] = frozenset({"goal_y", "goal_z", "on_target"})


def _step(v: FloatArray, low: float, high: float) -> FloatArray:
    """Clip to ``[low, high]``, then round to the ``PLACEMENT_STEP`` grid (half to even)."""
    out: FloatArray = np.round(np.clip(v, low, high) / PLACEMENT_STEP) / _PER_METRE
    return out


def clipped(gy: npt.ArrayLike, gz: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether a goal-frame point lies beyond the frame plus the tolerance (and so is clipped).

    The one rule behind ``mouth_out`` and the cards' clipped count.

    Examples
    --------
    >>> clipped([0.0, 3.81, 3.82, 0.0], [1.0, 2.59, 1.0, 2.6]).tolist()
    [False, False, True, True]
    """
    y, z = np.asarray(gy, dtype=np.float64), np.asarray(gz, dtype=np.float64)
    out: npt.NDArray[np.bool_] = (np.abs(y) > EDGE_Y) | (z < 0.0) | (z > EDGE_Z)
    return out


def dead_zone(y: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Whether a shot from pitch-width ``y`` is within :data:`CENTRE_DEAD_ZONE` of the centre line.

    Such a shot has no near or far post (``far_side`` is 0).

    Examples
    --------
    >>> dead_zone([34.0, 34.49, 33.5, 20.0]).tolist()
    [True, True, False, False]
    """
    out: npt.NDArray[np.bool_] = (
        np.abs(pitch.GOAL_CENTRE_Y - np.asarray(y, dtype=np.float64)) < CENTRE_DEAD_ZONE
    )
    return out


def placement(shots: pl.DataFrame) -> pl.DataFrame:
    """The placement of on-target shots in the goal frame, one row per shot.

    ``goal_y`` / ``goal_z`` go through ``pitch.to_goal_frame`` (``y`` positive towards the
    shooter's right post), are clipped to the frame plus ``goalmouth.GOAL_TOLERANCE``
    (:data:`EDGE_Y`, :data:`EDGE_Z`) and rounded to :data:`PLACEMENT_STEP` with ``np.round``
    (half to even): a goal and a projected save in the same 0.1 m cell get the same values. A
    non-finite value is an error, never clipped.

    Parameters
    ----------
    shots : polars.DataFrame
        ``goal_y``, ``goal_z``, ``on_target``; ``y`` adds the side columns and ``angle_rad``
        the interactions.

    Returns
    -------
    polars.DataFrame
        Float64 ``mouth_y`` (signed), ``mouth_y_abs`` and ``mouth_z`` with their spline terms
        ``*_rcs1..3`` (:data:`MOUTH_Y_KNOTS`, :data:`MOUTH_Z_KNOTS`) and ``mouth_yz``;
        ``mouth_y_in = min(mouth_y_abs, INNER_Y)``, ``mouth_y_past = mouth_y_abs −
        mouth_y_in`` and ``mouth_out`` (1.0 when the point was recorded beyond the frame plus
        the tolerance, :func:`clipped`, else 0.0); with
        ``y``: ``far_side`` (+1 far post, -1 near post, 0 when the shot is within
        :data:`CENTRE_DEAD_ZONE` of the centre line or the ball is rounded to the centre) and
        ``mouth_far = far_side * mouth_y_abs``; with ``angle_rad``: ``angle_x_mouth_y`` and
        ``angle_x_mouth_z``.

    Raises
    ------
    ValueError
        A missing ``goal_y`` / ``goal_z`` / ``on_target``; a shot that is not on target (or a
        null ``on_target``); a null, NaN or infinite ``goal_y``, ``goal_z``, ``y`` or
        ``angle_rad`` (counts in the message).

    Examples
    --------
    >>> shots = pl.DataFrame({"goal_y": [30.34, 33.95, 33.75], "goal_z": [0.0, 2.7, 1.0],
    ...                       "on_target": [True] * 3, "y": [20.0, 50.0, 34.2]})
    >>> p = placement(shots)
    >>> p["mouth_y"].to_list(), p["mouth_z"].to_list(), p["far_side"].to_list()
    ([3.7, 0.0, 0.2], [0.0, 2.6, 1.0], [-1.0, 0.0, 0.0])
    >>> p["mouth_y_in"].to_list(), p["mouth_out"].to_list()
    ([3.55, 0.0, 0.2], [0.0, 1.0, 0.0])
    """
    if missing := sorted(_MOUTH - set(shots.columns)):
        raise ValueError(f"placement: the shots lack {missing}")
    if off := shots.height - int(shots["on_target"].fill_null(False).sum()):
        raise ValueError(
            f"placement: {off} shot(s) are not on target (or have a null on_target); xGOT "
            "placement is defined on on-target shots only"
        )
    read = [c for c in ("goal_y", "goal_z", "y", "angle_rad") if c in shots.columns]
    for c in read:
        if bad := shots.height - int(shots[c].cast(pl.Float64).is_finite().fill_null(False).sum()):
            raise ValueError(f"placement: {c} has {bad} null, NaN or infinite value(s)")
    v = {c: shots[c].cast(pl.Float64).to_numpy() for c in read}
    gy, gz = pitch.to_goal_frame(v["goal_y"], v["goal_z"])
    out_ = clipped(gy, gz).astype(np.float64)
    gy, gz = _step(gy, -EDGE_Y, EDGE_Y), _step(gz, 0.0, EDGE_Z)
    inner = np.minimum(np.abs(gy), INNER_Y)
    cols = {
        "mouth_y": gy,
        "mouth_y_abs": np.abs(gy),
        "mouth_z": gz,
        "mouth_yz": np.abs(gy) * gz,
        "mouth_y_in": inner,
        "mouth_y_past": np.abs(gy) - inner,
        "mouth_out": out_,
    }
    if "y" in v:
        off_centre = pitch.GOAL_CENTRE_Y - v["y"]
        side = np.where(dead_zone(v["y"]), 0.0, np.sign(off_centre))
        cols["far_side"] = -side * np.sign(gy) + 0.0  # + 0.0: no -0.0
        cols["mouth_far"] = cols["far_side"] * np.abs(gy) + 0.0
    if "angle_rad" in v:
        cols["angle_x_mouth_y"] = v["angle_rad"] * np.abs(gy)
        cols["angle_x_mouth_z"] = v["angle_rad"] * gz
    out = pl.DataFrame(cols)
    terms = [
        rcs_term(pl.col(col), knots, j).alias(f"{col}_rcs{j + 1}")
        for col, knots in (("mouth_y_abs", MOUTH_Y_KNOTS), ("mouth_z", MOUTH_Z_KNOTS))
        for j in range(len(knots) - 2)
    ]
    return out.with_columns(terms)


def _placed(name: str, extra: tuple[str, ...] = ()) -> DerivedFeature:
    sources: tuple[Source, ...] = tuple(("shots", c) for c in ("goal_y", "goal_z", "on_target"))
    return DerivedFeature(
        name,
        sources + tuple(("shots", c) for c in extra),
        lambda t: placement(t["shots"])[name],
        reads=_MOUTH,
    )


PLACEMENT: Final[Mapping[str, DerivedFeature]] = MappingProxyType(
    {
        f.name: f
        for f in (
            _placed("mouth_y_abs"),
            *(_placed(f"mouth_y_abs_rcs{j + 1}") for j in range(len(MOUTH_Y_KNOTS) - 2)),
            _placed("mouth_z"),
            *(_placed(f"mouth_z_rcs{j + 1}") for j in range(len(MOUTH_Z_KNOTS) - 2)),
            _placed("far_side", ("y",)),
            _placed("mouth_far", ("y",)),
            _placed("angle_x_mouth_y", ("angle_rad",)),
            _placed("angle_x_mouth_z", ("angle_rad",)),
            _placed("mouth_yz"),
            _placed("mouth_y_in"),
            _placed("mouth_y_past"),
            _placed("mouth_out"),
        )
    }
)
XGOT_DERIVED: Final[Mapping[str, DerivedFeature]] = MappingProxyType({**DERIVED, **PLACEMENT})


def check_xg_features(
    features: Sequence[str], derived: Mapping[str, DerivedFeature] = DERIVED
) -> tuple[str, ...]:
    """Check that every name is an allowed C3 shots or matches column or a derived feature.

    Parameters
    ----------
    features : sequence of str
        Candidate feature names.
    derived : mapping of str to DerivedFeature, default DERIVED
        Registered derived features, by name (their sources were checked when made).

    Returns
    -------
    tuple of str
        The names, in the given order.

    Raises
    ------
    TypeError
        If ``features`` is a single ``str``.
    ValueError
        A name that is neither a C3 shots or matches column nor registered (shared names are
        looked up in shots first; they have the same role in both), a column whose role is not
        ``pre_shot`` (the message names the role), a column in ``RESTRICTED_UNTIL`` (the
        message gives the reason), a registry key that is not its feature's name, a derived
        feature reading a restricted column that is not the :data:`DERIVED` object of that
        name, a derived feature that reads post-shot columns (an xGOT placement feature), a
        derived feature without the features in its ``needs``, or a duplicate name.

    Examples
    --------
    >>> check_xg_features(["distance_m", "body_part", "gender", "phase"])
    ('distance_m', 'body_part', 'gender', 'phase')
    >>> check_xg_features(["goal_y"])
    Traceback (most recent call last):
    ...
    ValueError: column 'goal_y' has role 'post_shot'; xG features must have role 'pre_shot'
    >>> check_xg_features(["gk_to_goal_m"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: derived feature 'gk_to_goal_m' is defined only together with ['gk_absent']; ...
    """
    if isinstance(features, str):
        raise TypeError("features must be a sequence of column names, not a single str")
    names = tuple(features)
    for name in names:
        if name in derived:
            feature = derived[name]
            if feature.name != name:
                raise ValueError(f"derived feature registered as {name!r} is {feature.name!r}")
            if feature.reads:
                raise ValueError(
                    f"derived feature {name!r} reads post-shot column(s) "
                    f"{sorted(feature.reads)}; only the xGOT guard accepts it"
                )
            cols = [c for t, c in feature.sources if t == "shots" and c in RESTRICTED_UNTIL]
            if cols and feature is not DERIVED.get(name):
                raise ValueError(
                    f"derived feature {name!r} reads restricted column(s) {cols}; only "
                    f"features.DERIVED[{name!r}] may read them"
                )
            continue
        table = "shots" if ("shots", name) in _ROLES else "matches"
        if (table, name) not in _ROLES:
            raise ValueError(
                f"{name!r} is not a column of the C3 shots table, the matches table or a "
                "registered derived feature"
            )
        _check_source(table, name)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    for name in names:
        if name in derived and (missing := [n for n in derived[name].needs if n not in names]):
            raise ValueError(
                f"derived feature {name!r} is defined only together with {missing}; add them "
                "to the features"
            )
    return names


Guard = Callable[[Sequence[str], Mapping[str, DerivedFeature]], tuple[str, ...]]
ABLATION_ONLY: Final[Mapping[str, str]] = MappingProxyType(
    {
        "big_chance": "Opta q214 is an analyst's label of the chance; it enters a labelled "
        "ablation only, never a headline model"
    }
)


def uses_ablation_only(names: Sequence[str]) -> bool:
    """True when any name is an :data:`ABLATION_ONLY` column: the spec is an ablation.

    The one rule behind "an ablation never qualifies" (``evaluation.best``) and the v1 check
    "big chance only in the labelled ablation".

    Raises
    ------
    TypeError
        If ``names`` is a single ``str``.

    Examples
    --------
    >>> uses_ablation_only(("distance_m", "big_chance")), uses_ablation_only(("distance_m",))
    (True, False)
    """
    if isinstance(names, str):
        raise TypeError("features must be a sequence of column names, not a single str")
    return any(n in ABLATION_ONLY for n in names)


def check_ablation_features(
    names: Sequence[str], derived: Mapping[str, DerivedFeature] = DERIVED
) -> tuple[str, ...]:
    """The guard of a labelled ablation: :func:`check_xg_features` plus ``ABLATION_ONLY``.

    Parameters
    ----------
    names : sequence of str
        Feature names, at least one of them in :data:`ABLATION_ONLY`.
    derived : mapping of str to DerivedFeature, default DERIVED
        Registered derived features.

    Returns
    -------
    tuple of str
        The names, in the given order.

    Raises
    ------
    TypeError
        If ``names`` is a single ``str``.
    ValueError
        No ``ABLATION_ONLY`` name, a duplicate, or any refusal of :func:`check_xg_features` on
        the other names (so post-shot and outcome columns stay refused).

    Examples
    --------
    >>> check_ablation_features(["distance_m", "big_chance"])
    ('distance_m', 'big_chance')
    >>> check_ablation_features(["goal_y", "big_chance"])
    Traceback (most recent call last):
    ...
    ValueError: column 'goal_y' has role 'post_shot'; xG features must have role 'pre_shot'
    """
    if isinstance(names, str):
        raise TypeError("features must be a sequence of column names, not a single str")
    out = tuple(names)
    if not uses_ablation_only(out):
        raise ValueError(f"an ablation guard needs one of {sorted(ABLATION_ONLY)}; got {out}")
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate feature names in {out}")
    check_xg_features([n for n in out if n not in ABLATION_ONLY], derived)
    return out


def check_xgot_features(
    features: Sequence[str], derived: Mapping[str, DerivedFeature] = XGOT_DERIVED
) -> tuple[str, ...]:
    """The xGOT guard: placement, shot context and the cross-fitted xG, never the keeper.

    ``xg_logit`` (:data:`XG_INPUT`) passes; a derived feature with ``reads`` passes only as the
    :data:`XGOT_DERIVED` object of its name; every other name goes through
    :func:`check_xg_features`, so its outcome, identity, lineage, restricted and ``needs``
    rules hold unchanged.

    Parameters
    ----------
    features : sequence of str
        Candidate xGOT feature names.
    derived : mapping of str to DerivedFeature, default XGOT_DERIVED
        Registered derived features.

    Returns
    -------
    tuple of str
        The names, in the given order.

    Raises
    ------
    TypeError
        If ``features`` is a single ``str``.
    ValueError
        A duplicate; a :data:`KEEPER_POSITION` name; ``goal_y``, ``goal_z`` or ``on_target``
        named directly; a derived feature with ``reads`` that is not the :data:`XGOT_DERIVED`
        object of that name; any refusal of :func:`check_xg_features` on the other names.

    Examples
    --------
    >>> check_xgot_features(["xg_logit", "mouth_y_abs", "far_side", "angle_rad"])
    ('xg_logit', 'mouth_y_abs', 'far_side', 'angle_rad')
    >>> check_xgot_features(["goal_y"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: column 'goal_y' is not an xGOT feature itself (placement, ...); read it ...
    >>> check_xgot_features(["xg_logit", "gk_absent"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: 'gk_absent' is a keeper-position feature; xGOT is keeper-neutral ...
    """
    if isinstance(features, str):
        raise TypeError("features must be a sequence of column names, not a single str")
    names = tuple(features)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    rest = []
    for name in names:
        if name in KEEPER_POSITION:
            raise ValueError(
                f"{name!r} is a keeper-position feature; xGOT is keeper-neutral (PM E3: "
                "placement, shot context and out-of-fold xG only, so goals saved keep the "
                "keeper's positioning)"
            )
        if name in XGOT_DECLARED:
            raise ValueError(
                f"column {name!r} is not an xGOT feature itself ({XGOT_DECLARED[name]}); read "
                f"it through the placement features {sorted(PLACEMENT)}"
            )
        feature = derived.get(name)
        if feature is not None and feature.reads and feature is not XGOT_DERIVED.get(name):
            raise ValueError(
                f"derived feature {name!r} reads post-shot column(s) {sorted(feature.reads)}; "
                f"only features.XGOT_DERIVED[{name!r}] may read them"
            )
        if name != XG_INPUT and not (feature is not None and feature.reads):
            rest.append(name)
    check_xg_features(rest, derived)
    return names


def check_keeper_neutral(features: Sequence[str]) -> tuple[str, ...]:
    """Refuse any :data:`KEEPER_POSITION` feature: the check run on the xGOT's xG input.

    Raises
    ------
    TypeError
        If ``features`` is a single ``str``.
    ValueError
        Naming the keeper-position features found.

    Examples
    --------
    >>> check_keeper_neutral(["distance_m", "angle_rad"])
    ('distance_m', 'angle_rad')
    >>> check_keeper_neutral(["distance_m", "open_goal", "gk_absent"])  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: the xG input reads keeper-position feature(s) ['open_goal', 'gk_absent']; ...
    """
    if isinstance(features, str):
        raise TypeError("features must be a sequence of column names, not a single str")
    names = tuple(features)
    if found := [n for n in names if n in KEEPER_POSITION]:
        raise ValueError(
            f"the xG input reads keeper-position feature(s) {found}; the xGOT input must be "
            "keeper-neutral (PM E3, X1)"
        )
    return names


def _derive(
    feature: DerivedFeature, shots: pl.DataFrame, freeze_frames: pl.DataFrame | None
) -> pl.Series:
    """Run ``feature.fn`` on its declared columns only."""
    # Match columns are read per shot, from the columns load_xg_data joined onto the shots,
    # never from the one-row-per-match table.
    frames = {"shots": shots, "freeze_frames": freeze_frames, "matches": shots}
    inputs: dict[schema.Table, pl.DataFrame] = {}
    for table in dict.fromkeys(t for t, _ in feature.sources):
        frame = frames[table]
        if frame is None:
            raise ValueError(f"derived feature {feature.name!r} reads {table}; pass {table}=")
        columns = dict.fromkeys(["shot_id", *(c for t, c in feature.sources if t == table)])
        if (missing := [c for c in columns if c not in frame.columns]) and table == "matches":
            raise ValueError(
                f"derived feature {feature.name!r}: shots lacks match column(s) {missing}; "
                "load with data.load_xg_data"
            )
        if missing:
            raise ValueError(f"derived feature {feature.name!r}: {table} lacks {missing}")
        inputs[table] = frame.select(list(columns))
    out = feature.fn(MappingProxyType(inputs))
    if not isinstance(out, pl.Series) or out.len() != shots.height:
        raise ValueError(
            f"derived feature {feature.name!r} must return a Series of {shots.height} value(s)"
        )
    return out


def _columns(shots: pl.DataFrame, col: str) -> list[tuple[str, pl.Expr]]:
    """The design columns of one feature, after the null / NaN / dtype checks."""
    if col not in shots.columns:
        raise ValueError(f"feature {col!r} is not a column of the frame")
    series = shots[col]
    dtype = series.dtype
    if series.null_count():
        raise ValueError(f"feature {col!r} has {series.null_count()} null value(s)")
    if dtype.is_float():
        bad = int((~series.is_finite()).sum())
        if bad:
            raise ValueError(f"feature {col!r} has {bad} NaN or infinite value(s)")
    if dtype.is_float() or dtype.is_integer() or dtype == pl.Boolean():
        return [(col, pl.col(col).cast(pl.Float64()))]
    if isinstance(dtype, pl.Enum):
        # One indicator per non-reference category, built from the dtype rather than the
        # data, so a category with no rows still gets its (all-zero) column.
        return [
            (f"{col}={category}", (pl.col(col) == category).cast(pl.Float64()))
            for category in dtype.categories[1:].to_list()
        ]
    raise ValueError(f"feature {col!r} has unsupported dtype {dtype}")


def design_matrix(
    shots: pl.DataFrame,
    features: Sequence[str],
    *,
    derived: Mapping[str, DerivedFeature] = DERIVED,
    freeze_frames: pl.DataFrame | None = None,
    guard: Guard | None = None,
) -> tuple[FloatArray, tuple[str, ...]]:
    """The float64 feature matrix of the given xG features, and its column names.

    Numbers become float64, booleans 0.0 / 1.0, and an Enum one indicator per category after
    the first (the reference), named ``f"{col}={category}"``. A derived feature is computed
    from its declared columns, then treated the same way.

    Parameters
    ----------
    shots : polars.DataFrame
        Shots holding the feature columns.
    features : sequence of str
        Feature names, checked by :func:`check_xg_features`.
    derived : mapping of str to DerivedFeature, default DERIVED
        Registered derived features.
    freeze_frames : polars.DataFrame, optional
        C3 freeze frames, for derived features that read them.
    guard : callable, optional
        ``guard(features, derived)`` checks the names first; ``None`` is
        :func:`check_xg_features`, looked up when called (a labelled ablation passes
        :func:`check_ablation_features`).

    Returns
    -------
    matrix : numpy.ndarray
        Shape ``(shots.height, number of design columns)``.
    names : tuple of str
        The design column names, in order.

    Raises
    ------
    TypeError, ValueError
        As the guard; also a feature missing from the frame, a null, NaN or
        infinite value, an unsupported dtype (List, String), or a derived feature without its
        table or returning the wrong length.

    Examples
    --------
    >>> shots = pl.DataFrame({"shot_id": ["sb:a", "sb:b"], "distance_m": [11.0, 20.0],
    ...                       "body_part": pl.Series(["head", "right_foot"],
    ...                                              dtype=schema.BODY_PART)})
    >>> matrix, names = design_matrix(shots, ["distance_m", "body_part"])
    >>> names
    ('distance_m', 'body_part=left_foot', 'body_part=head', 'body_part=other')
    >>> matrix.tolist()
    [[11.0, 0.0, 1.0, 0.0], [20.0, 0.0, 0.0, 0.0]]
    >>> near = DerivedFeature("near", (("shots", "distance_m"),),
    ...                       lambda t: t["shots"]["distance_m"] < 12)
    >>> design_matrix(shots, ["near"], derived={"near": near})[0].tolist()
    [[1.0], [0.0]]
    """
    names = (guard or check_xg_features)(features, derived)
    frame = shots
    for name in names:
        if name in derived:
            frame = frame.with_columns(_derive(derived[name], shots, freeze_frames).alias(name))
    columns = [c for col in names for c in _columns(frame, col)]
    if not columns:
        return np.empty((shots.height, 0), dtype=np.float64), ()
    design = tuple(name for name, _ in columns)
    out = frame.select(expr.alias(name) for name, expr in columns)
    matrix: FloatArray = np.asarray(out.to_numpy(), dtype=np.float64)
    return matrix, design
