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

The registry :data:`DERIVED` holds the M1.2 and M1.3 features. Every one is computed per shot
from that shot's own row and freeze frame, never from other shots: the spline knots are fixed
constants (rounded, label-free quantiles), so a fold fit cannot leak through them, and the
freeze-frame features are local to the shot (shooting cone, nearest opponent, keeper), in
canonical metres (``football_core.pitch``). The frame is recorded at the strike, so it is
pre-shot.

:func:`design_matrix` is the only way the models turn a frame into numbers, and it calls the
guard first. It never fills a missing value: a null, NaN or infinity is an error, because a
fill value would be a modelling decision taken silently.
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
from football_core import pitch, schema

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


def _check_source(table: str, column: str, reader: str | None = None) -> None:
    """The guard for one C3 column: ``pre_shot``, and not restricted unless ``reader`` may."""
    known = next((t for t in _TABLES if t == table), None)
    if known is None:
        raise ValueError(f"unknown table {table!r}; expected one of {_TABLES}")
    key: Source = (known, column)
    if key not in _ROLES:
        raise ValueError(f"{column!r} is not a column of the C3 {table} table")
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

    Raises
    ------
    ValueError
        A bad name, no sources, a source that fails the guard (the message names its role), or
        a bad ``needs``.

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

    def __post_init__(self) -> None:
        taken = ("shots", self.name) in _ROLES or ("matches", self.name) in _ROLES
        if not isinstance(self.name, str) or not self.name or taken:
            raise ValueError(
                f"derived feature name {self.name!r} is empty or a C3 shot column (or a matches "
                "column)"
            )
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError(f"derived feature {self.name!r} needs a tuple of (table, column)")
        for table, column in self.sources:
            _check_source(table, column, reader=self.name)
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
CONTACTS: Final = tuple(c for c in schema.TECHNIQUES if c != "lob")
CONTACT: Final = pl.Enum(CONTACTS)


def _phase(t: Tables) -> pl.Series:
    sit = pl.col("situation")
    wide = (pl.col("y") < BOX_RIGHT_Y) | (pl.col("y") > BOX_LEFT_Y)
    expr = (
        pl.when(sit.is_in(["open_play", "kick_off"]))
        .then(pl.lit("open_play"))
        .when(sit == "fast_break")
        .then(pl.lit("fast_break"))
        .when(sit == "from_corner")
        .then(pl.lit("corner"))
        .when((sit == "from_free_kick") | ((sit == "direct_free_kick") & wide))
        .then(pl.lit("free_kick"))
        .when(sit == "from_throw_in")
        .then(pl.lit("throw_in"))
        .when(sit == "direct_free_kick")
        .then(pl.lit("direct_free_kick"))
        .otherwise(None)  # penalty, direct_corner: excluded, so design_matrix raises on them
        .cast(PHASE)
    )
    return t["shots"].select(expr).to_series()


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
            *FRAME_FEATURES,
        )
    }
)


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
        name, a derived feature without the features in its ``needs``, or a duplicate name.

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

    Returns
    -------
    matrix : numpy.ndarray
        Shape ``(shots.height, number of design columns)``.
    names : tuple of str
        The design column names, in order.

    Raises
    ------
    TypeError, ValueError
        As :func:`check_xg_features`; also a feature missing from the frame, a null, NaN or
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
    names = check_xg_features(features, derived)
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
