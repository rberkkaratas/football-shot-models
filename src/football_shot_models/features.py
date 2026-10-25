"""The M1.0 leakage guard: xG features come only from C3 columns whose role is ``pre_shot``.

An xG model must only see what was known when the ball was struck. The C3 schema
(``football_core.schema``) gives every shot and freeze-frame column a role, so the allow-list
is derived from it rather than written by hand: a column added to C3 later is never a feature
unless C3 calls it ``pre_shot``. Outcome, post-shot (goal-mouth point, deflection), restricted
(big chance) and lineage columns (provider xG, raw payloads) are refused with their role in
the message.

Two additions sit on top of the roles:

- ``RESTRICTED_UNTIL`` refuses a ``pre_shot`` column that is known to be recorded on outcome,
  with the reason and the follow-up that will lift it. ``situation`` is one: StatsBomb logs
  direct corners and wide direct free kicks as shots mostly when they threaten goal (C4 F5).
- A :class:`DerivedFeature` (a freeze-frame count, a spline input) names the C3 columns it
  reads. Each source must pass the same check, and :func:`design_matrix` hands the feature's
  function only those columns (plus ``shot_id``, the join key), so it cannot read others.

:func:`design_matrix` is the only way the models turn a frame into numbers, and it calls the
guard first. It never fills a missing value: a null, NaN or infinity is an error, because a
fill value would be a modelling decision taken silently.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import schema

FloatArray = npt.NDArray[np.float64]
Source = tuple[schema.Table, str]
DeriveFn = Callable[[Mapping[schema.Table, pl.DataFrame]], pl.Series]

RESTRICTED_UNTIL: Final[Mapping[str, str]] = MappingProxyType(
    {"situation": "F5 (direct_corner / direct_free_kick shots are logged on outcome)"}
)
XG_ALLOWED: Final[frozenset[str]] = frozenset(
    schema.column_names("shots", roles="pre_shot")
) - frozenset(RESTRICTED_UNTIL)
_TABLES: Final[tuple[schema.Table, ...]] = ("shots", "freeze_frames")
_ROLES: Final[Mapping[Source, str]] = MappingProxyType(
    {("shots", c.name): c.role for c in schema.SHOT_COLUMNS}
    | {("freeze_frames", c.name): c.role for c in schema.FREEZE_FRAME_COLUMNS}
)


def _check_source(table: str, column: str) -> None:
    """The guard for one C3 column: a ``pre_shot`` column that is not restricted."""
    if table not in _TABLES:
        raise ValueError(f"unknown table {table!r}; expected one of {_TABLES}")
    key: Source = ("shots" if table == "shots" else "freeze_frames", column)
    if key not in _ROLES:
        raise ValueError(f"{column!r} is not a column of the C3 {table} table")
    role = _ROLES[key]
    prefix = "" if table == "shots" else "freeze_frames "
    if role != "pre_shot":
        raise ValueError(
            f"{prefix}column {column!r} has role {role!r}; xG features must have role 'pre_shot'"
        )
    if table == "shots" and column in RESTRICTED_UNTIL:
        raise ValueError(
            f"column {column!r} must not be an xG feature until {RESTRICTED_UNTIL[column]}"
        )


@dataclass(frozen=True)
class DerivedFeature:
    """A feature computed from C3 columns, checked like one.

    Attributes
    ----------
    name : str
        The feature name; not a C3 shot column.
    sources : tuple of (table, column)
        Every C3 column ``fn`` reads; each must be ``pre_shot`` (and not restricted) in
        ``"shots"`` or ``"freeze_frames"``.
    fn : callable
        ``fn(tables) -> polars.Series``, one value per shot in shots order. ``tables`` maps
        each source table to a frame holding ``shot_id`` and the declared columns only.

    Raises
    ------
    ValueError
        A bad name, no sources, or a source that fails the guard (the message names its role).

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

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or ("shots", self.name) in _ROLES:
            raise ValueError(f"derived feature name {self.name!r} is empty or a C3 shot column")
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError(f"derived feature {self.name!r} needs a tuple of (table, column)")
        for table, column in self.sources:
            _check_source(table, column)


DERIVED: Final[Mapping[str, DerivedFeature]] = MappingProxyType({})  # filled from M1.2 on


def check_xg_features(
    features: Sequence[str], derived: Mapping[str, DerivedFeature] = DERIVED
) -> tuple[str, ...]:
    """Check that every name is an allowed C3 shot column or a registered derived feature.

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
        A name that is neither a C3 shot column nor registered, a column whose role is not
        ``pre_shot`` (the message names the role), a column in ``RESTRICTED_UNTIL`` (the
        message gives the reason), a registry key that is not its feature's name, or a
        duplicate name.

    Examples
    --------
    >>> check_xg_features(["distance_m", "body_part"])
    ('distance_m', 'body_part')
    >>> check_xg_features(["goal_y"])
    Traceback (most recent call last):
    ...
    ValueError: column 'goal_y' has role 'post_shot'; xG features must have role 'pre_shot'
    """
    if isinstance(features, str):
        raise TypeError("features must be a sequence of column names, not a single str")
    names = tuple(features)
    for name in names:
        if name in derived:
            if derived[name].name != name:
                raise ValueError(
                    f"derived feature registered as {name!r} is {derived[name].name!r}"
                )
            continue
        if ("shots", name) not in _ROLES:
            raise ValueError(
                f"{name!r} is not a column of the C3 shots table or a registered derived feature"
            )
        _check_source("shots", name)
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    return names


def _derive(
    feature: DerivedFeature, shots: pl.DataFrame, freeze_frames: pl.DataFrame | None
) -> pl.Series:
    """Run ``feature.fn`` on its declared columns only."""
    frames = {"shots": shots, "freeze_frames": freeze_frames}
    inputs: dict[schema.Table, pl.DataFrame] = {}
    for table in dict.fromkeys(t for t, _ in feature.sources):
        frame = frames[table]
        if frame is None:
            raise ValueError(f"derived feature {feature.name!r} reads {table}; pass {table}=")
        columns = dict.fromkeys(["shot_id", *(c for t, c in feature.sources if t == table)])
        if missing := [c for c in columns if c not in frame.columns]:
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
