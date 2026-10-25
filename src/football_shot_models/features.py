"""The M1.0 leakage guard: xG features come only from C3 columns whose role is ``pre_shot``.

An xG model must only see what was known when the ball was struck. The C3 schema
(``football_core.schema``) gives every shot column a role, so the allow-list is derived from
it rather than written by hand: a column added to C3 later is never a feature unless C3 calls
it ``pre_shot``. Outcome, post-shot (goal-mouth point, deflection), restricted (big chance)
and lineage columns (provider xG, raw payloads) are refused with their role in the message.

:func:`design_matrix` is the only way the models turn a frame into numbers, and it calls the
guard first. It never fills a missing value: a null, NaN or infinity is an error, because a
fill value would be a modelling decision taken silently.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import schema

FloatArray = npt.NDArray[np.float64]

XG_ALLOWED: Final[frozenset[str]] = frozenset(schema.column_names("shots", roles="pre_shot"))
_ROLES: Final[Mapping[str, str]] = MappingProxyType({c.name: c.role for c in schema.SHOT_COLUMNS})


def check_xg_features(features: Sequence[str]) -> tuple[str, ...]:
    """Check that every name is a C3 shot column with role ``pre_shot``.

    Parameters
    ----------
    features : sequence of str
        Candidate feature columns.

    Returns
    -------
    tuple of str
        The names, in the given order.

    Raises
    ------
    TypeError
        If ``features`` is a single ``str``.
    ValueError
        A name that is not a C3 shot column, a column whose role is not ``pre_shot`` (the
        message names the role), or a duplicate name.

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
        if name not in _ROLES:
            raise ValueError(f"{name!r} is not a column of the C3 shots table")
        if name not in XG_ALLOWED:
            raise ValueError(
                f"column {name!r} has role {_ROLES[name]!r}; xG features must have role 'pre_shot'"
            )
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    return names


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
    shots: pl.DataFrame, features: Sequence[str]
) -> tuple[FloatArray, tuple[str, ...]]:
    """The float64 feature matrix of the given xG features, and its column names.

    Numbers become float64, booleans 0.0 / 1.0, and an Enum one indicator per category after
    the first (the reference), named ``f"{col}={category}"``.

    Parameters
    ----------
    shots : polars.DataFrame
        Shots holding the feature columns.
    features : sequence of str
        Feature columns, checked by :func:`check_xg_features`.

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
        infinite value, or an unsupported dtype (List, String).

    Examples
    --------
    >>> shots = pl.DataFrame({"distance_m": [11.0, 20.0],
    ...                       "body_part": pl.Series(["head", "right_foot"],
    ...                                              dtype=schema.BODY_PART)})
    >>> matrix, names = design_matrix(shots, ["distance_m", "body_part"])
    >>> names
    ('distance_m', 'body_part=left_foot', 'body_part=head', 'body_part=other')
    >>> matrix.tolist()
    [[11.0, 0.0, 1.0, 0.0], [20.0, 0.0, 0.0, 0.0]]
    """
    columns = [c for col in check_xg_features(features) for c in _columns(shots, col)]
    if not columns:
        return np.empty((shots.height, 0), dtype=np.float64), ()
    names = tuple(name for name, _ in columns)
    frame = shots.select(expr.alias(name) for name, expr in columns)
    matrix: FloatArray = np.asarray(frame.to_numpy(), dtype=np.float64)
    return matrix, names
