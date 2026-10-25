"""M0: the legacy v0 xG, scored as a test-only benchmark on the WhoScored shots.

v0 is the xG model of an earlier project on the same archive: a logistic regression on ten
inputs, stored as a JSON file of coefficients (``xg_model.json``). It is private, so it lives
under the git-ignored ``data/legacy/`` and :func:`read_v0` pins its SHA-256: another file is
an error, never a silently different benchmark.

**v0's own inputs, never C3's.** v0 was fitted on raw Opta 0-100 coordinates scaled linearly
to 105 × 68 m, with the distance clipped below at 0.5 m and its own arctan2 opening angle. C3
maps Opta piecewise between the IFAB landmarks (``football_core.pitch``), so v0 is scored on
the raw Opta point recovered by ``pitch.from_canonical``, through :func:`v0_inputs`, and never
on C3's ``distance_m`` / ``angle_rad``. Its seven flags come from C5 columns (header ⇔
``body_part == head``, set piece ⇔ ``situation`` in from free kick or from throw-in, ...).

v0 reads Opta's big chance (q214), an analyst's label that no candidate model may read
(``features.ABLATION_ONLY``). Here it is a benchmark input, read outside
``features.design_matrix``. A shot with a null v0 input is not covered: its score is null and
counted, never filled (:func:`covered`).

:func:`refit_v0` refits v0's own inputs on the train shots with some of them dropped: the
"remove" counterfactual of the v1 card (v0 without q214).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import pitch
from football_core.io import whoscored
from scipy.linalg import LinAlgWarning
from sklearn.exceptions import ConvergenceWarning

from football_shot_models import ablation
from football_shot_models.evaluation import labels

FloatArray = npt.NDArray[np.float64]
StrPath = str | os.PathLike[str]

V0_SHA256: Final[str] = "24c8f23d059b5dcab82858ad1ea6554ec3fc599a3da5071f3c2108c8c657b740"
V0_FEATURES: Final[tuple[str, ...]] = (
    "log_dist",
    "dist_m",
    "angle_rad",
    "header",
    "big_chance",
    "one_on_one",
    "fast_break",
    "from_corner",
    "set_piece",
    "direct_freekick",
)
V0_TITLE: Final[str] = "Legacy v0" + (
    f" (uses Opta big chance q{whoscored.BIG_CHANCE})" if "big_chance" in V0_FEATURES else ""
)
DEFAULT_V0: Final[str] = "data/legacy/xg_model.json"
# v0's own geometry: a linear 0-100 -> metres scale and its goal width
V0_PITCH_M: Final[tuple[float, float]] = (105.0, 68.0)
V0_GOAL_M: Final[float] = 7.32
V0_MIN_DIST_M: Final[float] = 0.5
COVER: Final[tuple[str, ...]] = ("x", "y", "body_part", "situation", "big_chance", "one_on_one")
_ISO_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True)
class LegacyModel:
    """The pinned v0 file, read.

    Attributes
    ----------
    coefficients : mapping of str to float
        One per ``V0_FEATURES`` name, in that order.
    intercept : float
        The logistic intercept.
    penalty_xg : float
        v0's constant P(goal) of an in-game penalty.
    trained_on : datetime.date
        The date v0 was fitted; it must precede the first test match.
    trained_shots : int
        The shots it was fitted on.
    sha256 : str
        The file's SHA-256 (equal to ``V0_SHA256``).
    """

    coefficients: Mapping[str, float]
    intercept: float
    penalty_xg: float
    trained_on: datetime.date
    trained_shots: int
    sha256: str


def read_v0(path: StrPath = DEFAULT_V0) -> LegacyModel:
    """Read the v0 JSON after checking its SHA-256 against ``V0_SHA256``.

    Behind the hash pin only what the run relies on is checked: the feature list and the
    coefficient keys, and ``trained_on.date`` as a real ``YYYY-MM-DD`` date.

    Parameters
    ----------
    path : str or path, default DEFAULT_V0
        The copied ``xg_model.json`` (``make legacy-v0``).

    Returns
    -------
    LegacyModel

    Raises
    ------
    ValueError
        A SHA-256 other than ``V0_SHA256`` (read when called; the message names both);
        features or coefficient keys other than ``V0_FEATURES``; a ``trained_on.date`` that is
        not a ``YYYY-MM-DD`` string of a real date.
    FileNotFoundError
        From the operating system, unchanged.

    Examples
    --------
    >>> read_v0().trained_shots  # doctest: +SKIP
    48882
    """
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != V0_SHA256:  # the module constant, read at call time (tests patch it)
        raise ValueError(f"{path}: SHA-256 {digest} is not the pinned v0 {V0_SHA256}")
    doc = json.loads(raw)
    if tuple(doc["features"]) != V0_FEATURES or set(doc["coefficients"]) != set(V0_FEATURES):
        raise ValueError(f"v0 features {doc['features']} are not {list(V0_FEATURES)}")
    when = doc["trained_on"]["date"]
    try:
        if not isinstance(when, str) or not _ISO_DATE.fullmatch(when):
            raise ValueError("not YYYY-MM-DD")
        trained_on = datetime.date.fromisoformat(when)  # also refuses 2026-13-01
    except ValueError as e:
        raise ValueError(f"v0 trained_on.date must be an ISO date, got {when!r}: {e}") from None
    return LegacyModel(
        MappingProxyType({f: float(doc["coefficients"][f]) for f in V0_FEATURES}),
        float(doc["intercept"]),
        float(doc["penalty_xg"]),
        trained_on,
        int(doc["trained_on"]["shots"]),
        digest,
    )


def covered(shots: pl.DataFrame) -> pl.Series:
    """True for the shots v0 can score: every v0 input (``COVER``) is non-null.

    Examples
    --------
    >>> shots = pl.DataFrame({"x": [94.0, None], "y": [34.0, 34.0],
    ...                       "body_part": ["head", "head"], "situation": ["open_play"] * 2,
    ...                       "big_chance": [False, False], "one_on_one": [False, False]})
    >>> covered(shots).to_list()
    [True, False]
    """
    return shots.select(
        pl.all_horizontal(pl.col(c).is_not_null() for c in COVER).alias("v0_covered")
    ).to_series()


def v0_inputs(shots: pl.DataFrame) -> dict[str, FloatArray]:
    """v0's ten inputs, by name, from C5 columns (covered shots only).

    The Opta point is recovered from canonical metres with ``pitch.from_canonical``, then
    scaled linearly as v0 was: distance to the goal centre clipped below at
    ``V0_MIN_DIST_M``, and the opening angle ``arctan2(w dx, dx² + dy² - (w/2)²)`` mapped into
    ``[0, π]``.

    Examples
    --------
    >>> x, y = pitch.to_canonical([88.5, 100.0], [50.0, 50.0], "opta")
    >>> shots = pl.DataFrame({"x": x, "y": y, "body_part": ["right_foot", "head"],
    ...                       "situation": ["open_play", "from_throw_in"],
    ...                       "big_chance": [False, True], "one_on_one": [False, False]})
    >>> inputs = v0_inputs(shots)
    >>> inputs["dist_m"].round(6).tolist(), inputs["angle_rad"].round(6).tolist()
    ([12.075, 0.5], [0.588607, 3.141593])
    >>> inputs["header"].tolist(), inputs["set_piece"].tolist()
    ([0.0, 1.0], [0.0, 1.0])
    """
    length, width = V0_PITCH_M
    half = V0_GOAL_M / 2
    ox, oy = pitch.from_canonical(shots["x"].to_numpy(), shots["y"].to_numpy(), "opta")
    dx = (100.0 - ox) / 100.0 * length
    dy = np.abs(oy - 50.0) / 100.0 * width
    dist = np.clip(np.sqrt(dx**2 + dy**2), V0_MIN_DIST_M, None)
    angle = np.arctan2(2 * half * dx, dx**2 + dy**2 - half**2)
    angle = np.where(angle < 0, angle + np.pi, angle)
    sit = shots["situation"].cast(pl.String)
    flags = {
        "header": shots["body_part"].cast(pl.String) == "head",
        "big_chance": shots["big_chance"],
        "one_on_one": shots["one_on_one"],
        "fast_break": sit == "fast_break",
        "from_corner": sit == "from_corner",
        "set_piece": sit.is_in(["from_free_kick", "from_throw_in"]),
        "direct_freekick": sit == "direct_free_kick",
    }
    out = {"log_dist": np.log(dist), "dist_m": dist, "angle_rad": angle}
    return out | {k: v.cast(pl.Float64).to_numpy() for k, v in flags.items()}


def v0_proba(
    x: Mapping[str, FloatArray], coef: Mapping[str, float], intercept: float
) -> FloatArray:
    """v0's P(goal): ``1 / (1 + exp(-clip(z, -35, 35)))`` with ``z`` its linear predictor.

    Examples
    --------
    >>> x = {f: np.zeros(1) for f in V0_FEATURES}
    >>> v0_proba(x, {f: 1.0 for f in V0_FEATURES}, 0.0).tolist()
    [0.5]
    """
    z = intercept + sum(coef[f] * x[f] for f in V0_FEATURES)
    p: FloatArray = 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
    return p


def score_v0(shots: pl.DataFrame, model: LegacyModel) -> pl.Series:
    """v0's xG of every shot: ``v0_xg`` (Float64), null where :func:`covered` is False.

    Examples
    --------
    >>> score_v0(held, read_v0()).null_count()  # doctest: +SKIP
    0
    """
    cov = covered(shots)
    out = pl.Series("v0_xg", [None] * shots.height, dtype=pl.Float64)
    if not cov.any():
        return out
    p = v0_proba(v0_inputs(shots.filter(cov)), model.coefficients, model.intercept)
    return out.scatter(cov.arg_true(), p)


def v0_design(shots: pl.DataFrame, drop: Sequence[str]) -> FloatArray:
    """v0's inputs as columns, in v0's order, without the ``drop`` names.

    Examples
    --------
    >>> x, y = pitch.to_canonical([88.5], [50.0], "opta")
    >>> shots = pl.DataFrame({"x": x, "y": y, "body_part": ["head"], "situation": ["open_play"],
    ...                       "big_chance": [True], "one_on_one": [False]})
    >>> v0_design(shots, ("big_chance",)).shape
    (1, 9)
    """
    x = v0_inputs(shots)
    return np.column_stack([x[f] for f in V0_FEATURES if f not in drop])


def refit_v0(dev: pl.DataFrame, held: pl.DataFrame, drop: Sequence[str]) -> FloatArray:
    """Refit v0's inputs minus ``drop`` on ``dev`` (``ablation.make_lr``); P(goal) on ``held``.

    A labelled counterfactual, never a candidate: it reads ``big_chance`` outside the
    leakage guard when it is not dropped. Only ``dev`` labels are read.

    Parameters
    ----------
    dev, held : polars.DataFrame
        Train and test shots, every row covered.
    drop : sequence of str
        ``V0_FEATURES`` names to leave out.

    Returns
    -------
    numpy.ndarray
        One probability per ``held`` row.

    Raises
    ------
    ValueError
        ``drop`` a bare ``str`` or naming a column outside ``V0_FEATURES``; an uncovered row
        in either frame; a fit that does not converge or has a singular Hessian.

    Examples
    --------
    >>> refit_v0(dev, held, drop=("big_chance",)).shape  # doctest: +SKIP
    (16164,)
    """
    if isinstance(drop, str) or set(drop) - set(V0_FEATURES):
        raise ValueError(f"drop must be a sequence of v0 inputs {list(V0_FEATURES)}, got {drop!r}")
    for name, frame in (("dev", dev), ("held", held)):
        if n := int((~covered(frame)).sum()):
            raise ValueError(f"refit_v0: {n} {name} row(s) lack a v0 input {list(COVER)}")
    lr = ablation.make_lr()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", LinAlgWarning)
        try:
            lr.fit(v0_design(dev, drop), labels(dev))
        except (ConvergenceWarning, LinAlgWarning) as w:
            raise ValueError(f"refit_v0 (drop {list(drop)}): the fit failed ({w})") from None
    p: FloatArray = np.ascontiguousarray(lr.predict_proba(v0_design(held, drop))[:, 1])
    return p
