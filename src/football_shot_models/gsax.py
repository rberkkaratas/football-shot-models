"""MS6 goalkeepers: goals saved above expected (P3.1), a keeper model (P3.2), stability (P3.3).

**Rating (P3.1).** A keeper's goals saved above expected (GSAx) is Σ xGOT − goals over the
on-target shots he faced. Every row's xGOT comes from fits that never saw its season
(``season_out.season_out``), and the rating is **against the league sum**: one logit shift δ
per rating group (:func:`league_shift`) makes Σ q = goals, ``q = expit(logit xGOT + δ)``, so the
xGOT level gap is not charged to keepers. Open play has one δ per competition-season;
penalties one δ per season-out source (PM 2026-10-28 (a): a season has too few kicks for its
own δ). A δ is fitted only where E2's minimum holds (:func:`too_small`), and team-selected
competition-seasons are rated at δ = 0 and flagged (``ZERO_FAMILIES``): their league sum is one
team's keeper against its opponents. A logit shift, not a ratio, keeps every q below 1 on
near-certain shots. GSAx is a record of shots faced, rated against the league; positive means
fewer goals conceded than a league-average keeper facing the same shots. The SD is the null SD
√Σ q(1 − q) of a league-average keeper; the interval is a match bootstrap with the rating held
fixed.

**Keeper model (P3.2).** ``goal ~ offset(logit q) + (1 | group)``, no fixed intercept (the
rated offset fixes the level), fitted by a Laplace approximation (:func:`fit_keeper_model`):
τ̂ maximises the approximate marginal likelihood, and each group's effect is its posterior
mode, shrunk towards 0 as its shots get fewer. It is tested on the next season's on-target
shots against the raw (unpooled) estimate and against xGOT alone (:func:`next_season`).

**Stability (P3.3).** Split-half reliability of q − goal per shot, alternating shots within a
keeper in time order (PM X5), for the keepers whose total shots faced reach each tier
(:func:`reliability`); a tier no keeper reaches is reported empty.

**Sensitivities (PR1, StatsBomb).** A projected save's ``goal_z`` is where the keeper met the
ball, not its height at the line: :func:`line_rated` re-rates GSAx on goals and the saves
stopped within ``LINE_YD`` of the line, and ``season_out(f, without=NOZ_FEATURES)`` refits the
headline without ``mouth_z`` (PM E2 (c)). Both are printed beside the headline, never as it,
and both are summarised over the keeper-seasons whose competition-season has a δ fitted on the
line subset: that subset keeps every goal and few saves, and a season at δ = 0 cannot re-level
it, so its shift is the selection, not ``goal_z`` (:func:`shift_summary`).

Pure: nothing here fits an xGOT, reads the cache, writes or prints; :func:`fit` calls
``season_out.season_out`` for every fit and :func:`assemble` is arithmetic on its rows, so this
module is hash-exempt (``cache.ORCHESTRATION_ONLY``). The model, next-season, reliability and
CI-count functions take the "without penalties" rows.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import goalmouth, pitch
from football_core.io import statsbomb
from scipy import optimize
from scipy.special import expit

from football_shot_models import data, features, metrics, season_out, xgot
from football_shot_models.metrics import Check, GainCI, SplitHalf

Result = Literal["met", "missed", "n/a"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SEASON_KEY: Final = season_out.SEASON_KEY
RATED_KINDS: Final = ("open_play", "penalty")
WITHOUT: Final = "without penalties"
WITH: Final = "with penalties"
VARIANTS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {WITHOUT: ("open_play",), WITH: RATED_KINDS}
)
KEEPER_SEASON: Final = ("gk_id", *SEASON_KEY)
CAREER: Final = ("gk_id",)
CI_LEVEL: Final[float] = 0.90
SUM_TOL: Final[float] = 1e-9  # goals
PRINT_FLOOR: Final[float] = 1e-12  # sums below it print as "< 1e-12": thread-order noise
TIERS: Final[tuple[int, ...]] = (100, 250, 500)
MIN_TIER_KEEPERS: Final[int] = 30  # Fisher-z 95 % half-width about 0.37 at 30 keepers
TABLE_MIN_SHOTS: Final[int] = 100
NEXT_SEASON_CHECK: Final[int] = 5  # the check row (1-based) the keeper-model card prints
TEAM_SELECTED: Final[str] = "team-selected"
# keeper-season shots from which a team-selected keeper is the selected side's, not an opponent's
SELECTED_SIDE: Final[int] = 30
# placement bands of the rated q (result audit): cut points; every band starts at 0
BAND_CUTS: Final[Mapping[str, tuple[float, ...]]] = MappingProxyType({
    "mouth_y_abs": xgot.Y_BANDS[1:], "mouth_z": (0.3, 0.6, 1.0, 1.5, 2.0, 2.3),
    "angle_rad": (0.3, 0.4, 0.5, 0.7, 1.0),
})  # fmt: skip
OUTSIDE: Final[str] = f"outside the frame + {goalmouth.GOAL_TOLERANCE:g} m"
POOLED: Final[str] = "all"
TAU_BOUNDS: Final[tuple[float, float]] = (1e-3, 3.0)
NEWTON_TOL: Final[float] = 1e-10
NEWTON_MAX: Final[int] = 100
MAX_STEP: Final[float] = 5.0
BRACKET: Final[float] = 80.0  # ±40 fails to bracket with about e^6 rows at the logit clip
_REQUIRED: Final = ("gk_id", "xgot", "is_goal", "match_id", *SEASON_KEY, "competition_name",
                    "season_name", "kind")  # fmt: skip
_RATE_REQUIRED: Final = (*_REQUIRED, "family", "source")
# one δ per group of these columns (PM 2026-10-28 (a)): a competition-season for open play, the
# season-out source for penalties (the provider is the fit)
RATING_GROUPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {"open_play": SEASON_KEY, "penalty": ("source",)}
)
ZERO_FAMILIES: Final[tuple[str, ...]] = (TEAM_SELECTED,)  # rated at δ = 0, flagged (PM 2026-10-28)
# every row at δ = 0 by rule: an E2-small season, a zero-family season, an E2-small penalty pool
AT_ZERO: Final = pl.col("small") | pl.col("zero_family") | pl.col("pool_small")
# judged GSAx bands outside metrics.BAND_SDS, listed on the card as a limitation (PM PR1 E1 (a))
ESCALATED_BANDS: Final[Mapping[str, frozenset[tuple[str, str, str]]]] = MappingProxyType({
    "statsbomb": frozenset(),
    "whoscored": frozenset({("mouth_y_abs", "[0, 1)", "holdout_fit"),
                            ("mouth_y_abs", "[3.3, 3.55)", "holdout_fit")}),
})  # fmt: skip
# goal_z sensitivity (PM after MS6): a save stopped within LINE_YD of the line; StatsBomb x 119
# less half a fine cell, so the fine grid's raw-119 saves are on it (coarse-grid matches are out)
LINE_YD: Final[float] = 1.0
LINE_X: Final[float] = float(
    pitch.to_canonical(120 - LINE_YD, 40, "statsbomb", cell_offset=statsbomb.FINE_OFFSET)[0]
)
LINE_PROVIDERS: Final = frozenset({"statsbomb"})  # a save's goal_z is the stop height there
NOZ_FEATURES: Final[tuple[str, ...]] = ("mouth_z",)  # PM E2 (c), PR1: the z-free refit
NOZ_PROVIDERS: Final = frozenset({"statsbomb"})


@dataclass(frozen=True)
class SmallLeague:
    """A competition-season too small for a league shift: at most ``max_keepers`` keepers or
    fewer than ``min_shots`` open-play rated shots."""

    max_keepers: int
    min_shots: int


# PM E2 (b), MS6: a league offset from two keepers absorbs their own performance
SMALL_LEAGUE: Final[SmallLeague | None] = SmallLeague(2, 50)


def too_small(units: int, rows: int, small: SmallLeague | None) -> bool:
    """E2's minimum, the one rule for every fitted league offset: a group with at most
    ``small.max_keepers`` keepers (``units``) or fewer than ``small.min_shots`` rows, or no row,
    is too small; never with ``small`` None.

    Not applied to a keeper's own shift (``next_season``'s raw estimate or the model's u): PM
    PR1 E3, the minimum protects a league offset from absorbing its keepers' performance.

    Examples
    --------
    >>> [too_small(k, n, SmallLeague(2, 50)) for k, n in ((2, 99), (3, 49), (3, 50), (9, 0))]
    [True, True, False, True]
    >>> too_small(0, 0, None)
    False
    """
    if small is None:
        return False
    return rows == 0 or units <= small.max_keepers or rows < small.min_shots


# ---- P3.1 rating ---------------------------------------------------------------------------


def league_shift(p: npt.ArrayLike, goals: npt.ArrayLike) -> float:
    """The logit shift δ with ``Σ expit(logit p + δ) = Σ goals``.

    ``brentq`` on [−``BRACKET``, ``BRACKET``] then Newton steps to machine precision; −inf
    with no goal, +inf when every row is a goal (no finite shift).

    Raises
    ------
    ValueError
        An empty input; shapes that differ; a ``p`` NaN or outside [0, 1]; a goal not 0 / 1.

    Examples
    --------
    >>> round(league_shift([0.2, 0.2], [1, 0]), 6) == round(math.log(4), 6)
    True
    >>> league_shift([0.3], [0]), league_shift([0.3], [1])
    (-inf, inf)
    """
    q, y = np.asarray(p, dtype=np.float64), np.asarray(goals, dtype=np.float64)
    if q.size == 0 or q.shape != y.shape:
        raise ValueError(
            f"league_shift: {q.size} probabilities for {y.size} goals (none or unequal)"
        )
    if not np.all((q >= 0) & (q <= 1)):
        raise ValueError("league_shift: a probability is NaN or outside [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("league_shift: a goal is not 0 / 1")
    total = float(y.sum())
    if total == 0:
        return -math.inf
    if total == y.size:
        return math.inf
    lp = metrics.logit(q)

    def gap(d: float) -> float:
        return float(expit(lp + d).sum()) - total

    d = float(optimize.brentq(gap, -BRACKET, BRACKET, xtol=1e-12))
    for _ in range(3):  # polish: brentq's xtol times Σ q(1 − q) can exceed SUM_TOL
        s = expit(lp + d)
        slope = float((s * (1 - s)).sum())
        if slope <= 0:
            break
        d -= gap(d) / slope
    return d


def _require(frame: pl.DataFrame, columns: Sequence[str], what: str) -> None:
    if missing := [c for c in columns if c not in frame.columns]:
        raise ValueError(f"{what}: missing column(s) {missing}")


def _validate(frame: pl.DataFrame, what: str, required: Sequence[str] = _REQUIRED) -> None:
    """The rows rule of every table: required columns, no null, xgot in [0, 1], a rated kind."""
    _require(frame, required, what)
    if nulls := {c: n for c in required if (n := frame[c].null_count())}:
        raise ValueError(f"{what}: null values {nulls}")
    x = frame["xgot"].cast(pl.Float64)
    if bad := int((x.is_nan() | (x < 0) | (x > 1)).sum()):
        raise ValueError(f"{what}: {bad} xgot value(s) are NaN or outside [0, 1]")
    if bad := frame.filter(~pl.col("kind").cast(pl.String).is_in(RATED_KINDS)).height:
        raise ValueError(f"{what}: {bad} row(s) of a kind outside {RATED_KINDS}")


def _zero_families(zero_families: Sequence[str]) -> tuple[str, ...]:
    if isinstance(zero_families, str):
        raise TypeError("zero_families must be a sequence of family names, not a single str")
    if bad := [v for v in zero_families if v not in data.FAMILIES]:
        raise ValueError(f"rate: zero_families {bad} are not families; {data.FAMILIES}")
    return tuple(zero_families)


def _season_flags(
    pred: pl.DataFrame, small: SmallLeague | None, zero: tuple[str, ...]
) -> pl.DataFrame:
    """``SEASON_KEY``, ``small`` (E2 on its open-play rows) and ``zero_family``, per season."""
    play = pl.col("kind").cast(pl.String) == "open_play"
    seasons = pred.group_by(SEASON_KEY).agg(
        families=pl.col("family").cast(pl.String).n_unique(),
        family=pl.col("family").cast(pl.String).first(),
        keepers=pl.col("gk_id").filter(play).n_unique(),
        shots=play.sum(),
    )
    if bad := seasons.filter(pl.col("families") > 1).height:
        raise ValueError(f"rate: {bad} competition-season(s) have more than one family")
    return seasons.select(
        *SEASON_KEY,
        small=pl.Series([too_small(k, n, small) for k, n in seasons.select("keepers", "shots")
                         .iter_rows()], dtype=pl.Boolean),
        zero_family=pl.col("family").is_in(zero),
    )  # fmt: skip


def rate(
    pred: pl.DataFrame, *, small: SmallLeague | None, zero_families: Sequence[str]
) -> pl.DataFrame:
    """Rate every row against its league: ``group``, ``delta``, ``q`` and three flags.

    In order: (1) a competition-season is ``small`` when its open-play rows fail E2's minimum
    (:func:`too_small`; no open-play row counts) and ``zero_family`` when its ``family`` is in
    ``zero_families``; every row of such a season, both kinds, gets δ = 0. (2) The other rows
    are grouped on ``RATING_GROUPS[kind]`` (a season for open play, the ``source`` for
    penalties); a group that fails E2's minimum gets δ = 0 and ``pool_small``, any other the
    :func:`league_shift` of its xGOT against its goals, so it sums to its league. (3) With
    ``small`` None no season or group is small. ``q = expit(logit xgot + delta)``, and
    ``q = goal`` when δ is infinite.

    Parameters
    ----------
    pred : polars.DataFrame
        ``season_out.season_out`` rows (``gk_id``, ``xgot``, ``is_goal``, ``match_id``,
        ``SEASON_KEY``, ``competition_name``, ``season_name``, ``kind``, ``family``,
        ``source``).
    small : SmallLeague or None
        E2's minimum; None fits every group.
    zero_families : sequence of str
        Families rated at δ = 0 (``ZERO_FAMILIES`` in :func:`fit`); values of
        ``data.FAMILIES``.

    Returns
    -------
    polars.DataFrame
        ``pred`` in its order plus ``group`` (String: ``"<competition_id> <season_id>"`` for
        open play and for the kicks of a small or zero-family season, else the ``source``),
        ``delta`` (Float64, ±inf allowed), ``q`` (Float64), ``small`` (the season's E2 flag),
        ``zero_family`` and ``pool_small`` (the penalty pool's E2 flag) (Boolean); empty in,
        empty out.

    Raises
    ------
    TypeError
        ``zero_families`` is a ``str``.
    ValueError
        A missing column; a null in a required column (count named); an ``xgot`` NaN or
        outside [0, 1]; a ``kind`` outside ``RATED_KINDS``; a ``zero_families`` value outside
        ``data.FAMILIES``; a ``competition_id`` or ``season_id`` with a space (the ``group``
        label must be unique); a competition-season with two families.

    Examples
    --------
    >>> pred = pl.DataFrame({"gk_id": ["g1", "g2", "g1"], "xgot": [0.2, 0.2, 0.5],
    ...                      "is_goal": [True, False, True], "match_id": ["m1", "m1", "m2"],
    ...                      "competition_id": ["c"] * 3, "season_id": ["s"] * 3,
    ...                      "competition_name": ["C"] * 3, "season_name": ["2020"] * 3,
    ...                      "kind": ["open_play", "open_play", "penalty"],
    ...                      "family": ["women"] * 3, "source": ["season_folds"] * 3})
    >>> r = rate(pred, small=None, zero_families=ZERO_FAMILIES)
    >>> [round(v, 4) for v in r["q"]], round(r["delta"][0], 4), r["group"].to_list()
    ([0.5, 0.5, 1.0], 1.3863, ['c s', 'c s', 'season_folds'])
    >>> rate(pred, small=None, zero_families=["women"])["q"].to_list()
    [0.2, 0.2, 0.5]
    """
    zero = _zero_families(zero_families)
    _validate(pred, "rate", _RATE_REQUIRED)
    spaced = pl.any_horizontal(pl.col(SEASON_KEY).cast(pl.String).str.contains(" ", literal=True))
    if bad := pred.filter(spaced).height:
        raise ValueError(f"rate: {bad} row(s) have a competition_id or season_id with a space")
    added = pl.Schema({
        "group": pl.String(), "delta": pl.Float64(), "q": pl.Float64(), "small": pl.Boolean(),
        "zero_family": pl.Boolean(), "pool_small": pl.Boolean(),
    })  # fmt: skip
    if pred.height == 0:
        return pred.with_columns(pl.lit(None, dtype=t).alias(c) for c, t in added.items())
    rows = pred.with_row_index("_i").join(
        _season_flags(pred, small, zero), on=SEASON_KEY, how="left", maintain_order="left"
    )
    at_season, kind = pl.col("small") | pl.col("zero_family"), pl.col("kind").cast(pl.String)
    season = pl.concat_str(pl.col(SEASON_KEY).cast(pl.String), separator=" ")
    parts = [rows.filter(at_season).with_columns(group=season, delta=pl.lit(0.0),
                                                  pool_small=pl.lit(False))]  # fmt: skip
    for k in RATED_KINDS:
        keys = RATING_GROUPS[k]
        for key, g in rows.filter((kind == k) & ~at_season).group_by(*keys):
            tiny = too_small(g["gk_id"].n_unique(), g.height, small)
            d = 0.0 if tiny else league_shift(g["xgot"].to_numpy(), g["is_goal"].to_numpy())
            label = " ".join(str(v) for v in key)
            parts.append(g.with_columns(group=pl.lit(label), delta=pl.lit(d, dtype=pl.Float64),
                                        pool_small=pl.lit(tiny)))  # fmt: skip
    out = pl.concat(parts).sort("_i")
    lq = metrics.logit(out["xgot"].cast(pl.Float64).to_numpy()) + out["delta"].to_numpy()
    q = pl.Series("q", expit(lq))
    q = pl.select(
        pl.when(out["delta"].is_infinite()).then(out["is_goal"].cast(pl.Float64)).otherwise(q)
    ).to_series()
    return out.with_columns(q.alias("q")).select(*pred.columns, *added).cast(added)


_GROUP_SCHEMA: Final = pl.Schema({
    "kind": pl.String(), "group": pl.String(), "rows": pl.Int64(), "keepers": pl.Int64(),
    "goals": pl.Int64(), "delta": pl.Float64(), "small": pl.Boolean(),
    "zero_family": pl.Boolean(), "pool_small": pl.Boolean(), "gap": pl.Float64(),
})  # fmt: skip


def rating_groups(rated: pl.DataFrame) -> pl.DataFrame:
    """One row per (``kind``, ``group``) of :func:`rate` output: its size, δ, flags and
    ``gap`` = Σ (q − goal), sorted by kind and group (the δ table a league chart reads).

    Raises
    ------
    ValueError
        ``delta``, ``small``, ``zero_family`` or ``pool_small`` not constant within a group
        (count of such groups named).

    Examples
    --------
    >>> r = pl.DataFrame({"kind": ["open_play"] * 2, "group": ["c s"] * 2, "gk_id": ["a", "b"],
    ...                   "is_goal": [True, False], "q": [0.5, 0.5], "delta": [0.1] * 2,
    ...                   "small": [False] * 2, "zero_family": [False] * 2,
    ...                   "pool_small": [False] * 2})
    >>> rating_groups(r).select("rows", "keepers", "goals", "delta", "gap").rows()
    [(2, 2, 1, 0.1, 0.0)]
    """
    flags = ("delta", "small", "zero_family", "pool_small")
    if rated.height == 0:
        return pl.DataFrame(schema=_GROUP_SCHEMA)
    out = rated.group_by(kind=pl.col("kind").cast(pl.String), group=pl.col("group")).agg(
        pl.col(flags).first(),
        rows=pl.len().cast(pl.Int64),
        keepers=pl.col("gk_id").n_unique().cast(pl.Int64),
        goals=pl.col("is_goal").sum().cast(pl.Int64),
        _varies=pl.max_horizontal(*(pl.col(c).n_unique() for c in flags)),
        gap=(pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum(),
    )
    if bad := out.filter(pl.col("_varies") > 1).height:
        raise ValueError(f"rating_groups: {bad} group(s) have more than one δ or flag value")
    return out.select(list(_GROUP_SCHEMA)).cast(_GROUP_SCHEMA).sort("kind", "group")


# ---- P3.1 tables ---------------------------------------------------------------------------


def _ci(frame: pl.DataFrame) -> tuple[float | None, float | None]:
    """The CI_LEVEL match-bootstrap interval of Σ (q − goal), rating fixed; None with 1 match."""
    ids = frame["match_id"].to_numpy()
    if frame["match_id"].n_unique() < 2:
        return None, None
    inv, w = metrics.group_weights(ids, metrics.N_BOOT, metrics.SEED)
    per = np.bincount(inv, weights=(frame["q"] - frame["is_goal"].cast(pl.Float64)).to_numpy())
    low, high = np.quantile(w @ per, [(1 - CI_LEVEL) / 2, (1 + CI_LEVEL) / 2])
    return float(low), float(high)


_KEEPER_SCHEMA: Final = pl.Schema({
    "gk_id": pl.String(), "competition_id": pl.String(), "season_id": pl.String(),
    "competition_name": pl.String(), "season_name": pl.String(), "family": data.FAMILY,
    "shots": pl.Int64(), "goals": pl.Int64(), "matches": pl.Int64(), "penalties": pl.Int64(),
    "open_goal": pl.Int64(), "xgot": pl.Float64(), "gsax_raw": pl.Float64(),
    "gsax": pl.Float64(), "open_goal_gsax": pl.Float64(), "sd": pl.Float64(),
    "ci_low": pl.Float64(), "ci_high": pl.Float64(),
})  # fmt: skip


def keeper_table(rated: pl.DataFrame) -> pl.DataFrame:
    """One row per keeper-season (``KEEPER_SEASON``) of the rows given, sorted by season key
    then ``gk_id``.

    ``goals`` counts rated shots only (own goals, direct corners and shoot-outs are not in it,
    so it is not goals conceded); ``gsax_raw`` = Σ xgot − goals, ``gsax`` = Σ q − goals; ``sd``
    = √Σ q(1 − q), the SD for a league-average keeper facing these shots (not the uncertainty
    of this keeper's estimate); ``ci_low`` / ``ci_high`` the ``CI_LEVEL`` match bootstrap of Σ
    (q − goal), null with one match. ``open_goal`` and ``open_goal_gsax`` are null when every
    row's ``open_goal`` is null (WhoScored records none).

    Examples
    --------
    >>> r = pl.DataFrame({"gk_id": ["g1"] * 2, "competition_id": ["c"] * 2,
    ...                   "season_id": ["s"] * 2, "competition_name": ["C"] * 2,
    ...                   "season_name": ["2020"] * 2, "family": ["women"] * 2,
    ...                   "match_id": ["m1", "m2"], "kind": ["open_play"] * 2,
    ...                   "is_goal": [True, False], "open_goal": [None, None],
    ...                   "xgot": [0.6, 0.3], "q": [0.5, 0.4]})
    >>> t = keeper_table(r)
    >>> t.select("shots", "goals", pl.col("gsax", "sd").round(4), "open_goal").rows()
    [(2, 1, -0.1, 0.7, None)]
    """
    _validate(rated, "keeper_table")
    if rated.height == 0:
        return pl.DataFrame(schema=_KEEPER_SCHEMA)
    og, goal = pl.col("open_goal"), pl.col("is_goal").cast(pl.Float64)
    none = og.null_count() == pl.len()
    table = rated.group_by(KEEPER_SEASON).agg(
        pl.col("competition_name", "season_name", "family").first(),
        shots=pl.len().cast(pl.Int64),
        goals=pl.col("is_goal").sum().cast(pl.Int64),
        matches=pl.col("match_id").n_unique().cast(pl.Int64),
        penalties=(pl.col("kind").cast(pl.String) == "penalty").sum().cast(pl.Int64),
        open_goal=pl.when(none).then(None).otherwise(og.sum()).cast(pl.Int64),
        xgot=pl.col("xgot").sum(),
        gsax_raw=pl.col("xgot").sum() - goal.sum(),
        gsax=pl.col("q").sum() - goal.sum(),
        open_goal_gsax=pl.when(none).then(None).otherwise(
            (pl.col("q") - goal).filter(og.fill_null(False)).sum()),
        sd=(pl.col("q") * (1 - pl.col("q"))).sum().sqrt(),
    ).sort(*SEASON_KEY, "gk_id")  # fmt: skip
    cis = {tuple(k): _ci(g) for k, g in rated.group_by(KEEPER_SEASON) if isinstance(k, tuple)}
    keys = list(table.select(KEEPER_SEASON).iter_rows())
    return table.with_columns(
        ci_low=pl.Series([cis[k][0] for k in keys], dtype=pl.Float64),
        ci_high=pl.Series([cis[k][1] for k in keys], dtype=pl.Float64),
    ).select(list(_KEEPER_SCHEMA))


def league_table(rated: pl.DataFrame) -> pl.DataFrame:
    """One row per competition-season (``SEASON_KEY``) of the rows given, sorted by key.

    ``gsax_raw`` is the league sum of xGOT − goals; ``delta`` the open-play shift and
    ``delta_penalty`` the penalty one (null without a kick).

    Examples
    --------
    >>> r = pl.DataFrame({"gk_id": ["g1", "g2"], "competition_id": ["c"] * 2,
    ...                   "season_id": ["s"] * 2, "competition_name": ["C"] * 2,
    ...                   "season_name": ["2020"] * 2, "family": ["women"] * 2,
    ...                   "match_id": ["m1"] * 2, "kind": ["open_play"] * 2,
    ...                   "is_goal": [True, False], "xgot": [0.6, 0.3], "delta": [0.1, 0.1],
    ...                   "small": [False] * 2})
    >>> league_table(r).select("keepers", "delta", "delta_penalty").rows()
    [(2, 0.1, None)]
    """
    _validate(rated, "league_table")
    kind = pl.col("kind").cast(pl.String)
    return (
        rated.group_by(SEASON_KEY)
        .agg(
            pl.col("competition_name", "season_name", "family").first(),
            shots=pl.len().cast(pl.Int64),
            goals=pl.col("is_goal").sum().cast(pl.Int64),
            keepers=pl.col("gk_id").n_unique().cast(pl.Int64),
            xgot=pl.col("xgot").sum(),
            gsax_raw=pl.col("xgot").sum() - pl.col("is_goal").cast(pl.Float64).sum(),
            delta=pl.col("delta").filter(kind == "open_play").first(),
            delta_penalty=pl.col("delta").filter(kind == "penalty").first(),
            small=pl.col("small").any(),
        )
        .sort(SEASON_KEY)
    )


def _open_play(rated: pl.DataFrame, what: str) -> pl.DataFrame:
    """The variant is pinned by code: these functions read "without penalties" rows only."""
    if bad := rated.filter(pl.col("kind").cast(pl.String) != "open_play").height:
        raise ValueError(f"{what}: {bad} row(s) are not open play; pass the {WITHOUT!r} rows")
    return rated


@dataclass(frozen=True)
class CiCounts:
    """Keeper-seasons whose ``CI_LEVEL`` interval excludes 0: observed, and with goals redrawn
    from q (no keeper differences); among those with >= ``TABLE_MIN_SHOTS`` shots and with
    >= 2 matches."""

    table_rows: int
    table_observed: int
    table_null: int
    multi_rows: int
    multi_observed: int
    multi_null: int


def ci_counts(rated: pl.DataFrame, *, seed: int) -> CiCounts:
    """How often the interval excludes 0, against goals drawn once as Bernoulli(q).

    Examples
    --------
    >>> ci_counts(rated_open, seed=metrics.SEED).table_rows  # doctest: +SKIP
    100
    """
    rated = _open_play(rated, "ci_counts")
    draw = np.random.default_rng(seed).random(rated.height) < rated["q"].to_numpy()
    obs = keeper_table(rated)
    null = keeper_table(rated.with_columns(is_goal=pl.Series(draw)))
    out = (pl.col("ci_low") > 0) | (pl.col("ci_high") < 0)
    big, multi = pl.col("shots") >= TABLE_MIN_SHOTS, pl.col("matches") >= 2
    return CiCounts(
        obs.filter(big).height, obs.filter(big & out).height, null.filter(big & out).height,
        obs.filter(multi).height, obs.filter(multi & out).height, null.filter(multi & out).height,
    )  # fmt: skip


def team_selected_split(table: pl.DataFrame) -> pl.DataFrame:
    """The team-selected keeper-seasons of a :func:`keeper_table`, by side.

    The selected side's keeper faces at least ``SELECTED_SIDE`` shots that season, the
    opponents' keepers fewer. At δ = 0 (PR1) each side's sum is its keepers' goals against
    xGOT; δ = 0 removes the league-sum coupling between the sides, not the gap between them,
    and this split cannot attribute that gap to finishing or keeping.

    Returns
    -------
    polars.DataFrame
        ``side`` ("selected side", then "opponents"; a side without rows is absent),
        ``keeper_seasons``, ``shots`` (Int64), ``gsax_raw`` (before δ), ``gsax`` and ``sd`` =
        √Σ sd² (Float64).

    Examples
    --------
    >>> t = pl.DataFrame({"family": [TEAM_SELECTED] * 3 + ["women"], "shots": [40, 5, 3, 9],
    ...                   "gsax_raw": [2.5, -1.0, -1.0, 4.0], "gsax": [2.0, -1.0, -0.5, 4.0],
    ...                   "sd": [3.0, 1.0, 0.0, 1.0]})
    >>> team_selected_split(t).rows()
    [('selected side', 1, 40, 2.5, 2.0, 3.0), ('opponents', 2, 8, -2.0, -1.5, 1.0)]
    """
    side = pl.when(pl.col("shots") >= SELECTED_SIDE).then(pl.lit("selected side"))
    return (
        table.filter(pl.col("family").cast(pl.String) == TEAM_SELECTED)
        .group_by(side.otherwise(pl.lit("opponents")).alias("side"))
        .agg(
            keeper_seasons=pl.len().cast(pl.Int64),
            shots=pl.col("shots").sum().cast(pl.Int64),
            gsax_raw=pl.col("gsax_raw").sum(),
            gsax=pl.col("gsax").sum(),
            sd=(pl.col("sd") ** 2).sum().sqrt(),
        )  # fmt: skip
        .sort("side", descending=True)
    )


def band_table(rated: pl.DataFrame, geometry: pl.DataFrame) -> pl.DataFrame:
    """Calibration of the rated q by band of ``|y|``, height and angle, per source and pooled.

    The xGOT card's band table judges the match out-of-fold xGOT; GSAx reads the season-out
    refits (on WhoScored, a fit on the test season alone), whose shape can differ, and the
    league shift δ removes a level gap, not a shape one. Bands of ``BAND_CUTS`` on
    ``features.placement``'s ``mouth_y_abs`` / ``mouth_z`` and on ``angle_rad``; the points
    recorded outside the frame are a ``|y|`` row of their own (``OUTSIDE``), as on the xGOT
    card. ``gap`` = Σ q − goals, ``sd`` = √Σ q(1 − q); ``judged`` and ``within`` are the
    xGOT card's rule (``xgot.judged``, ``metrics.within``, read when called).

    Parameters
    ----------
    rated : polars.DataFrame
        Open-play :func:`rate` rows (``shot_id``, ``q``, ``is_goal``, ``source``).
    geometry : polars.DataFrame
        ``shot_id`` (unique), ``goal_y``, ``goal_z``, ``on_target``, ``angle_rad`` of every
        rated row.

    Returns
    -------
    polars.DataFrame
        ``feature``, ``band``, ``source`` (String; ``POOLED`` first, then each source sorted),
        ``shots``, ``goals`` (Int64), ``gap``, ``sd`` (Float64), ``judged``, ``within``
        (Boolean); every band of every source, an empty band with 0 shots, not judged.

    Raises
    ------
    ValueError
        A row that is not open play; a rated row without geometry; from
        ``features.placement``.

    Examples
    --------
    >>> r = pl.DataFrame({"shot_id": ["a", "b"], "q": [0.5, 0.2], "is_goal": [True, False],
    ...                   "kind": ["open_play"] * 2, "source": ["holdout_fit"] * 2})
    >>> geo = pl.DataFrame({"shot_id": ["a", "b"], "goal_y": [34.0, 33.0], "goal_z": [0.5, 2.0],
    ...                     "on_target": [True, True], "angle_rad": [0.2, 1.2]})
    >>> t = band_table(r, geo)
    >>> t.filter((pl.col("source") == "all") & (pl.col("shots") > 0)).select(
    ...     "band", "shots", pl.col("gap").round(2)).rows()[2:]
    [('[0.3, 0.6)', 1, -0.5), ('[2, 2.3)', 1, 0.2), ('[0, 0.3)', 1, -0.5), ('>= 1', 1, 0.2)]
    >>> t.height, t.filter(pl.col("judged")).height  # 20 bands, 2 sources (all, holdout_fit)
    (40, 0)
    """
    rated = _open_play(rated, "band_table")
    cols = ("goal_y", "goal_z", "on_target", "angle_rad")
    if missing := rated.join(geometry, on="shot_id", how="anti").height:
        raise ValueError(f"band_table: {missing} rated row(s) have no geometry")
    rows = rated.select("shot_id", "q", "is_goal", "source").join(
        geometry.select("shot_id", *cols), on="shot_id", how="left", validate="m:1",
        maintain_order="left",
    )  # fmt: skip
    place = features.placement(rows.select(cols))
    values = {"mouth_y_abs": place["mouth_y_abs"], "mouth_z": place["mouth_z"],
              "angle_rad": rows["angle_rad"]}  # fmt: skip
    q, y = rows["q"].to_numpy(), rows["is_goal"].cast(pl.Float64).to_numpy()
    src = rows["source"].cast(pl.String).to_numpy()
    out = place["mouth_out"].to_numpy() > 0
    records = []
    for feature, cuts in BAND_CUTS.items():
        idx = np.searchsorted(cuts, values[feature].cast(pl.Float64).to_numpy(), side="right")
        bands = [f"[{lo:g}, {hi:g})" if math.isfinite(hi) else f">= {lo:g}"
                 for lo, hi in pairwise((0.0, *cuts, math.inf))]  # fmt: skip
        if feature == "mouth_y_abs":
            idx, bands = np.where(out, len(bands), idx), [*bands, OUTSIDE]
        for s in (POOLED, *sorted(set(src))):
            for i, band in enumerate(bands):
                m = (idx == i) & ((src == s) | (s == POOLED))
                n, goals = int(m.sum()), int(y[m].sum())
                gap, sd = metrics.poisson_binomial(y[m], q[m])
                records.append((feature, band, s, n, goals, gap, sd, metrics.within(gap, sd)))
    schema = {
        "feature": pl.String,
        "band": pl.String,
        "source": pl.String,
        "shots": pl.Int64,
        "goals": pl.Int64,
        "gap": pl.Float64,
        "sd": pl.Float64,
        "within": pl.Boolean,
    }
    table = pl.DataFrame(records, schema=schema, orient="row")
    return table.insert_column(7, xgot.judged(table))


# ---- P3.2 keeper model ---------------------------------------------------------------------


def keeper_modes(
    offset: FloatArray, goal: npt.ArrayLike, group: IntArray, *, tau: float
) -> tuple[FloatArray, FloatArray]:
    """Per-group posterior modes ``u`` of ``goal ~ offset + u[group]``, ``u ~ N(0, τ²)``.

    Vectorised Newton steps (clipped to ±``MAX_STEP``) until the largest step is below
    ``NEWTON_TOL``; ``h`` is Σ p(1 − p) per group at the mode.

    Raises
    ------
    ValueError
        An empty input; unequal lengths; a non-finite offset; a goal not 0 / 1; a code outside
        0 … G − 1 (G = max code + 1); ``tau`` not finite and > 0; no convergence in
        ``NEWTON_MAX`` steps.

    Examples
    --------
    >>> u, h = keeper_modes(np.zeros(4), [1, 1, 0, 0], np.array([0, 0, 1, 1]), tau=1.0)
    >>> [round(float(v), 4) for v in u]
    [0.6748, -0.6748]
    """
    if not (math.isfinite(tau) and tau > 0):
        raise ValueError(f"keeper_modes: tau must be finite and > 0, not {tau!r}")
    y = np.asarray(goal, dtype=np.float64)
    off, grp = np.asarray(offset, dtype=np.float64), np.asarray(group)
    if off.size == 0 or not off.shape == y.shape == grp.shape:
        raise ValueError(f"keeper_modes: {off.size} offsets, {y.size} goals, {grp.size} codes")
    if not np.all(np.isfinite(off)):
        raise ValueError("keeper_modes: an offset is not finite")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("keeper_modes: a goal is not 0 / 1")
    if not np.issubdtype(grp.dtype, np.integer) or int(grp.min()) < 0:
        raise ValueError("keeper_modes: group codes must be integers in 0 … G − 1")
    n = int(grp.max()) + 1
    u = np.zeros(n)
    for _ in range(NEWTON_MAX):
        p = expit(off + u[grp])
        grad = np.bincount(grp, y - p, n) - u / tau**2
        step = np.clip(grad / (np.bincount(grp, p * (1 - p), n) + 1 / tau**2), -MAX_STEP, MAX_STEP)
        u = u + step
        if float(np.abs(step).max()) < NEWTON_TOL:
            p = expit(off + u[grp])
            return u, np.bincount(grp, p * (1 - p), n)
    raise ValueError(f"keeper_modes: no convergence in {NEWTON_MAX} Newton steps (tau {tau})")


def _marginal(off: FloatArray, y: FloatArray, grp: IntArray, tau: float) -> float:
    """The Laplace log marginal likelihood at τ."""
    u, h = keeper_modes(off, y, grp, tau=tau)
    lp = off + u[grp]
    ll = np.bincount(grp, y * lp - np.logaddexp(0, lp), u.size)
    return float((ll - u**2 / (2 * tau**2) - np.log(tau) - 0.5 * np.log(h + 1 / tau**2)).sum())


@dataclass(frozen=True, eq=False)
class KeeperModel:
    """``goal ~ offset(logit q) + (1 | by)`` by Laplace.

    Attributes
    ----------
    by : tuple of str
        The grouping (``KEEPER_SEASON`` or ``CAREER``).
    tau : float
        τ̂, the SD of the effect on the logit scale.
    at_bound : bool
        τ̂ within 1e-3 (relative) of a ``TAU_BOUNDS`` bound.
    per_100 : float
        100 · τ̂ · q̄(1 − q̄): one SD of the effect in goals per 100 shots at the mean q.
    estimates : polars.DataFrame
        The ``by`` columns, ``shots`` (Int64), ``u`` and ``gsax_shrunk`` = Σ q − Σ expit(logit
        q + u) (Float64), one row per group, sorted by ``by``.
    """

    by: tuple[str, ...]
    tau: float
    at_bound: bool
    per_100: float
    estimates: pl.DataFrame


def _codes(rated: pl.DataFrame, by: Sequence[str]) -> tuple[pl.DataFrame, IntArray]:
    if isinstance(by, str):
        raise TypeError("by must be a sequence of column names, not a single str")
    keys = rated.select(by).unique().sort(by).with_row_index("code")
    codes = rated.select(by).join(keys, on=list(by), how="left", maintain_order="left")["code"]
    return keys.drop("code"), codes.to_numpy().astype(np.intp)


def fit_keeper_model(rated: pl.DataFrame, *, by: Sequence[str]) -> KeeperModel:
    """Fit τ̂ by the Laplace marginal and the per-group modes; code g is row g of the sorted keys.

    Raises
    ------
    TypeError
        ``by`` is a ``str``.
    ValueError
        A row that is not open play; from :func:`keeper_modes`.

    Examples
    --------
    >>> m = fit_keeper_model(rated_open, by=KEEPER_SEASON)  # doctest: +SKIP
    >>> m.estimates.columns  # doctest: +SKIP
    ['gk_id', 'competition_id', 'season_id', 'shots', 'u', 'gsax_shrunk']
    """
    rated = _open_play(rated, "fit_keeper_model")
    keys, grp = _codes(rated, by)
    q = rated["q"].to_numpy()
    off, y = metrics.logit(q), rated["is_goal"].cast(pl.Float64).to_numpy()
    lo, hi = TAU_BOUNDS
    best = optimize.minimize_scalar(
        lambda lt: -_marginal(off, y, grp, math.exp(lt)), bounds=(math.log(lo), math.log(hi)),
        method="bounded", options={"xatol": 1e-6},
    )  # fmt: skip
    tau = math.exp(float(best.x))
    u, _ = keeper_modes(off, y, grp, tau=tau)
    g = len(u)
    shrunk = np.bincount(grp, q, g) - np.bincount(grp, expit(off + u[grp]), g)
    qbar = float(q.mean())
    return KeeperModel(
        tuple(by), tau, any(abs(tau - b) <= 1e-3 * b for b in TAU_BOUNDS),
        100 * tau * qbar * (1 - qbar),
        keys.with_columns(shots=pl.Series(np.bincount(grp, minlength=g), dtype=pl.Int64),
                          u=pl.Series(u), gsax_shrunk=pl.Series(shrunk)),
    )  # fmt: skip


# ---- P3.2 next season ----------------------------------------------------------------------


def season_start(name: str) -> int:
    """The first year of a season name: ``"2015/2016"`` → 2015, ``"2018"`` → 2018.

    Raises
    ------
    ValueError
        Any other value (a trailing newline, non-ASCII digits or a second year that is not the
        first + 1 included), named.

    Examples
    --------
    >>> season_start("2015/2016"), season_start("2018")
    (2015, 2018)
    """
    m = re.fullmatch(r"([0-9]{4})(?:/([0-9]{4}))?", name) if isinstance(name, str) else None
    if m is None or (m[2] is not None and int(m[2]) != int(m[1]) + 1):
        raise ValueError(f"season_start: {name!r} is not 'YYYY' or 'YYYY/YYYY+1'")
    return int(m[1])


@dataclass(frozen=True)
class NextSeason:
    """The keeper model against raw GSAx and xGOT alone on the next season's shots.

    ``log_loss`` is (none, raw, shrunk); ``gain`` is shrunk over raw and ``gain_none`` shrunk
    over xGOT alone (``metrics.GAIN_LEVEL`` match-bootstrap intervals); the ``_loo`` pair uses
    the test season's δ fitted without the paired keepers' rows (δ = 0 in a season at δ = 0 by
    rule, and in ``loo_small`` seasons whose unpaired rows fail E2's minimum). A test season
    with no unpaired row or a non-finite δ without them is left out of the ``_loo`` pair:
    ``loo_missing`` seasons, ``loo_shots`` test shots and ``loo_pairs`` pairs (the ``_loo``
    gains are None when fewer than 2 test matches are left). ``kept_pairs`` are the scored
    pairs (``gk_id``, ``competition_id``, ``season_id``, ``next_season_id``), ``pairs`` of
    them. ``untestable`` names why there is no test, with the counts.
    """

    pairs: int
    left_out: int
    test_shots: int
    test_matches: int
    log_loss: tuple[float | None, float | None, float | None]
    gain: GainCI | None
    gain_none: GainCI | None
    gain_loo: GainCI | None
    gain_none_loo: GainCI | None
    loo_missing: int
    untestable: str | None
    loo_shots: int = 0
    loo_pairs: int = 0
    loo_small: int = 0
    kept_pairs: tuple[tuple[str, str, str, str], ...] = ()


def _log_loss(y: FloatArray, p: FloatArray) -> float:
    q = np.clip(p, metrics.EPS, 1 - metrics.EPS)
    return float(-(y * np.log(q) + (1 - y) * np.log1p(-q)).mean())


def next_season(rated: pl.DataFrame, *, by: Sequence[str]) -> NextSeason:
    """Pairs (same keeper and competition, next start year) scored on the next season.

    For each test year t + 1, the model is fitted on every rated row with start <= t
    (``by`` grouping); raw is the keeper's own no-pooling shift in his ``by`` group. A pair
    whose raw estimate is infinite (0 or all goals) is left out and counted.

    Examples
    --------
    >>> ns = next_season(rated_open, by=KEEPER_SEASON)  # doctest: +SKIP
    >>> ns.untestable is None  # doctest: +SKIP
    True
    """
    rated = _open_play(rated, "next_season")
    if isinstance(by, str):
        raise TypeError("by must be a sequence of column names, not a single str")
    by = tuple(by)
    rows = rated.with_columns(
        start=pl.col("season_name").map_elements(season_start, return_dtype=pl.Int64)
    )
    pairs = _pairs(rows)
    test_parts: list[pl.DataFrame] = []
    scored: list[tuple[str, str, str, str]] = []
    left_out = 0
    for t in sorted(set(pairs["start"])):
        train = rows.filter(pl.col("start") <= t)
        model = fit_keeper_model(train.drop("start"), by=by).estimates
        for pair in pairs.filter(pl.col("start") == t).iter_rows(named=True):
            key = {c: pair[c] for c in by}
            own = train.filter(pl.all_horizontal(pl.col(c) == v for c, v in key.items()))
            u_raw = league_shift(own["q"].to_numpy(), own["is_goal"].to_numpy())
            if not math.isfinite(u_raw):
                left_out += 1
                continue
            scored.append((pair["gk_id"], pair["competition_id"], pair["season_id"],
                           pair["next_season_id"]))  # fmt: skip
            hat = model.filter(pl.all_horizontal(pl.col(c) == v for c, v in key.items()))["u"]
            test_parts.append(rows.filter(
                (pl.col("gk_id") == pair["gk_id"]) & (pl.col("competition_id") ==
                pair["competition_id"]) & (pl.col("season_id") == pair["next_season_id"])
            ).with_columns(u_raw=pl.lit(u_raw), u_hat=pl.lit(float(hat[0]))))  # fmt: skip
    kept = len(scored)
    empty = NextSeason(kept, left_out, 0, 0, (None, None, None), None, None, None, None, 0, None,
                       kept_pairs=tuple(scored))  # fmt: skip
    if not test_parts:
        why = f"no pair with a finite raw estimate ({kept + left_out} pairs, {left_out} left out)"
        return dataclasses.replace(empty, untestable=why)
    test = pl.concat(test_parts)
    matches = test["match_id"].n_unique()
    if matches < 2:
        why = f"fewer than 2 test matches ({matches} match, {test.height} shots, {kept} pairs)"
        return dataclasses.replace(
            empty, test_shots=test.height, test_matches=matches, untestable=why
        )
    y = test["is_goal"].cast(pl.Float64).to_numpy()
    ids = test["match_id"].to_numpy()
    lq = metrics.logit(test["q"].to_numpy())
    ur, uh = test["u_raw"].to_numpy(), test["u_hat"].to_numpy()
    p = (expit(lq), expit(lq + ur), expit(lq + uh))
    loo, missing, loo_small = _loo_logits(rows, test)
    ok = np.isfinite(loo)
    gains_loo: tuple[GainCI | None, GainCI | None] = (None, None)
    if len(set(ids[ok])) >= 2:
        pp = (expit(loo[ok]), expit(loo[ok] + ur[ok]), expit(loo[ok] + uh[ok]))
        gains_loo = (metrics.bootstrap_gain(y[ok], pp[1], pp[2], ids[ok]),
                     metrics.bootstrap_gain(y[ok], pp[0], pp[2], ids[ok]))  # fmt: skip
    dropped = test.filter(pl.Series(~ok))
    return NextSeason(
        kept, left_out, test.height, matches,
        (_log_loss(y, p[0]), _log_loss(y, p[1]), _log_loss(y, p[2])),
        metrics.bootstrap_gain(y, p[1], p[2], ids), metrics.bootstrap_gain(y, p[0], p[2], ids),
        *gains_loo, missing, None, dropped.height,
        dropped.select(KEEPER_SEASON).n_unique() if dropped.height else 0,
        loo_small, tuple(scored),
    )  # fmt: skip


def _pairs(rows: pl.DataFrame) -> pl.DataFrame:
    """Keeper-seasons with the same keeper in the same competition the next start year:
    ``KEEPER_SEASON``, ``start`` and ``next_season_id``, sorted by ``KEEPER_SEASON``."""
    seasons = rows.select(*KEEPER_SEASON, "start").unique()
    return seasons.join(
        seasons.select("gk_id", "competition_id", next_season_id=pl.col("season_id"),
                       start=pl.col("start") - 1),
        on=["gk_id", "competition_id", "start"],
    ).sort(*KEEPER_SEASON)  # fmt: skip


def _loo_logits(rows: pl.DataFrame, test: pl.DataFrame) -> tuple[FloatArray, int, int]:
    """logit xgot + δ_loo per test row, δ_loo fitted on its season's rows of unpaired keepers.

    In order: no unpaired row → NaN; a season at δ = 0 by rule (``small`` or
    ``zero_family``, as :func:`rate`) → 0; unpaired rows that fail E2's minimum
    (:func:`too_small` with ``SMALL_LEAGUE``, read when called) → 0, counted; else their
    :func:`league_shift`, NaN when not finite. Returns the logits, the count of NaN seasons and
    the count of E2-small remainders."""
    faced = test.select("gk_id", *SEASON_KEY).unique()
    shifts: dict[tuple[str, str], float] = {}
    loo_small = 0
    for key in test.select(SEASON_KEY).unique().iter_rows():
        season = rows.filter((pl.col(SEASON_KEY[0]) == key[0]) & (pl.col(SEASON_KEY[1]) == key[1]))
        others = season.join(faced, on=["gk_id", *SEASON_KEY], how="anti")
        flagged = any(bool(season[c].any()) for c in ("small", "zero_family") if c in season)
        if others.height == 0:
            d = math.nan
        elif flagged:
            d = 0.0
        elif too_small(others["gk_id"].n_unique(), others.height, SMALL_LEAGUE):
            d, loo_small = 0.0, loo_small + 1
        else:
            d = league_shift(others["xgot"], others["is_goal"])
        shifts[key] = d if math.isfinite(d) else math.nan
    deltas = np.array([shifts[k] for k in test.select(SEASON_KEY).iter_rows()])
    missing = sum(math.isnan(d) for d in shifts.values())
    return metrics.logit(test["xgot"].to_numpy()) + deltas, missing, loo_small


def saw_test(ns: NextSeason, trained: pl.DataFrame) -> int:
    """The kept pairs of ``ns`` whose rated season's xGOT came from a fit that trained on the
    test season (PM E4, a disclosure): (``competition_id``, ``next_season_id``) is a training
    season of (``competition_id``, ``season_id``) in ``trained``
    (``season_out.trained_seasons``).

    Raises
    ------
    ValueError
        A kept pair whose rated season has no row in ``trained`` (count named): fail closed,
        never counted as unseen.

    Examples
    --------
    >>> ns = NextSeason(2, 0, 0, 0, (None, None, None), None, None, None, None, 0, None,
    ...                 kept_pairs=(("g", "c", "1", "2"), ("h", "c", "2", "3")))
    >>> trained = pl.DataFrame({"competition_id": ["c", "c"], "season_id": ["1", "2"],
    ...                         "trained_competition_id": ["c", "c"],
    ...                         "trained_season_id": ["2", "1"]})
    >>> saw_test(ns, trained)
    1
    """
    if not ns.kept_pairs:
        return 0
    names = [*KEEPER_SEASON, "next_season_id"]
    kept = pl.DataFrame(ns.kept_pairs, schema=dict.fromkeys(names, pl.String), orient="row")
    keys = trained.select(pl.col(SEASON_KEY).cast(pl.String)).unique()
    if missing := kept.join(keys, on=SEASON_KEY, how="anti").height:
        raise ValueError(f"saw_test: {missing} kept pair(s) have a rated season not in trained")
    seen = trained.filter(pl.col("trained_competition_id") == pl.col("competition_id")).select(
        pl.col(SEASON_KEY).cast(pl.String),
        next_season_id=pl.col("trained_season_id").cast(pl.String),
    )
    return kept.join(seen.unique(), on=[*SEASON_KEY, "next_season_id"], how="semi").height


# ---- P3.3 reliability ----------------------------------------------------------------------


@dataclass(frozen=True)
class Tier:
    """Split-half reliability at ``tier`` shots faced (PM X5): ``first`` on each keeper's first
    ``tier`` shots, ``all_shots`` on all of them; ``*_sb`` the Spearman-Brown step-up of r and
    its interval. ``keepers`` 0 is an empty tier."""

    tier: int
    keepers: int
    first: SplitHalf
    first_sb: SplitHalf
    all_shots: SplitHalf
    all_shots_sb: SplitHalf


def _stepped(s: SplitHalf) -> SplitHalf:
    sb = metrics.spearman_brown
    return SplitHalf(s.keys, sb(s.r), sb(s.low), sb(s.high))


def reliability(rated: pl.DataFrame) -> tuple[Tier, ...]:
    """One :class:`Tier` per ``TIERS`` value: keepers whose total rated open-play shots (all
    seasons) reach it; shots alternated in ``metrics.SPLIT_ORDER["row"]`` order, value q − goal.

    Examples
    --------
    >>> [t.keepers for t in reliability(rated_open.head(0))]  # doctest: +SKIP
    [0, 0, 0]
    """
    rated = _open_play(rated, "reliability")
    order = ["gk_id", *metrics.SPLIT_ORDER["row"]]
    rows = rated.with_columns(value=pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sort(order)
    total = pl.len().over("gk_id")
    out = []
    for t in TIERS:
        mine = rows.filter(total >= t)
        first = mine.filter(pl.int_range(pl.len()).over("gk_id") < t)
        halves = [metrics.split_half(f, by="gk_id", value="value", min_rows=t, unit="row")
                  for f in (first, mine)]  # fmt: skip
        out.append(Tier(t, mine["gk_id"].n_unique(), halves[0], _stepped(halves[0]), halves[1],
                        _stepped(halves[1])))  # fmt: skip
    return tuple(out)


# ---- PR1 sensitivities (StatsBomb) ---------------------------------------------------------


def stops(f: xgot.XgotFit) -> pl.DataFrame | None:
    """``shot_id`` and ``end_x`` (Float64, where the shot was stopped) of every open-play
    xGOT row of ``f`` (``f.inp.on_dev`` / ``on_held``); None when ``f.provider`` is not in
    ``LINE_PROVIDERS`` (read when called).

    Raises
    ------
    ValueError
        The frames have no ``end_x`` column.

    Examples
    --------
    >>> stops(xgot.fit("whoscored", cache="data/cache")) is None  # doctest: +SKIP
    True
    """
    if f.provider not in LINE_PROVIDERS:
        return None
    frames = (f.inp.on_dev, f.inp.on_held)
    if any("end_x" not in frame.columns for frame in frames):
        raise ValueError(f"stops: the {f.provider} xGOT rows have no end_x column")
    return pl.concat([fr.select("shot_id", pl.col("end_x").cast(pl.Float64)) for fr in frames])


def line_rated(pred: pl.DataFrame, stops: pl.DataFrame, *, coarse: pl.Series) -> pl.DataFrame:
    """The open-play rows of ``pred`` that are a goal or a save stopped at ``end_x >=
    LINE_X``, re-rated by :func:`rate` (δ and E2 smallness fitted on this subset).

    The goal_z sensitivity (PM after MS6): a projected save's ``goal_z`` is where the keeper
    met the ball, so only saves stopped within ``LINE_YD`` of the line keep a z near the line.
    Every goal stays. ``LINE_X`` is a fine-grid value; it holds because coarse-grid matches
    are out of GSAx (PM E1 (a)).

    Parameters
    ----------
    pred : polars.DataFrame
        ``season_out.season_out`` rows (penalties are dropped here).
    stops : polars.DataFrame
        :func:`stops` of the same fit.
    coarse : polars.Series
        ``XgotFit.coarse``: the coarse-grid ``match_id`` values.

    Raises
    ------
    ValueError
        A row of a ``coarse`` match; a rated row without a stop; a save with a null or NaN
        ``end_x`` (counts named); from :func:`rate`.

    Examples
    --------
    >>> pred = pl.DataFrame({"shot_id": ["a", "b", "c"], "gk_id": ["g"] * 3,
    ...                      "xgot": [0.3, 0.3, 0.3], "is_goal": [True, False, False],
    ...                      "match_id": ["m"] * 3, "competition_id": ["c"] * 3,
    ...                      "season_id": ["s"] * 3, "competition_name": ["C"] * 3,
    ...                      "season_name": ["2020"] * 3, "kind": ["open_play"] * 3,
    ...                      "family": ["women"] * 3, "source": ["season_folds"] * 3})
    >>> at = pl.DataFrame({"shot_id": ["a", "b", "c"], "end_x": [105.0, LINE_X, 100.0]})
    >>> line_rated(pred, at, coarse=pl.Series([], dtype=pl.String))["shot_id"].to_list()
    ['a', 'b']
    """
    op = pred.filter(pl.col("kind").cast(pl.String) == "open_play")
    if bad := op.filter(pl.col("match_id").is_in(coarse.cast(pl.String).implode())).height:
        raise ValueError(f"line_rated: {bad} row(s) of coarse-grid matches; LINE_X is fine-grid")
    rows = op.join(stops.select("shot_id", "end_x"), on="shot_id", how="left", validate="m:1",
                   maintain_order="left")  # fmt: skip
    if bad := op.join(stops, on="shot_id", how="anti").height:
        raise ValueError(f"line_rated: {bad} rated row(s) have no stop")
    end = pl.col("end_x")
    if bad := rows.filter(~pl.col("is_goal") & (end.is_null() | end.is_nan())).height:
        raise ValueError(f"line_rated: {bad} save(s) have a null or NaN end_x")
    kept = rows.filter(pl.col("is_goal") | (end >= LINE_X)).drop("end_x")
    return rate(kept, small=SMALL_LEAGUE, zero_families=ZERO_FAMILIES)


def noz_rated(pred: pl.DataFrame) -> pl.DataFrame:
    """The z-free season-out rows (``season_out(f, without=NOZ_FEATURES)``, open play) rated
    by :func:`rate` with ``SMALL_LEAGUE`` and ``ZERO_FAMILIES`` (read when called).

    Examples
    --------
    >>> noz_rated(season_out.season_out(f, without=NOZ_FEATURES))  # doctest: +SKIP
    """
    return rate(pred, small=SMALL_LEAGUE, zero_families=ZERO_FAMILIES)


# ---- fit and checks ------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GsaxFit:
    """Everything :func:`assemble` computed for one provider; cards and charts render from it.

    ``model_no_team`` is the keeper-season model without team-selected rows (None when there
    are none, or only those); ``saw_test`` is (S, P, L): S of the P kept next-season pairs
    whose rated-season xGOT trained on the test season (:func:`saw_test`), L left out;
    ``line`` and ``noz`` are the :func:`keeper_table` of the goal_z sensitivities
    (:func:`line_rated`, :func:`noz_rated`), None where not computed; ``line_seasons`` is
    :func:`season_flags` of the line subset (None with ``line``).
    """

    xgot: xgot.XgotFit
    predictions: pl.DataFrame
    rated: pl.DataFrame
    keepers: Mapping[str, pl.DataFrame]
    leagues: Mapping[str, pl.DataFrame]
    ci: CiCounts
    model: KeeperModel
    career: KeeperModel
    next_season: NextSeason
    next_career: NextSeason
    next_no_team: NextSeason | None
    tiers: tuple[Tier, ...]
    checks: tuple[Check, ...]
    model_no_team: KeeperModel | None
    saw_test: tuple[int, int, int]
    line: pl.DataFrame | None
    noz: pl.DataFrame | None
    line_seasons: pl.DataFrame | None


def season_flags(rated: pl.DataFrame) -> pl.DataFrame:
    """One row per competition-season of open-play :func:`rate` output: ``SEASON_KEY``, its
    ``delta`` and ``at_zero`` (``AT_ZERO``: δ = 0 by rule, small or zero-family), sorted by key.

    Raises
    ------
    ValueError
        ``rated`` holds a non-open-play row.

    Examples
    --------
    >>> r = pl.DataFrame({"competition_id": ["c", "c", "d"], "season_id": ["s"] * 3,
    ...                   "kind": ["open_play"] * 3, "delta": [0.2, 0.2, 0.0],
    ...                   "small": [False, False, True], "zero_family": [False] * 3,
    ...                   "pool_small": [False] * 3})
    >>> season_flags(r).rows()
    [('c', 's', 0.2, False), ('d', 's', 0.0, True)]
    """
    return (
        _open_play(rated, "season_flags")
        .group_by(SEASON_KEY)
        .agg(pl.col("delta").first(), at_zero=AT_ZERO.first())
        .sort(SEASON_KEY)
    )


def variant(rated: pl.DataFrame, name: str) -> pl.DataFrame:
    """The rows of one ``VARIANTS`` entry (a filter: the rating is the same).

    Examples
    --------
    >>> variant(g.rated, WITHOUT)["kind"].unique().to_list()  # doctest: +SKIP
    ['open_play']
    """
    return rated.filter(pl.col("kind").cast(pl.String).is_in(VARIANTS[name]))


def keeper_rows(g: GsaxFit) -> pl.DataFrame:
    """The card and chart rows: "without penalties" keeper-seasons with >= ``TABLE_MIN_SHOTS``
    shots, with ``gsax_shrunk`` (the keeper-season model), ``gsax_with`` (with penalties),
    ``gsax_line`` and ``gsax_noz`` (the goal_z sensitivities; null without a row there) and
    ``zero_family`` (the season flag), sorted by ``gsax`` descending.

    Examples
    --------
    >>> keeper_rows(g).columns[-5:]  # doctest: +SKIP
    ['gsax_shrunk', 'gsax_with', 'gsax_line', 'gsax_noz', 'zero_family']
    """
    with_ = g.keepers[WITH].select(*KEEPER_SEASON, gsax_with=pl.col("gsax"))
    sens = [
        pl.DataFrame(schema={**{c: pl.String for c in KEEPER_SEASON}, name: pl.Float64})
        if t is None else t.select(*KEEPER_SEASON, pl.col("gsax").alias(name))
        for name, t in (("gsax_line", g.line), ("gsax_noz", g.noz))
    ]  # fmt: skip
    flags = variant(g.rated, WITHOUT).group_by(SEASON_KEY).agg(pl.col("zero_family").first())
    out = (
        g.keepers[WITHOUT].filter(pl.col("shots") >= TABLE_MIN_SHOTS)
        .join(g.model.estimates.select(*KEEPER_SEASON, "gsax_shrunk"), on=KEEPER_SEASON,
              how="left")
        .join(with_, on=KEEPER_SEASON, how="left")
    )  # fmt: skip
    for t in sens:
        out = out.join(t, on=KEEPER_SEASON, how="left")
    return out.join(flags, on=SEASON_KEY, how="left").sort(
        pl.col("gsax"), *KEEPER_SEASON, descending=[True, False, False, False]
    )


@dataclass(frozen=True)
class LineSummary:
    """A goal_z sensitivity against the headline over the ``keeper_seasons`` with >=
    ``TABLE_MIN_SHOTS`` headline shots whose competition-season has a δ fitted on the line
    subset: ``saves`` open-play saves rated, ``far`` of them stopped more than ``LINE_YD``
    before the line; the |shift| (sensitivity − headline GSAx) median, 90th percentile and max
    beside the keeper-seasons' mean SD; ``rank_r`` the Spearman of the two GSAx, ``far_share_r``
    that of the shift and the keeper-season's far saves per shot. ``at_zero`` keeper-seasons
    (same shot floor) sit in a season at δ = 0 on the line subset and are left out, with their
    median |shift| ``at_zero_median_abs``. Each float is None with no keeper-season (a Spearman
    also with one, or a constant column)."""

    saves: int
    far: int
    keeper_seasons: int
    median_abs: float | None
    p90_abs: float | None
    max_abs: float | None
    mean_sd: float | None
    rank_r: float | None
    far_share_r: float | None
    at_zero: int
    at_zero_median_abs: float | None


SHIFTS: Final[Mapping[str, str]] = MappingProxyType({"gsax_line": "line", "gsax_noz": "noz"})


def _spearman(frame: pl.DataFrame, a: str, b: str) -> float | None:
    rows = frame.select(a, b).drop_nulls()
    r = rows.select(pl.corr(a, b, method="spearman")).item() if rows.height else None
    return None if r is None or math.isnan(r) else float(r)


def shift_summary(g: GsaxFit, column: str) -> LineSummary | None:
    """:class:`LineSummary` of one sensitivity column of :func:`keeper_rows` (``SHIFTS``);
    None when ``g`` lacks that sensitivity or the line one (the far-save counts and the
    population come from it).

    Both sensitivities are judged on one population: the keeper-seasons whose competition-season
    has a δ fitted on the line subset (``g.line_seasons``, not ``at_zero``; audit A10). The line
    subset keeps every goal and few saves; a fitted δ re-levels that selection, a season at δ = 0
    (team-selected, or E2-small on the subset) cannot, so its keeper-seasons are only counted.

    Raises
    ------
    ValueError
        ``column`` not in ``SHIFTS``.

    Examples
    --------
    >>> shift_summary(g_ws, "gsax_line") is None  # doctest: +SKIP
    True
    """
    if column not in SHIFTS:
        raise ValueError(f"shift_summary: {column!r} is not one of {sorted(SHIFTS)}")
    if getattr(g, SHIFTS[column]) is None or g.line is None or g.line_seasons is None:
        return None
    fitted = g.line_seasons.filter(~pl.col("at_zero")).select(SEASON_KEY)
    saves = pl.col("shots") - pl.col("goals")
    line = g.line.select(*KEEPER_SEASON, line_saves=saves)
    head = g.keepers[WITHOUT].select(*KEEPER_SEASON, "shots", saves=saves)
    far = head.join(line, on=KEEPER_SEASON, how="left").select(
        *KEEPER_SEASON, far=pl.col("saves") - pl.col("line_saves").fill_null(0),
        far_share=(pl.col("saves") - pl.col("line_saves").fill_null(0)) / pl.col("shots"),
    )  # fmt: skip
    every = (
        keeper_rows(g)
        .join(far, on=KEEPER_SEASON, how="left")
        .filter(pl.col(column).is_not_null())
        .with_columns(shift=pl.col(column) - pl.col("gsax"))
    )
    rows = every.join(fitted, on=SEASON_KEY, how="semi", maintain_order="left")
    zero = every.join(fitted, on=SEASON_KEY, how="anti")["shift"].abs().to_numpy()
    d, sd = rows["shift"].abs().to_numpy(), rows["sd"].to_numpy()
    med = p90 = top = mean_sd = None
    if d.size:
        med, p90, top = float(np.median(d)), float(np.quantile(d, 0.9)), float(d.max())
        mean_sd = float(sd.mean())
    return LineSummary(
        int(head["saves"].sum()), int(far["far"].sum()), rows.height, med, p90, top, mean_sd,
        _spearman(rows, "gsax", column), _spearman(rows, "shift", "far_share"),
        zero.size, float(np.median(zero)) if zero.size else None,
    )  # fmt: skip


def line_summary(g: GsaxFit) -> LineSummary | None:
    """``shift_summary(g, "gsax_line")``: GSAx on goals and saves stopped within ``LINE_YD``
    of the line against the headline; None without the line sensitivity.

    Examples
    --------
    >>> line_summary(g_sb).far_share_r < 0  # doctest: +SKIP
    True
    """
    return shift_summary(g, "gsax_line")


def oof_rated(g: GsaxFit) -> pl.DataFrame:
    """``g``'s rows rated as :func:`fit` rates them, from the xGOT card's match out-of-fold xGOT.

    Diagnostic only (that xGOT saw the rated season): beside the season-out rating it shows
    which band or keeper gap comes from the season-out refit rather than from xGOT as such.

    Examples
    --------
    Σ q = goals over the rows rated with a δ (not those at δ = 0 by rule, ``AT_ZERO``):

    >>> r = oof_rated(g).filter(~AT_ZERO)  # doctest: +SKIP
    >>> abs(r["q"].sum() - r["is_goal"].sum()) < 1e-6  # doctest: +SKIP
    True
    """
    oof = g.xgot.predictions.select("shot_id", oof="xgot")
    pred = g.predictions.join(oof, on="shot_id", how="left", validate="1:1", maintain_order="left")
    return rate(pred.with_columns(xgot=pl.col("oof")).drop("oof"), small=SMALL_LEAGUE,
                zero_families=ZERO_FAMILIES)  # fmt: skip


def placement_bands(g: GsaxFit) -> pl.DataFrame:
    """:func:`band_table` of ``g``'s open-play rows, plus ``gap_oof`` / ``sd_oof`` of the same
    rows rated by :func:`oof_rated` (diagnostic).

    Examples
    --------
    >>> placement_bands(g).columns[-2:]  # doctest: +SKIP
    ['gap_oof', 'sd_oof']
    """
    cols = ("shot_id", "goal_y", "goal_z", "on_target", "angle_rad")
    geo = pl.concat([g.xgot.inp.on_dev.select(cols), g.xgot.inp.on_held.select(cols)])
    keys = ["feature", "band", "source"]
    oof = band_table(variant(oof_rated(g), WITHOUT), geo).select(
        *keys, gap_oof=pl.col("gap"), sd_oof=pl.col("sd")
    )
    out = band_table(variant(g.rated, WITHOUT), geo)
    return out.join(oof, on=keys, how="left", maintain_order="left")


def _max(s: pl.Series) -> float:
    """The largest value as a float; 0 for an empty series."""
    return float(s.cast(pl.Float64).max() or 0.0)  # type: ignore[arg-type]


def _tiny(x: float) -> str:
    """A rounding-noise size: below ``PRINT_FLOOR`` it prints the floor, so the card's bytes do
    not depend on the summation order of threads."""
    return f"< {PRINT_FLOOR:g}" if x < PRINT_FLOOR else f"{x:.1e}"


def gain_text(c: GainCI | None) -> str:
    """A next-season gain and its interval as card text; ``n/a`` for None.

    Examples
    --------
    >>> gain_text(GainCI(0.001, -0.0005, 0.002, 2000, 40)), gain_text(None)
    ('+0.00100 [-0.00050, +0.00200]', 'n/a')
    """
    return "n/a" if c is None else f"{c.gain:+.5f} [{c.low:+.5f}, {c.high:+.5f}]"


def checks(g: GsaxFit) -> tuple[Check, ...]:
    """The six checks of the P3 criterion, in the card's order (spec §4.7).

    Rows 1-5 can be missed. Row 1: the rated rows carry ``g.predictions``' season-out xGOT
    (``season_out`` raises on a fold that saw its season) and some of it differs from the match
    out-of-fold xGOT, which saw the season. Row 5 is judged on the keeper-season grouping and
    prints the career grouping, ``(1 | keeper)``, beside it. Row 6 is a reporting check: it is
    met when every ``TIERS`` value is reported, an empty tier included.

    Examples
    --------
    >>> [c.result for c in checks(g)]  # doctest: +SKIP
    ['met', 'met', 'met', 'met', 'met', 'met']
    """
    by_source = g.rated.group_by("source").len().sort("source")
    xg_cols = ("shot_id", "xgot")
    read = (
        g.rated.select(xg_cols)
        .sort("shot_id")
        .equals(g.predictions.select(xg_cols).sort("shot_id"))
    )
    oof = g.predictions.join(g.xgot.predictions.select("shot_id", oof="xgot"), on="shot_id",
                             how="left")  # fmt: skip
    moved = oof.filter(pl.col("oof").is_null() | (pl.col("xgot") != pl.col("oof"))).height
    sums, worst = [], 0.0
    for name in VARIANTS:
        rows, table = variant(g.rated, name), g.keepers[name]
        lg = rows.group_by(SEASON_KEY).agg(
            raw=(pl.col("xgot") - pl.col("is_goal").cast(pl.Float64)).sum(),
            rated=(pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum(),
        )
        kp = table.group_by(SEASON_KEY).agg(pl.col("gsax_raw").sum(), pl.col("gsax").sum())
        j = lg.join(kp, on=SEASON_KEY, how="full", coalesce=True).select(
            (pl.col("raw") - pl.col("gsax_raw")).abs().fill_null(math.inf).alias("a"),
            (pl.col("rated") - pl.col("gsax")).abs().fill_null(math.inf).alias("b"),
        )
        worst = max(worst, _max(j["a"]), _max(j["b"]))
        sums.append(int(table["shots"].sum()) == rows.height)
    groups = rating_groups(g.rated)
    fitted = groups.filter(~AT_ZERO)
    level = _max(fitted["gap"].abs())
    n_fit = {k: fitted.filter(pl.col("kind") == k).height for k in RATED_KINDS}
    # season-level groups: every open-play one, and the kicks of a season at δ = 0
    by_season = groups.filter((pl.col("kind") == "open_play") | pl.col("small")
                              | pl.col("zero_family"))  # fmt: skip
    seasons = by_season.group_by("group").agg(pl.col("small", "zero_family").first(),
                                              gap=pl.col("gap").sum())  # fmt: skip
    team, other = pl.col("zero_family"), pl.col("small") & ~pl.col("zero_family")
    n_team, n_other = seasons.filter(team).height, seasons.filter(other).height
    n_both = seasons.filter(team & pl.col("small")).height
    pools = groups.filter(pl.col("pool_small")).height
    gap_team, gap_other = (float(seasons.filter(e)["gap"].sum()) for e in (team, other))
    tables = list(g.keepers.values())
    sd_ok = all(bool(t["sd"].is_finite().all()) for t in tables)
    ci_ok = all(t.filter(pl.col("ci_low").is_null() != (pl.col("matches") == 1)).height == 0
                for t in tables)  # fmt: skip
    one = int((g.keepers[WITHOUT]["matches"] == 1).sum())
    ns, nc = g.next_season, g.next_career
    five: Result = "n/a" if ns.untestable else "met" if ns.gain and ns.gain.low > 0 else "missed"
    career = nc.untestable or gain_text(nc.gain)
    tiers = "; ".join(f"{t.tier}: {t.keepers} keepers" + (" (empty)" if not t.keepers else "")
                      for t in g.tiers)  # fmt: skip
    return (
        Check("Every rated row's xGOT from fits that never saw its season (season_out raises "
              "otherwise); the rating reads those rows, not the match out-of-fold xGOT",
              ", ".join(f"{s} {n:,}" for s, n in by_source.iter_rows())
              + f"; {moved:,} of {oof.height:,} differ from the match out-of-fold xGOT",
              "met" if read and moved > 0 else "missed"),
        Check(f"Keeper totals sum to the league sum of xGOT − goals (raw and rated), both "
              f"variants, within {SUM_TOL:g}; table shots = rated rows",
              f"max abs. difference {_tiny(worst)}", "met" if worst <= SUM_TOL and all(sums)
              else "missed"),
        Check(f"Rated against the league sum: abs(Σ (q − goal)) <= {SUM_TOL:g} per rating "
              "group (open play: one δ per competition-season; penalties: one δ per season-out "
              "source, PM 2026-10-28 (a)), groups at δ = 0 (E2-small, team-selected or an "
              "E2-small pool) excepted",
              f"max {_tiny(level)}; {n_fit['open_play']:,} open-play and {n_fit['penalty']:,} "
              f"penalty groups rated; {n_team:,} team-selected competition-seasons ({n_both:,} "
              f"also small), {n_other:,} other small, {pools:,} penalty pool(s) at δ = 0 by E2; "
              f"Σ (q − goal) at δ = 0: team-selected {gap_team:+.1f}, other small "
              f"{gap_other:+.1f}", "met" if level <= SUM_TOL else "missed"),
        Check(f"Analytic SD and {CI_LEVEL:.0%} match-bootstrap CI per keeper-season",
              f"SD finite; CI null on the {one:,} one-match keeper-seasons", "met"
              if sd_ok and ci_ok else "missed"),
        Check("Shrunken estimates predict the next season better than raw GSAx (log-loss "
              f"gain, {metrics.GAIN_LEVEL:.0%} match-bootstrap interval above 0; judged on the "
              "keeper-season grouping, the career grouping (1 | keeper) beside it)",
              (ns.untestable or f"keeper-season {gain_text(ns.gain)} on {ns.test_shots:,} shots, "
               f"{ns.pairs} pairs") + f"; career {career}", five),
        Check(f"Split-half reliability at {' / '.join(map(str, TIERS))} shots faced, empty tiers "
              "reported (a reporting check: met when every tier is reported)", tiers,
              "met" if [t.tier for t in g.tiers] == list(TIERS) else "missed"),
    )  # fmt: skip


def assemble(
    f: xgot.XgotFit,
    pred: pl.DataFrame,
    *,
    stops: pl.DataFrame | None,
    trained: pl.DataFrame,
    noz: pl.DataFrame | None = None,
) -> GsaxFit:
    """GSAx tables, keeper models, next-season tests, reliability and checks from season-out
    rows: pure arithmetic on its arguments (no fit), so a chart or a test can build a
    :class:`GsaxFit` from frames.

    ``pred`` is ``season_out.season_out(f)``, rated with ``SMALL_LEAGUE`` and
    ``ZERO_FAMILIES`` (read when called); ``stops`` (:func:`stops`, None to skip) feeds the
    line sensitivity, ``trained`` (``season_out.trained_seasons``) the E4 count and ``noz``
    (``season_out(f, without=NOZ_FEATURES)``, None to skip) the z-free sensitivity.

    Raises
    ------
    ValueError
        From :func:`rate`, the tables, the models, :func:`saw_test` and :func:`line_rated`.

    Examples
    --------
    >>> g = assemble(f, season_out.season_out(f), stops=None,
    ...              trained=season_out.trained_seasons(f))  # doctest: +SKIP
    """
    rated = rate(pred, small=SMALL_LEAGUE, zero_families=ZERO_FAMILIES)
    keepers = {v: keeper_table(variant(rated, v)) for v in VARIANTS}
    leagues = {v: league_table(variant(rated, v)) for v in VARIANTS}
    open_ = variant(rated, WITHOUT)
    no_team = open_.filter(pl.col("family") != TEAM_SELECTED)
    partial = 0 < no_team.height < open_.height
    ns = next_season(open_, by=KEEPER_SEASON)
    line = None if stops is None else line_rated(pred, stops, coarse=f.coarse)
    g = GsaxFit(
        f, pred, rated, MappingProxyType(keepers), MappingProxyType(leagues),
        ci_counts(open_, seed=metrics.SEED), fit_keeper_model(open_, by=KEEPER_SEASON),
        fit_keeper_model(open_, by=CAREER), ns, next_season(open_, by=CAREER),
        None if no_team.height == open_.height else next_season(no_team, by=KEEPER_SEASON),
        reliability(open_), (),
        model_no_team=fit_keeper_model(no_team, by=KEEPER_SEASON) if partial else None,
        saw_test=(saw_test(ns, trained), ns.pairs, ns.left_out),
        line=None if line is None else keeper_table(line),
        noz=None if noz is None else keeper_table(noz_rated(noz)),
        line_seasons=None if line is None else season_flags(line),
    )  # fmt: skip
    return dataclasses.replace(g, checks=checks(g))


def fit(f: xgot.XgotFit) -> GsaxFit:
    """GSAx tables, keeper models, next-season tests and reliability of one provider.

    ``season_out.season_out(f)`` is the fit (through ``f``'s cache), then, for a provider in
    ``NOZ_PROVIDERS``, one serial z-free refit (``without=NOZ_FEATURES``, PM E2 (c)); the rest
    is :func:`assemble`. Every constant is read when called.

    Examples
    --------
    >>> g = fit(xgot.fit("statsbomb", cache="data/cache"))  # doctest: +SKIP
    >>> [c.result for c in g.checks][:4]  # doctest: +SKIP
    ['met', 'met', 'met', 'met']
    """
    pred = season_out.season_out(f)
    noz = season_out.season_out(f, without=NOZ_FEATURES) if f.provider in NOZ_PROVIDERS else None
    return assemble(f, pred, stops=stops(f), trained=season_out.trained_seasons(f), noz=noz)
