"""MS6 goalkeepers: goals saved above expected (P3.1), a keeper model (P3.2), stability (P3.3).

**Rating (P3.1).** A keeper's goals saved above expected (GSAx) is Σ xGOT − goals over the
on-target shots he faced. Every row's xGOT comes from fits that never saw its season
(``xgot.season_out``), and the rating is **against the season's league sum**: one logit shift δ
per competition-season and kind (:func:`league_shift`) makes Σ q = goals, ``q = expit(logit
xGOT + δ)``, so the xGOT level gap of a season is not charged to its keepers and the keepers of
a season sum to 0. A logit shift, not a ratio, keeps every q below 1 on near-certain shots.
Penalties are rated apart (their own δ), so GSAx with penalties minus GSAx without is exactly
a keeper's penalty GSAx. GSAx is a record of shots faced, rated against the league; positive
means fewer goals conceded than a league-average keeper facing the same shots. The SD is the
null SD √Σ q(1 − q) of a league-average keeper; the interval is a match bootstrap with the
rating held fixed.

**Keeper model (P3.2).** ``goal ~ offset(logit q) + (1 | group)``, no fixed intercept (the
rated offset fixes the level), fitted by a Laplace approximation (:func:`fit_keeper_model`):
τ̂ maximises the approximate marginal likelihood, and each group's effect is its posterior
mode, shrunk towards 0 as its shots get fewer. It is tested on the next season's on-target
shots against the raw (unpooled) estimate and against xGOT alone (:func:`next_season`).

**Stability (P3.3).** Split-half reliability of q − goal per shot, alternating shots within a
keeper in time order (PM X5), for the keepers whose total shots faced reach each tier
(:func:`reliability`); a tier no keeper reaches is reported empty.

Pure: nothing here fits an xGOT, reads the cache, writes or prints; :func:`fit` calls
``xgot.season_out`` for every fit, so this module is hash-exempt (``cache.ORCHESTRATION_ONLY``).
The model, next-season, reliability and CI-count functions take the "without penalties" rows.
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
from football_core import goalmouth
from scipy import optimize
from scipy.special import expit

from football_shot_models import data, features, metrics, xgot
from football_shot_models.metrics import Check, GainCI, SplitHalf

Result = Literal["met", "missed", "n/a"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

SEASON_KEY: Final = xgot.SEASON_KEY
RATED_KINDS: Final = ("open_play", "penalty")
WITHOUT: Final = "without penalties"
VARIANTS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {WITHOUT: ("open_play",), "with penalties": RATED_KINDS}
)
KEEPER_SEASON: Final = ("gk_id", *SEASON_KEY)
CAREER: Final = ("gk_id",)
CI_LEVEL: Final[float] = 0.90
SUM_TOL: Final[float] = 1e-9  # goals
PRINT_FLOOR: Final[float] = 1e-12  # sums below it print as "< 1e-12": thread-order noise
TIERS: Final[tuple[int, ...]] = (100, 250, 500)
MIN_TIER_KEEPERS: Final[int] = 30  # Fisher-z 95 % half-width about 0.37 at 30 keepers
TABLE_MIN_SHOTS: Final[int] = 100
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


@dataclass(frozen=True)
class SmallLeague:
    """A competition-season too small for a league shift: at most ``max_keepers`` keepers or
    fewer than ``min_shots`` open-play rated shots."""

    max_keepers: int
    min_shots: int


# PM E2 (b), MS6: a league offset from two keepers absorbs their own performance
SMALL_LEAGUE: Final[SmallLeague | None] = SmallLeague(2, 50)


# ---- P3.1 rating ---------------------------------------------------------------------------


def league_shift(p: npt.ArrayLike, goals: npt.ArrayLike) -> float:
    """The logit shift δ with ``Σ expit(logit p + δ) = Σ goals``.

    ``brentq`` on [−``BRACKET``, ``BRACKET``] then Newton steps to machine precision; −inf
    with no goal, +inf when every row is a goal (no finite shift).

    Raises
    ------
    ValueError
        An empty input; shapes that differ; a ``p`` NaN or outside [0, 1].

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


def _validate(frame: pl.DataFrame, what: str) -> None:
    """The rows rule of every table: required columns, no null, xgot in [0, 1], a rated kind."""
    _require(frame, _REQUIRED, what)
    if nulls := {c: n for c in _REQUIRED if (n := frame[c].null_count())}:
        raise ValueError(f"{what}: null values {nulls}")
    x = frame["xgot"].cast(pl.Float64)
    if bad := int((x.is_nan() | (x < 0) | (x > 1)).sum()):
        raise ValueError(f"{what}: {bad} xgot value(s) are NaN or outside [0, 1]")
    if bad := frame.filter(~pl.col("kind").cast(pl.String).is_in(RATED_KINDS)).height:
        raise ValueError(f"{what}: {bad} row(s) of a kind outside {RATED_KINDS}")


def rate(pred: pl.DataFrame, *, small: SmallLeague | None) -> pl.DataFrame:
    """Rate every row against its competition-season and kind: ``delta``, ``q``, ``small``.

    One δ per (``SEASON_KEY``, ``kind``) (:func:`league_shift` of xGOT against goals), so each
    (season, kind) sums to its league; ``q = expit(logit xgot + delta)``, and ``q = goal`` when
    δ is infinite. With ``small``, a competition-season whose open-play rows have at most
    ``small.max_keepers`` keepers or fewer than ``small.min_shots`` rows is rated at δ = 0 in
    both kinds and flagged ``small``.

    Parameters
    ----------
    pred : polars.DataFrame
        ``xgot.season_out`` rows (``gk_id``, ``xgot``, ``is_goal``, ``match_id``,
        ``SEASON_KEY``, ``competition_name``, ``season_name``, ``kind``).
    small : SmallLeague or None
        The small-league rule; None rates every competition-season.

    Returns
    -------
    polars.DataFrame
        ``pred`` in its order plus ``delta`` (Float64, ±inf allowed), ``q`` (Float64) and
        ``small`` (Boolean); empty in, empty out.

    Raises
    ------
    ValueError
        A missing column; a null in a required column (count named); an ``xgot`` NaN or
        outside [0, 1]; a ``kind`` outside ``RATED_KINDS``.

    Examples
    --------
    >>> pred = pl.DataFrame({"gk_id": ["g1", "g2", "g1"], "xgot": [0.2, 0.2, 0.5],
    ...                      "is_goal": [True, False, True], "match_id": ["m1", "m1", "m2"],
    ...                      "competition_id": ["c"] * 3, "season_id": ["s"] * 3,
    ...                      "competition_name": ["C"] * 3, "season_name": ["2020"] * 3,
    ...                      "kind": ["open_play", "open_play", "penalty"]})
    >>> r = rate(pred, small=None)
    >>> [round(v, 4) for v in r["q"]], round(r["delta"][0], 4)
    ([0.5, 0.5, 1.0], 1.3863)
    """
    _validate(pred, "rate")
    flags = (
        pred.filter(pl.col("kind").cast(pl.String) == "open_play")
        .group_by(SEASON_KEY)
        .agg(keepers=pl.col("gk_id").n_unique(), shots=pl.len())
    )
    smalls: set[tuple[str, ...]] = set()
    if small is not None:
        tiny = (pl.col("keepers") <= small.max_keepers) | (pl.col("shots") < small.min_shots)
        smalls = set(flags.filter(tiny).select(SEASON_KEY).iter_rows())
        smalls |= set(pred.select(SEASON_KEY).unique().iter_rows()) - set(
            flags.select(SEASON_KEY).iter_rows()
        )  # no open-play row: no keeper to rate against
    rows = []
    for key, g in pred.group_by(*SEASON_KEY, "kind"):
        season = tuple(key[:2])
        d = 0.0 if season in smalls else league_shift(g["xgot"].to_numpy(), g["is_goal"].to_numpy())
        rows.append((*season, str(key[2]), d, season in smalls))
    deltas = pl.DataFrame(
        rows, schema={**{k: pred.schema[k] for k in SEASON_KEY}, "_kind": pl.String,
                      "delta": pl.Float64, "small": pl.Boolean}, orient="row",
    )  # fmt: skip
    out = pred.with_columns(_kind=pl.col("kind").cast(pl.String)).join(
        deltas, on=[*SEASON_KEY, "_kind"], how="left", maintain_order="left"
    )
    lq = pl.Series(metrics.logit(out["xgot"].cast(pl.Float64).to_numpy())) + out["delta"]
    goal = out["is_goal"].cast(pl.Float64)
    q = pl.Series("q", expit(lq.to_numpy()))
    q = pl.select(pl.when(out["delta"].is_infinite()).then(goal).otherwise(q)).to_series()
    return out.drop("_kind").with_columns(q.alias("q")).select(*pred.columns, "delta", "q", "small")


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

    In a team-selected competition-season one δ mixes the selected side's finishing with its
    opponents', so the selected side's keeper (>= ``SELECTED_SIDE`` shots that season) is
    credited, and the opponents' keepers charged, for the gap between them.

    Returns
    -------
    polars.DataFrame
        ``side`` ("selected side", then "opponents"; a side without rows is absent),
        ``keeper_seasons``, ``shots`` (Int64), ``gsax`` and ``sd`` = √Σ sd² (Float64).

    Examples
    --------
    >>> t = pl.DataFrame({"family": [TEAM_SELECTED] * 3 + ["women"], "shots": [40, 5, 3, 9],
    ...                   "gsax": [2.0, -1.0, -0.5, 4.0], "sd": [3.0, 1.0, 0.0, 1.0]})
    >>> team_selected_split(t).rows()
    [('selected side', 1, 40, 2.0, 3.0), ('opponents', 2, 8, -1.5, 1.0)]
    """
    side = pl.when(pl.col("shots") >= SELECTED_SIDE).then(pl.lit("selected side"))
    return (
        table.filter(pl.col("family").cast(pl.String) == TEAM_SELECTED)
        .group_by(side.otherwise(pl.lit("opponents")).alias("side"))
        .agg(
            keeper_seasons=pl.len().cast(pl.Int64),
            shots=pl.col("shots").sum().cast(pl.Int64),
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
    card. ``gap`` = Σ q − goals, ``sd`` = √Σ q(1 − q); a band is judged with at least
    ``xgot.MIN_BAND_SHOTS`` shots (read when called) and ``within`` ``xgot.BAND_SDS`` SD.

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
    least, out = xgot.MIN_BAND_SHOTS, place["mouth_out"].to_numpy() > 0
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
                gap, sd = float(q[m].sum()) - goals, math.sqrt(float((q[m] * (1 - q[m])).sum()))
                records.append((feature, band, s, n, goals, gap, sd, n >= least,
                                abs(gap) <= xgot.BAND_SDS * sd))  # fmt: skip
    schema = {"feature": pl.String, "band": pl.String, "source": pl.String, "shots": pl.Int64,
              "goals": pl.Int64, "gap": pl.Float64, "sd": pl.Float64, "judged": pl.Boolean,
              "within": pl.Boolean}  # fmt: skip
    return pl.DataFrame(records, schema=schema, orient="row")


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
        0 … G − 1 (G = max code + 1); no convergence in ``NEWTON_MAX`` steps.

    Examples
    --------
    >>> u, h = keeper_modes(np.zeros(4), [1, 1, 0, 0], np.array([0, 0, 1, 1]), tau=1.0)
    >>> [round(float(v), 4) for v in u]
    [0.6748, -0.6748]
    """
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
        Any other value (a trailing newline included), named.

    Examples
    --------
    >>> season_start("2015/2016"), season_start("2018")
    (2015, 2018)
    """
    if not isinstance(name, str) or re.fullmatch(r"\d{4}(/\d{4})?", name) is None:
        raise ValueError(f"season_start: {name!r} is not 'YYYY' or 'YYYY/YYYY'")
    return int(name[:4])


@dataclass(frozen=True)
class NextSeason:
    """The keeper model against raw GSAx and xGOT alone on the next season's shots.

    ``log_loss`` is (none, raw, shrunk); ``gain`` is shrunk over raw and ``gain_none`` shrunk
    over xGOT alone (``metrics.GAIN_LEVEL`` match-bootstrap intervals); the ``_loo`` pair uses
    the test season's δ fitted without the paired keepers' rows (None, with ``loo_missing``
    test seasons counted, when one has no unpaired row or a non-finite δ). ``untestable``
    names why there is no test, with the counts.
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
    seasons = rows.select(*KEEPER_SEASON, "start").unique()
    pairs = seasons.join(
        seasons.select("gk_id", "competition_id", next_season_id=pl.col("season_id"),
                       start=pl.col("start") - 1),
        on=["gk_id", "competition_id", "start"],
    ).sort(*KEEPER_SEASON)  # fmt: skip
    test_parts, kept, left_out = [], 0, 0
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
            kept += 1
            hat = model.filter(pl.all_horizontal(pl.col(c) == v for c, v in key.items()))["u"]
            test_parts.append(rows.filter(
                (pl.col("gk_id") == pair["gk_id"]) & (pl.col("competition_id") ==
                pair["competition_id"]) & (pl.col("season_id") == pair["next_season_id"])
            ).with_columns(u_raw=pl.lit(u_raw), u_hat=pl.lit(float(hat[0]))))  # fmt: skip
    empty = NextSeason(kept, left_out, 0, 0, (None, None, None), None, None, None, None, 0, None)
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
    loo, missing = _loo_logits(rows, test)
    gains_loo: tuple[GainCI | None, GainCI | None] = (None, None)
    if loo is not None:
        pp = (expit(loo), expit(loo + ur), expit(loo + uh))
        gains_loo = (metrics.bootstrap_gain(y, pp[1], pp[2], ids),
                     metrics.bootstrap_gain(y, pp[0], pp[2], ids))  # fmt: skip
    return NextSeason(
        kept, left_out, test.height, matches,
        (_log_loss(y, p[0]), _log_loss(y, p[1]), _log_loss(y, p[2])),
        metrics.bootstrap_gain(y, p[1], p[2], ids), metrics.bootstrap_gain(y, p[0], p[2], ids),
        *gains_loo, missing, None,
    )  # fmt: skip


def _loo_logits(rows: pl.DataFrame, test: pl.DataFrame) -> tuple[FloatArray | None, int]:
    """logit xgot + δ_loo per test row, δ_loo fitted on its season's rows of unpaired keepers;
    None with the count of test seasons that have no unpaired row or a non-finite δ."""
    faced = test.select("gk_id", *SEASON_KEY).unique()
    shifts: dict[tuple[str, str], float] = {}
    missing = 0
    for key in test.select(SEASON_KEY).unique().iter_rows():
        season = rows.filter((pl.col(SEASON_KEY[0]) == key[0]) & (pl.col(SEASON_KEY[1]) == key[1]))
        others = season.join(faced, on=["gk_id", *SEASON_KEY], how="anti")
        d = league_shift(others["xgot"], others["is_goal"]) if others.height else math.nan
        missing += not math.isfinite(d)
        shifts[key] = d
    if missing:
        return None, missing
    deltas = np.array([shifts[k] for k in test.select(SEASON_KEY).iter_rows()])
    return metrics.logit(test["xgot"].to_numpy()) + deltas, 0


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


# ---- fit and checks ------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GsaxFit:
    """Everything :func:`fit` computed for one provider; cards and charts render from it."""

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
    shots, with ``gsax_shrunk`` (the keeper-season model) and ``gsax_with`` (with penalties),
    sorted by ``gsax`` descending.

    Examples
    --------
    >>> keeper_rows(g).columns[-2:]  # doctest: +SKIP
    ['gsax_shrunk', 'gsax_with']
    """
    with_ = g.keepers["with penalties"].select(*KEEPER_SEASON, gsax_with=pl.col("gsax"))
    return (
        g.keepers[WITHOUT].filter(pl.col("shots") >= TABLE_MIN_SHOTS)
        .join(g.model.estimates.select(*KEEPER_SEASON, "gsax_shrunk"), on=KEEPER_SEASON,
              how="left")
        .join(with_, on=KEEPER_SEASON, how="left")
        .sort(pl.col("gsax"), *KEEPER_SEASON, descending=[True, False, False, False])
    )  # fmt: skip


def oof_rated(g: GsaxFit) -> pl.DataFrame:
    """``g``'s rows rated as :func:`fit` rates them, from the xGOT card's match out-of-fold xGOT.

    Diagnostic only (that xGOT saw the rated season): beside the season-out rating it shows
    which band or keeper gap comes from the season-out refit rather than from xGOT as such.

    Examples
    --------
    >>> oof_rated(g)["q"].sum() - oof_rated(g)["is_goal"].sum()  # doctest: +SKIP
    0.0
    """
    oof = g.xgot.predictions.select("shot_id", oof="xgot")
    pred = g.predictions.join(oof, on="shot_id", how="left", validate="1:1", maintain_order="left")
    return rate(pred.with_columns(xgot=pl.col("oof")).drop("oof"), small=SMALL_LEAGUE)


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


def _gain(c: GainCI | None) -> str:
    return "n/a" if c is None else f"{c.gain:+.5f} [{c.low:+.5f}, {c.high:+.5f}]"


def checks(g: GsaxFit) -> tuple[Check, ...]:
    """The six checks of the P3 criterion, in the card's order (spec §4.7).

    Examples
    --------
    >>> [c.result for c in checks(g)]  # doctest: +SKIP
    ['met', 'met', 'met', 'met', 'met', 'met']
    """
    by_source = g.rated.group_by("source").len().sort("source")
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
    groups = g.rated.group_by(*SEASON_KEY, "kind").agg(
        s=(pl.col("q") - pl.col("is_goal").cast(pl.Float64)).sum(), small=pl.col("small").any()
    )
    level = _max(groups.filter(~pl.col("small"))["s"].abs())
    n_small = groups.filter(pl.col("small")).select(SEASON_KEY).n_unique()
    tables = list(g.keepers.values())
    sd_ok = all(bool(t["sd"].is_finite().all()) for t in tables)
    ci_ok = all(t.filter(pl.col("ci_low").is_null() != (pl.col("matches") == 1)).height == 0
                for t in tables)  # fmt: skip
    one = int((g.keepers[WITHOUT]["matches"] == 1).sum())
    ns = g.next_season
    five: Result = "n/a" if ns.untestable else "met" if ns.gain and ns.gain.low > 0 else "missed"
    tiers = "; ".join(f"{t.tier}: {t.keepers} keepers" + (" (empty)" if not t.keepers else "")
                      for t in g.tiers)  # fmt: skip
    return (
        Check("Every rated row's xGOT from fits that never saw its season (season_out raises "
              "otherwise)", ", ".join(f"{s} {n:,}" for s, n in by_source.iter_rows()), "met"),
        Check(f"Keeper totals sum to the league sum of xGOT − goals (raw and rated), both "
              f"variants, within {SUM_TOL:g}; table shots = rated rows",
              f"max abs. difference {_tiny(worst)}", "met" if worst <= SUM_TOL and all(sums)
              else "missed"),
        Check(f"Rated against the season's league sum: abs(Σ (q − goal)) <= {SUM_TOL:g} per "
              "competition-season and kind", f"max {_tiny(level)}; {n_small} small "
              "competition-season(s) at δ = 0", "met" if level <= SUM_TOL else "missed"),
        Check(f"Analytic SD and {CI_LEVEL:.0%} match-bootstrap CI per keeper-season",
              f"SD finite; CI null on the {one:,} one-match keeper-seasons", "met"
              if sd_ok and ci_ok else "missed"),
        Check("Shrunken estimates predict the next season better than raw GSAx (log-loss "
              f"gain, {metrics.GAIN_LEVEL:.0%} match-bootstrap interval above 0)",
              ns.untestable or f"{_gain(ns.gain)} on {ns.test_shots:,} shots, {ns.pairs} pairs",
              five),
        Check(f"Split-half reliability at {' / '.join(map(str, TIERS))} shots faced, empty tiers "
              "reported", tiers, "met" if [t.tier for t in g.tiers] == list(TIERS) else "missed"),
    )  # fmt: skip


def fit(f: xgot.XgotFit) -> GsaxFit:
    """GSAx tables, keeper models, next-season tests and reliability of one provider.

    ``xgot.season_out(f)`` is the only fit (through ``f``'s cache); everything else is
    arithmetic on its rows with ``rate(…, small=SMALL_LEAGUE)``, read when called.

    Examples
    --------
    >>> g = fit(xgot.fit("statsbomb", cache="data/cache"))  # doctest: +SKIP
    >>> [c.result for c in g.checks][:4]  # doctest: +SKIP
    ['met', 'met', 'met', 'met']
    """
    pred = xgot.season_out(f)
    rated = rate(pred, small=SMALL_LEAGUE)
    keepers = {v: keeper_table(variant(rated, v)) for v in VARIANTS}
    leagues = {v: league_table(variant(rated, v)) for v in VARIANTS}
    open_ = variant(rated, WITHOUT)
    no_team = open_.filter(pl.col("family") != TEAM_SELECTED)
    g = GsaxFit(
        f, pred, rated, MappingProxyType(keepers), MappingProxyType(leagues),
        ci_counts(open_, seed=metrics.SEED), fit_keeper_model(open_, by=KEEPER_SEASON),
        fit_keeper_model(open_, by=CAREER), next_season(open_, by=KEEPER_SEASON),
        next_season(open_, by=CAREER),
        None if no_team.height == open_.height else next_season(no_team, by=KEEPER_SEASON),
        reliability(open_), (),
    )  # fmt: skip
    return dataclasses.replace(g, checks=checks(g))
