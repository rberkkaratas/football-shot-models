"""Read fit outputs from the warm cache, refitting no model of record, and write every chart.

:func:`cached_evaluation` reads the ``xg-gbm`` hold-out predictions for the shot charts: the
spec is evaluated through the prediction cache with an estimator that refuses to fit. The
keeper and model-card charts read what the existing public functions return
(:func:`fits`: ``xgot.fit`` and ``gsax.fit``; :func:`candidates`: the xG candidates) inside
:func:`read_only`, which turns every cache miss and every cache write into an error. Two
estimations that guard cannot see are existing behaviour, run on every call as every warm
``make xgot`` / ``make gsax`` / ``make reproduce`` runs them: ``xgot.fit`` fits the shoot-out
penalty xGOT LR once per provider, uncached (drawn by no chart), and ``gsax.fit``
re-estimates the keeper model (τ̂ by ``fit_keeper_model``, plus the ``next_season`` rolling
fits), which the keeper chart draws. Both are deterministic, so no number moves; caching them is
in the maintenance round.

This is the only module that reads ``football_shot_models`` types for the model-card charts:
:func:`calibration_bins`, :func:`xg_view` and :func:`xgot_view` turn evaluations into the
plain :class:`~football_shot_charts.diagnostics.ModelView`. Every file goes through
``rows.write_all``, which refuses a chart not drawn in Inter, a WhoScored chart outside the
private folder, any path git would track, and any chart that fails ``viz.check``.
``python -m football_shot_charts.render`` (``make charts``) writes the 18 StatsBomb charts into
the git-ignored ``outputs/charts``, with ``charts.json`` (each chart's drawn title and subtitle,
:func:`titles`); ``--private`` (``make charts-private``) writes the 10 WhoScored ones into
``data/charts/private``. ``--publish docs/assets/charts`` (``make docs-charts``, :func:`publish`)
copies the StatsBomb charts the docs site shows (:data:`PUBLISH`) byte for byte, after checking
each carries the StatsBomb logo; WhoScored charts are never published.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import json
import os
import re
import shutil
import sys
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import viz
from football_core.io import snapshot, statsbomb
from matplotlib.text import Text

from football_shot_charts import diagnostics, keepers, rows
from football_shot_charts import shots as charts
from football_shot_charts.diagnostics import GainRow, ModelView
from football_shot_models import (
    cache,
    data,
    evaluation,
    gsax,
    gsax_cards,
    metrics,
    models,
    v1,
    xgot,
)
from football_shot_models.evaluation import Comparison, Evaluation, ModelSpec

StrPath = str | os.PathLike[str]

PRIVATE_DIR: Final = rows.PRIVATE_DIR
# this repo's own open-data cache by default (no sibling path); the Makefile passes core's
STATSBOMB_RAW: Final[str] = statsbomb.DEFAULT_CACHE
NOTE_PARTS: Final[Mapping[str, tuple[str, str, bool]]] = {
    # reason: (noun, plural, print the goals)
    "shootout": ("shoot-out kick", "shoot-out kicks", False),
    "penalty": ("penalty", "penalties", True),
    "direct_corner": ("direct corner", "direct corners", True),
    "no_freeze_frame": ("shot without freeze frame", "shots without freeze frame", True),
}


BEST_LR: Final = "best-lr"  # the headline rule's comparison (``models.select_headline``)
XG_REFS: Final[Mapping[str, tuple[tuple[str, str], ...]]] = MappingProxyType({
    "statsbomb": ((BEST_LR, "Best regression"),
                  ("xg-lr-distance-angle-body-part", "Standard baseline (S0)"),
                  (evaluation.BENCHMARK.name, "StatsBomb's xG")),
    "whoscored": ((BEST_LR, "Best regression"), (v1.DISTANCE_ANGLE, "Distance and angle")),
})  # fmt: skip
XGOT_REFS: Final = (("{prefix}lr", "Logistic regression"),)
# on a shared axis their +0.108 / +0.202 gains flatten the +0.0097 headline row (d1): the strip
XGOT_EXTRAS: Final = (("{prefix}xg-only", "xG only"), ("{prefix}base-rate", "base rate"))
BENCHMARK_TITLE: Final = "provider model, freeze-frame inputs"
BENCHMARK_NOTE: Final = (
    "StatsBomb's own model was likely trained on these matches, so it is a reference rather "
    "than an out-of-sample rival"
)
HEADS: Final[Mapping[str, str]] = MappingProxyType(
    {"statsbomb": models.GBM.name, "whoscored": v1.GBM.name}
)
SPLITS: Final[Mapping[str, tuple[str, str, str]]] = MappingProxyType({
    "hold-out": ("Hold-out season", "the hold-out season", "hold-out"),
    "test": ("Test season", "the test season", "test-season"),
})  # fmt: skip
COLD_HINT: Final[Mapping[str, str]] = MappingProxyType({
    "statsbomb": "no warm {name} entry; run make xg-models xgot gsax first",
    "whoscored": "no warm {name} entry; run make xg-v1 xgot gsax first",
})  # fmt: skip
CHART_OUT: Final = "outputs/charts"
PRIVATE_PREFIX: Final = "ws-"
TITLES: Final = "charts.json"  # {file name: {"title", "subtitle"}} of the last StatsBomb run
# the charts the docs site publishes: (file pattern with {theme}, kind); each (pattern, theme)
# must match exactly one chart of the run, so a second team or the keeper chart without
# team-selected seasons is never picked by accident
PUBLISH: Final[tuple[tuple[str, str], ...]] = (
    ("team-*-{theme}-square.png", "Team shot map"),
    ("player-*-{theme}-square.png", "Player shot map"),
    ("gsax-keepers-{theme}.png", "Keeper GSAx interval chart"),
    ("calibration-xg-{theme}.png", "xG calibration chart"),
    ("calibration-xgot-{theme}.png", "xGOT calibration chart"),
    ("comparison-xg-{theme}.png", "xG model comparison chart"),
    ("comparison-xgot-{theme}.png", "xGOT model comparison chart"),
)
_ALT_ESCAPE: Final = re.compile(r"([\\`*_\[\]])")


class CacheMiss(ValueError):
    """The chart asked for a fit: the cache entry is missing or stale (``name``: the spec)."""

    def __init__(self, message: str, name: str = "") -> None:
        super().__init__(message)
        self.name = name


class _NoFit:
    """An estimator that keeps the cache identity (``get_params``) and refuses to fit."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def get_params(self, deep: bool = True) -> Any:
        return self._inner.get_params(deep)

    def _refuse(self, *args: object, **kwargs: object) -> Any:
        raise CacheMiss("a chart never fits")

    fit = fit_design = predict_proba = _refuse


def cached_evaluation(
    xg: data.XgData, pc: cache.PredictionCache, *, spec: ModelSpec = models.GBM
) -> Evaluation:
    """``spec``'s evaluation (out-of-fold, folds, hold-out) read from the warm cache, no fit.

    The split and folds are the ones ``models.run`` uses (``pair_holdout``, ``split_holdout``,
    ``cv_folds``), so the entry's identity matches; the estimator is wrapped so that any fit
    raises.

    Raises
    ------
    ValueError
        No warm entry for ``spec`` (run ``make xg-models`` first); the cache wrote an entry.

    Examples
    --------
    >>> ev = cached_evaluation(data.load_xg_data(), cache.PredictionCache("data/cache",
    ...     data.load_xg_data().manifest.data_digest))  # doctest: +SKIP
    >>> ev.holdout_p.size  # doctest: +SKIP
    9816
    """
    dev, held = data.split_holdout(xg.shots, data.pair_holdout(xg.matches))
    folds = evaluation.cv_folds(dev)
    guarded = dataclasses.replace(spec, make=lambda: _NoFit(spec.make()))
    ff = models.freeze_frames_or_none(xg.freeze_frames)
    try:
        ev = evaluation.evaluate(guarded, dev, held, folds, freeze_frames=ff, cache=pc)
    except CacheMiss:
        raise ValueError(
            f"the prediction cache {pc.dir} has no warm {spec.name} entry for this data and "
            "code; run make xg-models first"
        ) from None
    if pc.written:
        raise ValueError(f"the prediction cache wrote {pc.written}; a chart must only read it")
    return ev


def holdout_xg(
    xg: data.XgData, pc: cache.PredictionCache, *, spec: ModelSpec = models.GBM
) -> pl.DataFrame:
    """The hold-out's kept shots with ``xg`` = ``spec``'s hold-out predictions (no fit).

    Examples
    --------
    >>> held = holdout_xg(xg, pc)  # doctest: +SKIP
    >>> held.height  # doctest: +SKIP
    9816
    """
    ev = cached_evaluation(xg, pc, spec=spec)
    held = data.split_holdout(xg.shots, data.pair_holdout(xg.matches))[1]
    return held.with_columns(pl.Series("xg", ev.holdout_p, dtype=pl.Float64))


def _n(count: int, noun: str, plural: str) -> str:
    return f"{count} {noun if count == 1 else plural}"


def excluded_note(
    shots_table: pl.DataFrame,
    exclusions: pl.DataFrame,
    *,
    team: str,
    player: str | None = None,
    holdout: tuple[str, str] = data.HOLDOUT,
) -> tuple[str, ...]:
    """Footer note parts for the hold-out rows the charts leave out, in ``EXCLUSION_REASONS``
    order: penalties, shoot-out kicks, direct corners and shots without a freeze frame of the
    team (and player), and on team charts the opponents' own goals credited to the team
    (an own-goal row carries the conceding side's ``team_id``), so the totals reconcile.

    Examples
    --------
    >>> shots = pl.DataFrame({"shot_id": ["sb:1", "sb:2"], "team_id": ["sb:22", "sb:9"],
    ...     "opponent_team_id": ["sb:9", "sb:22"], "player_id": ["sb:5", "sb:6"]})
    >>> excl = pl.DataFrame({"shot_id": ["sb:1", "sb:2"], "competition_id": ["sb:2"] * 2,
    ...     "season_id": ["sb:27"] * 2, "reason": ["penalty", "own_goal"],
    ...     "is_goal": [True, True]})
    >>> excluded_note(shots, excl, team="sb:22")
    ('1 own goal by opponents', '1 penalty (1 scored)')
    """
    cols = ["shot_id", "team_id", "opponent_team_id", "player_id"]
    rows = exclusions.filter(data.is_holdout(holdout)).join(
        shots_table.select(cols), on="shot_id", how="inner", validate="1:1"
    )
    own = (pl.col("player_id") == player) if player is not None else pl.lit(True)
    parts = []
    for reason in data.EXCLUSION_REASONS:
        r = rows.filter(pl.col("reason") == reason)
        if reason == "own_goal":
            if player is None and (n := r.filter(pl.col("opponent_team_id") == team).height):
                parts.append(_n(n, "own goal by opponents", "own goals by opponents"))
            continue
        r = r.filter((pl.col("team_id") == team) & own)
        if r.height:
            noun, plural, goals = NOTE_PARTS[reason]
            scored = f" ({int(r['is_goal'].sum())} scored)" if goals else ""
            parts.append(_n(r.height, noun, plural) + scored)
    return tuple(parts)


def default_player(held: pl.DataFrame, team: str) -> str:
    """The team's player with the most kept shots (ties: the smallest id as a string); the
    choice never reads an outcome.

    Raises
    ------
    ValueError
        The team has no kept shot.

    Examples
    --------
    >>> default_player(pl.DataFrame({"team_id": ["sb:1"] * 4,
    ...     "player_id": ["sb:3814", "sb:10960", "sb:3814", "sb:10960"]}), "sb:1")
    'sb:10960'
    """
    counts = held.filter(pl.col("team_id") == team).group_by("player_id").len()
    if counts.height == 0:
        raise ValueError(f"team {team!r} has no kept shot in the hold-out")
    return str(counts.sort(["len", "player_id"], descending=[True, False])["player_id"][0])


def model_text(context: str, quantity: str = "xG") -> str:
    """The footer's model credit, in reader words.

    Examples
    --------
    >>> model_text("Premier League 2015/2016")
    'xG: my own model, trained without Premier League 2015/2016'
    """
    return f"{quantity}: my own model, trained without {context}"


def write(c: viz.Canvas, path: StrPath, *, private: bool = False) -> Path:
    """One chart through ``rows.write_all``: written only where it may go, and only when it
    passes every machine check.

    Raises
    ------
    ValueError
        From ``rows.guard_write``; nothing is written.

    Examples
    --------
    >>> import tempfile
    >>> c = viz.canvas("light", "square", title="T", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> write(c, Path(tempfile.gettempdir()) / "c.png")
    Traceback (most recent call last):
    ...
    ValueError: ...: a WhoScored chart is private: pass private=True and a path under data/...
    """
    path = Path(path)
    return rows.write_all({path.name: c}, path.parent, private=private)[0]


def slug(name: str) -> str:
    """A file-name slug: ASCII, lower case, hyphens.

    Examples
    --------
    >>> slug("N'Golo Kanté")
    'n-golo-kante'
    """
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


# --- Reading fits without fitting ---------------------------------------------------------------


@contextlib.contextmanager
def read_only() -> Iterator[None]:
    """While open, a ``PredictionCache`` miss raises :class:`CacheMiss` (before any cached fit)
    and a write raises; restored on exit, an exception or a nested exit included.

    It patches the class for the process, so it is not thread-safe (the CLI and pytest without
    xdist are single-threaded). It cannot see a fit that bypasses the cache: ``xgot.fit`` fits
    the shoot-out penalty xGOT LR once per provider (existing behaviour, deterministic).

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp, read_only():
    ...     cache.PredictionCache(tmp, "d" * 64, "s" * 64).load("xg-x", "id", ())
    Traceback (most recent call last):
    ...
    football_shot_charts.render.CacheMiss: no warm xg-x entry in ...
    """
    load, save = cache.PredictionCache.load, cache.PredictionCache.save

    def guarded(
        self: cache.PredictionCache, name: str, identity: str, folds: cache.Folds
    ) -> cache.Predictions:
        found = load(self, name, identity, folds)
        if found is None:
            raise CacheMiss(f"no warm {name} entry in {self.dir}", name)
        return found

    def refuse(self: cache.PredictionCache, name: str, *args: object, **kwargs: object) -> None:
        raise CacheMiss(f"a chart wrote the cache entry {name}; it must only read it", name)

    cache.PredictionCache.load = guarded  # type: ignore[method-assign]
    cache.PredictionCache.save = refuse  # type: ignore[method-assign]
    try:
        yield
    finally:
        cache.PredictionCache.load = load  # type: ignore[method-assign]
        cache.PredictionCache.save = save  # type: ignore[method-assign]


def _hinted(provider: str, e: CacheMiss) -> CacheMiss:
    return CacheMiss(COLD_HINT[provider].format(name=e.name or "cache"), e.name)


def fits(provider: str, *, cache: StrPath | None) -> tuple[xgot.XgotFit, gsax.GsaxFit]:
    """``xgot.fit`` and ``gsax.fit`` of one provider from the warm cache, inside
    :func:`read_only`: no xG or xGOT prediction is refitted. Not cached, and run on every call
    as every warm ``make gsax`` runs them: ``xgot.fit``'s shoot-out penalty LR and
    ``gsax.fit``'s keeper-model estimation (τ̂ and the ``next_season`` rolling fits), which the
    keeper chart draws; both deterministic.

    Raises
    ------
    ValueError
        ``cache`` is None (before ``xgot.fit``); a cold entry (:class:`CacheMiss` with
        ``COLD_HINT``); an entry written; from the fits.

    Examples
    --------
    >>> fits("statsbomb", cache=None)
    Traceback (most recent call last):
    ...
    ValueError: charts read the warm prediction cache and never fit: pass a cache root
    """
    if cache is None:
        raise ValueError("charts read the warm prediction cache and never fit: pass a cache root")
    try:
        with read_only():
            f = xgot.fit(provider, cache=cache)
            g = gsax.fit(f)
    except CacheMiss as e:
        raise _hinted(provider, e) from None
    if f.cache is None or f.cache.written:
        raise ValueError(f"the prediction cache wrote {[] if f.cache is None else f.cache.written}")
    return f, g


def candidates(f: xgot.XgotFit) -> models.Candidates:
    """The provider's xG candidates (baselines, LR steps, GBM) on ``f.frames``, read from the
    warm cache inside :func:`read_only`: the set the xG card and the xGOT input read.

    Raises
    ------
    ValueError
        ``f`` has no cache; a cold entry (:class:`CacheMiss` with ``COLD_HINT``).

    Examples
    --------
    >>> candidates(fits("statsbomb", cache="data/cache")[0]).gbm.spec.name  # doctest: +SKIP
    'xg-gbm'
    """
    if f.cache is None:
        raise ValueError("charts read the warm prediction cache and never fit: f has no cache")
    dev, held = f.frames
    try:
        with read_only():
            return models.evaluate_candidates(
                *xgot.PROVIDERS[f.provider].candidate_specs(), dev, held,
                evaluation.cv_folds(dev), freeze_frames=f.xg.freeze_frames, cache=f.cache,
            )  # fmt: skip
    except CacheMiss as e:
        raise _hinted(f.provider, e) from None


# --- Model-card views ---------------------------------------------------------------------------


def calibration_bins(
    y: npt.ArrayLike, p: npt.ArrayLike, groups: npt.ArrayLike, *, bins: int = metrics.ECE_BINS
) -> pl.DataFrame:
    """Equal-count bins of ``p`` (``metrics.reliability``'s rule) with a match-bootstrap band of
    each bin's observed rate.

    The band resamples matches with ``metrics.group_weights(groups, N_BOOT, SEED)`` (read when
    called): a resample's rate is its weighted outcomes over its weighted rows in the bin (a
    resample with no row in the bin is dropped); ``low`` / ``high`` are the ``(1 ± GAIN_LEVEL)
    / 2`` percentiles. ``inside`` (``low <= mean_p <= high``) is judged on the unclipped band;
    then ``rate``, ``low`` and ``high`` are clipped to ``[P_FLOOR, 1 − P_FLOOR]`` for the logit
    axis and ``clipped`` marks a rate of 0 or 1.

    Returns
    -------
    polars.DataFrame
        ``diagnostics.BIN_COLUMNS``: ``bin``, ``shots`` (Int64), ``mean_p``, ``rate``, ``low``,
        ``high`` (Float64), ``inside``, ``clipped`` (Boolean).

    Raises
    ------
    ValueError
        Lengths differ; ``y`` not 0 / 1; ``p`` NaN, infinite or outside [0, 1]; a null group;
        a bin whose mean prediction is 0 or 1; a bin with no resample left; the bins differ
        from ``metrics.reliability`` (shots, or mean / rate by more than 1e-12).

    Examples
    --------
    >>> y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    >>> b = calibration_bins(y, np.linspace(0.1, 0.8, 8), np.array(["m1", "m2"] * 4), bins=2)
    >>> b.select("bin", "shots", "rate").rows()
    [(0, 4, 0.25), (1, 4, 0.75)]
    """
    y_a, p_a, g_a = np.asarray(y), np.asarray(p, dtype=np.float64), np.asarray(groups)
    if not (y_a.ndim == 1 and y_a.shape == p_a.shape == g_a.shape):
        raise ValueError(f"y, p and groups differ in shape: {y_a.shape}, {p_a.shape}, {g_a.shape}")
    if not np.all((y_a == 0) | (y_a == 1)):
        raise ValueError("y must be 0 / 1")
    if not np.all(np.isfinite(p_a)) or bool(((p_a < 0) | (p_a > 1)).any()):
        raise ValueError("p must be finite and in [0, 1]")
    if bad := sum(k is None or k != k for k in g_a.tolist()):  # k != k: NaN
        raise ValueError(f"groups has {bad} null or NaN key(s)")
    yf, n = y_a.astype(np.float64), p_a.size
    b = (
        pl.DataFrame({"p": p_a, "y": yf})
        .with_columns(bin=(pl.col("p").rank("ordinal").cast(pl.Int64) - 1) * bins // n)["bin"]
        .to_numpy()
    )  # the rule of metrics.reliability, re-typed; cross-checked below (export: maintenance)
    ref = metrics.reliability(y_a, p_a, bins)
    inv, w = metrics.group_weights(g_a, metrics.N_BOOT, metrics.SEED)
    q = ((1 - metrics.GAIN_LEVEL) / 2, (1 + metrics.GAIN_LEVEL) / 2)
    out = []
    for k, (shots, mean_xg, goal_rate) in enumerate(ref.select("shots", "mean_xg", "goal_rate")
                                                     .iter_rows()):  # fmt: skip
        mask = b == k
        mean_p, rate = float(p_a[mask].mean()), float(yf[mask].mean())
        if (
            int(mask.sum()) != shots
            or abs(mean_p - mean_xg) > 1e-12
            or abs(rate - goal_rate) > 1e-12
        ):
            raise ValueError(f"bin {k} differs from metrics.reliability")
        if mean_p in (0.0, 1.0):
            raise ValueError(f"bin {k} has a mean prediction of {mean_p}: no logit axis")
        m = w.shape[1]
        ws = w @ np.bincount(inv[mask], minlength=m)
        wg = w @ np.bincount(inv[mask], weights=yf[mask], minlength=m)
        keep = ws > 0
        if not keep.any():
            raise ValueError(f"bin {k}: no resample holds a row of it")
        low, high = (float(v) for v in np.quantile(wg[keep] / ws[keep], q))
        clip = (diagnostics.P_FLOOR, 1 - diagnostics.P_FLOOR)
        drawn = (float(np.clip(v, *clip)) for v in (rate, low, high))
        out.append((k, shots, mean_p, *drawn, low <= mean_p <= high, rate in (0.0, 1.0)))
    return pl.DataFrame(out, schema={
        "bin": pl.Int64, "shots": pl.Int64, "mean_p": pl.Float64, "rate": pl.Float64,
        "low": pl.Float64, "high": pl.Float64, "inside": pl.Boolean, "clipped": pl.Boolean,
    }, orient="row")  # fmt: skip


def gain_row(role: str, title: str, c: Comparison) -> GainRow:
    """A ``Comparison`` as a plain row (gain, low, high per split).

    Raises
    ------
    ValueError
        The two intervals come from different numbers of resamples.

    Examples
    --------
    >>> g = metrics.GainCI(0.01, 0.005, 0.015, 2000, 30)
    >>> gain_row("Reference", "LR", Comparison("over LR", g, g)).cv
    (0.01, 0.005, 0.015)
    """
    if c.cv.n_boot != c.holdout.n_boot:
        raise ValueError(f"{c.label}: {c.cv.n_boot} and {c.holdout.n_boot} resamples")
    return GainRow(role, title, (c.cv.gain, c.cv.low, c.cv.high),
                   (c.holdout.gain, c.holdout.low, c.holdout.high))  # fmt: skip


def _n_boot(comparisons: Sequence[Comparison]) -> int:
    counts = {c.cv.n_boot for c in comparisons} | {c.holdout.n_boot for c in comparisons}
    if len(counts) != 1:
        raise ValueError(f"the comparisons come from different resample counts: {sorted(counts)}")
    return counts.pop()


def contexts(dev: pl.DataFrame, held: pl.DataFrame, matches: pl.DataFrame) -> tuple[str, str]:
    """What cross-validation and the split cover: ``"{k} competition-seasons"``, or the one
    competition-season's name.

    Examples
    --------
    >>> m = pl.DataFrame({"competition_id": ["c", "d"], "season_id": ["s", "s"],
    ...                   "competition_name": ["Cup", "League"], "season_name": ["2015"] * 2})
    >>> contexts(m.head(2), m.tail(1), m)
    ('2 competition-seasons', 'League 2015')
    """
    key = ["competition_id", "season_id"]
    cv, split = (frame.select(key).unique() for frame in (dev, held))
    if split.height != 1:
        return f"{cv.height} competition-seasons", f"{split.height} competition-seasons"
    row = split.join(matches.select(*key, "competition_name", "season_name").unique(key), on=key)
    name = f"{row['competition_name'][0]} {row['season_name'][0]}" if row.height else "n/a"
    return f"{cv.height} competition-seasons", name


def _view(
    f: xgot.XgotFit,
    head: Evaluation,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    drawn: Sequence[tuple[GainRow, Comparison]],
    extras: Sequence[tuple[str, Comparison]],
    *,
    quantity: str,
    unit: str,
    notes: tuple[str, ...],
) -> ModelView:
    yd, yh = evaluation.labels(dev), evaluation.labels(held)
    gap = metrics.poisson_binomial(yh, head.holdout_p)  # (Σ p − goals, SD)
    return ModelView(
        quantity=quantity, outcome="goal", unit=unit, split=SPLITS[f.setup.split],
        contexts=contexts(dev, held, f.xg.matches),
        cv_bins=calibration_bins(yd, head.oof, dev["match_id"].to_numpy()),
        held_bins=calibration_bins(yh, head.holdout_p, held["match_id"].to_numpy()),
        cv_n=dev.height, held_n=held.height, cv_slope=head.cv.cal_slope, held_gap=-gap[0],
        held_sd=gap[1], held_level_ok=metrics.within(gap[0], gap[1]),
        band_sds=metrics.BAND_SDS, cv_log_loss=head.cv.log_loss,
        held_log_loss=head.holdout.log_loss,
        rows=tuple(r for r, _ in drawn),
        extras=tuple(diagnostics.Extra(label, c.cv.gain, c.holdout.gain) for label, c in extras),
        level=metrics.GAIN_LEVEL,
        n_boot=_n_boot([*(c for _, c in drawn), *(c for _, c in extras)]), notes=notes,
    )  # fmt: skip


def xg_view(f: xgot.XgotFit, cands: models.Candidates) -> ModelView:
    """The xG model card's view: the headline's calibration and its gain over ``XG_REFS``
    (picked by name; the head and repeats skipped), on the xG card's own frames and folds.

    Raises
    ------
    ValueError
        The headline is not ``HEADS[f.provider]``; a reference missing from ``cands``; from
        ``evaluation.compare`` and :func:`calibration_bins`.

    Examples
    --------
    >>> xg_view(f, candidates(f)).rows[0].role  # doctest: +SKIP
    'Best logistic regression'
    """
    dev, held = f.frames
    head, best_gain = models.select_headline(cands.lrs, cands.gbm, dev, held)
    if head.spec.name != HEADS[f.provider]:
        raise ValueError(f"the xG headline is {head.spec.name}, not {HEADS[f.provider]}")
    by_name = {ev.spec.name: ev for ev in (*cands.lrs, cands.gbm)}
    best_lr = evaluation.best(cands.lrs)
    seen, drawn = {id(head)}, []
    notes: tuple[str, ...] = ()
    for name, role in XG_REFS[f.provider]:
        if name == BEST_LR:
            ref, c, title = best_lr, best_gain, best_lr.spec.title
        elif name == evaluation.BENCHMARK.name:
            ref = evaluation.evaluate_benchmark(dev, held, evaluation.cv_folds(dev))
            c = evaluation.compare(f"over {ref.spec.title}", ref, head, dev, held)
            title, notes = BENCHMARK_TITLE, (BENCHMARK_NOTE,)
        elif name in by_name:
            ref = by_name[name]
            if id(ref) in seen:
                continue
            c = evaluation.compare(f"over {ref.spec.title}", ref, head, dev, held)
            title = ref.spec.title
        else:
            raise ValueError(f"the xG reference {name!r} is not among the candidates")
        if id(ref) in seen:
            continue
        seen.add(id(ref))
        drawn.append((gain_row(role, title, c), c))
    return _view(f, head, dev, held, drawn, (), quantity="xG", unit="shots", notes=notes)


def xgot_view(f: xgot.XgotFit) -> ModelView:
    """The xGOT model card's view: the headline's calibration, its gain over ``XGOT_REFS``
    (drawn) and ``XGOT_EXTRAS`` (on the strip), picked by name from ``f.evaluations``; pure,
    in memory (no cache).

    Raises
    ------
    ValueError
        A reference missing from ``f.evaluations``; no reference left once the head is
        skipped; from ``evaluation.compare``.

    Examples
    --------
    >>> xgot_view(f).rows[0].role  # doctest: +SKIP
    'Logistic regression'
    """
    dev, held = f.inp.on_dev, f.inp.on_held
    by_name = {ev.spec.name: ev for ev in f.evaluations}

    def comparison(name: str) -> tuple[Evaluation, Comparison]:
        key = name.format(prefix=f.setup.prefix)
        if key not in by_name:
            raise ValueError(f"the xGOT reference {key!r} is not among the evaluations")
        ref = by_name[key]
        return ref, evaluation.compare(f"over {ref.spec.title}", ref, f.head, dev, held)

    drawn = []
    for name, role in XGOT_REFS:
        ref, c = comparison(name)
        if ref is not f.head:
            drawn.append((gain_row(role, ref.spec.title, c), c))
    if not drawn:
        raise ValueError(f"no xGOT reference left once the head {f.head.spec.name} is skipped")
    extras = [(label, comparison(name)[1]) for name, label in XGOT_EXTRAS]
    return _view(f, f.head, dev, held, drawn, extras, quantity="xGOT", unit="on-target shots",
                 notes=())  # fmt: skip


def model_charts(
    f: xgot.XgotFit, cands: models.Candidates, *, logo: StrPath | None, prefix: str = ""
) -> dict[str, viz.Canvas]:
    """Calibration and comparison charts of the xG and xGOT headlines in every theme, file
    name → canvas (``{prefix}calibration-{xg|xgot}-{theme}.png``, ``…comparison…``)."""
    out: dict[str, viz.Canvas] = {}
    for key, view in (("xg", xg_view(f, cands)), ("xgot", xgot_view(f))):
        footer = viz.Footer((f.provider,), model=model_text(f.holdout.label, view.quantity))
        size = diagnostics.COMPARISON_SIZES[view.quantity]
        for theme in viz.THEMES:
            out[f"{prefix}calibration-{key}-{theme}.png"] = diagnostics.calibration_chart(
                view, theme=theme, footer=footer, logo=logo)  # fmt: skip
            out[f"{prefix}comparison-{key}-{theme}.png"] = diagnostics.comparison_chart(
                view, theme=theme, footer=footer, logo=logo, size=size)  # fmt: skip
    return out


# --- Publishing to the docs site ---------------------------------------------------------------


def titles(charts_: Mapping[str, viz.Canvas]) -> dict[str, dict[str, str]]:
    """``{file name: {"title": ..., "subtitle": ...}}`` read from each canvas's ``title`` and
    ``subtitle`` texts, line breaks as spaces: the alt text of the published charts.

    Raises
    ------
    ValueError
        A canvas without a title or subtitle text.

    Examples
    --------
    >>> c = viz.canvas("light", "square", title="Saves\\nabove", subtitle="S",
    ...                footer=viz.Footer(("whoscored",)))
    >>> titles({"c.png": c})
    {'c.png': {'title': 'Saves above', 'subtitle': 'S'}}
    """
    out: dict[str, dict[str, str]] = {}
    for name, c in charts_.items():
        texts = {t.get_gid(): t.get_text() for t in c.fig.findobj(Text)}
        if not ({"title", "subtitle"} <= texts.keys()):
            raise ValueError(f"{name}: the canvas has no title or subtitle text")
        out[name] = {k: str(texts[k]).replace("\n", " ") for k in ("title", "subtitle")}
    return out


def _dumps(entries: Mapping[str, Mapping[str, str]]) -> str:
    return json.dumps(entries, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def _published(name: str) -> tuple[str, str]:
    for pattern, kind_ in PUBLISH:
        for theme in viz.THEMES:
            if fnmatch.fnmatchcase(name, pattern.format(theme=theme)):
                return kind_, theme
    raise ValueError(f"{name} is not a published chart ({', '.join(p for p, _ in PUBLISH)})")


def kind(name: str) -> str:
    """The kind of a published chart file, from :data:`PUBLISH`.

    Raises
    ------
    ValueError
        ``name`` matches no :data:`PUBLISH` pattern.

    Examples
    --------
    >>> kind("gsax-keepers-dark.png")
    'Keeper GSAx interval chart'
    """
    return _published(name)[0]


def alt_text(name: str, entry: Mapping[str, str]) -> str:
    """The alt text of a published chart: its kind, then the drawn title and subtitle.

    Examples
    --------
    >>> alt_text("calibration-xg-light.png", {"title": "T", "subtitle": "S"})
    'xG calibration chart: T (S)'
    """
    return f"{kind(name)}: {entry['title']} ({entry['subtitle']})"


def image_markdown(name: str, entry: Mapping[str, str], href: str) -> str:
    """The Markdown of one published chart, shown in its own colour scheme only and linking to
    ``href``; Markdown metacharacters in the alt text are backslash-escaped so that the built
    page's alt attribute equals :func:`alt_text`.

    Examples
    --------
    >>> md = image_markdown("gsax-keepers-light.png", {"title": "A [b", "subtitle": "c_d"}, "#k")
    >>> print(md.replace("](", "]\\n("))
    [![Keeper GSAx interval chart: A \\[b (c\\_d)]
    (assets/charts/gsax-keepers-light.png#only-light){ loading=lazy }]
    (#k)
    """
    theme = _published(name)[1]
    alt = _ALT_ESCAPE.sub(r"\\\1", alt_text(name, entry))
    return f"[![{alt}](assets/charts/{name}#only-{theme}){{ loading=lazy }}]({href})"


def publish(src: StrPath, dest: StrPath) -> tuple[Path, ...]:
    """Copy the published charts of the last StatsBomb run (``src/charts.json``) into ``dest``,
    byte for byte, with their ``charts.json``; print one :func:`image_markdown` line per file.

    Raises
    ------
    ValueError
        No ``src/charts.json`` (run ``make charts`` first); a WhoScored (``ws-``) name in it; a
        (pattern, theme) of :data:`PUBLISH` matching no chart or several; a selected file
        missing; a selected PNG without the StatsBomb logo (``rows.logo_pixels`` under
        ``rows.LOGO_MIN_PX``); a PNG in ``dest`` outside the selection. Nothing is copied.

    Examples
    --------
    >>> publish("no-such-folder", "docs/assets/charts")
    Traceback (most recent call last):
    ...
    ValueError: no no-such-folder/charts.json; run make charts first
    """
    src, dest = Path(src), Path(dest)
    if not (src / TITLES).is_file():
        raise ValueError(f"no {src / TITLES}; run make charts first")
    entries: dict[str, dict[str, str]] = json.loads((src / TITLES).read_text(encoding="utf-8"))
    if private := sorted(n for n in entries if n.startswith(PRIVATE_PREFIX)):
        raise ValueError(f"{src / TITLES} lists WhoScored charts {private}; never published")
    selected: list[str] = []
    for pattern, _ in PUBLISH:
        for theme in viz.THEMES:
            found = fnmatch.filter(sorted(entries), pattern.format(theme=theme))
            if len(found) != 1:
                raise ValueError(f"{pattern.format(theme=theme)} matches {len(found)} charts of "
                                 f"{src / TITLES}, not exactly one: {found}")  # fmt: skip
            selected += found
    for name in selected:
        if not (src / name).is_file():
            raise ValueError(f"{src / name} is listed in {TITLES} but missing; run make charts")
        if (n := rows.logo_pixels(src / name)) < rows.LOGO_MIN_PX:
            raise ValueError(f"{src / name}: {n} StatsBomb-logo pixels (< {rows.LOGO_MIN_PX})")
    if extra := sorted(p.name for p in dest.glob("*.png") if p.name not in selected):
        raise ValueError(f"{dest} holds charts outside the selection: {extra}; remove them first")
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for name in selected:
        shutil.copyfile(src / name, dest / name)
        out.append(dest / name)
        print(image_markdown(name, entries[name], href=f"assets/charts/{name}"))
    (dest / TITLES).write_text(_dumps({n: entries[n] for n in selected}), encoding="utf-8")
    return tuple(out)


# --- Command line -------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point: the 18 StatsBomb charts (``--team`` and ``--team-name``
    needed), or with ``--private`` the 10 WhoScored ones; exit 1 on an error, with no traceback.

    Examples
    --------
    >>> main(["--team", "sb:22", "--team-name", "T", "--manifest", "missing.json"])
    1
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_charts.render",
        description="Write the shot-model charts: shot charts, keeper chart, model-card visuals; "
        "or copy the published ones into the docs site (--publish).",
    )
    parser.add_argument("--team", help="team id, e.g. sb:22 (StatsBomb charts)")
    parser.add_argument("--team-name", help="the team's display name")
    parser.add_argument("--player", help="player id (default: the most kept shots)")
    parser.add_argument("--statsbomb-raw", default=STATSBOMB_RAW, help="StatsBomb open-data cache")
    parser.add_argument("--out", help=f"git-ignored output folder (default {CHART_OUT}; "
                        f"{PRIVATE_DIR} with --private)")  # fmt: skip
    parser.add_argument("--cache", default="data/cache", help="prediction cache root")
    parser.add_argument("--manifest", default=data.DEFAULT_MANIFEST, help="StatsBomb manifest")
    parser.add_argument("--private", action="store_true",
                        help=f"WhoScored charts only, under {PRIVATE_DIR}")  # fmt: skip
    parser.add_argument("--drop-team-selected", action="store_true",
                        help="the keeper chart without team-selected seasons")  # fmt: skip
    parser.add_argument("--publish", metavar="DEST",
                        help="copy the published StatsBomb charts of --out into DEST")  # fmt: skip
    args = parser.parse_args(argv)
    try:
        if args.publish:
            if args.private:
                raise ValueError("--publish copies StatsBomb charts only; drop --private")
            publish(args.out or CHART_OUT, args.publish)
            return 0
        return _run_private(args) if args.private else _run(args)
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"charts: error: {e}", file=sys.stderr)
        return 1


def write_charts(charts_: Mapping[str, viz.Canvas], out: Path, *, private: bool) -> None:
    """Write a run's charts with ``rows.write_all`` (nothing is written when one chart fails) and
    print one ``chart:`` line per file; a StatsBomb run (not ``private``) also writes
    ``charts.json`` (:func:`titles`), checked git-ignored before anything is written.

    Raises
    ------
    ValueError
        A chart fails ``rows.write_all``'s checks, or git would track ``charts.json``.

    Examples
    --------
    >>> write_charts(model_charts(f, cands, logo=logo), CHART_OUT, private=False)  # doctest: +SKIP
    chart: outputs/charts/calibration-xg-dark.png
    """
    entries = None if private else titles(charts_)
    if entries is not None:
        snapshot.check_ignored(Path(os.path.abspath(out / TITLES)), what="chart titles")
    for path in rows.write_all(charts_, out, private=private):
        print(f"chart: {path}")
    if entries is not None:
        (out / TITLES).write_text(_dumps(entries), encoding="utf-8")
        print(f"titles: {out / TITLES}")
    print("checks: 0 problems")


def _mode(args: argparse.Namespace) -> keepers.Mode:
    return "drop" if args.drop_team_selected else "flag"


def _run_private(args: argparse.Namespace) -> int:
    if args.team or args.team_name or args.player:
        raise ValueError("--private draws the WhoScored charts only; --team, --team-name and "
                         "--player are StatsBomb options")  # fmt: skip
    f, g = fits("whoscored", cache=args.cache)
    cands = candidates(f)
    assert f.cache is not None  # fits refuses otherwise
    print(f"cache: hits {f.cache.hits} written {f.cache.written}")
    out = {f"{PRIVATE_PREFIX}{n}": c for n, c in gsax_cards.keeper_charts(
        g, names=None, logo=None, team_selected=_mode(args)).items()}  # fmt: skip
    out |= model_charts(f, cands, logo=None, prefix=PRIVATE_PREFIX)
    write_charts(out, Path(args.out or PRIVATE_DIR), private=True)
    return 0


def _run(args: argparse.Namespace) -> int:
    if not (args.team and args.team_name):
        raise ValueError("the StatsBomb charts need --team and --team-name (or pass --private)")
    if not Path(args.manifest).is_file():
        raise ValueError(f"no manifest at {args.manifest}; run make snapshot first")
    digest = snapshot.read_manifest(args.manifest).data_digest
    pc = cache.PredictionCache(args.cache, digest)
    if not (pc.dir / f"{models.GBM.name}.npz").is_file():
        raise ValueError(f"no {models.GBM.name} entry in {pc.dir}; run make xg-models first")
    checkout = Path(args.statsbomb_raw) / statsbomb.OPEN_DATA_COMMIT
    if not (checkout / "tree.json").is_file():
        raise ValueError(
            f"no StatsBomb checkout at {checkout} (player names); pass --statsbomb-raw"
        )
    xg = data.load_xg_data(args.manifest)
    held = holdout_xg(xg, pc, spec=models.GBM)
    table = snapshot.load_snapshot(args.manifest, tables=("shots",))["shots"]
    context = data.pair_holdout(xg.matches).label
    pid = args.player or default_player(held, args.team)
    m = xg.matches.filter(
        (pl.col("home_team_id") == args.team) | (pl.col("away_team_id") == args.team)
    )
    ids = m.sort("match_date", "match_id")["match_id"].to_list()
    names = statsbomb.player_names(checkout, ids, player_ids=[pid])
    if names.height == 0:
        raise ValueError(f"player {pid!r} is not in the lineups of {args.team}'s matches")
    player = str(names["player_name"][0])
    theirs = snapshot.read_manifest(xgot.PROVIDERS["statsbomb"].manifest).data_digest
    if theirs != digest:
        raise ValueError(
            f"--manifest {args.manifest} has data digest {digest[:16]}, but the keeper and model "
            f"charts read {xgot.PROVIDERS['statsbomb'].manifest} (digest {theirs[:16]}): the "
            "charts must read one dataset"
        )
    f, g = fits("statsbomb", cache=args.cache)
    cands = candidates(f)
    keeper_names = gsax_cards.keeper_names(g, args.statsbomb_raw)
    assert f.cache is not None  # fits refuses otherwise
    print(f"cache: hits {pc.hits + f.cache.hits} written {pc.written + f.cache.written}")
    footer = viz.Footer(("statsbomb",), model=model_text(context))
    team_rows = held.filter(pl.col("team_id") == args.team)
    player_rows = team_rows.filter(pl.col("player_id") == pid)
    logo = statsbomb.logo_file()
    team_note = excluded_note(table, xg.exclusions, team=args.team)
    player_note = excluded_note(table, xg.exclusions, team=args.team, player=pid)
    out: dict[str, viz.Canvas] = {}
    for theme in viz.THEMES:
        for size in viz.SIZES:
            team_c = charts.team_chart(
                team_rows, held, team=args.team_name, context=context, theme=theme, size=size,
                footer=footer, excluded=team_note, logo=logo,
            )  # fmt: skip
            player_c = charts.player_chart(
                player_rows, held, player=player, team=args.team_name, context=context,
                theme=theme, size=size, footer=footer, excluded=player_note, logo=logo,
            )  # fmt: skip
            for kind, who, c in (("team", args.team_name, team_c), ("player", player, player_c)):
                out[f"{kind}-{slug(who)}-{theme}-{size}.png"] = c
    out |= gsax_cards.keeper_charts(g, names=keeper_names, logo=logo, team_selected=_mode(args))
    out |= model_charts(f, cands, logo=logo)
    write_charts(out, Path(args.out or CHART_OUT), private=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
