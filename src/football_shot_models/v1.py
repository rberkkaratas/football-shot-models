"""M1.6: the WhoScored xG model, fitted on 2025-26 and tested on 2026-27 with the MS3 harness.

**Data and split.** The private WhoScored snapshot (builder ``whoscored``, no freeze frames)
is split by season: every match of :data:`TRAIN_SEASONS` is train, every match of
:data:`TEST_SEASONS` is test, and ``data.season_holdout`` asserts that the last train match is
before the first test match. Grouped CV by match runs on train only; the test season is scored
once by the fit on all of train and never used to choose. The few shots whose assisting event
could not be resolved (a null ``assist_type``) are dropped before the split, for every model,
and counted (:func:`population`).

**Candidates.** The MS2 baselines (base rate, distance + angle, + body part), two LR steps
(:data:`STEPS`: splines, then the context :data:`CONTEXT` without freeze frames; gender is one
value on every shot, so it is left out) and a monotone GBM (:data:`GBM`) on the same inputs
without splines. The GBM is the headline only if its CV gain over the best LR has a 95 %
match-bootstrap interval above 0 (``models.headline``).

**Big chance** (Opta q214) enters one labelled ablation (:func:`ablation_spec`), with the
ablation guard; ``evaluation.best`` can never pick it.

**The legacy v0** (:mod:`football_shot_models.legacy`) is a test-only benchmark: it was fitted
before the first test match (:func:`check_v0_boundary`), so it gets no CV row, and it is
compared on the test shots it covers with a match-bootstrap gain.

**Penalties** get the train-season Beta-binomial posterior; the test kicks score it and v0's
constant as constants.

Run ``python -m football_shot_models.v1`` (``make xg-v1``) to write ``docs/models/xg-v1.md``
and print the README block (:func:`football_shot_models.release.summary_block`).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import functools
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import polars as pl
from football_core.io import snapshot
from football_core.io.whoscored import BIG_CHANCE

from football_shot_models import (
    ablation,
    baselines,
    cards,
    data,
    features,
    gbm,
    legacy,
    metrics,
    models,
    penalties,
    release,
)
from football_shot_models.cache import PredictionCache
from football_shot_models.cards import fmt, gain_cell, pct, side
from football_shot_models.evaluation import (
    Comparison,
    Evaluation,
    ModelSpec,
    best,
    compare,
    cv_folds,
    evaluate,
    labels,
)
from football_shot_models.metrics import Check, GainCI, bootstrap_gain, score

StrPath = str | os.PathLike[str]

BUILDER: Final[str] = "whoscored"
DEFAULT_MANIFEST: Final[str] = "manifests/whoscored-v1.json"
TRAIN_SEASONS: Final[tuple[str, ...]] = ("2025/2026", "2026")
TEST_SEASONS: Final[tuple[str, ...]] = ("2026/2027",)
CONTEXT: Final[tuple[str, ...]] = ("phase", "assist", "one_on_one")
STEPS: Final[tuple[ModelSpec, ...]] = (
    ModelSpec(
        "ws-lr-splines",
        "LR splines + body part",
        ablation.SPLINES,
        ablation.make_lr,
        baselines.lr_coefficients,
    ),
    ModelSpec(
        "ws-lr-context",
        "LR splines + context",
        ablation.SPLINES + CONTEXT,
        ablation.make_lr,
        baselines.lr_coefficients,
    ),
)
GBM_SIGNS: Final = MappingProxyType({"distance_m": -1, "angle_rad": 1})
GBM: Final = ModelSpec(
    "ws-gbm",
    "Monotone GBM + Platt",
    ("distance_m", "angle_rad", "body_part", *CONTEXT),
    functools.partial(gbm.MonotoneGBM, GBM_SIGNS),
    gbm.gbm_settings,
)
DISTANCE_ANGLE: Final[str] = "xg-lr-distance-angle"  # the PM's start line, a BASELINES name
# PM brief, quoted: a test log loss above MISS_ABOVE is a miss reported with a reason
MISS_ABOVE: Final[float] = 0.275
WINDOW: Final[tuple[float, float]] = (0.26, 0.27)
START_LINE: Final[float] = 0.297
CARD_FILE: Final[str] = f"{release.V1_NAME}.md"


def ablation_spec(spec: ModelSpec) -> ModelSpec:
    """The labelled big-chance ablation of ``spec``: its features plus ``big_chance``.

    Its guard is ``features.check_ablation_features``, so only this spec may read the column,
    and ``evaluation.best`` never picks it.

    Examples
    --------
    >>> a = ablation_spec(STEPS[1])
    >>> a.name, a.features[-1], a.guard is features.check_ablation_features
    ('ws-lr-context-big-chance', 'big_chance', True)
    """
    return dataclasses.replace(
        spec,
        name=f"{spec.name}-big-chance",
        title=f"{spec.title} + big chance (labelled ablation)",
        features=spec.features + ("big_chance",),
        guard=features.check_ablation_features,
    )


def population(xg: data.XgData) -> tuple[pl.DataFrame, int]:
    """The v1 population: shots with a resolved ``assist_type``, and how many were dropped.

    Raises
    ------
    ValueError
        The shots have no ``assist_type`` column.

    Examples
    --------
    >>> xg = data.XgData(pl.DataFrame({"assist_type": ["pass", None, "none"]}), *[None] * 5)
    >>> shots, dropped = population(xg)
    >>> shots.height, dropped
    (2, 1)
    """
    if "assist_type" not in xg.shots.columns:
        raise ValueError("population: the shots have no assist_type column")
    shots = xg.shots.filter(pl.col("assist_type").is_not_null())
    return shots, xg.shots.height - shots.height


def check_v0_boundary(
    model: legacy.LegacyModel, matches: pl.DataFrame, holdout: data.Holdout
) -> None:
    """Refuse a v0 that was fitted on or after the first test match.

    Raises
    ------
    ValueError
        The test matches have no date, or ``model.trained_on`` is not before the first one
        (both dates in the message).

    Examples
    --------
    >>> from datetime import date
    >>> m = legacy.LegacyModel({}, 0.0, 0.78, date(2026, 8, 7), 10, "")
    >>> matches = pl.DataFrame({"match_id": ["ws:9"], "match_date": [date(2026, 8, 7)]})
    >>> check_v0_boundary(m, matches, data.Holdout(pl.col("match_id") == "ws:9", "t"))
    ... # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: v0 was fitted on data up to 2026-08-07, which is not before the first test ...
    """
    first = matches.filter(holdout.expr)["match_date"].min()
    if not isinstance(first, datetime.date):
        raise ValueError(f"the test matches have no first date, got {first!r}")
    if not model.trained_on < first:
        raise ValueError(
            f"v0 was fitted on data up to {model.trained_on}, which is not before the first "
            f"test match ({first}); its test comparison would not be out of sample"
        )


def v1_checks(
    head: Evaluation,
    gbm: Evaluation,
    gbm_gain: Comparison,
    da: Evaluation,
    v0_gain: GainCI,
    coverage: release.Coverage,
) -> tuple[Check, ...]:
    """The five MS4 checks of the headline model, in the card's order; bounds from constants.

    Rows: test log loss; at least as good as v0 on the same test shots (point gain >= 0);
    calibration slope on the test season; the GBM headline rule; big chance only in the
    ablation. Rows 4 and 5 restate rules the run applies, to pin them.

    Examples
    --------
    >>> [c.result for c in v1_checks(head, gbm, gg, da, g0, cov)]  # doctest: +SKIP
    ['missed', 'missed', 'met', 'met', 'met']
    """
    lo, hi = models.SLOPE_RANGE
    share = coverage.covered / coverage.population
    slope = head.holdout.cal_slope
    feats = head.spec.features if isinstance(head.spec, ModelSpec) else ()
    slope_target = f"Calibration slope in [{lo:g}, {hi:g}], test season"
    return (
        Check(
            f"Test log loss <= {MISS_ABOVE:g} (PM window about {WINDOW[0]:g}-{WINDOW[1]:g}; "
            f"{da.spec.title} start line: test {fmt(da.holdout.log_loss)}, PM about "
            f"{START_LINE:g})",
            f"{head.spec.title}: {fmt(head.holdout.log_loss)}",
            "met" if head.holdout.log_loss <= MISS_ABOVE else "missed",
        ),
        Check(
            f"At least as good as {legacy.V0_TITLE} on the same test shots "
            f"({coverage.covered:,} of {coverage.population:,} covered, {pct(share)})",
            f"test gain over v0 {gain_cell(v0_gain)}, an interval that {side(v0_gain)}",
            "met" if v0_gain.gain >= 0.0 else "missed",
        ),
        Check(slope_target, "not estimable", "n/a")
        if slope is None
        else Check(slope_target, fmt(slope), "met" if lo <= slope <= hi else "missed"),
        Check(
            f"GBM headline iff its CV gain {gbm_gain.label} has a 95% interval above 0",
            f"CV {gain_cell(gbm_gain.cv)}; headline {head.spec.title}",
            "met" if (head is gbm) == (gbm_gain.cv.low > 0) else "missed",
        ),
        Check(
            "Big chance only in the labelled ablation, never in the headline",
            f"headline features: {', '.join(feats) or 'none'}",
            "met" if not features.uses_ablation_only(feats) else "missed",
        ),
    )


def _constant_loss(goals: int, n: int, q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError(f"a constant penalty probability must be in (0, 1), got {q}")
    return -(goals * math.log(q) + (n - goals) * math.log1p(-q)) / n


def _penalties(
    kicks: tuple[penalties.PenaltyRow, ...], model: legacy.LegacyModel
) -> tuple[float, float]:
    by = {r.label: r for r in kicks}
    dev, held = (by.get(f"{penalties.IN_GAME}: {k}") for k in ("dev", "hold-out"))
    if dev is None or held is None:
        raise ValueError("penalties: the train or the test season has no in-game penalty")
    goals, n = held.rate.goals, held.rate.n
    return _constant_loss(goals, n, dev.rate.mean), _constant_loss(goals, n, model.penalty_xg)


def run(
    manifest: StrPath = DEFAULT_MANIFEST,
    *,
    v0: StrPath = legacy.DEFAULT_V0,
    docs_dir: StrPath = "docs/models",
    cache: StrPath | None = None,
) -> release.V1Run:
    """Evaluate every v1 model, score v0, write ``docs_dir/xg-v1.md`` and print the summary.

    Parameters
    ----------
    manifest : str or path, default DEFAULT_MANIFEST
        The WhoScored snapshot manifest (git-ignored).
    v0 : str or path, default legacy.DEFAULT_V0
        The pinned legacy ``xg_model.json``.
    docs_dir : str or path, default "docs/models"
        Where the card is written.
    cache : str or path, optional
        Prediction cache root; ``None`` fits everything and writes nothing.

    Returns
    -------
    release.V1Run

    Raises
    ------
    ValueError
        From the loader, ``read_v0``, the split, the v0 boundary, the fits; v0 covering no
        test shot; no in-game penalty in a season.

    Examples
    --------
    >>> run().head.spec.name  # doctest: +SKIP
    'ws-gbm'
    """
    xg = data.load_xg_data(manifest, builder=BUILDER)
    model = legacy.read_v0(v0)
    shots, dropped = population(xg)
    ho = data.season_holdout(xg.matches, TRAIN_SEASONS, TEST_SEASONS)
    check_v0_boundary(model, xg.matches, ho)
    dev, held = data.split_holdout(shots, ho)
    held_all = xg.shots.filter(data.holdout_mask(xg.shots, ho, "shots"))
    folds = cv_folds(dev)
    pc = None if cache is None else PredictionCache(cache, xg.manifest.data_digest)
    base = tuple(evaluate(b, dev, held, folds, cache=pc) for b in baselines.BASELINES)
    steps = tuple(evaluate(s, dev, held, folds, cache=pc) for s in STEPS)
    best_lr = best((*base, *steps))
    if not isinstance(best_lr.spec, ModelSpec):  # best returns a ModelSpec evaluation
        raise TypeError(f"best returned a benchmark: {best_lr.spec!r}")
    gbm_ev = evaluate(GBM, dev, held, folds, cache=pc)  # the module global: tests patch it
    abl = evaluate(ablation_spec(best_lr.spec), dev, held, folds, cache=pc)
    head, gbm_gain = models.headline(best_lr, gbm_ev, dev, held)
    da = next(e for e in base if e.spec.name == DISTANCE_ANGLE)

    cov = legacy.covered(held)
    coverage = release.Coverage(
        int(cov.sum()), held.height, held_all.height - held.height,
        int(legacy.covered(held_all).sum()), held_all.height,
    )  # fmt: skip
    if coverage.covered == 0:
        raise ValueError("v0 covers no test shot")
    mask = cov.to_numpy()
    held_cov = held.filter(cov)
    p0 = legacy.score_v0(held_cov, model).to_numpy()
    y0, g0 = labels(held)[mask], held["match_id"].to_numpy()[mask]
    v0_evs = tuple(dict.fromkeys((head, da, best_lr, gbm_ev, abl)))  # by identity, in order
    v0_gains = tuple(
        (ev.spec.name, bootstrap_gain(y0, p0, ev.holdout_p[mask], g0)) for ev in v0_evs
    )
    by_v0 = dict(v0_gains)
    p_r = legacy.refit_v0(dev.filter(legacy.covered(dev)), held_cov, drop=("big_chance",))
    add = release.Counterfactual(
        f"the best LR ({best_lr.spec.title}) plus q{BIG_CHANCE} alone",
        abl.holdout.log_loss,
        by_v0[abl.spec.name],
    )
    remove = release.Counterfactual(
        f"v0's inputs refitted on train without q{BIG_CHANCE}",
        score(y0, p_r).log_loss,
        bootstrap_gain(y0, p0, p_r, g0),
    )
    pairs = (*dict.fromkeys(((da, head), (da, best_lr))), (best_lr, abl))  # no repeated row
    gains = tuple(
        compare(f"{new.spec.title} over {ref.spec.title}", ref, new, dev, held)
        for ref, new in pairs
    )
    gbm_row = dataclasses.replace(gbm_gain, label=f"{gbm_ev.spec.title} {gbm_gain.label}")
    gains = (*gains[:-1], gbm_row, gains[-1])

    comp = held.join(
        xg.matches.select("match_id", "competition_name"),
        on="match_id", how="left", validate="m:1", maintain_order="left",
    )["competition_name"].cast(pl.String)  # fmt: skip
    groups = metrics.by_group(
        labels(held), head.holdout_p, comp, held["match_id"].to_numpy(), models.MIN_SEGMENT_SHOTS
    )
    v0_groups = metrics.by_group(y0, p0, comp.filter(cov), g0, models.MIN_SEGMENT_SHOTS)
    kicks = penalties.penalty_rows(xg.penalties, holdout=ho)
    context = cards.card_context(dataclasses.replace(xg, shots=shots), dev, held, ho)
    checks = v1_checks(head, gbm_ev, gbm_gain, da, by_v0[head.spec.name], coverage)
    reasons = tuple(
        line
        for row in (1, 2)
        if (line := release.miss_reason(row, checks[row - 1].result == "missed", add, remove,
                                        MISS_ABOVE)) is not None
    )  # fmt: skip
    comps = dev.join(xg.matches.select("match_id", "competition_name"), on="match_id")
    rel = metrics.reliability(labels(held), head.holdout_p)
    result = release.V1Run(
        context=context, train_seasons=TRAIN_SEASONS, test_seasons=TEST_SEASONS,
        dropped_assist=dropped, baselines=base, steps=steps, gbm=gbm_ev, best_lr=best_lr,
        head=head, ablation=abl, gbm_gain=gbm_gain, gains=gains, v0=model, coverage=coverage,
        v0_scores=score(y0, p0), v0_gains=v0_gains, add=add, remove=remove, groups=groups,
        v0_groups=v0_groups, penalties=kicks, penalty_losses=_penalties(kicks, model),
        checks=checks, reasons=reasons,
        train_competitions=tuple(sorted(set(comps["competition_name"].to_list()))),
        reliability=tuple(rel.iter_rows()),
    )  # fmt: skip
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / CARD_FILE).write_text(release.v1_card(result), encoding="utf-8", newline="\n")
    for ev in (*base, *steps, gbm_ev, abl):
        print(baselines.summary_line(ev))
    v0_loss = fmt(result.v0_scores.log_loss)
    print(f"{legacy.V0_TITLE}: test log_loss {v0_loss} on {coverage.covered:,} shots")
    for g in gains:
        print(f"gain | {g.label}: cv {gain_cell(g.cv)} | test {gain_cell(g.holdout)}")
    for name, g0_ci in v0_gains:
        print(f"gain over v0 | {name}: test {gain_cell(g0_ci)}")
    print(f"headline: {head.spec.name}")
    print(*(f"check | {c.target} | {c.measured} | {c.result}" for c in checks), sep="\n")
    print(*(f"reason | {line}" for line in reasons), sep="\n")
    print(release.summary_block(result), end="")
    print(f"card: {out / CARD_FILE}")
    if pc is not None:
        print(f"cache: {pc.dir} ({len(pc.hits)} read, {len(pc.written)} written)")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 0 even when a check is missed (it reports).

    Examples
    --------
    >>> main(["--v0", "missing.json"])  # doctest: +SKIP
    1
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.v1",
        description="Evaluate the WhoScored xG v1 models against the legacy v0; write xg-v1.md.",
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="WhoScored manifest")
    parser.add_argument("--v0", default=legacy.DEFAULT_V0, help="the legacy xg_model.json")
    parser.add_argument("--docs-dir", default="docs/models", help="where the card goes")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--cache", default="data/cache", help="prediction cache root (git-ignored)")
    where.add_argument("--no-cache", action="store_true", help="fit everything, write no cache")
    args = parser.parse_args(argv)
    try:
        run(
            args.manifest,
            v0=args.v0,
            docs_dir=args.docs_dir,
            cache=None if args.no_cache else args.cache,
        )
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
