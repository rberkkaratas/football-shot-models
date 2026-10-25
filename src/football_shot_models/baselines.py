"""The M1.1 xG baselines, their grouped cross-validation and their model cards.

Three baselines, each a step up from the last: the **base rate** (every shot gets the
training goal rate), **logistic regression on distance and angle**, and the same **plus body
part**. Every later xG model has to beat them on the same folds.

**Validation.** Shots from one match share the game state, the keepers and the annotators,
so folds are grouped by ``match_id`` (5-fold ``GroupKFold``, no shuffle). One whole
competition-season (Premier League 2015/2016) is held out as a second test, scored by the
fit on all the other shots. The logistic regression is unpenalised, with no class weights
(rebalancing would bias the probabilities), and its scaler is fitted on training rows only.

**Benchmark.** StatsBomb's own xG (``provider_xg``, a lineage column) is scored on the same
shots as a reference ceiling. It is read only for scoring, never as a feature.

Run ``python -m football_shot_models.baselines`` (``make xg-baselines``) to reproduce the
numbers and rewrite ``docs/models/xg-*.md``. A card holds aggregates only, never data rows.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import polars as pl
import sklearn
from football_core.io import snapshot
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from football_shot_models import data
from football_shot_models.features import design_matrix
from football_shot_models.metrics import (
    Scores,
    check_thresholds,
    fold_sd,
    mean_scores,
    reliability,
    score,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]
Folds = tuple[tuple[IntArray, IntArray], ...]
StrPath = str | os.PathLike[str]

# Read at call time, so a test can lower MAX_ITER to force a convergence failure.
N_SPLITS = 5
MAX_ITER = 1000
TOL = 1e-10  # solver precision: at the default 1e-4 lbfgs stops short of the MLE


@dataclass(frozen=True)
class Baseline:
    """A baseline: card file name, display title and xG features (``()`` = base rate)."""

    name: str
    title: str
    features: tuple[str, ...]


BASELINES: Final[tuple[Baseline, ...]] = (
    Baseline("xg-base-rate", "Base rate", ()),
    Baseline("xg-lr-distance-angle", "LR distance + angle", ("distance_m", "angle_rad")),
    Baseline(
        "xg-lr-distance-angle-body-part",
        "LR distance + angle + body part",
        ("distance_m", "angle_rad", "body_part"),
    ),
)
BENCHMARK: Final[Baseline] = Baseline("statsbomb-xg", "StatsBomb xG", ())


@dataclass(frozen=True, eq=False)
class Evaluation:
    """One model scored by grouped CV, in-sample on dev and on the hold-out.

    Attributes
    ----------
    baseline : Baseline
        The model.
    folds : tuple of Scores
        Per test fold.
    cv : Scores
        ``mean_scores(folds)``, the grouped-CV score.
    oof : numpy.ndarray
        Out-of-fold probabilities, in ``dev`` order.
    train, holdout : Scores
        The fit on all of dev, scored on dev (in-sample) and on the hold-out.
    coefficients : tuple of (str, float, float)
        ``(term, standardised, natural units)`` for the intercept and each design column of
        the fit on all of dev; ``()`` without features.
    """

    baseline: Baseline
    folds: tuple[Scores, ...]
    cv: Scores
    oof: FloatArray
    train: Scores
    holdout: Scores
    coefficients: tuple[tuple[str, float, float], ...]


@dataclass(frozen=True)
class CardContext:
    """Facts a model card prints besides the evaluations: provenance and population counts."""

    manifest_name: str
    data_digest: str
    source_revision: str | None
    sklearn_version: str
    holdout_label: str
    dev_shots: int
    dev_matches: int
    holdout_shots: int
    holdout_matches: int
    competition_seasons: int
    male_shots: int
    female_shots: int
    exclusions: tuple[tuple[str, int, int], ...]  # (reason, shots, goals), every reason


def make_model() -> Pipeline:
    """The unpenalised logistic regression pipeline every LR baseline uses.

    Examples
    --------
    >>> make_model()[-1].C
    inf
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, tol=TOL, max_iter=MAX_ITER))


def _labels(shots: pl.DataFrame) -> BoolArray:
    y: BoolArray = shots["is_goal"].to_numpy()
    return y


def _fit(baseline: Baseline, train: pl.DataFrame) -> Pipeline:
    x, _ = design_matrix(train, baseline.features)
    model = make_model()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(x, _labels(train))
        except ConvergenceWarning as w:
            raise ValueError(
                f"{baseline.name}: LogisticRegression did not converge in {MAX_ITER} iterations"
            ) from w
    return model


def _predict(model: Pipeline, baseline: Baseline, shots: pl.DataFrame) -> FloatArray:
    x, _ = design_matrix(shots, baseline.features)
    p: FloatArray = np.ascontiguousarray(model.predict_proba(x)[:, 1], dtype=np.float64)
    return p


def _rate(baseline: Baseline, train: pl.DataFrame) -> float:
    rate = train["is_goal"].mean()
    if not isinstance(rate, float):
        raise ValueError(f"{baseline.name}: empty training set")
    return rate


def fit_predict(baseline: Baseline, train: pl.DataFrame, test: pl.DataFrame) -> FloatArray:
    """Fit a baseline on ``train`` and return P(goal) for each ``test`` shot.

    Parameters
    ----------
    baseline : Baseline
        Without features, the prediction is the training goal rate.
    train, test : polars.DataFrame
        Shots with ``is_goal`` and the baseline's features.

    Returns
    -------
    numpy.ndarray
        One probability per test row.

    Raises
    ------
    ValueError
        An empty training set, a feature problem (``design_matrix``), or no convergence in
        ``MAX_ITER`` iterations.

    Examples
    --------
    >>> train = pl.DataFrame({"is_goal": [True, False, False, False]})
    >>> fit_predict(BASELINES[0], train, train.head(2)).tolist()
    [0.25, 0.25]
    """
    if not baseline.features:
        return np.full(test.height, _rate(baseline, train))
    return _predict(_fit(baseline, train), baseline, test)


def cv_folds(dev: pl.DataFrame) -> Folds:
    """``N_SPLITS`` grouped folds by ``match_id``, as (train, test) row-index pairs.

    Raises
    ------
    ValueError
        From scikit-learn, with fewer matches than folds.

    Examples
    --------
    >>> dev = pl.DataFrame({"match_id": [f"sb:{i // 2}" for i in range(10)]})
    >>> sorted(test.tolist() for _, test in cv_folds(dev))
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """
    splitter = GroupKFold(n_splits=N_SPLITS)
    groups = dev["match_id"].to_numpy()
    return tuple(
        (np.asarray(train, dtype=np.intp), np.asarray(test, dtype=np.intp))
        for train, test in splitter.split(np.zeros((dev.height, 1)), groups=groups)
    )


def _coefficients(model: Pipeline, names: tuple[str, ...]) -> tuple[tuple[str, float, float], ...]:
    """Standardised coefficients and the same model in natural units (metres, radians, 0/1)."""
    scaler, lr = model[0], model[-1]
    beta: FloatArray = lr.coef_[0]
    mean: FloatArray = scaler.mean_
    scale: FloatArray = scaler.scale_
    natural = beta / scale
    b0 = float(lr.intercept_[0])
    rows = [("intercept", b0, b0 - float(np.sum(beta * mean / scale)))]
    rows += [(n, float(b), float(r)) for n, b, r in zip(names, beta, natural, strict=True)]
    return tuple(rows)


def _evaluation(
    baseline: Baseline,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    fold_p: list[FloatArray],
    folds: Folds,
    p_train: FloatArray,
    p_held: FloatArray,
    coefficients: tuple[tuple[str, float, float], ...],
) -> Evaluation:
    y = _labels(dev)
    oof = np.full(dev.height, np.nan)
    per_fold = []
    for (_, test), p in zip(folds, fold_p, strict=True):
        oof[test] = p
        per_fold.append(score(y[test], p))
    return Evaluation(
        baseline=baseline,
        folds=tuple(per_fold),
        cv=mean_scores(per_fold),
        oof=oof,
        train=score(y, p_train),
        holdout=score(_labels(held), p_held),
        coefficients=coefficients,
    )


def evaluate(baseline: Baseline, dev: pl.DataFrame, held: pl.DataFrame, folds: Folds) -> Evaluation:
    """Grouped-CV, in-sample and hold-out scores of one baseline.

    Examples
    --------
    >>> ev = evaluate(BASELINES[1], dev, held, cv_folds(dev))  # doctest: +SKIP
    >>> round(ev.cv.log_loss, 4)  # doctest: +SKIP
    0.2968
    """
    fold_p = [fit_predict(baseline, dev[train], dev[test]) for train, test in folds]
    if not baseline.features:
        rate = _rate(baseline, dev)
        return _evaluation(
            baseline, dev, held, fold_p, folds,
            np.full(dev.height, rate), np.full(held.height, rate), (),
        )  # fmt: skip
    model = _fit(baseline, dev)
    _, names = design_matrix(dev.head(0), baseline.features)
    return _evaluation(
        baseline, dev, held, fold_p, folds,
        _predict(model, baseline, dev), _predict(model, baseline, held),
        _coefficients(model, names),
    )  # fmt: skip


def _provider_xg(shots: pl.DataFrame) -> FloatArray:
    nulls = shots["provider_xg"].null_count()
    if nulls:
        raise ValueError(f"provider_xg has {nulls} null value(s)")
    p: FloatArray = shots["provider_xg"].to_numpy()
    return p


def evaluate_benchmark(dev: pl.DataFrame, held: pl.DataFrame, folds: Folds) -> Evaluation:
    """StatsBomb xG (``provider_xg``) scored on the same folds and hold-out (no fitting).

    Examples
    --------
    >>> round(evaluate_benchmark(dev, held, folds).cv.log_loss, 4)  # doctest: +SKIP
    0.2674
    """
    p = _provider_xg(dev)
    fold_p = [p[test] for _, test in folds]
    return _evaluation(BENCHMARK, dev, held, fold_p, folds, p, _provider_xg(held), ())


def best(evaluations: Sequence[Evaluation]) -> Evaluation:
    """The baseline with the lowest grouped-CV log loss (the benchmark never qualifies).

    Raises
    ------
    ValueError
        If no evaluation is of a baseline in ``BASELINES``.

    Examples
    --------
    >>> best(evaluations).baseline.name  # doctest: +SKIP
    'xg-lr-distance-angle-body-part'
    """
    candidates = [ev for ev in evaluations if ev.baseline in BASELINES]
    if not candidates:
        raise ValueError("no baseline evaluation to choose from")
    return min(candidates, key=lambda ev: ev.cv.log_loss)


def card_context(xg: data.XgData, dev: pl.DataFrame, held: pl.DataFrame) -> CardContext:
    """Collect the provenance and counts a card prints.

    Examples
    --------
    >>> card_context(xg, dev, held).competition_seasons  # doctest: +SKIP
    80
    """
    label = xg.matches.filter(
        (pl.col("competition_id") == data.HOLDOUT[0]) & (pl.col("season_id") == data.HOLDOUT[1])
    ).select(pl.concat_str("competition_name", "season_name", separator=" "))
    log = xg.exclusions
    gender = xg.shots["gender"]
    return CardContext(
        manifest_name=xg.manifest.name,
        data_digest=xg.manifest.data_digest,
        source_revision=xg.manifest.source.revision,
        sklearn_version=sklearn.__version__,
        holdout_label=str(label.item(0, 0)) if label.height else "n/a",
        dev_shots=dev.height,
        dev_matches=dev["match_id"].n_unique(),
        holdout_shots=held.height,
        holdout_matches=held["match_id"].n_unique(),
        competition_seasons=xg.shots.select("competition_id", "season_id").n_unique(),
        male_shots=int((gender == "male").sum()),
        female_shots=int((gender == "female").sum()),
        exclusions=tuple(
            (
                reason,
                int((log["reason"] == reason).sum()),
                int(log.filter(pl.col("reason") == reason)["is_goal"].sum()),
            )
            for reason in data.EXCLUSION_REASONS
        ),
    )


# ---- model card ---------------------------------------------------------------------------

TASK: Final[str] = (
    "binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time "
    "included; direct free kicks and corners included; own goals and shoot-outs excluded)"
)
INTENDED_USE: Final[str] = (
    "A transparent floor for the xG models of this repo: how much of P(goal) shot geometry "
    "(and body part) alone explains on StatsBomb open data. Later models must beat it on the "
    "same folds. It is not for rating players or teams, not for penalties, and not for other "
    "providers' data without re-validation."
)
FILTER_LABELS: Final[dict[str, str]] = {
    "own_goal": "own goal (Own Goal Against events; not shots)",
    "shootout": "shoot-out (period 5; all are penalties)",
    "penalty": "penalty (in-game)",
    "no_freeze_frame": "no freeze frame",
}
FEATURE_ROWS: Final[dict[str, str]] = {
    "distance_m": "| `distance_m` | Metres from the shot location to the goal centre "
    "| `distance_m` (`pitch.shot_distance`) |",
    "angle_rad": "| `angle_rad` | Opening angle between the posts seen from the shot location, "
    "radians | `angle_rad` (`pitch.opening_angle`) |",
    "body_part": "| `body_part` | Indicators for left foot, head and other; reference right foot "
    "| `body_part` |",
}
VALIDATION: Final[str] = (
    "Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match "
    "has shots in two folds, and every model uses the same folds. **Validation** is the mean "
    "over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of "
    "dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one "
    "of the four complete 2015/16 men's league seasons, with its three sibling leagues in dev. "
    "It tests transfer to an unseen league in the same season and data spec, not across time "
    "or gender."
)
BENCHMARK_CAVEAT: Final[str] = (
    "StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: "
    "StatsBomb fits it on its own data, which likely includes these matches, and it uses "
    "freeze-frame and other inputs."
)
LIMITATIONS: Final[tuple[str, ...]] = (
    "Distance, angle and body part only: no defenders, keeper position or pressure.",
    "Men and women are pooled; see the gender and family segments.",
    "Team-selected seasons (one team in ≥ 50 % of a competition-season's matches) "
    "over-represent a few clubs.",
    "Direct corners and wide direct free kicks are logged as Shot events mostly when they "
    "threaten goal, so they are outcome-selected. They stay in the population, scored by "
    "geometry only, and `situation` must not become a feature before C3/C4 restrict those "
    "values (F5).",
    "`situation` is unused (`play_pattern` spans the possession, C4 F3).",
    "Penalties are a separate model (M1.5).",
)
THRESHOLD_NOTE: Final[str] = (
    "Σ xG and calibration on grouped CV are close to their targets by construction for the "
    "base rate and for an unpenalised logistic regression with an intercept; see the hold-out "
    "column and the reliability table."
)
RELIABILITY_NOTE: Final[str] = "This table stands in for the reliability plot until C7 (F1)."
LICENSING: Final[str] = (
    "Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never "
    "committed or redistributed; this card holds aggregates only."
)
METRIC_LABELS: Final[dict[str, str]] = {
    "log_loss": "log loss",
    "brier": "Brier",
    "auc": "ROC AUC",
    "xg_error": "abs(Σ xG − goals) / goals",  # no "|": it would split a table cell
}


def _f(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2%}"


def _ratio(xg: float, goals: int) -> str:
    return "n/a" if goals == 0 else _f(xg / goals)


def _measured(metric: str, x: float) -> str:
    return _pct(x) if metric == "xg_error" else _f(x)


def _segment(dev: pl.DataFrame, oof: FloatArray, column: str, value: str) -> dict[str, float]:
    """Pooled out-of-fold numbers of one segment; empty dict when it has no shots."""
    mask = (dev[column].cast(pl.String) == value).to_numpy()
    if not mask.any():
        return {}
    y = _labels(dev)[mask].astype(np.float64)
    p = oof[mask]
    return {
        "shots": float(mask.sum()),
        "goals": float(y.sum()),
        "xg": float(p.sum()),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _segments(dev: pl.DataFrame) -> tuple[tuple[str, str], ...]:
    dtype = dev["body_part"].dtype
    body = tuple(dtype.categories.to_list()) if isinstance(dtype, pl.Enum) else ()
    return (
        *(("family", v) for v in data.FAMILIES),
        *(("gender", v) for v in ("female", "male")),
        *(("body_part", v) for v in body),
    )


def _skill(metric: str, ev: Evaluation, base: Evaluation, split: str) -> str:
    def value(e: Evaluation) -> float:
        return float(getattr(getattr(e, split), metric))

    return _f(1.0 - value(ev) / value(base))


def _why(ev: Evaluation, base: Evaluation, benchmark: Evaluation, dev: pl.DataFrame) -> str:
    seg = {v: _segment(dev, ev.oof, "family", v) for v in data.FAMILIES}

    def scored(s: dict[str, float]) -> str:
        return f"{_f(s['log_loss'])} / {_f(s['brier'])}"

    def ratio(s: dict[str, float]) -> str:
        return f"Σ xG / goals {_ratio(s['xg'], int(s['goals']))}"

    clauses = []
    if s := seg["league or tournament"]:
        clauses.append(f"{scored(s)} on league or tournament shots ({ratio(s)})")
    if s := seg["team-selected"]:
        clauses.append(
            f"{scored(s)} on team-selected seasons ({_pct(s['shots'] / dev.height)} of dev "
            f"shots, goal rate {_pct(s['goals'] / s['shots'])}, {ratio(s)})"
        )
    if s := seg["women"]:
        clauses.append(f"{scored(s)} on women's shots ({ratio(s)})")
    listed = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]
    return (
        "**Why.** Absolute log loss and Brier depend on the goal rate and on which matches are "
        f"in the sample. The base rate alone scores {_f(base.cv.log_loss)} / "
        f"{_f(base.cv.brier)} on the CV shots (goal rate {_pct(ev.cv.goals / ev.cv.n)}) and "
        f"{_f(base.holdout.log_loss)} / {_f(base.holdout.brier)} on the hold-out "
        f"({_pct(ev.holdout.goals / ev.holdout.n)}). By segment, out of fold, this model "
        f"scores {listed}. Its log-loss skill against the base rate is "
        f"{_skill('log_loss', ev, base, 'cv')} on CV and "
        f"{_skill('log_loss', ev, base, 'holdout')} on the hold-out. StatsBomb xG, with "
        f"freeze-frame inputs, scores {_f(benchmark.cv.log_loss)} / {_f(benchmark.cv.brier)} "
        "on the CV shots. Thresholds come from a single-season pilot (roadmap); they are "
        "reported, not tuned to."
    )


def _metrics_section(ev: Evaluation, base: Evaluation) -> list[str]:
    def sd(metric: str) -> str:
        return f" ± {_f(fold_sd(ev.folds, metric))}"

    splits = (ev.train, ev.cv, ev.holdout)
    rows = [
        ("Log loss", [_f(ev.train.log_loss), _f(ev.cv.log_loss) + sd("log_loss"),
                      _f(ev.holdout.log_loss)]),
        ("Brier score", [_f(ev.train.brier), _f(ev.cv.brier) + sd("brier"),
                         _f(ev.holdout.brier)]),
        ("ROC AUC", [_f(ev.train.auc), _f(ev.cv.auc) + sd("auc"), _f(ev.holdout.auc)]),
        ("Log-loss skill", [_skill("log_loss", ev, base, s) for s in ("train", "cv", "holdout")]),
        ("Brier skill", [_skill("brier", ev, base, s) for s in ("train", "cv", "holdout")]),
        ("Calibration intercept", [_f(s.cal_intercept) for s in splits]),
        ("Calibration slope", [_f(s.cal_slope) for s in splits]),
        ("Σ xG", [_f(s.xg) for s in splits]),
        ("Goals", [f"{s.goals:,}" for s in splits]),
        (METRIC_LABELS["xg_error"], [_pct(s.xg_error) for s in splits]),
    ]  # fmt: skip
    return [
        "| Metric | Train | Validation | Test |",
        "|---|---|---|---|",
        *(f"| {label} | {' | '.join(cells)} |" for label, cells in rows),
    ]


def _thresholds_section(
    ev: Evaluation, base: Evaluation, benchmark: Evaluation, dev: pl.DataFrame
) -> list[str]:
    cv, held = check_thresholds(ev.cv), check_thresholds(ev.holdout)
    lines = [
        "### Milestone thresholds",
        "",
        "| Threshold | CV | Hold-out | CV result |",
        "|---|---|---|---|",
    ]
    for c, h in zip(cv, held, strict=True):
        t = c.threshold
        bound = _pct(t.value) if t.metric == "xg_error" else f"{t.value:g}"
        label = f"{METRIC_LABELS[t.metric]} {t.op} {bound}"
        result = "met" if c.passed else "missed"
        lines.append(
            f"| {label} | {_measured(t.metric, c.measured)} | "
            f"{_measured(t.metric, h.measured)} | {result} |"
        )
    lines += ["", THRESHOLD_NOTE]
    if not all(c.passed for c in cv):
        lines += ["", _why(ev, base, benchmark, dev)]
    return lines


def _segments_section(ev: Evaluation, dev: pl.DataFrame) -> list[str]:
    lines = [
        "### Segments",
        "",
        "Pooled out of fold.",
        "",
        "| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for column, value in _segments(dev):
        if not (s := _segment(dev, ev.oof, column, value)):
            continue
        shots, goals = int(s["shots"]), int(s["goals"])
        lines.append(
            f"| {column} = {value} | {shots:,} | {goals:,} | {_pct(goals / shots)} | "
            f"{_f(s['xg'])} | {_ratio(s['xg'], goals)} | {_f(s['log_loss'])} | "
            f"{_f(s['brier'])} |"
        )
    return lines


def render_card(
    ev: Evaluation,
    *,
    evaluations: Sequence[Evaluation],
    benchmark: Evaluation,
    dev: pl.DataFrame,
    context: CardContext,
) -> str:
    """The model card of one baseline, following ``docs/model-card-template.md``.

    Parameters
    ----------
    ev : Evaluation
        The baseline the card is about.
    evaluations : sequence of Evaluation
        The three baseline evaluations (the base rate is the skill reference).
    benchmark : Evaluation
        StatsBomb xG.
    dev : polars.DataFrame
        The development shots ``ev`` was evaluated on (for the segments).
    context : CardContext
        Provenance and population counts.

    Returns
    -------
    str
        Markdown, with no date or time, so a rerun gives identical bytes.

    Examples
    --------
    >>> card = render_card(ev, evaluations=evs, benchmark=b, dev=dev, context=c)  # doctest: +SKIP
    >>> card.splitlines()[0]  # doctest: +SKIP
    '# Model card: Base rate'
    """
    base = next(e for e in evaluations if e.baseline.name == "xg-base-rate")
    top = best(evaluations)
    is_best = (
        "This is the best baseline (lowest grouped-CV log loss)."
        if top is ev
        else f"Not the best baseline; that is {top.baseline.title}."
    )
    c = context
    filters = [
        f"| {FILTER_LABELS[reason]} | {shots:,} | {goals:,} |"
        for reason, shots, goals in c.exclusions
    ]
    if ev.baseline.features:
        features = [
            "| Feature | Definition | Source field(s) |",
            "|---|---|---|",
            *(FEATURE_ROWS[f] for f in ev.baseline.features),
        ]
        model = [
            "`make_pipeline(StandardScaler(), LogisticRegression(C=inf, tol=1e-10, "
            "max_iter=1000))`: unpenalised logistic regression (lbfgs), no class weights, no "
            "calibration step. The scaler is fitted on the training rows of each fit. "
            f"scikit-learn {c.sklearn_version}.",
            "",
            "Coefficients of the fit on all of dev. Natural units are log-odds per metre, per "
            "radian, or vs right foot at the same spot.",
            "",
            "| Term | Standardised | Natural units | Odds ratio |",
            "|---|---|---|---|",
            *(
                f"| {term} | {_f(std)} | {_f(nat)} | "
                f"{'n/a' if term == 'intercept' else f'{np.exp(nat):.2f}'} |"
                for term, std, nat in ev.coefficients
            ),
        ]
    else:
        features = ["None: every shot gets the goal rate of the training shots."]
        model = [
            "Constant prediction: the goal rate of the training shots "
            f"({_pct(ev.train.goals / ev.train.n)} on all of dev). Nothing else is fitted. "
            f"scikit-learn {c.sklearn_version} (metrics and folds).",
        ]
    benchmarks = [
        "| Model | CV log loss | CV Brier | CV AUC | Hold-out log loss | Hold-out Brier "
        "| Hold-out AUC |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {e.baseline.title} | {_f(e.cv.log_loss)} | {_f(e.cv.brier)} | {_f(e.cv.auc)} | "
            f"{_f(e.holdout.log_loss)} | {_f(e.holdout.brier)} | {_f(e.holdout.auc)} |"
            for e in (*evaluations, benchmark)
        ),
    ]
    rel = reliability(_labels(dev), ev.oof)
    reliability_rows = [
        f"| {b} | {n:,} | {_pct(m)} | {_pct(g)} |" for b, n, m, g in rel.iter_rows()
    ]
    lines = [
        f"# Model card: {ev.baseline.title}",
        "",
        "## Overview",
        f"- **Name / version:** `{ev.baseline.name}` v1",
        f"- **Task:** {TASK}",
        "- **Status:** experimental",
        f"- Baseline for M1.1. {is_best}",
        "",
        "## Intended use",
        INTENDED_USE,
        "",
        "## Data",
        f"- **Source(s):** StatsBomb open data, revision `{c.source_revision}`; snapshot "
        f"`{c.manifest_name}`, data digest `{c.data_digest}`",
        f"- **Competitions & seasons:** {c.competition_seasons:,} competition-seasons; hold-out "
        f"{c.holdout_label}",
        f"- **Rows:** train and validation (dev) {c.dev_shots:,} shots in {c.dev_matches:,} "
        f"matches / test (hold-out) {c.holdout_shots:,} shots in {c.holdout_matches:,} "
        f"matches; men {c.male_shots:,} shots, women {c.female_shots:,}",
        f"- **Filters:** excluded shots, by first matching reason. The per-shot log is snapshot "
        f"`{data.EXCLUSIONS_SNAPSHOT}` (git-ignored; its committed manifest holds counts only).",
        "",
        "| Filter | Shots | Goals |",
        "|---|---|---|",
        *filters,
        "",
        "## Target",
        "`is_goal`: the ball went in. Own goals are excluded, so every goal is the shooter's.",
        "",
        "## Features",
        *features,
        "",
        "## Model",
        *model,
        "",
        "## Validation design",
        VALIDATION,
        "",
        "## Metrics",
        *_metrics_section(ev, base),
        "",
        "Skill is `1 − metric / base-rate metric` on the same shots and column.",
        "",
        *_thresholds_section(ev, base, benchmark, dev),
        "",
        "### Reliability",
        "",
        "Ten equal-count bins of the out-of-fold predictions.",
        "",
        "| Bin | Shots | Mean xG | Goal rate |",
        "|---|---|---|---|",
        *reliability_rows,
        "",
        RELIABILITY_NOTE,
        "",
        *_segments_section(ev, dev),
        "",
        "## Benchmarks",
        *benchmarks,
        "",
        BENCHMARK_CAVEAT,
        "",
        "## Limitations & known biases",
        *(f"- {item}" for item in LIMITATIONS),
        "",
        "## Data licensing",
        LICENSING,
        "",
        "## Changelog",
        "- v1 (MS2): first baseline.",
    ]
    return "\n".join(lines) + "\n"


# ---- command line -------------------------------------------------------------------------


def _summary(ev: Evaluation) -> str:
    def part(s: Scores) -> str:
        return (
            f"log_loss {_f(s.log_loss)} brier {_f(s.brier)} auc {_f(s.auc)} "
            f"xg_error {_pct(s.xg_error)}"
        )

    return f"{ev.baseline.name}: cv {part(ev.cv)} | hold-out {part(ev.holdout)}"


def run(
    manifest: StrPath, *, root: StrPath = ".", docs_dir: StrPath = "docs/models"
) -> tuple[Evaluation, ...]:
    """Load the snapshot, log the exclusions, evaluate everything and write the model cards.

    Parameters
    ----------
    manifest : str or path
        The StatsBomb snapshot manifest.
    root : str or path, default "."
        Repo root for the exclusion-log snapshot.
    docs_dir : str or path, default "docs/models"
        Where ``<baseline name>.md`` is written.

    Returns
    -------
    tuple of Evaluation
        The three baselines in ``BASELINES`` order, then the benchmark.

    Examples
    --------
    >>> run("manifests/statsbomb-open-v1.json")  # doctest: +SKIP
    """
    xg = data.load_xg_data(manifest)
    logged = data.write_exclusion_log(xg.exclusions, manifest, root=root)
    dev, held = data.split_holdout(xg.shots)
    folds = cv_folds(dev)
    evaluations = tuple(evaluate(b, dev, held, folds) for b in BASELINES)
    benchmark = evaluate_benchmark(dev, held, folds)
    context = card_context(xg, dev, held)
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ev in evaluations:
        card = render_card(
            ev, evaluations=evaluations, benchmark=benchmark, dev=dev, context=context
        )
        (out / f"{ev.baseline.name}.md").write_text(card, encoding="utf-8", newline="\n")
    counts = ", ".join(f"{reason} {shots:,}" for reason, shots, _ in context.exclusions)
    print(f"exclusions: {counts}")
    print(f"snapshot {data.EXCLUSIONS_SNAPSHOT}: {logged.status}")
    for ev in (*evaluations, benchmark):
        print(_summary(ev))
    top = best(evaluations)
    print(f"best: {top.baseline.name}")
    for check in check_thresholds(top.cv):
        t = check.threshold
        verdict = "MET" if check.passed else "MISSED"
        print(f"{t.metric} {check.measured:.4f} {t.op} {t.value:g}: {verdict}")
    print(f"cards: {out}")
    return (*evaluations, benchmark)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 0 even when a threshold is missed (it reports).

    Examples
    --------
    >>> main(["--manifest", "manifests/statsbomb-open-v1.json"])  # doctest: +SKIP
    0
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.baselines",
        description="Evaluate the M1.1 xG baselines and write their model cards.",
    )
    parser.add_argument("--manifest", default=data.DEFAULT_MANIFEST, help="snapshot manifest")
    parser.add_argument("--root", default=".", help="repo root for the exclusion log")
    parser.add_argument("--docs-dir", default="docs/models", help="where the cards go")
    args = parser.parse_args(argv)
    try:
        run(args.manifest, root=args.root, docs_dir=args.docs_dir)
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
