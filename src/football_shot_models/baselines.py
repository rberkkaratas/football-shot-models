"""The M1.1 xG baselines, their model-card text and the command line that writes the cards.

Three baselines, each a step up from the last: the **base rate** (every shot gets the
training goal rate), **logistic regression on distance and angle**, and the same **plus body
part**. Every later xG model has to beat them on the same folds, through the same harness
(:mod:`football_shot_models.evaluation`).

The logistic regression is unpenalised, with no class weights (rebalancing would bias the
probabilities); its scaler sits inside the pipeline, so each fit scales by its own training
rows. StatsBomb's own xG (``provider_xg``, a lineage column) is the benchmark: scored on the
same shots as a reference ceiling, never a feature.

Run ``python -m football_shot_models.baselines`` (``make xg-baselines``) to reproduce the
numbers and rewrite ``docs/models/xg-*.md``. A card holds aggregates only, never data rows.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
from football_core.io import snapshot
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from football_shot_models import data
from football_shot_models.cards import CardContext, CardText, card_context, fmt, pct, render_card
from football_shot_models.evaluation import (
    BaseRate,
    Coefficients,
    Estimator,
    Evaluation,
    ModelSpec,
    best,
    cv_folds,
    evaluate,
    evaluate_benchmark,
)
from football_shot_models.metrics import Scores, check_thresholds, fold_xg_error

FloatArray = npt.NDArray[np.float64]
StrPath = str | os.PathLike[str]

# Read at call time, so a test can lower MAX_ITER to force a convergence failure.
MAX_ITER = 1000
TOL = 1e-10  # solver precision: at the default 1e-4 lbfgs stops short of the MLE


def make_model() -> Pipeline:
    """The unpenalised logistic regression pipeline every LR baseline uses.

    Examples
    --------
    >>> make_model()[-1].C
    inf
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=np.inf, tol=TOL, max_iter=MAX_ITER))


def lr_coefficients(model: Estimator, names: tuple[str, ...]) -> Coefficients:
    """Standardised coefficients of a :func:`make_model` fit, and the same in natural units.

    Natural units are per metre, per radian or per 0 / 1 indicator: the coefficient divided
    by the scaler's standard deviation, with the intercept shifted by the scaler's means.

    Examples
    --------
    >>> x = np.array([[1.0], [2.0], [3.0], [4.0]])
    >>> model = make_model().fit(x, np.array([False, True, False, True]))
    >>> [term for term, _, _ in lr_coefficients(model, ("distance_m",))]
    ['intercept', 'distance_m']
    """
    if not isinstance(model, Pipeline):
        raise TypeError(f"expected a make_model() pipeline, not {type(model).__name__}")
    scaler, lr = model[0], model[-1]
    beta: FloatArray = lr.coef_[0]
    mean: FloatArray = scaler.mean_
    scale: FloatArray = scaler.scale_
    natural = beta / scale
    b0 = float(lr.intercept_[0])
    rows = [("intercept", b0, b0 - float(np.sum(beta * mean / scale)))]
    rows += [(n, float(b), float(r)) for n, b, r in zip(names, beta, natural, strict=True)]
    return tuple(rows)


BASELINES: Final[tuple[ModelSpec, ...]] = (
    ModelSpec("xg-base-rate", "Base rate", (), BaseRate),
    ModelSpec(
        "xg-lr-distance-angle",
        "LR distance + angle",
        ("distance_m", "angle_rad"),
        make_model,
        lr_coefficients,
    ),
    ModelSpec(
        "xg-lr-distance-angle-body-part",
        "LR distance + angle + body part",
        ("distance_m", "angle_rad", "body_part"),
        make_model,
        lr_coefficients,
    ),
)


# ---- card text ----------------------------------------------------------------------------

INTENDED_USE: Final[str] = (
    "A transparent floor for the xG models of this repo: how much of P(goal) shot geometry "
    "(and body part) alone explains on StatsBomb open data. Later models must beat it on the "
    "same folds. It is not for rating players or teams, not for penalties, and not for other "
    "providers' data without re-validation."
)
FEATURE_ROWS: Final[dict[str, str]] = {
    "distance_m": "| `distance_m` | Metres from the shot location to the goal centre "
    "| `distance_m` (`pitch.shot_distance`) |",
    "angle_rad": "| `angle_rad` | Opening angle between the posts seen from the shot location, "
    "radians | `angle_rad` (`pitch.opening_angle`) |",
    "body_part": "| `body_part` | Indicators for left foot, head and other; reference right foot "
    "| `body_part` |",
}
LIMITATIONS: Final[tuple[str, ...]] = (
    "Distance, angle and body part only: no defenders, keeper position or pressure.",
    "Men and women are pooled; see the gender and family segments.",
    "Team-selected seasons (one team in ≥ 50 % of a competition-season's matches) "
    "over-represent a few clubs. The rule also takes every one- or two-match "
    "competition-season (Champions League, World Cup and Copa del Rey finals, among others), "
    "since both teams then play in all of its matches; see Data.",
    "Direct corners are logged as Shot events mostly when they threaten goal, so they are "
    "outcome-selected and excluded (see Filters). Wide direct free kicks are selected the same "
    "way; they stay in the population, scored by geometry only (F5).",
    "`situation` is unused (`play_pattern` spans the possession, C4 F3).",
    "Penalties are a separate model (M1.5).",
)
CHANGELOG: Final[tuple[str, ...]] = (
    "v1 (MS2): first baseline.",
    "v2 (MS3): direct corners excluded (logged on outcome), all numbers re-measured; Segments "
    "show the pooled-OOF calibration slope.",
    "v3 (MS3, M1.4): the benchmark table shows calibration slopes.",
)


def _base_rate_model(ev: Evaluation, c: CardContext) -> list[str]:
    return [
        "Constant prediction: the goal rate of the training shots "
        f"({pct(ev.train.goals / ev.train.n)} on all of dev). Nothing else is fitted. "
        f"scikit-learn {c.sklearn_version} (metrics and folds).",
    ]


def _lr_model(ev: Evaluation, c: CardContext) -> list[str]:
    lr = make_model()[-1]  # the settings every fit in this run used
    return [
        f"`make_pipeline(StandardScaler(), LogisticRegression(C={lr.C}, tol={lr.tol:g}, "
        f"max_iter={lr.max_iter}))`: unpenalised logistic regression ({lr.solver}), no class "
        "weights, no calibration step. The scaler is fitted on the training rows of each fit. "
        f"scikit-learn {c.sklearn_version}.",
        "",
        "Coefficients of the fit on all of dev. Natural units are log-odds per metre, per "
        "radian, or vs right foot at the same spot.",
        "",
        "| Term | Standardised | Natural units | Odds ratio |",
        "|---|---|---|---|",
        *(
            f"| {term} | {fmt(std)} | {fmt(nat)} | "
            f"{'n/a' if term == 'intercept' else f'{np.exp(nat):.2f}'} |"
            for term, std, nat in ev.coefficients
        ),
    ]


def _text(spec: ModelSpec) -> CardText:
    if not spec.features:
        features: tuple[str, ...] = ("None: every shot gets the goal rate of the training shots.",)
    else:
        features = (
            "| Feature | Definition | Source field(s) |",
            "|---|---|---|",
            *(FEATURE_ROWS[f] for f in spec.features),
        )
    return CardText(
        role="Baseline for M1.1.",
        intended_use=INTENDED_USE,
        features=features,
        model=_lr_model if spec.features else _base_rate_model,
        limitations=LIMITATIONS,
        changelog=CHANGELOG,
    )


CARDS: Final[dict[str, CardText]] = {spec.name: _text(spec) for spec in BASELINES}


# ---- command line -------------------------------------------------------------------------


def summary_line(ev: Evaluation) -> str:
    """One line of CV and hold-out scores, for the command-line output.

    Examples
    --------
    >>> summary_line(ev)  # doctest: +SKIP
    'xg-base-rate: cv log_loss 0.3190 brier 0.0896 auc 0.5000 xg_error 0.00% | hold-out ...'
    """

    def part(s: Scores) -> str:
        return (
            f"log_loss {fmt(s.log_loss)} brier {fmt(s.brier)} auc {fmt(s.auc)} "
            f"xg_error {pct(s.xg_error)}"
        )

    return f"{ev.spec.name}: cv {part(ev.cv)} | hold-out {part(ev.holdout)}"


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
            ev,
            CARDS[ev.spec.name],
            evaluations=evaluations,
            benchmark=benchmark,
            dev=dev,
            context=context,
        )
        (out / f"{ev.spec.name}.md").write_text(card, encoding="utf-8", newline="\n")
    counts = ", ".join(f"{reason} {shots:,}" for reason, shots, _ in context.exclusions)
    print(f"exclusions: {counts}")
    status: str = logged.status
    if status == "unchanged" and logged.manifest_written:
        status = "unchanged (manifest updated)"  # the same wording as `fc snapshot`
    print(f"snapshot {data.EXCLUSIONS_SNAPSHOT}: {status}")
    for ev in (*evaluations, benchmark):
        print(summary_line(ev))
    top = best(evaluations)
    print(f"best: {top.spec.name}")
    for check in check_thresholds(top.cv):
        t = check.threshold
        verdict = "MET" if check.passed else "MISSED"
        print(f"{t.metric} {check.measured:.4f} {t.op} {t.value:g}: {verdict}")
    mean, worst = fold_xg_error(top.folds)
    print(f"xg_error fold mean {mean:.4f}, max {worst:.4f} (the line above is pooled)")
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
