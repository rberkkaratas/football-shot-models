"""The M1.2 and M1.3 xG steps: splines, context and freeze frames, each an LR ablation.

Each step adds one group of features to the previous one and must earn its place on the same
folds (the MS2 harness, :mod:`football_shot_models.evaluation`):

- **S1** ``xg-lr-splines``: the MS2 best baseline (distance, angle, body part) plus
  restricted cubic spline terms of distance and angle, so P(goal) can bend where shots are
  many;
- **S2** ``xg-lr-context``: plus ``phase``, ``assist_type``, ``contact``, ``open_goal`` and
  ``one_on_one``. A candidate stays iff dropping it from S2 loses log loss with a CV interval
  above 0 (added for a rejected one: ``first_time`` and ``gender`` add nothing);
- **S3** ``xg-lr-frames``: plus the freeze-frame features (shooting cone, nearest opponent,
  keeper geometry).

Every gain is a :func:`~football_shot_models.evaluation.compare` row: the log-loss gain over
the previous step, over the MS2 baseline and over StatsBomb's own xG, with a 95 % interval
from resampling matches. The ``VARIANTS`` (one context column or one frame group dropped, a
rejected column added) give the drop-one rows that justify each feature group.

The steps use ``solver="newton-cholesky"``: the same unpenalised maximum-likelihood fit as
MS2's lbfgs, but it converges on the 35 columns of S3, where lbfgs stops short. The steps are
evaluated and their cards written by :func:`football_shot_models.models.run` (``make
xg-models``), next to the GBM, so every step card lists every model.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from football_core import schema
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from football_shot_models import baselines, features
from football_shot_models.cards import CardContext, CardText, fmt, gain_cell, render_card, side
from football_shot_models.evaluation import (
    Comparison,
    Evaluation,
    Folds,
    ModelSpec,
    compare,
    evaluate,
    labels,
)
from football_shot_models.metrics import Check


def make_lr() -> Pipeline:
    """The unpenalised logistic regression of every step, solved by Newton-Cholesky.

    Examples
    --------
    >>> make_lr()[-1].solver
    'newton-cholesky'
    """
    lr = LogisticRegression(
        C=np.inf, solver="newton-cholesky", tol=baselines.TOL, max_iter=baselines.MAX_ITER
    )
    return make_pipeline(StandardScaler(), lr)


SPLINES: Final[tuple[str, ...]] = (
    *("distance_m", "distance_m_rcs1", "distance_m_rcs2", "distance_m_rcs3"),
    *("angle_rad", "angle_rad_rcs1", "angle_rad_rcs2", "angle_rad_rcs3"),
    "body_part",
)
CONTEXT: Final[tuple[str, ...]] = ("phase", "assist_type", "contact", "open_goal", "one_on_one")
REJECTED: Final[tuple[str, ...]] = ("first_time", "gender")
FRAME_GROUPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "cone": ("opponents_in_cone", "teammates_in_cone", "gk_in_cone"),
        "nearest": ("nearest_opponent_m",),
        "keeper": ("gk_absent", "gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m"),
    }
)
FRAMES: Final[tuple[str, ...]] = tuple(c for group in FRAME_GROUPS.values() for c in group)

STEPS: Final[tuple[ModelSpec, ...]] = tuple(
    ModelSpec(name, title, feats, make_lr, baselines.lr_coefficients)
    for name, title, feats in (
        ("xg-lr-splines", "LR splines + body part", SPLINES),
        ("xg-lr-context", "LR splines + context", SPLINES + CONTEXT),
        ("xg-lr-frames", "LR splines + context + freeze frames", SPLINES + CONTEXT + FRAMES),
    )
)
_STEP: Final[Mapping[str, ModelSpec]] = MappingProxyType({s.name: s for s in STEPS})


@dataclass(frozen=True)
class Variant:
    """A step with some columns dropped, or added: the drop-one / add-one rows of its card.

    Examples
    --------
    >>> v = Variant("xg-lr-frames", "drop", FRAME_GROUPS["nearest"])
    >>> v.label, v.spec().name
    ('nearest group', 'xg-lr-frames-drop-nearest')
    """

    step: str
    kind: Literal["drop", "add"]
    columns: tuple[str, ...]

    @property
    def label(self) -> str:
        """The frame group's name (``"cone group"``), or the column names joined by ``+``."""
        group = next((g for g, cols in FRAME_GROUPS.items() if cols == self.columns), None)
        return f"{group} group" if group else "+".join(self.columns)

    @property
    def name(self) -> str:
        """The variant's spec name; needs no step lookup, so any split's steps can use it."""
        return f"{self.step}-{self.kind}-{self.label.removesuffix(' group')}"

    def spec(self) -> ModelSpec:
        """The step's spec with the columns dropped or added (no coefficients, no card)."""
        step = _STEP[self.step]
        kept = tuple(f for f in step.features if f not in self.columns)
        feats = kept if self.kind == "drop" else step.features + self.columns
        return ModelSpec(self.name, f"{step.title}, {self.kind} {self.label}", feats, make_lr)


VARIANTS: Final[tuple[Variant, ...]] = (
    *(Variant("xg-lr-context", "drop", (c,)) for c in CONTEXT),
    *(Variant("xg-lr-context", "add", (c,)) for c in REJECTED),
    *(Variant("xg-lr-frames", "drop", cols) for cols in FRAME_GROUPS.values()),
)


def _columns(v: Variant) -> str:
    cols = ", ".join(f"`{c}`" for c in v.columns)
    return f"{v.label} ({cols})" if v.label.endswith(" group") else cols


def comparisons(
    evaluations: Mapping[str, Evaluation],
    benchmark: Evaluation,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    *,
    chain: Sequence[str],
    variants: Sequence[Variant] = VARIANTS,
) -> dict[str, tuple[Comparison, ...]]:
    """The gain rows of every step after the first in ``chain``, keyed by step name.

    For ``chain[i]`` (i >= 1): over ``chain[i - 1]``, over ``chain[0]`` (i >= 2), over the
    benchmark, then one row per variant of that step (a dropped column: the step over the
    variant; an added one: the variant over the step). Pure wiring over
    :func:`~football_shot_models.evaluation.compare`, so another split and benchmark reuse it.

    Parameters
    ----------
    evaluations : mapping of str to Evaluation
        By spec name: every step in ``chain`` and every variant of those steps.
    benchmark : Evaluation
        The provider's xG on the same folds.
    dev, held : polars.DataFrame
        The frames everything was evaluated on.
    chain : sequence of str
        Step names in order; the first is the reference baseline.
    variants : sequence of Variant, default VARIANTS
        Variants; those of steps not in ``chain[1:]`` are ignored.

    Returns
    -------
    dict of str to tuple of Comparison

    Raises
    ------
    TypeError
        ``chain`` is a single ``str``.
    ValueError
        A name in ``chain`` or a variant without an evaluation; from ``compare``.

    Examples
    --------
    >>> rows = comparisons(evs, sb, dev, held, chain=("s0", "xg-lr-splines"))  # doctest: +SKIP
    >>> [row.label for row in rows["xg-lr-splines"]]  # doctest: +SKIP
    ['over LR distance + angle + body part', 'over StatsBomb xG']
    """
    if isinstance(chain, str):
        raise TypeError("chain must be a sequence of step names, not a single str")

    def get(name: str) -> Evaluation:
        if name not in evaluations:
            raise ValueError(f"no evaluation named {name!r}; have {sorted(evaluations)}")
        return evaluations[name]

    out: dict[str, tuple[Comparison, ...]] = {}
    for i, name in enumerate(chain[1:], start=1):
        new = get(name)
        refs = (get(chain[i - 1]), *((get(chain[0]),) if i >= 2 else ()), benchmark)
        rows = [compare(f"over {ref.spec.title}", ref, new, dev, held) for ref in refs]
        for v in (v for v in variants if v.step == name):
            other = get(v.name)
            if v.kind == "drop":
                rows.append(compare(f"{_columns(v)}: dropped", other, new, dev, held))
            else:
                rows.append(
                    compare(f"{_columns(v)}: added (not in the model)", new, other, dev, held)
                )
        out[name] = tuple(rows)
    return out


# ---- card text ----------------------------------------------------------------------------

INTENDED_USE: Final[str] = (
    "A transparent, reproducible xG model on StatsBomb open data, built one feature group at a "
    "time so each group's gain is measured on the same folds. For research and for comparing "
    "with StatsBomb's own xG; not for rating players or teams without that comparison, not for "
    "penalties, and not for other providers' data (it needs StatsBomb freeze frames and "
    "annotations) without re-fitting."
)


def _knots(knots: tuple[float, ...]) -> str:
    return ", ".join(f"{k:g}" for k in knots)


def _levels(categories: Sequence[str]) -> str:
    return ", ".join(c.replace("_", " ") for c in categories)


_FRAME_SRC: Final[str] = "freeze frame `is_teammate`, `is_keeper`, `x`, `y`; shot `x`, `y`"
_GK: Final[str] = f"capped at {features.GK_CAP_M:g}; 0 with `gk_absent`"
_BOX: Final[str] = f"y < {features.BOX_RIGHT_Y:g} or > {features.BOX_LEFT_Y:g}"
FEATURE_ROWS: Final[dict[str, str]] = {
    **baselines.FEATURE_ROWS,
    **{
        f"{col}_rcs{j + 1}": f"| `{col}_rcs{j + 1}` | Restricted cubic spline term {j + 1} of "
        f"`{col}`, knots {_knots(knots)} {unit} (`features.rcs_term`) | `{col}` |"
        for col, knots, unit in (
            ("distance_m", features.DISTANCE_KNOTS, "m"),
            ("angle_rad", features.ANGLE_KNOTS, "rad"),
        )
        for j in range(len(knots) - 2)
    },
    "phase": f"| `phase` | Indicators for {_levels(features.PHASES[1:])}; reference open play. "
    f"Kick-offs join open play; direct free kicks wide of the box ({_BOX}) join free kick "
    "(they are logged on outcome) | `situation`, `y` |",
    "assist_type": "| `assist_type` | Indicators for the assisting pass: "
    f"{_levels(schema.ASSIST_TYPES[1:])}; "
    "reference none | `assist_type` |",
    "contact": f"| `contact` | Indicators for {_levels(features.CONTACTS[1:])}; reference normal. "
    "Lob is merged into normal (it is read from the ball's flight) | `technique` |",
    "open_goal": "| `open_goal` | 1 when StatsBomb marks the goal as open (an annotator "
    "judgement) | `open_goal` |",
    "one_on_one": "| `one_on_one` | 1 when StatsBomb marks a one-on-one with the keeper (an "
    "annotator judgement) | `one_on_one` |",
    "opponents_in_cone": "| `opponents_in_cone` | Outfield opponents in the triangle shot, "
    f"right post, left post (edges included) | {_FRAME_SRC} |",
    "teammates_in_cone": "| `teammates_in_cone` | Teammates, own keeper included, in that "
    f"triangle | {_FRAME_SRC} |",
    "gk_in_cone": "| `gk_in_cone` | 1 when the opponent keeper is in that triangle; 0 when "
    f"absent | {_FRAME_SRC} |",
    "nearest_opponent_m": "| `nearest_opponent_m` | Metres from the shot to the nearest outfield "
    f"opponent, capped at {features.NEAREST_CAP_M:g} (also when none is in the frame) | "
    f"{_FRAME_SRC} |",
    "gk_absent": "| `gk_absent` | 1 when the frame has no opponent keeper (out of the recorded "
    f"area or far from his goal at the strike) | {_FRAME_SRC} |",
    "gk_to_goal_m": f"| `gk_to_goal_m` | Opponent keeper to the goal centre, metres, {_GK} | "
    f"{_FRAME_SRC} |",
    "gk_to_shot_m": f"| `gk_to_shot_m` | Opponent keeper to the shot location, metres, {_GK} | "
    f"{_FRAME_SRC} |",
    "gk_off_line_m": "| `gk_off_line_m` | Opponent keeper's distance from the bisector of the "
    "shot's angle (the line that splits the goal mouth in the ratio of the shot's distances "
    f"to the posts), metres, {_GK} | {_FRAME_SRC} |",
}

POOLED_LIMITS: Final[tuple[str, ...]] = (
    "Men and women are pooled: adding `gender` gives no gain with a CV interval above 0 (see "
    "the gain table of `xg-lr-context`); the Segments table gives each one's calibration slope.",
    "Direct corners are excluded (logged on outcome); penalties are a separate model (M1.5).",
)
CONTEXT_LIMITS: Final[tuple[str, ...]] = (
    "`open_goal` and `one_on_one` are StatsBomb annotator judgements; other providers record "
    "them differently or not at all.",
    "The context columns were selected on CV among seven candidates, a small optimism; their "
    "drop-one rows are in the gain table of `xg-lr-context`.",
    "Lob is merged into normal contact: it is labelled from the ball's flight, after the strike.",
    "No score state: the game situation enters only through `phase` and the freeze frame.",
)


def holdout_women(c: CardContext) -> str:
    """The card line on women in the hold-out, from the hold-out's own shot count.

    Examples
    --------
    >>> holdout_women(card_context(xg, dev, held))  # doctest: +SKIP
    "The hold-out (Premier League 2015/2016) has no women's shots, so women are judged ..."
    """
    if c.holdout_female_shots == 0:
        return (
            f"The hold-out ({c.holdout_label}) has no women's shots, so women are judged on the "
            "pooled out-of-fold predictions only."
        )
    return (
        f"The hold-out ({c.holdout_label}) has {c.holdout_female_shots:,} women's shots; the "
        "Segments table is pooled out of fold on dev only."
    )


def holdout_drops(rows: Sequence[Comparison]) -> str | None:
    """The card line on the hold-out side of the card's own drop-one rows; ``None`` without any.

    Built from the rows with :func:`~football_shot_models.cards.side`, so a refit cannot leave
    it stale.

    Examples
    --------
    >>> from football_shot_models.metrics import GainCI
    >>> cv, ho = GainCI(0.003, 0.002, 0.004, 2000, 9), GainCI(0.001, -0.001, 0.003, 2000, 9)
    >>> print(holdout_drops([Comparison("`phase`: dropped", cv, ho)]))
    On the hold-out the drop-one interval of `phase` (+0.0010 [-0.0010, +0.0030]) includes 0 \
(this card's gain table); the inclusion rule is decided on CV.
    >>> holdout_drops([]) is None
    True
    """
    drops = [r for r in rows if r.label.endswith(": dropped")]
    if not drops:
        return None
    weak = [
        f"{r.label.removesuffix(': dropped').split(' (')[0]} ({gain_cell(r.holdout)}) "
        f"{side(r.holdout)}"
        for r in drops
        if side(r.holdout) != "lies above 0"
    ]
    if not weak:
        return "On the hold-out every drop-one interval in this card's gain table lies above 0."
    return (
        f"On the hold-out the drop-one interval of {'; of '.join(weak)} (this card's gain "
        "table); the inclusion rule is decided on CV."
    )


FRAME_LIMITS: Final[tuple[str, ...]] = (
    "Freeze frames hold the players the broadcast camera shows, so the counts depend on the "
    "camera; the features are local to the shot (cone, nearest opponent, keeper) rather than "
    "whole-frame counts.",
    f"Keeper distances are capped at {features.GK_CAP_M:g} m, where few shots are (a keeper "
    "far out of goal); `gk_absent` marks a frame without the opponent keeper.",
)


def _model(ev: Evaluation, c: CardContext) -> list[str]:
    lr = make_lr()[-1]  # the settings every fit in this run used
    lines = [
        f"`make_pipeline(StandardScaler(), LogisticRegression(C={lr.C}, solver={lr.solver!r}, "
        f"tol={lr.tol:g}, max_iter={lr.max_iter}))`: unpenalised logistic regression, no class "
        "weights, no calibration step. The scaler is fitted on the training rows of each fit; a "
        "singular Hessian is an error, never a silent switch to another solver. "
        f"scikit-learn {c.sklearn_version}.",
        "",
        "Splines are restricted cubic splines with fixed knots (rounded quantiles of the shots, "
        f"no labels): distance {_knots(features.DISTANCE_KNOTS)} m, angle "
        f"{_knots(features.ANGLE_KNOTS)} rad.",
    ]
    if ev.spec.name == STEPS[0].name:
        lines += [
            "",
            "S0 (the MS2 baseline it is compared with) is fitted with lbfgs; on S0's features "
            "the two solvers agree to 1.0e-7 in every out-of-fold probability.",
        ]
    if ev.spec.name == STEPS[-1].name:
        lines += [
            "",
            f"Freeze-frame distances: nearest opponent capped at {features.NEAREST_CAP_M:g} m, "
            f"keeper distances at {features.GK_CAP_M:g} m.",
        ]
    return [
        *lines,
        "",
        "Coefficients of the fit on all of dev. Natural units are log-odds per unit of the "
        "feature (metre, radian, spline term) or vs the reference level at the same spot.",
        "",
        "| Term | Standardised | Natural units |",
        "|---|---|---|",
        *(f"| {term} | {fmt(std)} | {fmt(nat)} |" for term, std, nat in ev.coefficients),
    ]


CHANGELOG: Final[tuple[str, ...]] = (
    "v1 (MS3): first version.",
    "v2 (MS3, M1.4): the benchmark table lists the GBM and calibration slopes; the overview "
    "names the headline model (v1's 'best model' sentence held until the GBM was added).",
    "v3 (MS3 assessment): the hold-out drop-one and women's lines are computed; the family = "
    "women segment row (the gender = female row) is printed once.",
)


def _text(spec: ModelSpec, role: str, limitations: tuple[str, ...]) -> CardText:
    return CardText(
        role=role,
        intended_use=INTENDED_USE,
        features=(
            "| Feature | Definition | Source field(s) |",
            "|---|---|---|",
            *(FEATURE_ROWS[f] for f in spec.features),
        ),
        model=_model,
        limitations=limitations,
        changelog=CHANGELOG,
        best_label="model",
    )


CARDS: Final[dict[str, CardText]] = {
    STEPS[0].name: _text(
        STEPS[0],
        "M1.2 step S1: the MS2 best baseline plus spline terms of distance and angle.",
        ("Geometry and body part only: no context, defenders or keeper.", *POOLED_LIMITS),
    ),
    STEPS[1].name: _text(
        STEPS[1], "M1.2 step S2: S1 plus the context columns.", (*CONTEXT_LIMITS, *POOLED_LIMITS)
    ),
    STEPS[2].name: _text(
        STEPS[2],
        "M1.3 step S3: S2 plus the freeze-frame features.",
        (*FRAME_LIMITS, *CONTEXT_LIMITS, *POOLED_LIMITS),
    ),
}


def wide_dfk(ev: Evaluation, benchmark: Evaluation, dev: pl.DataFrame) -> str:
    """The card line on wide direct free kicks, from the out-of-fold predictions.

    Examples
    --------
    >>> wide_dfk(s2, sb, dev)  # doctest: +SKIP
    'Wide direct free kicks (y < 13.84 or > 54.16, n 378 in dev): goal rate 0.074, ...'
    """
    dfk = (pl.col("situation") == "direct_free_kick") & features.WIDE
    mask = dev.select(dfk).to_series().to_numpy()
    n = int(mask.sum())

    def mean(values: npt.NDArray[np.generic]) -> str:
        return "n/a" if n == 0 else f"{float(np.mean(values[mask])):.3f}"

    return (
        f"Wide direct free kicks ({_BOX}, n {n:,} in dev): goal rate {mean(labels(dev))}, this "
        f"model's mean xG {mean(ev.oof)}, StatsBomb {mean(benchmark.oof)}. They are logged as "
        "shots mostly when they threaten goal and join `phase = free_kick`, so the model rates "
        "them low."
    )


# ---- evaluation and cards -----------------------------------------------------------------


def evaluate_steps(
    dev: pl.DataFrame, held: pl.DataFrame, folds: Folds, freeze_frames: pl.DataFrame
) -> tuple[tuple[Evaluation, ...], tuple[Evaluation, ...]]:
    """The steps (``STEPS`` order) and their variants (``VARIANTS`` order), on the same folds.

    Examples
    --------
    >>> steps, variants = evaluate_steps(dev, held, folds, xg.freeze_frames)  # doctest: +SKIP
    >>> [ev.spec.name for ev in steps]  # doctest: +SKIP
    ['xg-lr-splines', 'xg-lr-context', 'xg-lr-frames']
    """
    ff = freeze_frames
    steps = tuple(evaluate(s, dev, held, folds, freeze_frames=ff) for s in STEPS)
    variants = tuple(evaluate(v.spec(), dev, held, folds, freeze_frames=ff) for v in VARIANTS)
    return steps, variants


def step_cards(
    steps: Sequence[Evaluation],
    rows: Mapping[str, Sequence[Comparison]],
    benchmark: Evaluation,
    *,
    evaluations: Sequence[Evaluation],
    dev: pl.DataFrame,
    context: CardContext,
    verdicts: Mapping[str, str],
    checks: Mapping[str, Sequence[Check]] | None = None,
) -> dict[str, str]:
    """The Markdown card of every step, keyed by step name.

    Parameters
    ----------
    steps : sequence of Evaluation
        The step evaluations, in ``STEPS`` order.
    rows : mapping of str to sequence of Comparison
        The gain rows of each step (:func:`comparisons`).
    benchmark : Evaluation
        The provider's xG on the same folds.
    evaluations : sequence of Evaluation
        Every model the benchmark tables list; the first is the base rate.
    dev : polars.DataFrame
        The development shots.
    context : CardContext
        Provenance and population counts.
    verdicts : mapping of str to str
        The overview sentence per step name; a step without one gets "best model" wording.
    checks : mapping of str to sequence of Check, optional
        The milestone checks, under the headline model's name when it is a step.

    Returns
    -------
    dict of str to str

    Examples
    --------
    >>> cards = step_cards(steps, rows, sb, evaluations=evs, dev=dev, context=c,
    ...                    verdicts={})  # doctest: +SKIP
    >>> sorted(cards)  # doctest: +SKIP
    ['xg-lr-context', 'xg-lr-frames', 'xg-lr-splines']
    """
    out = {}
    for ev in steps:
        name = ev.spec.name
        text = CARDS[name]
        wide = (wide_dfk(ev, benchmark, dev),) if "phase" in _STEP[name].features else ()
        drops = () if (line := holdout_drops(rows[name])) is None else (line,)
        limits = (*wide, *text.limitations, *drops, holdout_women(context))
        text = dataclasses.replace(text, limitations=limits)
        out[name] = render_card(
            ev,
            text,
            evaluations=evaluations,
            benchmark=benchmark,
            dev=dev,
            context=context,
            thresholds=(),
            comparisons=rows[name],
            verdict=verdicts.get(name),
            checks=(checks or {}).get(name, ()),
        )
    return out
