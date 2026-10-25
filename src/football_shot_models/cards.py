"""Model cards: one Markdown page per model, every number computed from its evaluation.

A card follows ``docs/model-card-template.md``. The parts that every xG card shares (data
provenance, validation design, metrics, thresholds, reliability, segments, benchmarks,
licensing) are rendered here from an :class:`~football_shot_models.evaluation.Evaluation`.
The parts that belong to one model (its role, intended use, features, model description,
limitations and changelog) come from a :class:`CardText` written next to the model's spec, so
a new model brings its own text instead of inheriting the M1.1 wording.

A card can also carry the log-loss gains of its model over other models
(:class:`~football_shot_models.evaluation.Comparison` rows, with match-bootstrap intervals),
can leave out the milestone thresholds when they do not apply to it, and can carry a verdict
sentence and the MS3 milestone checks (:class:`~football_shot_models.metrics.Check` rows).

A card prints only numbers it computes, holds aggregates only (never a data row or an id) and
carries no date or time, so rerunning on the same snapshot gives identical bytes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import polars as pl
import sklearn
from sklearn.metrics import brier_score_loss, log_loss

from football_shot_models import data
from football_shot_models.evaluation import Comparison, Evaluation, best, labels
from football_shot_models.metrics import (
    N_BOOT,
    SEED,
    THRESHOLDS,
    Check,
    GainCI,
    Metric,
    Scores,
    Threshold,
    calibration,
    check_thresholds,
    fold_sd,
    fold_xg_error,
    metric_value,
    reliability,
)

FloatArray = npt.NDArray[np.float64]
Split = Literal["train", "cv", "holdout"]


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
    holdout_female_shots: int  # women's shots in the hold-out (the cards' pooled-gender line)
    exclusions: tuple[tuple[str, int, int], ...]  # (reason, shots, goals), every reason
    team_selected_seasons: int  # competition-seasons in the team-selected family
    team_selected_short: int  # of those, with one or two matches (both teams reach 50 %)


@dataclass(frozen=True)
class CardText:
    """The parts of a card that belong to one model, written next to its spec.

    Attributes
    ----------
    role : str
        Overview line, e.g. ``"Baseline for M1.1."``.
    intended_use : str
        The Intended use section.
    features : tuple of str
        The Features section lines (a Markdown table or a sentence).
    model : callable
        ``model(ev, context)`` gives the Model section lines; it may print the fitted
        coefficients and the solver settings, read from the code that fitted them.
    limitations, changelog : tuple of str
        One bullet each.
    best_label : str, default "baseline"
        What the overview calls the best of the evaluations ("This is the best baseline").
    checks_title, checks_note : str, default CHECKS_TITLE, CHECKS_NOTE
        The heading and the sentence of the milestone checks section, so a later milestone's
        checks render under their own name.
    """

    role: str
    intended_use: str
    features: tuple[str, ...]
    model: Callable[[Evaluation, CardContext], Sequence[str]]
    limitations: tuple[str, ...]
    changelog: tuple[str, ...]
    best_label: str = "baseline"
    checks_title: str = "Milestone checks (MS3)"
    checks_note: str = "Thresholds re-baselined by the PM: targets are relative to the same shots."


CHECKS_TITLE: Final[str] = CardText.checks_title
CHECKS_NOTE: Final[str] = CardText.checks_note


def card_context(
    xg: data.XgData, dev: pl.DataFrame, held: pl.DataFrame, holdout: data.Holdout | None = None
) -> CardContext:
    """Collect the provenance and counts a card prints.

    Parameters
    ----------
    xg : data.XgData
        The loaded population.
    dev, held : polars.DataFrame
        The development and hold-out shots the cards describe.
    holdout : data.Holdout, optional
        The hold-out ``dev`` and ``held`` were split by; its label is the cards'. Default
        ``data.pair_holdout(xg.matches)``.

    Raises
    ------
    ValueError
        A ``held`` row that the hold-out does not hold out, or a ``dev`` row that it does;
        a hold-out that fails on either frame (:func:`~football_shot_models.data.holdout_mask`).

    Examples
    --------
    >>> card_context(xg, dev, held).competition_seasons  # doctest: +SKIP
    80
    """
    ho = data.pair_holdout(xg.matches) if holdout is None else holdout
    for name, frame, want in (("dev", dev, False), ("held", held, True)):
        if wrong := int((data.holdout_mask(frame, ho, name) != want).sum()):
            raise ValueError(
                f"card_context: {wrong} {name} row(s) are {'not ' * want}held out by the hold-out "
                f"{ho.label!r}; pass the hold-out the frames were split by"
            )
    log = xg.exclusions
    gender = xg.shots["gender"]
    pair = ("competition_id", "season_id")
    sizes = (
        data.match_family(xg.matches)
        .join(xg.matches.select("match_id", *pair), on="match_id")
        .filter(pl.col("family") == "team-selected")
        .group_by(pair)
        .agg(n=pl.len())
    )
    return CardContext(
        manifest_name=xg.manifest.name,
        data_digest=xg.manifest.data_digest,
        source_revision=xg.manifest.source.revision,
        sklearn_version=sklearn.__version__,
        holdout_label=ho.label,
        dev_shots=dev.height,
        dev_matches=dev["match_id"].n_unique(),
        holdout_shots=held.height,
        holdout_matches=held["match_id"].n_unique(),
        competition_seasons=xg.shots.select("competition_id", "season_id").n_unique(),
        male_shots=int((gender == "male").sum()),
        female_shots=int((gender == "female").sum()),
        holdout_female_shots=int((held["gender"] == "female").sum()),
        exclusions=tuple(
            (
                reason,
                int((log["reason"] == reason).sum()),
                int(log.filter(pl.col("reason") == reason)["is_goal"].sum()),
            )
            for reason in data.EXCLUSION_REASONS
        ),
        team_selected_seasons=sizes.height,
        team_selected_short=sizes.filter(pl.col("n") <= 2).height,
    )


TASK: Final[str] = (
    "binary classification: P(goal | shot) for non-penalty shots in periods 1–4 (extra time "
    "included; direct free kicks and corners from open play or set pieces included; direct "
    "corners, own goals and shoot-outs excluded)"
)
FILTER_LABELS: Final[dict[str, str]] = {
    "own_goal": "own goal (Own Goal Against events; not shots)",
    "shootout": "shoot-out (period 5; all are penalties)",
    "penalty": "penalty (in-game)",
    "direct_corner": "direct corner (logged as a shot on outcome)",
    "no_freeze_frame": "no freeze frame",
}
VALIDATION: Final[str] = (
    "Grouped 5-fold cross-validation by `match_id` on dev (`GroupKFold`, no shuffle): no match "
    "has shots in two folds, and every model uses the same folds. **Validation** is the mean "
    "over the out-of-fold test sets (Σ xG and goals are pooled). **Train** is the fit on all of "
    "dev, scored in-sample. **Test** is the hold-out, scored by that fit. The hold-out is one "
    "of the four 2015/16 men's league seasons in the data, with the other three in dev. "
    "It tests transfer to an unseen league in the same season and data spec, not across time "
    "or gender."
)
BENCHMARK_CAVEAT: Final[str] = (
    "StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: "
    "StatsBomb fits it on its own data, which likely includes these matches, and it uses "
    "freeze-frame and other inputs."
)
THRESHOLD_NOTE: Final[str] = (
    "Σ xG and calibration on grouped CV are close to their targets by construction for the "
    "base rate and for an unpenalised logistic regression with an intercept; see the hold-out "
    "column and the reliability table."
)
RELIABILITY_NOTE: Final[str] = (
    "This table stands in for a reliability plot until a reliability chart is added to `viz` (F1)."
)
LICENSING: Final[str] = (
    "Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never "
    "committed or redistributed; this card holds aggregates only."
)
METRIC_LABELS: Final[dict[Metric, str]] = {
    "log_loss": "log loss",
    "brier": "Brier",
    "auc": "ROC AUC",
    "xg_error": "abs(Σ xG − goals) / goals",  # no "|": it would split a table cell
}


def fmt(x: float | None) -> str:
    """A card number: four decimals, or ``n/a``.

    Examples
    --------
    >>> fmt(0.28801), fmt(None)
    ('0.2880', 'n/a')
    """
    return "n/a" if x is None else f"{x:.4f}"


def pct(x: float | None) -> str:
    """A card percentage, or ``n/a``.

    Examples
    --------
    >>> pct(0.0698)
    '6.98%'
    """
    return "n/a" if x is None else f"{x:.2%}"


def _ratio(xg: float, goals: int) -> str:
    return "n/a" if goals == 0 else fmt(xg / goals)


def _measured(metric: Metric, x: float) -> str:
    return pct(x) if metric == "xg_error" else fmt(x)


def _split(ev: Evaluation, split: Split) -> Scores:
    return {"train": ev.train, "cv": ev.cv, "holdout": ev.holdout}[split]


def _segment(dev: pl.DataFrame, oof: FloatArray, column: str, value: str) -> dict[str, float]:
    """Pooled out-of-fold numbers of one segment; empty dict when it has no shots.

    ``cal_slope`` is NaN when the slope is not estimable.
    """
    mask = (dev[column].cast(pl.String) == value).to_numpy()
    if not mask.any():
        return {}
    y = labels(dev)[mask].astype(np.float64)
    p = oof[mask]
    rate = float(y.mean())
    loss = float(log_loss(y, p, labels=[0, 1]))
    # skill against the segment's own goal rate, so segments with different rates compare;
    # a one-class segment has no such reference (its constant log loss is 0)
    one_class = rate in (0.0, 1.0)
    base = 0.0 if one_class else float(log_loss(y, np.full(y.size, rate), labels=[0, 1]))
    return {
        "shots": float(mask.sum()),
        "goals": float(y.sum()),
        "xg": float(p.sum()),
        "log_loss": loss,
        "brier": float(brier_score_loss(y, p)),
        "skill": float("nan") if one_class else 1.0 - loss / base,
        "cal_slope": float("nan") if (slope := calibration(y, p)[1]) is None else slope,
    }


def _segments(dev: pl.DataFrame) -> tuple[tuple[str, str], ...]:
    dtype = dev["body_part"].dtype
    body = tuple(dtype.categories.to_list()) if isinstance(dtype, pl.Enum) else ()
    genders = [(dev["gender"].cast(pl.String) == g).to_numpy() for g in ("female", "male")]
    # a family that is a gender row by construction (women = female) is printed once
    same = {
        v
        for v in data.FAMILIES
        if any(np.array_equal((dev["family"].cast(pl.String) == v).to_numpy(), g) for g in genders)
    }
    return (
        *(("family", v) for v in data.FAMILIES if v not in same),
        *(("gender", v) for v in ("female", "male")),
        *(("body_part", v) for v in body),
    )


def _seg_skill(s: dict[str, float], key: str = "skill") -> str:
    return "n/a" if np.isnan(s[key]) else fmt(s[key])


def _skill(metric: Metric, ev: Evaluation, base: Evaluation, split: Split) -> str:
    value = metric_value(_split(ev, split), metric)
    return fmt(1.0 - value / metric_value(_split(base, split), metric))


def _why(ev: Evaluation, base: Evaluation, benchmark: Evaluation, dev: pl.DataFrame) -> str:
    seg = {v: _segment(dev, ev.oof, "family", v) for v in data.FAMILIES}

    def scored(s: dict[str, float]) -> str:
        return f"{fmt(s['log_loss'])} / {fmt(s['brier'])} (log-loss skill {_seg_skill(s)})"

    def ratio(s: dict[str, float]) -> str:
        return f"Σ xG / goals {_ratio(s['xg'], int(s['goals']))}"

    clauses = []
    if s := seg["league or tournament"]:
        clauses.append(f"{scored(s)} on league or tournament shots ({ratio(s)})")
    if s := seg["team-selected"]:
        clauses.append(
            f"{scored(s)} on team-selected seasons ({pct(s['shots'] / dev.height)} of dev "
            f"shots, goal rate {pct(s['goals'] / s['shots'])}, {ratio(s)})"
        )
    if s := seg["women"]:
        clauses.append(f"{scored(s)} on women's shots ({ratio(s)})")
    listed = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + " and " + clauses[-1]
    return (
        "**Why.** Absolute log loss and Brier depend on the goal rate and on which matches are "
        f"in the sample. The base rate alone scores {fmt(base.cv.log_loss)} / "
        f"{fmt(base.cv.brier)} on the CV shots (goal rate {pct(ev.cv.goals / ev.cv.n)}) and "
        f"{fmt(base.holdout.log_loss)} / {fmt(base.holdout.brier)} on the hold-out "
        f"({pct(ev.holdout.goals / ev.holdout.n)}). By segment, out of fold, this model "
        f"scores {listed}; skill is against each segment's own goal rate. Its log-loss "
        "skill against the base rate is "
        f"{_skill('log_loss', ev, base, 'cv')} on CV and "
        f"{_skill('log_loss', ev, base, 'holdout')} on the hold-out. StatsBomb xG, with "
        f"freeze-frame inputs, scores {fmt(benchmark.cv.log_loss)} / {fmt(benchmark.cv.brier)} "
        "on the CV shots. Thresholds come from a single-season pilot (roadmap); they are "
        "reported, not tuned to."
    )


def _fold_error(ev: Evaluation) -> str:
    mean, worst = fold_xg_error(ev.folds)
    return f"{pct(mean)} (max {pct(worst)})"


def metrics_table(ev: Evaluation, base: Evaluation) -> list[str]:
    """The card's metrics table: train, validation (CV mean ± fold SD) and test columns.

    Skill is against ``base`` (the base rate) on the same split.

    Examples
    --------
    >>> metrics_table(gbm, base)[0]  # doctest: +SKIP
    '| Metric | Train | Validation | Test |'
    """

    def sd(metric: Metric) -> str:
        return f" ± {fmt(fold_sd(ev.folds, metric))}"

    splits = (ev.train, ev.cv, ev.holdout)
    names: tuple[Split, ...] = ("train", "cv", "holdout")
    rows = [
        ("Log loss", [fmt(ev.train.log_loss), fmt(ev.cv.log_loss) + sd("log_loss"),
                      fmt(ev.holdout.log_loss)]),
        ("Brier score", [fmt(ev.train.brier), fmt(ev.cv.brier) + sd("brier"),
                         fmt(ev.holdout.brier)]),
        ("ROC AUC", [fmt(ev.train.auc), fmt(ev.cv.auc) + sd("auc"), fmt(ev.holdout.auc)]),
        ("Log-loss skill", [_skill("log_loss", ev, base, s) for s in names]),
        ("Brier skill", [_skill("brier", ev, base, s) for s in names]),
        ("Calibration intercept", [fmt(s.cal_intercept) for s in splits]),
        ("Calibration slope", [fmt(s.cal_slope) for s in splits]),
        ("Σ xG", [fmt(s.xg) for s in splits]),
        ("Goals", [f"{s.goals:,}" for s in splits]),
        (METRIC_LABELS["xg_error"], [pct(s.xg_error) for s in splits]),
        (f"{METRIC_LABELS['xg_error']}, fold mean (max)", ["n/a", _fold_error(ev), "n/a"]),
    ]  # fmt: skip
    return [
        "| Metric | Train | Validation | Test |",
        "|---|---|---|---|",
        *(f"| {label} | {' | '.join(cells)} |" for label, cells in rows),
    ]


def _signed(x: float) -> str:
    """A gain with its sign, kept unless the value is exactly 0: -0.000004 prints ``-0.0000``."""
    return "+0.0000" if x == 0.0 else f"{x:+.4f}"


def gain_cell(g: GainCI) -> str:
    """A gain and its interval as a card cell.

    Examples
    --------
    >>> gain_cell(GainCI(0.0035, 0.0028, -0.000004, 2000, 9))
    '+0.0035 [+0.0028, -0.0000]'
    """
    return f"{_signed(g.gain)} [{_signed(g.low)}, {_signed(g.high)}]"


def side(g: GainCI) -> str:
    """Where a gain's interval lies: ``"lies above 0"``, ``"lies below 0"`` or ``"includes 0"``.

    A bound equal to 0 counts as including 0.

    Raises
    ------
    ValueError
        A NaN bound.

    Examples
    --------
    >>> side(GainCI(0.0014, -0.0005, 0.0034, 2000, 380))
    'includes 0'
    >>> side(GainCI(0.0035, 0.0028, 0.0043, 2000, 1500))
    'lies above 0'
    """
    if math.isnan(g.low) or math.isnan(g.high):
        raise ValueError(f"gain interval has a NaN bound: [{g.low}, {g.high}]")
    return "lies above 0" if g.low > 0 else "lies below 0" if g.high < 0 else "includes 0"


def checks_table(checks: Sequence[Check], title: str, note: str) -> list[str]:
    """A milestone checks section: heading, note and one row per check; ``[]`` without checks.

    Examples
    --------
    >>> lines = checks_table([Check("log loss <= 0.275", "0.2771", "missed")], "Checks", "Note.")
    >>> lines[0], lines[-2]
    ('### Checks', '| log loss <= 0.275 | 0.2771 | missed |')
    >>> checks_table([], "Checks (MS4)", "Note.")
    []
    """
    if not checks:
        return []
    return [
        f"### {title}",
        "",
        note,
        "",
        "| Target | Measured | Result |",
        "|---|---|---|",
        *(f"| {c.target} | {c.measured} | {c.result} |" for c in checks),
        "",
    ]


def _gain_section(comparisons: Sequence[Comparison]) -> list[str]:
    return [
        "### Log-loss gain (95% match-bootstrap CI)",
        "",
        "Gain = log loss of the reference minus this model, pooled out of fold over all dev "
        "shots (CV) and over the hold-out; positive means this model is better. The 95% "
        f"interval is the percentile interval of {N_BOOT:,} resamples of matches (seed {SEED}) "
        "for the fixed fold models: it covers which matches were sampled, not refitting.",
        "",
        "| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |",
        "|---|---|---|",
        *(f"| {c.label} | {gain_cell(c.cv)} | {gain_cell(c.holdout)} |" for c in comparisons),
        "",
    ]


THRESHOLDS_TITLE: Final[str] = "MS2 thresholds (retired; see the headline model's MS3 checks)"


def _thresholds_section(
    ev: Evaluation,
    base: Evaluation,
    benchmark: Evaluation,
    dev: pl.DataFrame,
    thresholds: Sequence[Threshold],
) -> list[str]:
    if not thresholds:
        return []
    cv, held = check_thresholds(ev.cv, thresholds), check_thresholds(ev.holdout, thresholds)
    lines = [
        f"### {THRESHOLDS_TITLE}",
        "",
        "| Threshold | CV | Hold-out | CV result |",
        "|---|---|---|---|",
    ]
    for c, h in zip(cv, held, strict=True):
        t = c.threshold
        bound = pct(t.value) if t.metric == "xg_error" else f"{t.value:g}"
        label = f"{METRIC_LABELS[t.metric]} {t.op} {bound}"
        result = "met" if c.passed else "missed"
        cv_cell = _measured(t.metric, c.measured)
        if t.metric == "xg_error":  # the pooled value hides fold errors of opposite sign
            cv_cell += f" pooled; fold mean {_fold_error(ev)}"
        lines.append(f"| {label} | {cv_cell} | {_measured(t.metric, h.measured)} | {result} |")
    lines += ["", THRESHOLD_NOTE]
    if not all(c.passed for c in cv):
        lines += ["", _why(ev, base, benchmark, dev)]
    return [*lines, ""]


def _segments_section(ev: Evaluation, dev: pl.DataFrame) -> list[str]:
    lines = [
        "### Segments",
        "",
        "Pooled out of fold. Log-loss skill is `1 − log loss / log loss of a constant at the "
        "segment's own goal rate`, so segments with different goal rates can be compared; "
        "`n/a` for a segment with one class.",
        "",
        "| Segment | Shots | Goals | Goal rate | Σ xG | Σ xG / goals | Log loss | Brier "
        "| Log-loss skill | Cal. slope (pooled OOF) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for column, value in _segments(dev):
        if not (s := _segment(dev, ev.oof, column, value)):
            continue
        shots, goals = int(s["shots"]), int(s["goals"])
        lines.append(
            f"| {column} = {value} | {shots:,} | {goals:,} | {pct(goals / shots)} | "
            f"{fmt(s['xg'])} | {_ratio(s['xg'], goals)} | {fmt(s['log_loss'])} | "
            f"{fmt(s['brier'])} | {_seg_skill(s)} | {_seg_skill(s, 'cal_slope')} |"
        )
    return lines


def render_card(
    ev: Evaluation,
    text: CardText,
    *,
    evaluations: Sequence[Evaluation],
    benchmark: Evaluation,
    dev: pl.DataFrame,
    context: CardContext,
    thresholds: Sequence[Threshold] = THRESHOLDS,
    comparisons: Sequence[Comparison] = (),
    verdict: str | None = None,
    checks: Sequence[Check] = (),
) -> str:
    """The model card of one model, following ``docs/model-card-template.md``.

    Parameters
    ----------
    ev : Evaluation
        The model the card is about.
    text : CardText
        That model's own text.
    evaluations : sequence of Evaluation
        Every model on the same folds; the first is the base rate (the skill reference).
    benchmark : Evaluation
        StatsBomb xG.
    dev : polars.DataFrame
        The development shots ``ev`` was evaluated on (for the segments).
    context : CardContext
        Provenance and population counts.
    thresholds : sequence of Threshold, default THRESHOLDS
        The retired MS2 targets; empty leaves out their section (``THRESHOLDS_TITLE``).
    comparisons : sequence of Comparison, default ()
        Log-loss gains of this model over others; non-empty adds the "Log-loss gain" section
        after the metrics table.
    verdict : str, optional
        The overview sentence after the role; replaces "This is the best ..." (the headline
        model's verdict, :func:`football_shot_models.models.verdict`).
    checks : sequence of Check, default ()
        Non-empty adds the ``text.checks_title`` section where the thresholds go.

    Returns
    -------
    str
        Markdown, with no date or time, so a rerun gives identical bytes.

    Examples
    --------
    >>> card = render_card(ev, text, evaluations=evs, benchmark=b, dev=dev,
    ...                    context=c)  # doctest: +SKIP
    >>> card.splitlines()[0]  # doctest: +SKIP
    '# Model card: Base rate'
    """
    base = evaluations[0]
    top = best(evaluations)
    label = text.best_label
    if verdict is None:
        verdict = (
            f"This is the best {label} (lowest grouped-CV log loss)."
            if top is ev
            else f"Not the best {label}; that is {top.spec.title}."
        )
    c = context
    filters = [
        f"| {FILTER_LABELS[reason]} | {shots:,} | {goals:,} |"
        for reason, shots, goals in c.exclusions
    ]
    benchmarks = [
        "| Model | CV log loss | CV Brier | CV AUC | CV cal. slope | Hold-out log loss "
        "| Hold-out Brier | Hold-out AUC | Hold-out cal. slope |",
        "|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {e.spec.title} | {fmt(e.cv.log_loss)} | {fmt(e.cv.brier)} | {fmt(e.cv.auc)} | "
            f"{fmt(e.cv.cal_slope)} | {fmt(e.holdout.log_loss)} | {fmt(e.holdout.brier)} | "
            f"{fmt(e.holdout.auc)} | {fmt(e.holdout.cal_slope)} |"
            for e in (*evaluations, benchmark)
        ),
    ]
    rel = reliability(labels(dev), ev.oof)
    reliability_rows = [f"| {b} | {n:,} | {pct(m)} | {pct(g)} |" for b, n, m, g in rel.iter_rows()]
    lines = [
        f"# Model card: {ev.spec.title}",
        "",
        "## Overview",
        f"- **Name / version:** `{ev.spec.name}` v{len(text.changelog)}",
        f"- **Task:** {TASK}",
        "- **Status:** experimental",
        f"- {text.role} {verdict}",
        "",
        "## Intended use",
        text.intended_use,
        "",
        "## Data",
        f"- **Source(s):** StatsBomb open data, revision `{c.source_revision}`; snapshot "
        f"`{c.manifest_name}`, data digest `{c.data_digest}`",
        f"- **Competitions & seasons:** {c.competition_seasons:,} competition-seasons; hold-out "
        f"{c.holdout_label}",
        f"- **Rows:** train and validation (dev) {c.dev_shots:,} shots in {c.dev_matches:,} "
        f"matches / test (hold-out) {c.holdout_shots:,} shots in {c.holdout_matches:,} "
        f"matches; men {c.male_shots:,} shots, women {c.female_shots:,}",
        f"- **Families (segments):** women; team-selected, where one team plays in at least half "
        f"of a competition-season's matches ({c.team_selected_seasons:,} competition-seasons, "
        f"{c.team_selected_short:,} of them with one or two matches, mostly single finals, "
        "where both teams qualify trivially); league or tournament, the rest",
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
        *text.features,
        "",
        "## Model",
        *text.model(ev, c),
        "",
        "## Validation design",
        VALIDATION,
        "",
        "## Metrics",
        *metrics_table(ev, base),
        "",
        "Skill is `1 − metric / base-rate metric` on the same shots and column.",
        "",
        *(_gain_section(comparisons) if comparisons else []),
        *_thresholds_section(ev, base, benchmark, dev, thresholds),
        *checks_table(checks, text.checks_title, text.checks_note),
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
        *(f"- {item}" for item in text.limitations),
        "",
        "## Data licensing",
        LICENSING,
        "",
        "## Changelog",
        *(f"- {item}" for item in text.changelog),
    ]
    return "\n".join(lines) + "\n"
