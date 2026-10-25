"""M1.7: the xG v1 model card and the README block, every number computed from the v1 run.

The v1 run (:func:`football_shot_models.v1.run`) collects everything it evaluated in one
:class:`V1Run`; :func:`v1_card` renders the card of record ``xg-v1`` from it and
:func:`summary_block` the README block that lists every model next to the legacy v0. No
WhoScored row, id, team, player, date or file name reaches either: they carry the snapshot's
name and data digest (the pin), counts and metrics.

**Misses carry a computed reason.** When the headline misses the test log-loss bound or the
"at least as good as v0" row, :func:`miss_reason` prints two labelled counterfactual refits
next to each other: *add* (the best LR plus Opta big chance alone) and *remove* (v0's own
inputs refitted on train without it). The line reports both numbers and whether each meets
the bound; it names big chance as the reason only when both refits agree (adding it meets
the bound, removing it misses), and otherwise says the run does not isolate the reason. The
card and the README block also print the headline's gain over the *remove* refit, the legacy
information set without big chance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from football_core import schema
from football_core.io import whoscored

from football_shot_models import ablation, cards, evaluation, features, gbm, legacy, models
from football_shot_models.cards import fmt, gain_cell, pct, side
from football_shot_models.evaluation import Comparison, Evaluation, ModelSpec
from football_shot_models.metrics import N_BOOT, SEED, Check, GainCI, GroupRow, Scores
from football_shot_models.penalties import IN_GAME, MATCH_XG, SHOOTOUT, PenaltyRow

V1_NAME: Final[str] = "xg-v1"
V1_TITLE: Final[str] = "xG v1 (WhoScored / Opta events)"
CHECKS_TITLE: Final[str] = "Milestone checks (MS4)"
CHANGELOG: Final[tuple[str, ...]] = ("v1 (MS4, M1.6 and M1.7): first version.",)
README_START: Final[str] = "<!-- v1:summary:start -->"
README_END: Final[str] = "<!-- v1:summary:end -->"


@dataclass(frozen=True)
class Coverage:
    """How many test shots v0 scores.

    Attributes
    ----------
    covered, population : int
        Covered test shots, of the v1 test population.
    dropped : int
        Test shots dropped from the v1 population (unresolved assist).
    scored_all, kept_all : int
        Covered test shots, of every kept test shot (before that drop).
    """

    covered: int
    population: int
    dropped: int
    scored_all: int
    kept_all: int


@dataclass(frozen=True)
class Counterfactual:
    """A labelled refit beside the candidates: its test log loss and test gain over v0."""

    label: str
    test_log_loss: float
    gain_over_v0: GainCI


@dataclass(frozen=True, eq=False)
class V1Run:
    """Everything the v1 run evaluated, as the card and the README block print it.

    ``v0_gains`` pairs a spec name with its test gain over v0 on the covered shots: the
    headline, LR distance + angle, the best LR, the GBM and the ablation, each once.
    ``head_over_remove`` is the headline's test gain over the *remove* counterfactual (v0's
    inputs refitted without big chance) on the same shots. ``train_competitions`` names the
    competitions with a train match; ``reliability`` holds the test season's ten bins (bin,
    shots, mean xG, goal rate).
    """

    context: cards.CardContext
    train_seasons: tuple[str, ...]
    test_seasons: tuple[str, ...]
    dropped_assist: int
    baselines: tuple[Evaluation, ...]
    steps: tuple[Evaluation, ...]
    gbm: Evaluation
    best_lr: Evaluation
    head: Evaluation
    ablation: Evaluation
    gbm_gain: Comparison
    gains: tuple[Comparison, ...]
    v0: legacy.LegacyModel
    coverage: Coverage
    v0_scores: Scores
    v0_gains: tuple[tuple[str, GainCI], ...]
    add: Counterfactual
    remove: Counterfactual
    head_over_remove: GainCI
    groups: tuple[GroupRow, ...]
    v0_groups: tuple[GroupRow, ...]
    penalties: tuple[PenaltyRow, ...]
    penalty_losses: tuple[float, float]
    checks: tuple[Check, ...]
    reasons: tuple[str, ...]
    train_competitions: tuple[str, ...]
    reliability: tuple[tuple[int, int, float, float], ...]


def _gain(g: GainCI) -> str:
    return f"gain over v0 {g.gain:+.4f} [{g.low:+.4f}, {g.high:+.4f}]"


def miss_reason(
    row: int, missed: bool, add: Counterfactual, remove: Counterfactual, bound: float
) -> str | None:
    """The computed reason line of a missed check row 1 (log loss) or 2 (gain over v0).

    Row 1 judges each counterfactual's test log loss against ``bound``; row 2 its point gain
    over v0 (>= 0). The line prints both values with "meets it" / "misses it", then one of
    three closing sentences chosen by which of them meets the bound: when adding big chance
    meets it and removing it misses, both refits agree and big chance is named as the reason;
    otherwise the line says the run does not isolate or identify one.

    Returns
    -------
    str or None
        ``None`` when the row is met.

    Raises
    ------
    ValueError
        A row other than 1 or 2.

    Examples
    --------
    >>> g = GainCI(0.0004, -0.0002, 0.0010, 2000, 587)
    >>> add = Counterfactual("add", 0.2611, g)
    >>> rem = Counterfactual("remove", 0.2797, GainCI(-0.0182, -0.0211, -0.0154, 2000, 587))
    >>> line = miss_reason(1, True, add, rem, 0.275)
    >>> line.split(". ")[0]
    'add: test log loss 0.2611 (meets it); remove: test log loss 0.2797 (misses it)'
    >>> line.split(". ")[1].split(":")[0]
    'Both labelled counterfactual refits agree that q214 accounts for the gap'
    >>> miss_reason(1, False, add, rem, 0.275) is None
    True
    """
    if row not in (1, 2):
        raise ValueError(f"miss_reason covers check rows 1 and 2, got {row}")
    if not missed:
        return None
    if row == 1:
        meets = [c.test_log_loss <= bound for c in (add, remove)]
        shown = [f"test log loss {c.test_log_loss:.4f}" for c in (add, remove)]
    else:
        meets = [c.gain_over_v0.gain >= 0.0 for c in (add, remove)]
        shown = [_gain(c.gain_over_v0) for c in (add, remove)]
    word = ["meets it" if ok else "misses it" for ok in meets]
    line = f"{add.label}: {shown[0]} ({word[0]}); {remove.label}: {shown[1]} ({word[1]})."
    q = f"q{whoscored.BIG_CHANCE}"
    if meets == [True, False]:
        return line + (
            f" Both labelled counterfactual refits agree that {q} accounts for the gap: adding "
            f"it alone meets the bound, and removing it from v0's inputs misses it."
        )
    if meets[0]:
        return line + f" Removing {q} does not reproduce the miss: this run does not isolate it."
    return line + f" Adding {q} does not meet the bound: this run does not identify the reason."


# ---- card text -------------------------------------------------------------------------------


def _levels(categories: Sequence[str]) -> str:
    return ", ".join(c.replace("_", " ") for c in categories)


def _codes(mapping: Mapping[int, str], skip: str = "") -> str:
    return ", ".join(f"q{k} {v.replace('_', ' ')}" for k, v in mapping.items() if v != skip)


_BOX: Final[str] = f"y < {features.BOX_RIGHT_Y:g} or > {features.BOX_LEFT_Y:g}"
FEATURE_ROWS: Final[dict[str, str]] = {
    **{f: ablation.FEATURE_ROWS[f] for f in ablation.SPLINES if f != "body_part"},
    "body_part": f"| `body_part` | Indicators for {_levels(schema.BODY_PARTS[1:])}; reference "
    f"{_levels(schema.BODY_PARTS[:1])} | Opta {_codes(whoscored.BODY_PARTS)} |",
    "phase": f"| `phase` | Indicators for {_levels(features.PHASES[1:])}; reference open play. "
    f"Direct free kicks wide of the box ({_BOX}) join free kick | Opta pattern "
    f"{_codes(whoscored.PATTERNS, skip='penalty')} (`situation`), `y` |",
    "assist": f"| `assist` | Indicators for the assisting event: {_levels(features.ASSISTS[1:])}; "
    "reference none. No cutback level: the archive has no pull-back qualifier | Opta "
    f"q{whoscored.RELATED} related event, its q{whoscored.CROSS} cross and "
    f"q{whoscored.THROUGH_BALL} through ball (`assist_type`) |",
    "one_on_one": f"| `one_on_one` | Opta 1-on-1 tag (annotator) | Opta q{whoscored.ONE_ON_ONE} |",
    "big_chance": "| `big_chance` | Opta big chance, an analyst's label of the chance: the "
    f"labelled ablation only | Opta q{whoscored.BIG_CHANCE} |",
}
INTENDED_USE: Final[str] = (
    "xG for WhoScored / Opta shot events: P(goal) of a non-penalty shot from its location, "
    "body part, phase of play, assisting event and Opta's 1-on-1 tag. For research and "
    "opposition analysis on this archive, and for comparing with the legacy v0 on the same test "
    "shots; not for rating players or teams without that comparison, not for penalties (a "
    "separate probability, below), and not for other providers' data without re-fitting."
)
LICENSING: Final[str] = (
    "Data: a private WhoScored / Opta archive, never committed or redistributed; this card holds "
    "counts and metrics only."
)
FILTER_LABELS: Final[dict[str, str]] = {
    "own_goal": f"own goal (Opta q{whoscored.OWN_GOAL})",
    "shootout": "shoot-out (period 5)",
    "penalty": "penalty (in-game; the penalty probability below)",
    "direct_corner": f"direct corner (Opta q{whoscored.DIRECT_CORNER})",
    "no_freeze_frame": "no freeze frame",
}


def _ci(ci: tuple[float, float] | None, f: str) -> str:
    return "n/a" if ci is None else f"[{ci[0]:{f}}, {ci[1]:{f}}]"


def _excludes(ci: tuple[float, float] | None, value: float) -> bool:
    return ci is not None and not ci[0] <= value <= ci[1]


def _is_gbm(ev: Evaluation) -> bool:
    return isinstance(ev.spec, ModelSpec) and ev.spec.coefficients is gbm.gbm_settings


def _model(ev: Evaluation, c: cards.CardContext) -> list[str]:
    if not _is_gbm(ev):
        return [
            "Unpenalised logistic regression (`ablation.make_lr`: standardised inputs, "
            f"Newton-Cholesky), no class weights. scikit-learn {c.sklearn_version}. "
            "Coefficients of the fit on all of train:",
            "",
            "| Term | Standardised | Natural units |",
            "|---|---|---|",
            *(f"| {term} | {fmt(std)} | {fmt(nat)} |" for term, std, nat in ev.coefficients),
        ]
    g, s = models.gbm_grid(ev), {t: v for t, v, _ in ev.coefficients}
    signs = [(t.removeprefix("monotone:"), v) for t, v in s.items() if t[:9] == "monotone:"]
    leaves, iters, grid = int(s["max_leaf_nodes"]), int(s["iterations"]), g["leaf_grid"]
    edges = [
        f"The chosen leaf count ({leaves}) is the grid's {word} ({', '.join(map(str, grid))})."
        for word, hit in (("smallest", leaves == min(grid)), ("largest", leaves == max(grid)))
        if hit
    ][:1]
    if iters == g["max_iter"]:
        edges.append(f"The chosen iteration count ({iters}) is the grid's largest ({iters}).")
    return [
        "Monotone gradient boosting (`gbm.MonotoneGBM`, scikit-learn "
        f"{c.sklearn_version}), tuned inside each fit by {g['inner_splits']}-fold `GroupKFold` "
        f"by match on its training rows (`max_leaf_nodes` in {grid}; {g['step']}, "
        f"{2 * g['step']}, ..., {g['max_iter']} iterations) and Platt-scaled on that inner "
        "out-of-fold score; the test season never reaches the tuning.",
        "",
        "Monotone constraints (every other column is free): "
        + "; ".join(
            f"`{col}` {'non-increasing' if v < 0 else 'non-decreasing'}"
            + (f" ({models.MONOTONE_REASONS[col]})" if col in models.MONOTONE_REASONS else "")
            for col, v in signs
        )
        + ".",
        "",
        "| Setting of the fit on all of train | Value |",
        "|---|---|",
        f"| max_leaf_nodes | {leaves} |",
        f"| iterations | {iters} |",
        f"| Platt intercept | {s['platt_intercept']:+.4f} |",
        f"| Platt slope | {s['platt_slope']:.4f} |",
        *(["", *edges] if edges else []),
    ]


def _gender_line(c: cards.CardContext, feats: Mapping[str, None]) -> str:
    total = c.male_shots + c.female_shots
    if c.female_shots == 0:
        return f"`gender` is left out: 0 of {total:,} shots are women's, so it is one value."
    used = "is" if "gender" in feats else "is not"
    return f"`gender` {used} a feature; {c.female_shots:,} of {total:,} shots are women's."


def _evaluations(r: V1Run) -> tuple[Evaluation, ...]:
    return (*r.baselines, *r.steps, r.gbm, r.ablation)


def _title(r: V1Run, ev: Evaluation) -> str:
    return f"**{ev.spec.title}**" if ev is r.head else ev.spec.title


def _signed(s: Scores) -> str:
    return f"{s.xg_signed_error:+.2%}"


def _comp_rows(r: V1Run) -> list[str]:
    v0 = {g.key: g for g in r.v0_groups}
    rows = []
    for g in r.groups:
        o = v0.get(g.key)
        err = "n/a" if g.signed_error is None else f"{g.signed_error:+.2%}"
        v0_err = "n/a" if o is None or o.signed_error is None else f"{o.signed_error:+.2%}"
        rows.append(
            f"| {g.key} | {g.shots:,} | {g.goals:,} | {g.xg:.1f} | {err} {_ci(g.error_ci, '+.2%')}"
            f" | {fmt(g.cal_slope)} {_ci(g.slope_ci, '.3f')} | {v0_err} "
            f"{_ci(None if o is None else o.error_ci, '+.2%')} |"
        )
    k = sum(_excludes(g.error_ci, 0.0) for g in r.groups)
    j = sum(_excludes(g.slope_ci, 1.0) for g in r.groups)
    n = len(r.groups)
    return [
        "| Competition | Shots | Goals | Σ xG | Σ xG error, signed [95% CI] | Cal. slope [95% CI] "
        "| v0 Σ xG error, signed [95% CI] |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"{k} of {n} signed-error intervals exclude 0; {j} of {n} slope intervals exclude 1.",
    ]


def _penalty_lines(r: V1Run) -> list[str]:
    by = {row.label: row for row in r.penalties}
    dev, held = by[f"{IN_GAME}: dev"].rate, by[f"{IN_GAME}: hold-out"].rate
    shoot = by.get(f"{SHOOTOUT}: all")
    lines = [
        "One probability for every in-game penalty: the train-season Beta-binomial posterior "
        f"(Jeffreys prior), {dev.mean:.4f} [95% interval {dev.low:.4f}, {dev.high:.4f}] from "
        f"{dev.n:,} kicks; v0's constant `penalty_xg` is {r.v0.penalty_xg:.4f}. The test season "
        f"has {held.n:,} in-game kicks and {held.goals:,} goals; constant log loss on the test "
        f"kicks: dev posterior {r.penalty_losses[0]:.4f}, v0 `penalty_xg` "
        f"{r.penalty_losses[1]:.4f}.",
        "",
        MATCH_XG,
    ]
    if shoot is not None:
        lines += ["", f"{shoot.rate.n:,} shoot-out kicks are reported apart, never pooled."]
    return lines


def v1_card(r: V1Run) -> str:
    """The ``xg-v1`` model card: every number from ``r``, no date, id or row.

    Examples
    --------
    >>> v1_card(run).splitlines()[0]  # doctest: +SKIP
    '# Model card: xG v1 (WhoScored / Opta events)'
    """
    c, head, cov, v0 = r.context, r.head, r.coverage, r.v0_scores
    base = r.baselines[0]
    test_comps = [g.key for g in r.groups]
    only = [
        f"{', '.join(names)} only in {side}"
        for side, names in (
            ("train", [k for k in r.train_competitions if k not in test_comps]),
            ("test", [k for k in test_comps if k not in r.train_competitions]),
        )
        if names
    ]
    titles = {ev.spec.name: ev.spec.title for ev in _evaluations(r)}
    feats = dict.fromkeys(f for ev in (*r.steps, r.gbm) if isinstance(ev.spec, ModelSpec)
                          for f in ev.spec.features)  # fmt: skip
    exclusions = [
        f"| {FILTER_LABELS[reason]} | n/a (no frames) | n/a |"
        if reason == "no_freeze_frame"
        else f"| {FILTER_LABELS[reason]} | {shots:,} | {goals:,} |"
        for reason, shots, goals in c.exclusions
    ]
    models_rows = [
        f"| {_title(r, ev)} | {fmt(ev.cv.log_loss)} | {fmt(ev.holdout.log_loss)} | "
        f"{fmt(ev.holdout.brier)} | {fmt(ev.holdout.auc)} | {fmt(ev.holdout.cal_slope)} | "
        f"{_signed(ev.holdout)} |"
        for ev in _evaluations(r)
    ]
    abl, best_lr = r.ablation, r.best_lr
    abl_gain = r.gains[-1]
    lines = [
        f"# Model card: {V1_TITLE}",
        "",
        "## Overview",
        f"- **Name / version:** `{V1_NAME}` v{len(CHANGELOG)}",
        f"- **Task:** {cards.TASK}",
        "- **Status:** experimental",
        f"- M1.6 / M1.7: the WhoScored xG model of record. Headline model: {head.spec.title}. "
        f"{models.verdict(head, head, r.gbm_gain, split='test')}",
        "",
        "## Intended use",
        INTENDED_USE,
        "",
        "## What it is not",
        "- Not a StatsBomb model: it is fitted and tested on WhoScored / Opta events only, and "
        "its numbers are not comparable with the StatsBomb cards.",
        "- No freeze frames: it sees no keeper or defender positions at the strike.",
        f"- It does not read Opta big chance (q{whoscored.BIG_CHANCE}); {legacy.V0_TITLE} and the "
        "labelled ablation do.",
        (
            f"- Men's competitions only ({c.male_shots:,} shots, 0 women's): "
            if c.female_shots == 0
            else f"- Men's and women's competitions ({c.male_shots:,} men's shots, "
            f"{c.female_shots:,} women's): "
        )
        + f"{len(r.train_competitions)} in train, {len(test_comps)} in test"
        + "".join(f"; {x}" for x in only)
        + ".",
        f"- One train season ({', '.join(r.train_seasons)}: {c.dev_matches:,} matches) and the "
        f"first {c.holdout_matches:,} test matches ({c.holdout_label}); not a multi-season model.",
        f"- The v0 comparison is test-only: v0 was fitted on {r.v0.trained_shots:,} shots before "
        "the first test match (checked), so it has no CV row.",
        "- No provider xG benchmark: v0 is the benchmark.",
        "",
        "## Data",
        f"- **Source(s):** private WhoScored / Opta event archive, source revision "
        f"`{c.source_revision}`; snapshot `{c.manifest_name}`, data digest `{c.data_digest}`; "
        f"v0 file SHA-256 `{r.v0.sha256[:16]}…`",
        f"- **Competitions & seasons:** {c.competition_seasons:,} competition-seasons; train "
        f"{', '.join(r.train_seasons)}, test {', '.join(r.test_seasons)}",
        f"- **Counts:** train {c.dev_shots:,} shots ({head.cv.goals:,} goals) in "
        f"{c.dev_matches:,} matches / test {c.holdout_shots:,} shots ({head.holdout.goals:,} "
        f"goals) in {c.holdout_matches:,} matches",
        "- **Filters:** excluded shots, by first matching reason.",
        "",
        "| Filter | Shots | Goals |",
        "|---|---|---|",
        *exclusions,
        "",
        f"{r.dropped_assist:,} kept shots whose assisting event (Opta q{whoscored.RELATED}) is "
        "unresolved (a null `assist_type`) are dropped for every model, before the split.",
        "",
        f"v0 covers {cov.covered:,} of {cov.population:,} test shots in the v1 population "
        f"({cov.dropped:,} dropped for unresolved assist, q{whoscored.RELATED}; v0 scores "
        f"{cov.scored_all:,} of {cov.kept_all:,} kept test shots).",
        "",
        "## Target",
        f"`is_goal`: the ball went in. Own goals (Opta q{whoscored.OWN_GOAL}) are excluded, so "
        "every goal is the shooter's.",
        "",
        "## Split",
        f"Train: season(s) {', '.join(r.train_seasons)}; test: {', '.join(r.test_seasons)}. "
        "Every train match precedes every test match (checked). "
        f"Grouped {evaluation.N_SPLITS}-fold CV by `match_id` "
        "on train only (**Validation**); **Train** is the fit on all of train, scored "
        "in-sample; **Test** is the test season, scored once by that fit and never used to "
        "choose. Any tuning (the GBM's grid, Platt) runs inside each fit on its training rows.",
        "",
        "## Features",
        "| Feature | Definition | Source field(s) |",
        "|---|---|---|",
        *(FEATURE_ROWS[f] for f in (*feats, "big_chance") if f in FEATURE_ROWS),
        "",
        _gender_line(c, feats),
        "",
        "## Model",
        *_model(head, c),
        "",
        "## Metrics",
        *cards.metrics_table(head, base),
        "",
        "Skill is `1 − metric / base-rate metric` on the same shots and column.",
        "",
        "## Models",
        "Every candidate on the same folds and test shots; the headline in bold. "
        f"v0 is scored on the {cov.covered:,} test shots it covers.",
        "",
        "| Model | CV log loss | Test log loss | Test Brier | Test AUC | Test cal. slope "
        "| Test Σ xG error (signed) |",
        "|---|---|---|---|---|---|---|",
        *models_rows,
        f"| {legacy.V0_TITLE} | n/a (test-only benchmark) | {fmt(v0.log_loss)} | "
        f"{fmt(v0.brier)} | {fmt(v0.auc)} | {fmt(v0.cal_slope)} | {_signed(v0)} |",
        "",
        "## Gains",
        "Gain = log loss of the reference minus the model's; positive means the model is "
        f"better. 95% percentile intervals of {N_BOOT:,} match resamples (seed {SEED}).",
        "",
        "| Comparison | CV gain [95% CI] | Test gain [95% CI] |",
        "|---|---|---|",
        *(f"| {g.label} | {gain_cell(g.cv)} | {gain_cell(g.holdout)} |" for g in r.gains),
        "",
        f"Test gain over {legacy.V0_TITLE} on the {cov.covered:,} covered test shots:",
        "",
        "| Model | Test gain over v0 [95% CI] | Interval |",
        "|---|---|---|",
        *(f"| {titles[n]} | {gain_cell(g)} | {side(g)} |" for n, g in r.v0_gains),
        f"| {r.remove.label} (test only, counterfactual) | {gain_cell(r.remove.gain_over_v0)} | "
        f"{side(r.remove.gain_over_v0)} |",
        f"| {head.spec.title} over {r.remove.label} (test only, counterfactual reference in "
        f"place of v0) | {gain_cell(r.head_over_remove)} | {side(r.head_over_remove)} |",
        "",
        "## Checks",
        *cards.checks_table(
            r.checks,
            CHECKS_TITLE,
            f"Targets from the MS4 brief, judged on the test season ({c.holdout_label}). The "
            "last two rows restate rules the run applies.",
        ),
        *(f"- {line}" for line in r.reasons),
        *([""] if r.reasons else []),
        "## Reliability",
        "Ten equal-count bins of the headline model's test predictions.",
        "",
        "| Bin | Shots | Mean xG | Goal rate |",
        "|---|---|---|---|",
        *(f"| {b} | {n:,} | {pct(m)} | {pct(g)} |" for b, n, m, g in r.reliability),
        "",
        "## Test by competition",
        f"The headline model on each competition's test shots; intervals from {N_BOOT:,} "
        "resamples of that competition's matches. A slope needs "
        f"{models.MIN_SEGMENT_SHOTS:,} shots.",
        "",
        *_comp_rows(r),
        "",
        "## Big-chance ablation",
        f"{abl.spec.title}: CV log loss {fmt(abl.cv.log_loss)}, test {fmt(abl.holdout.log_loss)};"
        f" gain over {best_lr.spec.title} CV {gain_cell(abl_gain.cv)}, test "
        f"{gain_cell(abl_gain.holdout)}. It is never a candidate: its guard is "
        "`features.check_ablation_features`, and `evaluation.best` skips any spec with an "
        "ablation-only feature.",
        "",
        "| Counterfactual (labelled, never a candidate) | Test log loss | Test gain over v0 "
        "[95% CI] |",
        "|---|---|---|",
        *(
            f"| {cf.label} | {fmt(cf.test_log_loss)} | {gain_cell(cf.gain_over_v0)} |"
            for cf in (r.add, r.remove)
        ),
        "",
        "## Penalties",
        *_penalty_lines(r),
        "",
        "## Limitations & known biases",
        f"- One train season and {c.holdout_matches:,} test matches in {len(test_comps)} "
        "competitions: the per-competition intervals above are wide.",
        "- `assist` and `one_on_one` are Opta annotations, read as recorded.",
        "- No score state and no freeze frame: the game situation enters only through `phase`.",
        f"- {r.dropped_assist:,} shots with an unresolved assisting event are left out of every "
        "model, train and test.",
        "- Penalties, own goals, shoot-outs and direct corners are outside the xG population "
        "(see Data).",
        "",
        "## Data licensing",
        LICENSING,
        "",
        "## Changelog",
        *(f"- {item}" for item in CHANGELOG),
    ]
    return "\n".join(lines) + "\n"


def summary_block(r: V1Run) -> str:
    """The README block: every model next to v0 on the test season, then three lines.

    Examples
    --------
    >>> print(summary_block(run))  # doctest: +SKIP
    | Model | CV log loss | Test log loss | Test AUC | ...
    """
    gains = dict(r.v0_gains)
    v0 = r.v0_scores
    head_gain = gains[r.head.spec.name]
    pen = {row.label: row for row in r.penalties}[f"{IN_GAME}: dev"].rate
    c = r.context
    lines = [
        "| Model | CV log loss | Test log loss | Test AUC | Test cal. slope | Test Σ xG error "
        "(signed) | Test gain over v0 [95% CI] |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {_title(r, ev)} | {fmt(ev.cv.log_loss)} | {fmt(ev.holdout.log_loss)} | "
            f"{fmt(ev.holdout.auc)} | {fmt(ev.holdout.cal_slope)} | {_signed(ev.holdout)} | "
            + (gain_cell(gains[ev.spec.name]) if ev.spec.name in gains else "n/a")
            + " |"
            for ev in _evaluations(r)
        ),
        f"| {legacy.V0_TITLE} | n/a (test-only) | {fmt(v0.log_loss)} | {fmt(v0.auc)} | "
        f"{fmt(v0.cal_slope)} | {_signed(v0)} | n/a (reference) |",
        "",
        f"Headline model: {r.head.spec.title}. On the {r.coverage.covered:,} test shots v0 "
        f"covers, its gain over v0 is {gain_cell(head_gain)}, an interval that "
        f"{side(head_gain)}. Over {r.remove.label} (a labelled counterfactual: v0's "
        f"information without big chance) its gain is {gain_cell(r.head_over_remove)}, an "
        f"interval that {side(r.head_over_remove)}.",
        "",
        f"Penalties (in-game): P(goal) {pen.mean:.4f} [95% interval {pen.low:.4f}, "
        f"{pen.high:.4f}] from {pen.n:,} train kicks; constant log loss on the test kicks "
        f"{r.penalty_losses[0]:.4f} (v0 {r.penalty_losses[1]:.4f}).",
        "",
        f"WhoScored test season, {c.holdout_shots:,} shots, goal rate "
        f"{r.head.holdout.goals / r.head.holdout.n:.4f}; not comparable with the StatsBomb block.",
    ]
    return "\n".join(lines) + "\n"
