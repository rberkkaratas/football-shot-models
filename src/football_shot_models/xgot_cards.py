"""Render and write the xGOT cards from an :class:`~football_shot_models.xgot.XgotFit`.

:func:`render` turns one provider's fit into its three cards (LR, GBM, penalty xGOT) as
Markdown, and nothing else: no fit, no write, no print. :func:`run` is fit, render, write and
the printed summary; ``python -m football_shot_models.xgot_cards --provider both`` (``make
xgot``) runs it for both providers after ``make xg-models xg-v1``, so the xG candidates come
from the prediction cache.

This module is render-only for the prediction cache (``cache.RENDER_ONLY``): no fit-side module
imports it, so editing a card's prose leaves every cached prediction warm. The dependency is
one-way, ``xgot_cards`` → ``xgot``.

The WhoScored cards, README block and output hold counts and metrics only (Rule 1).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import polars as pl
from football_core import goalmouth, pitch
from football_core.io import snapshot

from football_shot_models import (
    ablation,
    baselines,
    cards,
    data,
    evaluation,
    features,
    metrics,
    models,
    xgot,
)
from football_shot_models.cards import CardContext, CardText, fmt, gain_cell, signed
from football_shot_models.evaluation import Evaluation, labels
from football_shot_models.metrics import SplitHalf
from football_shot_models.xgot import (
    BAND_SDS,
    ECE_MAX,
    FLOOR_REPS,
    LR_FEATURES,
    MIN_SHOTS,
    PENALTY_FEATURES,
    BoolArray,
    EceStats,
    Provider,
    Reconciliation,
    XgotFit,
    band_table,
    ece_stats,
    judged,
)

StrPath = str | os.PathLike[str]

PENALTY_FOLDS: Final[str] = (
    f"Grouped {evaluation.N_SPLITS}-fold cross-validation by `match_id` on the in-game dev "
    "kicks (their own "
    "folds); the {split} kicks are scored once by the fit on all of them, and the shoot-out "
    "kicks of every season by that fit too. The penalty xGOT reads no xG input."
)
XGOT_MONOTONE_REASONS: Final[Mapping[str, str]] = MappingProxyType({
    features.XG_INPUT: "a higher xG at the same placement never lowers xGOT",
})  # fmt: skip
XGOT_MONOTONE_NOTE: Final[str] = (
    "The constraint holds along the xG input with every other column held fixed. Placement "
    "carries no monotone constraint, so the trees shape the goal rate across the goal mouth and "
    "towards the posts from the data; the calibration-by-placement-band table below gives the "
    "goal rate per band."
)


def ece_lines(
    ev: Evaluation, y_dev: BoolArray, y_held: BoolArray, *, split: str = "hold-out"
) -> list[str]:
    """The ECE section of every xGOT card: a one-row table and the binning sentence.

    Examples
    --------
    >>> ece_lines(head, y_dev, y_held)[0]  # doctest: +SKIP
    '### Expected calibration error'
    """
    s = ece_stats(ev, y_dev, y_held)
    return [
        "### Expected calibration error",
        "",
        f"ECE ({metrics.ECE_BINS} equal-count bins): the shot-weighted mean of |mean prediction "
        "− goal rate| over the bins of the reliability table. CV is "
        "judged on the fold mean, with the worst fold and the pooled out-of-fold value beside "
        f"it; the {split} value has beside it its noise floor, the mean ECE of outcomes drawn "
        f"from its own predictions ({FLOOR_REPS} draws, seed {metrics.SEED}), which a perfectly "
        "calibrated model reaches by chance.",
        "",
        f"| CV fold mean | CV max | CV pooled | {(cap := split.capitalize())} | {cap} floor |",
        "|---|---|---|---|---|",
        f"| {fmt(s.mean)} | {fmt(s.max)} | {fmt(s.pooled)} | {fmt(s.held)} | {fmt(s.floor)} |",
        "",
    ]


# ---- cards ---------------------------------------------------------------------------------

_PLACED: Final[str] = (
    f"clipped to the frame + {goalmouth.GOAL_TOLERANCE:g} m and rounded to "
    f"{features.PLACEMENT_STEP:g} m"
)
_SRC: Final[str] = "`goal_y`, `goal_z` via `pitch.to_goal_frame`; on target only"
FEATURE_ROWS: Final[Mapping[str, str]] = MappingProxyType({
    features.XG_INPUT: "| `xg_logit` | Logit of the shot's cross-fitted xG (the xG input, out "
    "of sample for the row) | xG model |",
    "mouth_y_abs": f"| `mouth_y_abs` | Distance from the goal centre across the goal, m, "
    f"{_PLACED} | {_SRC} |",
    "mouth_z": f"| `mouth_z` | Height above the ground, m, {_PLACED} | {_SRC} |",
    **{
        f"{col}_rcs{j + 1}": f"| `{col}_rcs{j + 1}` | Restricted cubic spline term {j + 1} of "
        f"`{col}`, knots {', '.join(f'{k:g}' for k in knots)} m | `{col}` |"
        for col, knots in (("mouth_y_abs", features.MOUTH_Y_KNOTS),
                           ("mouth_z", features.MOUTH_Z_KNOTS))
        for j in range(len(knots) - 2)
    },
    "far_side": "| `far_side` | +1 far post, −1 near post (the ball's side against the shot's), "
    f"0 for a shot within {features.CENTRE_DEAD_ZONE:g} m of the centre line or a ball at the "
    f"centre | `y`, {_SRC} |",
    "mouth_far": "| `mouth_far` | `far_side` × `mouth_y_abs` | `y`, " + _SRC + " |",
    "angle_x_mouth_y": "| `angle_x_mouth_y` | `angle_rad` × `mouth_y_abs` | `angle_rad`, "
    + _SRC + " |",
    "angle_x_mouth_z": "| `angle_x_mouth_z` | `angle_rad` × `mouth_z` | `angle_rad`, "
    + _SRC + " |",
    "mouth_yz": "| `mouth_yz` | `mouth_y_abs` × `mouth_z` | " + _SRC + " |",
    "mouth_y_in": f"| `mouth_y_in` | `min(mouth_y_abs, {features.INNER_Y:g})`: the distance from "
    "the goal centre up to where a ball starts to touch the post, m | `mouth_y_abs` |",
    "mouth_y_past": "| `mouth_y_past` | `mouth_y_abs − mouth_y_in`: how far past that point "
    "the ball crossed, m | `mouth_y_abs` |",
    "mouth_out": f"| `mouth_out` | 1 when the point was recorded beyond the frame + "
    f"{goalmouth.GOAL_TOLERANCE:g} m (before clipping), else 0 | {_SRC} |",
    "angle_rad": ablation.FEATURE_ROWS["angle_rad"],
})  # fmt: skip
INTENDED_USE: Final[str] = (
    "xGOT (post-shot xG) for on-target shots: P(goal) of a shot that reached the goal frame, "
    "from where it crossed the line and its pre-shot xG. Keeper-neutral (no keeper position), "
    "so goals saved against it credit the keeper's positioning too. For research and as the "
    "input of the MS6 goalkeeper models; not a rating of finishers, and not for other "
    "providers' data without re-fitting."
)
TASK: Final[str] = (
    "binary classification: P(goal | on-target shot) for the on-target shots of the non-penalty "
    "xG population (a goal or a keeper save; own goals are never on target)"
)
PENALTY_TASK: Final[str] = (
    "P(goal | on-target penalty) from placement only; fitted on in-game kicks, shoot-out kicks "
    "scored apart"
)
PENALTY_USE: Final[str] = (
    "Penalty xGOT: P(goal) of an on-target penalty from where it crossed the line, kept apart "
    "from the open-play xGOT. For research and as the penalty input of the MS6 goalkeeper "
    "models; not a rating of takers or keepers on its own."
)
TARGET: Final[str] = (
    "`is_goal` of an on-target shot: the ball went in rather than being saved. Blocked shots "
    "(clearances off the line included), woodwork and misses are not on target (C3 "
    "`on_target`), so they are not in the population."
)
CHANGELOG: Final[tuple[str, ...]] = (
    "v1 (MS5, P2.2–P2.5): first version.",
    "v2 (MS5 assessment): the GBM is monotone in `mouth_y_in` only, with `mouth_y_past` and "
    "`mouth_out` free; the LR reads `mouth_y_past`; calibration by placement band; the "
    "scored-once split's level gap; penalty calibration judged bound by bound.",
    "v3 (MS6): saves recorded where the keeper stopped the ball and projected to the line are "
    "snapped to the 0.1 yd goal-mouth lattice of recorded points (football-core); the GBM's "
    "placement features carry no monotone constraint (chosen by the band rule); the band table "
    "prints the SD; the band rule is a check.",
    f"v4 (MS6, GSAx): matches recorded on the coarse (1 yd) grid leave the xGOT fit (PM E1); a "
    f"band is judged by the band rule only with at least {xgot.MIN_BAND_SHOTS} shots.",
)


def _lr_model(ev: Evaluation, c: CardContext) -> list[str]:
    lr = ablation.make_lr()[-1]
    return [
        f"`make_pipeline(StandardScaler(), LogisticRegression(C={lr.C}, solver={lr.solver!r}, "
        f"tol={lr.tol:g}, max_iter={lr.max_iter}))`: unpenalised logistic regression, no class "
        "weights, no calibration step; the scaler is fitted on the training rows of each fit. "
        f"scikit-learn {c.sklearn_version}.",
        "",
        "Coefficients of the fit on all of dev (natural units: log-odds per unit of the feature).",
        "",
        "| Term | Standardised | Natural units |",
        "|---|---|---|",
        *(f"| {term} | {fmt(std)} | {fmt(nat)} |" for term, std, nat in ev.coefficients),
    ]


def _feature_table(names: Sequence[str]) -> tuple[str, ...]:
    return (
        "| Feature | Definition | Source field(s) |",
        "|---|---|---|",
        *(FEATURE_ROWS[f] for f in names),
    )


def _limits(p: Provider, on: pl.DataFrame, *, flagged: bool = True) -> tuple[str, ...]:
    """Limitations computed from the on-target rows ``on`` (dev and the scored-once split);
    ``flagged`` when the card's models read ``mouth_out``."""
    gy, gz = pitch.to_goal_frame(on["goal_y"].to_numpy(), on["goal_z"].to_numpy())
    clipped = int(features.clipped(gy, gz).sum())
    dead = int(features.dead_zone(on["y"].to_numpy()).sum())
    deflected = on.filter(pl.col("deflected").fill_null(False) & ~pl.col("is_goal")).height
    psxg = on["provider_psxg"].null_count()
    lines = [
        f"{clipped:,} of {on.height:,} on-target shots ({p.split} included) are recorded "
        f"beyond the frame + {goalmouth.GOAL_TOLERANCE:g} m and are clipped to its edge"
        + (
            " and flagged by `mouth_out`, since where they would have crossed the line is not "
            "known; "
            if flagged
            else "; "
        )
        + f"{dead:,} are struck within {features.CENTRE_DEAD_ZONE:g} m of the centre line, so they "
        "have no near or far post.",
        f"`provider_psxg` is null on {psxg:,} of {on.height:,} on-target shots, so there is no "
        "provider xGOT benchmark.",
    ]
    if p.mouth_note is not None:
        lines.append(p.mouth_note.format(deflected=deflected))
    return tuple(lines)


def band_lines(ev: Evaluation, dev: pl.DataFrame) -> list[str]:
    """Calibration by placement band: ``xgot.band_table`` of ``ev.oof`` on ``dev``, rendered.

    Examples
    --------
    >>> band_lines(head, inp.on_dev)[0]  # doctest: +SKIP
    '### Calibration by placement band'
    """
    least = xgot.MIN_BAND_SHOTS  # read when called, like ``xgot.judged``
    rows = _band_rows(band_table(ev.oof, dev))
    chance = math.erfc(BAND_SDS / math.sqrt(2))
    return [
        "### Calibration by placement band",
        "",
        "Dev out-of-fold xGOT by the ball's distance from the goal centre (`mouth_y_abs`, m, "
        f"after rounding); a ball beyond {features.INNER_Y:g} m touches the post. The "
        "reliability table bins by prediction, so it cannot show a placement the model "
        f"misprices. SD is √Σ p(1 − p), the spread of the goal count if xGOT is calibrated; "
        f"*Within {BAND_SDS:g} SD* is the PM's screening rule (MS6), not a proof of "
        f"calibration: a calibrated model leaves a band outside it by chance in about "
        f"{chance:.1%} of cases. The rule is a normal approximation, so a band with fewer than "
        f"{least} shots is printed but not judged (PM, MS6).",
        "",
        *rows,
        "",
    ]


def _band_rows(table: pl.DataFrame) -> list[str]:
    """The header and rows of a :func:`xgot.band_table`, judged by the PM's rule."""
    least, out = xgot.MIN_BAND_SHOTS, []
    for r, ok in zip(table.iter_rows(named=True), judged(table), strict=True):
        gap = f"{r['gap']:+.1f}" if r["shots"] else "n/a"
        within = ("yes" if r["within"] else "no") if ok else f"not judged (n < {least})"
        out.append(
            f"| {r['band']} | {r['shots']:,} | {r['goals']:,} | {fmt(r['goal_rate'])} | "
            f"{fmt(r['mean_xgot'])} | {gap} | {r['sd']:.1f} | {within} |"
        )
    return [
        f"| abs(y) band (m) | Shots | Goals | Goal rate | Mean xGOT | Σ xGOT − goals | SD | "
        f"Within {BAND_SDS:g} SD |",
        "|---|---|---|---|---|---|---|---|",
        *out,
    ]


def held_band_lines(ev: Evaluation, held: pl.DataFrame, split: str) -> list[str]:
    """The band table of ``ev.holdout_p`` on ``held`` (the hold-out or test split), rendered.

    The PM names the hold-out / test splits as the clean evidence (after MS5); check row 12
    judges dev out of fold, so this table is printed beside it with the same rule, not judged
    as a check.

    Examples
    --------
    >>> held_band_lines(head, inp.on_held, "test")[0]  # doctest: +SKIP
    '### Calibration by placement band, test'
    """
    n = held.height
    return [
        f"### Calibration by placement band, {split}",
        "",
        f"The same table on the {n:,} on-target shots of the {split} (the fit on all of dev), "
        f"with the same rule and the same {xgot.MIN_BAND_SHOTS}-shot minimum; printed beside "
        "check row 12, which judges dev out of fold.",
        "",
        *_band_rows(band_table(ev.holdout_p, held)),
        "",
    ]


def band_limit(ev: Evaluation, head: Evaluation, dev: pl.DataFrame) -> str | None:
    """The limitation of a card whose model is not the headline and fails the band rule.

    The band rule judges the headline xGOT only (PM E1): another model's card names its
    failing bands with their counts; None when every band is within or ``ev`` is the headline.

    Examples
    --------
    >>> band_limit(lr, head, inp.on_dev)  # doctest: +SKIP
    'Calibration by placement band: 2 of 7 bands are outside 2 SD ([3, 3.3): ...'
    """
    table = band_table(ev.oof, dev)
    bad = table.filter(~pl.col("within") & judged(table))
    if ev is head or bad.height == 0:
        return None
    listed = "; ".join(
        f"{r['band']}: {r['shots']:,} shots, {r['goals']:,} goals, Σ xGOT − goals "
        f"{signed(r['gap'])} (SD {r['sd']:.1f})"
        for r in bad.iter_rows(named=True)
    )
    return (
        f"Calibration by placement band: {bad.height} of {table.height} bands are outside "
        f"{BAND_SDS:g} SD on dev out of fold ({listed}). This model's placement terms misprice "
        f"them; the headline is {head.spec.title}, which the band rule judges (PM E1)."
    )


def grid_lines(grid: pl.DataFrame) -> list[str]:
    """The coarse-grid matches left out of the xGOT (PM E1 (a)), per competition-season;
    nothing when there is none (WhoScored).

    Examples
    --------
    >>> grid_lines(fit.grid)[0]  # doctest: +SKIP
    '### Coarse-grid matches left out'
    """
    if grid.height == 0:
        return []
    return [
        "### Coarse-grid matches left out",
        "",
        "Matches whose shot locations are all whole yards were recorded on the coarse (1 yd) "
        "grid (football-core's per-match rule), so their end locations are whole yards too and "
        "placement on them is a recording artefact. Their on-target shots are left out of the "
        "xGOT fit and of the goalkeeper models (PM E1 (a), MS6), reason *coarse recording "
        "grid* in the Filters table:",
        "",
        "| Competition | Season | Matches | On-target shots | Goals |",
        "|---|---|---|---|---|",
        *(
            f"| {r['competition_name']} | {r['season_name']} | {r['matches']:,} | "
            f"{r['shots']:,} | {r['goals']:,} |"
            for r in grid.iter_rows(named=True)
        ),
        f"| **Total** | | {int(grid['matches'].sum()):,} | {int(grid['shots'].sum()):,} | "
        f"{int(grid['goals'].sum()):,} |",
        "",
    ]


def _level(ev: Evaluation, split: str) -> str:
    """The scored-once split's level gap, a limitation the ECE does not show."""
    h = ev.holdout
    gap = f"{h.xg_signed_error:+.2%}" if h.goals else "n/a"
    return (
        f"On the {split}, Σ xGOT is {h.xg:,.1f} against {h.goals:,} goals "
        f"({gap}; calibration intercept {fmt(h.cal_intercept)}). A level gap "
        "moves every keeper's goals saved against this xGOT the same way, so a season's "
        "keepers read better or worse as a group."
    )


def _text(
    role: str, feats: tuple[str, ...], model: Callable[[Evaluation, CardContext], Sequence[str]],
    limits: tuple[str, ...], *, task: str = TASK,
) -> CardText:  # fmt: skip
    return CardText(
        role=role, intended_use=INTENDED_USE, features=feats, model=model, limitations=limits,
        changelog=CHANGELOG, task=task, target=TARGET, best_label="xGOT model",
        checks_title=xgot.CHECKS_TITLE,
        checks_note="Bounds from the PM brief (MS5) and the band rule from the PM brief (MS6); "
        "AUC >= 0.80 reads \"about 0.80 or better\" strictly. ECE is judged on the CV fold mean "
        "and the point value of the scored-once split (PM X3).",
        quantity="xGOT",
    )  # fmt: skip


def xgot_context(
    xg: data.XgData,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    recon: Reconciliation,
    *,
    holdout: data.Holdout,
) -> CardContext:
    """``cards.card_context`` with every count taken from the on-target frames passed.

    Examples
    --------
    >>> c = xgot_context(xg, inp.on_dev, inp.on_held, recon, holdout=ho)  # doctest: +SKIP
    >>> c.dev_shots == inp.on_dev.height  # doctest: +SKIP
    True
    """
    both = pl.concat([dev.select("competition_id", "season_id", "gender"),
                      held.select("competition_id", "season_id", "gender")])  # fmt: skip
    return dataclasses.replace(
        cards.card_context(xg, dev, held, holdout),
        competition_seasons=both.select("competition_id", "season_id").n_unique(),
        male_shots=int((both["gender"] == "male").sum()),
        female_shots=int((both["gender"] == "female").sum()),
        exclusions=recon.rows,
    )


def split_half_cell(s: SplitHalf, digits: int = 3) -> str:
    """A split-half r and its interval as a card cell; ``n/a`` when r or a bound is None.

    Examples
    --------
    >>> split_half_cell(SplitHalf(30, 0.4, 0.05, 0.66), 2)
    '+0.40 [+0.05, +0.66]'
    >>> split_half_cell(SplitHalf(3, None, None, None))
    'n/a'
    """
    if s.r is None or s.low is None or s.high is None:
        return "n/a"
    return f"{s.r:+.{digits}f} [{s.low:+.{digits}f}, {s.high:+.{digits}f}]"


def _halves(rows: Sequence[tuple[int, bool, SplitHalf, SplitHalf]]) -> list[str]:
    return [
        "### Shooting goals added (descriptive)",
        "",
        "Shooting goals added (SGA) is xGOT − xG per shot, next to goals − xG; both subtract the "
        "same xG, the xG input above. Blocked and off-target shots count as 0 xGOT. Each "
        "player's matches are ordered by date (then match id) and alternated into two halves, "
        "so a match never spans halves; r is the Pearson correlation across players of the "
        "per-shot value in the two halves, with a 95% interval by Fisher z (n/a under 4 "
        "players). A descriptive measure of shots beating their xG on target, not a "
        "finishing-skill claim.",
        "",
        "| Min shots | Blocked shots | Players | SGA per shot: r [95% CI] | Goals − xG per "
        "shot: r [95% CI] |",
        "|---|---|---|---|---|",
        *(
            f"| {m} | {'in' if blocked else 'left out'} | {a.keys:,} | {split_half_cell(a)} | "
            f"{split_half_cell(b)} |"
            for m, blocked, a, b in rows
        ),
        "",
    ]


def _recon_lines(recon: Reconciliation) -> list[str]:
    return [
        "### Reconciliation with C3",
        "",
        "Every C3 on-target shot is in one row of the Filters table; the scored rows are the "
        "open-play and penalty xGOT predictions, compared with C3 less the rows a PM ruling "
        "leaves out.",
        "",
        "| | Shots | Goals |",
        "|---|---|---|",
        f"| C3 on target | {recon.c3_shots:,} | {recon.c3_goals:,} |",
        *(
            [
                f"| Left out by a PM ruling (coarse-grid matches) | {recon.ruled[0]:,} | "
                f"{recon.ruled[1]:,} |"
            ]
            if recon.ruled[0]
            else []
        ),
        f"| Scored by xGOT | {recon.shots:,} | {recon.goals:,} |",
        f"| Difference | {recon.shots_gap:+.3%} | {recon.goals_gap:+.3%} |",
        "",
        f"Population rows without a prediction: {recon.unscored:,}.",
        "",
    ]


# ---- render and run ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class XgotRun(XgotFit):
    """An :class:`~football_shot_models.xgot.XgotFit` plus ``cards``: card name → the Markdown
    written. Every MS5 ``XgotRun`` attribute keeps its name (``penalty`` is a property)."""

    cards: Mapping[str, str]


def _y(pred: pl.DataFrame, kind: str, split: str) -> BoolArray:
    rows = pred.filter((pl.col("kind") == kind) & (pl.col("split") == split) & pl.col("on_target"))
    y: BoolArray = rows["is_goal"].to_numpy()
    return y


def summary_block(runs: Sequence[XgotFit]) -> str:
    """The README block: every xGOT model per provider, then the headline, penalty and SGA lines.

    Each run's source wording is its own ``setup``, as read when it was fitted.

    Examples
    --------
    >>> print(summary_block([sb_fit]))  # doctest: +SKIP
    | Provider | Model | CV AUC | CV ECE | CV cal. slope | ...
    """
    lines = [
        "| Provider | Model | CV AUC | CV ECE | CV cal. slope | Hold-out / test AUC | "
        "Hold-out / test ECE | Hold-out / test cal. slope |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for r in runs:
        source = r.setup.text.source
        for ev, kind in (*((e, "open_play") for e in r.evaluations), (r.penalty, "penalty")):
            s = ece_stats(ev, _y(r.predictions, kind, "dev"), _y(r.predictions, kind, "holdout"))
            title = f"**{ev.spec.title}**" if ev is r.head else ev.spec.title
            lines.append(
                f"| {source} | {title} | {fmt(ev.cv.auc)} | {fmt(s.mean)} | "
                f"{fmt(ev.cv.cal_slope)} | {fmt(ev.holdout.auc)} | {fmt(s.held)} | "
                f"{fmt(ev.holdout.cal_slope)} |"
            )
        met = sum(c.result == "met" for c in r.checks)
        so = r.shootouts
        sga40 = next(row for row in r.split_half if row[0] == MIN_SHOTS[1] and row[1])
        notes += [
            f"{source}: headline {r.head.spec.title}; xG input {r.xg_input.spec.title} (provider "
            f"headline xG {r.xg_headline.spec.title}); {met} of {len(r.checks)} checks met "
            f"({xgot.CHECKS_MILESTONES}); {r.recon.shots:,} of {r.recon.c3_shots:,} C3 on-target "
            f"shots scored, {r.recon.ruled[0]:,} left out by a PM ruling "
            f"({r.recon.shots_gap:+.3%}), goals {r.recon.goals_gap:+.3%}.",
            "",
            f"{source}, penalty xGOT (placement only, in-game kicks): CV AUC "
            f"{fmt(r.penalty.cv.auc)}, CV cal. slope {fmt(r.penalty.cv.cal_slope)}; shoot-out "
            + ("kicks: none with both outcomes." if so is None else
               f"kicks scored apart: {so.n:,} kicks, {so.goals:,} goals, Σ xGOT {so.xg:.1f}, log "
               f"loss {fmt(so.log_loss)}."),
            "",
            f"{source}, shooting goals added (descriptive), split-half r at >= {sga40[0]} shots, "
            f"{sga40[2].keys:,} players: SGA per shot {split_half_cell(sga40[2])}, goals − xG "
            f"per shot {split_half_cell(sga40[3])}.",
            "",
        ]  # fmt: skip
    return "\n".join([*lines, "", *notes[:-1]]) + "\n"


def _penalty_calibration(pen: Evaluation, s: EceStats, split: str) -> str:
    """The penalty card's calibration line: each open-play bound judged on its own."""
    lo, hi = models.SLOPE_RANGE

    def within(v: float | None) -> str:
        return "not estimable" if v is None else "inside" if lo <= v <= hi else "outside"

    slopes = (pen.cv.cal_slope, pen.holdout.cal_slope)
    inside = all(v is not None and lo <= v <= hi for v in slopes) and s.mean < ECE_MAX
    line = (
        f"Against the open-play bounds, each on its own: CV calibration slope "
        f"{fmt(slopes[0])} ({within(slopes[0])} [{lo:g}, {hi:g}]), {split} slope "
        f"{fmt(slopes[1])} ({within(slopes[1])}), CV ECE {fmt(s.mean)} (fold mean; "
        f"{'below' if s.mean < ECE_MAX else 'not below'} {ECE_MAX:g})."
    )
    if inside:
        return line
    return (
        f"{line} Its probabilities are not calibrated to the open-play standard; as a ranking "
        f"of placements its AUC is {fmt(pen.cv.auc)} on CV and {fmt(pen.holdout.auc)} on the "
        f"{split} (0.5 is no ranking)."
    )


def render(fit: XgotFit) -> Mapping[str, str]:
    """The three cards of one provider's fit, card name → Markdown; pure (no fit, no write).

    Reads ``fit.setup`` and the other fields only, never ``xgot.PROVIDERS``.

    Raises
    ------
    ValueError
        A malformed fit (shape errors from ``xgot.band_table``).

    Examples
    --------
    >>> sorted(render(xgot.fit("statsbomb", cache="data/cache")))  # doctest: +SKIP
    ['xgot-gbm', 'xgot-lr', 'xgot-penalty']
    """
    p, evs, head, kicks = fit.setup, fit.evaluations, fit.head, fit.kicks
    on_dev, on_held = fit.inp.on_dev, fit.inp.on_held
    y_dev, y_held = labels(on_dev), labels(on_held)
    limits = _limits(p, pl.concat([on_dev, on_held]))
    lead = (f"xG input: {fit.xg_input.spec.title}, cross-fitted; provider headline xG: "
            f"{fit.xg_headline.spec.title}.", "")  # fmt: skip
    lr_limits = (*limits, _level(evs[3], p.split))
    if (lr_band := band_limit(evs[3], head, on_dev)) is not None:
        lr_limits = (*lr_limits, lr_band)
    gbm_limits = (*limits, _level(evs[4], p.split))
    if (gbm_band := band_limit(evs[4], head, on_dev)) is not None:
        gbm_limits = (*gbm_limits, gbm_band)
    texts = (
        _text("P2.3: logistic regression of the goal on the logit of the cross-fitted xG, "
              "placement splines, the width past the post's inner edge, angle × placement and "
              "the near / far post.",
              (*lead, *_feature_table(LR_FEATURES)), _lr_model, lr_limits),
        _text("P2.4: gradient boosting on the xG input (monotone), placement, side and angle, "
              "tuned and Platt-calibrated inside each fit by grouped CV.",
              (*lead, *models.gbm_features(evs[4], FEATURE_ROWS, XGOT_MONOTONE_REASONS,
                                           XGOT_MONOTONE_NOTE)),
              models.gbm_model, gbm_limits),
    )  # fmt: skip
    ctx = xgot_context(fit.xg, on_dev, on_held, fit.recon, holdout=fit.holdout)
    written = {
        ev.spec.name: cards.render_card(
            ev, text, evaluations=evs, benchmark=None, dev=on_dev, context=ctx, provider=p.text,
            validation=p.validation, thresholds=(), comparisons=(gain,),
            verdict=models.verdict(ev, head, fit.gbm_gain, split=p.split),
            checks=fit.checks if ev is head else (),
            extra=[*ece_lines(ev, y_dev, y_held, split=p.split), *band_lines(ev, on_dev),
                   *(held_band_lines(ev, on_held, p.split) if ev is evs[4] else ()),
                   *((*_recon_lines(fit.recon), *_halves(fit.split_half)) if ev is head else ()),
                   *grid_lines(fit.grid)],
        )
        for ev, text, gain in ((evs[3], texts[0], fit.far_gain), (evs[4], texts[1], fit.gbm_gain))
    }  # fmt: skip
    pen, k_dev, k_held = fit.penalty, labels(kicks.dev), labels(kicks.held)
    k_shoot = labels(kicks.shootouts)
    calib = _penalty_calibration(pen, ece_stats(pen, k_dev, k_held), p.split)
    shoot = [
        "### Shoot-out kicks (reported apart)", "",
        f"{k_shoot.size:,} on-target shoot-out kicks ({int(k_shoot.sum()):,} goals), scored by "
        "the fit on all in-game dev kicks and never pooled with them: Σ xGOT "
        f"{float(kicks.shootout_p.sum()):.1f}, log loss "
        f"{fmt(None if fit.shootouts is None else fit.shootouts.log_loss)}.", "",
    ]  # fmt: skip
    pen_text = dataclasses.replace(_text(
        "P2.4: penalty xGOT, a logistic regression on placement only; it reads no xG, since "
        "every in-game penalty has the same pre-shot xG.",
        _feature_table(PENALTY_FEATURES), _lr_model,
        (calib, *_limits(p, pl.concat([kicks.dev, kicks.held]), flagged=False)),
        task=PENALTY_TASK,
    ), intended_use=PENALTY_USE)  # fmt: skip
    written[pen.spec.name] = cards.render_card(
        pen, pen_text, evaluations=(fit.penalty_base, pen), benchmark=None, dev=kicks.dev,
        context=xgot_context(fit.xg, kicks.dev, kicks.held, fit.recon, holdout=fit.holdout),
        provider=p.text, validation=PENALTY_FOLDS.format(split=p.split), thresholds=(),
        verdict="Placement-only, reported apart from the open-play xGOT.",
        extra=[*ece_lines(pen, k_dev, k_held, split=p.split), *shoot, *grid_lines(fit.grid)],
    )  # fmt: skip
    return MappingProxyType(written)


def write_cards(cards: Mapping[str, str], docs_dir: StrPath) -> Path:
    """Write each card to ``docs_dir/<name>.md`` (UTF-8, ``\\n`` line ends); return the folder.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     out = write_cards({"xgot-x": "# x\\n"}, tmp)
    ...     (out / "xgot-x.md").read_text(encoding="utf-8")
    '# x\\n'
    """
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, card in cards.items():
        (out / f"{name}.md").write_text(card, encoding="utf-8", newline="\n")
    return out


def run(provider: str, *, docs_dir: StrPath, cache: StrPath | None) -> XgotRun:
    """Fit one provider, render and write its three cards, and print the results.

    Parameters
    ----------
    provider : str
        A key of ``xgot.PROVIDERS``.
    docs_dir : str or path
        Where ``<name>.md`` is written.
    cache : str or path, optional
        Prediction cache root; ``None`` fits everything and writes no cache.

    Returns
    -------
    XgotRun

    Raises
    ------
    ValueError
        As ``xgot.fit``.

    Examples
    --------
    >>> run("statsbomb", docs_dir="docs/models", cache="data/cache").head.spec.name
    ... # doctest: +SKIP
    'xgot-gbm'
    """
    f = xgot.fit(provider, cache=cache)
    written = render(f)
    out = write_cards(written, docs_dir)
    for ev in dict.fromkeys((f.xg_headline, f.xg_input, *f.evaluations, f.penalty)):
        print(baselines.summary_line(ev))
    for g in (f.gbm_gain, f.far_gain):
        print(f"gain | {g.label}: cv {gain_cell(g.cv)} | {f.setup.split} {gain_cell(g.holdout)}")
    print(f"headline: {f.head.spec.name}")
    print(*(f"check | {c.target} | {c.measured} | {c.result}" for c in f.checks), sep="\n")
    print(f"cards: {out}")
    if f.cache is not None:
        print(f"cache: {f.cache.dir} ({len(f.cache.hits)} read, {len(f.cache.written)} written)")
    fields = {field.name: getattr(f, field.name) for field in dataclasses.fields(f)}
    return XgotRun(**fields, cards=written)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 0 even when a check is missed (it reports).

    Examples
    --------
    >>> main(["--provider", "statsbomb"])  # doctest: +SKIP
    0
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.xgot_cards",
        description="Evaluate the xGOT models (MS5, MS6) and write their cards.",
    )
    providers = tuple(xgot.PROVIDERS)
    parser.add_argument("--provider", choices=(*providers, "both"), default="both")
    parser.add_argument("--docs-dir", default="docs/models", help="where the cards go")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--cache", default="data/cache", help="prediction cache root (git-ignored)")
    where.add_argument("--no-cache", action="store_true", help="fit everything, write no cache")
    args = parser.parse_args(argv)
    keys = providers if args.provider == "both" else (args.provider,)
    cache = None if args.no_cache else args.cache
    try:
        runs = [run(k, docs_dir=args.docs_dir, cache=cache) for k in keys]
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    print(summary_block(runs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
