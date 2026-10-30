"""The ``xg-v2`` model card and README block, rendered from a :class:`~.v2.V2Fit`.

Render-only (``cache.RENDER_ONLY``): it fits nothing and no other module imports it, so a
wording fix here keeps the prediction cache warm. Every number comes from the fit; the card
holds counts, metrics and Opta vocabulary only (no id, name, team, date or file).

Run ``python -m football_shot_models.v2_cards`` (``make xg-v2``, after ``make xg-v1``) to
write ``docs/models/xg-v2.md`` and print the README block.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from football_core.io import snapshot, whoscored

from football_shot_models import cards, evaluation, models, release, v1, v2
from football_shot_models.cards import CardContext, fmt, gain_cell, pct
from football_shot_models.evaluation import Evaluation
from football_shot_models.metrics import N_BOOT

StrPath = str | os.PathLike[str]

CHANGELOG: Final = ("v1 (PR1, xg-context): first version.",)
README_START: Final = "<!-- v2:summary:start -->"
README_END: Final = "<!-- v2:summary:end -->"
CARD_FILE: Final = f"{v2.NAME}.md"
FEATURE_ROWS: Final = {
    **release.FEATURE_ROWS,
    "prev_action": "| `prev_action` | Indicators for the last play event before the shot "
    "(stoppages skipped): the shooting team's take-on, regain, earlier shot or other action, the "
    "opponent's clearance, outfield block, keeper action or other; reference pass | Event types "
    f"before the shot in file order; Opta q{whoscored.OUTFIELD_BLOCK} on a previous Save |",
    "rebound": "| `rebound` | The team's previous shot of the period is at most "
    f"{whoscored.REBOUND_S} s earlier | Shot event clocks |",
}
CONTEXT_NAMES: Final = " and ".join(f"`{c}`" for c in v2.CONTEXT_COLUMNS)
# PR1 design history: the context set was chosen by a probe that scored the test season
CHOSEN_ON_TEST: Final = (
    f"The design probe that chose {CONTEXT_NAMES} scored its candidates on the test season, so "
    "the test gain over xG v1 carries selection optimism"
)
VALIDATION: Final = (
    f"Train: season(s) {', '.join(v1.TRAIN_SEASONS)}; test: {', '.join(v1.TEST_SEASONS)}, the "
    "population and split of xG v1. Every train match precedes every test match (checked). "
    f"Grouped {evaluation.N_SPLITS}-fold CV by `match_id` on train only (**Validation**); "
    "**Train** is the fit on all of train, scored in-sample; **Hold-out** below is the test "
    f"season, scored by that fit. {CHOSEN_ON_TEST}: the evidence the test season did not choose "
    "is the CV gain over xG v1 (training season only), in the gain table."
)


def cv_evidence(fit: v2.V2Fit) -> str:
    """The sentence naming the CV gain over v1 as the clean evidence (overview, README block).

    Examples
    --------
    >>> cv_evidence(fit).startswith("The design probe")  # doctest: +SKIP
    True
    """
    g = fit.gain_over_v1.cv
    return (
        f"{CHOSEN_ON_TEST}; the CV gain over xG v1 on the training season, which the probe "
        f"never scored, is {gain_cell(g)}, an interval that {cards.side(g)}."
    )


def _model(ev: Evaluation, c: CardContext) -> list[str]:
    if models.is_gbm(ev):
        return models.gbm_model(ev, c)
    return [
        "Unpenalised logistic regression (`ablation.make_lr`), coefficients of the fit on all of "
        "train:",
        "",
        "| Term | Standardised | Natural units |",
        "|---|---|---|",
        *(f"| {t} | {fmt(s)} | {fmt(n)} |" for t, s, n in ev.coefficients),
    ]


def text(fit: v2.V2Fit) -> cards.CardText:
    """The card's own text; the features table names the headline's columns.

    Examples
    --------
    >>> text(fit).role.startswith("PR1")  # doctest: +SKIP
    True
    """
    feats = fit.head.spec.features if isinstance(fit.head.spec, evaluation.ModelSpec) else ()
    return cards.CardText(
        role="PR1: WhoScored xG with the pre-shot context of the previous play event (v1's "
        f"inputs plus {CONTEXT_NAMES}), beside xG v1. {cv_evidence(fit)}",
        intended_use="The context-model result beside xG v1 for WhoScored / Opta shot events. "
        "xG v1's card stays the record, and v1's keeper-neutral headline stays the WhoScored "
        "xGOT input (PM PR1 E4 (a); `prev_action` and `rebound` read the keeper's previous "
        "event, so they are refused in the xGOT input chain, PM MS6 E5). Not for penalties, not "
        "for other providers' data without re-fitting.",
        features=(
            "| Feature | Definition | Source field(s) |",
            "|---|---|---|",
            *(FEATURE_ROWS[f] for f in feats if f in FEATURE_ROWS),
        ),
        model=_model,
        limitations=(
            "`prev_action` and `rebound` are derived from the event order of the archive (file "
            "order with a fail-closed clock check); Opta records no carry.",
            "One train season; the test season is partial.",
            f"Big chance (q{whoscored.BIG_CHANCE}) stays out: it grades the chance; the gap "
            "share below says how much of its ablation gain the situation columns recover.",
        ),
        changelog=CHANGELOG,
        task=cards.TASK,
        target=cards.TARGET,
        best_label="model",
        checks_title="Milestone checks (PR1)",
        checks_note="Judged on the test season; the last two rows report, they do not judge.",
    )


def _groups(fit: v2.V2Fit) -> list[str]:
    return [
        "### Test by competition",
        "",
        f"The headline on each competition's test shots; intervals from {N_BOOT:,} resamples of "
        "that competition's matches.",
        "",
        "| Competition | Shots | Goals | Σ xG | Σ xG error, signed [95% CI] |",
        "|---|---|---|---|---|",
        *(
            f"| {g.key} | {g.shots:,} | {g.goals:,} | {g.xg:.1f} | "
            + ("n/a" if g.signed_error is None else f"{g.signed_error:+.2%}")
            + ("" if g.error_ci is None else f" [{g.error_ci[0]:+.2%}, {g.error_ci[1]:+.2%}]")
            + " |"
            for g in fit.groups
        ),
        "",
    ]


def _gap(fit: v2.V2Fit) -> list[str]:
    v1h, abl, head = fit.v1_head, fit.v1_ablation, fit.head
    rows = [
        (split, v2.gap_text(v1h, head, abl, key), getattr(v1h, attr).log_loss,
         getattr(abl, attr).log_loss, getattr(head, attr).log_loss)
        for split, key, attr in (("test", "test", "holdout"), ("CV", "cv", "cv"))
    ]  # fmt: skip
    return [
        "### Share of the big-chance gap",
        "",
        *(
            f"- {split}: {share} = (v1 {a:.4f} − v2 {b:.4f}) / (v1's headline "
            f"({v1h.spec.title}, {a:.4f}) minus v1's labelled ablation ({abl.spec.title}, "
            f"{c:.4f}))."
            for split, share, a, c, b in rows
        ),
        "",
    ]


def _census(fit: v2.V2Fit) -> list[str]:
    q, ctx = fit.census.qualifiers, fit.census.context
    return [
        "### Census of pre-shot information (counts only)",
        "",
        "Every Opta qualifier on the population's shots, by class; situation qualifiers may "
        "enter xG, a grading, post-shot or outcome-derived one never does.",
        "",
        "| Qualifier | Class | Train shots | Train goals | Test shots |",
        "|---|---|---|---|---|",
        *(
            f"| q{r['id']} {r['name']} | {r['class'] or 'unclassified'} | {r['train_shots']:,} | "
            f"{r['train_goals']:,} | {r['test_shots']:,} |"
            for r in q.iter_rows(named=True)
        ),
        "",
        "| Context column = value | Class | Keeper-derived (xG only) | Train shots | Train goals "
        "| Test shots |",
        "|---|---|---|---|---|---|",
        *(
            f"| {r['column']} = {r['value']} | {r['class']} | {r['keeper_derived']} | "
            f"{r['train_shots']:,} | {r['train_goals']:,} | {r['test_shots']:,} |"
            for r in ctx.iter_rows(named=True)
        ),
        "",
        f"- {v2.CARRY_LINE}.",
        "",
    ]


def render(fit: v2.V2Fit) -> str:
    """The ``xg-v2`` card.

    Examples
    --------
    >>> render(fit).splitlines()[0]  # doctest: +SKIP
    '# Model card: Monotone GBM + Platt, context v2'
    """
    c = fit.candidates
    return cards.render_card(
        fit.head, text(fit), evaluations=(*c.baselines, *c.steps, c.gbm), benchmark=None,
        dev=fit.dev, context=fit.context, provider=release.WHOSCORED_TEXT,
        validation=VALIDATION, thresholds=(), comparisons=(fit.gain_over_v1, fit.gbm_gain),
        verdict=models.verdict(fit.head, fit.head, fit.gbm_gain, split="test"),
        checks=fit.checks, extra=[*_groups(fit), *_gap(fit), *_census(fit)],
    )  # fmt: skip


def summary_block(fit: v2.V2Fit) -> str:
    """The README block: v2's headline beside v1's on the test season, with the checks met.

    Examples
    --------
    >>> print(summary_block(fit))  # doctest: +SKIP
    | Model | CV log loss | Test log loss | ...
    """
    h, v, g = fit.head, fit.v1_head, fit.gain_over_v1.holdout
    share = v2.gap_text(v, h, fit.v1_ablation, "test")
    judged = [ch for ch in fit.checks if ch.result != "n/a"]
    lines = [
        "| Model | CV log loss | Test log loss | Test AUC | Test cal. slope | Test Σ xG error "
        "(signed) |",
        "|---|---|---|---|---|---|",
        *(
            f"| {label} ({e.spec.title}) | {fmt(e.cv.log_loss)} | {fmt(e.holdout.log_loss)} | "
            f"{fmt(e.holdout.auc)} | {fmt(e.holdout.cal_slope)} | "
            f"{e.holdout.xg_signed_error:+.2%} |"
            for label, e in (("xG v1", v), ("**xG v2**", h))
        ),
        "",
        f"Test gain of v2 over v1: {gain_cell(g)}, an interval that {cards.side(g)}. "
        f"{cv_evidence(fit)} Share of the big-chance ablation gap closed on test: {share}. "
        f"PR1 checks met: {sum(ch.result == 'met' for ch in judged)} of {len(judged)}. v1 stays "
        f"the WhoScored xG of record and the xGOT input; test goal rate "
        f"{pct(h.holdout.goals / h.holdout.n)}.",
    ]
    return "\n".join(lines) + "\n"


def run(
    manifest: StrPath = v1.DEFAULT_MANIFEST,
    *,
    docs_dir: StrPath = "docs/models",
    cache: StrPath | None = None,
) -> v2.V2Fit:
    """Fit, write ``docs_dir/xg-v2.md`` and print the checks and the README block.

    Examples
    --------
    >>> run(cache="data/cache").head.spec.name  # doctest: +SKIP
    'ws-gbm-v2'
    """
    fit = v2.fit(manifest, cache=cache)
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / CARD_FILE).write_text(render(fit), encoding="utf-8", newline="\n")
    print(f"headline: {fit.head.spec.name}")
    print(*(f"check | {c.target} | {c.measured} | {c.result}" for c in fit.checks), sep="\n")
    print(summary_block(fit), end="")
    print(f"card: {out / CARD_FILE}")
    return fit


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 1 when a judged check is missed (the gate reads it).

    Examples
    --------
    >>> main(["--manifest", "missing.json"])  # doctest: +SKIP
    1
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.v2_cards",
        description="Evaluate the WhoScored xG v2 (pre-shot context) and write xg-v2.md.",
    )
    parser.add_argument("--manifest", default=v1.DEFAULT_MANIFEST, help="WhoScored manifest")
    parser.add_argument("--docs-dir", default="docs/models", help="where the card goes")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--cache", default="data/cache", help="prediction cache root (git-ignored)")
    where.add_argument("--no-cache", action="store_true", help="fit everything, write no cache")
    args = parser.parse_args(argv)
    try:
        fit = run(args.manifest, docs_dir=args.docs_dir,
                  cache=None if args.no_cache else args.cache)  # fmt: skip
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    return 1 if any(c.result == "missed" for c in fit.checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
