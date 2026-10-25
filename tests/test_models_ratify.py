"""Ratify tests for the Engineer's MS3.models decisions beyond the spec (``build.md`` R1-R14, QA
choices 4 and 12): one fast test per decision that states behaviour and is not already pinned
elsewhere.

Already pinned (not repeated here): R1 by ``test_models_headline.py::test_statsbomb_sentence``
(StatsBomb's hold-out Σ xG comes only from the ``benchmark`` argument); R4 by
``::test_verdict_sentences`` (exact sentences with one "0"); R13's checks mapping by
``test_models_run.py::test_models_run_fixture`` (the fixture's headline is a step, so the checks
section appears on that step card only); QA choice 4 by ``::test_gbm_card_monotone_lines``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
from football_core import schema
from test_baselines_cards import _header, _rows, _section
from test_cards_gain import GAIN

from football_shot_models import ablation, baselines, cards, data, evaluation, metrics

S0, S3 = baselines.BASELINES[2], ablation.STEPS[2]
STEP_NAMES = tuple(s.name for s in ablation.STEPS)
CHECKS = "### Milestone checks (MS3)"
LR_GAIN = evaluation.Comparison(
    f"over {S3.title}", mc.gain(0.0035, 0.0028, 0.0043), mc.gain(0.0014, -0.0005, 0.0034)
)
SB_ABOVE_THEN_NOT = evaluation.Comparison(
    "over StatsBomb xG", mc.gain(0.0021, 0.0012, 0.0031), mc.gain(-0.0013, -0.004, 0.0016)
)
SB_INCLUDES = evaluation.Comparison(
    "over StatsBomb xG", mc.gain(0.0001, -0.0002, 0.0004), mc.gain(-0.0013, -0.004, 0.0016)
)
SB_ABOVE_BOTH = evaluation.Comparison(
    "over StatsBomb xG", mc.gain(0.0021, 0.0012, 0.0031), mc.gain(0.0010, 0.0001, 0.0019)
)
BENCHMARK_HEADER = [
    "Model", "CV log loss", "CV Brier", "CV AUC", "CV cal. slope", "Hold-out log loss",
    "Hold-out Brier", "Hold-out AUC", "Hold-out cal. slope",
]  # fmt: skip


def _models() -> Any:
    from football_shot_models import models

    return models


def _settings_rows(leaves: float = 16, a: float = -0.0122, b: float = 0.9923) -> Any:
    return (
        ("max_leaf_nodes", leaves, leaves), ("iterations", 200, 200), ("platt_intercept", a, a),
        ("platt_slope", b, b), ("monotone:distance_m", -1, -1), ("monotone:angle_rad", 1, 1),
    )  # fmt: skip


def _goals(n: int, rate: float) -> list[bool]:
    return [i < round(n * rate) for i in range(n)]


# ---- R2: milestone_checks without s0 --------------------------------------------------------


def test_milestone_checks_s0_quote() -> None:
    """R2: with ``s0`` the first target quotes S0's CV log loss on these folds; without it, only
    the MS2 number."""
    models = _models()
    frame = pl.DataFrame(
        {
            "is_goal": pl.Series([True, False] * 5, dtype=pl.Boolean()),
            "gender": pl.Series(["male"] * 10, dtype=schema.GENDER),
        }
    )
    p = np.full(10, 0.5)
    ev = mc.fake_eval(models.GBM, oof=p, holdout_p=p)
    kwargs = {
        "s0_gain": evaluation.Comparison(f"over {S0.title}", LR_GAIN.cv, LR_GAIN.holdout),
        "sb_gain": SB_ABOVE_THEN_NOT, "gbm_gain": LR_GAIN, "dev": frame, "held": frame,
    }  # fmt: skip
    bench = mc.fake_eval(evaluation.BENCHMARK)
    plain = models.milestone_checks(ev, bench, **kwargs)[0].target
    assert plain == (
        "CV log-loss gain over S0 with a 95% interval above 0 (the MS2 best baseline 0.2880 is "
        "the MS2 population)"
    )
    s0 = mc.fake_eval(S0, cv=mc.scores(log_loss=0.2872))
    quoted = models.milestone_checks(ev, bench, s0=s0, **kwargs)[0].target
    assert quoted == (
        f"CV log-loss gain over S0 with a 95% interval above 0 ({S0.title} 0.2872 on these "
        "folds; the MS2 best baseline 0.2880 is the MS2 population)"
    )


# ---- R3: verdict identity -------------------------------------------------------------------


def test_verdict_gbm_identity() -> None:
    """R3: "is the GBM" is the ``gbm_settings`` hook, not the name (a renamed GBM spec, the MS4
    seam, keeps the GBM sentences; an LR named ``xg-gbm`` does not get them); "is the headline"
    is identity, not an equal copy or the same name."""
    models = _models()
    renamed = mc.fake_eval(dataclasses.replace(models.GBM, name="xg-v1", title="xG v1"))
    lr_head, stays = mc.fake_eval(S3), dataclasses.replace(LR_GAIN, cv=mc.gain(4e-4, -3e-4, 1e-3))
    assert models.verdict(renamed, renamed, LR_GAIN).startswith(
        f"This is the headline model: its CV log-loss gain over {S3.title} is"
    )
    assert models.verdict(renamed, lr_head, stays).startswith("Not the headline model: its CV")
    impostor = mc.fake_eval(dataclasses.replace(S3, name="xg-gbm", title="LR named xg-gbm"))
    assert models.verdict(impostor, lr_head, stays) == (
        f"Not the headline model; that is {S3.title}."
    )
    head = mc.fake_eval(models.GBM)
    copy = dataclasses.replace(head)
    assert models.verdict(copy, head, LR_GAIN) == (
        "Not the headline model; that is Monotone GBM + Platt."
    )


# ---- R5, R6: penalty_rows labels and null keys ----------------------------------------------


def test_penalty_row_labels() -> None:
    """R5: the row labels in order; ``penalty_card`` finds rows by label and refuses rows that do
    not start with the in-game all row."""
    models = _models()
    rows = models.penalty_rows(mc.penalties_frame())
    assert [r.label for r in rows] == [
        "In-game: all", "In-game: dev", "In-game: hold-out", "In-game: men", "In-game: women",
        "In-game: league or tournament", "In-game: team-selected",
        "Shoot-out: all", "Shoot-out: men", "Shoot-out: women",
    ]  # fmt: skip
    spread = models.exchangeability(mc.penalties_frame())
    for bad in (rows[1:], ()):
        with pytest.raises(ValueError, match="In-game: all"):
            models.penalty_card(bad, spread, mc.card_context())


@pytest.mark.parametrize(
    ("column", "row", "holdout", "match"),
    [
        ("competition_id", 2, None, "hold-out mask has 1 null"),  # sb:k03, a hold-out-season kick
        ("season_id", 0, None, "hold-out mask has 1 null"),  # sb:k01, a hold-out-competition kick
        ("season_id", 5, pl.col("season_id") == "sb:44", "hold-out mask has 1 null"),  # custom
        ("reason", 4, None, "reason has 1 null"),
    ],
    ids=["null competition", "null season", "null in a custom mask", "null reason"],
)
def test_penalty_rows_null_holdout_key(
    column: str, row: int, holdout: pl.Expr | None, match: str
) -> None:
    """R6: a null in the hold-out mask raises instead of dropping the kick from both dev and the
    hold-out; a null ``reason`` raises too."""
    models = _models()
    frame = mc.with_null(mc.penalties_frame(), column, row)
    with pytest.raises(ValueError, match=match):
        models.penalty_rows(frame, holdout=holdout)


# ---- R7, R8: penalty card margin and benchmarks ---------------------------------------------


def _card(rows: Any, frame: pl.DataFrame) -> str:
    models = _models()
    return str(models.penalty_card(rows, models.exchangeability(frame), mc.card_context()))


@pytest.mark.parametrize(
    ("mean", "low", "high", "measured", "sentence"),
    [
        (0.75, 0.73, 0.77, "0.0100 below the lower bound", "against the 0.04 window, and extends "
         "below it"),
        (0.81, 0.79, 0.83, "0.0100 above the upper bound", "against the 0.04 window, and extends "
         "above it"),
        (0.78, 0.75, 0.81, "0.0200 above the lower bound", "wider than the 0.04 window, and "
         "extends below and above it"),
        (0.79, 0.775, 0.805, "0.0100 below the upper bound", "against the 0.04 window, and "
         "extends above it"),
    ],
    ids=["below the window", "above the window", "tie goes to the lower", "narrow, above"],
)  # fmt: skip
def test_penalty_check_margin_forms(
    mean: float, low: float, high: float, measured: str, sentence: str
) -> None:
    """R7: outside the window the margin is "below the lower" / "above the upper" bound; a mean
    equidistant from both bounds (0.78, an exact float tie) measures from the lower; "wider
    than" only when the interval is wider than the window, "against" otherwise."""
    models = _models()
    frame = mc.penalties_frame()
    rows = models.penalty_rows(frame)
    pooled = dataclasses.replace(rows[0], rate=metrics.Rate(1118, 853, 853 / 1118, mean, low, high))
    card = _card((pooled, *rows[1:]), frame)
    (row,) = [r for r in _rows(card) if r[0].startswith("posterior mean in [0.76, 0.80]")]
    assert row[1] == f"{mean:.4f} [{low:.4f}, {high:.4f}]; {measured}"
    assert f"{high - low:.3f} wide, {sentence}: the check is on the point estimate." in card


def test_penalty_benchmarks_statsbomb_constant() -> None:
    """R8: StatsBomb's constant is the hold-out row's mean provider xG (not the pooled row's); a
    constant outside (0, 1) prints ``n/a``; with no dev kick both sections say so."""
    models = _models()
    frame = mc.penalties_frame()
    for i in range(3):  # sb:k01-k03, the hold-out kicks
        frame = mc.set_value(frame, "provider_xg", i, 0.9)
    rows = models.penalty_rows(frame)
    assert rows[2].provider_xg == pytest.approx(0.9) and rows[0].provider_xg != pytest.approx(0.9)
    loss = -(2 * math.log(0.9) + math.log(0.1)) / 3
    bench = _rows(_section(_card(rows, frame), "## Benchmarks"))
    assert bench[1] == ["StatsBomb mean xG (0.9000)", "3", f"{loss:.4f}"]
    for q in (1.0, 0.0):
        held = dataclasses.replace(rows[2], provider_xg=q)
        card = _card((*rows[:2], held, *rows[3:]), frame)
        assert _rows(_section(card, "## Benchmarks"))[1] == [
            f"StatsBomb mean xG ({q:.4f})",
            "3",
            "n/a",
        ]
    held_only = tuple(k for k in mc.PEN_KICKS if k[1] in ("sb:m01", "sb:m02") or k[2] == 5)
    frame = mc.penalties_frame(held_only)
    card = _card(models.penalty_rows(frame), frame)
    for title in ("## Validation design", "## Benchmarks"):
        assert "The dev has no in-game penalties." in _section(card, title)


# ---- R9, R10, QA 12: GBM card text ----------------------------------------------------------


def _dev(rates: tuple[float, float, float], n: int = 20) -> pl.DataFrame:
    """``n`` dev shots per family with the given goal rates; wide direct free kicks for
    ``wide_dfk`` on the first two rows."""
    family = [f for f in data.FAMILIES for _ in range(n)]
    goals = [g for r in rates for g in _goals(n, r)]
    m = len(family)
    return pl.DataFrame(
        {
            "family": pl.Series(family, dtype=data.FAMILY),
            "is_goal": pl.Series(goals, dtype=pl.Boolean()),
            "situation": pl.Series(["direct_free_kick"] * 2 + ["open_play"] * (m - 2)),
            "y": pl.Series([5.0] * 2 + [34.0] * (m - 2)),
        }
    )


def _gbm_text(
    *, rates: tuple[float, float, float] = (0.1, 0.2, 0.15), xg: float = 110.0,
    sb_gain: evaluation.Comparison = SB_ABOVE_THEN_NOT, leaves: float = 16,
    spec: Any = None,
) -> tuple[Any, evaluation.Evaluation, evaluation.Evaluation, pl.DataFrame]:  # fmt: skip
    models = _models()
    dev = _dev(rates)
    held = pl.DataFrame({"is_goal": pl.Series(_goals(40, 0.25), dtype=pl.Boolean())})
    ev = mc.fake_eval(
        models.GBM if spec is None else spec,
        holdout=mc.scores(goals=100, xg=xg),
        oof=np.full(dev.height, 0.1),
        coefficients=_settings_rows(leaves),
    )
    bench = mc.fake_eval(evaluation.BENCHMARK, oof=np.full(dev.height, 0.2))
    text = models.gbm_text(
        ev, bench, gain=LR_GAIN, sb_gain=sb_gain, dev=dev, held=held, context=mc.card_context()
    )
    return text, ev, bench, dev


@pytest.mark.parametrize(
    ("rates", "xg", "sb_gain", "word", "direction", "clause"),
    [
        ((0.1, 0.2, 0.15), 110.0, SB_ABOVE_THEN_NOT, "convert more than", "over", True),
        ((0.1, 0.2, 0.05), 90.0, SB_INCLUDES, "differ in goal rate from", "under", False),
        ((0.1, 0.1, 0.15), 110.0, SB_ABOVE_BOTH, "differ in goal rate from", "over", False),
    ],
    ids=["more, over, clause", "women below league, under", "team equal to league"],
)
def test_gbm_training_mix_line(
    rates: tuple[float, float, float],
    xg: float,
    sb_gain: evaluation.Comparison,
    word: str,
    direction: str,
    clause: bool,
) -> None:
    """R9: "convert more than" only when both the team-selected and the women's dev goal rates
    exceed the league rate (strictly); over- or under-prediction by the sign of the hold-out Σ xG
    error; the StatsBomb clause only when the CV gain over StatsBomb lies above 0 and the
    hold-out gain does not."""
    text, ev, _, _ = _gbm_text(rates=rates, xg=xg, sb_gain=sb_gain)
    league, team, women = (cards.pct(r) for r in rates)
    signed = ev.holdout.xg_signed_error
    tail = " and of the CV gain over StatsBomb not carrying over" if clause else ""
    assert text.limitations[1] == (
        f"The CV population mixes women's and team-selected seasons that {word} a men's league "
        f"(dev goal rate {league} league, {team} team-selected, {women} women; hold-out 25.00%); "
        f"a likely cause, not a measured one, of the hold-out {direction}-prediction "
        f"({signed:+.2%}){tail}."
    )


def test_gbm_limitations_order_and_largest_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    """R10, QA choice 12: limitations are the hold-out gain line, the training-mix line, the
    largest-leaf line (only when the dev fit chose ``max(LEAVES)``, the module constant), the
    wide-DFK line, then the card's fixed limitations."""
    models = _models()
    text, ev, bench, dev = _gbm_text(leaves=32)
    largest = "The grid's largest leaf count was chosen; a wider grid is a follow-up."
    assert text.limitations[0].startswith(f"The hold-out gain {LR_GAIN.label} is +0.0014")
    assert text.limitations[1].startswith("The CV population mixes")
    assert text.limitations[2:] == (
        largest,
        ablation.wide_dfk(ev, bench, dev),
        *models.CARD.limitations,
    )
    assert largest not in _gbm_text(leaves=16)[0].limitations
    monkeypatch.setattr(models, "LEAVES", (4, 8, 16))
    assert _gbm_text(leaves=16)[0].limitations[2] == largest


def test_gbm_card_layout_from_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    """R10: the Model section prints the fixed settings and the grid from the module constants
    (a swapped constant shows up), then the dev fit's four settings as a table; the Features
    section is the table, the constraints header, one line per fitted sign, then the arc
    sentence."""
    models = _models()
    for name, value in (("LEAVES", (4, 8)), ("STEP", 10), ("MAX_ITER", 40), ("INNER_SPLITS", 3),
                        ("LEARNING_RATE", 0.1), ("L2_REGULARIZATION", 2.0)):  # fmt: skip
        monkeypatch.setattr(models, name, value)
    ev = mc.fake_eval(models.GBM, coefficients=_settings_rows())
    lines = models.CARD.model(ev, mc.card_context())
    text = "\n".join(lines)
    assert "learning_rate=0.1, l2_regularization=2," in text
    assert "3-fold `GroupKFold`" in text
    assert "`max_leaf_nodes` in (4, 8) and 10, 20, ..., 40 iterations" in text
    i = lines.index("| Setting | Value |")
    assert lines[i : i + 6] == [
        "| Setting | Value |", "|---|---|", "| max_leaf_nodes | 16 |", "| iterations | 200 |",
        "| Platt intercept | -0.0122 |", "| Platt slope | 0.9923 |",
    ]  # fmt: skip
    feats = models.gbm_features(ev)
    assert feats[:2] == ("| Feature | Definition | Source field(s) |", "|---|---|---|")
    assert feats[-7:] == (
        "",
        "Monotone constraints of the fitted model (every other column is free):",
        "",
        f"- distance_m: non-increasing — {models.MONOTONE_REASONS['distance_m']}",
        f"- angle_rad: non-decreasing — {models.MONOTONE_REASONS['angle_rad']}",
        "",
        models.MONOTONE_NOTE,
    )


# ---- R11, R12, R13, R14: run wiring, benchmark table, step cards, summary ----------------------


def test_gbm_rows_order(models_run: dict[str, Any]) -> None:
    """R11: ``ModelsRun.rows`` is the step rows then ``xg-gbm``; the GBM's rows are over the best
    LR, over S0, over StatsBomb, and the GBM card's gain table prints them in that order."""
    run, dev, held = models_run["run"], models_run["dev"], models_run["held"]
    assert list(run.rows) == [*STEP_NAMES, "xg-gbm"]
    gbm = run.evaluations[-1]
    best_lr = evaluation.best(run.evaluations[:-1])
    s0 = next(e for e in run.evaluations if e.spec.name == S0.name)
    want = [
        evaluation.compare(label, ref, gbm, dev, held)
        for label, ref in (
            (f"over {best_lr.spec.title}", best_lr), (f"over {S0.title}", s0),
            ("over StatsBomb xG", run.benchmark),
        )
    ]  # fmt: skip
    got = run.rows["xg-gbm"]
    assert [(g.label, g.cv, g.holdout) for g in got] == [(w.label, w.cv, w.holdout) for w in want]
    card = (models_run["docs"] / "xg-gbm.md").read_text(encoding="utf-8")
    assert [r[0] for r in _rows(_section(card, GAIN))] == [w.label for w in want]


def test_benchmark_column_order(models_run: dict[str, Any], run_result: Any) -> None:
    """R12: the benchmark table's columns, in order, on the MS2 cards and the MS3 cards."""
    _, ms2_docs = run_result
    paths = [ms2_docs / f"{b.name}.md" for b in baselines.BASELINES]
    paths += [models_run["docs"] / f"{n}.md" for n in (*STEP_NAMES, "xg-gbm")]
    for path in paths:
        block = _section(path.read_text(encoding="utf-8"), "## Benchmarks")
        assert _header(block) == BENCHMARK_HEADER, path.name


def test_step_cards_checks_and_default_verdict(models_run: dict[str, Any]) -> None:
    """R13: ``step_cards`` renders the checks on the step they are keyed under, only there; a
    step without a verdict gets the "best model" wording."""
    run, dev, held = models_run["run"], models_run["dev"], models_run["held"]
    steps = tuple(e for e in run.evaluations if e.spec.name in STEP_NAMES)
    checks = (metrics.Check("a target", "a measure", "met"),)
    out = ablation.step_cards(
        steps, run.rows, run.benchmark, evaluations=run.evaluations, dev=dev,
        context=cards.card_context(models_run["xg"], dev, held), verdicts={},
        checks={STEP_NAMES[0]: checks},
    )  # fmt: skip
    top = evaluation.best(run.evaluations)
    for ev in steps:
        card = out[ev.spec.name]
        assert (CHECKS in card.splitlines()) == (ev.spec.name == STEP_NAMES[0])
        wording = (
            "This is the best model (lowest grouped-CV log loss)."
            if top is ev
            else f"Not the best model; that is {top.spec.title}."
        )
        assert wording in _section(card, "## Overview")
    assert _rows(_section(out[STEP_NAMES[0]], CHECKS)) == [["a target", "a measure", "met"]]


@pytest.mark.parametrize("head_is_gbm", [True, False], ids=["GBM head", "LR head"])
def test_summary_opens_with_headline(head_is_gbm: bool) -> None:
    """R14: the sentence after the table opens with "Headline model: {title}." so the StatsBomb
    sentence's "it" has a referent."""
    models = _models()
    gbm, s3 = mc.fake_eval(models.GBM), mc.fake_eval(S3)
    sb = mc.fake_eval(evaluation.BENCHMARK)
    head = gbm if head_is_gbm else s3
    penalty = models.PenaltyRow("In-game: all", metrics.beta_interval(7, 10), 6, 0.7835)
    text = models.summary_table((s3, gbm), (sb,), head, penalty, SB_ABOVE_THEN_NOT)
    lines = text.splitlines()
    i = lines.index("")
    assert lines[i + 1] == (
        f"Headline model: {head.spec.title}. "
        f"{models.statsbomb_sentence(head, SB_ABOVE_THEN_NOT, sb)}"
    )
