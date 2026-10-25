"""Model-card additions of MS3.features (spec §6): the log-loss gain table with match-bootstrap
CIs, cards without milestone thresholds, the pooled-OOF calibration slope per segment and the
``best_label`` wording."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from test_baselines_cards import _header, _rows, _section

from football_shot_models import baselines, cards, evaluation, metrics

THRESH = "### MS2 thresholds (retired; see the headline model's MS3 checks)"

GAIN = "### Log-loss gain (95% match-bootstrap CI)"
GAIN_HEADER = "| Comparison | CV gain [95% CI] | Hold-out gain [95% CI] |"


def _render(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any, ev_index: int, **kwargs: Any
) -> str:
    evaluations, _ = run_result
    dev, held = dev_held
    ev = evaluations[ev_index]
    return cards.render_card(
        ev,
        kwargs.pop("text", baselines.CARDS[ev.spec.name]),
        evaluations=evaluations[:3],
        benchmark=evaluations[3],
        dev=dev,
        context=cards.card_context(xg_data, dev, held),
        **kwargs,
    )


def _cell(ci: metrics.GainCI) -> str:
    # MS3.models §6: the sign is kept unless the value is exactly 0 (MS3.features dropped it
    # for anything that rounded to zero)
    def f(x: float) -> str:
        return "+0.0000" if x == 0.0 else f"{x:+.4f}"

    return f"{f(ci.gain)} [{f(ci.low)}, {f(ci.high)}]"


def test_card_gain_section(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    evaluations, _ = run_result
    dev, held = dev_held
    real = evaluation.compare("body part: added", evaluations[1], evaluations[2], dev, held)
    tiny = evaluation.Comparison(
        "format check",
        metrics.GainCI(-3e-5, -6e-5, 4.99e-5, 2000, 12),  # -0.00003 prints -0.0000 (MS3.models)
        metrics.GainCI(0.0123456, 0.00049, 0.02, 2000, 2),
    )
    card = _render(run_result, xg_data, dev_held, 2, comparisons=(real, tiny))
    lines = card.splitlines()
    assert lines.index("## Metrics") < lines.index(GAIN) < lines.index("### Reliability")
    block = _section(card, GAIN)
    sentence = (
        "Gain = log loss of the reference minus this model, pooled out of fold over all dev "
        "shots (CV) and over the hold-out; positive means this model is better. The 95% interval "
        f"is the percentile interval of {metrics.N_BOOT:,} resamples of matches (seed "
        f"{metrics.SEED}) for the fixed fold models: it covers which matches were sampled, not "
        "refitting."
    )
    assert sentence in block
    assert "2,000 resamples of matches (seed 0)" in block
    assert GAIN_HEADER in block.splitlines()
    assert _rows(block) == [
        ["body part: added", _cell(real.cv), _cell(real.holdout)],
        ["format check", "-0.0000 [-0.0001, +0.0000]", "+0.0123 [+0.0005, +0.0200]"],
    ]
    # without comparisons there is no gain section (the MS2 cards)
    assert GAIN not in _render(run_result, xg_data, dev_held, 2).splitlines()


def test_card_thresholds_omitted(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    """``thresholds=()`` drops the Milestone thresholds section and its Why paragraph; the
    default keeps the MS2 thresholds (the base rate misses them, so its Why is there)."""
    default = _render(run_result, xg_data, dev_held, 0)
    assert THRESH in default.splitlines() and "**Why.**" in default
    explicit = _render(run_result, xg_data, dev_held, 0, thresholds=metrics.THRESHOLDS)
    assert explicit == default
    none = _render(run_result, xg_data, dev_held, 0, thresholds=())
    assert THRESH not in none.splitlines()
    assert "**Why.**" not in none
    assert "### Reliability" in none.splitlines() and "## Metrics" in none.splitlines()


def test_card_segment_cal_slope(run_result: tuple[Any, Path], dev_held: Any) -> None:
    """Each segment row prints the calibration slope of its pooled out-of-fold predictions;
    the Metrics table keeps the fold mean."""
    evaluations, docs = run_result
    dev, _ = dev_held
    y = evaluation.labels(dev).astype(np.float64)
    for ev in evaluations[1:3]:
        card = (docs / f"{ev.spec.name}.md").read_text(encoding="utf-8")
        block = _section(card, "### Segments")
        col = _header(block).index("Cal. slope (pooled OOF)")
        rows = _rows(block)
        assert rows
        for row in rows:
            column, value = (part.strip() for part in row[0].split("=", 1))
            mask = (dev[column].cast(pl.String) == value).to_numpy()
            slope = metrics.calibration(y[mask], ev.oof[mask])[1]
            assert row[col] == cards.fmt(slope), row
        assert any(row[col] != "n/a" for row in rows)
        metrics_rows = {r[0]: r for r in _rows(_section(card, "## Metrics"))}
        assert metrics_rows["Calibration slope"][2] == cards.fmt(ev.cv.cal_slope)


def test_card_best_label(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    evaluations, _ = run_result
    top = evaluation.best(evaluations[:3])
    assert baselines.CARDS[top.spec.name].best_label == "baseline"  # the MS2 default
    for i, ev in enumerate(evaluations[:3]):
        text = replace(baselines.CARDS[ev.spec.name], best_label="model")
        overview = _section(_render(run_result, xg_data, dev_held, i, text=text), "## Overview")
        if ev is top:
            assert "This is the best model (lowest grouped-CV log loss)." in overview
        else:
            assert f"Not the best model; that is {top.spec.title}." in overview
        assert "baseline (lowest" not in overview and "best baseline" not in overview


def test_card_filter_and_task_text() -> None:
    assert cards.FILTER_LABELS["direct_corner"] == "direct corner (logged as a shot on outcome)"
    assert "direct corners, own goals and shoot-outs excluded" in cards.TASK


def test_card_gain_before_thresholds(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    """Ratify R2: with thresholds on, the gain section still comes right after the Metrics
    table, before Milestone thresholds and Reliability."""
    evaluations, _ = run_result
    dev, held = dev_held
    row = evaluation.compare("body part: added", evaluations[1], evaluations[2], dev, held)
    lines = _render(run_result, xg_data, dev_held, 2, comparisons=(row,)).splitlines()
    order = [
        "## Metrics",
        GAIN,
        THRESH,
        "### Reliability",
    ]
    assert [lines.index(h) for h in order] == sorted(lines.index(h) for h in order)


def test_card_version_from_changelog(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    """Ratify R3: the card version is the number of changelog entries, not a literal."""
    evaluations, docs = run_result
    name = evaluations[2].spec.name
    text = replace(baselines.CARDS[name], changelog=("v1: a.", "v2: b.", "v3: c."))
    overview = _section(_render(run_result, xg_data, dev_held, 2, text=text), "## Overview")
    assert f"- **Name / version:** `{name}` v3" in overview.splitlines()
    written = (docs / f"{name}.md").read_text(encoding="utf-8").splitlines()
    assert f"- **Name / version:** `{name}` v{len(baselines.CHANGELOG)}" in written
