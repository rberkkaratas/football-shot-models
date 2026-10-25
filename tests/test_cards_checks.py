"""Model-card additions of MS3.models (spec §6): calibration slopes in the benchmark table, the
``verdict`` overview sentence and the ``Milestone checks (MS3)`` section.

Renders the MS2 fixture's baseline evaluations (session fixture ``run_result``).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from test_baselines_cards import _header, _rows, _section
from test_cards_gain import GAIN, _render

from football_shot_models import baselines, cards, evaluation, metrics

CHECKS = "### Milestone checks (MS3)"


def test_card_benchmark_slopes(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    """§1.9, §6: every benchmark row prints its CV fold-mean and hold-out calibration slopes
    (``n/a`` for None), also in the MS2 baseline cards ``baselines.run`` writes; their changelog
    says so."""
    evaluations, docs = run_result
    for ev in evaluations[:3]:
        block = _section((docs / f"{ev.spec.name}.md").read_text(encoding="utf-8"), "## Benchmarks")
        header = _header(block)
        cv, ho = header.index("CV cal. slope"), header.index("Hold-out cal. slope")
        rows = _rows(block)
        assert [r[0] for r in rows] == [e.spec.title for e in evaluations]
        assert [(r[cv], r[ho]) for r in rows] == [
            (cards.fmt(e.cv.cal_slope), cards.fmt(e.holdout.cal_slope)) for e in evaluations
        ]
    assert "v3 (MS3, M1.4): the benchmark table shows calibration slopes." in baselines.CHANGELOG
    none = dataclasses.replace(
        evaluations[1],
        cv=dataclasses.replace(evaluations[1].cv, cal_slope=None),
        holdout=dataclasses.replace(evaluations[1].holdout, cal_slope=None),
    )
    dev, held = dev_held
    card = cards.render_card(
        evaluations[2], baselines.CARDS[evaluations[2].spec.name],
        evaluations=(evaluations[0], none, evaluations[2]), benchmark=evaluations[3], dev=dev,
        context=cards.card_context(xg_data, dev, held),
    )  # fmt: skip
    block = _section(card, "## Benchmarks")
    header = _header(block)
    row = _rows(block)[1]
    assert (row[header.index("CV cal. slope")], row[header.index("Hold-out cal. slope")]) == (
        "n/a",
        "n/a",
    )


def test_card_verdict(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    """§6: a ``verdict`` replaces the "best {label}" sentence after the role."""
    evaluations, _ = run_result
    name = evaluations[2].spec.name
    sentence = "Not the headline model; that is Monotone GBM + Platt."
    overview = _section(_render(run_result, xg_data, dev_held, 2, verdict=sentence), "## Overview")
    assert f"- {baselines.CARDS[name].role} {sentence}" in overview.splitlines()
    assert "best baseline" not in overview and "Not the best" not in overview


def test_card_checks_section(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    """§6: non-empty ``checks`` add the MS3 section where the thresholds go (after the gain table,
    before Reliability) with the PM sentence and one row per check; without them it is absent."""
    evaluations, _ = run_result
    dev, held = dev_held
    row = evaluation.compare("body part: added", evaluations[1], evaluations[2], dev, held)
    checks = (
        metrics.Check("slope in [0.9, 1.1], CV fold mean", "1.0100", "met"),
        metrics.Check("slope, hold-out women", "n/a (0 women's shots in the hold-out)", "n/a"),
        metrics.Check("fold Σ xG error mean <= 3.00%", "3.20%", "missed"),
    )
    card = _render(
        run_result, xg_data, dev_held, 2, thresholds=(), comparisons=(row,), checks=checks
    )
    lines = card.splitlines()
    order = ["## Metrics", GAIN, CHECKS, "### Reliability"]
    assert [lines.index(h) for h in order] == sorted(lines.index(h) for h in order)
    block = _section(card, CHECKS)
    assert "Thresholds re-baselined by the PM: targets are relative to the same shots." in block
    assert _header(block) == ["Target", "Measured", "Result"]
    assert _rows(block) == [[c.target, c.measured, c.result] for c in checks]
    plain = _render(run_result, xg_data, dev_held, 2, thresholds=(), comparisons=(row,))
    assert CHECKS not in plain.splitlines()


def test_card_checks_title_from_text(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    """run3-ms3 (architect 1): the checks heading and sentence come from the ``CardText``, so a
    later milestone's checks are not rendered under "MS3"; the defaults are the MS3 ones."""
    evaluations, _ = run_result
    assert (cards.CardText.checks_title, cards.CardText.checks_note) == (
        "Milestone checks (MS3)",
        "Thresholds re-baselined by the PM: targets are relative to the same shots.",
    )
    name = evaluations[2].spec.name
    text = dataclasses.replace(
        baselines.CARDS[name], checks_title="Milestone checks (MS4)", checks_note="M1.7 targets."
    )
    checks = (metrics.Check("log loss in [0.26, 0.27]", "0.2650", "met"),)
    card = _render(run_result, xg_data, dev_held, 2, text=text, thresholds=(), checks=checks)
    lines = card.splitlines()
    assert "### Milestone checks (MS4)" in lines and "M1.7 targets." in lines
    assert CHECKS not in lines and "re-baselined" not in card
