"""The card renderer's xGOT seams (spec ``runs/MS5/xgot`` §1, §9; test plan 8).

``CardText.quantity`` names what the metrics, segments and reliability tables sum and average
("xG" by default, "xGOT" for the xGOT cards); ``render_card`` takes an optional benchmark and
``extra`` lines; ``models.gbm_features`` takes the feature rows and monotone reasons. Defaults
reproduce today's bytes: the default-bytes test compares the code with itself, and the criterion
"the 8 StatsBomb cards and the v1 card stay byte-identical" is shown by the build-gate commands
(masked diff against HEAD, ``shasum -c`` against the schema-1.2 regeneration) quoted in the gate.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import models_cases as mc
import pytest
from test_cards_provider import FAILING, _text

from football_shot_models import ablation, cards, evaluation, models


def _render(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any, text: Any, **kw: Any) -> str:
    evaluations, _ = run_result
    dev, held = dev_held
    args: dict[str, Any] = {
        "evaluations": evaluations[:3],
        "benchmark": evaluations[3],
        "dev": dev,
        "context": cards.card_context(xg_data, dev, held),
        "provider": cards.STATSBOMB_TEXT,
        **kw,
    }
    return cards.render_card(evaluations[2], text, **args)


def _headers(card: str) -> list[str]:
    return [x for x in card.splitlines() if x.startswith("|")]


def test_quantity_default_bytes(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    """``quantity`` defaults to "xG", ``extra`` to (), and passing the defaults explicitly gives
    the same bytes as leaving them out."""
    field = {f.name: f for f in dataclasses.fields(cards.CardText)}["quantity"]
    assert field.default == "xG"
    text = _text()
    plain = _render(run_result, xg_data, dev_held, text)
    explicit = _render(run_result, xg_data, dev_held, dataclasses.replace(text, quantity="xG"),
                       extra=())  # fmt: skip
    assert plain == explicit
    assert "| Bin | Shots | Mean xG | Goal rate |" in plain
    assert any("Σ xG" in x for x in _headers(plain))


def test_quantity_extra_and_no_benchmark(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    """With ``quantity="xGOT"`` every table label says xGOT; ``extra`` lines go right before
    "## Benchmarks"; ``benchmark=None`` leaves the benchmark row out and refuses thresholds."""
    evaluations, _ = run_result
    text = dataclasses.replace(_text(), quantity="xGOT")
    extra = ["### ECE", "", "| ECE | CV |", "|---|---|", "| x | 0.0100 |"]
    card = _render(run_result, xg_data, dev_held, text, extra=extra, benchmark=None,
                   thresholds=())  # fmt: skip
    assert "| Bin | Shots | Mean xGOT | Goal rate |" in card
    assert any("Σ xGOT" in x for x in _headers(card))
    assert [x for x in _headers(card) if re.search(r"(Σ|Mean) xG(?!OT)", x)] == []
    lines = card.splitlines()
    at = lines.index("## Benchmarks")
    start = lines.index(extra[0])
    assert lines[start : start + len(extra)] == extra
    assert all(x == "" for x in lines[start + len(extra) : at])
    bench = card[card.index("## Benchmarks") :].split("\n## ")[0]
    titles = [e.spec.title for e in evaluations[:3]]
    rows = [x for x in bench.splitlines() if x.startswith("| ") and not x.startswith("| Model")]
    assert [r.split(" | ")[0].removeprefix("| ") for r in rows] == titles
    assert evaluations[3].spec.title not in bench
    with pytest.raises(ValueError, match="benchmark"):
        _render(run_result, xg_data, dev_held, text, benchmark=None, thresholds=FAILING)


def test_gbm_features_parameters() -> None:
    """``gbm_features`` reads the feature rows and monotone reasons it is given; the defaults are
    the xG ones."""
    spec = evaluation.ModelSpec("xgot-gbm", "xGOT GBM", ("xg_logit", "mouth_y_abs", "mouth_z"),
                                ablation.make_lr)  # fmt: skip
    coef = (("max_leaf_nodes", 4.0, 4.0), ("monotone:xg_logit", 1.0, 1.0),
            ("monotone:mouth_y_abs", 1.0, 1.0))  # fmt: skip
    ev = mc.fake_eval(spec, coefficients=coef)
    rows = {f: f"| {f} | row of {f} | src |" for f in spec.features}
    reasons = {"xg_logit": "higher xG never lowers xGOT"}
    out = models.gbm_features(ev, rows=rows, reasons=reasons)
    for f in spec.features:
        assert rows[f] in out
    assert "- xg_logit: non-decreasing — higher xG never lowers xGOT" in out
    assert "- mouth_y_abs: non-decreasing" in out
    xg_spec = dataclasses.replace(spec, features=("distance_m", "angle_rad"))
    xg_ev = mc.fake_eval(xg_spec, coefficients=(("monotone:distance_m", -1.0, -1.0),))
    assert models.gbm_features(xg_ev) == models.gbm_features(
        xg_ev, rows=ablation.FEATURE_ROWS, reasons=models.MONOTONE_REASONS
    )
