"""The card renderer's provider seam (spec ``runs/MS5/goalmouth`` §8, §10.8): ``ProviderText``
carries every provider string ``render_card`` prints; ``CardText`` carries the task and target.

The custom wording below is invented ("Acme"), so a provider literal left in the renderer shows
up as "StatsBomb" in a card that names no StatsBomb anywhere. The StatsBomb strings are typed
here from the MS4 cards, so ``STATSBOMB_TEXT`` is held to the text the cards printed before.
"""

from __future__ import annotations

import dataclasses
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from test_baselines_cards import _rows, _section

from football_shot_models import cards, data, metrics

FAILING = (metrics.Threshold("auc", ">=", 0.999),)  # so ``_why`` prints
ACME_LABELS = MappingProxyType({r: f"acme filter {r}" for r in data.EXCLUSION_REASONS})


def _provider(**over: Any) -> Any:
    fields: dict[str, Any] = {
        "source": "Acme event feed",
        "licensing": "Acme licence: aggregates only.",
        "filter_labels": ACME_LABELS,
        "benchmark_name": "Acme xG, with tracking inputs",
        "benchmark_caveat": "Acme caveat sentence.",
        "families": lambda c: f"acme families ({c.team_selected_seasons:,} groups)",
        "exclusions_log": "Acme log sentence.",
    }
    return cards.ProviderText(**{**fields, **over})


def _text() -> Any:
    return cards.CardText(
        role="Acme role.",
        intended_use="Acme intended use.",
        features=("Acme features.",),
        model=lambda ev, c: ["Acme model."],
        limitations=("Acme limitation.",),
        changelog=("v1: acme.",),
        task="acme task",
        target="Acme target sentence.",
    )


def _render(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any, provider: Any) -> str:
    evaluations, _ = run_result
    dev, held = dev_held
    bench = evaluations[3]
    bench = dataclasses.replace(bench, spec=dataclasses.replace(bench.spec, title="Acme bench"))
    return cards.render_card(
        evaluations[2],
        _text(),
        evaluations=evaluations[:3],
        benchmark=bench,
        dev=dev,
        context=cards.card_context(xg_data, dev, held),
        thresholds=FAILING,
        provider=provider,
    )


def test_render_card_provider_text(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    p = _provider()
    card = _render(run_result, xg_data, dev_held, p)
    c = cards.card_context(xg_data, *dev_held)
    assert "**Why.**" in card  # precondition: the failing threshold prints ``_why``
    assert "StatsBomb" not in card
    assert data.EXCLUSIONS_SNAPSHOT not in card  # the StatsBomb log sentence is provider text
    assert "- **Task:** acme task" in card.splitlines()
    assert _section(card, "## Target").strip() == "Acme target sentence."
    assert "- **Source(s):** Acme event feed, revision `" in card
    assert p.families(c) in card
    assert "excluded shots, by first matching reason. Acme log sentence." in card
    filters = _rows(_section(card, "## Data"))
    assert [r[0] for r in filters] == [p.filter_labels[reason] for reason, _, _ in c.exclusions]
    assert "Acme xG, with tracking inputs" in card
    assert "Acme caveat sentence." in _section(card, "## Benchmarks")
    assert _section(card, "## Data licensing").strip() == "Acme licence: aggregates only."


def test_no_caveat_no_families(run_result: tuple[Any, Path], xg_data: Any, dev_held: Any) -> None:
    """None fields print nothing: no benchmark sentence in ``_why``, no caveat paragraph, no
    Families bullet and no men / women clause, nothing after the Filters sentence."""
    p = _provider(benchmark_name=None, benchmark_caveat=None, families=None, exclusions_log=None)
    card = _render(run_result, xg_data, dev_held, p)
    assert "**Why.**" in card
    assert "on the hold-out. Thresholds come from" in card  # the benchmark sentence is gone
    bench = _section(card, "## Benchmarks").splitlines()
    assert [x for x in bench if x.strip() and not x.lstrip().startswith("|")] == []
    assert "Families" not in card
    rows = next(x for x in card.splitlines() if x.startswith("- **Rows:**"))
    assert "men" not in rows and "women" not in rows
    filters = next(x for x in card.splitlines() if x.startswith("- **Filters:**"))
    assert filters.endswith("excluded shots, by first matching reason.")
    assert "None" not in card


def test_filter_label_unknown_reason() -> None:
    sb = cards.STATSBOMB_TEXT
    assert isinstance(sb.filter_labels, MappingProxyType)
    assert sb.label("direct_corner") == "direct corner (logged as a shot on outcome)"
    with pytest.raises(ValueError, match="nope") as info:
        sb.label("nope")
    assert "own_goal" in str(info.value)  # names the known reasons


def test_statsbomb_text_and_seam() -> None:
    """STATSBOMB_TEXT holds the strings the MS4 cards printed; the old module constants are
    gone; ``provider`` is required and keyword-only; task and target are CardText fields."""
    sb = cards.STATSBOMB_TEXT
    assert sb.source == "StatsBomb open data"
    assert sb.licensing == (
        "Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never "
        "committed or redistributed; this card holds aggregates only."
    )
    assert dict(sb.filter_labels) == {
        "own_goal": "own goal (Own Goal Against events; not shots)",
        "shootout": "shoot-out (period 5; all are penalties)",
        "penalty": "penalty (in-game)",
        "direct_corner": "direct corner (logged as a shot on outcome)",
        "no_freeze_frame": "no freeze frame",
    }
    assert sb.benchmark_name == "StatsBomb xG, with freeze-frame inputs"
    assert sb.benchmark_caveat is not None and sb.benchmark_caveat.startswith(
        "StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor"
    )
    assert sb.exclusions_log is not None and data.EXCLUSIONS_SNAPSHOT in sb.exclusions_log
    for name in ("LICENSING", "FILTER_LABELS", "BENCHMARK_CAVEAT"):
        assert not hasattr(cards, name), name
    param = inspect.signature(cards.render_card).parameters["provider"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty
    assert cards.TARGET == (
        "`is_goal`: the ball went in. Own goals are excluded, so every goal is the shooter's."
    )
    fields = {f.name: f for f in dataclasses.fields(cards.CardText)}
    for name in ("task", "target"):
        assert fields[name].default is dataclasses.MISSING, name


def test_families_bullet_added_by_renderer(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    """Build ratify 2: ``ProviderText.families`` returns the bullet text without its ``- ``;
    ``render_card`` adds it, so the card has exactly one bullet marker."""
    p = _provider()
    card = _render(run_result, xg_data, dev_held, p)
    c = cards.card_context(xg_data, *dev_held)
    assert f"- {p.families(c)}" in card.splitlines()
    sb = cards.STATSBOMB_TEXT.families
    assert sb is not None and sb(c).startswith("**Families (segments):** ")


def test_benchmark_name_is_an_appositive(
    run_result: tuple[Any, Path], xg_data: Any, dev_held: Any
) -> None:
    """Build ratify 3: ``_why`` prints ``benchmark_name`` followed by ", scores X / Y on the
    CV shots.", so the name reads as an appositive phrase."""
    card = _render(run_result, xg_data, dev_held, _provider())
    why = next(x for x in card.splitlines() if "**Why.**" in x)
    bench = run_result[0][3].cv
    scores = f"{cards.fmt(bench.log_loss)} / {cards.fmt(bench.brier)}"
    assert f" Acme xG, with tracking inputs, scores {scores} on the CV shots. " in why
