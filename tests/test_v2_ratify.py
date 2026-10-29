"""``v2`` / ``v2_cards``: one test per build ratify item that states behaviour (build log
``runs/PR1/xg-context/build.md`` "Architect: ratify" 11-18).

The fit-based tests read the session ``conftest.v2_fit`` (no refit); the census and CLI tests use
hand-written frames and fakes. Expected values are written by hand or recomputed here by another
route (``v2_cases.qualifier_ids``), never read from the module under test.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
import v2_cases as v2c

from football_shot_models import ablation, models, v1, v2, v2_cards
from football_shot_models.metrics import Check

NA_TEXT = "n/a (the ablation is not better than v1)"


def test_spline_terms_map_to_no_qualifier() -> None:
    """Ratify 11: the six spline terms read x / y only, so ``FEATURE_QUALIFIERS`` maps each to
    ``()``; every feature of every v2 candidate is mapped, so check row 4 can be met by an LR
    headline (``_feature_classes`` finds nothing to refuse on the LR v2 step)."""
    splines = {f"{b}_rcs{k}" for b in ("distance_m", "angle_rad") for k in (1, 2, 3)}
    assert splines == {f for f in ablation.SPLINES if "_rcs" in f}  # precondition
    for f in splines:
        assert v2.FEATURE_QUALIFIERS[f] == (), f
    for spec in (*v2.STEPS, v2.GBM):
        assert set(spec.features) <= set(v2.FEATURE_QUALIFIERS), spec.name
        assert v2._feature_classes(spec.features) == [], spec.name
    assert v2._feature_classes(("distance_m_rcs4",)) == ["distance_m_rcs4 (no qualifier map)"]


def test_extra_qualifier_classes() -> None:
    """Ratify 12: q9 (penalty) and q263 (direct corner) are "pattern" (``phase`` reads them, so
    each id of ``FEATURE_QUALIFIERS["phase"]`` has an allowed class); q83-87 are "post-shot";
    an id nobody listed (q88) stays unclassified, so check row 6 can fail."""
    assert (v2.QUALIFIER_CLASS[9], v2.QUALIFIER_CLASS[263]) == ("pattern", "pattern")
    assert {9, 263} <= set(v2.FEATURE_QUALIFIERS["phase"])  # precondition
    for q in v2.FEATURE_QUALIFIERS["phase"]:
        assert v2.QUALIFIER_CLASS[q] not in v2.REFUSED_CLASSES, q
    assert {v2.QUALIFIER_CLASS[q] for q in range(83, 88)} == {"post-shot"}
    assert 88 not in v2.QUALIFIER_CLASS


def test_v2_checks_signature_and_reporting_rows(
    v2_fit: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratify 13: ``v2_checks`` is keyword-only with ``V2Fit`` field names plus
    ``xgot_features`` (None reads ``xgot_input_features()`` when called); rows 1-6, then exactly
    two reporting rows with result ``n/a`` whose targets say so."""
    fit = v2_fit["fit"]
    params = inspect.signature(v2.v2_checks).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    fields = {f.name for f in dataclasses.fields(fit)}
    assert set(params) - fields == {"xgot_features"}
    assert params["xgot_features"].default is None
    kw = {n: getattr(fit, n) for n in params if n in fields}
    with pytest.raises(TypeError):
        v2.v2_checks(*kw.values())  # type: ignore[misc]
    checks = v2.v2_checks(**kw)
    assert len(checks) == 8
    assert [c.result for c in checks[6:]] == ["n/a", "n/a"]
    assert all(c.target.startswith("Reporting row (not a check)") for c in checks[6:])
    assert all(c.result != "n/a" for c in checks[:6])
    assert v2.v2_checks(**kw, xgot_features=("rebound",))[4].result == "missed"
    assert v2.v2_checks(**kw, xgot_features=("distance_m",))[4].result == "met"
    assert checks[4].result == "met"  # precondition of the read-when-called case
    monkeypatch.setattr(v2, "xgot_input_features", lambda: ("prev_action",))
    assert v2.v2_checks(**kw)[4].result == "missed"


def test_gap_text_na_reaches_card_and_readme(v2_fit: dict[str, Any]) -> None:
    """Ratify 14: when the ablation is not better than v1 (here: the ablation is v1's headline
    itself, denominator 0) ``gap_share`` raises and the card's gap section and the README block
    print ``n/a (the ablation is not better than v1)``; with the fit's own ablation they print a
    percentage."""
    fit = v2_fit["fit"]
    with pytest.raises(ValueError):
        v2.gap_share(fit.v1_head, fit.head, fit.v1_head, "test")
    same = dataclasses.replace(fit, v1_ablation=fit.v1_head)
    card = v2_cards.render(same)
    assert f"- test: {NA_TEXT} = " in card and f"- CV: {NA_TEXT} = " in card
    assert f"closed on test: {NA_TEXT}." in v2_cards.summary_block(same)
    assert v2.gap_text(fit.v1_head, fit.head, fit.v1_head, "cv") == NA_TEXT
    if fit.v1_head.holdout.log_loss > fit.v1_ablation.holdout.log_loss:  # fixture-dependent
        assert NA_TEXT not in v2.gap_text(fit.v1_head, fit.head, fit.v1_ablation, "test")


def test_v1_head_is_v1_gbm_not_v1_headline_rule(v2_fit: dict[str, Any]) -> None:
    """Ratify 15: ``v1_head`` is ``evaluate(v1.GBM)`` (name, title, features of ``v1.GBM``),
    not ``select_headline`` over v1's candidates, which on this fixture picks an LR (the
    precondition that tells the two apart)."""
    fit = v2_fit["fit"]
    spec = fit.v1_head.spec
    assert (spec.name, spec.title, spec.features) == (v1.GBM.name, v1.GBM.title, v1.GBM.features)
    assert models.is_gbm(fit.v1_head)
    v1_lrs = (*fit.candidates.baselines, *fit.candidates.steps[: len(v1.STEPS)])
    lrs = tuple(e for e in v1_lrs if not models.is_gbm(e))
    rule_head, _ = models.select_headline(lrs, fit.v1_head, fit.dev, fit.held)
    assert rule_head is not fit.v1_head  # precondition: the headline rule would pick an LR


@pytest.mark.parametrize(
    ("results", "code"),
    [(("met", "met", "n/a"), 0), (("met", "missed", "n/a"), 1), (("missed", "n/a"), 1),
     (("n/a", "n/a"), 0)],
)  # fmt: skip
def test_main_exits_1_on_a_missed_check(
    results: tuple[str, ...], code: int, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Ratify 16: ``v2_cards.main`` exits 1 when a check row is ``missed`` and 0 otherwise
    (reporting rows are ``n/a``), unlike ``models.main`` / ``v1.main``."""
    fake = SimpleNamespace(checks=tuple(Check("t", "m", r) for r in results))  # type: ignore[arg-type]
    monkeypatch.setattr(v2_cards, "run", lambda *a, **k: fake)
    assert v2_cards.main(["--manifest", str(tmp_path / "m.json"), "--no-cache"]) == code


def _raw(*items: tuple[int, str]) -> str:
    return json.dumps([{"type": {"value": q, "displayName": n}} for q, n in items])


def test_census_once_per_shot_and_name(v2_fit: dict[str, Any]) -> None:
    """Ratify 17: an id repeated on one shot counts once; its name is the most common Opta
    ``displayName`` (ties alphabetical); the fit's census covers v1's population, train plus
    test (per id: train shots = dev shots carrying it, test shots = held shots, recomputed with
    ``v2_cases.qualifier_ids``)."""
    shots = pl.DataFrame(
        {
            "raw_qualifiers": [
                _raw((20, "RightFoot"), (20, "RightFoot"), (22, "Zeta")),
                _raw((20, "RightFoot"), (22, "Alpha")),
                _raw((20, "Right foot"), (22, "Zeta"), (22, "Alpha")),
            ],
            "is_goal": [True, False, False],
            "prev_action": ["pass", "pass", "shot"],
            "rebound": [False, False, True],
        },
        schema_overrides={"prev_action": pl.Enum(v2c.PREV_LEVELS)},
    )
    c = v2.census(shots, pl.Series([True, True, False]))
    rows = {r[0]: r for r in c.qualifiers.iter_rows()}
    assert rows[20][1] == "RightFoot" and rows[20][3:] == (2, 1, 1)  # shot 0 counted once
    assert rows[22][1] == "Alpha" and rows[22][3:] == (2, 1, 1)  # Alpha 2, Zeta 2: alphabetical
    fit = v2_fit["fit"]
    train = Counter(q for raw in fit.dev["raw_qualifiers"] for q in v2c.qualifier_ids(raw))
    test = Counter(q for raw in fit.held["raw_qualifiers"] for q in v2c.qualifier_ids(raw))
    got = {r["id"]: (r["train_shots"], r["test_shots"])
           for r in fit.census.qualifiers.iter_rows(named=True)}  # fmt: skip
    assert got == {q: (train[q], test[q]) for q in train.keys() | test.keys()}
    assert any(n for n, _ in got.values()) and any(n for _, n in got.values())  # both splits


def test_card_sections(v2_fit: dict[str, Any]) -> None:
    """Ratify 18: the card is ``render_card`` with the checks section titled
    ``Milestone checks (PR1)``, the WhoScored validation paragraph (``v2_cards.VALIDATION``,
    naming v1's train and test seasons) and three extra sections in order: test by competition,
    share of the big-chance gap, census of pre-shot information (counts only)."""
    card = v2_cards.render(v2_fit["fit"])
    assert "### Milestone checks (PR1)" in card and "Milestone checks (MS3)" not in card
    assert v2_cards.VALIDATION in card
    for season in (*v1.TRAIN_SEASONS, *v1.TEST_SEASONS):
        assert season in v2_cards.VALIDATION, season
    heads = ["### Test by competition", "### Share of the big-chance gap",
             "### Census of pre-shot information (counts only)"]  # fmt: skip
    at = [card.index(h) for h in heads]
    assert at == sorted(at) and card.count(heads[0]) == 1
