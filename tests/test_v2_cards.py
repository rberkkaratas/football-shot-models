"""``v2_cards``: the WhoScored xG v2 card rendered from a fit object, with no refit (spec
``runs/PR1/xg-context`` §3.4, §4 data policy, §8 chart seam). Reads ``conftest.v2_fit``."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import v2_cases as v2c

from football_shot_models import data, evaluation


def _cards() -> Any:
    from football_shot_models import v2_cards

    return v2_cards


@pytest.fixture(scope="module")
def card(v2_fit: dict[str, Any]) -> str:
    text: str = _cards().render(v2_fit["fit"])
    return text


def test_render_from_fit(v2_fit: dict[str, Any], card: str) -> None:
    """§3.4: the card prints the census (every ``prev_action`` level, the carry line), the gap
    share naming both denominator models by title, every test competition's row, every check
    target, and says v1 stays the WhoScored xGOT input (E4 (a), MS6 E5); rendering twice gives
    the same text."""
    fit = v2_fit["fit"]
    for level in v2c.PREV_LEVELS:
        assert level in card, level
    assert "carry: not recorded by Opta; dribble = TakeOn" in card
    assert fit.v1_head.spec.title in card and fit.v1_ablation.spec.title in card
    assert f"{fit.v1_head.holdout.log_loss:.4f}" in card
    assert f"{fit.v1_ablation.holdout.log_loss:.4f}" in card
    for group in fit.groups:
        assert group.key in card, group.key
    for check in fit.checks:
        assert check.target in card, check.target
    assert re.search(r"v1[^\n]*xGOT|xGOT[^\n]*v1", card)
    assert _cards().render(fit) == card


def test_card_holds_no_ids(v2_fit: dict[str, Any], card: str) -> None:
    """§4 data policy: counts, metrics and Opta vocabulary only: no ``ws:`` prefix and no fixture
    shot, match, team or player id; the README block likewise."""
    xg = data.load_xg_data(v2_fit["manifest"], builder="whoscored")
    block = _cards().summary_block(v2_fit["fit"])
    ids = set(xg.shots["shot_id"]) | set(xg.matches["match_id"]) | set(xg.shots["player_id"])
    ids |= set(xg.matches["home_team_id"]) | set(xg.matches["away_team_id"])
    for text in (card, block):
        assert re.search(r"\bws:", text) is None
        for i in ids:
            assert i.removeprefix("ws:") not in text, i


def test_summary_block(v2_fit: dict[str, Any], card: str) -> None:
    """§3.4: ``summary_block`` is the README text between ``<!-- v2:summary:start/end -->`` (the
    markers live in the README, not in the block) and quotes the headline's test log loss;
    ``main`` on a missing manifest exits 1 with ``fsm: error`` and no traceback."""
    fit = v2_fit["fit"]
    block = _cards().summary_block(fit)
    assert block.strip() and "<!--" not in block
    assert f"{fit.head.holdout.log_loss:.4f}" in block


def test_main_bad_manifest(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """§3.4 / playbook: the CLI checks its input before any fit."""
    code = _cards().main(["--manifest", str(tmp_path / "missing.json"), "--docs-dir",
                          str(tmp_path / "docs"), "--no-cache"])  # fmt: skip
    err = capsys.readouterr().err
    assert code == 1 and "fsm: error" in err and "Traceback" not in err
    assert not (tmp_path / "docs" / "xg-v2.md").exists()


def test_fit_arrays_align_with_frames(v2_fit: dict[str, Any]) -> None:
    """§8 chart seam: every evaluation on ``V2Fit`` has one out-of-fold prediction per
    ``dev.shot_id`` and one hold-out prediction per ``held.shot_id`` (recomputed rows: the
    headline's test Σ xG error by competition sums back to its pooled Σ xG)."""
    fit = v2_fit["fit"]
    evs = (fit.head, fit.v1_head, fit.v1_ablation, *fit.candidates.lrs, fit.candidates.gbm)
    for ev in evs:
        assert ev.oof.size == fit.dev.height and ev.holdout_p.size == fit.held.height
        assert not np.isnan(ev.oof).any()
    assert fit.dev["shot_id"].is_unique().all() and fit.held["shot_id"].is_unique().all()
    y = evaluation.labels(fit.held)
    assert fit.head.holdout.goals == int(y.sum())
    assert sum(g.xg for g in fit.groups) == pytest.approx(float(fit.head.holdout_p.sum()))
    assert sum(g.goals for g in fit.groups) == int(y.sum())
    assert isinstance(fit.held, pl.DataFrame)


def test_render_gbm_headline(v2_fit: dict[str, Any], card: str) -> None:
    """Integration census (PR1.xg-context): the fixture's headline is an LR (its card prints the
    coefficient table), so the GBM-headline branch the real data takes is rendered here from the
    same fit with the GBM as headline: the Model section is the GBM's (settings, Platt), not
    the LR's coefficient table, and the feature rows name the GBM's columns."""
    import dataclasses

    from football_shot_models import models

    fit = v2_fit["fit"]
    gbm_ev = fit.candidates.gbm
    assert fit.head is not gbm_ev and not models.is_gbm(fit.head)  # precondition
    assert "| Term | Standardised | Natural units |" in card
    text = _cards().render(dataclasses.replace(fit, head=gbm_ev))
    assert "`HistGradientBoostingClassifier(" in text and "Platt scaling" in text
    assert "| Term | Standardised | Natural units |" not in text
    assert gbm_ev.spec.title in text and "`rebound`" in text


def test_card_says_the_test_season_chose(v2_fit: dict[str, Any], card: str) -> None:
    """Run7 assessment (football 1, QA 1, football 6): the design probe chose the context set on
    the test season, so the card and the README block name the CV gain over v1 (recomputed cell)
    as the evidence the test did not choose, and never say the test was "never used to choose";
    the split sentence says every train match precedes every test match (checked)."""
    from football_shot_models import cards

    fit = v2_fit["fit"]
    cv = cards.gain_cell(fit.gain_over_v1.cv)
    block = _cards().summary_block(fit)
    for text in (card, block):
        assert "scored its candidates on the test season" in text
        assert (
            f"the CV gain over xG v1 on the training season, which the probe never scored, is {cv}"
            in text
        )
        assert "never used to choose" not in text
    overview = card.split("## Intended use")[0]
    assert cv in overview
    assert "Every train match precedes every test match (checked)." in card
