"""``models``: the level-fix specs, the ADOPTED guard and what ``run`` evaluates (spec
``runs/PR1/xg-context`` §3.2, §3.3).

``run`` goes through the ``models_cases`` snapshot with fixture-sized GBMs
(``models_cases.patch_level``). The ADOPTED-index branch stubs the rule (``level.adopt``) to
agree, so the spy sees the spec names that branch evaluates on any fixture data; the mismatch
test uses the real rule with every fix in ``DOWNSTREAM_MISSED`` (the rule then picks None).
"""

from __future__ import annotations

import contextlib
import functools
import io
from collections import Counter
from pathlib import Path
from typing import Any

import models_cases as mc
import pytest

from football_shot_models import evaluation, gbm, models

TEAM = "team-selected"


def test_level_specs() -> None:
    """§3.3: POOLED is the GBM on ``GBM_FEATURES`` with no training field; the two fixes in the
    brief's order (indicator, then league-trained) differ from it only in name, title and one
    field; with ``ADOPTED`` = 1 the shipped ``GBM`` is LEAGUE renamed ``xg-gbm`` (its title,
    ``train_exclude`` = (team-selected,)), and nothing is refused downstream."""
    p = models.POOLED
    assert (p.name, p.features, p.train_exclude, p.train_indicator) == (
        "xg-gbm-pooled", models.GBM_FEATURES, (), ()
    )  # fmt: skip
    assert p.coefficients is gbm.gbm_settings
    ind, league = models.LEVEL_FIXES
    assert (ind.name, ind.train_indicator, ind.train_exclude) == (
        "xg-gbm-ts-indicator",
        (TEAM,),
        (),
    )
    assert ind.title == "Monotone GBM + Platt, team-selected indicator (0 at prediction)"
    assert (league.name, league.train_exclude, league.train_indicator) == (
        "xg-gbm-league-trained", (TEAM,), ()
    )  # fmt: skip
    assert league.title == "Monotone GBM + Platt, trained without team-selected seasons"
    for fix in (ind, league):
        assert (fix.features, fix.make, fix.coefficients) == (p.features, p.make, p.coefficients)
    assert models.ADOPTED == 1  # §1 F2, pre-registered
    assert models.DOWNSTREAM_MISSED == ()
    g = models.GBM
    assert (g.name, g.title, g.train_exclude, g.train_indicator) == (
        "xg-gbm", league.title, (TEAM,), ()
    )  # fmt: skip
    assert (g.features, g.make) == (p.features, p.make)


def test_specs_downstream_carry_train_exclude() -> None:
    """§3.2, §4 F2 paths: the StatsBomb xGOT input (``xgot.keeper_neutral``) and the season-out
    xG refit (``season_out._renamed``) carry the headline's ``train_exclude``, so they train on
    the same population; the xGOT run reads ``models.GBM`` itself."""
    from football_shot_models import season_out, xgot

    spec = models.GBM
    assert spec.train_exclude == (TEAM,)
    assert "open_goal" in spec.features  # precondition: keeper_neutral builds a new spec
    neutral = xgot.keeper_neutral(spec)
    assert neutral is not spec and "open_goal" not in neutral.features
    assert neutral.train_exclude == spec.train_exclude
    refit = season_out._renamed(neutral, "-so")
    assert (refit.name, refit.train_exclude) == (f"{neutral.name}-so", (TEAM,))
    ind = models.LEVEL_FIXES[0]
    assert season_out._renamed(ind, "-so").train_indicator == (TEAM,)
    assert xgot.PROVIDERS["statsbomb"].candidate_specs()[1] is models.GBM


def _run(models_run: dict[str, Any], tmp_path: Path, mp: pytest.MonkeyPatch) -> Any:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return models.run(models_run["manifest"], docs_dir=tmp_path / "docs")


def test_adopted_mismatch_raises(
    models_run: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.3: ``run`` raises ValueError when the rule's pick differs from ``ADOPTED`` (here the
    rule picks None: every fix is in ``DOWNSTREAM_MISSED``; ADOPTED says 0)."""
    names = tuple(f.name for f in models.LEVEL_FIXES)
    mc.patch_level(models, monkeypatch, adopted=0, downstream=names)
    with pytest.raises(ValueError, match="ADOPTED is out of date"):
        _run(models_run, tmp_path, monkeypatch)


def test_run_evaluates_each_level_spec_once(
    models_run: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§3.3: with ``ADOPTED`` = 1 the run evaluates ``xg-gbm`` (the adopted fix), POOLED and the
    indicator fix once each, and never the adopted fix under its own name (its row reuses the
    headline evaluation); ``ModelsRun.level`` holds the probe with one row per POOLED and fix.
    With ``ADOPTED`` None (the session run) every fix is evaluated and the GBM card carries the
    limitation sentence."""
    from football_shot_models import level

    mc.patch_level(models, monkeypatch, adopted=1)
    seen: list[str] = []
    specs: dict[str, Any] = {}
    real = models.evaluate

    def spy(spec: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(spec.name)
        specs[spec.name] = spec
        return real(spec, *args, **kwargs)

    def agree(*args: Any, **kwargs: Any) -> int | None:
        return models.ADOPTED  # the stubbed rule agrees with ADOPTED = 1

    monkeypatch.setattr(models, "evaluate", spy)
    monkeypatch.setattr(level, "adopt", agree)
    monkeypatch.setattr(models, "adopt", agree, raising=False)
    result = _run(models_run, tmp_path, monkeypatch)
    counts = Counter(n for n in seen if n.startswith("xg-gbm"))
    assert counts == {"xg-gbm": 1, "xg-gbm-pooled": 1, "xg-gbm-ts-indicator": 1}, seen
    # each name is evaluated with its own spec (not POOLED renamed): the training fields reach it
    assert specs["xg-gbm-pooled"] is models.POOLED
    assert specs["xg-gbm-ts-indicator"] is models.LEVEL_FIXES[0]
    assert specs["xg-gbm"] is models.GBM and models.GBM.train_exclude == (TEAM,)
    assert isinstance(result.level, level.LevelProbe)
    assert len(result.level.rows) == 1 + len(models.LEVEL_FIXES)
    assert result.level.fixes[1] is next(e for e in result.evaluations if e.spec.name == "xg-gbm")
    session = models_run["run"].level
    assert len(session.rows) == 3 and not any(r.adopted for r in session.rows)
    card = (models_run["docs"] / "xg-gbm.md").read_text(encoding="utf-8")
    assert "none: the decomposition is a limitation" in card


def test_tiny_level_patch_matches_spec() -> None:
    """Fixture guard: ``patch_level`` rebuilds ``GBM`` exactly as §3.3 writes it (the ADOPTED
    index's fix renamed ``xg-gbm``; POOLED renamed with the plain title when None)."""
    with pytest.MonkeyPatch.context() as mp:
        mc.patch_level(models, mp, adopted=1)
        assert models.GBM.train_exclude == (TEAM,) and models.GBM.name == "xg-gbm"
        assert isinstance(models.GBM.make, functools.partial)
        mc.patch_level(models, mp, adopted=None)
        assert (models.GBM.train_exclude, models.GBM.title) == ((), "Monotone GBM + Platt")
        assert tuple(f.name for f in models.LEVEL_FIXES) == models.DOWNSTREAM_MISSED
    assert isinstance(models.GBM, evaluation.ModelSpec)


def test_level_fixture_census(models_run: dict[str, Any]) -> None:
    """Integration census (PR1.xg-context, workflow v8): the session ``models.run`` reaches every
    branch the level wiring feeds. Dev holds all three families (``models_cases.TS_COPIES``), so
    the league-trained fix drops training rows and its predictions differ from POOLED's; the dev
    360-frame kind has two levels and the hold-out ones one; a segment has
    ``MIN_JUDGED_SHOTS`` + shots (the worst-segment line); the team-selected family's dev OOF line
    has a value; one fix is refused downstream (meets the MS3 rule) and one is not."""
    import numpy as np
    import polars as pl

    from football_shot_models import data, metrics

    probe = models_run["run"].level
    dev = probe.dev
    assert set(dev["family"].cast(pl.String)) == set(data.FAMILIES)
    assert int((dev["family"] == TEAM).sum()) == 36 * len(mc.TS_COPIES)
    league = probe.fixes[[f.name for f in models.LEVEL_FIXES].index("xg-gbm-league-trained")]
    assert not np.allclose(league.oof, probe.pooled.oof)
    seg = probe.segments
    levels = dict(seg.group_by("split", "kind").len().select(
        pl.concat_str("split", "kind", separator="/"), "len").iter_rows())  # fmt: skip
    assert levels["dev OOF/360 frame"] == 2 and levels["dev OOF/family"] == 3
    assert levels["hold-out/360 frame"] == 1  # the one-level line
    assert (seg.filter(pl.col("split") == "hold-out")["n"] >= metrics.MIN_JUDGED_SHOTS).any()
    assert sorted(r.refused for r in probe.rows[1:]) == [False, True]
    card = (models_run["docs"] / "xg-gbm.md").read_text(encoding="utf-8")
    assert "team-selected, under pooled training: n/a" not in card
    assert "Largest headline hold-out segment gap" in card
    for r in probe.rows[1:]:
        assert (f"{r.title} met the MS3 rule but" in card) == r.refused, r.name
