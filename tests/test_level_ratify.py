"""``level`` / ``models.run`` level wiring: one test per build ratify item that states behaviour
(build log ``runs/PR1/xg-context/build.md`` "Architect: ratify" 6-10).

The probe tests run ``level`` on the ``xg_cases`` snapshot (hold-out 20 shots, 8 goals) with
hand-built evaluations; the wiring tests run ``models.run`` once on the ``models_cases``
snapshot with fixture-sized GBMs, ``ADOPTED`` = 1 and the stubbed rule agreeing with it.
Expected values are written or recomputed here, never read from the module under test.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import re
from pathlib import Path
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest

from football_shot_models import data, level, models
from football_shot_models.metrics import Check

MET, NA = Check("t", "m", "met"), Check("t", "m", "n/a")


# --- level_probe on the xg_cases snapshot -----------------------------------------------------


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> tuple[pl.DataFrame, ...]:
    import xg_cases as xc

    manifest = xc.write_fixture_snapshot(tmp_path_factory.mktemp("level-ratify"))
    xg = data.load_xg_data(manifest)
    dev, held = data.split_holdout(xg.shots)
    assert (held.height, int(held["is_goal"].sum())) == (20, 8)  # precondition
    return dev, held, xg.freeze_frames


def _ev(spec: Any, dev: pl.DataFrame, oof: float, p: float) -> Any:
    folds = tuple(mc.scores(n=24, goals=6, xg=6.0, log_loss=0.27) for _ in range(5))
    held_p = np.full(20, p)
    return mc.fake_eval(
        spec, folds=folds, oof=np.full(dev.height, oof), holdout_p=held_p,
        holdout=mc.scores(n=20, goals=8, xg=float(held_p.sum()), log_loss=0.26),
    )  # fmt: skip


def _probe(frames: tuple[pl.DataFrame, ...], held: pl.DataFrame | None = None) -> Any:
    dev, held0, ff = frames
    pooled = _ev(models.POOLED, dev, 0.30, 0.7)
    ind = _ev(models.LEVEL_FIXES[0], dev, 0.28, 0.6)
    head = _ev(models.GBM, dev, 0.25, 0.3)  # under-predicts: Σ xG 6 for 8 goals, z < 0
    return level.level_probe(
        pooled, (ind, head), head, checks_of=lambda ev: (MET, NA), dev=dev,
        held=held0 if held is None else held, freeze_frames=ff,
    )  # fmt: skip


def test_segments_null_level_and_band_order(frames: tuple[pl.DataFrame, ...]) -> None:
    """Ratify 9: a null level is its own segment printed ``null`` (the kind still sums to the
    totals); distance bands come nearest first in ``band_labels()`` order (not alphabetical,
    which would put "(11, 16] m" before "(6, 11] m"); other kinds alphabetically."""
    _, held, _ = frames
    dist = [3.0, 8.0, 13.0, 18.0, 22.0, 30.0] * 3 + [8.0, 13.0]
    moved = held.with_columns(
        pl.Series("distance_m", dist, dtype=pl.Float64),
        pl.when(pl.int_range(pl.len()) == 0).then(None).otherwise(pl.col("body_part"))
        .alias("body_part"),
    )  # fmt: skip
    seg = _probe(frames, moved).segments.filter(pl.col("split") == "hold-out")
    bands = seg.filter(pl.col("kind") == "distance")["level"].to_list()
    assert bands == ["<= 6 m", "(6, 11] m", "(11, 16] m", "(16, 20] m", "(20, 25] m", "> 25 m"]
    body = seg.filter(pl.col("kind") == "body part")
    assert body.filter(pl.col("level") == "null")["n"].to_list() == [1]
    assert int(body["n"].sum()) == 20 and body.height >= 3
    levels = body["level"].to_list()
    assert levels == sorted(levels)  # alphabetical for a kind without bands


def test_level_section_heading_bullets_true_minus(frames: tuple[pl.DataFrame, ...]) -> None:
    """Ratify 10: the section starts ``### Hold-out level (PR1)``; each computed sentence
    (adopted, largest segment, dev OOF lines) is a bullet; a negative z and gap print a true
    minus (the headline: Σ xG 6.0 for 8 goals, gap −2.0, z −2 / √4.2 = −0.98)."""
    probe = _probe(frames)
    rows = tuple(dataclasses.replace(r, adopted=r.title == models.GBM.title) for r in probe.rows)
    assert sum(r.adopted for r in rows) == 1  # precondition: the adopted branch
    lines = level.level_lines(dataclasses.replace(probe, rows=rows), steps_train_exclude=())
    assert lines[0] == "### Hold-out level (PR1)"
    computed = [ln for ln in lines if re.match(r"(- )?(Adopted|Largest|Dev out of fold)", ln)]
    assert len(computed) >= 3 and all(ln.startswith("- ") for ln in computed), computed
    z = -2.0 / np.sqrt(20 * 0.3 * 0.7)
    assert f"{z:+.2f}" == "-0.98"  # precondition
    text = "\n".join(lines)
    assert "−0.98" in text and "−2.0" in text
    assert re.search(r"(?<![\w|])-\d", text) is None, re.findall(r".{20}(?<![\w|])-\d.{5}", text)


# --- models.level_fixes and models.run ---------------------------------------------------------


@pytest.mark.parametrize("adopted", [None, 1])
def test_pooled_evaluated_under_its_own_name(
    adopted: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratify 7: ``level_fixes`` evaluates POOLED itself (its own spec and name) whether or not
    a fix is adopted, even though with ``ADOPTED`` None the headline GBM is POOLED renamed; the
    adopted fix is never evaluated again (its slot is the headline evaluation)."""
    monkeypatch.setattr(models, "ADOPTED", adopted)
    gbm_ev = object()
    seen: list[Any] = []

    def evaluate_spec(spec: Any) -> Any:
        seen.append(spec)
        return ("ev", spec.name)

    pooled, fixes = models.level_fixes(gbm_ev, evaluate_spec)  # type: ignore[arg-type]
    assert pooled == ("ev", "xg-gbm-pooled") and seen[0] is models.POOLED
    if adopted is None:
        assert seen == [models.POOLED, *models.LEVEL_FIXES]
        assert fixes == tuple(("ev", f.name) for f in models.LEVEL_FIXES)
    else:
        assert seen == [models.POOLED, models.LEVEL_FIXES[0]]
        assert fixes[1] is gbm_ev


@pytest.fixture(scope="module")
def adopted_run(models_run: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> Any:
    """``models.run`` with ``ADOPTED`` = 1, the adopted fix (league-trained) named in
    ``DOWNSTREAM_MISSED`` and the rule stubbed to agree; a spy records ``level_probe``'s
    ``skip``."""
    docs = tmp_path_factory.mktemp("adopted") / "docs"
    seen: dict[str, Any] = {}
    real = level.level_probe

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen["skip"] = list(kwargs.get("skip", ()))
        return real(*args, **kwargs)

    def agree(*args: Any, **kwargs: Any) -> int | None:
        return models.ADOPTED

    with pytest.MonkeyPatch.context() as mp:
        mc.patch_level(models, mp, adopted=1, downstream=("xg-gbm-league-trained",))
        mp.setattr(level, "level_probe", spy)
        mp.setattr(level, "adopt", agree)
        with contextlib.redirect_stdout(io.StringIO()):
            result = models.run(models_run["manifest"], docs_dir=docs)
    return {"result": result, "skip": seen["skip"], "card": (docs / "xg-gbm.md")}


def test_downstream_names_map_to_evaluation_names(adopted_run: dict[str, Any]) -> None:
    """Ratify 6: ``run`` passes ``DOWNSTREAM_MISSED`` to ``level_probe`` as the evaluations'
    names: the adopted fix (league-trained) is evaluated as ``xg-gbm``, so ``skip`` holds
    ``xg-gbm``, not the fix's spec name (which no probe row carries)."""
    assert adopted_run["skip"] == ["xg-gbm"]
    names = [r.name for r in adopted_run["result"].level.rows]
    assert names == ["xg-gbm-pooled", "xg-gbm-ts-indicator", "xg-gbm"]


def test_lr_steps_population_line_reads_best_lr(adopted_run: dict[str, Any]) -> None:
    """Ratify 8: ``level_lines`` gets the LR steps' population from the best LR
    (``head_exclude(best_lr)`` = ``()``): with the league-trained fix adopted the GBM card says
    the GBM-over-LR gain mixes training population with model class."""
    run = adopted_run["result"]
    # QA fix: the probe's head is the GBM after the fix (LevelProbe.head), not the run's
    # headline, which is an LR on this fixture (before the fix the LR's predictions filled the
    # segments' "_head" columns and the line below vanished)
    assert run.head.spec.name != "xg-gbm"  # precondition: an LR heads the fixture run
    assert run.level.head.spec.name == "xg-gbm" and run.level.head is run.level.fixes[1]
    card = Path(adopted_run["card"]).read_text(encoding="utf-8")
    assert models.GBM.train_exclude == ("team-selected",)  # restored constant, same as patched
    assert (
        "- The LR steps are trained on all of dev (pooled) and the headline on dev without "
        "team-selected seasons, so the GBM-over-LR gain mixes training population with model "
        "class."
    ) in card
