"""``models.run`` (spec MS3.models §1.10, §3.4, §10): one writer for the step cards, the GBM card
and the penalty card, the headline choice and the MS3 checks, the summary, the CLI; and, in
``make reproduce``, the same run on the real snapshot.

The fast tests share the session fixture ``models_run`` (``conftest``): the ablation fixture
plus penalties, with a fixture-sized GBM grid swapped in for ``models.GBM``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import models_cases as mc
import pytest
import xg_cases as xc
from test_baselines_cards import _header, _rows, _section
from test_cards_gain import GAIN

from football_shot_models import ablation, baselines, cards, data, evaluation, metrics

REPO = Path(__file__).resolve().parents[1]
STEP_NAMES = tuple(s.name for s in ablation.STEPS)
CARD_NAMES = (*STEP_NAMES, "xg-gbm", "xg-penalty")
CHECKS = "### Milestone checks (MS3)"
ARC = (
    "Each constraint holds along one column with every other column held fixed: at a fixed "
    "opening angle, more distance is a move along an arc through both posts."
)


def _gains(r: dict[str, Any]) -> dict[str, Any]:
    """The comparisons ``run`` is specified to make, recomputed from its evaluations."""
    from football_shot_models import models

    run, dev, held = r["run"], r["dev"], r["held"]
    evs = run.evaluations
    gbm, bench = evs[-1], run.benchmark
    best_lr = evaluation.best(evs[:-1])
    lr_gain = evaluation.compare(f"over {best_lr.spec.title}", best_lr, gbm, dev, held)
    head = gbm if lr_gain.cv.low > 0 else best_lr
    s0 = next(e for e in evs if e.spec.name == baselines.BASELINES[2].name)
    return {
        "models": models,
        "gbm": gbm,
        "bench": bench,
        "best_lr": best_lr,
        "lr_gain": lr_gain,
        "head": head,
        "s0": s0,
        "s0_gain": evaluation.compare(f"over {s0.spec.title}", s0, head, dev, held),
        "sb_gain": evaluation.compare("over StatsBomb xG", bench, head, dev, held),
        "sb_gain_gbm": evaluation.compare("over StatsBomb xG", bench, gbm, dev, held),
    }


def _sign_line(column: str, rest: str, text: str) -> bool:
    """``"{column}: {rest}"`` is in ``text``, the column optionally in backticks (Markdown)."""
    return re.search(rf"`?{re.escape(column)}`?: {re.escape(rest)}", text) is not None


def _card(r: dict[str, Any], name: str) -> str:
    return (r["docs"] / f"{name}.md").read_text(encoding="utf-8")


def test_models_run_fixture(models_run: dict[str, Any]) -> None:
    """§1.6, §1.10, §3.2, §3.4: five cards; evaluations = baselines, steps, GBM; the headline by
    the CV rule over the best LR (baselines and steps only); the checks of that headline; every
    model card lists every model and StatsBomb with calibration slopes; the checks table only on
    the headline card."""
    r = models_run
    g = _gains(r)
    models, run = g["models"], r["run"]
    assert sorted(p.name for p in r["docs"].iterdir()) == sorted(f"{n}.md" for n in CARD_NAMES)
    assert [e.spec.name for e in run.evaluations] == [
        *(b.name for b in baselines.BASELINES), *STEP_NAMES, "xg-gbm",
    ]  # fmt: skip
    assert run.benchmark.spec is evaluation.BENCHMARK
    assert run.head is g["head"]
    assert run.checks == models.milestone_checks(
        g["head"], g["bench"], s0_gain=g["s0_gain"], sb_gain=g["sb_gain"],
        gbm_gain=g["lr_gain"], dev=r["dev"], held=r["held"], s0=g["s0"],
    )  # fmt: skip
    assert len(run.checks) == 11
    assert [c.result for c in run.checks[4:8]] == ["n/a"] * 4  # every gender group < 1,000
    titles = [*(e.spec.title for e in run.evaluations), "StatsBomb xG"]
    for name in CARD_NAMES[:-1]:
        card = _card(r, name)
        block = _section(card, "## Benchmarks")
        assert [row[0] for row in _rows(block)] == titles, name
        header = _header(block)
        slopes = [row[header.index("CV cal. slope")] for row in _rows(block)]
        assert slopes == [cards.fmt(e.cv.cal_slope) for e in (*run.evaluations, run.benchmark)]
        lines = card.splitlines()
        assert (CHECKS in lines) == (name == g["head"].spec.name), name
        assert "### Milestone thresholds" not in lines
        overview = _section(card, "## Overview")
        ev = next(e for e in run.evaluations if e.spec.name == name)
        assert models.verdict(ev, g["head"], g["lr_gain"]) in overview, name
        assert "best model" not in overview
    if g["head"].spec.name in CARD_NAMES:
        block = _section(_card(r, g["head"].spec.name), CHECKS)
        assert _rows(block) == [[c.target, c.measured, c.result] for c in run.checks]
    for name in STEP_NAMES:
        changelog = _section(_card(r, name), "## Changelog")
        assert "v1 (MS3): first version." in changelog
        assert (
            "v2 (MS3, M1.4): the benchmark table lists the GBM and calibration slopes; the "
            "overview names the headline model (v1's 'best model' sentence held until the GBM "
            "was added)."
        ) in changelog


def test_models_run_gbm_card(models_run: dict[str, Any]) -> None:
    """§3.4: the GBM card's overview (verdict, then the StatsBomb sentence), features with the
    fitted sign map's lines, settings from the dev fit, gain rows over the best LR and
    StatsBomb, and the wide-DFK and training-mix limitations."""
    r = models_run
    g = _gains(r)
    models, gbm = g["models"], g["gbm"]
    card = _card(r, "xg-gbm")
    assert card.splitlines()[0] == "# Model card: Monotone GBM + Platt"
    overview = _section(card, "## Overview")
    verdict = models.verdict(gbm, g["head"], g["lr_gain"])
    sentence = mc.statsbomb_sentence(models, gbm, g["sb_gain_gbm"], g["bench"])
    assert verdict in overview and sentence in overview
    assert overview.index(verdict) < overview.index(sentence)
    feats = _section(card, "## Features")
    for f in models.GBM_FEATURES:
        assert ablation.FEATURE_ROWS[f] in feats, f
    for column, sign in models.MONOTONE.items():
        word = "non-increasing" if sign < 0 else "non-decreasing"
        assert _sign_line(column, f"{word} — {models.MONOTONE_REASONS[column]}", feats)
    assert ARC in feats
    assert [c[0] for c in gbm.coefficients] == [
        "max_leaf_nodes", "iterations", "platt_intercept", "platt_slope",
        *(f"monotone:{c}" for c in models.MONOTONE),
    ]  # fmt: skip
    model = _section(card, "## Model")
    assert "Tuning and calibration see only the rows of the fit they belong to" in model
    gains = _rows(_section(card, GAIN))
    for row in (g["lr_gain"], g["sb_gain_gbm"]):
        assert [row.label, cards.gain_cell(row.cv), cards.gain_cell(row.holdout)] in gains
    limits = _section(card, "## Limitations & known biases")
    assert ablation.wide_dfk(gbm, g["bench"], r["dev"]) in limits
    assert "The CV population mixes women's and team-selected seasons" in limits
    assert "sb:" not in card and "beats" not in card


def test_models_run_penalties_and_stdout(models_run: dict[str, Any]) -> None:
    """§3.3, §3.4: the penalty rows and spread of ``XgData.penalties``, the penalty card from
    them, one summary line per evaluation and the summary table on stdout."""
    r = models_run
    g = _gains(r)
    models, run = g["models"], r["run"]
    pens = r["xg"].penalties
    assert run.penalties == models.penalty_rows(pens)
    assert run.spread == models.exchangeability(pens)
    context = cards.card_context(r["xg"], r["dev"], r["held"])
    assert _card(r, "xg-penalty") == models.penalty_card(run.penalties, run.spread, context)
    out = r["stdout"]
    for ev in (*run.evaluations, run.benchmark):
        assert baselines.summary_line(ev) in out
    table = models.summary_table(
        run.evaluations, (run.benchmark,), run.head, run.penalties[0], g["sb_gain"]
    )
    assert table.strip() in out
    for check in run.checks:
        assert check.target in out


def test_gbm_card_monotone_lines(models_run: dict[str, Any], tmp_path: Path) -> None:
    """§3.4, §12: the card prints the fitted instance's sign map, not ``MONOTONE``: with a
    two-sign map, a column with a reason gets it, one without gets the sign only, and the
    default's other columns get no line."""
    from football_shot_models import models

    docs = tmp_path / "models"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(models, "GBM", mc.tiny_gbm(models, {"distance_m": -1, "gk_to_shot_m": 1}))
        result = models.run(models_run["manifest"], docs_dir=docs)
    gbm = result.evaluations[-1]
    assert [c for c in gbm.coefficients if c[0].startswith("monotone:")] == [
        ("monotone:distance_m", -1, -1),
        ("monotone:gk_to_shot_m", 1, 1),
    ]
    feats = _section((docs / "xg-gbm.md").read_text(encoding="utf-8"), "## Features")
    reason = models.MONOTONE_REASONS["distance_m"]
    assert _sign_line("distance_m", f"non-increasing — {reason}", feats)
    (line,) = [x for x in feats.splitlines() if re.search(r"gk_to_shot_m`?: non-decreasing", x)]
    assert "—" not in line
    for column in ("angle_rad", "opponents_in_cone", "nearest_opponent_m"):
        assert not re.search(rf"{column}`?: non-", feats)


def test_models_run_deterministic(models_run: dict[str, Any], tmp_path: Path) -> None:
    """Five cards with the same bytes from two more runs, each in a subprocess, with
    ``POLARS_MAX_THREADS=1`` and with the default thread count."""
    code = (
        "import sys\n"
        "import models_cases as mc\n"
        "from football_shot_models import models\n"
        "models.GBM = mc.tiny_gbm(models)\n"
        "models.run(sys.argv[1], docs_dir=sys.argv[2])\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "POLARS_MAX_THREADS"}
    for label, extra in (("one", {"POLARS_MAX_THREADS": "1"}), ("default", {})):
        out = tmp_path / label
        done = subprocess.run(
            [sys.executable, "-c", code, str(models_run["manifest"]), str(out)],
            cwd=Path(__file__).parent, env={**env, **extra}, capture_output=True, text=True,
            check=False,
        )  # fmt: skip
        assert done.returncode == 0, done.stderr
        for name in CARD_NAMES:
            first = (models_run["docs"] / f"{name}.md").read_bytes()
            assert (out / f"{name}.md").read_bytes() == first, (label, name)


def test_models_cli_error_no_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """§3.4, §8: ``main`` as ``baselines.main``; ``ablation`` no longer runs anything and the
    Makefile has ``xg-models`` instead of ``xg-ablation``."""
    from football_shot_models import models

    missing = tmp_path / "manifests" / "nope.json"
    argv = ["--manifest", str(missing), "--docs-dir", str(tmp_path / "cards")]
    assert models.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("fsm: error: ")
    assert "Traceback" not in err
    assert not (tmp_path / "cards").exists() or not any((tmp_path / "cards").iterdir())
    assert not hasattr(ablation, "run") and not hasattr(ablation, "main")
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    assert "xg-models:" in makefile and "football_shot_models.models" in makefile
    assert "xg-ablation" not in makefile


# ---- real snapshot (make reproduce) -----------------------------------------------------------


@pytest.fixture(scope="module")
def real_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One ``models.run`` on the real snapshot (about 8 min), shared by the four slow tests."""
    if not (REPO / "data" / "snapshots" / xc.SNAPSHOT).is_dir():
        pytest.skip("needs the real snapshot (make snapshot)")
    from football_shot_models import models

    manifest = REPO / "manifests" / f"{xc.SNAPSHOT}.json"
    docs = tmp_path_factory.mktemp("real") / "models"
    result = models.run(manifest, docs_dir=docs)
    xg = data.load_xg_data(manifest)
    dev, held = data.split_holdout(xg.shots)
    return {"run": result, "docs": docs, "xg": xg, "dev": dev, "held": held}


@pytest.mark.slow
def test_model_cards_reproduce(real_run: dict[str, Any]) -> None:
    """The five committed cards come back byte for byte from the real snapshot."""
    for name in CARD_NAMES:
        committed = (REPO / "docs" / "models" / f"{name}.md").read_bytes()
        assert (real_run["docs"] / f"{name}.md").read_bytes() == committed, name


@pytest.mark.slow
def test_inclusion_rule_holds(real_run: dict[str, Any]) -> None:
    """MS3.features §1.5, moved to ``models.run``: each step beats the previous one with a CV CI
    above 0; every kept context column and frame group has a drop-one CI above 0; each rejected
    column's add-one CI reaches 0."""
    rows = real_run["run"].rows
    for name in STEP_NAMES:
        over_previous = rows[name][0]
        assert over_previous.cv.low > 0, (name, over_previous)
        assert over_previous.holdout.gain > 0, (name, over_previous)
    variants = [row for name in STEP_NAMES[1:] for row in rows[name]]
    dropped = [row for row in variants if row.label.endswith(": dropped")]
    added = [row for row in variants if row.label.endswith(": added (not in the model)")]
    assert (len(dropped), len(added)) == (8, 2)
    assert all(row.cv.low > 0 for row in dropped)
    assert all(row.cv.low <= 0 for row in added)


@pytest.mark.slow
def test_readme_summary_matches_run(real_run: dict[str, Any]) -> None:
    """The README summary block is the run's ``summary_table``, pasted unchanged."""
    g = _gains(real_run)
    run = real_run["run"]
    table = g["models"].summary_table(
        run.evaluations, (run.benchmark,), run.head, run.penalties[0], g["sb_gain"]
    )
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- models:summary:start -->", "<!-- models:summary:end -->"
    block = readme[readme.index(start) + len(start) : readme.index(end)]
    assert block.strip() == table.strip()
    assert "CI inside" not in readme and "beats StatsBomb" not in readme


@pytest.mark.slow
def test_ms3_checks_on_real_data(real_run: dict[str, Any]) -> None:
    """§2: the GBM is headline; every MS3 check is met except hold-out women (n/a: no women's
    shots in the hold-out); the penalty point estimate is inside 0.76-0.80; the spread and the
    shoot-out row are as measured."""
    run = real_run["run"]
    assert run.head.spec.name == "xg-gbm"
    assert [c.result for c in run.checks] == ["met"] * 7 + ["n/a"] + ["met"] * 3
    pooled = run.penalties[0].rate
    assert (pooled.n, pooled.goals) == (1118, 853)
    assert pooled.mean == pytest.approx(0.762735, abs=1e-6)
    assert 0.76 <= pooled.mean <= 0.80
    assert (pooled.low, pooled.high) == (pytest.approx(0.737384, abs=1e-6),
                                          pytest.approx(0.787196, abs=1e-6))  # fmt: skip
    shootout = next(r for r in run.penalties if r.rate.n == 363)
    assert shootout.rate == metrics.beta_interval(242, 363)
    s = run.spread
    assert (s.groups, s.df, s.top_taker_kicks, s.kicks) == (55, 54, 80, 1118)
    assert s.dispersion == pytest.approx(0.94, abs=0.005)
