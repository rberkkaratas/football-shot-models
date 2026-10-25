"""Model cards, the CLI and the reproduction from the real snapshot (spec §5, §6)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from sklearn.metrics import brier_score_loss, log_loss

from football_shot_models import baselines, cards, data, evaluation, metrics

THRESH = "### MS2 thresholds (retired; see the headline model's MS3 checks)"

REPO = Path(__file__).resolve().parents[1]
LICENCE = (
    "Data: StatsBomb open data (github.com/statsbomb/open-data), credit StatsBomb. Never "
    "committed or redistributed; this card holds aggregates only."
)
CAVEAT = (
    "StatsBomb xG is a reference ceiling, not a like-for-like out-of-sample competitor: "
    "StatsBomb fits it on its own data, which likely includes these matches, and it uses "
    "freeze-frame and other inputs."
)
RELIABILITY = (
    "This table stands in for a reliability plot until a reliability chart is added to `viz` (F1)."
)
THRESHOLD_NOTE = (
    "Σ xG and calibration on grouped CV are close to their targets by construction for the "
    "base rate and for an unpenalised logistic regression with an intercept; see the hold-out "
    "column and the reliability table."
)
FILTERS = {  # Data > Filters row label: (shots, goals) in the fixture
    "own goal (Own Goal Against events; not shots)": ("1", "1"),
    "shoot-out (period 5; all are penalties)": ("1", "0"),
    "penalty (in-game)": ("1", "1"),
    "direct corner (logged as a shot on outcome)": ("0", "0"),  # MS3.features: none here
    "no freeze frame": ("1", "0"),
}
THRESHOLD_LINE = re.compile(r"^(log_loss|brier|auc|xg_error) \S+ (<=|>=) \S+: (MET|MISSED)$")


def _card(docs: Path, baseline: evaluation.ModelSpec) -> str:
    return (docs / f"{baseline.name}.md").read_text(encoding="utf-8")


def _section(text: str, heading: str) -> str:
    """The body under ``heading`` (``## X`` or ``### X``) up to the next heading of that level."""
    level = heading.split(" ")[0]
    lines = text.splitlines()
    start = lines.index(heading) + 1
    end = next(
        (i for i in range(start, len(lines)) if re.match(rf"#{{1,{len(level)}}} ", lines[i])),
        len(lines),
    )
    return "\n".join(lines[start:end])


def _rows(block: str) -> list[list[str]]:
    """Table body rows (header and separator dropped) as stripped cells; ``\\|`` is a literal."""
    rows = [
        [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        for line in block.splitlines()
        if line.lstrip().startswith("|")
    ]
    body = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    return body[1:]


def _header(block: str) -> list[str]:
    """The header cells of the first table in ``block``."""
    line = next(x for x in block.splitlines() if x.lstrip().startswith("|"))
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def _fmt(metric: str, value: float) -> str:
    return f"{value:.2%}" if metric == "xg_error" else f"{value:.4f}"


def test_card_headings_follow_template(run_result: tuple[Any, Path]) -> None:
    template = (REPO / "docs" / "model-card-template.md").read_text(encoding="utf-8")
    headings = [line for line in template.splitlines() if line.startswith("## ")]
    assert len(headings) == 12
    _, docs = run_result
    for baseline in baselines.BASELINES:
        lines = _card(docs, baseline).splitlines()
        assert lines[0] == f"# Model card: {baseline.title}"
        assert [line for line in lines if line.startswith("## ")] == headings
        for sub in (
            THRESH,
            "### Reliability",
            "### Segments",
        ):
            assert sub in lines


def test_card_text_rules(run_result: tuple[Any, Path]) -> None:
    evaluations, docs = run_result
    for ev in evaluations[:3]:
        text = _card(docs, ev.spec)
        assert "sb:" not in text  # no id of any kind
        assert "well calibrated" not in text.lower()
        assert LICENCE in _section(text, "## Data licensing")
        overview = _section(text, "## Overview")
        assert "experimental" in overview
        assert "Baseline for M1.1" in overview
        assert CAVEAT in _section(text, "## Benchmarks")
        assert RELIABILITY in _section(text, "### Reliability")
        assert len(_rows(_section(text, "### Reliability"))) == 10
        assert "v1 (MS2): first baseline." in _section(text, "## Changelog")
        data = _section(text, "## Data")
        assert "xg-exclusions-v1" in data
        filters = {r[0]: tuple(r[1:3]) for r in _rows(data) if r[0] in FILTERS}
        assert filters == FILTERS
    base_metrics = _rows(_section(_card(docs, evaluations[0].spec), "## Metrics"))
    skill = [r for r in base_metrics if r[0].lower().startswith("log-loss skill")]
    assert skill and skill[0][1:4] == ["0.0000"] * 3  # skill of the base rate against itself


def test_card_reports_missed_threshold(run_result: tuple[Any, Path]) -> None:
    evaluations, docs = run_result
    base, bench = evaluations[0], evaluations[3]
    missed_any = False
    for ev in evaluations[:3]:
        text = _card(docs, ev.spec)
        log_loss_row = next(r for r in _rows(_section(text, "## Metrics")) if r[0] == "Log loss")
        sd = metrics.fold_sd(ev.folds, "log_loss")
        assert f"{ev.train.log_loss:.4f}" in log_loss_row[1]
        assert f"{ev.cv.log_loss:.4f}" in log_loss_row[2] and f"{sd:.4f}" in log_loss_row[2]
        assert f"{ev.holdout.log_loss:.4f}" in log_loss_row[3]
        block = _section(text, THRESH)
        assert "hold-out" in block.lower()
        assert THRESHOLD_NOTE in block
        rows = _rows(block)
        checks = metrics.check_thresholds(ev.cv)
        held = metrics.check_thresholds(ev.holdout)
        assert len(rows) == len(metrics.THRESHOLDS) == len(checks)
        for row, cv, ho in zip(rows, checks, held, strict=True):
            assert _fmt(cv.threshold.metric, cv.measured) in row[1]
            assert _fmt(ho.threshold.metric, ho.measured) in row[2]
            assert row[3].lower() == ("met" if cv.passed else "missed")
        missed = not all(c.passed for c in checks)
        missed_any |= missed
        assert ("**Why.**" in text) == missed
        if missed:
            why = text[text.index("**Why.**") :].split("\n\n")[0]
            for value in (
                base.cv.log_loss,
                base.cv.brier,
                base.holdout.log_loss,
                bench.cv.log_loss,
            ):
                assert f"{value:.4f}" in why
            assert f"{base.cv.goals / base.cv.n:.2%}" in why  # the dev goal rate
            assert "they are reported, not tuned to" in why
    assert missed_any  # the base rate's AUC is 0.5, so its card always explains a miss


# run3-ms3 (football 11): family = women is the gender = female row by construction, so it is
# printed once, as the gender row
SEGMENTS = (
    r"league or tournament", r"team-selected",
    r"\bfemale\b", r"\bmale\b",
    r"right[_ ]foot", r"left[_ ]foot", r"\bhead\b",
)  # fmt: skip


def test_card_segments(run_result: tuple[Any, Path], dev_held: tuple[pl.DataFrame, Any]) -> None:
    evaluations, docs = run_result
    dev, _ = dev_held
    ev = evaluations[2]
    block = _section(_card(docs, ev.spec), "### Segments")
    # MS3.features adds a column (Cal. slope (pooled OOF)): cells are read by header name
    header = _header(block)
    ms2 = [
        "Segment", "Shots", "Goals", "Goal rate", "Σ xG", "Σ xG / goals", "Log loss", "Brier",
        "Log-loss skill",
    ]  # fmt: skip
    assert [h for h in header if h in ms2] == ms2 and "Cal. slope (pooled OOF)" in header
    col = {name: i for i, name in enumerate(header)}
    rows = _rows(block)
    labels = [r[0] for r in rows]
    found = [next(i for i, lab in enumerate(labels) if re.search(p, lab, re.I)) for p in SEGMENTS]
    assert found == list(range(len(SEGMENTS)))  # FAMILIES, then gender, then body part order
    assert len(rows) == len(SEGMENTS)  # body_part=other has no shot: no row
    assert not any(re.search(r"\bother\b", lab, re.I) for lab in labels)
    shots = [r[col["Shots"]] for r in rows]
    assert shots == ["48", "48", "24", "96", "52", "44", "24"]
    assert not any(re.search(r"\bwomen\b", lab) for lab in labels)
    assert all(len(r) == len(header) == 10 for r in rows)
    women = rows[2]  # gender = female; one class: no goals, a log loss and a Brier; no skill
    mask = (dev["family"] == "women").to_numpy()
    p = ev.oof[mask]
    assert women[col["Goals"]] == "0"
    assert women[col["Σ xG / goals"]] == "n/a"
    assert women[col["Log loss"]] == f"{log_loss(np.zeros(p.size), p, labels=[0, 1]):.4f}"
    assert women[col["Brier"]] == f"{brier_score_loss(np.zeros(p.size), p):.4f}"
    assert women[col["Log-loss skill"]] == "n/a"
    assert rows[3][col["Σ xG / goals"]] != "n/a"
    # skill against the segment's own goal rate, not the pooled one
    league = (dev["family"] == "league or tournament").to_numpy()
    y = dev["is_goal"].to_numpy()[league].astype(float)
    own = log_loss(y, np.full(y.size, y.mean()), labels=[0, 1])
    skill = f"{1 - log_loss(y, ev.oof[league], labels=[0, 1]) / own:.4f}"
    assert rows[0][col["Log-loss skill"]] == skill


def test_card_fold_xg_error(run_result: tuple[Any, Path]) -> None:
    """The pooled CV Σ xG error cancels fold errors of opposite sign, so the card also prints
    the fold mean and maximum, in the Metrics table and in the threshold row."""
    evaluations, docs = run_result
    for ev in evaluations[:3]:
        mean, worst = metrics.fold_xg_error(ev.folds)
        assert mean == pytest.approx(np.mean([f.xg_error for f in ev.folds]))
        text = _card(docs, ev.spec)
        cell = f"{mean:.2%} (max {worst:.2%})"
        row = next(r for r in _rows(_section(text, "## Metrics")) if "fold mean" in r[0])
        assert row == ["abs(Σ xG − goals) / goals, fold mean (max)", "n/a", cell, "n/a"]
        threshold = _rows(_section(text, THRESH))[3]
        assert f"{ev.cv.xg_error:.2%} pooled; fold mean {cell}" == threshold[1]


def test_card_context_counts(xg_data: Any, dev_held: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """m1 ratify 1: the provenance and population counts, pinned on the fixture."""
    dev, held = dev_held
    c = cards.card_context(xg_data, dev, held)
    assert (c.manifest_name, c.source_revision) == (xc.SNAPSHOT, "0" * 40)
    assert c.data_digest == xg_data.manifest.data_digest
    assert c.holdout_label == "Premier League 2015/2016"
    assert (c.dev_shots, c.dev_matches, c.holdout_shots, c.holdout_matches) == (120, 12, 20, 2)
    assert (c.competition_seasons, c.male_shots, c.female_shots) == (4, 116, 24)
    assert c.exclusions == tuple(
        (r, 0 if r == "direct_corner" else 1, 1 if r in ("own_goal", "penalty") else 0)
        for r in data.EXCLUSION_REASONS
    )  # the shared fixture has no direct corner (MS3.features adds the reason)
    # the hold-out pair (2 matches) and the sb:t13 season (4 matches) are team-selected
    assert (c.team_selected_seasons, c.team_selected_short) == (2, 1)


def test_card_model_text(
    run_result: tuple[Any, Path],
    xg_data: Any,
    dev_held: tuple[pl.DataFrame, pl.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """m1 ratify 7: the base-rate card's Features and Model text; the LR card prints the solver
    settings the fit used, read from make_model(), never a literal."""
    evaluations, docs = run_result
    base = _card(docs, evaluations[0].spec)
    assert _section(base, "## Features").strip() == (
        "None: every shot gets the goal rate of the training shots."
    )
    rate = evaluations[0].train.goals / evaluations[0].train.n
    model = _section(base, "## Model")
    assert f"the goal rate of the training shots ({rate:.2%} on all of dev)" in model
    assert "Nothing else is fitted." in model
    lr = _card(docs, evaluations[1].spec)
    assert "LogisticRegression(C=inf, tol=1e-10, max_iter=1000)" in _section(lr, "## Model")
    monkeypatch.setattr(baselines, "TOL", 1e-8)
    monkeypatch.setattr(baselines, "MAX_ITER", 50)
    context = cards.card_context(xg_data, *dev_held)
    text = baselines.CARDS[evaluations[1].spec.name].model(evaluations[1], context)
    assert "LogisticRegression(C=inf, tol=1e-08, max_iter=50)" in text[0]


def _args(manifest: Path, root: Path, docs: Path) -> list[str]:
    return ["--manifest", str(manifest), "--root", str(root), "--docs-dir", str(docs)]


def test_cli_fixture_end_to_end(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], run_result: tuple[Any, Path]
) -> None:
    manifest = xc.write_fixture_snapshot(tmp_path)
    docs = tmp_path / "cards"
    assert baselines.main(_args(manifest, tmp_path, docs)) == 0
    names = sorted(f"{b.name}.md" for b in baselines.BASELINES)
    assert sorted(p.name for p in docs.iterdir()) == names
    assert (tmp_path / "manifests" / "xg-exclusions-v1.json").is_file()
    out = capsys.readouterr().out
    assert "sb:" not in out
    evaluations, run_docs = run_result
    best = evaluation.best(evaluations)
    assert f"best: {best.spec.name}" in out.splitlines()
    lines = [m for m in map(THRESHOLD_LINE.match, out.splitlines()) if m]
    checks = metrics.check_thresholds(best.cv)
    assert [m[1] for m in lines] == [t.metric for t in metrics.THRESHOLDS]
    assert [m[3] == "MET" for m in lines] == [c.passed for c in checks]
    mean, worst = metrics.fold_xg_error(best.folds)
    assert f"xg_error fold mean {mean:.4f}, max {worst:.4f} (the line above is pooled)" in out
    first = {n: (docs / n).read_bytes() for n in names}
    assert first == {n: (run_docs / n).read_bytes() for n in names}  # no path, date or time
    assert baselines.main(_args(manifest, tmp_path, docs)) == 0
    assert "snapshot xg-exclusions-v1: unchanged\n" in capsys.readouterr().out
    assert {n: (docs / n).read_bytes() for n in names} == first
    # same data, different manifest bytes: say so, as `fc snapshot` does
    log_manifest = tmp_path / "manifests" / "xg-exclusions-v1.json"
    raw = json.loads(log_manifest.read_text(encoding="utf-8"))
    raw["environment"]["football_core_git"] = "f" * 40
    log_manifest.write_text(json.dumps(raw), encoding="utf-8")
    assert baselines.main(_args(manifest, tmp_path, docs)) == 0
    assert "snapshot xg-exclusions-v1: unchanged (manifest updated)\n" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_make_reproduce_fails_without_snapshot(tmp_path: Path) -> None:
    """A fresh clone has no snapshot: ``make reproduce`` must fail, not pass with a skip."""
    out = subprocess.run(
        ["make", "-C", str(tmp_path), "-f", str(REPO / "Makefile"), "reproduce"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode != 0
    assert "no snapshot at data/snapshots/statsbomb-open-v1" in out.stderr


@pytest.mark.parametrize(
    "case", ["missing manifest", "tampered data", "other builder", "no convergence"]
)
def test_cli_errors_exit_1(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    if case == "missing manifest":
        manifest = tmp_path / "manifests" / "nope.json"
    else:
        manifest = xc.write_fixture_snapshot(
            tmp_path, builder="parquet" if case == "other builder" else "statsbomb"
        )
    if case == "tampered data":
        target = tmp_path / "data" / "snapshots" / xc.SNAPSHOT / "matches.parquet"
        raw = bytearray(target.read_bytes())
        raw[len(raw) // 2] ^= 0x01
        target.write_bytes(bytes(raw))
    if case == "no convergence":
        monkeypatch.setattr(baselines, "MAX_ITER", 1)
    assert baselines.main(_args(manifest, tmp_path, tmp_path / "cards")) == 1
    err = capsys.readouterr().err
    assert err.startswith("fsm: error: ")
    assert "Traceback" not in err
    if case == "no convergence":
        assert "xg-lr-distance-angle: LogisticRegression did not converge in 1 iterations" in err


@pytest.mark.slow
def test_cards_reproduce_from_snapshot(tmp_path: Path) -> None:
    """``make reproduce``: the committed cards come back byte for byte from the real snapshot."""
    if not (REPO / "data" / "snapshots" / "statsbomb-open-v1").is_dir():
        pytest.skip("needs the real snapshot (make snapshot)")
    manifest = REPO / "manifests" / "statsbomb-open-v1.json"
    baselines.run(manifest, root=tmp_path, docs_dir=tmp_path / "models")
    for baseline in baselines.BASELINES:
        name = f"{baseline.name}.md"
        assert (tmp_path / "models" / name).read_bytes() == (
            REPO / "docs" / "models" / name
        ).read_bytes()
