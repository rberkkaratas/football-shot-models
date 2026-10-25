"""Model cards, the CLI and the reproduction from the real snapshot (spec §5, §6)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from sklearn.metrics import brier_score_loss, log_loss

from football_shot_models import baselines, metrics

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
RELIABILITY = "This table stands in for the reliability plot until C7 (F1)."
THRESHOLD_NOTE = (
    "Σ xG and calibration on grouped CV are close to their targets by construction for the "
    "base rate and for an unpenalised logistic regression with an intercept; see the hold-out "
    "column and the reliability table."
)
FILTERS = {  # Data > Filters row label: (shots, goals) in the fixture
    "own goal (Own Goal Against events; not shots)": ("1", "1"),
    "shoot-out (period 5; all are penalties)": ("1", "0"),
    "penalty (in-game)": ("1", "1"),
    "no freeze frame": ("1", "0"),
}
THRESHOLD_LINE = re.compile(r"^(log_loss|brier|auc|xg_error) \S+ (<=|>=) \S+: (MET|MISSED)$")


def _card(docs: Path, baseline: baselines.Baseline) -> str:
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
        for sub in ("### Milestone thresholds", "### Reliability", "### Segments"):
            assert sub in lines


def test_card_text_rules(run_result: tuple[Any, Path]) -> None:
    evaluations, docs = run_result
    for ev in evaluations[:3]:
        text = _card(docs, ev.baseline)
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
    base_metrics = _rows(_section(_card(docs, evaluations[0].baseline), "## Metrics"))
    skill = [r for r in base_metrics if r[0].lower().startswith("log-loss skill")]
    assert skill and skill[0][1:4] == ["0.0000"] * 3  # skill of the base rate against itself


def test_card_reports_missed_threshold(run_result: tuple[Any, Path]) -> None:
    evaluations, docs = run_result
    base, bench = evaluations[0], evaluations[3]
    missed_any = False
    for ev in evaluations[:3]:
        text = _card(docs, ev.baseline)
        log_loss_row = next(r for r in _rows(_section(text, "## Metrics")) if r[0] == "Log loss")
        sd = metrics.fold_sd(ev.folds, "log_loss")
        assert f"{ev.train.log_loss:.4f}" in log_loss_row[1]
        assert f"{ev.cv.log_loss:.4f}" in log_loss_row[2] and f"{sd:.4f}" in log_loss_row[2]
        assert f"{ev.holdout.log_loss:.4f}" in log_loss_row[3]
        block = _section(text, "### Milestone thresholds")
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


SEGMENTS = (
    r"league or tournament", r"team-selected", r"women",
    r"\bfemale\b", r"\bmale\b",
    r"right[_ ]foot", r"left[_ ]foot", r"\bhead\b",
)  # fmt: skip


def test_card_segments(run_result: tuple[Any, Path], dev_held: tuple[pl.DataFrame, Any]) -> None:
    evaluations, docs = run_result
    dev, _ = dev_held
    ev = evaluations[2]
    rows = _rows(_section(_card(docs, ev.baseline), "### Segments"))
    labels = [r[0] for r in rows]
    found = [next(i for i, lab in enumerate(labels) if re.search(p, lab, re.I)) for p in SEGMENTS]
    assert found == list(range(len(SEGMENTS)))  # FAMILIES, then gender, then body part order
    assert len(rows) == len(SEGMENTS)  # body_part=other has no shot: no row
    assert not any(re.search(r"\bother\b", lab, re.I) for lab in labels)
    shots = [r[1] for r in rows]
    assert shots == ["48", "48", "24", "24", "96", "52", "44", "24"]
    women = rows[2]  # one class: no goals, but a log loss and a Brier score
    mask = (dev["family"] == "women").to_numpy()
    p = ev.oof[mask]
    assert women[2] == "0"
    assert women[5] == "n/a"
    assert women[6] == f"{log_loss(np.zeros(p.size), p, labels=[0, 1]):.4f}"
    assert women[7] == f"{brier_score_loss(np.zeros(p.size), p):.4f}"
    assert rows[4][5] != "n/a"


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
    best = baselines.best(evaluations)
    assert f"best: {best.baseline.name}" in out.splitlines()
    lines = [m for m in map(THRESHOLD_LINE.match, out.splitlines()) if m]
    checks = metrics.check_thresholds(best.cv)
    assert [m[1] for m in lines] == [t.metric for t in metrics.THRESHOLDS]
    assert [m[3] == "MET" for m in lines] == [c.passed for c in checks]
    first = {n: (docs / n).read_bytes() for n in names}
    assert first == {n: (run_docs / n).read_bytes() for n in names}  # no path, date or time
    assert baselines.main(_args(manifest, tmp_path, docs)) == 0
    assert "unchanged" in capsys.readouterr().out
    assert {n: (docs / n).read_bytes() for n in names} == first


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
