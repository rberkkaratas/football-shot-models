"""The CH1 keeper and model-card charts against the committed cards on the real snapshots
(``make reproduce``; spec ``runs/CH1/keeper-model-charts`` §5.2, §6, §8).

The xGOT fit is the session's ``conftest.real_xgot`` run (shared with ``test_xgot_run``), so no
extra xGOT fit runs here. Every number a chart draws for a criterion element is compared with
the card row that states it: the xG gain cells (``xg-gbm`` / ``xg-v1``), the xGOT LR row, the
reliability bins, each keeper-season's shrunk total and the † number. In a cold session
(``FSM_CACHE=off``) the run's fit has no cache and keeps no xG candidates, so the xG gain rows
and the keeper rows (a second cold season-out) are checked warm only; the test says so.
``test_fit_spy`` pins PM E1 (a): under the read-only guard the only fit is the one uncached
shoot-out penalty LR per provider.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from conftest import REPO, real_xgot

from football_shot_models import cache, evaluation, gsax, xgot

PROVIDERS = ["statsbomb", pytest.param("whoscored", marks=pytest.mark.whoscored)]
XG_CARD = {"statsbomb": "xg-gbm", "whoscored": "xg-v1"}


def _card(name: str) -> str:
    return (REPO / "docs" / "models" / f"{name}.md").read_text(encoding="utf-8")


def _cell(t: tuple[float, float, float]) -> str:
    return f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"


def _bin_rows(card: str) -> tuple[str, list[tuple[int, int, str, str]]]:
    """The reliability section's sentence and its rows (bin, shots, mean %, rate %)."""
    lines = card.splitlines()
    i = next(k for k, x in enumerate(lines) if re.fullmatch(r"#+ Reliability", x.strip()))
    sentence = next(x for x in lines[i + 1 :] if x.strip())
    rows = []
    for x in lines[i + 1 :]:
        m = re.fullmatch(r"\| (\d+) \| ([\d,]+) \| ([\d.]+%) \| ([\d.]+%) \|", x.strip())
        if m:
            rows.append((int(m[1]), int(m[2].replace(",", "")), m[3], m[4]))
        elif rows:
            break
    return sentence, rows


def _check_bins(bins: pl.DataFrame, card: str) -> None:
    _, rows = _bin_rows(card)
    assert len(rows) == bins.height == 10
    for (b, shots, mean, rate), r in zip(rows, bins.iter_rows(named=True), strict=True):
        assert (b, shots) == (r["bin"], r["shots"])
        assert mean == f"{100 * r['mean_p']:.2f}%" and rate == f"{100 * r['rate']:.2f}%", (b, r)


def _held_card(card: str) -> bool:
    """Whether the card's reliability table is of the scored-once split (``xg-v1``: test)."""
    sentence, _ = _bin_rows(card)
    return "out-of-fold" not in sentence


@pytest.mark.slow
@pytest.mark.parametrize("p", PROVIDERS)
def test_numbers_match_cards(
    real: dict[str, Any], p: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """xG: the gain cells of every drawn reference and the reliability bins equal the
    provider's xG card; xGOT: the LR row and the bins equal the head's card; StatsBomb keepers:
    each shown row's shrunk total is a ``gsax.md`` table value and the † number is the card's
    selected-side GSAx per 100 shots."""
    from football_shot_charts import keepers, render
    from football_shot_models import gsax_cards

    f = real_xgot(real, p)["run"]
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO)
        xv = render.xgot_view(f)
    head_card = _card(f.head.spec.name)
    for row in xv.rows:
        assert f"| {_cell(row.cv)} | {_cell(row.held)} |" in head_card, row
    _check_bins(xv.held_bins if _held_card(head_card) else xv.cv_bins, head_card)
    xg_card = _card(XG_CARD[p])
    if f.cache is None:
        dev, held = f.frames
        ev = f.xg_headline
        y, pp, grp = ((evaluation.labels(held), ev.holdout_p, held["match_id"].to_numpy())
                      if _held_card(xg_card) else
                      (evaluation.labels(dev), ev.oof, dev["match_id"].to_numpy()))  # fmt: skip
        _check_bins(render.calibration_bins(y, pp, grp), xg_card)
        print("xG references: checked warm only")
        print("keepers: checked warm only")
        assert "checked warm only" in capsys.readouterr().out
        return
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO)
        view = render.xg_view(f, render.candidates(f))
    assert view.rows
    for row in view.rows:
        assert f"| {_cell(row.cv)} | {_cell(row.held)} |" in xg_card, row
    _check_bins(view.held_bins if _held_card(xg_card) else view.cv_bins, xg_card)
    if p != "statsbomb":
        return
    g = gsax.fit(f)
    kv = keepers.keeper_view(g, names=None, caveat="", flag=gsax_cards.FLAG,
                             team_gap_100=gsax_cards.team_gap_100(g))  # fmt: skip
    card = _card("gsax")
    lines = card.splitlines()
    at = next(k for k, x in enumerate(lines) if x.startswith("| # | Keeper |"))
    col = [c.strip() for c in lines[at].strip("|").split("|")].index("Shrunken GSAx")
    table = []  # the rows under that header only (the tier table's rows also start "| 100 |")
    for x in lines[at + 2 :]:
        if not re.match(r"\| \d+ \| ", x):
            break
        table.append(x)
    cells = Counter(x.strip("|").split("|")[col].strip().replace("−", "-") for x in table)
    mine = Counter(f"{r['shrunk'] * r['shots'] / 100:+.1f}" for r in kv.rows.iter_rows(named=True))
    assert kv.n == len(table) and mine == cells
    m = re.search(r"selected side: [\d,]+ keeper-seasons, ([\d,]+) shots, GSAx ([+−-][\d.]+)", card)
    gap = gsax_cards.team_gap_100(g)
    if m is None:
        assert gap is None
    else:
        shots, total = int(m[1].replace(",", "")), float(m[2].replace("−", "-"))
        assert gap is not None and abs(gap - 100 * total / shots) <= 100 * 0.05 / shots + 1e-9


@pytest.mark.slow
@pytest.mark.parametrize("p", PROVIDERS)
def test_fit_spy(p: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """PM E1 (a): warm, ``fits(p)`` runs exactly one ``fit_predict`` — the uncached shoot-out
    penalty xGOT LR (``{prefix}penalty``) — and no other estimator fit (``evaluation._fit``
    outside it), writes no cache entry; cold (``FSM_CACHE=off``), ``fits`` refuses
    ``cache=None`` before any fit."""
    from football_shot_charts import render

    manifest = REPO / xgot.PROVIDERS[p].manifest
    if not manifest.is_file():
        pytest.skip(f"needs the {p} snapshot")
    root = cache.cache_root(REPO / "data" / "cache")
    monkeypatch.chdir(REPO)
    if root is None:
        with pytest.raises(ValueError):
            render.fits(p, cache=None)
        print("cold run: fits refuses cache=None (the spy applies warm)")
        return
    calls: list[str] = []
    stray: list[str] = []
    depth: list[int] = []
    real_fp, real_fit = evaluation.fit_predict, evaluation._fit

    def fit_predict(spec: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(spec.name)
        depth.append(1)
        try:
            return real_fp(spec, *args, **kwargs)
        finally:
            depth.pop()

    def fit(spec: Any, *args: Any, **kwargs: Any) -> Any:
        if not depth:
            stray.append(spec.name)
        return real_fit(spec, *args, **kwargs)

    monkeypatch.setattr(evaluation, "fit_predict", fit_predict)
    monkeypatch.setattr(evaluation, "_fit", fit)
    f, g = render.fits(p, cache=root)
    prefix = xgot.PROVIDERS[p].prefix
    assert calls == [f"{prefix}penalty"], calls
    assert stray == []
    assert f.cache is not None and f.cache.written == []
    assert g.xgot is f and np.isfinite(g.model.tau)


def test_charts_paths_ignored() -> None:
    """The real renders' folders are git-ignored (``make charts``, ``make gsax``, the private
    WhoScored folder), so no real chart image can be staged."""
    import subprocess

    for path in ("outputs/charts/gsax-keepers-light.png", "data/charts/gsax-keepers-dark.png",
                 "data/charts/private/ws-calibration-xg-light.png"):  # fmt: skip
        done = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", path], check=False)
        assert done.returncode == 0, path


def test_expect_warm_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§6: the ``FSM_EXPECT_WARM=1`` mechanism (``conftest.expect_warm_guard``) makes a write
    into the guarded cache root fail with a message naming the entry; loads still work (a warm
    run reads); a cache elsewhere (a test's tmp cache) is not guarded."""
    from conftest import expect_warm_guard

    pc = cache.PredictionCache(tmp_path / "cache", "0" * 64)
    expect_warm_guard(monkeypatch, root=tmp_path / "cache")
    with pytest.raises(AssertionError, match="xgot-gbm"):
        pc.save("xgot-gbm", "identity", (), None)  # type: ignore[arg-type]
    assert pc.load("xgot-gbm", "identity", ()) is None  # type: ignore[arg-type]
    assert not (tmp_path / "cache").exists() or not list((tmp_path / "cache").rglob("*.npz"))
    other = cache.PredictionCache(tmp_path / "elsewhere", "0" * 64)
    with pytest.raises(Exception) as info:  # the real save runs (and rejects None predictions)
        other.save("xgot-gbm", "identity", (), None)  # type: ignore[arg-type]
    assert not isinstance(info.value, AssertionError)
