"""Season-out xGOT for the rated rows: ``xgot.refit`` and ``xgot.season_out`` (spec
``runs/MS6/gsax`` §3.4, test plan "Season-out xGOT").

The session fixture ``gsax_fits`` (``conftest``) runs ``gsax.fit`` on the ``xgot_fits`` fits with
``N_SPLITS = 2`` and a pass-through spy on the ``evaluate`` name ``xgot`` reads, so every fit of
the season-out chain is recorded with its frames and folds. StatsBomb rates dev rows from
season-grouped folds and the PL 2015/16 hold-out from the dev fit; WhoScored rates 2026/27 from
the train-only fit (``f.predictions`` as it is) and 2025/26 + 2026 from the swapped fit, which
trains on 2026/27 only.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xgot_cases as xg

from football_shot_models import data

KEY = ("competition_id", "season_id")
WS_TEST = ("ws:710", "ws:72627")  # 2026/2027


def _xgot() -> Any:
    from football_shot_models import xgot

    return xgot


def _seasons(frame: pl.DataFrame, idx: np.ndarray | None = None) -> set[tuple[str, str]]:
    rows = frame if idx is None else frame[idx]
    return set(rows.select(KEY).rows())


def _f_names(f: Any) -> set[str]:
    evs = (f.xg_input, f.xg_headline, *f.evaluations, f.head, f.penalty, f.penalty_base)
    return {e.spec.name for e in evs}


def _rated_ids(f: Any) -> set[str]:
    on = f.predictions.filter(pl.col("on_target") & (pl.col("kind") != "shootout"))
    return set(on["shot_id"].to_list())


def test_constants() -> None:
    """§3.4 constants as the spec writes them."""
    xgot = _xgot()
    assert xgot.SEASON_KEY == KEY
    assert pl.Enum(("season_folds", "holdout_fit", "swapped_fit")) == xgot.SOURCE
    assert dict(xgot.SEASON_OUT) == {"statsbomb": "season_folds", "whoscored": "swapped_fit"}
    assert dict(xgot.SUFFIX) == {"season_folds": "-season-out", "swapped_fit": "-swap"}


def test_season_out_statsbomb(gsax_fits: dict[str, Any]) -> None:
    """Every fit of the StatsBomb season-out chain (xG input, xGOT, penalty xGOT) runs on folds
    whose test seasons never appear among their training rows; its spec name ends in
    ``-season-out`` and is never a name ``xgot.fit`` used (no cache entry is shared). Dev rows
    are ``season_folds``, the hold-out rows ``holdout_fit``."""
    f, g = gsax_fits["xgot"]["statsbomb"], gsax_fits["fits"]["statsbomb"]
    calls = gsax_fits["calls"]["statsbomb"]
    assert len(calls) >= 3  # xG input, xGOT headline, penalty xGOT
    names = _f_names(f)
    held_seasons = _seasons(f.inp.held)
    for call in calls:
        name = call["spec"].name
        assert name.endswith("-season-out"), name
        assert name not in names, name
        dev, folds = call["dev"], call["folds"]
        assert len(folds) == 2  # N_SPLITS, one dev season per fold here
        for train, test in folds:
            assert not _seasons(dev, test) & _seasons(dev, train), name
        assert not _seasons(dev) & held_seasons, name  # PL 2015/16 is not a dev season
    pred = g.predictions
    split = pred.select("shot_id", "split").join(
        f.predictions.select("shot_id", f_split="split"), on="shot_id"
    )
    assert split.height == pred.height and (split["split"] == split["f_split"]).all()
    src = pred.select(pl.col("split").cast(pl.String), pl.col("source").cast(pl.String))
    assert set(src.unique().rows()) == {("dev", "season_folds"), ("holdout", "holdout_fit")}


def test_season_out_whoscored_swap(gsax_fits: dict[str, Any]) -> None:
    """2026/27 rows are ``f.predictions`` as they are (``holdout_fit``); 2025/26 and 2026 rows
    are ``swapped_fit``, from fits whose training rows are 2026/27 only and whose names end in
    ``-swap``; ``split`` is ``f.predictions``'."""
    f, g = gsax_fits["xgot"]["whoscored"], gsax_fits["fits"]["whoscored"]
    calls = gsax_fits["calls"]["whoscored"]
    assert len(calls) >= 3
    names = _f_names(f)
    for call in calls:
        name = call["spec"].name
        assert name.endswith("-swap") and name not in names, name
        assert _seasons(call["dev"]) == {WS_TEST}, name
        assert WS_TEST not in _seasons(call["held"]), name
    pred = g.predictions
    is_test = (pl.col("competition_id") == WS_TEST[0]) & (pl.col("season_id") == WS_TEST[1])
    test_rows = pred.filter(is_test).sort("shot_id")
    assert test_rows.height and (test_rows["source"].cast(pl.String) == "holdout_fit").all()
    ours = f.predictions.filter(pl.col("shot_id").is_in(test_rows["shot_id"].implode()))
    ours = ours.sort("shot_id")
    assert test_rows["xgot"].to_list() == ours["xgot"].to_list()
    train_rows = pred.filter(~is_test)
    assert train_rows.height and (train_rows["source"].cast(pl.String) == "swapped_fit").all()
    joined = pred.select("shot_id", "split").join(
        f.predictions.select("shot_id", f_split="split"), on="shot_id"
    )
    assert joined.height == pred.height and (joined["split"] == joined["f_split"]).all()


@pytest.mark.parametrize("p", ["statsbomb", "whoscored"])
def test_season_out_same_rows(gsax_fits: dict[str, Any], p: str) -> None:
    """The rated rows are exactly ``f.predictions``' on-target non-shoot-out rows, each once,
    with the ``xgot.predictions`` columns plus ``competition_name``, ``season_name``, ``family``
    (``data.FAMILY``) and ``source`` (``xgot.SOURCE``), none null; ``xgot`` in [0, 1]."""
    xgot = _xgot()
    f, g = gsax_fits["xgot"][p], gsax_fits["fits"][p]
    pred = g.predictions
    assert pred["shot_id"].n_unique() == pred.height
    assert set(pred["shot_id"].to_list()) == _rated_ids(f)
    assert "shootout" not in set(pred["kind"].cast(pl.String).to_list())
    for c in f.predictions.columns:
        assert c in pred.columns, c
    assert pred.schema["family"] == data.FAMILY and pred.schema["source"] == xgot.SOURCE
    assert pred.schema["competition_name"] == pl.String and pred.schema["season_name"] == pl.String
    for c in ("competition_name", "season_name", "family", "source", "xgot", "gk_id"):
        assert pred[c].null_count() == 0, c
    assert pred["xgot"].is_between(0.0, 1.0).all()
    names = pred.select(*KEY, "season_name").unique()
    matches = f.xg.matches.select(*KEY, "season_name").unique()
    assert set(names.rows()) <= set(matches.rows())


def test_season_out_unknown_provider(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``XgotFit`` whose provider is not in ``SEASON_OUT`` raises, naming it (self-check)."""
    from football_shot_models import evaluation

    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    f = gsax_fits["xgot"]["statsbomb"]
    with pytest.raises(ValueError, match="opta"):
        _xgot().season_out(dataclasses.replace(f, provider="opta"))


def _by_shot(frame: pl.DataFrame, p: np.ndarray) -> dict[str, float]:
    return dict(zip(frame["shot_id"].to_list(), p.tolist(), strict=True))


def _refit_values(
    calls: list[dict[str, Any]], name: str, *, dev: bool, held: bool
) -> dict[str, float]:
    """shot_id -> the spied evaluation's out-of-fold (``dev``) and dev-fit (``held``) xGOT."""
    call = next(c for c in calls if c["spec"].name == name)
    ev, out = call["_result"], {}
    if dev:
        out |= _by_shot(call["dev"], ev.oof)
    if held:
        out |= _by_shot(call["held"], ev.holdout_p)
    return out


def test_season_out_values_are_the_refits(gsax_fits: dict[str, Any]) -> None:
    """Run6-ms6 (QA 1, the leakage boundary): every rated row's ``xgot`` is the season-out
    refit's own prediction, read from the spied evaluations. StatsBomb: the ``-season-out``
    headline's out-of-fold values on dev and its dev fit on the hold-out, the ``-season-out``
    penalty xGOT likewise for kicks. WhoScored: the ``-swap`` fits' hold-out values (the train
    seasons) on ``swapped_fit`` rows. Precondition: each differs from the match out-of-fold
    xGOT of ``f.predictions`` on some row, so rating from ``f.predictions`` fails here."""
    for p, suffix, dev, held in (("statsbomb", "-season-out", True, True),
                                 ("whoscored", "-swap", False, True)):  # fmt: skip
        f, g = gsax_fits["xgot"][p], gsax_fits["fits"][p]
        calls = gsax_fits["calls"][p]
        want = {
            **_refit_values(calls, f.head.spec.name + suffix, dev=dev, held=held),
            **_refit_values(calls, f.penalty.spec.name + suffix, dev=dev, held=held),
        }
        rows = g.predictions
        if p == "whoscored":
            rows = rows.filter(pl.col("source").cast(pl.String) == "swapped_fit")
        assert rows.height and set(rows["shot_id"].to_list()) <= set(want), p
        got = dict(zip(rows["shot_id"].to_list(), rows["xgot"].to_list(), strict=True))
        assert all(got[k] == pytest.approx(want[k], abs=1e-12) for k in got), p
        oof = dict(zip(f.predictions["shot_id"].to_list(), f.predictions["xgot"].to_list(),
                       strict=True))  # fmt: skip
        assert any(abs(got[k] - oof[k]) > 1e-6 for k in got), p  # precondition


def test_season_out_xg_input_keeps_coarse(
    xgot_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 in the season-out chain (run6-ms6 architect A1, QA 5): with one StatsBomb dev match on
    the coarse grid, the ``-season-out`` xG input is fitted on frames that hold it (as
    ``xgot.fit``'s), while the season-out xGOT and penalty xGOT never see it and no rated row
    comes from it."""
    import inspect

    from football_shot_models import evaluation

    xgot = _xgot()
    base = xgot_fits["fits"]["statsbomb"]
    dev_matches = base.inp.on_dev["match_id"].unique().sort().to_list()
    f, mid = None, None
    monkeypatch.setattr(
        xgot, "PROVIDERS", xg.patched_providers(xgot, xgot_fits["manifests"], monkeypatch)
    )
    monkeypatch.chdir(tmp_path)
    for cand in dev_matches:
        monkeypatch.setattr(xgot, "coarse_matches", lambda _c3, m=cand: pl.Series([m]))
        try:
            f = xgot.fit("statsbomb", cache=None)
        except ValueError:  # a fold left without a test row: try the next match
            continue
        mid = cand
        break
    assert f is not None and mid is not None
    hit = pl.col("match_id") == mid
    assert f.frames[0].filter(hit).height and not f.inp.dev.filter(hit).height  # precondition
    assert f.coarse.to_list() == [mid]
    seen: list[dict[str, Any]] = []
    real = xgot.evaluate

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(inspect.signature(real).bind(*args, **kwargs).arguments))
        return real(*args, **kwargs)

    monkeypatch.setattr(xgot, "evaluate", spy)
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    out = xgot.season_out(f)
    by_name = {c["spec"].name: c for c in seen}
    xg_in = by_name[f.xg_input.spec.name + "-season-out"]
    assert xg_in["dev"].filter(hit).height == f.frames[0].filter(hit).height > 0
    for name in (f.head.spec.name, f.penalty.spec.name):
        call = by_name[name + "-season-out"]
        assert not call["dev"].filter(hit).height and not call["held"].filter(hit).height, name
    assert not out.filter(hit).height
