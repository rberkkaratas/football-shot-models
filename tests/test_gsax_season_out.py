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
from typing import Any

import numpy as np
import polars as pl
import pytest

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
