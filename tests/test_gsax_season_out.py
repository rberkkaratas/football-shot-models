"""Season-out xGOT for the rated rows: ``season_out.refit``, ``season_out.season_out``,
``fold_seasons`` and ``trained_seasons`` (spec ``runs/MS6/gsax`` §3.4; ``runs/PR1/keeper-fixes``
§2, §9 "SEASON_OUT['whoscored'] == 'season_folds'"; PM 2026-10-28 and PM E2 (c), PR1).

The session fixture ``gsax_fits`` (``conftest``) runs ``gsax.fit`` on the ``xgot_fits`` fits with
``N_SPLITS = 2`` and a pass-through spy on the ``evaluate`` name ``season_out`` reads, so every
fit of the season-out chain is recorded with its frames and folds. Both providers rate dev rows
from competition-season folds and the hold-out / test rows from that refit's dev fit
(``holdout_fit``): StatsBomb's 2 dev seasons and PL 2015/16, WhoScored's 2025/26 and 2026 (train)
and 2026/27 (test). The year swap is gone (PM 2026-10-28).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import xgot_cases as xg
from conftest import WS_TRAIN

from football_shot_models import data

KEY = ("competition_id", "season_id")
WS_TEST = ("ws:710", "ws:72627")  # 2026/2027
TRAINED = ("competition_id", "season_id", "trained_competition_id", "trained_season_id")


def _so() -> Any:
    from football_shot_models import season_out

    return season_out


def _seasons(frame: pl.DataFrame, idx: np.ndarray | None = None) -> set[tuple[str, str]]:
    rows = frame if idx is None else frame[idx]
    return set(rows.select(KEY).rows())


def _f_names(f: Any) -> set[str]:
    evs = (f.xg_input, f.xg_headline, *f.evaluations, f.head, f.penalty, f.penalty_base)
    return {e.spec.name for e in evs}


def _rated_ids(f: Any, kinds: tuple[str, ...] = ("open_play", "penalty")) -> set[str]:
    on = f.predictions.filter(pl.col("on_target") & pl.col("kind").cast(pl.String).is_in(kinds))
    return set(on["shot_id"].to_list())


def _main_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The calls of the headline chain (not of an E2 (c) ``-no-<names>`` refit)."""
    return [c for c in calls if "-no-" not in c["spec"].name]


def test_constants() -> None:
    """§2 constants as the spec writes them: one mode for both providers (PM 2026-10-28), no
    ``swapped_fit`` source or ``-swap`` suffix."""
    so = _so()
    assert so.SEASON_KEY == KEY
    assert pl.Enum(("season_folds", "holdout_fit")) == so.SOURCE
    assert dict(so.SEASON_OUT) == {"statsbomb": "season_folds", "whoscored": "season_folds"}
    assert dict(so.SUFFIX) == {"season_folds": "-season-out"}


def test_season_out_statsbomb(gsax_fits: dict[str, Any]) -> None:
    """Every fit of the StatsBomb season-out chain (xG input, xGOT, penalty xGOT) runs on folds
    whose test seasons never appear among their training rows; its spec name ends in
    ``-season-out`` and is never a name ``xgot.fit`` used (no cache entry is shared). Dev rows
    are ``season_folds``, the hold-out rows ``holdout_fit``."""
    f, g = gsax_fits["xgot"]["statsbomb"], gsax_fits["fits"]["statsbomb"]
    calls = _main_calls(gsax_fits["calls"]["statsbomb"])
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


def test_season_out_whoscored(gsax_fits: dict[str, Any]) -> None:
    """PM 2026-10-28: WhoScored runs the competition-season folds. Each fit of the chain trains
    on the train seasons only (never 2026/27) with 2 folds whose (train, test) *season sets* are
    ({2026}, {2025/26}) and ({2025/26}, {2026}); names end in ``-season-out``. The 2025/26 and
    2026 rows are ``season_folds``, the 2026/27 rows ``holdout_fit`` (spec O2: the refit's dev
    fit, not ``f.predictions``); ``split`` is ``f.predictions``'."""
    f, g = gsax_fits["xgot"]["whoscored"], gsax_fits["fits"]["whoscored"]
    calls = _main_calls(gsax_fits["calls"]["whoscored"])
    assert len(calls) >= 3
    names = _f_names(f)
    a, b = WS_TRAIN
    want = {(frozenset({b}), frozenset({a})), (frozenset({a}), frozenset({b}))}
    for call in calls:
        name = call["spec"].name
        assert name.endswith("-season-out") and name not in names, name
        assert not name.endswith("-swap"), name
        dev = call["dev"]
        assert _seasons(dev) == set(WS_TRAIN), name
        assert _seasons(call["held"]) == {WS_TEST}, name
        got = {(frozenset(_seasons(dev, tr)), frozenset(_seasons(dev, te)))
               for tr, te in call["folds"]}  # fmt: skip
        assert got == want, (name, got)
    pred = g.predictions
    is_test = (pl.col("competition_id") == WS_TEST[0]) & (pl.col("season_id") == WS_TEST[1])
    src = pred.select(is_test.alias("t"), pl.col("source").cast(pl.String))
    assert set(src.unique().rows()) == {(True, "holdout_fit"), (False, "season_folds")}
    joined = pred.select("shot_id", "split").join(
        f.predictions.select("shot_id", f_split="split"), on="shot_id"
    )
    assert joined.height == pred.height and (joined["split"] == joined["f_split"]).all()


@pytest.mark.parametrize("p", ["statsbomb", "whoscored"])
def test_season_out_same_rows(gsax_fits: dict[str, Any], p: str) -> None:
    """The rated rows are exactly ``f.predictions``' on-target non-shoot-out rows, each once,
    with the ``xgot.predictions`` columns plus ``competition_name``, ``season_name``, ``family``
    (``data.FAMILY``) and ``source`` (``season_out.SOURCE``), none null; ``xgot`` in [0, 1]."""
    so = _so()
    f, g = gsax_fits["xgot"][p], gsax_fits["fits"][p]
    pred = g.predictions
    assert pred["shot_id"].n_unique() == pred.height
    assert set(pred["shot_id"].to_list()) == _rated_ids(f)
    assert "shootout" not in set(pred["kind"].cast(pl.String).to_list())
    for c in f.predictions.columns:
        assert c in pred.columns, c
    assert pred.schema["family"] == data.FAMILY and pred.schema["source"] == so.SOURCE
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
        _so().season_out(dataclasses.replace(f, provider="opta"))


def test_season_out_other_mode_raises(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """§2: a ``SEASON_OUT`` value other than ``"season_folds"`` raises ``ValueError`` naming it
    (the swap branch is deleted, not kept behind the constant)."""
    from types import MappingProxyType

    from football_shot_models import evaluation

    so = _so()
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    monkeypatch.setattr(so, "SEASON_OUT", MappingProxyType(
        {"statsbomb": "season_folds", "whoscored": "swapped_fit"}))  # fmt: skip
    with pytest.raises(ValueError, match="swapped_fit"):
        so.season_out(gsax_fits["xgot"]["whoscored"])


def test_season_out_leaky_fold_raises(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run7 assessment (QA 4), ``Raises``: a fold whose train rows hold one of its test seasons
    (``_folds`` patched to train fold 0 on every dev row) → ValueError before any fit."""
    so = _so()
    f = gsax_fits["xgot"]["statsbomb"]

    def leaky(dev: pl.DataFrame) -> Any:
        every = np.arange(dev.height)
        return ((every, every[:1]),)

    assert f.frames[0].height > 1  # precondition: fold 0's test row is also a train row
    monkeypatch.setattr(so, "_folds", leaky)
    monkeypatch.setattr(so, "refit", lambda *a, **k: pytest.fail("refit reached"))
    with pytest.raises(ValueError, match=r"open-play fold 0 trains on 1 test season"):
        so.season_out(f)


def test_season_out_held_season_in_dev_raises(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run7 assessment (QA 4), ``Raises``: a rated hold-out season that is also a training
    season of the fit that scores it (``f.frames``' held given a dev row) → ValueError."""
    from football_shot_models import evaluation

    so = _so()
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    monkeypatch.setattr(so, "refit", lambda *a, **k: pytest.fail("refit reached"))
    f = gsax_fits["xgot"]["statsbomb"]
    dev, held = f.frames
    leaked = pl.concat([held, dev.head(1)], how="vertical")
    assert _seasons(dev.head(1)) <= _seasons(dev)  # precondition: the added season trains
    with pytest.raises(ValueError, match=r"1 hold-out season\(s\) are in the fit that rates them"):
        so.season_out(dataclasses.replace(f, frames=(dev, leaked)))


def test_season_out_two_families_raises(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run7 assessment (QA 4), ``Raises``: a competition-season with two families → ValueError.
    The refit is replaced by ``f``'s own predictions (the guard reads only ids, keys and
    families); near miss: with the real ``match_family`` the same call passes."""
    from football_shot_models import evaluation

    so = _so()
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    f = gsax_fits["xgot"]["statsbomb"]
    monkeypatch.setattr(so, "refit", lambda *a, **k: f.predictions)
    assert so.season_out(f).height  # near miss: one family per competition-season
    real = data.match_family

    def split(matches: pl.DataFrame) -> pl.DataFrame:
        out = real(matches)
        alt = pl.int_range(pl.len()) % 2 == 0
        other = pl.when(pl.col("family") == data.FAMILIES[0]).then(pl.lit(data.FAMILIES[1]))
        return out.with_columns(
            family=pl.when(alt).then(other.otherwise(pl.lit(data.FAMILIES[0])))
            .otherwise(pl.col("family")).cast(data.FAMILY)
        )  # fmt: skip

    rated = f.predictions.filter(pl.col("on_target"))
    per = rated.join(f.xg.matches.select("match_id", *KEY), on="match_id").group_by(KEY)
    assert per.agg(pl.col("match_id").n_unique())["match_id"].max() >= 2  # precondition
    monkeypatch.setattr(data, "match_family", split)
    with pytest.raises(ValueError, match="two families"):
        so.season_out(f)


def _by_shot(frame: pl.DataFrame, p: np.ndarray) -> dict[str, float]:
    return dict(zip(frame["shot_id"].to_list(), p.tolist(), strict=True))


def _refit_values(calls: list[dict[str, Any]], name: str) -> dict[str, float]:
    """shot_id -> the spied evaluation's out-of-fold (dev) and dev-fit (hold-out) xGOT."""
    call = next(c for c in calls if c["spec"].name == name)
    ev = call["_result"]
    return _by_shot(call["dev"], ev.oof) | _by_shot(call["held"], ev.holdout_p)


@pytest.mark.parametrize("p", ["statsbomb", "whoscored"])
def test_season_out_values_are_the_refits(gsax_fits: dict[str, Any], p: str) -> None:
    """Run6-ms6 (QA 1, the leakage boundary), both providers since PR1: every rated row's
    ``xgot`` is the season-out refit's own prediction, read from the spied evaluations: the
    ``-season-out`` headline's out-of-fold values on dev and its dev fit on the hold-out / test
    season, the ``-season-out`` penalty xGOT likewise for kicks. Precondition: each provider
    has rows of both sources, and some value differs from the match out-of-fold xGOT of
    ``f.predictions`` (on WhoScored's test season too: O2), so rating from ``f.predictions``
    fails here."""
    f, g = gsax_fits["xgot"][p], gsax_fits["fits"][p]
    calls = gsax_fits["calls"][p]
    suffix = "-season-out"
    want = _refit_values(calls, f.head.spec.name + suffix)
    want |= _refit_values(calls, f.penalty.spec.name + suffix)
    rows = g.predictions
    assert set(rows["source"].cast(pl.String).unique().to_list()) == {"season_folds", "holdout_fit"}
    assert set(rows["shot_id"].to_list()) <= set(want), p
    got = dict(zip(rows["shot_id"].to_list(), rows["xgot"].to_list(), strict=True))
    assert all(got[k] == pytest.approx(want[k], abs=1e-12) for k in got), p
    oof = dict(zip(f.predictions["shot_id"].to_list(), f.predictions["xgot"].to_list(),
                   strict=True))  # fmt: skip
    held = rows.filter(pl.col("source").cast(pl.String) == "holdout_fit")["shot_id"].to_list()
    assert any(abs(got[k] - oof[k]) > 1e-6 for k in got), p  # precondition
    assert any(abs(got[k] - oof[k]) > 1e-6 for k in held), p  # O2: not f.predictions


def test_season_out_xg_input_keeps_coarse(
    xgot_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1 in the season-out chain (run6-ms6 architect A1, QA 5): with one StatsBomb dev match on
    the coarse grid, the ``-season-out`` xG input is fitted on frames that hold it (as
    ``xgot.fit``'s), while the season-out xGOT and penalty xGOT never see it and no rated row
    comes from it."""
    import inspect

    from football_shot_models import evaluation, xgot

    so = _so()
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
    real = so.evaluate

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(inspect.signature(real).bind(*args, **kwargs).arguments))
        return real(*args, **kwargs)

    monkeypatch.setattr(so, "evaluate", spy)
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    out = so.season_out(f)
    by_name = {c["spec"].name: c for c in seen}
    xg_in = by_name[f.xg_input.spec.name + "-season-out"]
    assert xg_in["dev"].filter(hit).height == f.frames[0].filter(hit).height > 0
    for name in (f.head.spec.name, f.penalty.spec.name):
        call = by_name[name + "-season-out"]
        assert not call["dev"].filter(hit).height and not call["held"].filter(hit).height, name
    assert not out.filter(hit).height


# ---- fold_seasons / trained_seasons (§2; the E4 count reads them) ---------------------------

# X = (c1, s1), Y = (c1, s2), Z = (c2, s1): X and Y share a competition, X and Z a season id.
_X, _Y, _Z, _H = ("sb:c1", "sb:s1"), ("sb:c1", "sb:s2"), ("sb:c2", "sb:s1"), ("sb:c9", "sb:s9")


def _keyed(keys: list[tuple[str, str]]) -> pl.DataFrame:
    return pl.DataFrame(keys, schema={"competition_id": pl.String, "season_id": pl.String},
                        orient="row").with_columns(row=pl.int_range(pl.len()))  # fmt: skip


def _fold_case() -> tuple[pl.DataFrame, pl.DataFrame, tuple[tuple[np.ndarray, np.ndarray], ...]]:
    """dev: X 3 rows, Y 1, Z 2, out of key order; held: H 2 rows. Fold 0 tests X and trains on
    Y only (Z in neither: not every other season trains); fold 1 tests Y and Z, trains on X."""
    dev = _keyed([_Z, _X, _Y, _X, _Z, _X])
    held = _keyed([_H, _H])
    folds = (
        (np.array([2], dtype=np.intp), np.array([1, 3, 5], dtype=np.intp)),
        (np.array([1, 3, 5], dtype=np.intp), np.array([0, 2, 4], dtype=np.intp)),
    )
    return dev, held, folds


def test_fold_seasons() -> None:
    """One row per (rated season, training season): fold i's test seasons × its train seasons,
    and each held season × every dev season; String columns, none null, unique, sorted. Seasons
    sharing a competition or a season id stay apart."""
    dev, held, folds = _fold_case()
    out = _so().fold_seasons(dev, held, folds)
    assert out.columns == list(TRAINED)
    assert all(out.schema[c] == pl.String for c in TRAINED)
    want = sorted([(*_X, *_Y), (*_Y, *_X), (*_Z, *_X), (*_H, *_X), (*_H, *_Y), (*_H, *_Z)])
    assert out.rows() == want
    assert out.null_count().sum_horizontal().item() == 0


def _no_test() -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """A partition with a fold that has train rows but no test row (fold 0)."""
    a = lambda *v: np.array(v, dtype=np.intp)  # noqa: E731
    return (a(1, 3, 5), a()), (a(2), a(1, 3, 5)), (a(1, 3, 5), a(0, 2, 4))


@pytest.mark.parametrize(
    "case", ["not a partition", "no train row", "no test row", "missing key", "null dev key",
             "null held key"],
)  # fmt: skip
def test_fold_seasons_errors(case: str) -> None:
    """§2 / §7 ``Raises``: folds that are not a partition of ``dev``; a fold with no train or no
    test row; a missing key column (named); a null key in ``dev`` or ``held`` (count named)."""
    dev, held, folds = _fold_case()
    match = None
    if case == "not a partition":
        folds = (folds[0],)  # Y and Z rows are in no test fold
    elif case == "no train row":
        folds = ((np.array([], dtype=np.intp), folds[0][1]), folds[1])
    elif case == "no test row":
        folds = _no_test()
    elif case == "missing key":
        dev, match = dev.drop("season_id"), "season_id"
    else:
        frame = dev if case == "null dev key" else held
        frame = frame.with_columns(season_id=pl.when(pl.col("row") < 2).then(None)
                                   .otherwise(pl.col("season_id")))  # fmt: skip
        dev, held = (frame, held) if case == "null dev key" else (dev, frame)
        match = r"\b2\b"
    with pytest.raises(ValueError, match=match):
        _so().fold_seasons(dev, held, folds)


@pytest.mark.parametrize("p", ["statsbomb", "whoscored"])
def test_trained_seasons_reads_chain_folds(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch, p: str
) -> None:
    """``trained_seasons(f)`` is ``fold_seasons(*f.frames, season_folds(f.frames[0]))``: the
    chain's competition-season folds (coarse-grid matches included), never ``f.inp.folds`` (match
    folds, where a season trains its own rows). Pinned on the fixture: each dev season trained on
    the other, the hold-out / test season on both, and no season on itself."""
    from football_shot_models import evaluation

    so = _so()
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    f = gsax_fits["xgot"][p]
    out = so.trained_seasons(f)
    dev, held = f.frames
    assert out.equals(so.fold_seasons(dev, held, evaluation.season_folds(dev, by=KEY)))
    a, b = sorted(_seasons(dev))
    (h,) = _seasons(held)
    assert out.rows() == sorted([(*a, *b), (*b, *a), (*h, *a), (*h, *b)])
    itself = (pl.col("competition_id") == pl.col("trained_competition_id")) & (
        pl.col("season_id") == pl.col("trained_season_id")
    )
    assert not out.filter(itself).height


def test_season_out_without_mouth_z(
    gsax_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """PM E2 (c), PR1: ``season_out(f, without=("mouth_z",))`` refits the open-play headline
    season-out without ``mouth_z`` (its name ends ``-no-mouth_z`` after ``-season-out``; its
    other features are the headline's), rates the open-play rows only (no penalty xGOT refit:
    the penalty xGOT reads ``mouth_z`` too), on the same competition-season folds; its values
    differ from the headline season-out values (precondition: ``mouth_z`` matters)."""
    import inspect

    from football_shot_models import evaluation

    so = _so()
    monkeypatch.setattr(evaluation, "N_SPLITS", 2)
    f, g = gsax_fits["xgot"]["statsbomb"], gsax_fits["fits"]["statsbomb"]
    seen: list[dict[str, Any]] = []
    real = so.evaluate

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(dict(inspect.signature(real).bind(*args, **kwargs).arguments))
        return real(*args, **kwargs)

    monkeypatch.setattr(so, "evaluate", spy)
    out = so.season_out(f, without=("mouth_z",))
    head = [c for c in seen if c["spec"].name.startswith(f.head.spec.name)]
    assert len(head) == 1, [c["spec"].name for c in seen]
    spec = head[0]["spec"]
    assert spec.name.endswith("-no-mouth_z") and "-season-out" in spec.name, spec.name
    assert spec.features == tuple(c for c in f.head.spec.features if c != "mouth_z")
    assert "mouth_z" in f.head.spec.features  # precondition
    for train, test in head[0]["folds"]:
        assert not _seasons(head[0]["dev"], test) & _seasons(head[0]["dev"], train)
    assert not [c for c in seen if c["spec"].name.startswith(f.penalty.spec.name)]
    assert set(out["shot_id"].to_list()) == _rated_ids(f, ("open_play",))
    assert set(out["kind"].cast(pl.String).unique().to_list()) == {"open_play"}
    assert out.schema["source"] == so.SOURCE
    base = dict(zip(g.predictions["shot_id"].to_list(), g.predictions["xgot"].to_list(),
                    strict=True))  # fmt: skip
    assert any(abs(x - base[s]) > 1e-6 for s, x in out.select("shot_id", "xgot").rows())
