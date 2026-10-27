"""Per-shot predictions, shooting goals added and split-half correlations (spec
``runs/MS5/xgot`` §8; test plan 7).

SGA is ``xGOT − xG`` per shot, with an off-target or blocked shot at 0 xGOT, so its SGA is
``−xG``; ``goals − xG`` sits beside it, and both subtract the same ``xg`` column (the xG input
cross-fitted into xGOT, never the provider's headline xG). It is descriptive: the split-half
correlation says how much of it repeats between alternate matches of a player, nothing more.
Every frame below is hand-written; expected values are computed by hand.
"""

from __future__ import annotations

import datetime as dt
import math
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xgot_cases as xg
from football_core import schema

from football_shot_models import ablation, evaluation

# ---- predictions ---------------------------------------------------------------------------

# shot id, match, player, keeper, outcome, open_goal, input xg
_DEV = (
    ("sb:d1", "sb:m1", "sb:p1", "sb:k2", "goal", False, 0.30),
    ("sb:d2", "sb:m1", "sb:p2", "sb:k1", "saved", True, 0.12),
    ("sb:d3", "sb:m1", "sb:p1", "sb:k2", "off_target", False, 0.05),
    ("sb:d4", "sb:m2", "sb:p2", "sb:k1", "blocked", False, 0.08),
    ("sb:d5", "sb:m2", "sb:p1", "sb:k2", "saved", None, 0.20),
    ("sb:d6", "sb:m2", "sb:p2", None, "off_target", False, 0.03),  # gk_id null off target: ok
)
_HELD = (
    ("sb:h1", "sb:m3", "sb:p3", "sb:k4", "goal", False, 0.40),
    ("sb:h2", "sb:m3", "sb:p3", "sb:k4", "off_target", False, 0.06),
    ("sb:h3", "sb:m3", "sb:p4", "sb:k3", "saved", False, 0.15),
)
_KICKS_DEV = (("sb:k01", "sb:m1", "sb:p1", "sb:k2", "goal", False, None),
              ("sb:k02", "sb:m2", "sb:p2", "sb:k1", "saved", False, None))  # fmt: skip
_KICKS_HELD = (("sb:k03", "sb:m3", "sb:p3", "sb:k4", "goal", False, None),)
_SHOOTOUTS = (("sb:so1", "sb:m2", "sb:p1", "sb:k2", "goal", False, None),
              ("sb:so2", "sb:m2", "sb:p2", "sb:k1", "saved", False, None))  # fmt: skip
XGOT = {"sb:d1": 0.6, "sb:d2": 0.25, "sb:d5": 0.35, "sb:h1": 0.7, "sb:h3": 0.2,
        "sb:k01": 0.8, "sb:k02": 0.7, "sb:k03": 0.75, "sb:so1": 0.78, "sb:so2": 0.72}  # fmt: skip
INPUT_XG = {r[0]: r[6] for r in (*_DEV, *_HELD)}
DATES = {"sb:m1": dt.date(2019, 1, 5), "sb:m2": dt.date(2019, 1, 12),
         "sb:m3": dt.date(2016, 1, 5)}  # fmt: skip


def _shots(rows: tuple[tuple[Any, ...], ...], with_xg: bool) -> pl.DataFrame:
    frame = pl.DataFrame({
        "shot_id": [r[0] for r in rows],
        "match_id": [r[1] for r in rows],
        "player_id": [r[2] for r in rows],
        "competition_id": ["sb:2" if r[1] == "sb:m3" else "sb:11" for r in rows],
        "season_id": ["sb:27" if r[1] == "sb:m3" else "sb:4" for r in rows],
        "gk_id": pl.Series([r[3] for r in rows], dtype=pl.String()),
        "is_goal": [r[4] == "goal" for r in rows],
        "on_target": [r[4] in ("goal", "saved") for r in rows],
        "outcome": pl.Series([r[4] for r in rows], dtype=schema.OUTCOME),
        "open_goal": pl.Series([r[5] for r in rows], dtype=pl.Boolean()),
    })  # fmt: skip
    if with_xg:
        xgv = pl.Series("xg", [r[6] for r in rows], dtype=pl.Float64())
        frame = frame.with_columns(xgv, xg_logit=np.log(xgv / (1 - xgv)))
    return frame


def _matches() -> pl.DataFrame:
    return pl.DataFrame({
        "match_id": list(DATES),
        "match_date": list(DATES.values()),
        "competition_id": ["sb:11", "sb:11", "sb:2"],
        "season_id": ["sb:4", "sb:4", "sb:27"],
        "competition_name": ["Fixture League", "Fixture League", "Premier League"],
        "season_name": ["2018/2019", "2018/2019", "2015/2016"],
    })  # fmt: skip


def _predict(dev_rows: tuple[tuple[Any, ...], ...] = _DEV, matches: pl.DataFrame | None = None
             ) -> pl.DataFrame:  # fmt: skip
    from football_shot_models import xgot

    dev, held = _shots(dev_rows, True), _shots(_HELD, True)
    on_dev, on_held = dev.filter("on_target"), held.filter("on_target")
    inp = xgot.XgInput(dev=dev, held=held, on_dev=on_dev, on_held=on_held,
                       folds=xg.folds_from_tests(on_dev.height, [[0], [1], [2]]),
                       spec_name="xg-input-t")  # fmt: skip
    head = mc.fake_eval(xg.lr_spec(), oof=np.array([XGOT[s] for s in on_dev["shot_id"]]),
                        holdout_p=np.array([XGOT[s] for s in on_held["shot_id"]]))  # fmt: skip
    pen = evaluation.ModelSpec("xgot-penalty", "Penalty xGOT", xg.PENALTY_FEATURES,
                               ablation.make_lr)  # fmt: skip
    kd, kh, so = _shots(_KICKS_DEV, False), _shots(_KICKS_HELD, False), _shots(_SHOOTOUTS, False)
    kev = mc.fake_eval(pen, oof=np.array([XGOT[s] for s in kd["shot_id"]]),
                       holdout_p=np.array([XGOT[s] for s in kh["shot_id"]]))  # fmt: skip
    kicks = xgot.Kicks(dev=kd, held=kh, shootouts=so, ev=kev,
                       shootout_p=np.array([XGOT[s] for s in so["shot_id"]]))  # fmt: skip
    out: pl.DataFrame = xgot.predictions(
        inp, head, kicks, _matches() if matches is None else matches
    )
    return out


def test_predictions_columns() -> None:
    """One row per non-penalty shot (dev and hold-out) and per on-target kick; the spec's dtypes;
    ``xg`` null exactly on kicks, ``xgot`` null exactly on off-target open play; out-of-fold xGOT on
    dev, the dev fit on the hold-out, the penalty xGOT on kicks; ``match_date`` joined from the
    matches table; a missing match, a keeper missing on target or a duplicate id raise."""
    pred = _predict().sort("shot_id")
    assert pred.height == len(_DEV) + len(_HELD) + 3 + 2
    want = {
        "shot_id": pl.String(), "match_id": pl.String(), "player_id": pl.String(),
        "competition_id": pl.String(), "season_id": pl.String(), "match_date": pl.Date(),
        "gk_id": pl.String(), "is_goal": pl.Boolean(), "on_target": pl.Boolean(),
        "outcome": schema.OUTCOME, "open_goal": pl.Boolean(), "xg": pl.Float64(),
        "xgot": pl.Float64(),
    }  # fmt: skip
    for col, dtype in want.items():
        assert pred.schema[col] == dtype, col
    assert isinstance(pred.schema["kind"], pl.Enum) and isinstance(pred.schema["split"], pl.Enum)
    assert pred.schema["kind"].categories.to_list() == ["open_play", "penalty", "shootout"]
    assert pred.schema["split"].categories.to_list() == ["dev", "holdout"]
    rows = {r["shot_id"]: r for r in pred.iter_rows(named=True)}
    for sid, r in rows.items():
        kind = "shootout" if sid.startswith("sb:so") else "penalty" if sid[3] == "k" else (
            "open_play")  # fmt: skip
        assert r["kind"] == kind, sid
        assert r["xgot"] == (pytest.approx(XGOT[sid]) if sid in XGOT else None), sid
        assert r["xg"] == (pytest.approx(INPUT_XG[sid]) if kind == "open_play" else None), sid
        assert r["match_date"] == DATES[r["match_id"]], sid
        assert r["split"] is not None
        for col in ("shot_id", "match_id", "player_id", "competition_id", "season_id", "kind",
                    "is_goal", "on_target", "outcome"):  # fmt: skip
            assert r[col] is not None, (sid, col)
    for sid in ("sb:d1", "sb:d4", "sb:k01", "sb:k02"):
        assert rows[sid]["split"] == "dev"
    for sid in ("sb:h1", "sb:h2", "sb:k03"):
        assert rows[sid]["split"] == "holdout"
    assert rows["sb:d2"]["open_goal"] is True and rows["sb:d5"]["open_goal"] is None
    assert rows["sb:d6"]["gk_id"] is None  # off target: allowed
    assert pred.filter("on_target")["gk_id"].null_count() == 0
    with pytest.raises(ValueError, match="sb:m2|match"):
        _predict(matches=_matches().filter(pl.col("match_id") != "sb:m2"))
    no_keeper = tuple((*r[:3], None, *r[4:]) if r[0] == "sb:d1" else r for r in _DEV)
    with pytest.raises(ValueError, match="gk_id"):
        _predict(dev_rows=no_keeper)
    duplicate = (*_DEV, ("sb:k01", "sb:m1", "sb:p1", "sb:k2", "off_target", False, 0.1))
    with pytest.raises(ValueError, match="shot_id|duplicate"):
        _predict(dev_rows=duplicate)


def test_sga_reads_input_xg() -> None:
    """SGA and goals − xG of open-play rows both subtract the xG input carried in ``xg``; kicks
    are not in ``sga``; no second xG column is carried."""
    from football_shot_models import xgot

    pred = _predict()
    assert [c for c in pred.columns if "xg" in c and c not in ("xg", "xgot")] == []
    out = {r["shot_id"]: r for r in xgot.sga(pred).iter_rows(named=True)}
    assert set(out) == set(INPUT_XG)
    by_hand = {  # xgot − xg, is_goal − xg
        "sb:d1": (0.6 - 0.30, 1 - 0.30), "sb:d2": (0.25 - 0.12, -0.12),
        "sb:d5": (0.35 - 0.20, -0.20), "sb:h1": (0.7 - 0.40, 1 - 0.40),
        "sb:h3": (0.2 - 0.15, -0.15),
    }  # fmt: skip
    for sid, (sga, gax) in by_hand.items():
        assert out[sid]["sga"] == pytest.approx(sga, abs=1e-12), sid
        assert out[sid]["gax"] == pytest.approx(gax, abs=1e-12), sid


def test_sga_off_target_and_blocked() -> None:
    """Off-target and blocked shots count 0 xGOT through the explicit fill, so SGA is −xG;
    ``predictions`` keeps their ``xgot`` null (MS6 must not read it as 0)."""
    from football_shot_models import xgot

    pred = _predict()
    out = {r["shot_id"]: r for r in xgot.sga(pred).iter_rows(named=True)}
    for sid in ("sb:d3", "sb:d4", "sb:d6", "sb:h2"):
        assert pred.filter(pl.col("shot_id") == sid)["xgot"].item() is None
        assert out[sid]["sga"] == pytest.approx(-INPUT_XG[sid], abs=1e-12), sid
        assert out[sid]["gax"] == pytest.approx(-INPUT_XG[sid], abs=1e-12), sid


# ---- split half ----------------------------------------------------------------------------


def _half_frame(spec: dict[str, list[tuple[str, dt.date, list[float]]]]) -> pl.DataFrame:
    """Rows from {key: [(match, date, values)]}, shuffled so no order holds."""
    rows = [(k, m, d, v) for k, ms in spec.items() for m, d, vs in ms for v in vs]
    order = np.random.default_rng(11).permutation(len(rows))
    rows = [rows[i] for i in order]
    return pl.DataFrame({
        "player_id": [r[0] for r in rows],
        "match_id": [r[1] for r in rows],
        "match_date": pl.Series([r[2] for r in rows], dtype=pl.Date()),
        "v": pl.Series([float(r[3]) for r in rows], dtype=pl.Float64()),
    })  # fmt: skip


def _d(month: int, day: int) -> dt.date:
    return dt.date(2020, month, day)


FOUR = {
    "sb:pa": [("sb:a1", _d(1, 1), [1, 0]), ("sb:a3", _d(1, 15), [1, 1, 0]),
              ("sb:a2", _d(1, 8), [0]), ("sb:a4", _d(1, 22), [0, 0])],
    "sb:pb": [("sb:b3", _d(1, 15), [0, 0, 0]), ("sb:b1", _d(1, 1), [1]),
              ("sb:b2", _d(1, 8), [1, 0])],
    # two matches on one date: ranked by match_id (c1 first)
    "sb:pc": [("sb:c2", _d(2, 1), [0]), ("sb:c1", _d(2, 1), [1, 1]),
              ("sb:c3", _d(2, 8), [1, 0, 0, 0])],
    "sb:pd": [("sb:d1", _d(3, 1), [0]), ("sb:d2", _d(3, 8), [1, 1, 1]),
              ("sb:d3", _d(3, 15), [1]), ("sb:d4", _d(3, 22), [0, 1]),
              ("sb:d5", _d(3, 29), [1])],
}  # fmt: skip
ODD = [3 / 5, 1 / 4, 3 / 6, 2 / 3]  # rank 1, 3, 5 matches; Σ value / rows
EVEN = [0.0, 1 / 2, 0.0, 4 / 5]


def test_split_half_r() -> None:
    """Halves alternate a key's matches by date (then match id), so a match never spans halves;
    a key below ``min_rows`` or with an empty half is left out; r is Pearson's r of the halves'
    rates and its 95 % interval is Fisher's z."""
    from football_shot_models import xgot

    spec = {
        **FOUR,
        "sb:pe": [("sb:e1", _d(4, 1), [1]), ("sb:e2", _d(4, 8), [0])],  # 2 rows < min_rows
        "sb:pf": [("sb:f1", _d(4, 1), [1, 0, 1, 0, 1])],  # one match: an empty half
    }
    out = xgot.split_half(_half_frame(spec), by="player_id", value="v", min_rows=4)
    r = float(np.corrcoef(ODD, EVEN)[0, 1])
    assert out.keys == 4
    assert out.r == pytest.approx(r, abs=1e-12)
    z, se = math.atanh(r), 1.0  # 1 / sqrt(4 - 3)
    assert out.low == pytest.approx(math.tanh(z - 1.96 * se), abs=1e-3)
    assert out.high == pytest.approx(math.tanh(z + 1.96 * se), abs=1e-3)
    c = _half_frame(FOUR).filter(pl.col("player_id") == "sb:pc")
    # the tie-break is by match id, so renaming the matches moves the row with value 0 to rank 1
    flipped = xgot.split_half(
        pl.concat([_half_frame(FOUR).filter(pl.col("player_id") != "sb:pc"),
                   c.with_columns(match_id=pl.col("match_id").replace({"sb:c1": "sb:c9"}))]),
        by="player_id", value="v", min_rows=4,
    )  # fmt: skip
    odd_c, even_c = 1 / 5, 2 / 2  # c2 (0) and c3 first and third; c9 (1, 1) second
    want = float(np.corrcoef([3 / 5, 1 / 4, odd_c, 2 / 3], [0.0, 1 / 2, even_c, 4 / 5])[0, 1])
    assert flipped.r == pytest.approx(want, abs=1e-12)
    assert flipped.r != pytest.approx(out.r, abs=1e-6)


def test_split_half_boundaries() -> None:
    """Under 4 keys, or with a constant half (r is NaN), r and the interval are None; r = ±1
    gives an interval [r, r]."""
    from football_shot_models import xgot

    three = {k: v for k, v in FOUR.items() if k != "sb:pd"}
    out = xgot.split_half(_half_frame(three), by="player_id", value="v", min_rows=1)
    assert (out.keys, out.r, out.low, out.high) == (3, None, None, None)
    flat = {k: [(m, d, [0] * len(vs)) if i % 2 else (m, d, vs) for i, (m, d, vs) in
                enumerate(sorted(ms, key=lambda x: (x[1], x[0])))]
            for k, ms in FOUR.items()}  # fmt: skip
    out = xgot.split_half(_half_frame(flat), by="player_id", value="v", min_rows=1)
    assert (out.keys, out.r, out.low, out.high) == (4, None, None, None)
    for sign, evens in ((1.0, [0, 1, 0, 1]), (-1.0, [1, 0, 1, 0])):
        spec = {
            f"sb:q{i}": [(f"sb:q{i}a", _d(5, 1), [o]), (f"sb:q{i}b", _d(5, 8), [e])]
            for i, (o, e) in enumerate(zip([0, 1, 0, 1], evens, strict=True))
        }
        out = xgot.split_half(_half_frame(spec), by="player_id", value="v", min_rows=2)
        assert (out.keys, out.r, out.low, out.high) == (4, sign, sign, sign)


@pytest.mark.parametrize(
    "case", ["no player_id", "no v", "no match_date", "no match_id", "null key", "null value"]
)
def test_split_half_errors(case: str) -> None:
    """A missing column or a null in ``by`` or ``value`` is an error."""
    from football_shot_models import xgot

    frame = _half_frame(FOUR)
    if case.startswith("no "):
        frame = frame.drop(case.removeprefix("no "))
    else:
        col = "player_id" if case == "null key" else "v"
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col(col)).alias(col)
        )
    with pytest.raises(ValueError):
        xgot.split_half(frame, by="player_id", value="v", min_rows=1)
