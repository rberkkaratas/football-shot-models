"""Ratify items of ``runs/MS6/xgot-ready/build.md`` that state behaviour and had no test of their
own: 1 (``XgotFit.cache`` and the run's cache line), 4 (``cards.signed``, a true minus in check
row 12 and the band limitation), 5 (an empty band prints ``n/a``) and 6 (PM E1: the band
limitation bullet on a non-headline card with failing bands, never on the headline card).

Frames are typed column by column in goal-frame metres (``goal_y = 34 − y``); every expected
number is worked out by hand in the comments.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xgot_cases as xg

from football_shot_models import cards

# |y| of two shots per band (sign alternating), the last pair outside the frame + 0.15 m
AY = [0.5, 0.6, 1.5, 1.4, 2.5, 2.6, 3.1, 3.2, 3.4, 3.5, 3.7, 3.8, 5.0, 9.0]
GOAL = [True, False, False, False, True, True, False, True, True, False, False, False, True, True]
# every band within 2 SD but the outside pair: Σp 0.4, 2 goals, gap −1.6, sd √0.30 = 0.548
P_OUT = np.array([0.2, 0.3, 0.4, 0.1, 0.5, 0.7, 0.6, 0.4, 0.5, 0.5, 0.6, 0.4, 0.3, 0.1])
# the outside pair at 0.9 each: gap −0.2, sd √0.18 = 0.42, within; every band within
P_IN = np.where(np.arange(14) >= 12, 0.9, P_OUT)
LIMIT = (
    "Calibration by placement band: 1 of 7 bands are outside 2 SD on dev out of fold (outside "
    "the frame + 0.15 m: 2 shots, 2 goals, Σ xGOT − goals −1.6 (SD 0.5)). This model's "
    "placement terms misprice them; the headline is xGOT GBM, which the band rule judges (PM E1)."
)


def _dev(ay: list[float] = AY, goal: list[bool] = GOAL) -> pl.DataFrame:
    n = len(ay)
    return pl.DataFrame({
        "shot_id": pl.Series([f"sb:r{i:02d}" for i in range(n)], dtype=pl.String()),
        "goal_y": pl.Series([34.0 - (1 - 2 * (i % 2)) * a for i, a in enumerate(ay)],
                            dtype=pl.Float64()),
        "goal_z": pl.Series([1.0] * n, dtype=pl.Float64()),
        "on_target": pl.Series([True] * n, dtype=pl.Boolean()),
        "is_goal": pl.Series(goal, dtype=pl.Boolean()),
    })  # fmt: skip


def _cells(lines: list[str], band: str) -> list[str]:
    row = next(x for x in lines if x.startswith(f"| {band} |"))
    return [c.strip() for c in row.strip("|").split("|")]


# ---- ratify 6 (PM E1) and 4 (true minus) -------------------------------------------------------


def test_band_limit_non_headline_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 6: a card whose model is not the headline and has a band outside 2 SD gets one
    limitation naming each failing band with its shots, goals, signed gap (true minus, ratify 4)
    and SD, and the headline's title; the headline itself gets none even with the same failing
    band, and a non-headline model with every band within gets none."""
    from football_shot_models import xgot, xgot_cards

    dev = _dev()
    table = xgot.band_table(P_OUT, dev)
    assert table.filter(~pl.col("within"))["band"].to_list() == ["outside the frame + 0.15 m"]
    assert bool(xgot.band_table(P_IN, dev)["within"].all())  # precondition of the last case
    lr = mc.fake_eval(xg.lr_spec(), oof=P_OUT)
    head = mc.fake_eval(xg.gbm_spec(), oof=P_IN)
    # PM band minimum: the failing band has 2 shots (< 100), so it is not judged and not listed
    assert xgot_cards.band_limit(lr, head, dev) is None
    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 0)  # judge every band: the rule itself
    assert xgot_cards.band_limit(lr, head, dev) == LIMIT
    assert xgot_cards.band_limit(lr, lr, dev) is None  # the headline: never flagged
    within = mc.fake_eval(xg.lr_spec(), oof=P_IN)
    assert xgot_cards.band_limit(within, head, dev) is None


def test_band_limit_on_rendered_cards(
    xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratify 6 on the rendered cards: the LR and GBM cards of each fixture fit carry the
    ``band_limit`` bullet exactly when it is not None, and the headline card never carries one.
    Precondition: at least one fixture card carries it. The fixture bands hold fewer than the
    PM's 100 shots, so every band is judged here (minimum 0)."""
    from football_shot_models import xgot, xgot_cards

    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 0)

    carried = 0
    for f in xgot_fits["fits"].values():
        written = xgot_cards.render(f)
        for ev in f.evaluations[3:5]:
            card = written[ev.spec.name]
            limit = xgot_cards.band_limit(ev, f.head, f.inp.on_dev)
            if ev is f.head:
                assert limit is None
            assert ("Calibration by placement band: " in card) == (limit is not None), ev.spec.name
            if limit is not None:
                assert f"- {limit}" in card.splitlines(), ev.spec.name
                carried += 1
    assert carried >= 1
    # the real-data shape: the GBM is the headline and the LR misprices a band. The WhoScored
    # fixture's GBM fails [3.3, 3.55); give the LR those predictions and make the GBM the headline
    f = xgot_fits["fits"]["whoscored"]
    lr, gbm = f.evaluations[3:5]
    assert xgot_cards.band_limit(gbm, lr, f.inp.on_dev) is not None  # precondition
    swapped = dataclasses.replace(lr, oof=gbm.oof)
    g = dataclasses.replace(f, evaluations=(*f.evaluations[:3], swapped, gbm), head=gbm)
    written = xgot_cards.render(g)
    limit = xgot_cards.band_limit(swapped, gbm, f.inp.on_dev)
    assert limit is not None and f"- {limit}" in written[lr.spec.name].splitlines()
    assert "Calibration by placement band: " not in written[gbm.spec.name]


def test_check_row_true_minus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 4: check row 12's ``measured`` prints the gap through ``cards.signed``: a true
    minus (U+2212) for a negative gap, a plus for a positive one, one decimal. Every band is
    judged (minimum 0): the 50 stand-in shots cannot reach the PM's 100-shot minimum."""
    from football_shot_models import xgot

    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 0)
    assert cards.signed(-39.44) == "−39.4" and cards.signed(1.0) == "+1.0"
    assert cards.signed(-0.04) == "−0.0"
    # [2, 3): 10 shots p 0.2, 6 goals: gap −4.0, sd √1.6 = 1.26 (the worst); [0, 1): 30 at 0.5,
    # 20 goals: gap −5.0, sd 2.74; [3.3, 3.55): 10 at 0.9, 8 goals: gap +1.0, sd 0.95
    ys = [2.5] * 10 + [0.5] * 30 + [3.4] * 10
    gs = [i < 6 for i in range(10)] + [i < 20 for i in range(30)] + [i < 8 for i in range(10)]
    ps = [0.2] * 10 + [0.5] * 30 + [0.9] * 10
    rows = xg.call_checks(xg.check_inputs(frame_y=ys, is_goal=gs, oof=ps))
    assert rows[11].measured == "[2, 3): −4.0 (SD 1.3); 7 of 7 bands judged"
    # [3.3, 3.55): 2 shots p 1.0, 1 goal: gap +1.0, sd 0 (worst); the rest gap 0
    ys = [3.4] * 2 + [3.1] * 2 + [0.5] * 46
    gs = [True, False, False, False] + [i < 23 for i in range(46)]
    ps = [1.0, 1.0, 0.0, 0.0] + [0.5] * 46
    rows = xg.call_checks(xg.check_inputs(frame_y=ys, is_goal=gs, oof=ps))
    assert rows[11].measured == "[3.3, 3.55): +1.0 (SD 0.0); 7 of 7 bands judged"


# ---- ratify 5 (empty band) and the ASCII band column -------------------------------------------


def test_band_lines_empty_band_cells(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 5: an empty band prints rate, mean xGOT and Σ xGOT − goals as ``n/a``, SD ``0.0``
    and, under the PM's band minimum, ``not judged (n < 100)`` (``yes`` when every band is
    judged); a non-empty band's gap column keeps MS5's ASCII ``+.1f`` (ratify 4)."""
    from football_shot_models import xgot, xgot_cards

    keep = [i for i in range(14) if i not in (6, 7)]  # empty the [3, 3.3) band
    dev = _dev([AY[i] for i in keep], [GOAL[i] for i in keep])
    ev = mc.fake_eval(xg.lr_spec(), oof=P_OUT[keep])
    lines = xgot_cards.band_lines(ev, dev)
    small = "not judged (n < 100)"
    assert _cells(lines, "[3, 3.3)") == ["[3, 3.3)", "0", "0", "n/a", "n/a", "n/a", "0.0", small]
    assert _cells(lines, "outside the frame + 0.15 m")[-1] == small
    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 0)
    lines = xgot_cards.band_lines(ev, dev)
    assert _cells(lines, "[3, 3.3)") == ["[3, 3.3)", "0", "0", "n/a", "n/a", "n/a", "0.0", "yes"]
    out = _cells(lines, "outside the frame + 0.15 m")
    assert out[5] == "-1.6" and out[-1] == "no"  # ASCII minus in the table column
    assert _cells(lines, "[1, 2)")[5] == "+0.5"


# ---- ratify 1 (XgotFit.cache) --------------------------------------------------------------


def test_run_prints_cache_line(
    xgot_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratify 1: ``XgotFit``'s last field is ``cache`` (None for ``fit(p, cache=None)``);
    ``xgot_cards.run`` prints ``cache: {dir} ({hits} read, {written} written)`` from it as its
    last line, and no cache line without one. The fake cache has unequal counts, so swapping
    read and written is caught."""
    from football_shot_models import xgot, xgot_cards

    assert dataclasses.fields(xgot.XgotFit)[-1].name == "cache"
    assert all(f.cache is None for f in xgot_fits["fits"].values())
    fake = SimpleNamespace(dir=tmp_path / "c", hits=["a", "b"], written=["x", "y", "z"])
    base = xgot_fits["fits"]["whoscored"]
    seen: list[object] = []

    def fit(provider: str, *, cache: object) -> Any:
        seen.append(cache)
        return dataclasses.replace(base, cache=fake if cache is not None else None)

    monkeypatch.setattr(xgot, "fit", fit)
    monkeypatch.setattr(xgot_cards, "render", lambda f: {"xgot-stub": "stub\n"})
    for cache, want in (("croot", f"cache: {fake.dir} (2 read, 3 written)"), (None, None)):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = xgot_cards.run("whoscored", docs_dir=tmp_path / "docs", cache=cache)
        last = out.getvalue().splitlines()[-1]
        if want is None:
            assert not any(x.startswith("cache:") for x in out.getvalue().splitlines())
            assert r.cache is None
        else:
            assert last == want and r.cache is fake
    assert seen == ["croot", None]


# ---- run6-ms6 (football 6): the hold-out / test band table on the GBM cards -----------------


@pytest.mark.parametrize("p", ["statsbomb", "whoscored"])
def test_held_band_table_on_gbm_card(
    xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch, p: str
) -> None:
    """The GBM xGOT card (both providers) prints the band table of its hold-out / test
    predictions on the on-target hold-out / test rows (the clean split, PM after MS5), with the
    same rule and minimum (read when called); the LR card does not. Each row's shots, goals and
    Σ xGOT − goals are ``xgot.band_table(gbm.holdout_p, inp.on_held)``."""
    from football_shot_models import xgot, xgot_cards

    f = xgot_fits["fits"][p]
    gbm, lr = f.evaluations[4], f.evaluations[3]
    head = f"### Calibration by placement band, {f.setup.split}"
    monkeypatch.setattr(xgot, "MIN_BAND_SHOTS", 0)
    written = xgot_cards.render(f)
    assert head not in written[lr.spec.name]
    lines = written[gbm.spec.name].splitlines()
    start = lines.index(head)
    body = lines[start:]
    table = xgot.band_table(gbm.holdout_p, f.inp.on_held)
    assert table["shots"].sum() == f.inp.on_held.height > 0  # precondition
    for r in table.iter_rows(named=True):
        cells = _cells(body, r["band"])
        gap = f"{r['gap']:+.1f}" if r["shots"] else "n/a"
        within = "yes" if xgot.within(r["gap"], r["sd"]) else "no"
        assert cells[1:3] == [f"{r['shots']:,}", f"{r['goals']:,}"], r["band"]
        assert cells[5] == gap and cells[-1] == within, r["band"]
