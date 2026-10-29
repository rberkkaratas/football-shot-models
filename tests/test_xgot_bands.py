"""The calibration-by-band table and the band rule (spec ``runs/MS6/xgot-ready`` §3.1, §3.2, §6,
§8; PM E1: the rule is a check on the headline xGOT only).

``band_table(p, dev)`` gives per ``Y_BANDS`` band of rounded |y| (then the points outside the
frame) the shots, goals, goal rate, mean xGOT, gap = Σp − goals, sd = √Σp(1−p) and ``within`` =
|gap| ≤ ``metrics.BAND_SDS`` · sd. Check row 12 of ``xgot.checks`` is met iff every band is within,
and names the worst band by |gap| / sd. Every expected number is worked out by hand in the comments.
Frames are typed column by column in goal-frame metres (``goal_y = 34 − y``).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

import hypothesis.strategies as st
import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xgot_cases as xg
from hypothesis import given, settings

from football_shot_models import metrics

BANDS = ("[0, 1)", "[1, 2)", "[2, 3)", "[3, 3.3)", "[3.3, 3.55)", ">= 3.55",
         "outside the frame + 0.15 m")  # fmt: skip
TABLE_SCHEMA = pl.Schema({
    "band": pl.String(), "shots": pl.Int64(), "goals": pl.Int64(), "goal_rate": pl.Float64(),
    "mean_xgot": pl.Float64(), "gap": pl.Float64(), "sd": pl.Float64(), "within": pl.Boolean(),
})  # fmt: skip
# PM band minimum (MS6 gsax): only bands of at least ``metrics.MIN_JUDGED_SHOTS`` shots are judged
TARGET = ("Σ xGOT − goals within 2 SD (√Σp(1−p)) in every placement band of at least {least} "
          "shots, dev out of fold")  # fmt: skip
# one representative |y| per band (goal-frame metres; the last is beyond the frame + 0.15 m)
AT = {"[0, 1)": 0.5, "[1, 2)": 1.5, "[2, 3)": 2.5, "[3, 3.3)": 3.1, "[3.3, 3.55)": 3.4,
      ">= 3.55": 3.7, "outside the frame + 0.15 m": 5.0}  # fmt: skip


def _xgot() -> Any:
    from football_shot_models import xgot

    return xgot


def _dev(frame_y: Sequence[float], goals: Sequence[bool]) -> pl.DataFrame:
    n = len(frame_y)
    return pl.DataFrame({
        "shot_id": [f"sb:k{i:03d}" for i in range(n)],
        "match_id": [f"sb:m{i % 7}" for i in range(n)],
        "goal_y": pl.Series([34.0 - v for v in frame_y], dtype=pl.Float64()),
        "goal_z": pl.Series([1.0] * n, dtype=pl.Float64()),
        "on_target": pl.Series([True] * n, dtype=pl.Boolean()),
        "is_goal": pl.Series(list(goals), dtype=pl.Boolean()),
    })  # fmt: skip


def _rows(spec: Sequence[tuple[str, float, Sequence[float], int]]
          ) -> tuple[list[float], list[bool], list[float]]:  # fmt: skip
    """``(band, sign, p values, goals)`` groups as frame-y, goal and p lists; goals first."""
    ys, gs, ps = [], [], []
    for band, sign, p, goals in spec:
        for i, q in enumerate(p):
            ys.append(sign * AT[band])
            gs.append(i < goals)
            ps.append(q)
    return ys, gs, ps


# band, side (+1 right / -1 left), p of each shot, goals; listed out of band order
# fmt: off
CASE = [
    (">= 3.55", -1.0, [0.1, 0.3, 0.2], 2),                 # Σp 0.6, gap −1.4, sd √0.46: out
    ("[1, 2)", 1.0, [0.5] * 4, 4),                          # gap −2.0, sd 1.0: at −2 SD, within
    ("outside the frame + 0.15 m", 1.0, [0.9], 1),         # gap −0.1, sd 0.3: within
    ("[0, 1)", -1.0, [0.5] * 4, 0),                         # gap +2.0, sd 1.0: at +2 SD, within
    ("[3.3, 3.55)", 1.0, [0.2, 0.6], 1),                    # gap −0.2, sd √0.4: within
    ("[2, 3)", -1.0, [0.5, 0.5, 0.5, 0.5 + 1e-9], 0),       # gap 2 + 1e-9, sd 1.0: out
]
# band, shots, goals, goal rate, mean xGOT, gap, sd, within  ([3, 3.3) is empty)
WANT = [
    ("[0, 1)", 4, 0, 0.0, 0.5, 2.0, 1.0, True),
    ("[1, 2)", 4, 4, 1.0, 0.5, -2.0, 1.0, True),
    ("[2, 3)", 4, 0, 0.0, 0.5 + 0.25e-9, 2.0 + 1e-9, 1.0, False),
    ("[3, 3.3)", 0, 0, None, None, 0.0, 0.0, True),
    ("[3.3, 3.55)", 2, 1, 0.5, 0.4, -0.2, math.sqrt(0.4), True),
    (">= 3.55", 3, 2, 2 / 3, 0.2, -1.4, math.sqrt(0.46), False),
    ("outside the frame + 0.15 m", 1, 1, 1.0, 0.9, -0.1, 0.3, True),
]
# fmt: on


def test_band_table_values() -> None:
    """§3.1: one row per band in table order (rows given out of band order), the hand-worked
    numbers, an empty band within (0 ≤ 0), and the 2-SD boundary in float: |gap| exactly 2 sd
    is within on both signs, 1e-9 above is not."""
    xgot = _xgot()
    ys, gs, ps = _rows(CASE)
    p = np.array(ps)
    # preconditions: the boundary bands sit on the boundary in float
    assert sum([0.5] * 4) == 2.0 == 2.0 * math.sqrt(sum(q * (1 - q) for q in [0.5] * 4))
    near = [0.5, 0.5, 0.5, 0.5 + 1e-9]
    assert 0 < sum(near) - 2.0 * math.sqrt(sum(q * (1 - q) for q in near)) <= 2e-9
    table = xgot.band_table(p, _dev(ys, gs))
    assert table.schema == TABLE_SCHEMA
    assert metrics.BAND_SDS == 2.0
    assert table["band"].to_list() == list(BANDS)
    for got, want in zip(table.rows(), WANT, strict=True):
        band, n, g, rate, mean, gap, sd, within = want
        assert got[:3] == (band, n, g)
        assert (got[3] is None) == (rate is None) and (got[4] is None) == (mean is None)
        if rate is not None:
            assert got[3] == pytest.approx(rate, abs=1e-12)
            assert got[4] == pytest.approx(mean, abs=1e-12)
        assert got[5] == pytest.approx(gap, abs=1e-12), band
        assert got[6] == pytest.approx(sd, abs=1e-12), band
        assert got[7] is within, band


# fmt: off
ERRORS = {
    "p too short": (lambda p: p[:-1], None, None),
    "p two-dimensional": (lambda p: p.reshape(-1, 1), None, None),
    "NaN in p": (lambda p: np.where(np.arange(p.size) == 2, np.nan, p), None, None),
    "inf in p": (lambda p: np.where(np.arange(p.size) == 2, np.inf, p), None, None),
    "p above 1": (lambda p: np.where(np.arange(p.size) == 2, 1.0000001, p), None, None),
    "p below 0": (lambda p: np.where(np.arange(p.size) == 2, -1e-9, p), None, None),
    "off target row": (None, ("on_target", False), "target"),
    "null goal_y": (None, ("goal_y", None), "goal_y"),
}
# fmt: on


@pytest.mark.parametrize("case", list(ERRORS))
def test_band_table_errors(case: str) -> None:
    """§3.1 Raises: ``p.shape != (dev.height,)``, a non-finite ``p`` or one outside [0, 1], and
    ``features.placement``'s errors (a shot not on target, a null ``goal_y``). The unchanged
    inputs pass (near miss: p at exactly 0 and 1 is allowed)."""
    xgot = _xgot()
    ys, gs, ps = _rows(CASE)
    p, dev = np.array(ps), _dev(ys, gs)
    edge = np.where(np.arange(p.size) == 0, 0.0, np.where(np.arange(p.size) == 1, 1.0, p))
    assert xgot.band_table(edge, dev).height == len(BANDS)
    change_p, change_dev, match = ERRORS[case]
    if change_p is not None:
        p = change_p(p)
    if change_dev is not None:
        column, value = change_dev
        dtype = dev.schema[column]
        dev = dev.with_columns(
            pl.when(pl.int_range(pl.len()) == 2).then(pl.lit(value, dtype=dtype))
            .otherwise(pl.col(column)).alias(column)
        )  # fmt: skip
    with pytest.raises(ValueError, match=match):
        xgot.band_table(p, dev)


@settings(deadline=None, max_examples=60)
@given(
    data=st.lists(
        st.tuples(st.sampled_from(sorted(AT.values())), st.sampled_from([1.0, -1.0]),
                  st.floats(0.0, 1.0), st.booleans()),
        min_size=1, max_size=40,
    )
)  # fmt: skip
def test_band_table_property(data: list[tuple[float, float, float, bool]]) -> None:
    """Totals: the bands partition the shots (shots and goals add up, Σ gap = Σp − goals);
    ``within`` is exactly |gap| ≤ 2 · sd on every row; sd is never NaN."""
    xgot = _xgot()
    ys = [a * s for a, s, _, _ in data]
    p = np.array([q for _, _, q, _ in data])
    goals = [g for *_, g in data]
    table = xgot.band_table(p, _dev(ys, goals))
    assert int(table["shots"].sum()) == len(data)
    assert int(table["goals"].sum()) == sum(goals)
    assert float(table["gap"].sum()) == pytest.approx(float(p.sum()) - sum(goals), abs=1e-9)
    for row in table.iter_rows(named=True):
        assert not math.isnan(row["sd"])
        assert row["within"] is (abs(row["gap"]) <= 2.0 * row["sd"])


def _band_row(frame_y: Sequence[float], goals: Sequence[bool], oof: Sequence[float]) -> Any:
    """Row 12 of ``xgot.checks`` on the 50-shot stand-in inputs (every other row is met)."""
    inputs = xg.check_inputs(frame_y=frame_y, is_goal=goals, oof=oof)
    rows = xg.call_checks(inputs)
    assert len(rows) == 12
    return rows[11]


def _groups(*groups: tuple[str, int, float, int]) -> tuple[list[float], list[bool], list[float]]:
    """``(band, shots, p, goals)`` groups: frame y, goals (first ``goals`` of the group), p."""
    ys, gs, ps = [], [], []
    for band, n, q, g in groups:
        ys += [AT[band]] * n
        gs += [i < g for i in range(n)]
        ps += [q] * n
    assert len(ys) == 50  # the stand-in inputs have 50 dev shots
    return ys, gs, ps


def _measured(text: str, band: str, gap: str, sd: str, judged: int = 7) -> bool:
    """``"<band>: <gap> (SD <sd>); <judged> of 7 bands judged"``; a minus may be ASCII or
    U+2212, a plus optional."""
    sign = "[−-]" if gap.startswith("-") else r"\+?"
    value = re.escape(gap.lstrip("+-"))
    tail = rf"; {judged} of 7 bands judged"
    pattern = rf"{re.escape(band)}: {sign}{value} \(SD {re.escape(sd)}\){tail}"
    return re.fullmatch(pattern, text) is not None


@pytest.fixture
def judge_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 50-shot stand-in inputs cannot reach the PM's 100-shot band minimum: judge every band
    (minimum 0) so the 2-SD rule itself is tested; ``test_band_minimum`` tests the minimum."""
    monkeypatch.setattr(metrics, "MIN_JUDGED_SHOTS", 0)


def test_band_check_row(judge_all: None) -> None:
    """§3.2: the target sentence; ``met`` iff every band is within 2 SD; ``measured`` names the
    worst band by |gap| / sd. The largest |gap| ([0, 1), 5.0) and the smallest sd ([3.3, 3.55))
    are other bands than the worst ([2, 3), ratio 3.16), so a rule that ranks by |gap|, by sd or
    by sd / |gap| names another band."""
    ys, gs, ps = _groups(
        ("[2, 3)", 10, 0.2, 6),  # Σp 2, gap −4.0, sd √1.6 = 1.26, ratio 3.16: out
        ("[0, 1)", 30, 0.5, 20),  # Σp 15, gap −5.0, sd √7.5 = 2.74, ratio 1.83
        ("[3.3, 3.55)", 10, 0.9, 8),  # Σp 9, gap +1.0, sd √0.9 = 0.95, ratio 1.05
    )
    row = _band_row(ys, gs, ps)
    assert row.target == TARGET.format(least=0)
    assert row.result == "missed"
    assert _measured(row.measured, "[2, 3)", "-4.0", "1.3"), row.measured
    ys, gs, ps = _groups(
        ("[2, 3)", 10, 0.2, 2), ("[0, 1)", 30, 0.5, 20), ("[3.3, 3.55)", 10, 0.9, 8)
    )
    row = _band_row(ys, gs, ps)  # [2, 3) now gap 0: the worst is [0, 1), ratio 1.83: met
    assert row.result == "met"
    assert _measured(row.measured, "[0, 1)", "-5.0", "2.7"), row.measured


def test_band_check_row_empty_band(judge_all: None) -> None:
    """§3.2 / §6: an empty band (r = 0) beside the worst band, never ``nan`` in ``measured``; a
    band with sd = 0 and gap ≠ 0 (p = 1, one goal missing) has r = +inf and is the worst; p = 0
    with no goal has sd = gap = 0 (r = 0, within); a tie goes to the first band in table order
    even when its rows come later."""
    ys, gs, ps = _groups(
        ("[3.3, 3.55)", 2, 1.0, 1),  # sd 0, gap +1.0: r = inf, out
        ("[3, 3.3)", 2, 0.0, 0),  # sd 0, gap 0: r = 0
        ("[0, 1)", 46, 0.5, 23),  # gap 0
    )
    row = _band_row(ys, gs, ps)
    assert row.result == "missed"
    assert _measured(row.measured, "[3.3, 3.55)", "+1.0", "0.0"), row.measured
    assert "nan" not in row.measured.lower() and "inf" not in row.measured.lower()
    ys, gs, ps = _groups(
        ("[2, 3)", 4, 0.5, 0),  # gap +2.0, sd 1.0: r = 2.0 (rows first)
        ("[0, 1)", 4, 0.5, 0),  # gap +2.0, sd 1.0: r = 2.0 (first in table order)
        ("[1, 2)", 42, 0.5, 21),  # gap 0
    )
    row = _band_row(ys, gs, ps)
    assert row.result == "met"
    assert _measured(row.measured, "[0, 1)", "+2.0", "1.0"), row.measured


def test_band_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    """PM band minimum (MS6 gsax): a band is judged by the 2-SD rule only with at least
    ``metrics.MIN_JUDGED_SHOTS`` (100) shots; a smaller band out of 2 SD is printed with its counts
    as "not judged (n < 100)" and cannot miss check row 12. Boundary: 99 not judged, 100 judged."""
    from football_shot_models import xgot, xgot_cards

    assert metrics.MIN_JUDGED_SHOTS == 100
    table = pl.DataFrame({"shots": pl.Series([0, 99, 100, 101], dtype=pl.Int64())})
    assert xgot.judged(table).to_list() == [False, False, True, True]
    # [2, 3): 10 shots p 0.2, 6 goals, gap −4.0, sd 1.26: out of 2 SD, the rest within
    ys, gs, ps = _groups(("[2, 3)", 10, 0.2, 6), ("[0, 1)", 40, 0.5, 20))
    row = _band_row(ys, gs, ps)  # the real minimum: no band of the 50 shots is judged
    assert row.target == TARGET.format(least=100)
    assert (row.result, row.measured) == ("met", "no band judged; 0 of 7 bands judged")
    dev = _dev(ys, gs)
    lines = xgot_cards.band_lines(mc.fake_eval(xg.lr_spec(), oof=np.array(ps)), dev)
    out = next(x for x in lines if x.startswith("| [2, 3) "))
    assert out.rstrip(" |").endswith("| not judged (n < 100)"), out
    monkeypatch.setattr(metrics, "MIN_JUDGED_SHOTS", 10)  # [2, 3) at exactly the minimum: judged
    row = _band_row(ys, gs, ps)
    assert row.result == "missed"
    assert _measured(row.measured, "[2, 3)", "-4.0", "1.3", judged=2), row.measured
    monkeypatch.setattr(metrics, "MIN_JUDGED_SHOTS", 11)  # one shot short: not judged, met
    row = _band_row(ys, gs, ps)
    assert row.result == "met" and _measured(row.measured, "[0, 1)", "0.0", "3.2", judged=1)
