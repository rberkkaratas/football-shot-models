"""The on-target population and its reconciliation with C3 (spec ``runs/MS5/xgot`` §2, §3, §9;
test plan 4; PM X2).

On target is a goal or a keeper save that is not an own goal (C3 ``on_target``). Every C3
on-target shot lands in exactly one reconciliation row: its exclusion-log reason, else
``unresolved_assist`` (WhoScored rows ``v1.population`` drops), else ``modelled``. Shoot-out and
in-game penalty kicks on target are in the population, scored by the penalty xGOT. ``shots`` and
``goals`` count the rows actually scored. The hand-written frames hold every case the brief names:
own goal, deflected goal, StatsBomb "Saved Off Target" (C3 ``off_target``), a clearance off the
line (C3 ``blocked``), a direct corner on target, shoot-out kicks on and off target, in-game
penalties on and off target, a no-frame save (StatsBomb) and an unresolved-assist save
(WhoScored).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import xgot_cases as xg

from football_shot_models import cards, data, v1

# shot id, on target, goal, exclusion reason or None, pred kind or None (not scored)
_COMMON: tuple[tuple[str, bool, bool, str | None, str | None], ...] = (
    ("s01", False, True, "own_goal", None),  # own goal: never on target
    ("s02", True, True, None, "open_play"),  # deflected goal: modelled
    ("s03", False, False, None, "open_play"),  # StatsBomb Saved Off Target -> C3 off_target
    ("s04", False, False, None, "open_play"),  # clearance off the line -> C3 blocked
    ("s05", True, True, "direct_corner", None),  # direct corner on target: no xGOT
    ("s06", True, True, "shootout", "shootout"),
    ("s07", False, False, "shootout", None),  # shoot-out kick off target
    ("s08", True, False, "penalty", "penalty"),  # in-game penalty saved
    ("s09", False, False, "penalty", None),  # in-game penalty off target
    ("s11", True, False, None, "open_play"),
    ("s12", True, True, None, "open_play"),
)
_ODD = {
    "statsbomb": ("s10", True, False, "no_freeze_frame", None),
    "whoscored": ("s10", True, False, None, None),  # unresolved assist: in `unresolved`
}
_ORDER = (7, 2, 10, 0, 5, 11, 3, 9, 1, 6, 4, 8)  # rows not in shot_id order


def _frames(provider: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.DataFrame]:
    p = "sb:" if provider == "statsbomb" else "ws:"
    rows = [*_COMMON, _ODD[provider]]
    rows = [rows[i] for i in _ORDER]
    c3 = pl.DataFrame({
        "shot_id": [p + r[0] for r in rows],
        "on_target": [r[1] for r in rows],
        "is_goal": [r[2] for r in rows],
    })  # fmt: skip
    excl = [r for r in rows if r[3] is not None]
    exclusions = pl.DataFrame(
        {
            "shot_id": [p + r[0] for r in excl],
            "match_id": [p + "m1"] * len(excl),
            "competition_id": [p + "c1"] * len(excl),
            "season_id": [p + "y1"] * len(excl),
            "reason": [r[3] for r in excl],
            "is_goal": [r[2] for r in excl],
        },
        schema=data.EXCLUSION_SCHEMA,
    )
    unresolved = pl.Series("shot_id", [p + "s10"] if provider == "whoscored" else [],
                           dtype=pl.String())  # fmt: skip
    scored = [r for r in rows if r[4] is not None]
    pred = pl.DataFrame({
        "shot_id": [p + r[0] for r in scored],
        "kind": pl.Series([r[4] for r in scored], dtype=xg.KIND),
        "on_target": [r[1] for r in scored],
        "is_goal": [r[2] for r in scored],
    })  # fmt: skip
    return c3, exclusions, unresolved, pred


@pytest.mark.parametrize("provider", ["statsbomb", "whoscored"])
def test_reconcile_partition(provider: str) -> None:
    """Rows partition C3's on-target shots in ``RECONCILE_REASONS`` order, every row printed even
    when 0; off-target excluded rows (own goal, missed kicks) count nowhere."""
    from football_shot_models import xgot

    assert (*data.EXCLUSION_REASONS, "unresolved_assist", "modelled") == xgot.RECONCILE_REASONS
    assert xgot.POPULATION_REASONS == ("shootout", "penalty", "modelled")
    assert xgot.RECONCILE_TOL == 0.01
    r = xgot.reconcile(*_frames(provider))
    sb = provider == "statsbomb"
    assert r.rows == (
        ("own_goal", 0, 0),
        ("shootout", 1, 1),
        ("penalty", 1, 0),
        ("direct_corner", 1, 1),
        ("no_freeze_frame", 1 if sb else 0, 0),
        ("unresolved_assist", 0 if sb else 1, 0),
        ("modelled", 3, 2),
    )
    assert (r.c3_shots, r.c3_goals) == (7, 4)
    assert sum(n for _, n, _ in r.rows) == r.c3_shots
    assert sum(g for _, _, g in r.rows) == r.c3_goals
    assert (r.shots, r.goals, r.unscored) == (5, 3, 0)  # direct corner and the odd save unscored


def test_reconcile_counts_scored_rows() -> None:
    """``shots`` / ``goals`` come from the on-target rows of ``pred``: a missing modelled or
    shoot-out row lowers them and counts as ``unscored``; a ``pred`` on-target row that is not a
    C3 on-target shot is an error naming the count."""
    from football_shot_models import xgot

    c3, exclusions, unresolved, pred = _frames("statsbomb")
    for missing, goals in (("sb:s11", 3), ("sb:s06", 2)):
        r = xgot.reconcile(c3, exclusions, unresolved, pred.filter(pl.col("shot_id") != missing))
        assert (r.shots, r.goals, r.unscored) == (4, goals, 1), missing
        assert r.rows == xgot.reconcile(c3, exclusions, unresolved, pred).rows  # rows: C3 only
    claims = pred.with_columns(
        on_target=pl.when(pl.col("shot_id") == "sb:s03").then(True).otherwise(pl.col("on_target"))
    )
    with pytest.raises(ValueError, match=r"\b1\b"):
        xgot.reconcile(c3, exclusions, unresolved, claims)
    stranger = pl.concat([pred, pred.head(1).with_columns(shot_id=pl.lit("sb:zz"),
                                                          on_target=pl.lit(True))])  # fmt: skip
    with pytest.raises(ValueError, match=r"\b1\b"):
        xgot.reconcile(c3, exclusions, unresolved, stranger)


@pytest.mark.parametrize("excluded", ["sb:s05", "sb:s10"])
def test_reconcile_refuses_a_scored_excluded_shot(excluded: str) -> None:
    """A C3 on-target shot whose reason is outside ``POPULATION_REASONS`` (a direct corner, a
    no-frame save) that ``pred`` scores is an error naming the count, never reconciled
    (run5-ms5 Q8): a population bug must not move the gap towards 0."""
    from football_shot_models import xgot

    c3, exclusions, unresolved, pred = _frames("statsbomb")
    row = c3.filter(pl.col("shot_id") == excluded)
    assert row["on_target"].item()  # precondition: on target in C3, so the old check passed
    scored = pl.concat([pred, pl.DataFrame({
        "shot_id": [excluded], "kind": pl.Series(["open_play"], dtype=xg.KIND),
        "on_target": [True], "is_goal": row["is_goal"].to_list(),
    })])  # fmt: skip
    with pytest.raises(ValueError, match=r"\b1\b.*population"):
        xgot.reconcile(c3, exclusions, unresolved, scored)


@pytest.mark.parametrize(
    ("c3_shots", "c3_goals", "shots", "goals", "met"),
    [
        (1000, 300, 990, 300, (True, True)),  # shots -1 % exactly
        (1000, 300, 1010, 300, (True, True)),  # +1 %: the check is two-sided
        (10000, 300, 9899, 300, (False, True)),  # -1.01 %
        (10000, 300, 10101, 300, (False, True)),  # +1.01 %
        (1000, 1000, 1000, 990, (True, True)),  # goals -1 %
        (1000, 10000, 1000, 10101, (True, False)),  # goals +1.01 %
        (1000, 10000, 1000, 9899, (True, False)),  # goals -1.01 %
    ],
)
def test_reconcile_tolerance(
    c3_shots: int, c3_goals: int, shots: int, goals: int, met: tuple[bool, bool]
) -> None:
    """Checks 8 and 9: ``|shots - c3_shots| / c3_shots <= 0.01`` (inclusive, both signs), the
    same for goals; each can be missed on its own."""
    inputs = xg.check_inputs(recon=xg.reconciliation(c3_shots, c3_goals, shots, goals))
    rows = xg.call_checks(inputs)
    assert len(rows) == 11
    got = tuple(rows[i].result == "met" for i in (7, 8))
    assert got == met
    assert all(rows[i].result in ("met", "missed") for i in (7, 8))


def _goals_frame(c3: pl.DataFrame) -> pl.DataFrame:
    return c3.with_columns(on_target=pl.lit(False))


@pytest.mark.parametrize(
    "case",
    ["no on-target shot", "c3 shot_id", "c3 on_target", "c3 is_goal", "exclusions reason",
     "pred kind", "duplicate c3", "null c3", "duplicate pred"],
)  # fmt: skip
def test_reconcile_errors(case: str) -> None:
    """``ValueError`` on no C3 on-target shot, a missing column, and a duplicate or null id."""
    from football_shot_models import xgot

    c3, exclusions, unresolved, pred = _frames("statsbomb")
    xgot.reconcile(c3, exclusions, unresolved, pred)  # near miss: the frames are valid
    if case == "no on-target shot":
        c3 = _goals_frame(c3)
        pred = pred.filter(pl.lit(False))
    elif case.startswith("c3 "):
        c3 = c3.drop(case.removeprefix("c3 "))
    elif case == "exclusions reason":
        exclusions = exclusions.drop("reason")
    elif case == "pred kind":
        pred = pred.drop("kind")
    elif case == "duplicate c3":
        c3 = pl.concat([c3, c3.tail(1)])
    elif case == "null c3":
        c3 = c3.with_columns(
            shot_id=pl.when(pl.col("shot_id") == "sb:s03").then(None).otherwise(pl.col("shot_id"))
        )
    else:
        pred = pl.concat([pred, pred.tail(1)])
    with pytest.raises(ValueError):
        xgot.reconcile(c3, exclusions, unresolved, pred)


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for provider in ("statsbomb", "whoscored"):
        root: Path = tmp_path_factory.mktemp(provider)
        out[provider] = data.load_xg_data(xg.write_xgot_snapshot(root, provider), builder=provider)
    return out


@pytest.mark.parametrize("provider", ["statsbomb", "whoscored"])
def test_xgot_context(loaded: dict[str, Any], provider: str) -> None:
    """The card context counts the on-target frames passed, not the xG population; its exclusions
    are the reconciliation rows; the hold-out (WhoScored: the test season) is passed explicitly.
    Each provider's filter labels are keyed exactly by ``RECONCILE_REASONS``."""
    from football_shot_models import xgot

    xgd = loaded[provider]
    p = xgot.PROVIDERS[provider]
    assert (p.key, p.builder) == (provider, provider)
    assert p.prefix == ("xgot-" if provider == "statsbomb" else "ws-xgot-")
    assert p.split == ("hold-out" if provider == "statsbomb" else "test")
    assert set(p.text.filter_labels) == set(xgot.RECONCILE_REASONS)
    assert (p.text.benchmark_name, p.text.benchmark_caveat) == (None, None)
    assert p.text.exclusions_log.startswith("C3 on-target shots by first matching reason")
    assert xgot.XGOT_CROSS_FIT in p.validation and cards.VALIDATION not in p.validation
    if provider == "whoscored":
        assert "StatsBomb" not in p.validation and "StatsBomb" not in str(p.text)
    ho = p.holdout(xgd.matches)
    shots = xgd.shots if provider == "statsbomb" else v1.population(xgd)[0]
    dev, held = data.split_holdout(shots.filter("on_target"), ho)
    dev = dev.filter(pl.col("is_goal") | (pl.int_range(pl.len()) % 2 == 0))  # not every row
    recon = xg.reconciliation(dev.height + held.height, 1, dev.height + held.height, 1)
    recon = dataclasses.replace(
        recon, rows=tuple((r, i + 1, i) for i, r in enumerate(xgot.RECONCILE_REASONS))
    )
    c = xgot.xgot_context(xgd, dev, held, recon, holdout=ho)
    both = pl.concat([dev, held])
    assert (c.dev_shots, c.holdout_shots) == (dev.height, held.height)
    assert (c.dev_matches, c.holdout_matches) == (
        dev["match_id"].n_unique(), held["match_id"].n_unique(),
    )  # fmt: skip
    assert c.competition_seasons == both.select("competition_id", "season_id").n_unique()
    assert (c.male_shots, c.female_shots) == (both.height, 0)
    assert c.exclusions == recon.rows
    assert c.holdout_label == ho.label
    assert c.data_digest == xgd.manifest.data_digest
    assert c.dev_shots < xgd.shots.filter("on_target").height  # the frames, not the population
