"""``level``: the StatsBomb hold-out level probe, the adoption rule and its card text (spec
``runs/PR1/xg-context`` §3.3).

The probe runs on the ``xg_cases`` snapshot (dev 120 shots in three families of 48 / 48 / 24
out of key order, hold-out 20 shots with 8 goals, every shot with a freeze frame, no 360 area)
with hand-built evaluations: POOLED over-predicts the hold-out (z ≈ +2.93), the indicator fix
sits exactly on z = +2 in float (16 hold-out shots at 0.25 / 0.75 and 4 at 0.5: Σp = 12,
Σp(1 − p) = 4, both exact), the league-trained headline at z = 0. Check rows are hand-written
``metrics.Check`` values. Expected numbers are recomputed here with numpy, never read from the
module.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest

from football_shot_models import data, metrics, models
from football_shot_models.metrics import Check

MET, MISSED, NA = (Check("t", "m", r) for r in ("met", "missed", "n/a"))  # type: ignore[arg-type]


def _level() -> Any:
    from football_shot_models import level

    return level


# --- level_z ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("goal", "z"), [(False, 2.0), (True, -2.0)])
def test_level_z_boundary(goal: bool, z: float) -> None:
    """§3.3: (Σp − Σy, √Σp(1 − p), z), exactly ±2 in float on the boundary (within: |z| ≤
    ``LEVEL_SD``, equality within); 0.01 more expected goals is outside."""
    level = _level()
    assert level.LEVEL_SD == 2.0
    y, p = np.full(4, goal), np.full(4, 0.5)
    assert level.level_z(y, p) == (z, 1.0, z)
    assert abs(level.level_z(y, p)[2]) <= level.LEVEL_SD
    y2, p2 = np.append(y, False), np.append(p, 0.01)
    diff, sd, z2 = level.level_z(y2, p2)
    assert diff == pytest.approx(2.01 if not goal else -1.99)
    assert sd == pytest.approx(math.sqrt(1.0 + 0.01 * 0.99))
    if not goal:
        assert z2 > level.LEVEL_SD  # precondition of the near miss: 2.000124...


@pytest.mark.parametrize(
    "p",
    [np.array([]), np.array([0.5, np.nan]), np.array([0.5, np.inf]), np.array([0.0, 1.0])],
    ids=["empty", "nan", "inf", "sd-zero"],
)
def test_level_z_errors(p: np.ndarray) -> None:
    """§3.3: empty, NaN/inf p, or SD = 0 → ValueError."""
    with pytest.raises(ValueError):
        _level().level_z(np.zeros(p.size, dtype=bool), p)


# --- the adoption rule -----------------------------------------------------------------------

POOLED_CHECKS = (MET, MET, NA, MISSED)


def row(name: str, within: bool, **kw: Any) -> Any:
    """A ``LevelRow``; fields beyond the spec's list (a ``name``) get the spec name, and the
    title is the spec name too, so a rule that matches ``skip`` against either works."""
    level = _level()
    values: dict[str, Any] = {
        "title": name, "xg": 10.0, "goals": 10, "sd": 2.0, "z": 0.0, "within": within,
        "met": True, "judged": 3, "adopted": False, "cv_log_loss": 0.27,
        "holdout_log_loss": 0.26, "fold_xg_mean": 0.01, "fold_xg_max": 0.02,
    }  # fmt: skip
    values |= kw
    fields = {f.name for f in dataclasses.fields(level.LevelRow)}
    return level.LevelRow(**{k: values.get(k, name) for k in fields})


NAMES = ("xg-gbm-ts-indicator", "xg-gbm-league-trained")


@pytest.mark.parametrize(
    ("fix_checks", "within", "skip", "expected"),
    [
        (((MET, MET, NA, MISSED), (MET, MET, NA, MET)), (True, True), (), 0),
        (((MET, MET, NA, MISSED), (MET, MET, NA, MET)), (False, True), (), 1),
        (((MET, MISSED, NA, MET), (MET, MET, NA, MISSED)), (True, True), (), 1),
        (((MET, NA, NA, MET), (MET, MET, NA, MET)), (True, False), (), None),
        (((MET, MET, NA, MET), (MET, MET, NA, MET)), (True, True), (NAMES[0],), 1),
        (((MET, MET, NA, MET), (MET, MET, NA, MET)), (True, True), NAMES, None),
        (((MET, MET, NA, MET), (MISSED, MET, NA, MET)), (False, True), (), None),
    ],
    ids=["first-adoptable", "first-outside", "met-to-missed-blocks", "met-to-na-blocks",
         "downstream-skip", "all-skipped", "none"],
)  # fmt: skip
def test_adopt_rule(
    fix_checks: tuple[tuple[Check, ...], ...], within: tuple[bool, bool],
    skip: tuple[str, ...], expected: int | None,
) -> None:  # fmt: skip
    """§3.3 rule: the first fix in ``LEVEL_FIXES`` order that is within, keeps every row that is
    ``met`` for POOLED ``met`` (a pooled n/a or missed row is not judged), and is not in ``skip``
    (``DOWNSTREAM_MISSED``); otherwise None."""
    rows = tuple(row(n, w) for n, w in zip(NAMES, within, strict=True))
    assert _level().adopt(POOLED_CHECKS, fix_checks, rows, skip=skip) == expected


def test_adopt_ignores_na_rows() -> None:
    """§3.3: rows n/a for POOLED stay out of the rule whatever the fix gets there (n/a, missed);
    a pooled ``missed`` row may stay missed."""
    pooled = (NA, MET, MISSED, NA)
    fixes = ((NA, MET, MISSED, MISSED), (MET, MET, MET, MET))
    rows = (row(NAMES[0], True), row(NAMES[1], True))
    assert _level().adopt(pooled, fixes, rows, skip=()) == 0


# --- level_probe and level_lines -------------------------------------------------------------


@pytest.fixture(scope="module")
def frames(tmp_path_factory: pytest.TempPathFactory) -> tuple[pl.DataFrame, ...]:
    import xg_cases as xc

    manifest = xc.write_fixture_snapshot(tmp_path_factory.mktemp("level"))
    xg = data.load_xg_data(manifest)
    dev, held = data.split_holdout(xg.shots)
    assert (dev.height, held.height, int(held["is_goal"].sum())) == (120, 20, 8)  # precondition
    return dev, held, xg.freeze_frames


def _p_exactly_two(n: int) -> np.ndarray:
    """4 × 0.25, 12 × 0.75, 4 × 0.5: Σp = 12, Σp(1 − p) = 4 (dyadic, exact in float)."""
    assert n == 20
    return np.array([0.25] * 4 + [0.75] * 12 + [0.5] * 4)


def _ev(spec: Any, dev: pl.DataFrame, oof: float, held_p: np.ndarray, ll: float) -> Any:
    folds = tuple(mc.scores(n=24, goals=6, xg=6.0 + k * 0.3, log_loss=ll) for k in range(5))
    return mc.fake_eval(
        spec, folds=folds, oof=np.full(dev.height, oof), holdout_p=held_p,
        holdout=mc.scores(n=20, goals=8, xg=float(held_p.sum()), log_loss=ll - 0.01),
    )  # fmt: skip


@pytest.fixture(scope="module")
def probe(frames: tuple[pl.DataFrame, ...]) -> dict[str, Any]:
    level = _level()
    dev, held, ff = frames
    pooled = _ev(models.POOLED, dev, 0.30, np.full(20, 0.7), 0.270)
    indicator = _ev(models.LEVEL_FIXES[0], dev, 0.28, _p_exactly_two(20), 0.272)
    head = _ev(models.GBM, dev, 0.25, np.full(20, 0.4), 0.268)
    checks = {id(pooled): (MET, MET, NA), id(indicator): (MET, MISSED, NA),
              id(head): (MET, MET, NA)}  # fmt: skip
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        p = level.level_probe(
            pooled, (indicator, head), head, checks_of=lambda ev: checks[id(ev)], dev=dev,
            held=held, freeze_frames=ff,
        )  # fmt: skip
    assert out.getvalue() == ""  # pure: no print
    return {"probe": p, "pooled": pooled, "indicator": indicator, "head": head}


def test_probe_rows(probe: dict[str, Any], frames: tuple[pl.DataFrame, ...]) -> None:
    """§3.3: one row per evaluation (POOLED and each fix) with Σ xG, goals, SD and z of its
    hold-out predictions (recomputed here), ``within`` two-sided with equality within (the
    indicator sits exactly on z = 2), judged = the non-n/a check rows, and the evaluation's CV /
    hold-out log loss and fold Σ xG mean / max."""
    level = _level()
    _, held, _ = frames
    y = held["is_goal"].to_numpy()
    p = probe["probe"]
    by_title = {r.title: r for r in p.rows}
    for key in ("pooled", "indicator"):
        ev = probe[key]
        r = by_title[ev.spec.title]
        hp = ev.holdout_p
        sd = math.sqrt(float((hp * (1 - hp)).sum()))
        assert r.xg == pytest.approx(hp.sum()) and r.goals == int(y.sum())
        assert r.sd == pytest.approx(sd) and r.z == pytest.approx((hp.sum() - y.sum()) / sd)
        assert r.judged == 2
        assert r.cv_log_loss == ev.cv.log_loss and r.holdout_log_loss == ev.holdout.log_loss
        assert (r.fold_xg_mean, r.fold_xg_max) == pytest.approx(metrics.fold_xg_error(ev.folds))
    assert by_title[probe["indicator"].spec.title].z == 2.0  # in float, on the boundary
    assert by_title[probe["indicator"].spec.title].within is True
    assert by_title[probe["pooled"].spec.title].within is False
    assert level.LEVEL_SD == 2.0


def _keys(seg: pl.DataFrame) -> list[str]:
    """The segment key: ``kind`` plus any other text column but ``level`` (a split label)."""
    text = [c for c, t in seg.schema.items() if not t.is_numeric() and t != pl.Boolean()]
    return [c for c in text if c != "level"]


def _groups(seg: pl.DataFrame) -> pl.DataFrame:
    floats = [c for c, t in seg.schema.items() if t.is_float()]
    return seg.group_by(_keys(seg)).agg(pl.col("n", "goals").sum(), pl.col(floats).sum())


def test_segments_sum_to_totals(probe: dict[str, Any], frames: tuple[pl.DataFrame, ...]) -> None:
    """§3.3: one row per (kind, level); per kind the shots and goals sum to the split's totals
    (hold-out kinds: phase, body part, distance band, assist type, shot freeze frame, 360 frame;
    dev OOF kinds: family, shot freeze frame, 360 frame); for each split a Σ xG column sums per
    kind to POOLED's and one to the headline's Σ p; distance bands are (a, b] on
    ``DISTANCE_BANDS``; the family kind has the fixture's 24 / 48 / 48."""
    level = _level()
    dev, held, _ = frames
    seg = probe["probe"].segments
    assert {"kind", "level", "n", "goals"} <= set(seg.columns)
    keys, groups = _keys(seg), _groups(seg)
    floats = [c for c, t in seg.schema.items() if t.is_float()]
    assert set(groups["n"].to_list()) == {held.height, dev.height}, groups
    for frame, kinds, ev_p in ((held, 6, lambda e: e.holdout_p), (dev, 3, lambda e: e.oof)):
        sums = groups.filter(pl.col("n") == frame.height)
        assert sums.height == kinds, sums
        assert (sums["goals"] == int(frame["is_goal"].sum())).all()
        for key in ("pooled", "head"):
            target = float(ev_p(probe[key]).sum())
            assert any(np.allclose(sums[c].to_numpy(), target) for c in floats), (key, sums)

    def rows_of(word: str, n: int) -> pl.DataFrame:
        (key,) = [
            k for k in groups.filter(pl.col("n") == n).select(keys).iter_rows()
            if any(word in str(v).lower() for v in k)
        ]  # fmt: skip
        mask = pl.all_horizontal(pl.col(c) == v for c, v in zip(keys, key, strict=True))
        return seg.filter(mask & (pl.col("n") > 0))

    band = np.searchsorted(np.array(level.DISTANCE_BANDS), held["distance_m"].to_numpy())
    expected = sorted(int(c) for c in np.bincount(band) if c)
    assert sorted(rows_of("distance", held.height)["n"].to_list()) == expected
    assert level.DISTANCE_BANDS == (6.0, 11.0, 16.0, 20.0, 25.0)
    assert sorted(rows_of("family", dev.height)["n"].to_list()) == [24, 48, 48]


def test_probe_holds_aligned_frames(
    probe: dict[str, Any], frames: tuple[pl.DataFrame, ...]
) -> None:
    """§3.3, §8: the probe keeps the evaluations and frames a chart needs with no refit: each
    ``holdout_p`` lines up with ``held.shot_id`` and each ``oof`` with ``dev.shot_id``."""
    dev, held, _ = frames
    p = probe["probe"]
    assert p.pooled is probe["pooled"] and p.head is probe["head"]
    assert tuple(p.fixes) == (probe["indicator"], probe["head"])
    assert p.held["shot_id"].to_list() == held["shot_id"].to_list()
    assert p.dev["shot_id"].to_list() == dev["shot_id"].to_list()
    for ev in (p.pooled, *p.fixes, p.head):
        assert ev.holdout_p.size == p.held.height and ev.oof.size == p.dev.height


def test_level_lines(probe: dict[str, Any]) -> None:
    """§3.3 card text: the limitation sentence iff no row is adopted; a kind with one level
    prints "one level only (n = …)" (no 360 area on the hold-out); the pooled-trained LR-steps
    line appears only when the headline's ``train_exclude`` differs from the steps'."""
    level = _level()
    p = probe["probe"]
    head_title = probe["head"].spec.title
    none = dataclasses.replace(p, rows=tuple(dataclasses.replace(r, adopted=False) for r in p.rows))
    one = dataclasses.replace(
        p, rows=tuple(dataclasses.replace(r, adopted=r.title == head_title) for r in p.rows)
    )
    assert any(r.adopted for r in one.rows)  # precondition
    excl = models.GBM.train_exclude
    text_none = "\n".join(level.level_lines(none, steps_train_exclude=excl))
    text_one = "\n".join(level.level_lines(one, steps_train_exclude=excl))
    assert "none: the decomposition is a limitation" in text_none
    assert "none: the decomposition is a limitation" not in text_one
    assert "one level only (n = 20" in text_one
    other = () if excl else ("team-selected",)
    mixed = set(level.level_lines(one, steps_train_exclude=other)) - set(
        level.level_lines(one, steps_train_exclude=excl)
    )
    assert any("LR" in line for line in mixed), mixed


def test_min_judged_shots_is_one_rule() -> None:
    """§3.3: ``metrics.MIN_JUDGED_SHOTS`` = 100 (PM MS6) and ``xgot.MIN_BAND_SHOTS`` is it."""
    from football_shot_models import xgot

    assert metrics.MIN_JUDGED_SHOTS == 100
    assert xgot.MIN_BAND_SHOTS == metrics.MIN_JUDGED_SHOTS


def test_refused_sentence_is_computed(
    probe: dict[str, Any], frames: tuple[pl.DataFrame, ...]
) -> None:
    """Integration census (PR1.xg-context, ratify): ``LevelRow.refused`` and the card's "met the
    MS3 rule but an xGOT / GSAx check refused it" are computed: a fix in ``skip`` that meets the
    rule (the headline: within at z = 0, every pooled ``met`` row kept) is refused and named;
    one that fails it (the indicator: a pooled ``met`` row ``missed``) is not, although skipped."""
    level = _level()
    dev, held, ff = frames
    ind, head, pooled = probe["indicator"], probe["head"], probe["pooled"]
    checks = {id(pooled): (MET, MET, NA), id(ind): (MET, MISSED, NA), id(head): (MET, MET, NA)}
    names = (ind.spec.name, head.spec.name)
    assert len(set(names)) == 2  # precondition: the skip names tell the two fixes apart
    p = level.level_probe(
        pooled, (ind, head), head, checks_of=lambda ev: checks[id(ev)], dev=dev, held=held,
        freeze_frames=ff, skip=names,
    )  # fmt: skip
    assert p.adopted is None
    assert [r.refused for r in p.rows] == [False, False, True]
    assert p.rows[1].within  # precondition: only the kept-checks clause fails for the indicator
    text = "\n".join(level.level_lines(p, steps_train_exclude=()))
    assert f"{head.spec.title} met the MS3 rule but an xGOT / GSAx check refused it." in text
    assert f"{ind.spec.title} met the MS3 rule" not in text
