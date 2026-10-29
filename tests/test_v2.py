"""``v2``: WhoScored xG v2 with pre-shot context, its qualifier census and checks (spec
``runs/PR1/xg-context`` §3.4, §4, §6).

The fit runs once per session on the ``v2_cases`` snapshot (``conftest.v2_fit``). Check rows are
driven to ``missed`` by calling ``v2.v2_checks`` with one part of the fit replaced: its
parameters are looked up by name among the ``V2Fit`` fields (a choice pinned in ``tests.md``).
Expected counts are recomputed here from the fixture frames.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import v2_cases as v2c
from conftest import V2_MIN_SEGMENT

from football_shot_models import data, evaluation, features, metrics, models, v1, xgot
from football_shot_models.metrics import Check

TEAM_CONTEXT = ("prev_action", "rebound")


def _v2() -> Any:
    from football_shot_models import v2

    return v2


def call_checks(fit: Any, **over: Any) -> tuple[Check, ...]:
    """``v2.v2_checks`` with each parameter taken from the ``V2Fit`` field of its name, or from
    ``over``."""
    v2 = _v2()
    params = inspect.signature(v2.v2_checks).parameters
    assert not set(over) - set(params), sorted(set(over) - set(params))
    fields = {f.name for f in dataclasses.fields(fit)}
    kwargs: dict[str, Any] = {}
    for name, param in params.items():
        if name in over:
            kwargs[name] = over[name]
        elif name in fields:
            kwargs[name] = getattr(fit, name)
        else:
            assert param.default is not inspect.Parameter.empty, (
                f"v2_checks parameter {name!r} is not a V2Fit field"
            )
    return tuple(v2.v2_checks(**kwargs))


def results(checks: tuple[Check, ...]) -> list[str]:
    return [c.result for c in checks[:6]]


# --- constants -------------------------------------------------------------------------------


def test_v2_specs() -> None:
    """§3.4: the name, the context (v1's plus the two new columns), the steps (v1's, then the
    context-v2 LR), the GBM on the same inputs without splines and v1's sign map; the classes."""
    from football_shot_models import ablation, gbm

    v2 = _v2()
    assert v2.NAME == "xg-v2"
    assert (*v1.CONTEXT, *TEAM_CONTEXT) == v2.CONTEXT_V2
    assert v2.STEPS[: len(v1.STEPS)] == v1.STEPS
    (step,) = v2.STEPS[len(v1.STEPS) :]
    assert (step.name, step.features) == ("ws-lr-context-v2", ablation.SPLINES + v2.CONTEXT_V2)
    assert step.make is ablation.make_lr
    g = v2.GBM
    assert (g.name, g.features) == ("ws-gbm-v2", ("distance_m", "angle_rad", "body_part",
                                                  *v2.CONTEXT_V2))  # fmt: skip
    assert g.coefficients is gbm.gbm_settings
    assert dict(g.make().get_params()["monotone"]) == dict(v1.GBM_SIGNS)
    assert v2.QUALIFIER_CLASSES == (
        "body part",
        "pattern",
        "location zone",
        "assist link",
        "situation",
        "grading",
        "post-shot",
        "outcome-derived",
    )
    assert frozenset({"grading", "post-shot", "outcome-derived"}) == v2.REFUSED_CLASSES
    for spec in (*v2.STEPS, g):
        assert not features.uses_ablation_only(spec.features), spec.name
        assert features.check_xg_features(spec.features) == spec.features


# Every shot qualifier id of the census (§1: q60-82 and q89, no q83-88), plus q2 / q4 (assist
# pass) and q94 (Save).
CENSUS_IDS = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 55, 56, *range(60, 83), 89,
              100, 101, 102, 103, 108, 146, 147, 154, 160, 178, 214, 215, 233, 328,
              11111, 11112, 11113, 2, 4, 94)  # fmt: skip
SPEC_CLASS = {
    **dict.fromkeys((15, 20, 21, 72), "body part"),
    **dict.fromkeys((22, 23, 24, 25, 26, 160), "pattern"),
    **dict.fromkeys((55, 2, 4), "assist link"),
    **dict.fromkeys((89, 108, 154, 215, 328, 94), "situation"),
    214: "grading",
    **dict.fromkeys((*range(73, 88), 82, 100, 101, 102, 103, 146, 147, 178, 233), "post-shot"),
    **dict.fromkeys((11111, 11113), "outcome-derived"),
}


def test_qualifier_classes() -> None:
    """§1, §3.4: the classes the spec states (big chance grades the chance; q233 and the miss /
    placement ids are post-shot; key pass and intentional goal assist are outcome-derived;
    q89 / q108 / q154 / q215 / q328 / q94 describe the situation); q11112 is refused (E1 escalated:
    outcome-derived or grading, refused either way); every census id has a class of
    ``QUALIFIER_CLASSES``; ``FEATURE_QUALIFIERS`` maps each WhoScored column to the ids it reads."""
    from football_core.io import whoscored

    v2 = _v2()
    # the §1 census (census_shot_qualifiers.out lists 56 ids; §1's "57" is a miscount), by hand
    assert len(set(CENSUS_IDS) - {2, 4, 94}) == 56
    for qid, cls in SPEC_CLASS.items():
        assert v2.QUALIFIER_CLASS[qid] == cls, qid
    assert v2.QUALIFIER_CLASS[11112] in v2.REFUSED_CLASSES
    assert all(v2.QUALIFIER_CLASS[q] in v2.QUALIFIER_CLASSES for q in CENSUS_IDS)
    fq = v2.FEATURE_QUALIFIERS
    assert fq["distance_m"] == fq["angle_rad"] == fq["rebound"] == ()
    assert set(fq["body_part"]) == set(whoscored.BODY_PARTS)
    assert set(fq["phase"]) == {*whoscored.PATTERNS, whoscored.DIRECT_CORNER}
    assert set(fq["assist"]) == {55, 2, 4}
    assert (fq["one_on_one"], fq["first_time"], fq["big_chance"]) == ((89,), (328,), (214,))
    assert fq["prev_action"] == (94,)  # q233 is a guard, not a value
    assert v2.QUALIFIER_CLASS[214] in v2.REFUSED_CLASSES


# --- wiring ----------------------------------------------------------------------------------


def test_fit_uses_v1_population_and_split(v2_fit: dict[str, Any]) -> None:
    """§3.4: v2's population and split are v1's by call (spy, not a self-comparison):
    ``v1.population`` once, ``data.season_holdout`` with ``v1.TRAIN_SEASONS`` /
    ``TEST_SEASONS`` (the patched objects themselves); dev / held are the population split by
    that hold-out; the CV folds are ``cv_folds(dev)``; fit neither prints nor writes."""
    fit, calls = v2_fit["fit"], v2_fit["calls"]
    assert len(calls["population"]) == 1
    shots, _ = calls["population"][0][2]
    (args, kwargs, ho) = calls["season_holdout"][0]
    bound = inspect.signature(data.season_holdout).bind(*args, **kwargs).arguments
    assert bound["train"] is v2_fit["seasons"][0] and bound["test"] is v2_fit["seasons"][1]
    dev, held = data.split_holdout(shots, ho)
    assert fit.dev["shot_id"].to_list() == dev["shot_id"].to_list()
    assert fit.held["shot_id"].to_list() == held["shot_id"].to_list()
    folds = evaluation.cv_folds(dev)
    assert len(fit.head.fold_test) == len(folds)
    assert all(np.array_equal(a, t) for a, (_, t) in zip(fit.head.fold_test, folds, strict=True))
    assert v2_fit["stdout"] == "" and v2_fit["stderr"] == ""
    assert list(v2_fit["cwd"].iterdir()) == []


def test_fit_evaluations(v2_fit: dict[str, Any]) -> None:
    """§3.4: the headline by ``models.select_headline`` over v2's candidates; ``v1_head`` is
    ``v1.GBM`` and ``v1_ablation`` the big-chance ablation of v1's best LR; ``gain_over_v1``
    compares them on the same rows (recomputed with ``metrics.bootstrap_gain``); ``groups`` is
    the per-competition Σ xG table of the headline on test (one row per test competition, shot
    counts recomputed)."""
    fit = v2_fit["fit"]
    c = fit.candidates
    assert c.gbm.spec.name == "ws-gbm-v2" and [e.spec.name for e in c.steps][-1] == (
        "ws-lr-context-v2"
    )
    expect = c.gbm if fit.gbm_gain.cv.low > 0 else evaluation.best(c.lrs)
    assert fit.head is expect
    assert fit.v1_head.spec.name == "ws-gbm"
    assert fit.v1_ablation.spec.name.endswith("-big-chance")
    assert "big_chance" in fit.v1_ablation.spec.features
    # run7 architect 5: v1's ablation is picked among v1's candidates by name, not by position
    from football_shot_models import baselines

    v1_names = {s.name for s in (*baselines.BASELINES, *v1.STEPS)}
    v1_lrs = [e for e in c.lrs if e.spec.name in v1_names]
    assert len(v1_lrs) == len(c.lrs) - 1  # precondition: only the context-v2 LR is not v1's
    assert fit.v1_ablation.spec == v1.ablation_spec(evaluation.best(v1_lrs).spec)
    y = evaluation.labels(fit.held)
    g = metrics.bootstrap_gain(
        y, fit.v1_head.holdout_p, fit.head.holdout_p, fit.held["match_id"].to_numpy()
    )
    assert (fit.gain_over_v1.holdout.gain, fit.gain_over_v1.holdout.low) == pytest.approx(
        (g.gain, g.low)
    )  # fmt: skip
    matches = data.load_xg_data(v2_fit["manifest"], builder="whoscored").matches
    comp = fit.held.join(matches.select("match_id", "competition_name"), on="match_id")
    counts = dict(comp.group_by("competition_name").len().iter_rows())
    assert {r.key: r.shots for r in fit.groups} == counts
    assert len(counts) >= 2 and min(counts.values()) < models.MIN_SEGMENT_SHOTS
    # integration census: the fixture's segment minimum (conftest) leaves both kinds of row
    small = {k for k, n in counts.items() if n < V2_MIN_SEGMENT}
    assert small and small != set(counts)
    for r in fit.groups:
        assert (r.cal_slope is None) == (r.key in small), r.key
    v2c.census_reached(fit.dev, comp)


# --- census ----------------------------------------------------------------------------------


def _q(*ids: int) -> str:
    return json.dumps([{"type": {"value": q, "displayName": f"Q{q}"}} for q in ids])


CENSUS_FRAME = pl.DataFrame(
    {
        "raw_qualifiers": [_q(20, 22, 214), _q(72, 22), _q(20, 23, 214, 328), _q(15, 25),
                           _q(20, 22, 214), _q(72, 22, 99_999)],
        "is_goal": [True, False, True, False, True, True],
        "prev_action": ["opp_keeper", "pass", "opp_block", "pass", "opp_keeper", "shot"],
        "rebound": [True, False, False, False, True, False],
    },
    schema_overrides={"prev_action": pl.Enum(v2c.PREV_LEVELS)},
)  # fmt: skip
CENSUS_TRAIN = pl.Series("train", [True, True, True, False, False, True])


def _row(frame: pl.DataFrame, *cells: object) -> tuple[Any, ...]:
    """The one row holding every cell (by value; position-free)."""
    found = [r for r in frame.iter_rows() if all(c in r for c in cells)]
    assert len(found) == 1, (cells, found)
    return found[0]


def test_census_counts() -> None:
    """§3.4: qualifier rows (id, Opta name, class, train shots, train goals, test shots) and
    context rows (each ``prev_action`` level and ``rebound`` value: class "situation",
    keeper-derived, train shots, train goals, test shots), counted by hand here; an id the spec
    does not class gets no class."""
    v2 = _v2()
    # precondition (run7 QA 2): test row 4 is a q214 goal, so a test goal counted as a train
    # goal would move train_goals
    assert CENSUS_FRAME["is_goal"][4] and not CENSUS_TRAIN[4]
    census = v2.census(CENSUS_FRAME, CENSUS_TRAIN)
    q = census.qualifiers
    assert _row(q, 214)[-3:] == (2, 2, 1)  # train rows 0, 2 (both goals); test row 4 (a goal)
    assert "grading" in _row(q, 214)
    assert _row(q, 22)[-3:] == (3, 2, 1)
    assert _row(q, 15)[-3:] == (0, 0, 1)
    assert _row(q, 328)[-3:] == (1, 1, 0) and "situation" in _row(q, 328)
    unknown = _row(q, 99_999)
    assert unknown[-3:] == (1, 1, 0)
    assert not set(unknown) & set(v2.QUALIFIER_CLASSES)
    ctx = census.context
    assert _row(ctx, "opp_keeper")[-3:] == (1, 1, 1)
    assert _row(ctx, "pass")[-3:] == (1, 0, 1)
    assert _row(ctx, "opp_block")[-3:] == (1, 1, 0)
    assert "situation" in _row(ctx, "opp_keeper")
    reb = [r for r in ctx.iter_rows() if any("rebound" in str(c) for c in r)]
    assert sorted(r[-3:] for r in reb) == [(1, 1, 1), (3, 2, 1)]  # True: rows 0, 4; False: rest


@pytest.mark.parametrize(
    "case",
    ["not-boolean", "short", "bad-json"],
)
def test_census_errors(case: str) -> None:
    """§3.4: ``train`` not Boolean or not aligned with ``shots``, and unreadable
    ``raw_qualifiers`` JSON (the message gives the count) → ValueError."""
    v2 = _v2()
    frame, train = CENSUS_FRAME, CENSUS_TRAIN
    if case == "not-boolean":
        train = train.cast(pl.Int8)
    elif case == "short":
        train = train.head(5)
    else:
        frame = frame.with_columns(
            pl.when(pl.int_range(pl.len()) < 2).then(pl.lit("{not json"))
            .otherwise(pl.col("raw_qualifiers")).alias("raw_qualifiers")
        )  # fmt: skip
    with pytest.raises(ValueError) as err:
        v2.census(frame, train)
    if case == "bad-json":
        assert "2" in str(err.value)


# --- gap share -------------------------------------------------------------------------------


def test_gap_share() -> None:
    """§3.4: (L_v1 − L_v2) / (L_v1 − L_ablation) on the chosen split (test: hold-out log loss; cv:
    the CV log loss); a denominator ≤ 0 → ValueError."""
    v2 = _v2()
    spec = evaluation.ModelSpec("x", "X", (), evaluation.BaseRate)

    def ev(cv: float, test: float) -> Any:
        return mc.fake_eval(spec, cv=mc.scores(log_loss=cv), holdout=mc.scores(log_loss=test))

    a, b, abl = ev(0.280, 0.2771), ev(0.279, 0.2752), ev(0.262, 0.2611)
    assert v2.gap_share(a, b, abl, "test") == pytest.approx((0.2771 - 0.2752) / (0.2771 - 0.2611))
    assert v2.gap_share(a, b, abl, "cv") == pytest.approx((0.280 - 0.279) / (0.280 - 0.262))
    assert v2.gap_share(a, a, abl, "test") == 0.0
    for bad in (ev(0.280, 0.2771), ev(0.281, 0.2790)):  # equal, then worse than v1
        with pytest.raises(ValueError):
            v2.gap_share(a, b, bad, "test")


# --- checks ----------------------------------------------------------------------------------


def test_checks_on_the_fixture(v2_fit: dict[str, Any]) -> None:
    """§3.4: rows 1-6 computed from the fit (recomputed here); ``call_checks`` on the fit itself
    gives the fit's rows; row 6 is missed because the fixture holds an unclassified id, and met
    once that census row is gone; reporting rows follow the six."""
    fit = v2_fit["fit"]
    assert len(fit.checks) >= 6
    assert [c.result for c in call_checks(fit)] == [c.result for c in fit.checks]
    lo, hi = models.SLOPE_RANGE
    slope = fit.head.holdout.cal_slope
    assert results(fit.checks)[:3] == [
        "met" if fit.gain_over_v1.holdout.low > 0 else "missed",
        "met" if slope is not None and lo <= slope <= hi else "missed",
        "met" if (fit.head is fit.candidates.gbm) == (fit.gbm_gain.cv.low > 0) else "missed",
    ]
    assert results(fit.checks)[3:] == ["met", "met", "missed"]
    q = fit.census.qualifiers
    kept = pl.DataFrame([r for r in q.iter_rows() if v2c.UNKNOWN_ID not in r], schema=q.schema,
                        orient="row")  # fmt: skip
    assert kept.height == q.height - 1  # precondition
    census = dataclasses.replace(fit.census, qualifiers=kept)
    assert results(call_checks(fit, census=census))[5] == "met"


def _gain(low: float) -> metrics.GainCI:
    return mc.gain(0.001, low, 0.003)


@pytest.mark.parametrize("row", [1, 2, 3, 5])
def test_checks_driven_to_missed(v2_fit: dict[str, Any], row: int,
                                 monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """§3.4: rows 1, 2, 3 and 5 each go to ``missed`` on their own, at their exact boundary
    (interval low = 0; slope 1.1 met, just above it missed; the GBM headline with CV low = 0; a
    WhoScored xGOT candidate reading ``rebound``)."""
    fit = v2_fit["fit"]
    base = results(call_checks(fit))
    if row == 1:
        over = {"gain_over_v1": dataclasses.replace(fit.gain_over_v1, holdout=_gain(0.0))}
        assert results(call_checks(fit, gain_over_v1=dataclasses.replace(
            fit.gain_over_v1, holdout=_gain(1e-9))))[0] == "met"  # fmt: skip
    elif row == 2:
        h = fit.head
        at = dataclasses.replace(h, holdout=dataclasses.replace(h.holdout, cal_slope=1.1))
        assert results(call_checks(fit, head=at))[1] == "met"
        over = {
            "head": dataclasses.replace(
                h, holdout=dataclasses.replace(h.holdout, cal_slope=np.nextafter(1.1, 2.0))
            )
        }
    elif row == 3:
        gg = dataclasses.replace(fit.gbm_gain, cv=_gain(0.0))
        over = {"gbm_gain": gg, "head": fit.candidates.gbm}
    else:
        p = xgot.PROVIDERS["whoscored"]
        steps, g = p.candidate_specs()
        bad = dataclasses.replace(g, features=(*g.features, "rebound"))
        patched = dataclasses.replace(p, candidate_specs=lambda: (steps, bad))
        monkeypatch.setattr(xgot, "PROVIDERS", {**xgot.PROVIDERS, "whoscored": patched})
        over = {}
    got = results(call_checks(fit, **over))
    assert got[row - 1] == "missed", got
    assert base[row - 1] == "met" or row in (1, 2, 3)  # fixture-dependent rows may start missed


@pytest.mark.parametrize(
    "feature", ["big_chance", "gender", "prev_action-refused", "prev_action-unclassified"]
)
def test_check4_refused(v2_fit: dict[str, Any], feature: str,
                        monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """§3.4 row 4: missed when a headline feature is ablation-only (big chance), is not in
    ``FEATURE_QUALIFIERS``, or reads an id whose class is refused (q94 re-classed "grading") or
    that has no class (q94 left out of ``QUALIFIER_CLASS``, run7 QA 3: "unclassified")."""
    v2 = _v2()
    fit = v2_fit["fit"]
    head = fit.head
    assert results(call_checks(fit))[3] == "met"  # precondition
    if feature.startswith("prev_action-"):
        classes = {**v2.QUALIFIER_CLASS, 94: "grading"}
        if feature == "prev_action-unclassified":
            classes = {q: c for q, c in v2.QUALIFIER_CLASS.items() if q != 94}
        monkeypatch.setattr(v2, "QUALIFIER_CLASS", classes)
        if "prev_action" not in head.spec.features:  # an LR headline without it: add it
            head = dataclasses.replace(head, spec=dataclasses.replace(
                head.spec, features=(*head.spec.features, "prev_action")))  # fmt: skip
    else:
        head = dataclasses.replace(head, spec=dataclasses.replace(
            head.spec, features=(*head.spec.features, feature)))  # fmt: skip
    row = call_checks(fit, head=head)[3]
    assert row.result == "missed"
    if feature == "prev_action-unclassified":
        assert "unclassified" in row.measured, row.measured
