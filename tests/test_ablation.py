"""The M1.2 / M1.3 ablation (spec MS3.features §7): the steps, their variants, the gain rows over
the previous step with match-bootstrap CIs, and one model card per step with measured numbers.

The card tests read the step cards that ``models.run`` writes (MS3.models §8: ``ablation.run``
moved there) on the hand-built ``ablation_cases`` snapshot plus penalties; the CLI, determinism
and real-snapshot tests moved to ``test_models_run.py``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from test_baselines_cards import _rows, _section
from test_cards_gain import GAIN, _cell

from football_shot_models import baselines, evaluation, features

THRESH = "### MS2 thresholds (retired; see the headline model's MS3 checks)"

STEP_NAMES = ("xg-lr-splines", "xg-lr-context", "xg-lr-frames")
STEP_TITLES = (
    "LR splines + body part",
    "LR splines + context",
    "LR splines + context + freeze frames",
)
SPLINES = (
    "distance_m", "distance_m_rcs1", "distance_m_rcs2", "distance_m_rcs3", "angle_rad",
    "angle_rad_rcs1", "angle_rad_rcs2", "angle_rad_rcs3", "body_part",
)  # fmt: skip
CONTEXT = ("phase", "assist_type", "contact", "open_goal", "one_on_one")
FRAME_GROUPS = {
    "cone": ("opponents_in_cone", "teammates_in_cone", "gk_in_cone"),
    "nearest": ("nearest_opponent_m",),
    "keeper": ("gk_absent", "gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m"),
}


def test_steps_nest_and_use_make_lr(xg_data: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from football_shot_models import ablation

    (_, scaler), (_, lr) = ablation.make_lr().steps
    assert isinstance(scaler, StandardScaler) and isinstance(lr, LogisticRegression)
    assert (lr.C, lr.solver, lr.class_weight) == (math.inf, "newton-cholesky", None)
    assert (lr.tol, lr.max_iter) == (baselines.TOL, baselines.MAX_ITER)
    monkeypatch.setattr(baselines, "MAX_ITER", 7)  # read at call time, as make_model
    assert ablation.make_lr()[-1].max_iter == 7
    monkeypatch.undo()
    assert ablation.SPLINES == SPLINES
    assert ablation.CONTEXT == CONTEXT
    assert ablation.REJECTED == ("first_time", "gender")
    assert dict(ablation.FRAME_GROUPS) == FRAME_GROUPS
    assert [s.name for s in ablation.STEPS] == list(STEP_NAMES)
    assert [s.title for s in ablation.STEPS] == list(STEP_TITLES)
    frames = tuple(c for group in FRAME_GROUPS.values() for c in group)
    assert [s.features for s in ablation.STEPS] == [
        SPLINES, SPLINES + CONTEXT, SPLINES + CONTEXT + frames,
    ]  # fmt: skip
    empty, ff = xg_data.shots.head(0), xg_data.freeze_frames
    widths = [len(features.design_matrix(empty, s.features, freeze_frames=ff)[1]) for s in
              ablation.STEPS]  # fmt: skip
    assert widths == [11, 27, 35]
    for spec in ablation.STEPS:
        assert spec.make()[-1].get_params() == ablation.make_lr()[-1].get_params()
        assert spec.coefficients is baselines.lr_coefficients
    # the step before the splines is the MS2 best baseline
    assert baselines.BASELINES[2].features == ("distance_m", "angle_rad", "body_part")


def test_variants() -> None:
    from football_shot_models import ablation

    expected = [
        *(("xg-lr-context", "drop", (c,)) for c in CONTEXT),
        *(("xg-lr-context", "add", (c,)) for c in ("first_time", "gender")),
        *(("xg-lr-frames", "drop", cols) for cols in FRAME_GROUPS.values()),
    ]
    assert [(v.step, v.kind, v.columns) for v in ablation.VARIANTS] == expected
    assert len(ablation.VARIANTS) == 10


def test_variant_names() -> None:
    """Ratify R5: the full variant names (a frame group by its name, other columns joined by
    "+"), equal to the spec names, and no step lookup for a step outside STEPS (R14)."""
    from football_shot_models import ablation

    expected = [
        *(f"xg-lr-context-drop-{c}" for c in CONTEXT),
        "xg-lr-context-add-first_time",
        "xg-lr-context-add-gender",
        *(f"xg-lr-frames-drop-{g}" for g in FRAME_GROUPS),
    ]
    assert [v.name for v in ablation.VARIANTS] == expected
    assert [v.spec().name for v in ablation.VARIANTS] == expected
    assert ablation.Variant("m16-step", "add", ("a", "b")).name == "m16-step-add-a+b"
    assert ablation.Variant("m16-step", "drop", FRAME_GROUPS["cone"]).name == "m16-step-drop-cone"


class _Lazy(Mapping[str, Any]):
    """Evaluations by name; a name it has not seen gets the next evaluation of ``pool``, so a
    test can check which variant evaluation each gain row used, whatever the variant names."""

    def __init__(self, fixed: dict[str, Any], pool: list[Any]) -> None:
        self.fixed, self.pool, self.made = fixed, pool, dict[str, Any]()

    def __getitem__(self, key: str) -> Any:
        if key in self.fixed:
            return self.fixed[key]
        if key not in self.made:
            if not self.pool:
                raise KeyError(key)
            self.made[key] = self.pool.pop(0)
        return self.made[key]

    def __iter__(self) -> Iterator[str]:
        return iter([*self.fixed, *self.made])

    def __len__(self) -> int:
        return len(self.fixed) + len(self.made)


def test_comparisons_wiring(dev_held: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """Rows for chain[i], i >= 1: over chain[i - 1], over chain[0] (from i = 2), vs the
    benchmark, then one per variant of that step (drop: the variant is the reference; add: the
    step is). Pure wiring over ``compare``: the evaluations here are cheap stand-ins under the
    step names, so a row that pairs the wrong two shows up as a different GainCI."""
    from football_shot_models import ablation

    dev, held = dev_held
    folds = evaluation.cv_folds(dev)

    def ev(name: str, feats: tuple[str, ...]) -> evaluation.Evaluation:
        make = evaluation.BaseRate if not feats else baselines.make_model
        return evaluation.evaluate(evaluation.ModelSpec(name, name, feats, make), dev, held, folds)

    s0, ctx, frm = baselines.BASELINES[2].name, "xg-lr-context", "xg-lr-frames"
    chain = {
        s0: ev("a", ()),
        ctx: ev("b", ("distance_m",)),
        frm: ev("c", ("distance_m", "angle_rad")),
    }
    pool = [
        ev("v1", ("angle_rad",)),
        ev("v2", ("distance_m", "body_part")),
        ev("v3", ("body_part",)),
    ]
    bench = evaluation.evaluate_benchmark(dev, held, folds)
    variants = [
        ablation.Variant(ctx, "drop", ("phase",)),
        ablation.Variant(ctx, "add", ("gender",)),
        ablation.Variant(frm, "drop", FRAME_GROUPS["cone"]),
    ]
    lazy = _Lazy(chain, list(pool))
    rows = ablation.comparisons(lazy, bench, dev, held, chain=(s0, ctx, frm), variants=variants)
    assert set(rows) == {ctx, frm}
    # each variant was looked up once, under a name f"{step}-{kind}-..."
    made = {
        (v.step, v.kind): next(
            e for k, e in lazy.made.items() if k.startswith(f"{v.step}-{v.kind}-")
        )
        for v in variants
    }
    assert len(lazy.made) == 3

    def gains(ref: Any, new: Any) -> tuple[Any, Any]:
        c = evaluation.compare("x", ref, new, dev, held)
        return c.cv, c.holdout

    assert [(r.cv, r.holdout) for r in rows[ctx]] == [
        gains(chain[s0], chain[ctx]),
        gains(bench, chain[ctx]),
        gains(made[(ctx, "drop")], chain[ctx]),
        gains(chain[ctx], made[(ctx, "add")]),
    ]
    assert [(r.cv, r.holdout) for r in rows[frm]] == [
        gains(chain[ctx], chain[frm]),
        gains(chain[s0], chain[frm]),
        gains(bench, chain[frm]),
        gains(made[(frm, "drop")], chain[frm]),
    ]
    drop, add, group = rows[ctx][2].label, rows[ctx][3].label, rows[frm][3].label
    assert "phase" in drop and drop.endswith(": dropped")
    assert "gender" in add and add.endswith(": added (not in the model)")
    assert group.endswith(": dropped")
    labels = [r.label for r in (*rows[ctx], *rows[frm])]
    assert len(set(labels[:2])) == 2 and len(set(labels[4:7])) == 3  # rows are told apart


def test_comparisons_any_chain(dev_held: tuple[pl.DataFrame, pl.DataFrame]) -> None:
    """Ratify R14: a chain and variants of steps outside ``ablation.STEPS`` (M1.6's own split)
    are wired by ``Variant.name``, with no lookup in the module's steps."""
    from football_shot_models import ablation

    dev, held = dev_held
    folds = evaluation.cv_folds(dev)

    def ev(name: str, feats: tuple[str, ...]) -> evaluation.Evaluation:
        make = evaluation.BaseRate if not feats else baselines.make_model
        return evaluation.evaluate(evaluation.ModelSpec(name, name, feats, make), dev, held, folds)

    drop, add = (
        ablation.Variant("m16-b", "drop", ("phase",)),
        ablation.Variant("m16-b", "add", ("gender",)),
    )
    evs = {
        "m16-a": ev("a", ()),
        "m16-b": ev("b", ("distance_m",)),
        "m16-b-drop-phase": ev("v1", ("angle_rad",)),
        "m16-b-add-gender": ev("v2", ("distance_m", "body_part")),
    }
    bench = evaluation.evaluate_benchmark(dev, held, folds)
    rows = ablation.comparisons(
        evs, bench, dev, held, chain=("m16-a", "m16-b"), variants=(drop, add)
    )
    assert list(rows) == ["m16-b"]
    got = [(r.cv, r.holdout) for r in rows["m16-b"]]
    pairs = [
        ("m16-a", "m16-b"), (None, "m16-b"), ("m16-b-drop-phase", "m16-b"),
        ("m16-b", "m16-b-add-gender"),
    ]  # fmt: skip
    want = []
    for ref, new in pairs:
        c = evaluation.compare("x", bench if ref is None else evs[ref], evs[new], dev, held)
        want.append((c.cv, c.holdout))
    assert got == want


@pytest.fixture(scope="module")
def fixture_run(models_run: dict[str, Any]) -> dict[str, Any]:
    """The step cards of ``models.run`` (MS3.models §8: ``ablation.run`` moved there), on the
    ablation fixture plus penalties, with the loaded dev / hold-out."""
    run = models_run["run"]
    return {
        "docs": models_run["docs"],
        "run": run,
        "evs": {e.spec.name: e for e in run.evaluations},
        "rows": run.rows,
        "dev": models_run["dev"],
        "held": models_run["held"],
    }


def test_models_run_writes_step_cards(fixture_run: dict[str, Any]) -> None:
    """The MS3.features criterion, now through ``models.run``: one card per step, with the gain
    over the previous step (and S0, and StatsBomb, and the variants) and a match-bootstrap CI;
    the benchmark table lists the GBM too (MS3.models §1.10)."""
    from football_shot_models import ablation

    docs, evs, gain_rows = fixture_run["docs"], fixture_run["evs"], fixture_run["rows"]
    assert {*STEP_NAMES, *(b.name for b in baselines.BASELINES), "xg-gbm"} <= set(evs)
    assert {k: len(gain_rows[k]) for k in STEP_NAMES} == dict(
        zip(STEP_NAMES, (2, 10, 6), strict=True)
    )
    s0, s1 = evs[baselines.BASELINES[2].name], evs["xg-lr-splines"]
    first = evaluation.compare("x", s0, s1, fixture_run["dev"], fixture_run["held"])
    assert (gain_rows["xg-lr-splines"][0].cv, gain_rows["xg-lr-splines"][0].holdout) == (
        first.cv,
        first.holdout,
    )
    titles = [
        *(b.title for b in baselines.BASELINES), *STEP_TITLES, "Monotone GBM + Platt",
        "StatsBomb xG",
    ]  # fmt: skip
    for spec in ablation.STEPS:
        card = (docs / f"{spec.name}.md").read_text(encoding="utf-8")
        lines = card.splitlines()
        assert lines[0] == f"# Model card: {spec.title}"
        assert "sb:" not in card
        assert THRESH not in lines and "**Why.**" not in card
        assert [r[0] for r in _rows(_section(card, "## Benchmarks"))] == titles
        overview = _section(card, "## Overview")
        assert "headline model" in overview and "best model" not in overview
        gain = [[c.label, _cell(c.cv), _cell(c.holdout)] for c in gain_rows[spec.name]]
        assert _rows(_section(card, GAIN)) == gain
        feature_block = _section(card, "## Features")
        for f in spec.features:
            assert ablation.FEATURE_ROWS[f] in feature_block, f
        model = _section(card, "## Model")
        assert "newton-cholesky" in model
        # the S1 card says its reference S0 is lbfgs and how closely the two solvers agree
        assert ("lbfgs" in model and "1.0e-7" in model) == (spec.name == "xg-lr-splines")
        assert "v1 (MS3): first version." in _section(card, "## Changelog")
        log_loss_row = next(r for r in _rows(_section(card, "## Metrics")) if r[0] == "Log loss")
        assert f"{evs[spec.name].cv.log_loss:.4f}" in log_loss_row[2]  # measured numbers
    for name in ("distance_m", "angle_rad", "body_part"):
        assert ablation.FEATURE_ROWS[name] == baselines.FEATURE_ROWS[name]


def test_card_wide_dfk_row(fixture_run: dict[str, Any]) -> None:
    """The context and frame cards report the wide direct free kicks (logged on outcome) out of
    fold: count, goal rate, this model's mean xG and StatsBomb's, from the dev shots."""
    dev, evs = fixture_run["dev"], fixture_run["evs"]
    wide = (pl.col("situation") == "direct_free_kick") & (
        (pl.col("y") < features.BOX_RIGHT_Y) | (pl.col("y") > features.BOX_LEFT_Y)
    )
    mask = dev.select(wide).to_series().to_numpy()
    n = int(mask.sum())
    assert n >= 20  # the fixture puts wide direct free kicks in dev
    rate = float(dev["is_goal"].to_numpy()[mask].mean())
    sb = float(dev["provider_xg"].to_numpy()[mask].mean())
    for name in STEP_NAMES:
        card = (fixture_run["docs"] / f"{name}.md").read_text(encoding="utf-8")
        x = float(np.mean(evs[name].oof[mask]))
        line = (
            f"Wide direct free kicks (y < 13.84 or > 54.16, n {n} in dev): goal rate {rate:.3f}, "
            f"this model's mean xG {x:.3f}, StatsBomb {sb:.3f}"
        )
        assert (line in card) == (name != "xg-lr-splines"), name
    assert "Wide direct free kicks" not in (fixture_run["docs"] / "xg-lr-splines.md").read_text()


# The CLI, determinism and real-snapshot tests moved to test_models_run.py with ``models.run``.


def test_holdout_lines() -> None:
    """run3-ms3 (architect 4, football 9): the drop-one line names only drop rows whose hold-out
    interval does not lie above 0, with ``side``; added rows are ignored; no drop rows -> None.
    The women's line follows the hold-out's women's shot count."""
    import dataclasses

    import models_cases as mc

    from football_shot_models import ablation, cards

    cv = mc.gain(0.003, 0.002, 0.004)
    above, zero, below = (
        mc.gain(0.002, 0.0001, 0.004),
        mc.gain(0.001, 0.0, 0.002),
        mc.gain(-0.002, -0.004, -0.001),
    )
    rows = [
        evaluation.Comparison("over S1", cv, below),
        evaluation.Comparison("`open_goal`: dropped", cv, above),
        evaluation.Comparison("`phase`: dropped", cv, zero),
        evaluation.Comparison("keeper group (`gk_absent`, `gk_to_goal_m`): dropped", cv, below),
        evaluation.Comparison("`gender`: added (not in the model)", cv, below),
    ]
    assert ablation.holdout_drops(rows) == (
        "On the hold-out the drop-one interval of `phase` (+0.0010 [+0.0000, +0.0020]) includes 0;"
        " of keeper group (-0.0020 [-0.0040, -0.0010]) lies below 0 (this card's gain table); "
        "the inclusion rule is decided on CV."
    )
    assert ablation.holdout_drops(rows[:2]) == (
        "On the hold-out every drop-one interval in this card's gain table lies above 0."
    )
    assert ablation.holdout_drops(rows[:1]) is None and ablation.holdout_drops([]) is None
    c = mc.card_context()
    assert ablation.holdout_women(c) == (
        "The hold-out (Premier League 2015/2016) has no women's shots, so women are judged on "
        "the pooled out-of-fold predictions only."
    )
    some = dataclasses.replace(c, holdout_female_shots=1234)
    assert "has 1,234 women's shots" in ablation.holdout_women(some)
    assert isinstance(some, cards.CardContext)
