"""M1.6 / M1.7: ``v1.run`` on the synthetic WhoScored snapshot (spec MS4.v1 §4, §5, §9), the
v1 checks and miss reasons, the v0 boundary, the CLI; and, in ``make reproduce``, the same run
on the private snapshot (marked ``slow`` and ``whoscored``).

The fast tests share one module-scoped run (``v1_cases``: 15 train and 12 test matches, three
test competitions of unequal size, two null-assist shots, penalties in both seasons) with a
fixture-sized GBM grid swapped in for ``v1.GBM`` and the fixture v0's hash patched into
``legacy.V0_SHA256``. ``models.headline`` is wrapped by a spy so the headline call can be
checked.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import math
import re
from pathlib import Path
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import v1_cases as vc
from football_core.io import whoscored
from test_baselines_cards import _rows

from football_shot_models import baselines, cards, data, evaluation, metrics, models

REPO = Path(__file__).resolve().parents[1]
CHECKS_TITLE = "Milestone checks (MS4)"
SECTIONS = (
    "Overview", "Intended use", "What it is not", "Data", "Split", "Features", "Model",
    "Metrics", "Models", "Gains", "Reliability", "Test by competition", "Big-chance ablation",
    "Penalties", "Limitations", "Data licensing", "Changelog",
)  # fmt: skip


def _mods() -> tuple[Any, Any, Any]:
    from football_shot_models import legacy, release, v1

    return legacy, release, v1


def _block(card: str, name: str) -> str:
    """The text under the first ``##``/``###`` heading starting with ``name``, up to the next
    heading of the same or a higher level."""
    lines = card.splitlines()
    start = next(i for i, x in enumerate(lines) if re.match(rf"#{{2,3}} {re.escape(name)}", x))
    level = len(lines[start].split(" ")[0])
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(rf"#{{1,{level}}} ", lines[i])),
        len(lines),
    )
    return "\n".join(lines[start + 1 : end])


@pytest.fixture(scope="module")
def snap(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("v1")
    manifest = vc.write_v1_snapshot(root)
    v0 = root / "data" / "legacy" / "xg_model.json"
    return {"root": root, "manifest": manifest, "v0": v0, "sha": vc.write_v0(v0)}


@pytest.fixture(scope="module")
def v1r(snap: dict[str, Any]) -> dict[str, Any]:
    """One ``v1.run`` on the fixture; its stdout, the headline spy's calls and the frames."""
    legacy, _, v1 = _mods()
    calls: list[tuple[tuple[Any, ...], Any]] = []
    real = models.headline

    def spy(*args: Any) -> Any:
        out = real(*args)
        calls.append((args, out))
        return out

    docs = snap["root"] / "docs" / "models"
    out = io.StringIO()
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
        mp.setattr(legacy, "V0_SHA256", snap["sha"])
        mp.setattr(v1, "GBM", mc.tiny_gbm(v1, dict(v1.GBM_SIGNS)))
        mp.setattr(models, "headline", spy)
        result = v1.run(snap["manifest"], v0=snap["v0"], docs_dir=docs, cache=None)
    xg = data.load_xg_data(snap["manifest"], builder="whoscored")
    ho = data.season_holdout(xg.matches, vc.TRAIN_SEASONS, vc.TEST_SEASONS)
    shots = xg.shots.filter(pl.col("assist_type").is_not_null())
    dev, held = data.split_holdout(shots, ho)
    _, held_all = data.split_holdout(xg.shots, ho)
    return {"run": result, "stdout": out.getvalue(), "calls": calls, "docs": docs, "xg": xg,
            "dev": dev, "held": held, "held_all": held_all, "real_headline": real}  # fmt: skip


def test_v1_specs() -> None:
    """§4.1: the constants, the two steps, the GBM (monotone distance -1 / angle +1) and the
    labelled ablation of a spec; no candidate reads an ablation-only column, gender, first_time
    or a freeze-frame feature."""
    from football_shot_models import ablation, features, gbm

    _, _, v1 = _mods()
    assert (v1.BUILDER, v1.DEFAULT_MANIFEST) == ("whoscored", "manifests/whoscored-v1.json")
    assert (v1.TRAIN_SEASONS, v1.TEST_SEASONS) == (("2025/2026", "2026"), ("2026/2027",))
    assert v1.CONTEXT == ("phase", "assist", "one_on_one")
    assert [(s.name, s.features) for s in v1.STEPS] == [
        ("ws-lr-splines", ablation.SPLINES), ("ws-lr-context", ablation.SPLINES + v1.CONTEXT),
    ]  # fmt: skip
    assert all(s.make is ablation.make_lr and s.guard is None for s in v1.STEPS)
    assert v1.GBM.name == "ws-gbm" and v1.GBM.coefficients is gbm.gbm_settings
    assert v1.GBM.features == ("distance_m", "angle_rad", "body_part", *v1.CONTEXT)
    assert dict(v1.GBM_SIGNS) == {"distance_m": -1, "angle_rad": 1}
    assert dict(v1.GBM.make().get_params()["monotone"]) == dict(v1.GBM_SIGNS)
    assert (v1.MISS_ABOVE, v1.WINDOW, v1.START_LINE) == (0.275, (0.26, 0.27), 0.297)
    for spec in (*baselines.BASELINES, *v1.STEPS, v1.GBM):
        assert not features.uses_ablation_only(spec.features), spec.name
        assert not {"gender", "first_time", "big_chance"} & set(spec.features), spec.name
        for name in spec.features:
            sources = features.DERIVED[name].sources if name in features.DERIVED else ()
            assert all(t != "freeze_frames" for t, _ in sources), (spec.name, name)
    abl = v1.ablation_spec(v1.STEPS[1])
    assert abl.name == "ws-lr-context-big-chance"
    assert abl.title == "LR splines + context + big chance (labelled ablation)"
    assert abl.features == (*v1.STEPS[1].features, "big_chance")
    assert abl.guard is features.check_ablation_features
    assert (abl.make, abl.coefficients) == (v1.STEPS[1].make, v1.STEPS[1].coefficients)


def test_v1_public_seams() -> None:
    """§2.1, §2.5, §3: the frameless WhoScored builder, the legacy constants (the pinned hash,
    v0's feature order, the title computed from the q number), and the renamed public helpers
    (no output change; ``PenaltyRow`` says "provider mean xG")."""
    from football_shot_models import penalties

    legacy, _, _ = _mods()
    assert dict(data.FRAMES_BY_BUILDER) == {"statsbomb": True, "whoscored": False}
    assert legacy.V0_SHA256 == "24c8f23d059b5dcab82858ad1ea6554ec3fc599a3da5071f3c2108c8c657b740"
    assert legacy.V0_FEATURES[4] == "big_chance" and len(legacy.V0_FEATURES) == 10
    title = f"Legacy v0 (uses Opta big chance q{whoscored.BIG_CHANCE})"
    assert title == legacy.V0_TITLE
    assert legacy.DEFAULT_V0 == "data/legacy/xg_model.json"
    assert callable(cards.metrics_table) and callable(cards.checks_table)
    assert callable(models.gbm_grid)
    doc = penalties.PenaltyRow.__doc__ or ""
    assert "provider mean xG" in doc and "StatsBomb" not in doc


def test_v1_population(snap: dict[str, Any]) -> None:
    """§4.2: ``population`` keeps the shots with a non-null ``assist_type`` in order and counts
    the dropped ones (all seasons); a frame without the column is a ``ValueError``."""
    _, _, v1 = _mods()
    xg = data.load_xg_data(snap["manifest"], builder="whoscored")
    shots, dropped = v1.population(xg)
    assert dropped == vc.DROPPED_ASSIST == xg.shots["assist_type"].null_count()
    assert shots.equals(xg.shots.filter(pl.col("assist_type").is_not_null()))
    assert {vc.NULL_ASSIST_TRAIN, vc.NULL_ASSIST_TEST} <= set(xg.shots["shot_id"])
    with pytest.raises(ValueError, match="assist_type"):
        v1.population(dataclasses.replace(xg, shots=xg.shots.drop("assist_type")))


def test_v1_run_fixture(v1r: dict[str, Any]) -> None:
    """§4.2, §5: one card of record (``xg-v1.md``) with every section; the candidates, the best
    LR from baselines and steps only, the ablation built from it (never headline though its CV
    loss is lowest), the comparisons, the v0 gains and counterfactuals from the run's own
    predictions, the 5 checks, the coverage line on both denominators, the competitions table,
    no id, date or file name, and the README block."""
    legacy, release, v1 = _mods()
    r, dev, held, held_all = v1r["run"], v1r["dev"], v1r["held"], v1r["held_all"]
    assert isinstance(r, release.V1Run)
    assert sorted(p.name for p in v1r["docs"].iterdir()) == ["xg-v1.md"]
    card = (v1r["docs"] / "xg-v1.md").read_text(encoding="utf-8")
    # population and split
    assert (r.train_seasons, r.test_seasons) == (v1.TRAIN_SEASONS, v1.TEST_SEASONS)
    assert r.dropped_assist == vc.DROPPED_ASSIST
    assert (held_all.height, held.height) == (vc.TEST_MATCHES * vc.SHOTS_PER_MATCH,
                                              vc.TEST_MATCHES * vc.SHOTS_PER_MATCH - 1)  # fmt: skip
    assert r.context.holdout_label == "2026/2027"
    assert (r.context.dev_shots, r.context.holdout_shots) == (dev.height, held.height)
    # candidates and the headline
    assert [e.spec.name for e in r.baselines] == [s.name for s in baselines.BASELINES]
    assert [e.spec.name for e in r.steps] == ["ws-lr-splines", "ws-lr-context"]
    assert r.best_lr is evaluation.best((*r.baselines, *r.steps))
    assert r.ablation.spec.name == v1.ablation_spec(r.best_lr.spec).name
    assert r.ablation.spec.features == (*r.best_lr.spec.features, "big_chance")
    assert r.ablation.cv.log_loss < min(e.cv.log_loss for e in (*r.baselines, *r.steps, r.gbm))
    assert r.head is not r.ablation and r.head in (r.gbm, r.best_lr)
    da = r.baselines[1]
    # §4.2 step 5 in order; a pair repeated by identity (head is the best LR) is one row
    pairs = [(da, r.head), (da, r.best_lr), (r.best_lr, r.gbm), (r.best_lr, r.ablation)]
    unique = [p for i, p in enumerate(pairs) if all(p[0] is not q[0] or p[1] is not q[1]
                                                    for q in pairs[:i])]  # fmt: skip
    expected = [evaluation.compare("x", a, b, dev, held) for a, b in unique]
    assert [(g.cv, g.holdout) for g in r.gains] == [(g.cv, g.holdout) for g in expected]
    assert any(g.cv == r.gbm_gain.cv for g in r.gains)
    # v0: coverage, scores, gains, counterfactuals
    cov = r.coverage
    assert (cov.covered, cov.population, cov.dropped) == (held.height, held.height, 1)
    assert (cov.scored_all, cov.kept_all) == (held_all.height - 1, held_all.height)
    y, groups = evaluation.labels(held), held["match_id"].to_numpy()
    p0 = legacy.score_v0(held, r.v0).to_numpy()
    assert r.v0_scores == metrics.score(y, p0)
    names = [n for n, _ in r.v0_gains]
    spec_order = list(dict.fromkeys(e.spec.name for e in (r.head, da, r.best_lr, r.ablation)))
    assert [n for n in names if n in spec_order] == spec_order  # extra rows allowed (choice 2)
    assert len(set(names)) == len(names) and names[0] == r.head.spec.name
    by_name = {e.spec.name: e for e in (*r.baselines, *r.steps, r.gbm, r.ablation)}
    for name, g in r.v0_gains:
        assert g == metrics.bootstrap_gain(y, p0, by_name[name].holdout_p, groups), name
    q = f"q{whoscored.BIG_CHANCE}"
    assert r.add.label == f"the best LR ({r.best_lr.spec.title}) plus {q} alone"
    assert r.add.test_log_loss == r.ablation.holdout.log_loss
    assert r.add.gain_over_v0 == dict(r.v0_gains)[r.ablation.spec.name]
    assert r.remove.label == f"v0's inputs refitted on train without {q}"
    p_remove = legacy.refit_v0(dev, held, ("big_chance",))
    assert r.remove.test_log_loss == pytest.approx(metrics.score(y, p_remove).log_loss, abs=1e-12)
    # run4-ms4 #12: the headline against the remove refit, v0's information without q214
    over = metrics.bootstrap_gain(y, p_remove, r.head.holdout_p, groups)
    got = r.head_over_remove
    assert (got.gain, got.low, got.high) == pytest.approx((over.gain, over.low, over.high),
                                                          abs=1e-12)  # fmt: skip
    assert got.gain != r.v0_gains[0][1].gain  # not the gain over v0 itself
    # checks and reasons
    assert r.checks == v1.v1_checks(r.head, r.gbm, r.gbm_gain, da, r.v0_gains[0][1], cov)
    assert len(r.checks) == 5 and [c.result for c in r.checks[3:]] == ["met", "met"]
    missed = [c.result == "missed" for c in r.checks[:2]]
    reasons = [release.miss_reason(i + 1, m, r.add, r.remove, v1.MISS_ABOVE)
               for i, m in enumerate(missed)]  # fmt: skip
    assert r.reasons == tuple(x for x in reasons if x is not None)
    # by competition
    comps = vc.TEST_COMPETITIONS
    assert [g.key for g in r.groups] == sorted(comps) == [g.key for g in r.v0_groups]
    shots = {k: n * vc.SHOTS_PER_MATCH for k, n in comps.items()}
    shots["Fixture League A"] -= 1  # the null-assist test shot (ws:m01)
    assert [g.shots for g in r.groups] == [shots[k] for k in sorted(comps)]
    assert [g.shots for g in r.v0_groups] == [g.shots for g in r.groups]
    # run4-ms4 #15: the values, against by_group on an independently joined key
    comp = held.join(
        v1r["xg"].matches.select("match_id", "competition_name"),
        on="match_id", how="left", maintain_order="left",
    )["competition_name"].cast(pl.String)  # fmt: skip
    assert comp.to_list() != sorted(comp.to_list())  # shot order is not competition order
    seg = models.MIN_SEGMENT_SHOTS
    assert r.groups == metrics.by_group(y, r.head.holdout_p, comp, groups, seg)
    assert r.v0_groups == metrics.by_group(y, p0, comp, groups, seg)
    # the card
    lines = card.splitlines()
    for name in SECTIONS:
        assert any(re.match(rf"#{{2,3}} {re.escape(name)}", x) for x in lines), name
    assert r.context.data_digest in card and r.v0.sha256[:12] in card
    coverage_line = (
        f"v0 covers {cov.covered:,} of {cov.population:,} test shots in the v1 population "
        f"({cov.dropped:,} dropped for unresolved assist, q{whoscored.RELATED}; v0 scores "
        f"{cov.scored_all:,} of {cov.kept_all:,} kept test shots)"
    )
    assert coverage_line in card
    assert "n/a (no frames)" in _block(card, "Data")
    assert "every train match precedes every test match (checked)" in _block(card, "Split").lower()
    models_block = _block(card, "Models")
    for title in (*(e.spec.title for e in (*r.baselines, *r.steps, r.gbm, r.ablation)),
                  legacy.V0_TITLE):  # fmt: skip
        assert title in models_block, title
    v0_row = next(
        x for x in models_block.splitlines() if x.startswith("|") and legacy.V0_TITLE in x
    )
    assert "n/a (test-only benchmark)" in v0_row
    head_row = next(x for x in models_block.splitlines()
                    if x.startswith(f"| **{r.head.spec.title}** |"))  # fmt: skip
    assert r.head.holdout.xg_signed_error != 0.0 and r.v0_scores.xg_signed_error != 0.0
    assert head_row.endswith(f"| {r.head.holdout.xg_signed_error:+.2%} |"), head_row
    assert v0_row.endswith(f"| {r.v0_scores.xg_signed_error:+.2%} |"), v0_row
    over_row = (f"| {r.head.spec.title} over {r.remove.label} (test only, counterfactual "
                f"reference in place of v0) | {cards.gain_cell(r.head_over_remove)} | "
                f"{cards.side(r.head_over_remove)} |")  # fmt: skip
    assert over_row in _block(card, "Gains").splitlines()
    table = cards.checks_table(r.checks, CHECKS_TITLE, "note")
    assert all(x in lines for x in table if x.startswith("|") or CHECKS_TITLE in x)
    assert all(reason in card for reason in r.reasons)
    by_comp = _block(card, "Test by competition")
    assert sorted(row[0] for row in _rows(by_comp)) == sorted(comps)
    k = sum(g.error_ci is not None and (g.error_ci[0] > 0 or g.error_ci[1] < 0) for g in r.groups)
    j = sum(g.slope_ci is not None and (g.slope_ci[0] > 1 or g.slope_ci[1] < 1) for g in r.groups)
    n = len(r.groups)
    assert (f"{k} of {n} signed-error intervals exclude 0; {j} of {n} slope intervals "
            "exclude 1") in by_comp  # fmt: skip
    g, o = r.groups[0], r.v0_groups[0]
    assert g.signed_error is not None and o.signed_error is not None
    assert g.error_ci is not None and o.error_ci is not None and g.signed_error != o.signed_error
    line = next(x for x in by_comp.splitlines() if x.startswith(f"| {g.key} |"))
    assert line.startswith(
        f"| {g.key} | {g.shots:,} | {g.goals:,} | {g.xg:.1f} | {g.signed_error:+.2%} "
        f"[{g.error_ci[0]:+.2%}, {g.error_ci[1]:+.2%}] | {cards.fmt(g.cal_slope)} "
    ), line
    assert line.endswith(f"| {o.signed_error:+.2%} [{o.error_ci[0]:+.2%}, {o.error_ci[1]:+.2%}] |")
    assert re.search(r"private WhoScored ?/ ?Opta archive, never committed or redistributed", card)
    assert re.search(r"\bws:", card) is None  # a word boundary: "Rows:" is not an id
    assert "xg_model.json" not in card
    assert re.search(r"\b\d{4}-\d{2}-\d{2}\b", card) is None
    # the README block and the stdout
    block = release.summary_block(r)
    assert legacy.V0_TITLE in block and f"**{r.head.spec.title}**" in block
    assert r.ablation.spec.title in block and re.search(r"\bws:", block) is None
    rate = held["is_goal"].cast(pl.Float64).mean()
    assert (f"WhoScored test season, {held.height:,} shots, goal rate {rate:.4f}; not comparable "
            "with the StatsBomb block") in block  # fmt: skip
    assert block in v1r["stdout"] and all(x in v1r["stdout"] for x in r.reasons)
    assert (f"Over {r.remove.label} (a labelled counterfactual: v0's information without big "
            f"chance) its gain is {cards.gain_cell(r.head_over_remove)}, an interval that "
            f"{cards.side(r.head_over_remove)}.") in block  # fmt: skip


def test_v1_headline_on_cv(v1r: dict[str, Any]) -> None:
    """§4.2 step 3, §4.3 row 4: the run calls ``models.headline`` once with the best LR and the
    GBM, and its headline is that call's; test predictions that make the GBM's test gain lie
    above or below 0 leave the headline and the CV gain unchanged."""
    r = v1r["run"]
    ((args, (head, gain)),) = v1r["calls"]
    best_lr, gbm, dev, held = args
    assert best_lr is r.best_lr and gbm is r.gbm and r.head is head
    assert (r.head is r.gbm) == (r.gbm_gain.cv.low > 0) and gain.cv == r.gbm_gain.cv
    y = evaluation.labels(held)
    for p, side in ((np.where(y, 0.999, 0.001), 1), (np.where(y, 0.001, 0.999), -1)):
        patched = dataclasses.replace(gbm, holdout_p=p, holdout=metrics.score(y, p))
        head2, gain2 = v1r["real_headline"](best_lr, patched, dev, held)
        assert (gain2.holdout.low > 0) if side > 0 else (gain2.holdout.high < 0)
        assert head2.spec.name == r.head.spec.name and gain2.cv == r.gbm_gain.cv


def test_v1_penalty_train_only(v1r: dict[str, Any]) -> None:
    """§4.2 step 7, §1.8: the v1 penalty probability is the posterior of the train kicks only
    (5 kicks, 4 goals; not the 8 pooled, not the shoot-outs); the test kicks (3, 1 goal) are
    scored with it and with v0's ``penalty_xg`` as constants, and the card prints both."""
    r = v1r["run"]
    rows = {row.label: row for row in r.penalties}
    dev_rate = rows["In-game: dev"].rate
    assert (dev_rate.n, dev_rate.goals) == (vc.TRAIN_KICKS, vc.TRAIN_KICK_GOALS)
    held_rate = rows["In-game: hold-out"].rate
    assert (held_rate.n, held_rate.goals) == (vc.TEST_KICKS, vc.TEST_KICK_GOALS)
    assert rows["In-game: all"].rate.n == vc.TRAIN_KICKS + vc.TEST_KICKS
    assert dev_rate.mean == metrics.beta_interval(vc.TRAIN_KICK_GOALS, vc.TRAIN_KICKS).mean

    def constant(q: float) -> float:
        g, n = vc.TEST_KICK_GOALS, vc.TEST_KICKS
        return -(g * math.log(q) + (n - g) * math.log1p(-q)) / n

    assert r.penalty_losses == pytest.approx((constant(dev_rate.mean), constant(vc.V0_PENALTY)))
    pooled = metrics.beta_interval(vc.TRAIN_KICK_GOALS + vc.TEST_KICK_GOALS, 8).mean
    assert abs(constant(pooled) - r.penalty_losses[0]) > 0.01  # the near miss is visible
    text = _block((v1r["docs"] / "xg-v1.md").read_text(encoding="utf-8"), "Penalties")
    a, b = (f"{x:.4f}" for x in r.penalty_losses)
    pattern = (rf"constant log loss on the test kicks: dev posterior {a}, v0 `?penalty_xg`?"
               rf"( \({vc.V0_PENALTY:.4f}\))? {b}")  # fmt: skip
    assert re.search(pattern, text), text


def _cf(label: str, loss: float, gain: float) -> Any:
    _, release, _ = _mods()
    return release.Counterfactual(label, loss, metrics.GainCI(gain, gain - 0.003, gain + 0.003,
                                                              2000, 587))  # fmt: skip


ADD = "the best LR (LR splines + context) plus q214 alone"
REMOVE = "v0's inputs refitted on train without q214"


def test_v1_miss_reason() -> None:
    """§4.3: ``None`` when met; row 1 judges each counterfactual's test log loss against the
    bound (equality meets), row 2 its point gain over v0 (0 meets); each of the three branches
    prints both computed values; a row other than 1 or 2 is a ``ValueError`` even when met."""
    _, release, _ = _mods()
    add, rem = _cf(ADD, 0.2608, 0.0006), _cf(REMOVE, 0.2797, -0.0182)
    both = (
        " Both labelled counterfactual refits agree that q214 accounts for the gap: adding "
        "it alone meets the bound, and removing it from v0's inputs misses it."
    )
    assert release.miss_reason(1, True, add, rem, 0.275) == (
        f"{ADD}: test log loss 0.2608 (meets it); {REMOVE}: test log loss 0.2797 (misses it)."
        + both
    )
    assert release.miss_reason(2, True, add, rem, 0.275) == (
        f"{ADD}: gain over v0 +0.0006 [-0.0024, +0.0036] (meets it); {REMOVE}: gain over v0 "
        "-0.0182 [-0.0212, -0.0152] (misses it)." + both
    )
    assert release.miss_reason(1, False, add, rem, 0.275) is None
    assert release.miss_reason(2, False, add, rem, 0.275) is None
    not_isolated = release.miss_reason(1, True, add, rem, 0.2797)  # remove meets at equality
    assert not_isolated == (
        f"{ADD}: test log loss 0.2608 (meets it); {REMOVE}: test log loss 0.2797 (meets it). "
        "Removing q214 does not reproduce the miss: this run does not isolate it."
    )
    not_identified = release.miss_reason(2, True, _cf(ADD, 0.25, -0.0001), _cf(REMOVE, 0.3, 0.0),
                                         0.275)  # fmt: skip
    assert not_identified == (
        f"{ADD}: gain over v0 -0.0001 [-0.0031, +0.0029] (misses it); {REMOVE}: gain over v0 "
        "+0.0000 [-0.0030, +0.0030] (meets it). Adding q214 does not meet the bound: this run "
        "does not identify the reason."
    )
    for row in (0, 3):
        for missed in (True, False):
            with pytest.raises(ValueError, match=f"got {row}"):
                release.miss_reason(row, missed, add, rem, 0.275)


def _ev(spec: Any, loss: float = 0.27, slope: float | None = 1.0) -> Any:
    return mc.fake_eval(spec, holdout=mc.scores(log_loss=loss, slope=slope))


def _comparison(cv_low: float) -> Any:
    return evaluation.Comparison("over LR", mc.gain(cv_low + 0.001, cv_low, cv_low + 0.002),
                                 mc.gain(0.001, -0.001, 0.003))  # fmt: skip


V1_CASES = {
    # head, test log loss, gain over v0, slope, GBM CV gain low -> rows 1-5
    "all met at the bounds": ("gbm", 0.275, 0.0, 0.9, 0.0005, ["met"] * 5),
    "just outside": ("lr", 0.2751, -0.0001, 1.1001, 0.0, ["missed"] * 3 + ["met"] * 2),
    "slope n/a, GBM headline without CI": ("gbm", 0.26, 0.001, None, 0.0,
                                           ["met", "met", "n/a", "missed", "met"]),
    "LR headline though CI above 0": ("lr", 0.26, 0.001, 0.8999, 0.0001,
                                      ["met", "met", "missed", "missed", "met"]),
    "ablation headline": ("ablation", 0.26, 0.001, 1.1, -0.001, ["met"] * 4 + ["missed"]),
}  # fmt: skip


@pytest.mark.parametrize("case", list(V1_CASES))
def test_v1_checks(case: str) -> None:
    """§4.3: the five rows in order, each met exactly when its sentence holds (bounds from the
    constants, equality meets, two-sided slope range); row 1 prints the bound and D + A's
    computed test value, row 2 the coverage counts."""
    from football_shot_models import ablation, gbm

    _, release, v1 = _mods()
    head_kind, loss, gain, slope, low, expected = V1_CASES[case]
    gbm_spec = evaluation.ModelSpec("ws-gbm", "GBM", ("distance_m",), evaluation.BaseRate,
                                    gbm.gbm_settings)  # fmt: skip
    lr_spec = evaluation.ModelSpec("ws-lr", "LR", ("distance_m",), ablation.make_lr)
    abl_spec = v1.ablation_spec(lr_spec)
    g = _ev(gbm_spec, loss, slope)
    head = {"gbm": g, "lr": _ev(lr_spec, loss, slope), "ablation": _ev(abl_spec, loss, slope)}
    da = _ev(baselines.BASELINES[1], 0.2971)
    coverage = release.Coverage(359, 359, 1, 359, 360)
    rows = v1.v1_checks(head[head_kind], g, _comparison(low), da, mc.gain(gain, gain - 0.002,
                        gain + 0.002), coverage)  # fmt: skip
    assert [c.result for c in rows] == expected
    assert "0.275" in rows[0].target and "0.2971" in rows[0].target + rows[0].measured
    assert f"{loss:.4f}" in rows[0].measured
    assert "359 of 359" in rows[1].target + rows[1].measured


def test_v1_v0_boundary(snap: dict[str, Any], tmp_path: Path,
                        monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    """§4.2 step 2, §8: v0 is compared only when its ``trained_on`` date precedes the first test
    match: a day before passes; the first test date itself and a date inside the test season
    are a ``ValueError`` naming both dates, raised before any fit; no test match date is a
    ``ValueError``."""
    import datetime as dt

    legacy, _, v1 = _mods()
    matches = data.load_xg_data(snap["manifest"], builder="whoscored").matches
    ho = data.season_holdout(matches, vc.TRAIN_SEASONS, vc.TEST_SEASONS)

    def model(day: dt.date) -> Any:
        return dataclasses.replace(vc.legacy_model(vc.V0_COEF), trained_on=day)

    assert v1.check_v0_boundary(model(vc.FIRST_TEST - dt.timedelta(days=1)), matches, ho) is None
    with pytest.raises(ValueError, match=r"(?s)2026-08-08.*2026-08-08"):
        v1.check_v0_boundary(model(vc.FIRST_TEST), matches, ho)
    none = data.Holdout(pl.col("match_id") == "ws:none", "none")
    with pytest.raises(ValueError):
        v1.check_v0_boundary(model(dt.date(2020, 1, 1)), matches, none)

    def refuse_fit(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a fit ran before the v0 boundary check")

    monkeypatch.setattr(evaluation, "_fit", refuse_fit)
    for when, message in (("2026-08-08", r"(?s)2026-08-08.*2026-08-08"),
                          ("2026-09-01", r"(?s)(?=.*2026-09-01)(?=.*2026-08-08)")):  # fmt: skip
        path = tmp_path / when / "xg_model.json"
        monkeypatch.setattr(legacy, "V0_SHA256", vc.write_v0(path, vc.v0_doc(when=when)))
        with pytest.raises(ValueError, match=message):
            v1.run(snap["manifest"], v0=path, docs_dir=tmp_path / "docs", cache=None)
    assert not (tmp_path / "docs" / "xg-v1.md").exists()


def test_v1_cli(snap: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    """§4.2 ``main``: a missing v0 file, a v0 that does not hash as pinned and a missing
    manifest each exit 1 with ``fsm: error:`` and no traceback; a full ``--no-cache`` run
    exits 0, writes the card and creates no cache directory."""
    legacy, _, v1 = _mods()
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "out"
    base = ["--manifest", str(snap["manifest"]), "--docs-dir", str(docs), "--no-cache"]
    for argv in (
        [*base, "--v0", str(tmp_path / "missing.json")],
        [*base, "--v0", str(snap["v0"])],  # the pin is not patched: wrong hash
        ["--manifest", str(tmp_path / "nope.json"), "--v0", str(snap["v0"]), "--no-cache"],
    ):
        assert v1.main(argv) == 1, argv
        err = capsys.readouterr().err
        assert err.startswith("fsm: error:") and "Traceback" not in err, err
    monkeypatch.setattr(legacy, "V0_SHA256", snap["sha"])
    monkeypatch.setattr(v1, "GBM", mc.tiny_gbm(v1, dict(v1.GBM_SIGNS)))
    assert v1.main([*base, "--v0", str(snap["v0"])]) == 0
    assert (docs / "xg-v1.md").is_file()
    assert not (tmp_path / "data").exists()


# ---- build ratify items (build.md "Architect: ratify"), one fast test each ------------------


TRAIN_ONLY, TEST_ONLY = "Fixture Friendly", "Fixture Shield"


@pytest.fixture(scope="module")
def v1r_gbm(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A second fixture run with the GBM forced to headline (the real run's branch), on a
    matches table where one train match is in a train-only competition and one test match in a
    test-only one."""
    legacy, _, v1 = _mods()
    root = tmp_path_factory.mktemp("v1-gbm")
    _, _, m = vc.v1_tables()
    relabel = {"ws:m27": TRAIN_ONLY, "ws:m12": TEST_ONLY}  # m27 is train (2026), m12 test
    m = m.with_columns(
        pl.col("match_id").replace_strict(relabel, default=pl.col("competition_name"))
        .alias("competition_name")
    )  # fmt: skip
    manifest = vc.write_v1_snapshot(root, matches=m)
    v0 = root / "data" / "legacy" / "xg_model.json"
    sha = vc.write_v0(v0)
    real = models.headline

    def gbm_head(best_lr: Any, gbm_ev: Any, dev: Any, held: Any) -> Any:
        return gbm_ev, real(best_lr, gbm_ev, dev, held)[1]

    docs = root / "docs" / "models"
    with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
        mp.setattr(legacy, "V0_SHA256", sha)
        mp.setattr(v1, "GBM", mc.tiny_gbm(v1, dict(v1.GBM_SIGNS)))
        mp.setattr(models, "headline", gbm_head)
        result = v1.run(manifest, v0=v0, docs_dir=docs, cache=None)
    return {"run": result, "card": (docs / "xg-v1.md").read_text(encoding="utf-8")}


def test_ratify_v1_run_fields(v1r: dict[str, Any], v1r_gbm: dict[str, Any]) -> None:
    """Ratify 1 and 5: ``train_competitions`` holds the competitions with a train match (not
    the test ones), ``reliability`` the headline's ten test bins, and "What it is not" names
    the competitions only in train and only in test (none when they are the same)."""
    r, xg, held = v1r["run"], v1r["xg"], v1r["held"]
    train = xg.matches.filter(pl.col("season_name").is_in(vc.TRAIN_SEASONS))
    assert (
        r.train_competitions
        == tuple(sorted(set(train["competition_name"])))
        == tuple(sorted(vc.TEST_COMPETITIONS))
    )
    rel = metrics.reliability(evaluation.labels(held), r.head.holdout_p)
    assert r.reliability == tuple(rel.iter_rows()) and len(r.reliability) == 10
    card = (v1r["docs"] / "xg-v1.md").read_text(encoding="utf-8")
    assert "3 in train, 3 in test." in _block(card, "What it is not")
    assert " only in " not in _block(card, "What it is not")
    g = v1r_gbm["run"]
    assert g.train_competitions == tuple(sorted({*vc.TEST_COMPETITIONS, TRAIN_ONLY}))
    assert TEST_ONLY in {row.key for row in g.groups} and TRAIN_ONLY not in {
        row.key for row in g.groups
    }
    assert (f"4 in train, 4 in test; {TRAIN_ONLY} only in train; {TEST_ONLY} only in test."
            in _block(v1r_gbm["card"], "What it is not"))  # fmt: skip


def test_ratify_v0_gains_order(v1r: dict[str, Any], v1r_gbm: dict[str, Any]) -> None:
    """Ratify 2: ``v0_gains`` is the headline, D + A, the best LR, the GBM and the ablation,
    each once by identity, in that order; the headline is always first."""
    for fx in (v1r, v1r_gbm):
        r = fx["run"]
        order = (r.head, r.baselines[1], r.best_lr, r.gbm, r.ablation)
        once = [ev for i, ev in enumerate(order) if all(ev is not o for o in order[:i])]
        assert len(once) == 4  # one of the five repeats in either branch
        assert [n for n, _ in r.v0_gains] == [ev.spec.name for ev in once]
    lr, gbm_head = v1r["run"], v1r_gbm["run"]
    assert lr.head is lr.best_lr and gbm_head.head is gbm_head.gbm  # both branches exercised
    assert [n for n, _ in lr.v0_gains][2] == "ws-gbm"  # the GBM row the LR branch adds


def test_ratify_gains_rows(v1r: dict[str, Any], v1r_gbm: dict[str, Any]) -> None:
    """Ratify 3: the gains drop a pair repeated by identity, and the GBM row is
    ``headline()``'s comparison relabelled "<GBM title> over <best LR title>", while
    ``gbm_gain`` keeps ``headline()``'s label; the Gains table prints those labels."""
    for fx, n in ((v1r, 3), (v1r_gbm, 4)):
        r = fx["run"]
        da, lr, gbm_t = r.baselines[1].spec.title, r.best_lr.spec.title, r.gbm.spec.title
        rows = [f"{lr} over {da}", f"{gbm_t} over {lr}", f"{r.ablation.spec.title} over {lr}"]
        if r.head is r.gbm:
            rows.insert(0, f"{gbm_t} over {da}")
        assert [g.label for g in r.gains] == rows and len(rows) == n
        gbm_row = r.gains[-2]
        assert (gbm_row.cv, gbm_row.holdout) == (r.gbm_gain.cv, r.gbm_gain.holdout)
        assert r.gbm_gain.label == f"over {lr}"
    gains_rows = [x for x in _block(v1r_gbm["card"], "Gains").splitlines() if x.startswith("| ")]
    for g in v1r_gbm["run"].gains:
        assert sum(x.startswith(f"| {g.label} |") for x in gains_rows) == 1, g.label


def test_ratify_card_wording(v1r: dict[str, Any]) -> None:
    """Ratify 4: the Data line is labelled "**Counts:**" (never "**Rows:**"); the filter rows
    carry the Opta q numbers and "n/a (no frames) | n/a" for the frame filter; the dropped
    assist count is a sentence; the penalty line states v0's ``penalty_xg``; the Models intro
    names v0 by "v0", so the first line holding ``V0_TITLE`` is the table row."""
    legacy, _, _ = _mods()
    r = v1r["run"]
    card = (v1r["docs"] / "xg-v1.md").read_text(encoding="utf-8")
    data_block = _block(card, "Data")
    assert "- **Counts:** train " in data_block and "**Rows:**" not in card
    exc = {reason: (shots, goals) for reason, shots, goals in r.context.exclusions}
    for label, reason in (
        (f"own goal (Opta q{whoscored.OWN_GOAL})", "own_goal"),
        (f"direct corner (Opta q{whoscored.DIRECT_CORNER})", "direct_corner"),
    ):
        shots, goals = exc[reason]
        assert f"| {label} | {shots:,} | {goals:,} |" in data_block.splitlines(), label
    assert "| no freeze frame | n/a (no frames) | n/a |" in data_block.splitlines()
    assert (f"{r.dropped_assist:,} kept shots whose assisting event (Opta q{whoscored.RELATED}) "
            "is unresolved") in data_block  # fmt: skip
    assert f"v0's constant `penalty_xg` is {vc.V0_PENALTY:.4f}." in _block(card, "Penalties")
    models_block = _block(card, "Models")
    assert f"v0 is scored on the {r.coverage.covered:,} test shots it covers." in models_block
    first = next(x for x in models_block.splitlines() if legacy.V0_TITLE in x)
    assert first.startswith(f"| {legacy.V0_TITLE} |")


def test_card_facts_computed(
    v1r: dict[str, Any], v1r_gbm: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """run4-ms4 #4: the CV fold count comes from ``evaluation.N_SPLITS``; the men-only and
    gender lines print only with 0 women's shots, else computed lines; the Overview verdict
    names the split "test", never "hold-out"."""
    _, release, _ = _mods()
    r = v1r["run"]
    card = release.v1_card(r)
    assert f"Grouped {evaluation.N_SPLITS}-fold CV" in card
    assert r.context.female_shots == 0
    assert f"- Men's competitions only ({r.context.male_shots:,} shots, 0 women's): " in card
    assert "shots are women's, so it is one value." in card
    gbm_card = v1r_gbm["card"]
    assert v1r_gbm["run"].head is v1r_gbm["run"].gbm  # the verdict's split branch
    assert "on the test it is " in _block(gbm_card, "Overview")
    assert "hold-out" not in _block(gbm_card, "Overview") + _block(card, "Overview")
    monkeypatch.setattr(evaluation, "N_SPLITS", 7)
    assert "Grouped 7-fold CV" in release.v1_card(r)
    mixed = dataclasses.replace(r, context=dataclasses.replace(r.context, female_shots=12))
    text = release.v1_card(mixed)
    assert "Men's competitions only" not in text and "one value" not in text
    men = r.context.male_shots
    assert f"- Men's and women's competitions ({men:,} men's shots, 12 women's): " in text
    assert f"`gender` is not a feature; 12 of {men + 12:,} shots are women's." in text


def test_ratify_readme_na(v1r: dict[str, Any]) -> None:
    """Ratify 6: in the README block a candidate without a computed gain over v0 prints
    ``n/a`` in the gain column, the others their ``gain_cell``; v0's row ends
    ``n/a (reference)``."""
    legacy, release, _ = _mods()
    r = v1r["run"]
    gains = dict(r.v0_gains)
    rows = release.summary_block(r).splitlines()
    evs = (*r.baselines, *r.steps, r.gbm, r.ablation)
    assert sum(ev.spec.name not in gains for ev in evs) >= 2  # the n/a branch is exercised
    for ev in evs:
        title = f"**{ev.spec.title}**" if ev is r.head else ev.spec.title
        row = next(x for x in rows if x.startswith(f"| {title} |"))
        cell = cards.gain_cell(gains[ev.spec.name]) if ev.spec.name in gains else "n/a"
        assert row.endswith(f"| {cell} |"), row
    v0_row = next(x for x in rows if x.startswith(f"| {legacy.V0_TITLE} |"))
    assert v0_row.endswith("| n/a (reference) |")


def test_ratify_penalty_refusals(v1r: dict[str, Any]) -> None:
    """Ratify 7: ``v1._penalties`` refuses a season with no in-game kick and a v0
    ``penalty_xg`` outside (0, 1); with both rows and a valid constant it gives the run's
    losses."""
    _, _, v1 = _mods()
    r = v1r["run"]
    assert v1._penalties(r.penalties, r.v0) == r.penalty_losses
    for gone in ("In-game: dev", "In-game: hold-out"):
        kicks = tuple(row for row in r.penalties if row.label != gone)
        assert len(kicks) == len(r.penalties) - 1
        with pytest.raises(ValueError, match="no in-game penalty"):
            v1._penalties(kicks, r.v0)
    for q in (0.0, 1.0, -0.2, 1.5, math.nan):
        with pytest.raises(ValueError, match=r"\(0, 1\)"):
            v1._penalties(r.penalties, dataclasses.replace(r.v0, penalty_xg=q))


@pytest.mark.parametrize("warning", ["ConvergenceWarning", "LinAlgWarning"])
def test_ratify_refit_v0_warning(warning: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 8: ``refit_v0`` turns a ``ConvergenceWarning`` or a ``LinAlgWarning`` from the
    fit into a ``ValueError`` (fail closed), never a warning and a prediction."""
    import warnings

    from scipy.linalg import LinAlgWarning
    from sklearn.exceptions import ConvergenceWarning

    legacy, _, _ = _mods()
    cls = {"ConvergenceWarning": ConvergenceWarning, "LinAlgWarning": LinAlgWarning}[warning]
    fits: list[int] = []

    class Warns:
        def fit(self, x: Any, y: Any) -> Warns:
            fits.append(len(y))
            warnings.warn("fixture", cls, stacklevel=1)
            return self

        def predict_proba(self, x: Any) -> Any:
            return np.full((len(x), 2), 0.5)

    shots, _, matches = vc.v1_tables()
    kept = vc.kept_open_play(shots)
    test_ids = matches.filter(pl.col("season_name").is_in(vc.TEST_SEASONS))["match_id"]
    held = pl.col("match_id").is_in(test_ids.implode())
    monkeypatch.setattr(legacy.ablation, "make_lr", Warns)
    with pytest.raises(ValueError, match="the fit failed"):
        legacy.refit_v0(kept.filter(~held), kept.filter(held), ("big_chance",))
    assert len(fits) == 1  # the stub was the fit


def test_ratify_v0_boundary_null_date() -> None:
    """Ratify 9: test matches whose dates are all null (or none at all) are a ``ValueError``,
    not a ``TypeError`` from comparing a date with ``None``."""
    import datetime as dt

    _, _, v1 = _mods()
    model = vc.legacy_model(vc.V0_COEF)
    matches = pl.DataFrame(
        {"match_id": ["ws:a", "ws:b"], "match_date": [dt.date(2026, 8, 8), None]},
        schema={"match_id": pl.String(), "match_date": pl.Date()},
    )
    for ids in (["ws:b"], ["ws:none"]):
        ho = data.Holdout(pl.col("match_id").is_in(ids), "t")
        with pytest.raises(ValueError, match="no first date"):
            v1.check_v0_boundary(model, matches, ho)
    assert v1.check_v0_boundary(model, matches, data.Holdout(pl.col("match_id") == "ws:a",
                                                             "t")) is None  # fmt: skip


def test_ratify_gbm_edge_lines(v1r: dict[str, Any], v1r_gbm: dict[str, Any]) -> None:
    """Ratify 10: on a one-value leaf grid the leaf line says "smallest" only; the iteration
    line reads "The chosen iteration count (N) is the grid's largest (N)." when the count is
    the grid's largest, and is absent otherwise; a leaf count inside the grid gets no line."""
    from football_shot_models import gbm

    _, release, v1 = _mods()

    def edges(card: str) -> list[str]:
        return [x for x in _block(card, "Model").splitlines() if x.startswith("The chosen")]

    g = v1r_gbm["run"].gbm
    settings = {t: v for t, v, _ in g.coefficients}
    assert (settings["max_leaf_nodes"], settings["iterations"]) == (4, mc.TINY["max_iter"])
    assert mc.TINY["leaf_grid"] == (4,)
    assert edges(v1r_gbm["card"]) == [
        "The chosen leaf count (4) is the grid's smallest (4).",
        f"The chosen iteration count ({mc.TINY['max_iter']}) is the grid's largest "
        f"({mc.TINY['max_iter']}).",
    ]
    assert isinstance(g.spec, evaluation.ModelSpec)
    r = v1r["run"]
    for leaves, iters, expected in (
        (16, 10, ["The chosen leaf count (16) is the grid's largest (4, 8, 16)."]),
        (8, 10, []),
    ):
        make = functools.partial(gbm.MonotoneGBM, dict(v1.GBM_SIGNS), leaf_grid=(4, 8, 16),
                                 max_iter=20, step=10)  # fmt: skip
        coef = tuple(
            (t, {"max_leaf_nodes": leaves, "iterations": iters}.get(t, v), n)
            for t, v, n in g.coefficients
        )
        head = dataclasses.replace(g, spec=dataclasses.replace(g.spec, make=make),
                                   coefficients=coef)  # fmt: skip
        assert edges(release.v1_card(dataclasses.replace(r, head=head))) == expected


def test_ratify_cli_cache_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 12: ``--cache`` defaults to ``data/cache`` and ignores ``FSM_CACHE``;
    ``--no-cache`` passes ``None``; an explicit ``--cache`` is passed as given."""
    _, _, v1 = _mods()
    seen: list[Any] = []
    monkeypatch.setattr(v1, "run", lambda *a, **k: seen.append(k["cache"]))
    monkeypatch.setenv("FSM_CACHE", "/elsewhere/cache")
    for argv in ([], ["--no-cache"], ["--cache", "x/y"]):
        assert v1.main(argv) == 0
    assert seen == ["data/cache", None, "x/y"]


# ---- make reproduce: the private snapshot (slow, whoscored) -----------------------------------


@pytest.fixture(scope="module")
def real_v1(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One ``v1.run`` on the private snapshot with the prediction cache, as ``real_run``."""
    manifest = REPO / "manifests" / "whoscored-v1.json"
    v0 = REPO / "data" / "legacy" / "xg_model.json"
    if not ((REPO / "data" / "snapshots" / "whoscored-v1").is_dir() and manifest.is_file()
            and v0.is_file()):  # fmt: skip
        pytest.skip("needs the WhoScored snapshot and v0 (make snapshot-whoscored legacy-v0)")
    from football_shot_models import cache

    _, _, v1 = _mods()
    docs = tmp_path_factory.mktemp("real-v1") / "models"
    result = v1.run(manifest, v0=v0, docs_dir=docs,
                    cache=cache.cache_root(REPO / "data" / "cache"))  # fmt: skip
    xg = data.load_xg_data(manifest, builder="whoscored")
    return {"run": result, "docs": docs, "xg": xg}


def _readme_block() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- v1:summary:start -->", "<!-- v1:summary:end -->"
    assert text.count(start) == 1 and text.count(end) == 1
    return text.split(start)[1].split(end)[0].strip("\n")


@pytest.mark.slow
@pytest.mark.whoscored
def test_v1_card_reproduces(real_v1: dict[str, Any]) -> None:
    """§5, §6: the regenerated ``xg-v1`` card is byte-identical to the committed one."""
    committed = (REPO / "docs" / "models" / "xg-v1.md").read_bytes()
    assert (real_v1["docs"] / "xg-v1.md").read_bytes() == committed


@pytest.mark.slow
@pytest.mark.whoscored
def test_v1_readme_block(real_v1: dict[str, Any]) -> None:
    """§5: the README holds ``summary_block`` of the run between its markers, after the MS3
    block."""
    _, release, _ = _mods()
    assert _readme_block() == release.summary_block(real_v1["run"]).strip("\n")
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert text.index("<!-- v1:summary:start -->") > text.index("<!-- models:summary:end -->")


@pytest.mark.slow
@pytest.mark.whoscored
def test_v1_card_holds_no_ids(real_v1: dict[str, Any]) -> None:
    """§6 Rule 1: neither the card nor the README block holds a ``ws:`` id, a 7-digit token that
    is a snapshot match id, or a 5-6 digit token that is a snapshot player id; the
    by-competition table has exactly one row per test competition."""
    xg = real_v1["xg"]
    match_ids = {m.removeprefix("ws:") for m in xg.matches["match_id"]}
    players = {p.removeprefix("ws:") for p in pl.concat([xg.shots, xg.penalties],
                                                       how="diagonal")["player_id"]}  # fmt: skip
    card = (real_v1["docs"] / "xg-v1.md").read_text(encoding="utf-8")
    for text in (card, _readme_block()):
        assert re.search(r"\bws:", text) is None  # "Rows:" holds "ws:" without a boundary
        assert not set(re.findall(r"\b\d{7}\b", text)) & match_ids
        assert not set(re.findall(r"\b\d{5,6}\b", text)) & players
    ho = data.season_holdout(xg.matches, vc.TRAIN_SEASONS, vc.TEST_SEASONS)
    test_comps = xg.matches.filter(ho.expr)["competition_name"].n_unique()
    assert len(_rows(_block(card, "Test by competition"))) == test_comps
