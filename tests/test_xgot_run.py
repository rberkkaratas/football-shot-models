"""xGOT LR, GBM and penalty xGOT on both providers, the checks, the cards and Rule 1 (spec
``runs/MS5/xgot`` §6, §7, §9, §10; test plan 6 and 9; MS6 ``runs/MS6/xgot-ready`` §3, §4, §8).

The fast tests share one module-scoped ``xgot_cards.run`` per provider (MS6 §4: ``run`` = ``fit``
+ ``render`` + write) on the synthetic snapshots of ``xgot_cases`` (StatsBomb- and
WhoScored-shaped), with ``xgot.PROVIDERS`` pointed at them and a fixture-sized xG candidate set
and GBM grid swapped in through the module globals ``Provider.candidate_specs`` reads (the
StatsBomb candidates read ``open_goal``, so the keeper-neutral refit of PM X1 (b) runs). The
WhoScored fixture's ids are 6-digit sentinels that the Rule 1 test looks for in every output.

``make reproduce`` runs the slow tests at the end: ``pytest -m slow`` runs ``test_models_run``
and ``test_v1_run`` first, which warm the prediction cache with the xG candidates these runs read.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import re
from pathlib import Path
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xgot_cases as xg
from conftest import real_xgot
from scipy.special import expit

from football_shot_models import cards, data, evaluation, features, gbm, metrics, models, v1

REPO = Path(__file__).resolve().parents[1]
PROVIDERS = ("statsbomb", "whoscored")
XG_HEADLINE_CARD = {"statsbomb": "xg-gbm.md", "whoscored": "xg-v1.md"}
CARDS = {
    "statsbomb": ("xgot-lr", "xgot-gbm", "xgot-penalty"),
    "whoscored": ("ws-xgot-lr", "ws-xgot-gbm", "ws-xgot-penalty"),
}
CHECKS_TITLE = "Milestone checks (MS5, MS6)"
ROW_NAMES = (
    "cv auc", "holdout auc", "cv ece", "holdout ece", "cv slope", "holdout slope",
    "gbm headline", "recon shots", "recon goals", "xg input", "keeper neutral", "bands",
)  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from football_shot_models import xgot, xgot_cards

    root = tmp_path_factory.mktemp("xgot-run")
    manifests = {}
    for p in PROVIDERS:
        (root / p).mkdir()
        manifests[p] = xg.write_xgot_snapshot(root / p, p)
    out: dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, manifests, mp))
        for p in PROVIDERS:
            docs = root / p / "docs" / "models"
            docs.mkdir(parents=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = xgot_cards.run(p, docs_dir=docs, cache=None)
            out[p] = {"run": result, "docs": docs, "stdout": buf.getvalue(),
                      "xg": data.load_xg_data(manifests[p], builder=p)}  # fmt: skip
        out["block_ws"] = xgot_cards.summary_block([out["whoscored"]["run"]])
        out["block"] = xgot_cards.summary_block([out[p]["run"] for p in PROVIDERS])
    return out


def _population(p: str, xgd: Any) -> pl.DataFrame:
    return xgd.shots if p == "statsbomb" else v1.population(xgd)[0]


def _headers(card: str) -> list[str]:
    return [x for x in card.splitlines() if x.startswith("|")]


@pytest.mark.parametrize("p", PROVIDERS)
def test_run_writes_cards(runs: dict[str, Any], p: str) -> None:
    """Three cards per provider and nothing else in the docs dir; every card says xGOT in its
    tables, carries the ECE table and the reconciliation rows as its filters, never the xG
    validation paragraph; the cross-fitting sentence is on every open-play card and not on the
    penalty card, whose folds are its own (run5-ms5 B1, disputes.md); the checks only on the
    headline card; the xG input line on the LR and GBM cards; no "StatsBomb" in WhoScored cards.
    The xG input is keeper-neutral (X1 (b)); the open-play model sees no kick."""
    from football_shot_models import xgot

    r, docs = runs[p]["run"], runs[p]["docs"]
    names = CARDS[p]
    assert sorted(f.name for f in docs.iterdir()) == sorted(f"{n}.md" for n in names)
    assert set(r.cards) == set(names)
    prefix = xgot.PROVIDERS[p].prefix
    assert [e.spec.name for e in r.evaluations] == [
        f"{prefix}{n}" for n in ("base-rate", "xg-only", "lr-no-far", "lr", "gbm")
    ]
    lr, g = r.evaluations[3], r.evaluations[4]
    head_card = f"{prefix}{'gbm' if r.head is g else 'lr'}"
    xg_line = (f"xG input: {r.xg_input.spec.title}, cross-fitted; provider headline xG: "
               f"{r.xg_headline.spec.title}")  # fmt: skip
    text = xgot.PROVIDERS[p].text
    for name in names:
        card = (docs / f"{name}.md").read_text(encoding="utf-8")
        assert card == r.cards[name], name
        assert f"ECE ({metrics.ECE_BINS} equal-count bins)" in card, name
        assert "| Bin | Shots | Mean xGOT | Goal rate |" in card, name
        assert [x for x in _headers(card) if re.search(r"(Σ|Mean) xG(?!OT)", x)] == [], name
        assert (xgot.XGOT_CROSS_FIT in card) == (not name.endswith("penalty")), name
        assert cards.VALIDATION not in card, name
        assert (CHECKS_TITLE in card) == (name == head_card), name
        assert (xg_line in card) == (not name.endswith("penalty")), name
        for reason in xgot.RECONCILE_REASONS:
            assert f"| {text.label(reason)} |" in card, (name, reason)
        if p == "whoscored":
            assert "StatsBomb" not in card, name
    assert len(r.checks) == 12
    if p == "statsbomb":
        assert r.xg_input.spec.name == f"{r.xg_headline.spec.name}-keeper-neutral"
        assert "open_goal" in r.xg_headline.spec.features
    else:
        assert r.xg_input.spec is r.xg_headline.spec
    assert not set(r.xg_input.spec.features) & features.KEEPER_POSITION
    for ev in (lr, g):
        assert ev.spec.xg_input == r.xg_input.spec.name
    pop = _population(p, runs[p]["xg"])
    kinds = r.predictions.group_by("kind").len().rows()
    kicks = runs[p]["xg"].penalties.filter("on_target")
    assert dict((str(k), n) for k, n in kinds) == {
        "open_play": pop.height,
        "penalty": kicks.filter(pl.col("reason") == "penalty").height,
        "shootout": kicks.filter(pl.col("reason") == "shootout").height,
    }
    assert lr.cv.n + lr.holdout.n == pop.filter("on_target").height  # no kick in the open play
    assert r.recon.unscored == 0
    for ev in (lr, g):
        assert ev.spec.title in runs["block"], ev.spec.name


def _patch_ece(mp: pytest.MonkeyPatch, value: float) -> list[int]:
    """Every ECE the checks compute is ``value`` (to test the bound at its exact value)."""
    from football_shot_models import xgot

    calls: list[int] = []

    def fake(y: Any, p: Any, bins: int = metrics.ECE_BINS) -> float:
        calls.append(1)
        return value

    mp.setattr(metrics, "ece", fake)
    if hasattr(xgot, "ece"):
        mp.setattr(xgot, "ece", fake)
    return calls


CHECK_CASES: dict[str, tuple[dict[str, Any], float, dict[str, str]]] = {
    "all met": ({}, 0.01, {}),
    "auc at 0.80": ({"cv_auc": 0.80, "ho_auc": 0.80}, 0.01, {}),
    "cv auc 0.7999": ({"cv_auc": 0.7999}, 0.01, {"cv auc": "missed"}),
    "hold-out auc 0.7999": ({"ho_auc": 0.7999}, 0.01, {"holdout auc": "missed"}),
    "ece at 0.02 (strict)": ({}, 0.02, {"cv ece": "missed", "holdout ece": "missed"}),
    "ece 0.0199": ({}, 0.0199, {}),
    "slopes at 0.9": ({"cv_slope": 0.9, "ho_slope": 0.9}, 0.01, {}),
    "slopes at 1.1": ({"cv_slope": 1.1, "ho_slope": 1.1}, 0.01, {}),
    "cv slope 0.8999": ({"cv_slope": 0.8999}, 0.01, {"cv slope": "missed"}),
    "hold-out slope 1.1001": ({"ho_slope": 1.1001}, 0.01, {"holdout slope": "missed"}),
    "no slope": ({"cv_slope": None, "ho_slope": None}, 0.01,
                 {"cv slope": "n/a", "holdout slope": "n/a"}),
    "gbm head, low > 0": ({"gbm_head": True, "low": 0.001}, 0.01, {}),
    "gbm head, low == 0": ({"gbm_head": True, "low": 0.0}, 0.01, {"gbm headline": "missed"}),
    "lr head, low == 0": ({"gbm_head": False, "low": 0.0}, 0.01, {}),
    "lr head, low > 0": ({"gbm_head": False, "low": 0.001}, 0.01, {"gbm headline": "missed"}),
}  # fmt: skip


@pytest.mark.parametrize("case", list(CHECK_CASES))
def test_checks_rows(case: str) -> None:
    """Rows 1-9 are met exactly when their sentence holds, each at its exact bound: AUC >= 0.80
    ("about 0.80 or better" read strictly), ECE < 0.02 (strict) on the CV fold mean and the
    hold-out, slope in [0.9, 1.1] (None -> n/a), the GBM headline iff its CV gain CI lies above
    0 (``low == 0`` keeps the LR); every other row stays met."""
    kw, ece, changed = CHECK_CASES[case]
    with pytest.MonkeyPatch.context() as mp:
        calls = _patch_ece(mp, ece)
        rows = xg.call_checks(xg.check_inputs(**kw))
    assert calls, "the checks never read metrics.ece"
    want = {name: changed.get(name, "met") for name in ROW_NAMES}
    assert dict(zip(ROW_NAMES, (c.result for c in rows), strict=True)) == want
    from football_shot_models import xgot

    assert (xgot.AUC_MIN, xgot.ECE_MAX, xgot.RECONCILE_TOL) == (0.80, 0.02, 0.01)
    assert models.SLOPE_RANGE == (0.9, 1.1)


def test_checks_ece_row_reads_the_fold_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """PM X3: the CV ECE row is judged on the fold mean, never on the pooled out-of-fold value
    (≈ 0 by construction for a Platt-calibrated fit). Folds at 0.03 and pooled at 0.005 miss
    row 3; the hold-out at 0.01 still meets row 4."""
    from football_shot_models import xgot

    inputs = xg.check_inputs()
    n_dev, n_held = inputs["dev"].height, inputs["held"].height
    fold_n = {t.size for t in inputs["head"].fold_test}
    assert len(fold_n | {n_dev, n_held}) == 3  # precondition: sizes tell the calls apart
    by_size = {**dict.fromkeys(fold_n, 0.03), n_dev: 0.005, n_held: 0.01}

    def fake(y: Any, p: Any, bins: int = metrics.ECE_BINS) -> float:
        return by_size[len(y)]

    monkeypatch.setattr(metrics, "ece", fake)
    s = xgot.ece_stats(inputs["head"], evaluation.labels(inputs["dev"]),
                       evaluation.labels(inputs["held"]))  # fmt: skip
    assert (s.mean, s.pooled, s.held) == (0.03, 0.005, 0.01)
    rows = dict(zip(ROW_NAMES, (c.result for c in xg.call_checks(inputs)), strict=True))
    assert (rows["cv ece"], rows["holdout ece"]) == ("missed", "met")


@pytest.mark.parametrize(
    "case", ["keeper feature in the headline", "keeper feature in the xG input", "ablation input"]
)
def test_checks_guard_rows(case: str) -> None:
    """Rows 10-11 are guard-backed: a keeper-position feature in the headline xGOT or in the xG
    input, or an ablation-only xG input, raises rather than printing a missed row; with clean
    inputs both rows are met and row 10 names the xG input."""
    rows = xg.call_checks(xg.check_inputs())
    assert [rows[i].result for i in (9, 10)] == ["met", "met"]
    assert "xG input" in rows[9].target + rows[9].measured
    kw: dict[str, Any] = {
        "keeper feature in the headline": {"head_features": (*xg.LR_FEATURES, "gk_x")},
        "keeper feature in the xG input": {"xg_features": ("distance_m", "open_goal")},
        "ablation input": {"xg_features": ("distance_m", "big_chance")},
    }[case]
    with pytest.raises(ValueError):
        xg.call_checks(xg.check_inputs(**kw))


def test_headline_rule(runs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The xGOT headline is the GBM iff its CV gain over the LR has ``low > 0`` (both runs), and
    ``low == 0`` exactly keeps the LR."""
    for p in PROVIDERS:
        r = runs[p]["run"]
        lr, g = r.evaluations[3], r.evaluations[4]
        assert r.head is (g if r.gbm_gain.cv.low > 0 else lr), p
        assert r.gbm_gain.label == f"over {lr.spec.title}"
    inputs = xg.check_inputs()
    head = inputs["head"]
    g = dataclasses.replace(head, spec=xg.gbm_spec())
    for low, want in ((0.0, head), (1e-12, g), (-0.001, head)):
        gain = evaluation.Comparison("over xGOT LR", mc.gain(0.001, low, 0.002),
                                     mc.gain(0.0, -0.01, 0.01))  # fmt: skip
        monkeypatch.setattr(models, "compare", lambda *a, _g=gain, **k: _g)
        assert models.headline(head, g, inputs["dev"], inputs["held"])[0] is want, low


def test_penalty_placement_only(runs: dict[str, Any]) -> None:
    """The penalty xGOT reads placement only (a subset of ``PLACEMENT``, no xG input); its card
    prints the CV and hold-out calibration slope and the ECE table, and no xG input line."""
    from football_shot_models import xgot

    pen = next(s for s in xgot.xgot_specs("xgot-", "xg-x") if s.name == "xgot-penalty")
    assert set(pen.features) <= set(xg.placement_names())
    assert features.XG_INPUT not in pen.features and pen.xg_input is None
    for p in PROVIDERS:
        r = runs[p]["run"]
        assert set(r.penalty.spec.features) == set(xg.PENALTY_FEATURES)
        card = r.cards[CARDS[p][2]]
        for value in (r.penalty.cv.cal_slope, r.penalty.holdout.cal_slope):
            assert cards.fmt(value) in card, p
        assert f"ECE ({metrics.ECE_BINS} equal-count bins)" in card
        assert "xG input:" not in card


@pytest.mark.parametrize("p", PROVIDERS)
def test_penalty_shootouts_scored_by_dev_fit(runs: dict[str, Any], p: str) -> None:
    """The penalty xGOT is fitted on in-game dev kicks on target only; the shoot-out kicks on
    target (all seasons) are scored by a second fit on those kicks and reported apart (X2)."""
    from football_shot_models import xgot

    r, xgd = runs[p]["run"], runs[p]["xg"]
    kicks = xgd.penalties.filter("on_target")
    ingame = kicks.filter(pl.col("reason") == "penalty")
    held_mask = data.holdout_mask(ingame, xgot.PROVIDERS[p].holdout(xgd.matches), "kicks")
    dev_k, held_k = ingame.filter(~held_mask), ingame.filter(held_mask)
    shoot = kicks.filter(pl.col("reason") == "shootout")
    assert shoot.height == 4 and xgd.penalties.filter(pl.col("reason") == "shootout").height == 6
    assert (r.penalty.cv.n, r.penalty.holdout.n) == (dev_k.height, held_k.height)
    p_so = evaluation.fit_predict(r.penalty.spec, dev_k, shoot)
    s = metrics.score(evaluation.labels(shoot), p_so)
    assert (r.shootouts.n, r.shootouts.goals) == (s.n, s.goals)
    assert r.shootouts.xg == pytest.approx(s.xg, abs=1e-9)
    assert r.shootouts.log_loss == pytest.approx(s.log_loss, abs=1e-9)
    pred = r.predictions.filter(pl.col("kind") == "shootout").sort("shot_id")
    assert pred["shot_id"].to_list() == shoot["shot_id"].sort().to_list()
    want = dict(zip(shoot["shot_id"].to_list(), p_so.tolist(), strict=True))
    assert pred["xgot"].to_list() == pytest.approx([want[s] for s in pred["shot_id"]], abs=1e-9)
    pens = r.predictions.filter(pl.col("kind") == "penalty")
    oof = dict(zip(dev_k["shot_id"].to_list(), r.penalty.oof.tolist(), strict=True))
    got = {k: v for k, v in pens.select("shot_id", "xgot").iter_rows() if k in oof}
    assert got == pytest.approx(oof, abs=1e-12)


def test_gbm_placement_free() -> None:
    """MS6 §3 (candidate (b), replaces the MS5 monotone-placement test): the GBM reads the xG
    input, ``mouth_y_abs``, ``mouth_out``, ``mouth_z``, ``far_side`` and ``angle_rad``, and its
    sign map constrains the xG input only (placement is free). The card's reasons and note
    follow: one reason, for the xG input; the note states no number. The LR keeps its features
    (``mouth_y_past`` included) and the bands are not re-cut."""
    from football_shot_models import xgot, xgot_cards

    spec = next(s for s in xgot.xgot_specs("xgot-", "xg-x") if s.name == "xgot-gbm")
    assert spec.features == xg.GBM_FEATURES
    assert xgot.GBM_FEATURES == xg.GBM_FEATURES
    assert dict(xgot.GBM_SIGNS) == xg.GBM_SIGNS == {features.XG_INPUT: 1}
    assert spec.make().get_params()["monotone"] == xg.GBM_SIGNS
    assert "mouth_y_in" not in spec.features and "mouth_y_past" not in spec.features
    assert set(xgot_cards.XGOT_MONOTONE_REASONS) == {features.XG_INPUT}
    assert re.search(r"\d", xgot_cards.XGOT_MONOTONE_NOTE) is None
    assert "band" in xgot_cards.XGOT_MONOTONE_NOTE
    lr = next(s for s in xgot.xgot_specs("xgot-", "xg-x") if s.name == "xgot-lr")
    assert lr.features == xg.LR_FEATURES
    assert xgot.Y_BANDS == (0.0, 1.0, 2.0, 3.0, 3.3, features.INNER_Y) and features.INNER_Y == 3.55


def test_gbm_beyond_post_not_top() -> None:
    """run5-ms5 (FDS M1, QA Q1): a ball beyond ``features.INNER_Y`` touches the post, so the
    on-target shots there are mostly saves, and a StatsBomb save stopped just in front of the
    shot projects far outside the frame. A fit of the ``gbm`` spec on data where that holds
    scores neither a ball at 3.8 m nor that projected save above a ball at 3.4 m."""
    from football_core import pitch as fc_pitch

    from football_shot_models import xgot

    projected = float(fc_pitch.project_to_goal_line([90.0], [34.0], [90.5], [36.0])[0])
    assert projected == 68.0  # precondition: the save lands on the touchline, not near the post
    assert features.clipped([fc_pitch.GOAL_CENTRE_Y - projected], [1.0]).all()
    spec = next(s for s in xgot.xgot_specs("xgot-", "xg-x") if s.name == "xgot-gbm")
    signs = spec.make().get_params()["monotone"]
    tiny = dataclasses.replace(spec, make=functools.partial(gbm.MonotoneGBM, signs, **mc.SIGNAL))
    rng = np.random.default_rng(29)
    n = 1200
    out_ = rng.uniform(size=n) < 0.15
    gy = np.where(out_, rng.uniform(4.0, 20.0, n), rng.uniform(0.0, 3.8, n))
    gy *= np.where(rng.uniform(size=n) < 0.5, -1.0, 1.0)
    gz, xgl = rng.uniform(0.05, 2.35, n), rng.normal(-1.5, 0.8, n)
    inner = np.minimum(np.abs(gy), features.INNER_Y)
    p = expit(0.9 * xgl + 0.9 + 0.9 * (inner - 1.8) + 0.4 * (gz - 1.2))
    p = np.where(np.abs(gy) > features.INNER_Y, 0.08, p)  # beyond the post: saves
    train = pl.DataFrame({
        "shot_id": [f"sb:s{i:04d}" for i in range(n)],
        "match_id": [f"sb:m{i // 10:03d}" for i in range(n)],
        "is_goal": rng.uniform(size=n) < p,
        "on_target": [True] * n,
        "goal_y": 34.0 - gy, "goal_z": gz, "y": rng.uniform(15.0, 53.0, n),
        "angle_rad": rng.uniform(0.2, 1.2, n), "xg_logit": xgl,
    })  # fmt: skip
    grid = pl.DataFrame({
        "shot_id": ["sb:g0", "sb:g1", "sb:g2"],
        "match_id": ["sb:g"] * 3,
        "is_goal": [False] * 3,
        "on_target": [True] * 3,
        "goal_y": [34.0 - 3.4, 34.0 - 3.8, projected], "goal_z": [1.0] * 3, "y": [20.0] * 3,
        "angle_rad": [0.6] * 3, "xg_logit": [-1.0] * 3,
    })  # fmt: skip
    top, post, saved = evaluation.fit_predict(tiny, train, grid)
    assert post < top and saved < top, (top, post, saved)


def test_band_lines_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """The calibration-by-placement table: shots, goals, goal rate, mean xGOT, Σ xGOT − goals,
    and (MS6 §3.1) the SD √Σp(1−p) and ``Within 2 SD`` per ``Y_BANDS`` band of rounded |y|,
    points outside the frame in their own row; worked out by hand from a frame with two shots
    per band (the outside row misses: |−1.6| > 2 · 0.548). An emptied band prints 0 shots, no
    rate or mean, SD 0.0 and ``yes`` (0 ≤ 0). Every band is judged here (minimum 0); at the
    PM's 100-shot minimum (MS6 gsax) every 2-shot band prints ``not judged (n < 100)``."""
    from football_shot_models import xgot, xgot_cards

    ay = [0.5, 0.6, 1.5, 1.4, 2.5, 2.6, 3.1, 3.2, 3.4, 3.5, 3.7, 3.8, 5.0, 9.0]
    sign = [1, -1] * 7
    goal = [True, False, False, False, True, True, False, True, True, False, False, False,
            True, True]  # fmt: skip
    p = np.array([0.2, 0.3, 0.4, 0.1, 0.5, 0.7, 0.6, 0.4, 0.5, 0.5, 0.6, 0.4, 0.3, 0.1])
    dev = pl.DataFrame({
        "shot_id": [f"sb:b{i:02d}" for i in range(14)],
        "goal_y": [34.0 - s * a for s, a in zip(sign, ay, strict=True)],
        "goal_z": [1.0] * 14, "on_target": [True] * 14, "is_goal": goal,
    })  # fmt: skip
    ev = mc.fake_eval(xg.lr_spec(), oof=p)
    small = [x for x in xgot_cards.band_lines(ev, dev) if x.startswith("| ") and "abs" not in x]
    assert [x.rstrip(" |").rsplit("| ", 1)[-1] for x in small] == ["not judged (n < 100)"] * 7
    monkeypatch.setattr(metrics, "MIN_JUDGED_SHOTS", 0)
    lines = xgot_cards.band_lines(ev, dev)
    rows = [x for x in lines if x.startswith("| ") and not x.startswith("| abs")]
    assert xgot.Y_BANDS == (0.0, 1.0, 2.0, 3.0, 3.3, features.INNER_Y)
    assert metrics.BAND_SDS == 2.0
    want = [  # band, shots, goals, rate, mean, Σ − goals, SD, within: pairs of rows i, i + 1
        ("[0, 1)", 2, 1, 0.5, 0.25, -0.5, "0.6", "yes"),  # √(0.16 + 0.21)
        ("[1, 2)", 2, 0, 0.0, 0.25, 0.5, "0.6", "yes"),  # √(0.24 + 0.09)
        ("[2, 3)", 2, 2, 1.0, 0.6, -0.8, "0.7", "yes"),  # √(0.25 + 0.21)
        ("[3, 3.3)", 2, 1, 0.5, 0.5, 0.0, "0.7", "yes"),  # √(0.24 + 0.24)
        ("[3.3, 3.55)", 2, 1, 0.5, 0.5, 0.0, "0.7", "yes"),  # √0.5
        (">= 3.55", 2, 0, 0.0, 0.5, 1.0, "0.7", "yes"),  # 1.0 ≤ 2 · 0.693
        ("outside the frame + 0.15 m", 2, 2, 1.0, 0.2, -1.6, "0.5", "no"),  # √(0.21 + 0.09)
    ]  # fmt: skip
    assert len(rows) == len(want)
    for row, (band, n, g, rate, mean, gap, sd, within) in zip(rows, want, strict=True):
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells == [band, str(n), str(g), cards.fmt(rate), cards.fmt(mean), f"{gap:+.1f}",
                         sd, within]  # fmt: skip
    header = next(x for x in lines if x.startswith("| abs"))
    assert [c.strip() for c in header.strip("|").split("|")][-2:] == ["SD", "Within 2 SD"]
    keep = [i for i in range(14) if i not in (6, 7)]  # empty the [3, 3.3) band
    empty = xgot_cards.band_lines(mc.fake_eval(xg.lr_spec(), oof=p[keep]), dev[keep])
    row = next(x for x in empty if x.startswith("| [3, 3.3) |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:3] == ["0", "0"] and cells[-2:] == ["0.0", "yes"]
    assert cells[3] == cells[4] and cells[3] in ("–", cards.fmt(None))


def test_level_and_penalty_calibration_lines(runs: dict[str, Any]) -> None:
    """The open-play cards state the scored-once split's Σ xGOT against goals (FDS m4); the
    penalty card judges each open-play bound on its own and prints the ranking AUC beside the
    advice when one is outside (FDS m1). Unit case: both slopes inside, ECE outside."""
    from football_shot_models import xgot, xgot_cards

    for p in PROVIDERS:
        r = runs[p]["run"]
        split = xgot.PROVIDERS[p].split
        for ev, name in ((r.evaluations[3], CARDS[p][0]), (r.evaluations[4], CARDS[p][1])):
            h = ev.holdout
            assert f"On the {split}, Σ xGOT is {h.xg:,.1f} against {h.goals:,} goals" in (
                r.cards[name]), name  # fmt: skip
        pen_card = r.cards[CARDS[p][2]]
        assert f"CV calibration slope {cards.fmt(r.penalty.cv.cal_slope)} (" in pen_card
    pen = mc.fake_eval(
        xg.lr_spec(name="xgot-penalty"), cv=mc.scores(auc=0.64, slope=0.986),
        holdout=mc.scores(auc=0.566, slope=0.9168),
    )  # fmt: skip
    s = xgot.EceStats(mean=0.0892, max=0.1, pooled=0.04, held=0.12, floor=0.1)
    line = xgot_cards._penalty_calibration(pen, s, "hold-out")
    assert "slope 0.9860 (inside [0.9, 1.1])" in line
    assert "hold-out slope 0.9168 (inside)" in line
    assert "CV ECE 0.0892 (fold mean; not below 0.02)" in line
    assert "AUC is 0.6400 on CV and 0.5660 on the hold-out" in line
    ok = xgot_cards._penalty_calibration(pen, dataclasses.replace(s, mean=0.0199), "hold-out")
    assert "AUC" not in ok and "(fold mean; below 0.02)" in ok


@pytest.mark.parametrize("p", PROVIDERS)
def test_run_shootout_split_and_penalty_benchmarks(runs: dict[str, Any], p: str) -> None:
    """Gate ratify 10: a shoot-out kick in a hold-out competition-season is ``holdout``, one in a
    dev pair ``dev`` (hold-out matches taken from the fixture's matches table here). Ratify 9:
    the penalty card's Benchmarks lists the penalty base rate first, then the penalty xGOT."""
    r, xgd = runs[p]["run"], runs[p]["xg"]
    m = xgd.matches
    held_matches = set(
        m.filter((pl.col("competition_id") == "sb:2") & (pl.col("season_id") == "sb:27"))[
            "match_id"] if p == "statsbomb" else m.filter(pl.col("season_name") == "2026/2027")[
            "match_id"]
    )  # fmt: skip
    so = r.predictions.filter(pl.col("kind") == "shootout")
    got = {(row["match_id"] in held_matches, str(row["split"])) for row in so.iter_rows(
        named=True)}  # fmt: skip
    assert got == {(True, "holdout"), (False, "dev")}  # precondition: both kinds present
    card = r.cards[CARDS[p][2]]
    bench = card.split("## Benchmarks", 1)[1].split("## ", 1)[0]
    rows = [x for x in bench.splitlines() if x.startswith("| ") and not x.startswith("| Model")]
    assert rows[0].startswith("| Penalty base rate (on target) |"), rows
    assert rows[1].startswith(f"| {r.penalty.spec.title} |"), rows


@pytest.mark.parametrize("p", PROVIDERS)
def test_split_half_blocked_left_out(runs: dict[str, Any], p: str) -> None:
    """The split-half rows with blocked shots "left out" equal ``metrics.split_half`` on the
    per-shot frame without blocked shots, computed here, and differ from the "in" rows."""
    from football_shot_models import xgot

    r = runs[p]["run"]
    per_shot = xgot.sga(r.predictions)
    assert per_shot.filter(pl.col("outcome") == "blocked").height > 0  # precondition
    kept = per_shot.filter(pl.col("outcome") != "blocked")
    rows = {(m, blocked): (a, b) for m, blocked, a, b in r.split_half}
    for m in xgot.MIN_SHOTS:
        want = tuple(
            metrics.split_half(kept, by="player_id", value=v, min_rows=m, unit="match")
            for v in ("sga", "gax")
        )
        for got, exp in zip(rows[(m, False)], want, strict=True):
            assert got.keys == exp.keys, m
            assert (got.r, got.low, got.high) == pytest.approx((exp.r, exp.low, exp.high)), m
    in_, out = rows[(xgot.MIN_SHOTS[0], True)], rows[(xgot.MIN_SHOTS[0], False)]
    assert in_[0].r is not None and in_[0] != out[0]
    assert f"| {xgot.MIN_SHOTS[0]} | left out | {out[0].keys:,} |" in r.cards[r.head.spec.name]


def test_candidates_read_module_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """MS6 §5 (rewritten from run5-ms5 B2): ``Provider.candidate_specs`` returns the xG run's
    step specs and GBM, read from the module globals when called (StatsBomb ``ablation.STEPS``
    and ``models.GBM``, WhoScored ``v1.STEPS`` and ``v1.GBM``), so a patch is seen; the old
    ``candidates`` field and private helpers are gone."""
    from football_shot_models import ablation, xgot

    sb_steps, sb_gbm, ws_steps, ws_gbm = (object(),), object(), (object(),), object()
    monkeypatch.setattr(ablation, "STEPS", sb_steps)
    monkeypatch.setattr(models, "GBM", sb_gbm)
    monkeypatch.setattr(v1, "STEPS", ws_steps)
    monkeypatch.setattr(v1, "GBM", ws_gbm)
    sb = xgot.PROVIDERS["statsbomb"].candidate_specs()
    ws = xgot.PROVIDERS["whoscored"].candidate_specs()
    assert sb[0] is sb_steps and sb[1] is sb_gbm
    assert ws[0] is ws_steps and ws[1] is ws_gbm
    assert not hasattr(xgot.PROVIDERS["statsbomb"], "candidates")
    for name in ("_candidates", "_sb_candidates", "_ws_candidates"):
        assert not hasattr(xgot, name), name


def test_rule1_ws_outputs(runs: dict[str, Any]) -> None:
    """Rule 1: no WhoScored match, team, player or keeper id (prefixed or bare) appears in a
    WhoScored card, the README block or stdout; only the cards are written."""
    sentinels = xg.ws_sentinels()
    assert len(sentinels) > 50  # precondition: the fixture's ids
    ids = runs["whoscored"]["xg"].shots.select("match_id", "team_id", "player_id", "gk_id")
    used = {v.removeprefix("ws:") for c in ids.columns for v in ids[c].drop_nulls().to_list()}
    assert used <= sentinels and len(used) > 50
    texts = {
        **{n: runs["whoscored"]["run"].cards[n] for n in CARDS["whoscored"]},
        "block": runs["block_ws"],
        "stdout": runs["whoscored"]["stdout"],
    }
    for where, text in texts.items():
        assert re.search(r"ws:\d", text) is None, where
        assert not set(re.findall(r"\b\d{6}\b", text)) & sentinels, where
    assert sorted(f.suffix for f in runs["whoscored"]["docs"].iterdir()) == [".md"] * 3


def test_ece_lines_values() -> None:
    """The ECE table on every card: CV fold mean, CV max, CV pooled, hold-out and its floor, each
    recomputed here from the folds; one table row and the binning sentence."""
    from football_shot_models import xgot_cards

    rng = np.random.default_rng(23)
    n, nh = 60, 30
    y, yh = np.arange(n) % 3 == 0, np.arange(nh) % 3 == 0  # both classes in every fold
    dev = pl.DataFrame({"match_id": [f"sb:m{i // 4:02d}" for i in range(n)], "is_goal": y})
    held = pl.DataFrame({"match_id": [f"sb:n{i // 3:02d}" for i in range(nh)], "is_goal": yh})
    folds = xg.folds_from_tests(n, [list(range(k, n, 5)) for k in range(5)])
    oof, ph = rng.uniform(0.05, 0.8, n), rng.uniform(0.05, 0.8, nh)
    spec = evaluation.ModelSpec("xgot-lr", "xGOT LR", xg.LR_FEATURES, evaluation.BaseRate)
    ev = xg.make_eval(spec, dev, held, folds, oof, ph)
    per_fold = [metrics.ece(y[t], oof[t]) for _, t in folds]
    values = (np.mean(per_fold), max(per_fold), metrics.ece(y, oof), metrics.ece(yh, ph),
              metrics.ece_floor(ph))  # fmt: skip
    lines = xgot_cards.ece_lines(ev, y, yh)
    text = "\n".join(lines)
    for value in values:
        assert cards.fmt(float(value)) in text, value
    assert len([x for x in lines if x.startswith("|")]) == 3  # header, rule, one row
    assert f"{metrics.ECE_BINS} equal-count bins" in text


def test_main_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` exits 1 with ``fsm: error:`` and no traceback on a missing manifest."""
    from football_shot_models import xgot, xgot_cards

    monkeypatch.setenv("FSM_CACHE", "off")
    missing = {p: tmp_path / "manifests" / f"{p}.json" for p in PROVIDERS}
    monkeypatch.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, missing, monkeypatch))
    assert xgot_cards.main(["--provider", "statsbomb"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("fsm: error: ") and "Traceback" not in err


# ---- real snapshots (make reproduce) ----------------------------------------------------------


def _readme_block() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- xgot:summary:start -->", "<!-- xgot:summary:end -->"
    assert text.count(start) == 1 and text.count(end) == 1
    return text.split(start)[1].split(end)[0].strip("\n")


def _sb(real: dict[str, Any]) -> dict[str, Any]:
    return real_xgot(real, "statsbomb")  # the session fixture ``real`` lives in conftest


def _ws(real: dict[str, Any]) -> dict[str, Any]:
    return real_xgot(real, "whoscored")


def _ms6_checks(r: Any) -> None:
    """MS6 package 1: every check row is met (the MS5 rows and row 12, the band rule on the
    headline), the headline is the GBM and its CV gain over the LR has an interval above 0."""
    assert len(r.checks) == 12
    assert [c.result for c in r.checks] == ["met"] * 12, [
        (c.target, c.measured) for c in r.checks if c.result != "met"
    ]
    assert models.is_gbm(r.head) and r.gbm_gain.cv.low > 0


@pytest.mark.slow
def test_reproduce_statsbomb(real: dict[str, Any]) -> None:
    """The three StatsBomb xGOT cards come back byte for byte; the reconciliation is within 1 %
    and every on-target shot of the population is scored."""
    sb = _sb(real)
    for name in CARDS["statsbomb"]:
        committed = (REPO / "docs" / "models" / f"{name}.md").read_bytes()
        assert (sb["docs"] / f"{name}.md").read_bytes() == committed, name
    r = sb["run"]
    assert [r.checks[i].result for i in (7, 8)] == ["met", "met"]
    assert r.recon.unscored == 0
    _ms6_checks(r)


@pytest.mark.slow
@pytest.mark.whoscored
def test_reproduce_whoscored(real: dict[str, Any]) -> None:
    """The three WhoScored xGOT cards and the README block (both providers) come back byte for
    byte; no ``ws:`` id is in them."""
    from football_shot_models import xgot_cards

    sb = _sb(real)
    ws = _ws(real)
    for name in CARDS["whoscored"]:
        committed = (REPO / "docs" / "models" / f"{name}.md").read_bytes()
        assert (ws["docs"] / f"{name}.md").read_bytes() == committed, name
    block = xgot_cards.summary_block([sb["run"], ws["run"]])
    assert _readme_block() == block.strip("\n"), (
        "the README MS5 block is stale: `make xgot` prints it; paste it between the "
        "xgot:summary markers"
    )
    for text in (block, *(ws["run"].cards[n] for n in CARDS["whoscored"])):
        assert re.search(r"ws:\d", text) is None
    r = ws["run"]
    assert [r.checks[i].result for i in (7, 8)] == ["met", "met"]
    assert r.recon.unscored == 0
    _ms6_checks(r)


@pytest.mark.slow
@pytest.mark.parametrize("p", [pytest.param("statsbomb"),
                               pytest.param("whoscored", marks=pytest.mark.whoscored)])  # fmt: skip
def test_reproduce_xg_headline_on_card(real: dict[str, Any], p: str) -> None:
    """MS6 §5, §8: the xGOT run's provider headline xG is the committed xG card's headline (its
    title, and its CV log loss on that card's row), and the committed headline xGOT card names
    the run's xG input. The run is ``fit`` + render (``test_run_is_fit_render_write``), so its
    ``xg_headline`` and ``xg_input`` are ``fit(p)``'s."""
    r = (_sb(real) if p == "statsbomb" else _ws(real))["run"]
    title = r.xg_headline.spec.title
    card = (REPO / "docs" / "models" / XG_HEADLINE_CARD[p]).read_text(encoding="utf-8")
    if p == "statsbomb":
        assert card.startswith(f"# Model card: {title}\n") and "This is the headline model" in card
    else:
        assert f"Headline model: {title}." in card
    row = next(x for x in card.splitlines() if x.startswith((f"| {title} |", f"| **{title}** |")))
    assert row.split("|")[2].strip() == cards.fmt(r.xg_headline.cv.log_loss), row
    xgot_card = (REPO / "docs" / "models" / f"{r.head.spec.name}.md").read_text(encoding="utf-8")
    assert f"xG input: {r.xg_input.spec.title}, cross-fitted; provider headline xG: {title}." in (
        xgot_card)  # fmt: skip
