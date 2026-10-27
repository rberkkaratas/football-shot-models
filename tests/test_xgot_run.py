"""xGOT LR, GBM and penalty xGOT on both providers, the checks, the cards and Rule 1 (spec
``runs/MS5/xgot`` §6, §7, §9, §10; test plan 6 and 9).

The fast tests share one module-scoped ``xgot.run`` per provider on the synthetic snapshots of
``xgot_cases`` (StatsBomb- and WhoScored-shaped), with ``xgot.PROVIDERS`` pointed at them and a
fixture-sized xG candidate set and GBM grid swapped in (the StatsBomb candidates read
``open_goal``, so the keeper-neutral refit of PM X1 (b) runs). The WhoScored fixture's ids are
6-digit sentinels that the Rule 1 test looks for in every output.

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
from scipy.special import expit

from football_shot_models import cards, data, evaluation, features, gbm, metrics, models, v1

REPO = Path(__file__).resolve().parents[1]
PROVIDERS = ("statsbomb", "whoscored")
CARDS = {
    "statsbomb": ("xgot-lr", "xgot-gbm", "xgot-penalty"),
    "whoscored": ("ws-xgot-lr", "ws-xgot-gbm", "ws-xgot-penalty"),
}
CHECKS_TITLE = "Milestone checks (MS5)"
ROW_NAMES = (
    "cv auc", "holdout auc", "cv ece", "holdout ece", "cv slope", "holdout slope",
    "gbm headline", "recon shots", "recon goals", "xg input", "keeper neutral",
)  # fmt: skip


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from football_shot_models import xgot

    root = tmp_path_factory.mktemp("xgot-run")
    manifests = {}
    for p in PROVIDERS:
        (root / p).mkdir()
        manifests[p] = xg.write_xgot_snapshot(root / p, p)
    out: dict[str, Any] = {}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, manifests))
        mp.setattr(xgot, "xgot_specs", xg.tiny_specs(xgot.xgot_specs))
        for p in PROVIDERS:
            docs = root / p / "docs" / "models"
            docs.mkdir(parents=True)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = xgot.run(p, docs_dir=docs, cache=None)
            out[p] = {"run": result, "docs": docs, "stdout": buf.getvalue(),
                      "xg": data.load_xg_data(manifests[p], builder=p)}  # fmt: skip
        out["block_ws"] = xgot.summary_block([out["whoscored"]["run"]])
        out["block"] = xgot.summary_block([out[p]["run"] for p in PROVIDERS])
    return out


def _population(p: str, xgd: Any) -> pl.DataFrame:
    return xgd.shots if p == "statsbomb" else v1.population(xgd)[0]


def _headers(card: str) -> list[str]:
    return [x for x in card.splitlines() if x.startswith("|")]


@pytest.mark.parametrize("p", PROVIDERS)
def test_run_writes_cards(runs: dict[str, Any], p: str) -> None:
    """Three cards per provider and nothing else in the docs dir; every card says xGOT in its
    tables, carries the ECE table, the provider's xGOT validation paragraph (cross-fitting
    sentence, not the xG one) and the reconciliation rows as its filters; the checks only on the
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
    xg_line = (f"xG input: {r.xg.spec.title}, cross-fitted; provider headline xG: "
               f"{r.xg_headline.spec.title}")  # fmt: skip
    text = xgot.PROVIDERS[p].text
    for name in names:
        card = (docs / f"{name}.md").read_text(encoding="utf-8")
        assert card == r.cards[name], name
        assert f"ECE ({metrics.ECE_BINS} equal-count bins)" in card, name
        assert "| Bin | Shots | Mean xGOT | Goal rate |" in card, name
        assert [x for x in _headers(card) if re.search(r"(Σ|Mean) xG(?!OT)", x)] == [], name
        assert xgot.XGOT_CROSS_FIT in card and cards.VALIDATION not in card, name
        assert (CHECKS_TITLE in card) == (name == head_card), name
        assert (xg_line in card) == (not name.endswith("penalty")), name
        for reason in xgot.RECONCILE_REASONS:
            assert f"| {text.label(reason)} |" in card, (name, reason)
        if p == "whoscored":
            assert "StatsBomb" not in card, name
    assert len(r.checks) == 11
    if p == "statsbomb":
        assert r.xg.spec.name == f"{r.xg_headline.spec.name}-keeper-neutral"
        assert "open_goal" in r.xg_headline.spec.features
    else:
        assert r.xg.spec is r.xg_headline.spec
    assert not set(r.xg.spec.features) & features.KEEPER_POSITION
    for ev in (lr, g):
        assert ev.spec.xg_input == r.xg.spec.name
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


def test_gbm_monotone_in_placement() -> None:
    """A fit of the ``gbm`` spec (its own sign map, a fixture-sized grid) never lowers xGOT as
    the ball moves away from the centre at a fixed side (near or far post), height, xG and angle."""
    from football_shot_models import xgot

    spec = next(s for s in xgot.xgot_specs("xgot-", "xg-x") if s.name == "xgot-gbm")
    signs = spec.make().get_params()["monotone"]
    assert signs == xg.GBM_SIGNS
    assert "mouth_far" not in spec.features and "far_side" in spec.features
    tiny = dataclasses.replace(spec, make=functools.partial(gbm.MonotoneGBM, signs, **mc.SIGNAL))
    rng = np.random.default_rng(17)
    n = 800
    gy, gz = rng.uniform(-3.6, 3.6, n), rng.uniform(0.05, 2.35, n)
    y, angle, xgl = rng.uniform(15.0, 53.0, n), rng.uniform(0.2, 1.2, n), rng.normal(-1.5, 0.8, n)
    far = -np.sign(34.0 - y) * np.sign(gy)
    p = expit(0.9 * xgl + 0.9 + 0.9 * (np.abs(gy) - 1.8) + 0.3 * far + 0.4 * (gz - 1.2))
    train = pl.DataFrame({
        "shot_id": [f"sb:s{i:03d}" for i in range(n)],
        "match_id": [f"sb:m{i // 10:02d}" for i in range(n)],
        "is_goal": rng.uniform(size=n) < p,
        "on_target": [True] * n,
        "goal_y": 34.0 - gy, "goal_z": gz, "y": y, "angle_rad": angle, "xg_logit": xgl,
    })  # fmt: skip
    v = np.round(np.arange(0.2, 3.81, 0.2), 1)
    for side in (1.0, -1.0):  # from y = 20: ball at frame y > 0 is the near post
        grid = pl.DataFrame({
            "shot_id": [f"sb:g{i:02d}" for i in range(v.size)],
            "match_id": ["sb:g"] * v.size,
            "is_goal": [False] * v.size,
            "on_target": [True] * v.size,
            "goal_y": 34.0 - side * v, "goal_z": [1.0] * v.size, "y": [20.0] * v.size,
            "angle_rad": [0.6] * v.size, "xg_logit": [-1.0] * v.size,
        })  # fmt: skip
        out = evaluation.fit_predict(tiny, train, grid)
        assert np.all(np.diff(out) >= -1e-12), (side, out)


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
    from football_shot_models import xgot

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
    lines = xgot.ece_lines(ev, y, yh)
    text = "\n".join(lines)
    for value in values:
        assert cards.fmt(float(value)) in text, value
    assert len([x for x in lines if x.startswith("|")]) == 3  # header, rule, one row
    assert f"{metrics.ECE_BINS} equal-count bins" in text


def test_main_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` exits 1 with ``fsm: error:`` and no traceback on a missing manifest."""
    from football_shot_models import xgot

    monkeypatch.setenv("FSM_CACHE", "off")
    missing = {p: tmp_path / "manifests" / f"{p}.json" for p in PROVIDERS}
    monkeypatch.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, missing))
    assert xgot.main(["--provider", "statsbomb"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("fsm: error: ") and "Traceback" not in err


# ---- real snapshots (make reproduce) ----------------------------------------------------------


def _real_run(p: str, tmp: Path) -> dict[str, Any]:
    from football_shot_models import cache, xgot

    manifest = REPO / xgot.PROVIDERS[p].manifest
    snap = "statsbomb-open-v1" if p == "statsbomb" else "whoscored-v1"
    if not ((REPO / "data" / "snapshots" / snap).is_dir() and manifest.is_file()):
        pytest.skip(f"needs the {p} snapshot")
    docs = tmp / p / "models"
    result = xgot.run(p, docs_dir=docs, cache=cache.cache_root(REPO / "data" / "cache"))
    return {"run": result, "docs": docs}


@pytest.fixture(scope="module")
def real(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return {"tmp": tmp_path_factory.mktemp("real-xgot")}


def _readme_block() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- xgot:summary:start -->", "<!-- xgot:summary:end -->"
    assert text.count(start) == 1 and text.count(end) == 1
    return text.split(start)[1].split(end)[0].strip("\n")


def _sb(real: dict[str, Any]) -> dict[str, Any]:
    if "statsbomb" not in real:
        real["statsbomb"] = _real_run("statsbomb", real["tmp"])
    return dict(real["statsbomb"])


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


@pytest.mark.slow
@pytest.mark.whoscored
def test_reproduce_whoscored(real: dict[str, Any]) -> None:
    """The three WhoScored xGOT cards and the README block (both providers) come back byte for
    byte; no ``ws:`` id is in them."""
    from football_shot_models import xgot

    sb = _sb(real)
    ws = _real_run("whoscored", real["tmp"])
    for name in CARDS["whoscored"]:
        committed = (REPO / "docs" / "models" / f"{name}.md").read_bytes()
        assert (ws["docs"] / f"{name}.md").read_bytes() == committed, name
    block = xgot.summary_block([sb["run"], ws["run"]])
    assert _readme_block() == block.strip("\n")
    for text in (block, *(ws["run"].cards[n] for n in CARDS["whoscored"])):
        assert re.search(r"ws:\d", text) is None
    r = ws["run"]
    assert [r.checks[i].result for i in (7, 8)] == ["met", "met"]
    assert r.recon.unscored == 0
