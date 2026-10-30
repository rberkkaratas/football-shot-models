"""Warm-cache read with no refit, the hold-out frame, the write guards, the CLI, the exclusion
note, the default player and the placement outside the cache hash (spec
``runs/CH1/style-shots`` §1, §3.2, §4, §6)."""

from __future__ import annotations

import ast
import dataclasses
import io
import json
import re
import subprocess
import tomllib
import types
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import charts_cases as cc
import numpy as np
import polars as pl
import pytest
from football_core import viz
from football_core.io import snapshot, statsbomb
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from test_cache_render_only import _imports, _module

from football_shot_charts import render
from football_shot_models import (
    baselines,
    cache,
    cards,
    data,
    evaluation,
    metrics,
    models,
    v1,
    xgot,
)

REPO = Path(__file__).resolve().parents[1]
LR = baselines.BASELINES[2]  # distance, angle, body part: no freeze-frame feature


class Counting:
    """Proxy estimator that counts ``fit`` calls (same params, so the same cache identity)."""

    fits = 0

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def get_params(self, deep: bool = True) -> Any:
        return self.inner.get_params(deep)

    def fit(self, *args: Any, **kwargs: Any) -> Counting:
        Counting.fits += 1
        self.inner.fit(*args, **kwargs)
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


# no coefficients: the card reader expects the bare pipeline, and the cache identity ignores them
COUNTED = dataclasses.replace(LR, make=lambda: Counting(LR.make()), coefficients=None)


def _split(xg: Any) -> tuple[pl.DataFrame, pl.DataFrame, Any]:
    dev, held = data.split_holdout(xg.shots, data.pair_holdout(xg.matches))
    return dev, held, evaluation.cv_folds(dev)


@pytest.fixture
def warm(xg_data: Any, tmp_path: Path) -> Iterator[tuple[Path, Any]]:
    """A cache root warmed for ``COUNTED`` by ``evaluation.evaluate`` (fits counted, then reset)."""
    dev, held, folds = _split(xg_data)
    pc = cache.PredictionCache(tmp_path / "cache", xg_data.manifest.data_digest)
    Counting.fits = 0
    ev = evaluation.evaluate(COUNTED, dev, held, folds, cache=pc)
    assert pc.written == [COUNTED.name] and Counting.fits > 0  # precondition: it really fitted
    Counting.fits = 0
    yield tmp_path / "cache", ev


def _no_fit(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def spy(name: str) -> Any:
        def refuse(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            raise AssertionError(f"{name} called: a refit")

        return refuse

    monkeypatch.setattr(evaluation, "fit_predict", spy("fit_predict"))
    monkeypatch.setattr(evaluation, "_fit", spy("_fit"))
    return calls


def test_cached_evaluation_warm_no_fit(
    xg_data: Any, warm: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm entry is read with no fit (spies on ``fit_predict``, ``_fit`` and the estimator's
    ``fit``); the predictions are the warm ones; nothing is written."""
    root, ev0 = warm
    assert ev0.oof.size and ev0.holdout_p.size  # census: dev (120) and hold-out (20) both read
    calls = _no_fit(monkeypatch)
    pc = cache.PredictionCache(root, xg_data.manifest.data_digest)
    ev = render.cached_evaluation(xg_data, pc, spec=COUNTED)
    assert calls == [] and Counting.fits == 0
    assert pc.hits == [COUNTED.name] and pc.written == []
    assert np.array_equal(ev.oof, ev0.oof)
    assert np.array_equal(ev.holdout_p, ev0.holdout_p)


def test_cached_evaluation_cold_refuses(xg_data: Any, tmp_path: Path) -> None:
    """A cold cache raises ``ValueError`` naming the spec and ``make xg-models``; no estimator
    is fitted and nothing is written."""
    pc = cache.PredictionCache(tmp_path / "cache", xg_data.manifest.data_digest)
    Counting.fits = 0
    with pytest.raises(ValueError) as info:
        render.cached_evaluation(xg_data, pc, spec=COUNTED)
    msg = str(info.value)
    assert COUNTED.name in msg and "make xg-models" in msg and "no warm" in msg
    assert Counting.fits == 0 and pc.written == []
    assert list((tmp_path / "cache").rglob("*.npz")) == []


def test_written_raises(xg_data: Any, warm: tuple[Path, Any]) -> None:
    """If the cache records a write during the call, ``cached_evaluation`` raises (explicitly,
    not by ``assert``)."""
    root, _ = warm

    class Growing(cache.PredictionCache):
        def load(self, *args: Any, **kwargs: Any) -> Any:
            hit = super().load(*args, **kwargs)
            assert hit is not None  # precondition: the entry is warm
            self.written.append("phantom")
            return hit

    with pytest.raises(ValueError):
        render.cached_evaluation(xg_data, Growing(root, xg_data.manifest.data_digest), spec=LR)


def test_holdout_xg_rows(
    xg_data: Any, warm: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hold-out rows, in order, with ``xg = ev.holdout_p``."""
    root, ev0 = warm
    _no_fit(monkeypatch)
    held = _split(xg_data)[1]
    assert held.height > 0  # census: the hold-out rows the chart reads are there
    out = render.holdout_xg(xg_data, cache.PredictionCache(root, xg_data.manifest.data_digest),
                            spec=COUNTED)  # fmt: skip
    assert out["shot_id"].to_list() == held["shot_id"].to_list()
    assert out.schema["xg"] == pl.Float64
    assert np.array_equal(out["xg"].to_numpy(), ev0.holdout_p)


def test_model_text() -> None:
    text = render.model_text("Premier League 2015/2016")
    assert text == "xG: own model, not trained on Premier League 2015/2016"
    assert "never used to choose" not in text


# --- write ------------------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo (cwd) ignoring ``data/`` and ``outputs/``, like the real one."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("data/\noutputs/\n")
    monkeypatch.chdir(root)
    return root


def _canvas(provider: str, logo: Path) -> viz.Canvas:
    footer = viz.Footer((provider,), model="xG: own model")
    return viz.canvas("light", "square", title="A synthetic title", subtitle="Team · League",
                      footer=footer, logo=logo)  # fmt: skip


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("logo") / "logo.png")


def test_write_refuses_unignored(repo: Path, logo: Path) -> None:
    ok = render.write(_canvas("statsbomb", logo), Path("outputs/charts/team-x-light-square.png"))
    assert ok.is_file()
    target = repo / "charts" / "x.png"
    with pytest.raises(ValueError):
        render.write(_canvas("statsbomb", logo), target)
    assert not target.exists()


def test_write_refuses_whoscored_unless_private(repo: Path, logo: Path) -> None:
    public = repo / "outputs" / "charts" / "ws.png"
    with pytest.raises(ValueError):
        render.write(_canvas("whoscored", logo), public)
    assert not public.exists()
    with pytest.raises(ValueError):  # private, but not under data/charts/private
        render.write(_canvas("whoscored", logo), public, private=True)
    assert not public.exists()
    private = Path("data/charts/private/ws.png")
    assert render.write(_canvas("whoscored", logo), private, private=True).is_file()


def test_private_path_containment(repo: Path, logo: Path) -> None:
    """``data/charts/private/../x.png`` is not under the private folder (normpath, no resolve);
    neither is a sibling whose name only starts like it."""
    for bad in ("data/charts/private/../x.png", "data/charts/private-not/x.png"):
        with pytest.raises(ValueError):
            render.write(_canvas("whoscored", logo), Path(bad), private=True)
        assert not (repo / bad).exists() and not (repo / "data/charts/x.png").exists()


def test_write_refuses_failed_checks(repo: Path, logo: Path) -> None:
    c = _canvas("statsbomb", logo)
    for word in ("Overlapping", "Underneath"):
        c.fig.text(0.5, 0.5, word, fontsize=14)
    target = repo / "outputs" / "charts" / "bad.png"
    with pytest.raises(ValueError) as info:
        render.write(c, target)
    assert "Overlapping" in str(info.value) or "Underneath" in str(info.value)
    assert not target.exists()


def test_outputs_ignored_in_repo() -> None:
    """The real repo ignores the ``make charts`` folder and the private WhoScored folder."""
    for path in ("outputs/charts/team-sb-22-light-portrait.png", "data/charts/private/x.png"):
        done = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", path], check=False)
        assert done.returncode == 0, path


# --- main -------------------------------------------------------------------------------------


def _argv(manifest: Path, raw: Path, cache_root: Path, out: Path) -> list[str]:
    return ["--team", "sb:t01", "--team-name", "Team One", "--statsbomb-raw", str(raw),
            "--out", str(out), "--cache", str(cache_root), "--manifest", str(manifest)]  # fmt: skip


def _layout(tmp_path: Path, manifest: Path, *, entry: bool, tree: bool) -> tuple[Path, Path]:
    raw, root = tmp_path / "raw", tmp_path / "cache"
    if tree:
        (raw / statsbomb.OPEN_DATA_COMMIT).mkdir(parents=True)
        (raw / statsbomb.OPEN_DATA_COMMIT / "tree.json").write_text("{}")
    if entry:
        digest = snapshot.read_manifest(manifest).data_digest
        d = cache.PredictionCache(root, digest).dir
        d.mkdir(parents=True)
        (d / f"{models.GBM.name}.npz").write_bytes(b"")
    return raw, root


def _run(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = render.main(argv)
    return rc, out.getvalue(), err.getvalue()


def _recorder(calls: list[object], real: Any) -> Any:
    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        return real(*args, **kwargs)

    return spy


@pytest.mark.parametrize("case", ["no-manifest", "cold-cache", "no-tree"])
def test_main_bad_input_exit_1(
    case: str, manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing manifest, cache entry or ``tree.json``: exit 1, ``charts: error:`` on stderr, no
    traceback, and nothing is loaded before the checks."""
    loads: list[object] = []
    real = data.load_xg_data
    for mod in (data, render):
        if hasattr(mod, "load_xg_data"):
            monkeypatch.setattr(mod, "load_xg_data", _recorder(loads, real))
    raw, root = _layout(tmp_path, manifest, entry=case != "cold-cache", tree=case != "no-tree")
    m = tmp_path / "missing.json" if case == "no-manifest" else manifest
    rc, _, err = _run(_argv(m, raw, root, tmp_path / "out"))
    assert rc == 1 and err.startswith("charts: error:") and "Traceback" not in err
    assert loads == []


def test_main_reads_xg_gbm(manifest: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` reads the model of record: ``holdout_xg`` with ``spec`` the xg-gbm headline and
    the cache keyed by the manifest's digest (stopped there by the spy)."""
    raw, root = _layout(tmp_path, manifest, entry=True, tree=True)
    seen: list[tuple[Any, Any]] = []

    def spy(xg: Any, pc: Any, *, spec: Any = models.GBM) -> Any:
        seen.append((pc, spec))
        raise ValueError("spy stop")

    monkeypatch.setattr(render, "holdout_xg", spy)
    rc, _, err = _run(_argv(manifest, raw, root, tmp_path / "out"))
    assert rc == 1 and "spy stop" in err and "Traceback" not in err
    ((pc, spec),) = seen
    assert spec is models.GBM and models.GBM.name == "xg-gbm"
    assert isinstance(pc, cache.PredictionCache)
    assert pc.dir == cache.PredictionCache(root, snapshot.read_manifest(manifest).data_digest).dir


# --- exclusion note and default player ---------------------------------------------------------

HO = data.HOLDOUT


def _excl_frames() -> tuple[pl.DataFrame, pl.DataFrame]:
    """One snapshot row per exclusion reason for the team (two of most, so plurals are fixed),
    an own goal *for* the team (opponent_team_id == team) and one *against* it (team_id ==
    team), another team's penalty, and a team penalty outside the hold-out pair."""
    t, o = cc.TEAM, cc.OTHER
    rows = [  # (shot_id, team, opponent, player, reason, goal, pair)
        ("p1", t, o, cc.PLAYER, "penalty", True, HO), ("p2", t, o, "sb:5", "penalty", True, HO),
        ("p3", t, o, "sb:5", "penalty", False, HO), ("s1", t, o, "sb:5", "shootout", True, HO),
        ("s2", t, o, "sb:5", "shootout", False, HO),
        ("c1", t, o, "sb:5", "direct_corner", False, HO),
        ("c2", t, o, cc.PLAYER, "direct_corner", True, HO),
        ("f1", t, o, "sb:5", "no_freeze_frame", True, HO),
        ("f2", t, o, "sb:5", "no_freeze_frame", False, HO),
        ("g1", o, t, "sb:91", "own_goal", True, HO), ("g2", o, t, "sb:92", "own_goal", True, HO),
        ("g3", t, o, cc.PLAYER, "own_goal", True, HO),  # the team's own goal: conceded
        ("x1", o, t, "sb:91", "penalty", True, HO),  # another team's penalty
        ("y1", t, o, cc.PLAYER, "penalty", True, ("sb:9", "sb:1")),  # not the hold-out pair
    ]  # fmt: skip
    shots_table = pl.DataFrame(
        {
            "shot_id": [f"sb:{r[0]}" for r in rows],
            "match_id": ["sb:m1" if r[6] == HO else "sb:m9" for r in rows],
            "team_id": [r[1] for r in rows],
            "opponent_team_id": [r[2] for r in rows],
            "player_id": [r[3] for r in rows],
            "is_goal": [r[5] for r in rows],
        }
    )
    exclusions = pl.DataFrame(
        {
            "shot_id": [f"sb:{r[0]}" for r in rows],
            "match_id": ["sb:m1" if r[6] == HO else "sb:m9" for r in rows],
            "competition_id": [r[6][0] for r in rows],
            "season_id": [r[6][1] for r in rows],
            "reason": [r[4] for r in rows],
            "is_goal": [r[5] for r in rows],
        },
        schema=data.EXCLUSION_SCHEMA,
    ).sort("shot_id")
    return shots_table, exclusions


def test_excluded_note_counts() -> None:
    """Parts per reason in ``EXCLUSION_REASONS`` order; own goals found by ``opponent_team_id``
    (not ``team_id``), team charts only; plurals computed; the player note filters by player."""
    shots_table, exclusions = _excl_frames()
    parts = render.excluded_note(shots_table, exclusions, team=cc.TEAM)
    assert set(parts) == {
        "3 penalties (2 scored)", "2 shoot-out kicks", "2 direct corners (1 scored)",
        "2 shots without freeze frame (1 scored)", "2 own goals by opponents",
    }  # fmt: skip
    order = [p for p in parts if "own goal" not in p]
    assert order == ["2 shoot-out kicks", "3 penalties (2 scored)", "2 direct corners (1 scored)",
                     "2 shots without freeze frame (1 scored)"]  # fmt: skip
    mine = render.excluded_note(shots_table, exclusions, team=cc.TEAM, player=cc.PLAYER)
    assert mine == ("1 penalty (1 scored)", "1 direct corner (1 scored)")


def _held(rows: list[tuple[str, str, bool]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "shot_id": [f"sb:s{i}" for i in range(len(rows))],
            "team_id": [r[1] for r in rows],
            "player_id": [r[0] for r in rows],
            "is_goal": [r[2] for r in rows],
        }
    ).reverse()  # out of key order


def test_default_player_tie() -> None:
    """Most kept shots for the team; a tie goes to the smallest id as a string ("sb:10960" <
    "sb:2" < "sb:3814"); rows of another team do not count; outcomes are never read."""
    base = (
        [("sb:3814", cc.TEAM, True)] * 5 + [("sb:2", cc.TEAM, False)] * 3
        + [("sb:10960", cc.TEAM, False)] * 5 + [("sb:2", cc.OTHER, False)] * 9
    )  # fmt: skip
    held = _held(base)
    assert render.default_player(held, cc.TEAM) == "sb:10960"  # sb:2 has 12 rows, 3 for the team
    assert render.default_player(held.with_columns(~pl.col("is_goal")), cc.TEAM) == "sb:10960"
    assert render.default_player(_held(base + [("sb:2", cc.TEAM, False)] * 3), cc.TEAM) == "sb:2"


# --- placement --------------------------------------------------------------------------------


def test_outside_hash_and_one_way() -> None:
    """``football_shot_charts`` sits outside every ``cache.ROOTS`` (so ``source_hash`` never sees
    it), ships beside ``football_shot_models`` in the wheel, ``cache.RENDER_ONLY`` is unchanged
    (E3); no module under ``cache.ROOTS`` outside ``RENDER_ONLY`` imports it, and
    ``gsax_cards`` imports only ``football_shot_charts.keepers`` (amended, CH1
    keeper-model-charts §1)."""
    pkg = REPO / "src" / "football_shot_charts"
    for name in ("shots", "render", "rows", "keepers", "diagnostics"):
        assert (pkg / f"{name}.py").is_file(), name
    for root in cache.ROOTS:
        assert not pkg.resolve().is_relative_to(Path(root).resolve()), root
    assert frozenset({
        "football_shot_models/xgot_cards.py", "football_shot_models/gsax_cards.py",
        "football_shot_models/v2_cards.py", "football_core/viz.py",
    }) == cache.RENDER_ONLY  # fmt: skip
    conf = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    names = conf["tool"]["uv"]["build-backend"]["module-name"]
    assert sorted(names) == ["football_shot_charts", "football_shot_models"]
    assert _imports("import football_shot_charts.render", "x.y", False)  # detector precondition
    seen = set()
    for root in cache.ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            mods = _imports(path.read_text(encoding="utf-8"), _module(Path(root), path),
                            is_package=path.name == "__init__.py")  # fmt: skip
            found = {m for m in mods if m.split(".")[0] == "football_shot_charts"}
            rel = f"{Path(root).name}/{path.relative_to(root).as_posix()}"
            if rel == "football_shot_models/gsax_cards.py":
                seen.add(rel)
                assert "football_shot_charts.keepers" in found, found
                assert found <= {"football_shot_charts", "football_shot_charts.keepers"}, found
            else:
                assert found == set(), (path, found)
    assert seen == {"football_shot_models/gsax_cards.py"}  # the detector reached it
    tree = ast.parse((pkg / "shots.py").read_text(encoding="utf-8"))
    assert not any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "fit"
                   for n in ast.walk(tree))  # fmt: skip


# ==== CH1 keeper-model-charts (spec runs/CH1/keeper-model-charts §4.2, §5, §8) =================

# --- calibration bins ---------------------------------------------------------------------------


def _floor() -> float:
    """``P_FLOOR`` (§4.2, 1e-3): ``render``'s, or the drawing module's when it lives there."""
    from football_shot_charts import diagnostics

    return float(getattr(render, "P_FLOOR", None) or diagnostics.P_FLOOR)  # noqa: B009


def _bin_inputs(n: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``n`` shots out of p order: p from 0.02 to 0.6 (a few ties), 3 unequal match groups in
    every bin (5, 3, 2 shots), a goal in every match group of bins 2-9, bin 0 all misses."""
    rng = np.random.default_rng(7)
    p = np.round(np.linspace(0.02, 0.6, n), 3)
    p[10:12] = p[10]  # a tie inside bin 1
    order = rng.permutation(n)
    y = np.zeros(n, dtype=bool)
    groups = np.empty(n, dtype=object)
    for b in range(n // 10):
        idx = np.arange(10 * b, 10 * b + 10)  # positions in p order
        sizes = (5, 3, 2)
        start = 0
        for g, size in enumerate(sizes):
            for k in idx[start : start + size]:
                groups[k] = f"sb:m{(b + g) % 7}"
            if b >= 2:
                y[idx[start]] = True
            start += size
    return y[order], p[order], groups[order].astype(str)


def test_bins_equal_reliability() -> None:
    """Per bin, ``shots``, ``mean_p`` and ``rate`` are ``metrics.reliability``'s (the same bin
    rule); dtypes as specified; ``inside`` / ``clipped`` Boolean."""
    y, p, groups = _bin_inputs()
    out = render.calibration_bins(y, p, groups)
    ref = metrics.reliability(y, p, metrics.ECE_BINS)
    assert out["bin"].to_list() == ref["bin"].to_list() == list(range(10))
    assert out["shots"].to_list() == ref["shots"].to_list()
    assert out["mean_p"].to_numpy() == pytest.approx(ref["mean_xg"].to_numpy(), abs=1e-12)
    assert out["rate"].to_numpy() == pytest.approx(
        np.clip(ref["goal_rate"].to_numpy(), _floor(), 1 - _floor()), abs=1e-12
    )
    assert dict(out.schema) == {
        "bin": pl.Int64, "shots": pl.Int64, "mean_p": pl.Float64, "rate": pl.Float64,
        "low": pl.Float64, "high": pl.Float64, "inside": pl.Boolean, "clipped": pl.Boolean,
    }  # fmt: skip
    assert (out["low"] <= out["high"]).all()


def test_bins_ties_on_edge() -> None:
    """Tied predictions straddling a bin edge (input order breaks the tie, as
    ``metrics.reliability``): the same bins, counts and means for 4 bins of 5."""
    p = np.array([0.3, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.1, 0.5, 0.6, 0.2, 0.7, 0.8, 0.9,
                  0.2, 0.25, 0.35, 0.45, 0.55])  # fmt: skip
    y = np.array([i % 3 == 0 for i in range(p.size)])
    groups = np.array([f"sb:m{i % 4}" for i in range(p.size)])
    out = render.calibration_bins(y, p, groups, bins=4)
    ref = metrics.reliability(y, p, 4)
    assert (p == 0.2).sum() == 7  # precondition: 7 ties over the 5-shot edge
    assert out["shots"].to_list() == ref["shots"].to_list() == [5, 5, 5, 5]
    assert out["mean_p"].to_numpy() == pytest.approx(ref["mean_xg"].to_numpy(), abs=1e-12)


def test_bins_cross_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``metrics.reliability`` that disagrees by one shot in one bin: ``ValueError`` (the bin
    rule is re-typed and cross-checked)."""
    y, p, groups = _bin_inputs()
    real = metrics.reliability

    def off(*args: Any, **kwargs: Any) -> pl.DataFrame:
        r = real(*args, **kwargs)
        return r.with_columns(shots=pl.when(pl.col("bin") == 3).then(pl.col("shots") + 1)
                              .otherwise(pl.col("shots")))  # fmt: skip

    monkeypatch.setattr(metrics, "reliability", off)
    with pytest.raises(ValueError):
        render.calibration_bins(y, p, groups)


def _band(y: np.ndarray, p: np.ndarray, groups: np.ndarray, b: int, level: float,
          bins: int = 10) -> tuple[float, float]:  # fmt: skip
    """One bin's match-bootstrap band by an explicit loop over the ``group_weights`` rows."""
    rank = pl.Series(p).rank("ordinal").cast(pl.Int64).to_numpy() - 1
    mask = rank * bins // p.size == b
    inv, w = metrics.group_weights(groups, metrics.N_BOOT, metrics.SEED)
    m = w.shape[1]
    shots = np.bincount(inv[mask], minlength=m).astype(float)
    goals = np.bincount(inv[mask], weights=y[mask].astype(float), minlength=m)
    rates = []
    for row in w:
        s = float(row @ shots)
        if s == 0:
            continue
        rates.append(float(row @ goals) / s)
    lo, hi = np.quantile(rates, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def test_band_independent() -> None:
    """The band of bins 4 and 8 equals the explicit loop over the match resamples (weighted
    goals / weighted shots, resamples without a shot in the bin dropped, percentiles at
    ``(1 ± GAIN_LEVEL) / 2``); ``inside`` is ``low ≤ mean_p ≤ high``."""
    y, p, groups = _bin_inputs()
    out = render.calibration_bins(y, p, groups)
    for b in (4, 8):
        lo, hi = _band(y, p, groups, b, metrics.GAIN_LEVEL)
        assert lo > _floor()  # precondition: nothing clipped in these bins
        row = out.row(b, named=True)
        assert row["low"] == pytest.approx(lo, abs=1e-12)
        assert row["high"] == pytest.approx(hi, abs=1e-12)
        assert row["inside"] == (lo <= row["mean_p"] <= hi)


def test_gain_level_patched(monkeypatch: pytest.MonkeyPatch) -> None:
    """``metrics.GAIN_LEVEL`` is read when called: at 0.5 the band is the 25-75 % range."""
    y, p, groups = _bin_inputs()
    monkeypatch.setattr(metrics, "GAIN_LEVEL", 0.5)
    out = render.calibration_bins(y, p, groups)
    lo, hi = _band(y, p, groups, 6, 0.5)
    assert out.row(6, named=True)["low"] == pytest.approx(lo, abs=1e-12)
    assert out.row(6, named=True)["high"] == pytest.approx(hi, abs=1e-12)


def test_inside_unclipped_and_clip() -> None:
    """``inside`` is judged on the unclipped band, then rate / low / high are clipped to
    ``[P_FLOOR, 1 − P_FLOOR]`` for drawing; ``clipped`` iff the observed rate was 0 or 1.
    Bin 0 (p 0.0008, one goal in one of 3 matches): band [0, h] holds 0.0008 unclipped (it
    would not once clipped at 0.001). Bin 1 (no goal): rate 0 → drawn at ``P_FLOOR``. Bin 9
    (all goals): rate 1 → drawn at 1 − ``P_FLOOR``."""
    assert _floor() == 1e-3
    n = 100
    p = np.concatenate([np.full(10, 0.0008), np.full(10, 0.05), np.linspace(0.1, 0.6, 70),
                        np.full(10, 0.95)])  # fmt: skip
    y = np.zeros(n, dtype=bool)
    y[0] = True
    y[20:90:3] = True
    y[90:] = True
    groups = np.array([f"sb:m{i % 3}" if i < 10 else f"sb:m{i % 5}" for i in range(n)])
    out = render.calibration_bins(y, p, groups)
    b0, b1, b9 = (out.row(i, named=True) for i in (0, 1, 9))
    assert b0["mean_p"] == pytest.approx(0.0008) and b0["inside"] and not b0["clipped"]
    assert b0["low"] == pytest.approx(_floor())  # drawn clipped
    assert b1["clipped"] and b1["rate"] == pytest.approx(_floor()) and not b1["inside"]
    assert b9["clipped"] and b9["rate"] == pytest.approx(1 - _floor())
    assert not out.filter(pl.col("bin").is_between(2, 8))["clipped"].any()


def _all_dropped(monkeypatch: pytest.MonkeyPatch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``N_BOOT`` 1 and the top bin's shots all in a match the single resample never draws."""
    monkeypatch.setattr(metrics, "N_BOOT", 1)
    ids = np.array([f"sb:m{i:02d}" for i in range(20)])
    _, w = metrics.group_weights(ids, 1, metrics.SEED)
    zero = ids[int(np.flatnonzero(w[0] == 0)[0])]
    others = [i for i in ids if i != zero]
    p = np.linspace(0.05, 0.6, 100)
    groups = np.array([others[i % len(others)] for i in range(90)] + [zero] * 10)
    y = np.array([i % 4 == 0 for i in range(100)])
    return y, p, groups


@pytest.mark.parametrize(
    "case", ["length", "y-2", "p-nan", "p-inf", "p-1.2", "mean-0", "null-group", "all-dropped"]
)
def test_bins_refusals(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ValueError``: unequal lengths; y not 0/1; p NaN, ±inf or outside [0, 1]; a bin whose
    mean prediction is 0 (an infinite logit axis); a null group; every resample of a bin
    dropped."""
    y, p, groups = _bin_inputs()
    y = y.astype(np.int64)
    groups_o = groups.astype(object)
    if case == "length":
        p = p[:-1]
    elif case == "y-2":
        y[3] = 2
    elif case == "p-nan":
        p[5] = np.nan
    elif case == "p-inf":
        p[5] = np.inf
    elif case == "p-1.2":
        p[5] = 1.2
    elif case == "mean-0":
        p[np.argsort(p)[:10]] = 0.0
    elif case == "null-group":
        groups_o[4] = None
        groups = groups_o
    elif case == "all-dropped":
        y, p, groups = _all_dropped(monkeypatch)
    with pytest.raises(ValueError):
        render.calibration_bins(y, p, groups)


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.tuples(st.floats(min_value=0.001, max_value=0.999), st.booleans(),
                       st.integers(min_value=0, max_value=5)), min_size=30, max_size=80)
)  # fmt: skip
def test_bins_property(rows: list[tuple[float, bool, int]]) -> None:
    """For any valid input: one row per bin, shots summing to n and equal to the reliability
    rule's, low ≤ high, every drawn value inside [P_FLOOR, 1 − P_FLOOR], ``clipped`` iff the
    observed rate is 0 or 1."""
    p = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    groups = np.array([f"sb:m{r[2]}" for r in rows])
    assume(np.unique(groups).size >= 2)
    out = render.calibration_bins(y, p, groups)
    ref = metrics.reliability(y, p, metrics.ECE_BINS)
    assert out["shots"].sum() == len(rows) and out["shots"].to_list() == ref["shots"].to_list()
    assert (out["low"] <= out["high"]).all()
    for col in ("rate", "low", "high"):
        assert out[col].is_between(_floor(), 1 - _floor()).all()
    raw = ref["goal_rate"].to_numpy()
    assert out["clipped"].to_list() == [bool(r in (0.0, 1.0)) for r in raw]


# --- references and views -----------------------------------------------------------------------


def _renamed(ev: Any, name: str) -> Any:
    return dataclasses.replace(ev, spec=dataclasses.replace(ev.spec, name=name))


def _forced_head(monkeypatch: pytest.MonkeyPatch) -> None:
    """``models.headline`` keeps the GBM whatever its gain (the fixture's GBM does not clear
    the rule; the rule itself is ``test_models_headline``'s)."""

    def gbm_head(best_lr: Any, gbm: Any, dev: pl.DataFrame, held: pl.DataFrame) -> Any:
        return gbm, evaluation.compare(f"over {best_lr.spec.title}", best_lr, gbm, dev, held)

    monkeypatch.setattr(models, "headline", gbm_head)


def _sb_candidates(xgot_fits: dict[str, Any], *, reverse: bool = False) -> tuple[Any, Any]:
    """The StatsBomb fixture fit and a candidate set whose names are the real ones: the GBM as
    ``xg-gbm``, the fixture LR as S0 and the keeper-neutral input (lowest CV log loss) as the
    best LR; ``reverse`` lists the baselines the other way round."""
    f = xgot_fits["fits"]["statsbomb"]
    c = xgot_fits["calls"]["statsbomb"][0][1]
    s0 = _renamed(c.lrs[0], "xg-lr-distance-angle-body-part")
    best = _renamed(f.xg_input, "xg-lr-fixture-best")
    assert best.cv.log_loss < s0.cv.log_loss  # precondition: S0 is not the best LR
    base = (best, s0)[::-1] if reverse else (best, s0)
    return f, models.Candidates(baselines=base, steps=(), gbm=_renamed(c.gbm, models.GBM.name))


def _gain(c: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (c.cv.gain, c.cv.low, c.cv.high), (c.holdout.gain, c.holdout.low, c.holdout.high)


def test_refs_per_provider(xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """StatsBomb: the best LR, S0 and StatsBomb's xG, picked by spec name (the same rows when
    the baselines are listed the other way round), each gain the ``evaluation.compare`` of the
    head over that reference; ``n_boot`` the comparisons'. WhoScored's set names v1's distance
    + angle LR and no benchmark."""
    _forced_head(monkeypatch)
    f, cands = _sb_candidates(xgot_fits)
    dev, held = f.frames
    view = render.xg_view(f, cands)
    assert [r.role for r in view.rows] == [
        "Best logistic regression", "Standard baseline (S0)", "StatsBomb's xG"]  # fmt: skip
    best, s0 = cands.baselines
    bench = evaluation.evaluate_benchmark(dev, held, evaluation.cv_folds(dev))
    for row, ref in zip(view.rows, (best, s0, bench), strict=True):
        cv, ho = _gain(evaluation.compare("x", ref, cands.gbm, dev, held))
        assert row.cv == pytest.approx(cv, abs=1e-15) and row.held == pytest.approx(ho, abs=1e-15)
    assert view.rows[2].title == "provider model, freeze-frame inputs"
    assert view.n_boot == metrics.N_BOOT
    _, flipped = _sb_candidates(xgot_fits, reverse=True)
    again = render.xg_view(f, flipped)
    assert [(r.role, r.cv, r.held) for r in again.rows] == [(r.role, r.cv, r.held)
                                                             for r in view.rows]  # fmt: skip
    ws = dict(render.XG_REFS["whoscored"])
    assert v1.DISTANCE_ANGLE in ws and render.BEST_LR in ws
    assert evaluation.BENCHMARK.name not in ws
    assert dict(render.HEADS) == {"statsbomb": models.GBM.name, "whoscored": v1.GBM.name}


def test_xg_view_head_guard(xgot_fits: dict[str, Any]) -> None:
    """Without the forced rule the fixture's head is an LR, not ``HEADS["statsbomb"]``:
    ``ValueError`` (a chart never draws another model as the headline)."""
    f, cands = _sb_candidates(xgot_fits)
    head, _ = models.select_headline(cands.lrs, cands.gbm, *f.frames)
    assert head.spec.name != render.HEADS["statsbomb"]  # precondition
    with pytest.raises(ValueError):
        render.xg_view(f, cands)


def test_xgot_view_refs_by_name(xgot_fits: dict[str, Any]) -> None:
    """xGOT: the one drawn reference is ``{prefix}lr``, the strip extras ``{prefix}xg-only`` and
    ``{prefix}base-rate``, each by name (the same view with ``evaluations`` reversed), gains the
    ``evaluation.compare`` of the head; outcome counts and bins from the on-target frames."""
    f = xgot_fits["fits"]["statsbomb"]
    assert f.head.spec.name == "xgot-gbm"  # precondition: the head is not the LR
    view = render.xgot_view(f)
    on_dev, on_held = f.inp.on_dev, f.inp.on_held
    by = {e.spec.name: e for e in f.evaluations}
    (row,) = view.rows
    assert row.role == "Logistic regression"
    cv, ho = _gain(evaluation.compare("x", by["xgot-lr"], f.head, on_dev, on_held))
    assert row.cv == pytest.approx(cv, abs=1e-15) and row.held == pytest.approx(ho, abs=1e-15)
    extras = {e.label: e for e in view.extras}
    assert set(extras) == {"xG only", "base rate"}
    for label, name in (("xG only", "xgot-xg-only"), ("base rate", "xgot-base-rate")):
        c = evaluation.compare("x", by[name], f.head, on_dev, on_held)
        assert extras[label].cv == pytest.approx(c.cv.gain, abs=1e-15)
        assert extras[label].held == pytest.approx(c.holdout.gain, abs=1e-15)
    flipped = render.xgot_view(dataclasses.replace(f, evaluations=f.evaluations[::-1]))
    assert flipped.rows == view.rows and flipped.extras == view.extras
    assert view.quantity == "xGOT" and view.unit == "on-target shots"
    assert view.cv_n == on_dev.height and view.held_n == on_held.height
    assert view.cv_bins["shots"].sum() == on_dev.height
    yh, ph = evaluation.labels(on_held), f.head.holdout_p
    assert view.held_gap == pytest.approx(float(yh.sum() - ph.sum()), abs=1e-9)
    assert view.held_sd == pytest.approx(float(np.sqrt(np.sum(ph * (1 - ph)))), abs=1e-9)
    assert render.BENCHMARK_NOTE not in view.notes


def test_xgot_view_whoscored(xgot_fits: dict[str, Any]) -> None:
    """Integration census (CH1 keeper-model-charts integrate): the WhoScored fixture's xGOT head
    is its LR, so the one reference is the head and is skipped: ``ValueError`` (a chart with no
    row), not an empty view. With the GBM as head, the references and extras are the ``ws-``
    prefixed evaluations and the split is the test season."""
    f = xgot_fits["fits"]["whoscored"]
    by = {e.spec.name: e for e in f.evaluations}
    assert f.head is by["ws-xgot-lr"]  # precondition: the reference is the head
    with pytest.raises(ValueError, match="no xGOT reference"):
        render.xgot_view(f)
    g = dataclasses.replace(f, head=by["ws-xgot-gbm"])
    view = render.xgot_view(g)
    (row,) = view.rows
    c = evaluation.compare("x", by["ws-xgot-lr"], g.head, g.inp.on_dev, g.inp.on_held)
    assert row.cv == pytest.approx(_gain(c)[0], abs=1e-15)
    assert [e.label for e in view.extras] == ["xG only", "base rate"]
    base = evaluation.compare("x", by["ws-xgot-base-rate"], g.head, g.inp.on_dev, g.inp.on_held)
    assert view.extras[1].held == pytest.approx(base.holdout.gain, abs=1e-15)
    assert view.split[0] == "Test season"


def test_views_need_no_cache(xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Both views are pure on an in-memory fit with ``f.cache is None`` (no fit: the fit
    functions are refused); the xG view's bins are the head's out-of-fold and hold-out
    predictions; contexts count the dev competition-seasons; the level is ``GAIN_LEVEL`` read
    when called."""
    _forced_head(monkeypatch)
    f, cands = _sb_candidates(xgot_fits)
    assert f.cache is None  # precondition
    calls = _no_fit(monkeypatch)
    monkeypatch.setattr(metrics, "GAIN_LEVEL", 0.8)
    view = render.xg_view(f, cands)
    xv = render.xgot_view(f)
    assert calls == []
    assert view.level == xv.level == 0.8
    dev, held = f.frames
    ref = metrics.reliability(evaluation.labels(dev), cands.gbm.oof, metrics.ECE_BINS)
    assert view.cv_bins["mean_p"].to_numpy() == pytest.approx(ref["mean_xg"].to_numpy())
    ref_h = metrics.reliability(evaluation.labels(held), cands.gbm.holdout_p, metrics.ECE_BINS)
    assert view.held_bins["shots"].to_list() == ref_h["shots"].to_list()
    assert view.cv_slope == cands.gbm.cv.cal_slope
    k = dev.select("competition_id", "season_id").unique().height
    assert view.contexts[0] == f"{k} competition-seasons"
    assert held.select("competition_id", "season_id").unique().height == 1  # precondition
    assert "competition-seasons" not in view.contexts[1] and view.contexts[1].strip()
    assert view.split[0] == "Hold-out" and view.unit == "shots" and view.quantity == "xG"


def test_benchmark_note_iff_row(xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """``BENCHMARK_NOTE`` is a footer note part iff the benchmark row is drawn (StatsBomb xG;
    not WhoScored xG, not xGOT); the card claim it shortens still stands."""
    _forced_head(monkeypatch)
    f, cands = _sb_candidates(xgot_fits)
    assert render.BENCHMARK_NOTE in render.xg_view(f, cands).notes
    fw = xgot_fits["fits"]["whoscored"]
    cw = xgot_fits["calls"]["whoscored"][0][1]
    ws_cands = models.Candidates(baselines=(_renamed(cw.lrs[0], v1.DISTANCE_ANGLE),), steps=(),
                                 gbm=_renamed(cw.gbm, v1.GBM.name))  # fmt: skip
    ws = render.xg_view(fw, ws_cands)
    assert render.BENCHMARK_NOTE not in ws.notes
    assert "StatsBomb's xG" not in [r.role for r in ws.rows]
    assert ws.split[0] == "Test season"
    claim = cards.STATSBOMB_TEXT.benchmark_caveat
    assert claim is not None
    assert "likely includes these matches" in claim
    assert "not a like-for-like out-of-sample competitor" in claim


def test_benchmark_note_not_on_calibration(
    xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Result audit A1: a benchmark view's calibration chart draws no StatsBomb xG, so its footer
    does not carry ``BENCHMARK_NOTE``; the comparison chart, which draws the row, does (both
    themes, through ``model_charts``). The synthetic logo, so a fresh clone passes (run8-ch1
    item 14: ``logo=None`` read the git-ignored real logo)."""
    _forced_head(monkeypatch)
    f, cands = _sb_candidates(xgot_fits)
    assert render.BENCHMARK_NOTE in render.xg_view(f, cands).notes  # precondition
    charts = render.model_charts(f, cands, logo=cc.write_logo(tmp_path / "logo.png"))
    for theme in viz.THEMES:
        cal = " ".join(t.get_text() for t in cc.by_gid(charts[f"calibration-xg-{theme}.png"],
                                                         viz.NOTE_GID))  # fmt: skip
        cmp_ = " ".join(t.get_text() for t in cc.by_gid(charts[f"comparison-xg-{theme}.png"],
                                                         viz.NOTE_GID))  # fmt: skip
        assert cc.collapse(render.BENCHMARK_NOTE) not in cc.collapse(cal)
        assert cc.collapse(render.BENCHMARK_NOTE) in cc.collapse(cmp_)


def test_gain_row_n_boot() -> None:
    """``gain_row`` copies (gain, low, high) of CV and the split; comparisons whose ``n_boot``
    differ raise ``ValueError``."""
    a = metrics.GainCI(0.003, 0.002, 0.004, 2000, 300)
    b = metrics.GainCI(0.001, -0.001, 0.003, 2000, 40)
    row = render.gain_row("Role", "Title", evaluation.Comparison("x", a, b))
    assert (row.role, row.title, row.cv, row.held) == ("Role", "Title", (0.003, 0.002, 0.004),
                                                       (0.001, -0.001, 0.003))  # fmt: skip
    with pytest.raises(ValueError):
        render.gain_row(
            "Role", "Title", evaluation.Comparison("x", a, dataclasses.replace(b, n_boot=1000))
        )


# --- the read-only guard and fits -----------------------------------------------------------------


def test_fits_refuses_no_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """``fits(p, cache=None)`` raises ``ValueError`` before ``xgot.fit`` is called."""
    called: list[object] = []
    monkeypatch.setattr(xgot, "fit", lambda *a, **k: called.append(a))
    with pytest.raises(ValueError):
        render.fits("statsbomb", cache=None)
    assert called == []


def test_read_only_refuses_before_fit(
    xg_data: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside ``read_only()`` a cold cache raises ``CacheMiss`` (a ``ValueError``) at the load,
    before any fit; ``save`` raises; no file is written."""
    dev, held, folds = _split(xg_data)
    pc = cache.PredictionCache(tmp_path / "cache", xg_data.manifest.data_digest)
    calls = _no_fit(monkeypatch)
    Counting.fits = 0
    assert issubclass(render.CacheMiss, ValueError)
    with render.read_only():
        with pytest.raises(render.CacheMiss):
            evaluation.evaluate(COUNTED, dev, held, folds, cache=pc)
        with pytest.raises(render.CacheMiss):
            pc.load(COUNTED.name, "any identity", folds)
        with pytest.raises((render.CacheMiss, ValueError, RuntimeError)):
            pc.save(COUNTED.name, "any identity", folds, None)  # type: ignore[call-arg, arg-type]
    assert calls == [] and Counting.fits == 0 and pc.written == []
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


def test_read_only_restores(xg_data: Any, tmp_path: Path) -> None:
    """``PredictionCache.load`` / ``save`` are restored on exit, after an exception and after
    nesting; outside the guard a cold load returns None again."""
    load, save = cache.PredictionCache.load, cache.PredictionCache.save
    with render.read_only():
        with render.read_only():
            assert cache.PredictionCache.load is not load
        assert cache.PredictionCache.load is not load  # still guarded after the inner exit
    assert cache.PredictionCache.load is load and cache.PredictionCache.save is save
    with pytest.raises(RuntimeError, match="boom"), render.read_only():
        raise RuntimeError("boom")
    assert cache.PredictionCache.load is load and cache.PredictionCache.save is save
    _, _, folds = _split(xg_data)
    pc = cache.PredictionCache(tmp_path / "cache", xg_data.manifest.data_digest)
    assert pc.load(COUNTED.name, "any identity", folds) is None


class _FakeCache:
    def __init__(self, written: list[str]) -> None:
        self.hits: list[str] = ["xgot-gbm"]
        self.written = written


def test_fits_wiring(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``fits``: ``xgot.fit(provider, cache=cache)`` runs inside the guard, ``gsax.fit``
    receives that fit, the pair is returned; a fit whose cache wrote an entry raises."""
    from football_shot_models import gsax

    pcache = cache.PredictionCache
    load = pcache.load
    seen: list[tuple[Any, ...]] = []
    f = types.SimpleNamespace(provider="statsbomb", cache=_FakeCache([]))
    g = object()

    def fake_fit(provider: str, *, cache: Any) -> Any:
        seen.append(("xgot", provider, cache, pcache.load is not load))
        return f

    monkeypatch.setattr(xgot, "fit", fake_fit)
    monkeypatch.setattr(gsax, "fit", lambda x: seen.append(("gsax", x)) or g)
    assert render.fits("statsbomb", cache=tmp_path / "c") == (f, g)
    assert seen[0] == ("xgot", "statsbomb", tmp_path / "c", True)
    assert seen[1] == ("gsax", f)
    f.cache = _FakeCache(["xgot-gbm"])
    with pytest.raises(ValueError):
        render.fits("statsbomb", cache=tmp_path / "c")


def test_candidates_wiring(xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """``candidates(f)`` calls ``models.evaluate_candidates`` inside the guard with the
    provider's candidate specs, ``f.frames``, ``cv_folds(f.frames[0])``, the fit's freeze
    frames and ``f.cache``."""
    import xgot_cases as xc

    providers = xc.patched_providers(xgot, xgot_fits["manifests"], monkeypatch)
    monkeypatch.setattr(xgot, "PROVIDERS", providers)
    f = dataclasses.replace(xgot_fits["fits"]["statsbomb"], cache=_FakeCache([]))
    load = cache.PredictionCache.load
    seen: list[tuple[tuple[Any, ...], dict[str, Any], bool]] = []
    sentinel = object()

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append((args, kwargs, cache.PredictionCache.load is not load))
        return sentinel

    monkeypatch.setattr(models, "evaluate_candidates", spy)
    assert render.candidates(f) is sentinel
    ((args, kwargs, guarded),) = seen
    steps, gbm_spec = xgot.PROVIDERS["statsbomb"].candidate_specs()
    assert guarded
    assert [s.name for s in args[0]] == [s.name for s in steps] and args[1].name == gbm_spec.name
    assert args[2] is f.frames[0] and args[3] is f.frames[1]
    folds = evaluation.cv_folds(f.frames[0])
    assert len(args[4]) == len(folds)
    for (a_tr, a_te), (b_tr, b_te) in zip(args[4], folds, strict=True):
        assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te)
    assert kwargs["freeze_frames"] is f.xg.freeze_frames and kwargs["cache"] is f.cache


# --- CLI ------------------------------------------------------------------------------------------

FAMILIES = (
    "gsax-keepers-light.png", "gsax-keepers-dark.png",
    "calibration-xg-light.png", "calibration-xg-dark.png",
    "calibration-xgot-light.png", "calibration-xgot-dark.png",
    "comparison-xg-light.png", "comparison-xg-dark.png",
    "comparison-xgot-light.png", "comparison-xgot-dark.png",
)  # fmt: skip


@pytest.fixture
def harness(
    repo: Path,
    xgot_fits: dict[str, Any],
    gsax_fits: dict[str, Any],
    logo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """``main``'s inputs on the synthetic StatsBomb xGOT snapshot (``xgot.PROVIDERS`` patched to
    the fixture manifests): a cache root with an (empty) ``xg-gbm`` entry, a checkout with
    ``tree.json``, the hold-out read faked (``holdout_xg``: the kept hold-out rows at xG 0.1),
    names and the logo faked. ``fake_fits`` replaces ``fits`` / ``candidates`` / the views with
    synthetic fits and views (keeper fits from ``keeper_cases``)."""
    import keeper_cases as kc
    import xgot_cases as xc

    from football_shot_models import gsax

    providers = xc.patched_providers(xgot, xgot_fits["manifests"], monkeypatch)
    monkeypatch.setattr(xgot, "PROVIDERS", providers)
    manifest = Path(xgot_fits["manifests"]["statsbomb"])
    raw, root = _layout(repo / "data" / "in", manifest, entry=True, tree=True)

    def holdout(xg: Any, pc: Any, *, spec: Any = models.GBM) -> pl.DataFrame:
        held = data.split_holdout(xg.shots, data.pair_holdout(xg.matches))[1]
        return held.with_columns(xg=pl.lit(0.1, dtype=pl.Float64))

    def names(checkout: Path, ids: Any, *, player_ids: Any) -> pl.DataFrame:
        ids = sorted(set(player_ids))
        return pl.DataFrame({"player_id": ids, "player_name": [f"Player {i[3:]}" for i in ids],
                             "names": pl.Series([1] * len(ids), dtype=pl.Int64)})  # fmt: skip

    monkeypatch.setattr(render, "holdout_xg", holdout)
    monkeypatch.setattr(statsbomb, "player_names", names)
    monkeypatch.setattr(statsbomb, "logo_file", lambda *a, **k: logo)
    xg = data.load_xg_data(manifest)
    held = holdout(xg, None)
    team = str(held.group_by("team_id").len().sort(["len", "team_id"], descending=[True, False])
               ["team_id"][0])  # fmt: skip
    fits_called: list[str] = []

    def fake_fits() -> None:
        sb = kc.fit_from(gsax_fits["fits"]["statsbomb"], kc.rated(kc.TWELVE))
        ws = kc.fit_from(gsax_fits["fits"]["whoscored"],
                         kc.rated([(f"ws:86{s[0][4:]}", *s[1:]) for s in kc.TWELVE]))  # fmt: skip
        pairs = {
            p: (dataclasses.replace(xgot_fits["fits"][p], cache=_FakeCache([])), g)
            for p, g in (("statsbomb", sb), ("whoscored", ws))
        }

        def fits(provider: str, *, cache: Any) -> Any:
            fits_called.append(provider)
            return pairs[provider]

        monkeypatch.setattr(render, "fits", fits)
        monkeypatch.setattr(render, "candidates", lambda f: object())
        monkeypatch.setattr(
            render,
            "xg_view",
            lambda f, c: (
                kc.model_view()
                if f.provider == "statsbomb"
                else kc.model_view(
                    split=("Test season", "the test season", "test-season"), notes=()
                )
            ),
        )
        monkeypatch.setattr(render, "xgot_view", lambda f: kc.xgot_view())
        assert gsax.keeper_rows(sb).height == 12  # precondition

    argv = ["--team", team, "--team-name", "Team One", "--statsbomb-raw", str(raw), "--out",
            "outputs/charts", "--cache", str(root), "--manifest", str(manifest)]  # fmt: skip
    return {"argv": argv, "root": root, "fake_fits": fake_fits, "fits_called": fits_called,
            "repo": repo, "sb_fit": xgot_fits["fits"]["statsbomb"]}  # fmt: skip


def test_main_writes_all_families(harness: dict[str, Any]) -> None:
    """``main`` (``make charts``): package 1's 8 files plus the keeper, calibration and
    comparison charts in both themes: 18 files, one ``chart:`` line each, ``written []`` and
    ``checks: 0 problems``; ``--drop-team-selected`` writes ``gsax-keepers-no-team-{theme}``
    instead. Amended for the docs sites (a ratify item): each run also writes ``charts.json`` (19
    files), whose keys are exactly that run's 18 PNG names, and prints ``titles: <path>``."""
    harness["fake_fits"]()
    rc, out, err = _run(harness["argv"])
    assert rc == 0, err
    charts_dir = harness["repo"] / "outputs" / "charts"
    files = sorted(p.name for p in charts_dir.iterdir())
    assert len(files) == 19 and "charts.json" in files
    pngs = [f for f in files if f.endswith(".png")]
    assert len(pngs) == 18
    assert set(FAMILIES) <= set(files)
    assert sum(f.startswith("team-team-one-") for f in files) == 4
    assert sum(f.startswith("player-") for f in files) == 4
    titles = json.loads((charts_dir / "charts.json").read_text(encoding="utf-8"))
    assert set(titles) == set(pngs)
    assert all(set(v) == {"title", "subtitle"} and v["title"] for v in titles.values())
    assert out.count("chart: ") == 18 and "checks: 0 problems" in out and "written []" in out
    assert out.count("titles: ") == 1
    assert harness["fits_called"] == ["statsbomb"]
    for p in charts_dir.iterdir():
        p.unlink()
    rc, out, err = _run([*harness["argv"], "--drop-team-selected"])
    assert rc == 0, err
    files_set = {p.name for p in charts_dir.iterdir()}
    assert {"gsax-keepers-no-team-light.png", "gsax-keepers-no-team-dark.png"} <= files_set
    assert not {"gsax-keepers-light.png", "gsax-keepers-dark.png"} & files_set
    assert len(files_set) == 19
    titles = json.loads((charts_dir / "charts.json").read_text(encoding="utf-8"))
    assert set(titles) == files_set - {"charts.json"}
    assert "gsax-keepers-no-team-light.png" in titles and "gsax-keepers-light.png" not in titles


def test_main_private(harness: dict[str, Any]) -> None:
    """``--private``: WhoScored only, 10 ``ws-`` files under ``PRIVATE_DIR`` (no ``--out``), no
    package-1 chart; ``--private --out outputs/x`` and ``--private --team`` exit 1 with no
    traceback and write nothing."""
    harness["fake_fits"]()
    root = harness["root"]
    rc, out, err = _run(["--private", "--cache", str(root)])
    assert rc == 0, err
    private = harness["repo"] / render.PRIVATE_DIR
    files = sorted(p.name for p in private.iterdir())
    assert files == sorted(f"ws-{f}" for f in FAMILIES)
    assert harness["fits_called"] == ["whoscored"]
    assert not (harness["repo"] / "outputs").exists()
    for argv in (["--private", "--cache", str(root), "--out", "outputs/x"],
                 ["--private", "--cache", str(root), "--team", "sb:1"]):  # fmt: skip
        rc, _, err = _run(argv)
        assert rc != 0 and "Traceback" not in err
        assert not (harness["repo"] / "outputs" / "x").exists()


@pytest.mark.parametrize("provider", ["statsbomb", "whoscored"])
def test_main_cold_hint(harness: dict[str, Any], provider: str) -> None:
    """A cold prediction cache (real ``fits`` under the guard): exit 1, ``charts: error:`` with
    the provider's ``COLD_HINT`` (StatsBomb ``make xg-models xgot gsax``, WhoScored ``make xg-v1
    xgot gsax``), no traceback, and no keeper or model-card chart written."""
    if provider == "statsbomb":
        argv = harness["argv"]
    else:
        argv = ["--private", "--cache", str(harness["root"])]
    assert [p.name for p in harness["root"].rglob("*.npz")] == [f"{models.GBM.name}.npz"]
    rc, _, err = _run(argv)
    assert rc == 1 and err.startswith("charts: error:") and "Traceback" not in err
    hint = "make xg-models xgot gsax" if provider == "statsbomb" else "make xg-v1 xgot gsax"
    assert "no warm" in err and hint in err, err
    written = [p.name for p in harness["repo"].rglob("*.png")]
    assert not [n for n in written if "gsax-keepers" in n or n.startswith(("calibration", "ws-"))]


def test_manifest_digests_agree(
    harness: dict[str, Any], manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``--manifest`` whose ``data_digest`` differs from ``xgot.PROVIDERS["statsbomb"]``'s
    (its own team, so package 1 could run): exit 1 naming the mismatch (digest or dataset)
    and the ``--manifest`` path, before ``fits`` is called."""
    harness["fake_fits"]()
    argv = list(harness["argv"])
    argv[argv.index("--manifest") + 1] = str(manifest)
    xg = data.load_xg_data(manifest)
    held = data.split_holdout(xg.shots, data.pair_holdout(xg.matches))[1]
    argv[argv.index("--team") + 1] = str(held["team_id"].mode().sort()[0])
    other = snapshot.read_manifest(manifest).data_digest
    d = cache.PredictionCache(harness["root"], other).dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{models.GBM.name}.npz").write_bytes(b"")
    assert other != snapshot.read_manifest(xgot.PROVIDERS["statsbomb"].manifest).data_digest
    rc, _, err = _run(argv)
    assert rc == 1 and err.startswith("charts: error:") and "Traceback" not in err
    assert re.search(r"digest|dataset", err) and str(manifest) in err, err
    assert harness["fits_called"] == []


def test_model_card_footers(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify R20 / R11: the model-card charts credit ``model_text(f.holdout.label, quantity)``
    (xG and xGOT) with the fit's provider; StatsBomb charts get the logo, WhoScored none;
    ``model_text`` keeps package 1's default."""
    from football_shot_charts import diagnostics

    assert render.model_text("Premier League 2015/2016") == (
        "xG: own model, not trained on Premier League 2015/2016"
    )
    assert render.model_text("Test League 2018", "xGOT") == (
        "xGOT: own model, not trained on Test League 2018"
    )
    harness["fake_fits"]()
    seen: list[tuple[str, Any, Any]] = []
    for name in ("calibration_chart", "comparison_chart"):
        real = getattr(diagnostics, name)

        def spy(view: Any, *a: Any, _real: Any = real, _name: str = name, **kw: Any) -> Any:
            seen.append((view.quantity, kw.get("footer"), kw.get("logo")))
            return _real(view, *a, **kw)

        monkeypatch.setattr(diagnostics, name, spy)
    rc, _, err = _run(harness["argv"])
    assert rc == 0, err
    assert {q for q, _, _ in seen} == {"xG", "xGOT"}  # both themes (and any re-render) alike
    label = _sb_label(harness)
    for quantity, footer, logo in seen:
        assert footer.providers == ("statsbomb",) and logo is not None
        assert footer.model == render.model_text(label, quantity)
    seen.clear()
    rc, _, err = _run(["--private", "--cache", str(harness["root"])])
    assert rc == 0, err
    assert {q for q, _, _ in seen} == {"xG", "xGOT"}
    for quantity, footer, logo in seen:
        assert footer.providers == ("whoscored",) and logo is None
        assert footer.model.startswith(f"{quantity}: own model, not trained on ")


def _sb_label(harness: dict[str, Any]) -> str:
    return str(harness["sb_fit"].holdout.label)


def test_main_all_or_nothing(harness: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify R22: ``main`` guards every chart before it writes any (a guard failing on the last
    chart leaves no file and no ``chart:`` line, exit 1); on success the ``cache:`` line is
    printed once, before the first ``chart:`` line."""
    from football_shot_charts import rows as rw

    harness["fake_fits"]()
    rc, out, err = _run(harness["argv"])
    assert rc == 0, err
    lines = out.splitlines()
    cache_at = [i for i, x in enumerate(lines) if x.startswith("cache: ")]
    chart_at = [i for i, x in enumerate(lines) if x.startswith("chart: ")]
    assert len(cache_at) == 1 and len(chart_at) == 18 and cache_at[0] < chart_at[0]
    folder = harness["repo"] / "outputs" / "charts"
    for p in folder.iterdir():
        p.unlink()
    real = rw.guard_write

    def guard(c: Any, path: Any, *, private: bool) -> Any:
        if Path(path).name == "comparison-xgot-dark.png":  # the last chart main writes
            raise ValueError("refused for the test")
        return real(c, path, private=private)

    monkeypatch.setattr(rw, "guard_write", guard)
    rc, out, err = _run(harness["argv"])
    assert rc == 1 and "refused for the test" in err and "Traceback" not in err
    assert "chart: " not in out and "checks: 0 problems" not in out
    assert list(folder.iterdir()) == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--private", "--team-name", "Team One"],
        ["--private", "--player", "sb:5"],
        ["--team-name", "Team One"],
        ["--team", "sb:1"],
    ],  # fmt: skip
)
def test_main_flag_refusals_exit_1(harness: dict[str, Any], extra: list[str]) -> None:
    """Ratify R23: ``--private`` also refuses ``--team-name`` and ``--player``; without
    ``--private`` a missing ``--team`` or ``--team-name`` is checked in code (exit 1 with
    ``charts: error:``, not argparse's exit 2); nothing is fitted or written."""
    harness["fake_fits"]()
    rc, _, err = _run([*extra, "--cache", str(harness["root"])])
    assert rc == 1 and err.startswith("charts: error:") and "Traceback" not in err
    assert harness["fits_called"] == []
    assert not list(harness["repo"].rglob("*.png"))


def test_band_linear_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify R19: the band's percentiles interpolate linearly between the two nearest sorted
    resample rates (``np.quantile``'s default), here by hand at 7 resamples of a bin whose 10
    shots sit in 10 matches, where ``(n − 1)·q`` falls between two distinct rates at both ends."""
    monkeypatch.setattr(metrics, "N_BOOT", 7)
    n = 100
    p = np.linspace(0.02, 0.6, n)  # distinct, in order: bin 4 is shots 40-49
    groups = np.array([f"sb:m{i:02d}" if 40 <= i < 50 else f"sb:o{i % 5}" for i in range(n)])
    y = np.zeros(n, dtype=bool)
    y[[40, 41, 43, 46, 49]] = True
    y[::7] = True
    out = render.calibration_bins(y, p, groups)
    b, level = 4, metrics.GAIN_LEVEL
    mask = (np.arange(n) >= 40) & (np.arange(n) < 50)
    inv, w = metrics.group_weights(groups, 7, metrics.SEED)
    shots = np.bincount(inv[mask], minlength=w.shape[1]).astype(float)
    goals = np.bincount(inv[mask], weights=y[mask].astype(float), minlength=w.shape[1])
    rates = sorted(float(r @ goals) / float(r @ shots) for r in w if float(r @ shots) > 0)
    want = []
    for q in ((1 - level) / 2, (1 + level) / 2):
        pos = (len(rates) - 1) * q
        i = int(pos)
        assert 0 < pos - i < 1 and rates[i] != rates[i + 1]  # precondition: between two values
        want.append(rates[i] + (pos - i) * (rates[i + 1] - rates[i]))
    row = out.row(b, named=True)
    assert row["low"] == pytest.approx(want[0], abs=1e-12)
    assert row["high"] == pytest.approx(want[1], abs=1e-12)


def test_manifest_digest_order_and_message(
    harness: dict[str, Any], manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ratify R24: the digest check runs after the package-1 reads (``holdout_xg`` has been
    called) and before ``fits``; its message names both digests (first 16 hex)."""
    harness["fake_fits"]()
    argv = list(harness["argv"])
    argv[argv.index("--manifest") + 1] = str(manifest)
    xg = data.load_xg_data(manifest)
    held = data.split_holdout(xg.shots, data.pair_holdout(xg.matches))[1]
    argv[argv.index("--team") + 1] = str(held["team_id"].mode().sort()[0])
    other = snapshot.read_manifest(manifest).data_digest
    theirs = snapshot.read_manifest(xgot.PROVIDERS["statsbomb"].manifest).data_digest
    assert other[:16] != theirs[:16]  # precondition
    d = cache.PredictionCache(harness["root"], other).dir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{models.GBM.name}.npz").write_bytes(b"")
    calls: list[object] = []
    monkeypatch.setattr(render, "holdout_xg", _recorder(calls, render.holdout_xg))
    rc, _, err = _run(argv)
    assert rc == 1 and len(calls) == 1 and harness["fits_called"] == []
    assert other[:16] in err and theirs[:16] in err, err


# ---- run8-ch1 assessment fixes -----------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, float("nan")])
def test_bins_refuse_a_null_match(bad: object) -> None:
    """Item 12 (R3): a null or NaN group is refused by name, never resampled as one more match
    (it would narrow the band the title counts against)."""
    y, p, groups = _bin_inputs()
    g = groups.astype(object)
    g[4] = bad
    with pytest.raises(ValueError, match="1 null or NaN key"):
        render.calibration_bins(y.astype(np.int64), p, g)


def test_view_judges_the_level_by_the_band_rule(
    xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item 8: ``_view`` carries ``held_level_ok = metrics.within(gap, SD)`` of the split's
    outcomes against Σ prediction and ``band_sds = metrics.BAND_SDS``, both read when called."""
    f = xgot_fits["fits"]["statsbomb"]
    view = render.xgot_view(f)
    assert view.held_level_ok is metrics.within(view.held_gap, view.held_sd)
    assert view.band_sds == metrics.BAND_SDS
    assert view.held_gap != 0  # precondition: a zero-width band fails it
    monkeypatch.setattr(metrics, "BAND_SDS", 0.0)
    tight = render.xgot_view(f)
    assert tight.held_level_ok is False and tight.band_sds == 0.0
