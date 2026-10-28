"""The pure fit, render and write split and the shared candidate set (spec ``runs/MS6/xgot-ready``
§4, §5, §7 Rule 1, §8).

``xgot.fit(provider, *, cache)`` computes everything for one provider and writes and prints
nothing; ``xgot_cards.render(fit)`` turns it into the card text; ``xgot_cards.run`` is fit, render
and write. ``models.evaluate_candidates`` is the one candidate set of ``models.run``, ``v1.run``
and ``xgot.fit``, so the xGOT run's provider headline is the xG run's headline.

The session fixture ``xgot_fits`` (``conftest``) fits both providers once on the ``xgot_cases``
snapshots. The xG runs (``models.run`` on the StatsBomb snapshot, ``v1.run`` on the WhoScored
one) are stopped right after their ``evaluate_candidates`` call returns: their cards need the
real step specs and fixtures, but the headline rule is ``models.select_headline(c.lrs, c.gbm,
dev, held)`` on what that call returned (§5), and the test checks that both runs pass the same
specs, rows and folds.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import io
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl
import pytest
import v1_cases as vc
import xgot_cases as xg
from conftest import tree

from football_shot_models import ablation, data, features, legacy, models, v1

PROVIDERS = ("statsbomb", "whoscored")
XGOT_FIT_FIELDS = (
    "provider", "setup", "xg", "holdout", "inp", "xg_input", "xg_headline", "evaluations", "head",
    "gbm_gain", "far_gain", "kicks", "penalty_base", "shootouts", "recon", "checks", "split_half",
    "predictions",
)  # fmt: skip


class _Stop(Exception):
    """Raised by the spy once an xG run's candidate set is evaluated."""


def _xgot() -> Any:
    from football_shot_models import xgot

    return xgot


def _cards() -> Any:
    from football_shot_models import xgot_cards

    return xgot_cards


@pytest.fixture(scope="module")
def xg_runs(xgot_fits: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """``models.run`` (StatsBomb) and ``v1.run`` (WhoScored) on the fit fixture's snapshots, with
    the same patched candidate globals, stopped after their ``evaluate_candidates`` returns."""
    root = tmp_path_factory.mktemp("xg-runs")
    manifests = xgot_fits["manifests"]
    v0 = root / "legacy" / "xg_model.json"
    sha = vc.write_v0(v0)
    out: dict[str, Any] = {}
    real = models.evaluate_candidates
    runs = {
        "statsbomb": lambda: models.run(manifests["statsbomb"], docs_dir=root / "docs-sb",
                                        cache=None),
        "whoscored": lambda: v1.run(manifests["whoscored"], v0=v0, docs_dir=root / "docs-ws",
                                    cache=None),
    }  # fmt: skip
    for p, call in runs.items():
        seen: list[tuple[dict[str, Any], Any]] = []

        def spy(*args: Any, _seen: list[Any] = seen, **kwargs: Any) -> Any:
            bound = inspect.signature(real).bind(*args, **kwargs)
            _seen.append((dict(bound.arguments), real(*args, **kwargs)))
            raise _Stop

        with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(io.StringIO()):
            xg.patch_candidates(mp)
            mp.setattr(legacy, "V0_SHA256", sha)
            mp.setattr(models, "evaluate_candidates", spy)
            specs = (ablation.STEPS, models.GBM) if p == "statsbomb" else (v1.STEPS, v1.GBM)
            with pytest.raises(_Stop):
                call()
        out[p] = {"calls": seen, "specs": specs}
    return out


# ---- §4 the pure fit -------------------------------------------------------------------------


def test_fit_is_pure(xgot_fits: dict[str, Any]) -> None:
    """§4, §7 Rule 1: ``fit(p, cache=None)`` prints nothing and writes no file (run from an
    empty working directory: no card, no cache, no data file appears anywhere under the tmp
    root); it returns an ``XgotFit`` with every §4 field and ``penalty`` as a property."""
    xgot = _xgot()
    assert xgot_fits["stdout"] == "" and xgot_fits["stderr"] == ""
    assert xgot_fits["after"] == xgot_fits["before"]
    assert not any((xgot_fits["root"] / "cwd").iterdir())
    for p in PROVIDERS:
        f = xgot_fits["fits"][p]
        assert isinstance(f, xgot.XgotFit)
        assert all(hasattr(f, name) for name in XGOT_FIT_FIELDS)
        assert f.provider == p
        assert f.setup.manifest == str(xgot_fits["manifests"][p])  # PROVIDERS[p] when fit ran
        assert f.penalty is f.kicks.ev
        assert isinstance(inspect.getattr_static(type(f), "penalty"), property)
        assert [e.spec.name for e in f.evaluations] == [
            f"{f.setup.prefix}{n}" for n in ("base-rate", "xg-only", "lr-no-far", "lr", "gbm")
        ]
        assert f.head in f.evaluations[3:]
        assert dataclasses.is_dataclass(f) and type(f).__dataclass_params__.frozen


def test_fit_signature_and_unknown_provider() -> None:
    """§4: ``cache`` is keyword-only and required (no default); an unknown provider raises
    ``ValueError`` naming the allowed keys."""
    xgot = _xgot()
    cache = inspect.signature(xgot.fit).parameters["cache"]
    assert cache.kind is inspect.Parameter.KEYWORD_ONLY
    assert cache.default is inspect.Parameter.empty
    with pytest.raises(ValueError, match="statsbomb") as err:
        xgot.fit("wyscout", cache=None)
    assert "whoscored" in str(err.value) and "wyscout" in str(err.value)


def test_fit_cache_writes_only_under_root(
    xgot_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4, §7: with a cache root, ``fit`` writes only under that root (the opt-in store), prints
    nothing, and gives the predictions of the fit without a cache."""
    xgot = _xgot()
    snapshots = tree(xgot_fits["root"])
    cwd, croot = tmp_path / "cwd", tmp_path / "cache"
    cwd.mkdir()
    providers = xg.patched_providers(xgot, xgot_fits["manifests"], monkeypatch)
    monkeypatch.setattr(xgot, "PROVIDERS", providers)
    monkeypatch.chdir(cwd)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        f = xgot.fit("whoscored", cache=croot)
    assert out.getvalue() == ""
    written = [p.relative_to(tmp_path).parts[0] for p in tmp_path.rglob("*") if p.is_file()]
    assert written and set(written) == {"cache"}
    assert tree(xgot_fits["root"]) == snapshots
    assert f.predictions.equals(xgot_fits["fits"]["whoscored"].predictions)
    # build ratify 1: the cache fit read and wrote is kept on the fit (cold: written, no hits)
    assert f.cache is not None and f.cache.dir.parent == croot
    assert f.cache.written and not f.cache.hits


def test_fit_refuses_off_lattice_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§4: right after the data load, a StatsBomb snapshot with a ``goal_y`` off the 0.1 yd
    lattice (a stale 1.2 snapshot) is refused with the problem and the rebuild command, before
    any candidate is evaluated. (The WhoScored fixture's ``goal_y`` are off the StatsBomb lattice
    and its fit runs: WhoScored rows are ignored.)"""
    xgot = _xgot()

    def nudge(shots: pl.DataFrame) -> pl.DataFrame:
        open_play = pl.col("on_target") & pl.col("shot_id").str.contains(r"-\d\d$")
        first = shots.select(open_play.arg_max()).item()  # an open-play shot of the xG population
        return shots.with_columns(
            pl.when(pl.int_range(pl.len()) == first).then(pl.col("goal_y") + 1e-6)
            .otherwise(pl.col("goal_y")).alias("goal_y")
        )  # fmt: skip

    (tmp_path / "statsbomb").mkdir()
    manifests = {
        "statsbomb": xg.write_xgot_snapshot(tmp_path / "statsbomb", "statsbomb", edit=nudge),
        "whoscored": tmp_path / "unused.json",
    }
    monkeypatch.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, manifests, monkeypatch))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a candidate was evaluated before the lattice guard")

    monkeypatch.setattr(models, "evaluate_candidates", refuse)
    with pytest.raises(ValueError, match="off the 0.1 yd goal-mouth lattice") as err:
        xgot.fit("statsbomb", cache=None)
    assert "1 of" in str(err.value) and "make snapshot OVERWRITE=--overwrite" in str(err.value)


# ---- §4 render and write ---------------------------------------------------------------------


def test_run_is_fit_render_write(
    xgot_fits: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    """§4: ``render(fit)`` is pure (no output, no file) and reads only the fit (called with the
    module globals unpatched); ``run(p)`` returns an ``XgotRun`` (an ``XgotFit`` with ``cards``)
    whose cards equal ``render(fit(p))``, and writes exactly those files; ``write_cards`` writes
    a card per name and returns the directory."""
    xgot, cards = _xgot(), _cards()
    f = xgot_fits["fits"]["statsbomb"]
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    with monkeypatch.context() as mp:  # render reads fit.setup, never the module global
        fake = MappingProxyType({
            k: dataclasses.replace(p, validation="Patched validation.",
                                   text=dataclasses.replace(p.text, source="Patched source"))
            for k, p in xgot.PROVIDERS.items()
        })  # fmt: skip
        mp.setattr(xgot, "PROVIDERS", fake)
        rendered = cards.render(f)
    assert not any("Patched" in text for text in rendered.values())
    assert capsys.readouterr() == ("", "")
    assert not any(cwd.iterdir())
    assert set(rendered) == {"xgot-lr", "xgot-gbm", "xgot-penalty"}
    out = cards.write_cards(rendered, tmp_path / "written")
    assert Path(out) == tmp_path / "written"
    assert tree(tmp_path / "written") == {
        f"{n}.md": text.encode("utf-8") for n, text in rendered.items()
    }
    with monkeypatch.context() as mp:
        mp.setattr(xgot, "PROVIDERS", xg.patched_providers(xgot, xgot_fits["manifests"], mp))
        r = cards.run("statsbomb", docs_dir=tmp_path / "docs", cache=None)
    stdout = capsys.readouterr().out
    assert isinstance(r, cards.XgotRun) and isinstance(r, xgot.XgotFit)
    assert dict(r.cards) == dict(rendered)
    assert tree(tmp_path / "docs") == {
        f"{n}.md": text.encode("utf-8") for n, text in rendered.items()
    }
    assert r.head.spec.name == f.head.spec.name and r.predictions.equals(f.predictions)
    assert f"headline: {r.head.spec.name}" in stdout.splitlines()
    assert not any(cwd.iterdir())  # run writes the cards to docs_dir only
    assert dict(cards.render(r)) == dict(r.cards)  # an XgotRun renders as its fit


# ---- §5 one shared candidate set ---------------------------------------------------------------


def test_candidates_shared(xgot_fits: dict[str, Any], xg_runs: dict[str, Any]) -> None:
    """§5: ``models.evaluate_candidates`` is called by ``models.run``, ``v1.run`` and
    ``xgot.fit`` with that run's own step specs and GBM (the module globals, by identity), with
    no cache in these runs; ``fit`` calls it exactly once per provider. Both the xG run and the
    fit pass the same dev and hold-out rows and folds."""
    for p in PROVIDERS:
        steps, gbm_spec = xgot_fits["specs"][p]
        fit_calls = xgot_fits["calls"][p]
        assert len(fit_calls) == 1, p
        args, result = fit_calls[0]
        assert args["steps"] is steps and args["gbm_spec"] is gbm_spec, p
        assert args.get("cache") is None
        assert isinstance(result, models.Candidates)
        run_args, _ = xg_runs[p]["calls"][0]
        assert len(xg_runs[p]["calls"]) == 1
        run_steps, run_gbm = xg_runs[p]["specs"]
        assert run_args["steps"] is run_steps and run_args["gbm_spec"] is run_gbm, p
        assert [s.name for s in run_args["steps"]] == [s.name for s in args["steps"]]
        assert run_args["gbm_spec"].name == args["gbm_spec"].name
        assert run_args["dev"].equals(args["dev"]) and run_args["held"].equals(args["held"]), p
        assert len(run_args["folds"]) == len(args["folds"])
        for (a_tr, a_te), (b_tr, b_te) in zip(run_args["folds"], args["folds"], strict=True):
            assert np.array_equal(a_tr, b_tr) and np.array_equal(a_te, b_te), p


def _same_make(a: Any, b: Any) -> bool:
    if isinstance(a, functools.partial) and isinstance(b, functools.partial):
        return a.func is b.func and a.args == b.args and a.keywords == b.keywords
    return a is b


@pytest.mark.parametrize("p", PROVIDERS)
def test_xgot_headline_is_xg_run_headline(
    xgot_fits: dict[str, Any], xg_runs: dict[str, Any], p: str
) -> None:
    """§5 reading of "the xGOT input equals the xG card's headline": ``fit(p).xg_headline`` is
    the xG run's headline (``select_headline(c.lrs, c.gbm, dev, held)`` on the candidates the xG
    run evaluated): same spec name, same out-of-fold and hold-out predictions. The input is that
    headline on WhoScored, and ``keeper_neutral(headline)`` on StatsBomb (PM X1 (b)), whose
    precondition is a headline that reads ``open_goal``."""
    xgot = _xgot()
    f = xgot_fits["fits"][p]
    args, c = xg_runs[p]["calls"][0]
    head, _ = models.select_headline(c.lrs, c.gbm, args["dev"], args["held"])
    assert f.xg_headline.spec.name == head.spec.name
    assert np.array_equal(f.xg_headline.oof, head.oof)
    assert np.array_equal(f.xg_headline.holdout_p, head.holdout_p)
    if p == "statsbomb":
        assert "open_goal" in f.xg_headline.spec.features  # precondition: the refit is real
        kn = xgot.keeper_neutral(f.xg_headline.spec)
        assert kn is not f.xg_headline.spec
        assert f.xg_input.spec.name == kn.name
        assert f.xg_input.spec.features == kn.features
        assert not set(f.xg_input.spec.features) & features.KEEPER_POSITION
        assert _same_make(f.xg_input.spec.make, kn.make)
        assert f.xg_input.spec.coefficients is kn.coefficients
    else:
        assert f.xg_input.spec is f.xg_headline.spec
        assert np.array_equal(f.xg_input.oof, f.xg_headline.oof)


def test_holdout_matches_provider(xgot_fits: dict[str, Any]) -> None:
    """§4: ``fit`` keeps the provider's hold-out (StatsBomb: the PL 2015/16 pair; WhoScored: the
    test season) and its ``xg`` data, so package 2 reads both from the fit alone."""
    for p in PROVIDERS:
        f = xgot_fits["fits"][p]
        assert isinstance(f.xg, data.XgData) and isinstance(f.holdout, data.Holdout)
        held = f.xg.matches.filter(f.holdout.expr)
        assert held.height > 0
        names = set(held["season_name"].to_list())
        assert names == ({"2015/2016"} if p == "statsbomb" else set(v1.TEST_SEASONS)), p
