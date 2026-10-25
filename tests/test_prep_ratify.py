"""Ratify tests for the Engineer's MS4.prep decisions beyond the spec (``build.md`` "Architect:
ratify" 1-7 and the amended ``real_run`` fixture): one fast test per decision that states
behaviour, each failing if that behaviour is reverted.

Already pinned elsewhere, and pinned here only where the old test left the decision open:
item 2's raise by ``test_data_holdout.py::test_card_context_follows_holdout`` (any
``ValueError``; the message is pinned here), item 6's value and allowed value by
``test_cache.py::test_cache_root`` (a lookahead; the exact text and the case-sensitive, unstripped
match are pinned here). Arrays and ids are invented; everything written goes to ``tmp_path``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import football_core
import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xg_cases as xc

from football_shot_models import baselines, data, evaluation

DIGEST = "0123456789abcdef" + "a" * 48
SOURCE = "fedcba9876543210" + "b" * 48
PL_LABEL = "Premier League 2015/2016"


# ---- 1: a cached entry whose arrays do not fit dev / held fails closed ---------------------------


@pytest.mark.parametrize("field", ["oof", "train", "holdout"])
def test_cache_entry_sizes(
    dev_held: tuple[Any, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    """Ratify 1 (§5.2): a hit whose ``oof`` / ``train`` / ``holdout`` size is not (dev, dev,
    held) raises the cache's own message naming the entry and the sizes, before any fit; the
    same entry with the right sizes is a hit (so the identity matches and only the size check
    can refuse it)."""
    from football_shot_models import cache

    dev, held = dev_held
    folds = evaluation.cv_folds(dev)
    spec = baselines.BASELINES[1]
    key = cache.identity(
        spec.name, spec.features, spec.make().get_params(), dev["shot_id"], held["shot_id"]
    )
    good = {"oof": np.full(dev.height, 0.1), "train": np.full(dev.height, 0.1),
            "holdout": np.full(held.height, 0.1)}  # fmt: skip
    c = cache.PredictionCache(tmp_path, DIGEST, SOURCE)
    c.save(spec.name, key, folds, cache.Predictions(**good, coefficients=()))
    assert c.load(spec.name, key, folds) is not None
    bad = dict(good)
    bad[field] = np.append(bad[field], 0.1)  # one row too many: indexing alone would not fail
    c.save(spec.name, key, folds, cache.Predictions(**bad, coefficients=()))

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fitted on a hit")

    monkeypatch.setattr(evaluation, "_fit", refuse)
    sizes = tuple(bad[f].size for f in ("oof", "train", "holdout"))
    assert sizes != (dev.height, dev.height, held.height)
    message = f"cache: the {spec.name} entry holds {sizes} predictions; delete it"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        evaluation.evaluate(spec, dev, held, folds, cache=c)


# ---- 2: card_context's message -------------------------------------------------------------------


def test_card_context_message(xg_data: Any) -> None:
    """Ratify 2 (§3.3): the refusal counts the wrong rows, names the side, says "not" only for
    held rows outside the hold-out, quotes the label and says what to pass."""
    from football_shot_models import cards

    ho = data.pair_holdout(xg_data.matches)
    dev, held = data.split_holdout(xg_data.shots, ho)
    assert held.height >= 2 and dev.height >= 2
    tail = f"by the hold-out {PL_LABEL!r}; pass the hold-out the frames were split by"
    cases = {
        f"card_context: 2 dev row(s) are held out {tail}": (
            pl.concat([dev, held.head(2)]),
            held.slice(2),
        ),
        f"card_context: 1 held row(s) are not held out {tail}": (
            dev.slice(1),
            pl.concat([held, dev.head(1)]),
        ),
    }
    for message, (d, h) in cases.items():
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            cards.card_context(xg_data, d, h, ho)


# ---- 3: --cache and --no-cache are exclusive ---------------------------------------------------


def test_main_cache_flags_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 3 (§5.3): each flag alone reaches ``run`` (default ``data/cache``, ``--cache DIR``,
    ``--no-cache`` -> ``None``); both together are an argparse usage error (exit 2) and ``run``
    is never called."""
    from football_shot_models import models

    calls: list[object] = []

    def fake_run(manifest: object, **kwargs: Any) -> None:
        calls.append(kwargs["cache"])

    monkeypatch.setattr(models, "run", fake_run)
    assert models.main([]) == 0
    assert models.main(["--cache", "elsewhere"]) == 0
    assert models.main(["--no-cache"]) == 0
    assert calls == ["data/cache", "elsewhere", None]
    err = io.StringIO()
    with contextlib.redirect_stderr(err), pytest.raises(SystemExit) as info:
        models.main(["--cache", "elsewhere", "--no-cache"])
    assert info.value.code == 2
    assert "not allowed with argument" in err.getvalue()
    assert len(calls) == 3


# ---- 4: PenaltyRow / Spread are not re-exported by models -------------------------------------


def test_models_no_penalty_reexport() -> None:
    """Ratify 4 (§1.1): ``models`` imports ``PenaltyRow`` / ``Spread`` for type checking only, so
    at run time they live in ``penalties`` alone; ``ModelsRun`` still annotates with them."""
    from football_shot_models import models, penalties

    for name in ("PenaltyRow", "Spread"):
        assert hasattr(penalties, name)
        assert not hasattr(models, name), name
    assert models.penalties is penalties  # the module, not a name taken from it
    hints = models.ModelsRun.__annotations__
    assert (hints["penalties"], hints["spread"]) == ("tuple[PenaltyRow, ...]", "Spread")


# ---- 5: the GBM card refuses a benchmark -------------------------------------------------------


def test_gbm_card_refuses_benchmark() -> None:
    """Ratify 5 (§2.3): the GBM card's grid comes from a ``ModelSpec``; a benchmark evaluation is
    a ``ValueError`` naming it, not an ``AttributeError`` from ``make``."""
    from football_shot_models import models

    rows = (("max_leaf_nodes", 16, 16), ("iterations", 200, 200),
            ("platt_intercept", 0.0, 0.0), ("platt_slope", 1.0, 1.0))  # fmt: skip
    ev = mc.fake_eval(evaluation.BENCHMARK, coefficients=rows)
    message = f"{evaluation.BENCHMARK.name}: a GBM card needs a ModelSpec, not a benchmark"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        models.CARD.model(ev, mc.card_context())


# ---- 6: FSM_CACHE accepts exactly "", "off" or unset --------------------------------------------


@pytest.mark.parametrize("value", ["no", "OFF", " off", "off\n", "0"])
def test_cache_root_message(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Ratify 6: any value other than unset, "" or "off" (case-sensitive, not stripped) is a
    ``ValueError`` naming the variable, the value and the allowed ones."""
    from football_shot_models import cache

    monkeypatch.setenv("FSM_CACHE", value)
    message = f"FSM_CACHE={value!r}; expected 'off' or unset (or empty)"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        cache.cache_root(Path("data") / "cache")


# ---- 7: ROOTS keep the package names, even through a symlink -----------------------------------


def test_roots_not_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ratify 7: ``ROOTS`` is ``(Path(__file__).parent, Path(*football_core.__path__))`` unresolved,
    so each root's name (hashed into :func:`source_hash`) is the package name even when the
    package directory is reached through a symlink; resolving would hash the target's name."""
    from football_shot_models import cache

    assert (Path(cache.__file__).parent, Path(*football_core.__path__)) == cache.ROOTS
    real_pkg, real_core = tmp_path / "pkg_target", tmp_path / "core_target"
    real_pkg.mkdir()
    real_core.mkdir()
    shutil.copy(cache.__file__, real_pkg / "cache.py")
    (tmp_path / "football_shot_models").symlink_to(real_pkg, target_is_directory=True)
    (tmp_path / "football_core").symlink_to(real_core, target_is_directory=True)
    monkeypatch.setattr(football_core, "__path__", [str(tmp_path / "football_core")])
    name = "_fsm_cache_via_symlink"
    spec = importlib.util.spec_from_file_location(
        name, tmp_path / "football_shot_models" / "cache.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert [r.name for r in module.ROOTS] == ["football_shot_models", "football_core"]
    assert [r.resolve().name for r in module.ROOTS] == ["pkg_target", "core_target"]


# ---- the amended real_run fixture: make reproduce passes the cache -------------------------------


@pytest.mark.parametrize("env", [None, "off"])
def test_real_run_passes_cache(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    env: str | None,
) -> None:
    """Spec §5.3 (the Engineer's amendment of ``test_models_run.py::real_run``): the slow fixture
    passes ``cache_root(REPO / "data" / "cache")`` to ``models.run``, so warm ``make reproduce``
    reads ``data/cache`` and ``make reproduce-cold`` (``FSM_CACHE=off``) passes ``None``."""
    import test_models_run as tmr

    from football_shot_models import models

    repo = tmp_path / "repo"
    (repo / "data" / "snapshots" / xc.SNAPSHOT).mkdir(parents=True)
    monkeypatch.setattr(tmr, "REPO", repo)
    if env is None:
        monkeypatch.delenv("FSM_CACHE", raising=False)
    else:
        monkeypatch.setenv("FSM_CACHE", env)
    seen: dict[str, object] = {}

    def fake_run(manifest: object, **kwargs: Any) -> str:
        seen.update(kwargs)
        return "run"

    monkeypatch.setattr(models, "run", fake_run)
    monkeypatch.setattr(data, "load_xg_data", lambda manifest: SimpleNamespace(shots=None))
    monkeypatch.setattr(data, "split_holdout", lambda shots: ("dev", "held"))
    result = tmr.real_run.__wrapped__(tmp_path_factory)
    assert result["run"] == "run"
    assert "cache" in seen
    assert seen["cache"] == (repo / "data" / "cache" if env is None else None)
