"""Render-only and orchestration-only modules in the cache key (spec ``runs/MS6/xgot-ready``
§4.1, §7, §8, §11; ``runs/MS6/gsax`` §3.3).

``cache.source_hash`` skips the files of ``cache.RENDER_ONLY`` (``"<root name>/<relative posix
path>"``), so an edit to card or chart rendering keeps every cached prediction warm. That is safe
only while no module under ``cache.ROOTS`` outside ``RENDER_ONLY`` imports one inside it, which
the AST test enforces (fail closed: a fit-side import of a render-only module would let a stale
prediction through).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def fsm_cache() -> ModuleType:
    return importlib.import_module("football_shot_models.cache")


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    """Two package roots named like the real ones, with a render-only and a fit-side file each,
    and a same-named file one folder down (not render-only: the path is relative to the root)."""
    shot, core = tmp_path / "a" / "football_shot_models", tmp_path / "b" / "football_core"
    (shot / "sub").mkdir(parents=True)
    core.mkdir(parents=True)
    (shot / "xgot.py").write_text("FIT = 1\n")
    (shot / "xgot_cards.py").write_text("CARD = 1\n")
    (shot / "gsax_cards.py").write_text("GSAX_CARD = 1\n")
    (shot / "gsax.py").write_text("ORCHESTRATION = 1\n")
    (shot / "sub" / "xgot_cards.py").write_text("NOT_RENDER_ONLY = 1\n")
    (core / "viz.py").write_text("CHART = 1\n")
    (core / "pitch.py").write_text("PITCH = 1\n")
    return shot, core


def test_source_hash_ignores_render_only(fsm_cache: ModuleType, tmp_path: Path) -> None:
    """Editing, adding or deleting ``football_shot_models/xgot_cards.py``, ``gsax_cards.py``
    (render-only), ``gsax.py`` (orchestration-only, gsax §3.3) or ``football_core/viz.py`` keeps
    the hash; editing a fit-side file, or a same-named file in a sub-folder, changes it."""
    assert frozenset(
        {"football_shot_models/xgot_cards.py", "football_shot_models/gsax_cards.py",
         "football_core/viz.py"}
    ) == fsm_cache.RENDER_ONLY  # fmt: skip
    assert frozenset({"football_shot_models/gsax.py"}) == fsm_cache.ORCHESTRATION_ONLY
    shot, core = _roots(tmp_path)
    roots = [shot, core]
    first = fsm_cache.source_hash(roots)
    (shot / "xgot_cards.py").write_text("CARD = 2\n")
    (shot / "gsax_cards.py").write_text("GSAX_CARD = 2\n")
    (shot / "gsax.py").write_text("ORCHESTRATION = 2\n")
    (core / "viz.py").write_text("CHART = 2\n")
    assert fsm_cache.source_hash(roots) == first
    (shot / "gsax.py").unlink()
    assert fsm_cache.source_hash(roots) == first
    (shot / "xgot_cards.py").unlink()
    assert fsm_cache.source_hash(roots) == first
    (shot / "xgot_cards.py").write_text("CARD = 3\n")
    (shot / "sub" / "xgot_cards.py").write_text("NOT_RENDER_ONLY = 2\n")
    second = fsm_cache.source_hash(roots)
    assert second != first
    (shot / "xgot.py").write_text("FIT = 2\n")
    assert fsm_cache.source_hash(roots) != second
    (core / "pitch.py").write_text("PITCH = 2\n")
    assert fsm_cache.source_hash(roots) != second


def _module(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = [root.name, *rel.parts]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(source: str, module: str, is_package: bool) -> set[str]:
    """Every module a source imports, absolute and relative imports resolved, ``from x import
    y`` counted as both ``x`` and ``x.y``."""
    package = module if is_package else module.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                base = ".".join([*anchor, base] if base else anchor)
            out.add(base)
            out |= {f"{base}.{alias.name}" for alias in node.names}
    return out


def test_render_only_import_one_way(fsm_cache: ModuleType) -> None:
    """§4.1: no module under ``cache.ROOTS`` outside ``RENDER_ONLY`` imports a module in it (by
    any absolute or relative form). The detector is checked on hand-written imports first."""
    render = {k.removesuffix(".py").replace("/", ".") for k in fsm_cache.RENDER_ONLY}
    assert render == {
        "football_shot_models.xgot_cards", "football_shot_models.gsax_cards", "football_core.viz"
    }  # fmt: skip
    for src, mod in (
        ("from football_core import viz", "football_core.goalmouth"),
        ("import football_core.viz as v", "football_core.goalmouth"),
        ("from football_core.viz import draw", "football_core.goalmouth"),
        ("from . import xgot_cards", "football_shot_models.xgot"),
        ("from .xgot_cards import render", "football_shot_models.xgot"),
        ("from ..viz import x", "football_core.io.statsbomb"),
    ):
        assert _imports(src, mod, is_package=False) & render, src
    assert not _imports("from football_core import pitch", "x.y", False) & render
    seen = 0
    for root in fsm_cache.ROOTS:
        for path in sorted(Path(root).rglob("*.py")):
            key = f"{Path(root).name}/{path.relative_to(root).as_posix()}"
            seen += key in fsm_cache.RENDER_ONLY
            if key in fsm_cache.RENDER_ONLY:
                continue
            module = _module(Path(root), path)
            found = _imports(path.read_text(encoding="utf-8"), module,
                             is_package=path.name == "__init__.py") & render  # fmt: skip
            assert not found, f"{key} imports render-only {sorted(found)}"
    assert seen == len(fsm_cache.RENDER_ONLY)  # precondition: both render-only files exist


GSAX_BARRED_MODULES = (
    "football_shot_models.evaluation", "football_shot_models.cache", "football_shot_models.gbm",
    "sklearn",
)  # fmt: skip
GSAX_BARRED_NAMES = ("evaluate", "penalty_kicks")


def test_orchestration_only_rules(fsm_cache: ModuleType) -> None:
    """gsax §3.3: ``gsax.py`` is skipped by ``source_hash``, which is exact only while it never
    fits. No ``src`` module imports ``gsax_cards``; none but ``gsax_cards`` imports ``gsax``;
    ``gsax.py`` imports none of ``evaluation``, ``evaluate``, ``cache``, ``penalty_kicks``,
    ``sklearn``, ``gbm`` (any form), reads no ``evaluate`` / ``penalty_kicks`` attribute and
    calls no ``ModelSpec(``. The detector is checked on hand-written sources first."""
    pkg = next(Path(r) for r in fsm_cache.ROOTS if Path(r).name == "football_shot_models")

    def barred(source: str) -> list[str]:
        found = []
        for mod in _imports(source, "football_shot_models.gsax", is_package=False):
            if any(mod == b or mod.startswith(f"{b}.") for b in GSAX_BARRED_MODULES):
                found.append(mod)
            if mod.rpartition(".")[2] in GSAX_BARRED_NAMES:
                found.append(mod)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Attribute) and node.attr in GSAX_BARRED_NAMES:
                found.append(f".{node.attr}")
            if isinstance(node, ast.Call):
                f = node.func
                name = (
                    f.id
                    if isinstance(f, ast.Name)
                    else f.attr
                    if isinstance(f, ast.Attribute)
                    else ""
                )
                if name == "ModelSpec":
                    found.append("ModelSpec(")
        return found

    for src in (
        "from football_shot_models import evaluation", "from . import cache",
        "from .evaluation import Folds", "from football_shot_models.xgot import penalty_kicks",
        "import sklearn.linear_model", "from football_shot_models import gbm",
        "from .evaluation import evaluate", "x = xgot.evaluate(a)", "s = ev.ModelSpec('a')",
        "s = ModelSpec('a')",
    ):  # fmt: skip
        assert barred(src), src
    assert not barred("from football_shot_models import metrics, xgot\nxgot.season_out(f)")
    source = (pkg / "gsax.py").read_text(encoding="utf-8")
    assert barred(source) == []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        mods = _imports(path.read_text(encoding="utf-8"), _module(pkg, path),
                        is_package=path.name == "__init__.py")  # fmt: skip
        assert "football_shot_models.gsax_cards" not in mods, rel
        if rel != "gsax_cards.py":
            assert "football_shot_models.gsax" not in mods, rel
