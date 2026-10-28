"""``xgot_cards``: the render-side module of the xGOT run (spec ``runs/MS6/xgot-ready`` §4, §8).

``summary_block`` reads each fit's own ``setup`` (never the module global ``xgot.PROVIDERS``)
and the checks title constant; ``python -m football_shot_models.xgot`` refuses to run and points
to ``xgot_cards``, and ``make xgot`` runs ``xgot_cards``. The fits come from the session fixture
``xgot_fits`` (``conftest``).
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_summary_block_reads_setup(
    xgot_fits: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """§4: with ``xgot.PROVIDERS`` patched after ``fit`` to providers whose source reads
    otherwise, the README block still names each fit's ``setup.text.source``; the count line is
    ``"{met} of {n} checks met (MS5, MS6)"`` and no ``MS5 checks`` literal is left."""
    from football_shot_models import xgot, xgot_cards

    fits = [xgot_fits["fits"][p] for p in ("statsbomb", "whoscored")]
    sources = [f.setup.text.source for f in fits]
    assert len(set(sources)) == 2  # precondition: the two providers' sources differ
    fake = MappingProxyType({
        k: dataclasses.replace(p, text=dataclasses.replace(p.text, source=f"Patched {k}"))
        for k, p in xgot.PROVIDERS.items()
    })  # fmt: skip
    monkeypatch.setattr(xgot, "PROVIDERS", fake)
    block = xgot_cards.summary_block(fits)
    assert "Patched" not in block
    for f, source in zip(fits, sources, strict=True):
        assert f"| {source} | " in block
        met = sum(c.result == "met" for c in f.checks)
        assert f"{met} of {len(f.checks)} checks met (MS5, MS6)" in block
    assert "MS5 checks" not in block


def test_xgot_module_cli_points_to_cards(tmp_path: Path) -> None:
    """§4: ``python -m football_shot_models.xgot`` exits 2 with ``fsm: error: run python -m
    football_shot_models.xgot_cards`` on stderr and no traceback, so the old command never
    succeeds silently; ``make xgot`` runs ``xgot_cards`` (both providers)."""
    done = subprocess.run(
        [sys.executable, "-m", "football_shot_models.xgot"],
        cwd=tmp_path, capture_output=True, text=True, check=False, timeout=300,
    )  # fmt: skip
    assert done.returncode == 2
    assert "fsm: error: run python -m football_shot_models.xgot_cards" in done.stderr
    assert "Traceback" not in done.stderr + done.stdout
    assert not any(tmp_path.iterdir())
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    recipe = makefile.split("\nxgot:\n", 1)[1].split("\n\n", 1)[0]
    assert "python -m football_shot_models.xgot_cards --provider both" in recipe
    assert "football_shot_models.xgot " not in recipe
