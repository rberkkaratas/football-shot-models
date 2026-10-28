"""Seams for the polish round (spec ``runs/MS6/gsax`` §10): the git dependency switch.

After the switch, football-core is installed from git, not an editable sibling checkout, so no
``src`` module may locate data or code through core's install path or a sibling path.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "football_shot_models"


def test_no_sibling_paths() -> None:
    """No ``src`` module contains ``football_core.__file__`` or ``../football-core`` (the
    detector is checked on hand-written text first)."""
    banned = ("football_core.__file__", "../football-core")
    assert any(b in "root = Path(football_core.__file__).parent" for b in banned)
    assert any(b in 'RAW = "../football-core/data/raw"' for b in banned)
    files = sorted(SRC.rglob("*.py"))
    assert files  # precondition: the package is there
    for path in files:
        text = path.read_text(encoding="utf-8")
        found = [b for b in banned if b in text]
        assert found == [], (path.name, found)
