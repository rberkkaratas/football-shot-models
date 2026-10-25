"""Hard rule 1 for MS4.v1 (spec §6): the WhoScored manifest, the v0 file and the snapshot tables
are git-ignored in this repo, so nothing private can be staged by accident.

One ``git check-ignore -q`` per path, reading only the exit code (0 = ignored); anything else
fails closed. The near misses are committed files next to them (the StatsBomb manifest, the
generated card, ``data/README.md``): a rule that ignores all of ``manifests/`` or ``docs/`` would
hide them.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PRIVATE = (
    "manifests/whoscored-v1.json",
    "data/legacy/xg_model.json",
    "data/snapshots/whoscored-v1/shots.parquet",
)
COMMITTED = ("manifests/statsbomb-open-v1.json", "docs/models/xg-v1.md", "data/README.md")


def _ignored(path: str) -> int:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(
        ["git", "check-ignore", "-q", path], cwd=REPO, env=env, capture_output=True, check=False
    ).returncode


def test_rule1_whoscored_paths_ignored() -> None:
    """§6 Rule 1: each private path is ignored (exit 0); each committed neighbour is not
    (exit 1)."""
    assert {p: _ignored(p) for p in PRIVATE} == dict.fromkeys(PRIVATE, 0)
    assert {p: _ignored(p) for p in COMMITTED} == dict.fromkeys(COMMITTED, 1)
