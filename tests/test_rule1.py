"""Hard rule 1 for MS4.v1 (spec §6): the WhoScored manifest, the v0 file and the snapshot tables
are git-ignored in this repo, so nothing private can be staged by accident.

One ``git check-ignore -q`` per path, reading only the exit code (0 = ignored); anything else
fails closed. The near misses are committed files next to them (the StatsBomb manifest, the
generated card, ``data/README.md``): a rule that ignores all of ``manifests/`` or ``docs/`` would
hide them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

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


SB_DIR, WS_DIR = "data/snapshots/statsbomb-open-v1", "data/snapshots/whoscored-v1"
WS_MANIFEST, V0 = "manifests/whoscored-v1.json", "data/legacy/xg_model.json"
IGNORED = "data/*\nmanifests/whoscored-*.json\n"  # the repo's two ignore lines
# target, directories, files, .gitignore of a tmp git repo (None: no repo), stderr needle.
# Each case fails exactly one check, so a check that stops failing reaches the real command.
MAKE_CASES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], str | None, str]] = {
    "reproduce, no WhoScored snapshot": ("reproduce", (SB_DIR,), (WS_MANIFEST, V0), None,
                                         "reproduce: no WhoScored snapshot"),
    "reproduce, snapshot without its manifest": ("reproduce", (SB_DIR, WS_DIR), (V0,), None,
                                                 "reproduce: no WhoScored snapshot"),
    "reproduce, no v0": ("reproduce", (SB_DIR, WS_DIR), (WS_MANIFEST,), None,
                         "reproduce: no legacy v0 at data/legacy/xg_model.json"),
    "snapshot-whoscored, no WHOSCORED_ARCHIVE": ("snapshot-whoscored", (), (), IGNORED,
                                                 "WHOSCORED_ARCHIVE must point at"),
    "snapshot-whoscored, manifest not ignored": ("snapshot-whoscored", (), (), "data/*\n",
                                                 f"{WS_MANIFEST} is not git-ignored"),
    "snapshot-whoscored, no git repo": ("snapshot-whoscored", (), (), None,
                                        f"{WS_MANIFEST} is not git-ignored"),
}  # fmt: skip


@pytest.mark.skipif(shutil.which("make") is None or shutil.which("git") is None,
                    reason="make and git are needed")  # fmt: skip
@pytest.mark.parametrize("case", list(MAKE_CASES))
def test_make_whoscored_targets_fail_closed(case: str, tmp_path: Path) -> None:
    """run4-ms4 #2, §6: ``make reproduce`` fails (never skips) without the WhoScored snapshot,
    its manifest or v0; ``make snapshot-whoscored`` exits non-zero without
    ``WHOSCORED_ARCHIVE`` and unless ``git check-ignore`` exits 0 on the manifest (a repo
    without the ignore line, or no repo at all)."""
    target, dirs, files, ignore, needle = MAKE_CASES[case]
    for d in dirs:
        (tmp_path / d).mkdir(parents=True)
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("{}", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("GIT_", "MAKE")) and k != "WHOSCORED_ARCHIVE"}  # fmt: skip
    if ignore is not None:
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, env=env, check=True)
        (tmp_path / ".gitignore").write_text(ignore, encoding="utf-8")
    else:  # no repo: check-ignore must answer 128 here, not 0 from an enclosing repo
        probe = subprocess.run(["git", "rev-parse"], cwd=tmp_path, env=env, capture_output=True)
        assert probe.returncode != 0
    argv = ["make", "-C", str(tmp_path), "-f", str(REPO / "Makefile"), target]
    if "no WHOSCORED_ARCHIVE" not in case and target == "snapshot-whoscored":
        argv.append(f"WHOSCORED_ARCHIVE={tmp_path / 'archive'}")
    out = subprocess.run(argv, capture_output=True, text=True, env=env, check=False)
    assert out.returncode != 0, out.stdout
    assert needle in out.stderr, out.stderr
    assert "uv run" not in out.stdout  # stopped before the recipe's real command
