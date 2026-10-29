"""GSAx on the real snapshots (``make reproduce``; spec ``runs/MS6/gsax`` §5, §7, §8).

``gsax_cards.run_from`` runs on the session ``XgotRun`` per provider (``conftest.real_xgot``, the
one ``test_xgot_run`` uses), so ``make reproduce`` adds only the season-out fits. The cards
must come back byte for byte; the StatsBomb chart, git-ignored under ``data/charts/`` (PM E3
(b)), is written to a tmp path here, and its SHA-256 is on the committed GSAx card, so the
card's byte check covers the chart without any local copy (run6-ms6 architect A2); the README
block (both providers) must match. A second ``gsax.fit`` on the same fit reads every season-out
fit from the cache (0 entries written); a cold run has no cache, which the test names.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import re
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from conftest import REPO, real_xgot

CARDS = {"statsbomb": ("gsax", "gk-model"), "whoscored": ("ws-gsax", "ws-gk-model")}
RESULTS = {1: "met", 2: "met", 3: "met", 4: "met", 6: "met"}


@pytest.fixture(scope="module")
def gsax_real(real: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return {"tmp": tmp_path_factory.mktemp("real-gsax"), "real": real}


def _raw() -> Path:
    from football_shot_models import gsax_cards

    raw = Path(os.environ.get("STATSBOMB_RAW") or gsax_cards.STATSBOMB_RAW)
    return raw if raw.is_absolute() else REPO / raw


def _run(state: dict[str, Any], p: str) -> dict[str, Any]:
    if p not in state:
        from football_shot_models import gsax_cards

        f = real_xgot(state["real"], p)["run"]
        docs = state["tmp"] / p / "models"
        chart = state["tmp"] / p / "gsax-keepers.svg"
        out = io.StringIO()
        with pytest.MonkeyPatch.context() as mp, contextlib.redirect_stdout(out):
            mp.setattr(gsax_cards, "CHART_PATH", chart)
            r = gsax_cards.run_from(f, docs_dir=docs, raw=_raw())
        state[p] = {"run": r, "fit": f, "docs": docs, "chart": chart, "stdout": out.getvalue()}
    return dict(state[p])


def _checks(r: Any) -> None:
    got = {i + 1: c.result for i, c in enumerate(r.checks)}
    assert len(got) == 6
    assert {k: got[k] for k in RESULTS} == RESULTS, [(c.target, c.measured) for c in r.checks]
    assert got[5] in ("met", "n/a"), r.checks[4]


def _warm_refit(f: Any) -> None:
    from football_shot_models import gsax

    if f.cache is None:
        print("cold run: f.cache is None, the second-fit cache assertion does not apply")
        return
    before = len(f.cache.written)
    gsax.fit(f)
    assert len(f.cache.written) == before, "a second gsax.fit wrote cache entries"


def _readme_block() -> str:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    start, end = "<!-- gsax:summary:start -->", "<!-- gsax:summary:end -->"
    assert text.count(start) == 1 and text.count(end) == 1
    return text.split(start)[1].split(end)[0].strip("\n")


@pytest.mark.slow
def test_reproduce_statsbomb(gsax_real: dict[str, Any]) -> None:
    """The StatsBomb GSAx and keeper-model cards and the keeper chart come back byte for byte;
    checks 1-4 and 6 met, check 5 met or n/a; a second fit writes no cache entry."""
    sb = _run(gsax_real, "statsbomb")
    for name in CARDS["statsbomb"]:
        committed = (REPO / "docs" / "models" / f"{name}.md").read_bytes()
        assert (sb["docs"] / f"{name}.md").read_bytes() == committed, name
    digest = hashlib.sha256(sb["chart"].read_bytes()).hexdigest()
    committed = (REPO / "docs" / "models" / "gsax.md").read_text(encoding="utf-8")
    assert f"its SHA-256 is `{digest}`" in committed, "the chart differs from the committed card's"
    _checks(sb["run"])
    _warm_refit(sb["fit"])


@pytest.mark.slow
@pytest.mark.whoscored
def test_reproduce_whoscored(gsax_real: dict[str, Any]) -> None:
    """The WhoScored cards and the README block (both providers) come back byte for byte; no
    WhoScored id, chart or keeper row is written; the checks as for StatsBomb."""
    from football_shot_models import gsax_cards

    sb = _run(gsax_real, "statsbomb")
    ws = _run(gsax_real, "whoscored")
    for name in CARDS["whoscored"]:
        committed = (REPO / "docs" / "models" / f"{name}.md").read_bytes()
        assert (ws["docs"] / f"{name}.md").read_bytes() == committed, name
    assert not ws["chart"].exists()
    assert sorted(x.name for x in ws["docs"].iterdir()) == sorted(
        f"{n}.md" for n in CARDS["whoscored"]
    )
    block = gsax_cards.summary_block([sb["run"], ws["run"]])
    assert _readme_block() == block.strip("\n"), (
        "the README gsax block is stale: `make gsax` prints it; paste it between the "
        "gsax:summary markers"
    )
    for text in (block, *(ws["run"].cards[n] for n in CARDS["whoscored"])):
        assert re.search(r"ws:\d", text) is None
    _checks(ws["run"])
    _warm_refit(ws["fit"])


@pytest.mark.slow
@pytest.mark.parametrize("p", ["statsbomb", pytest.param("whoscored", marks=pytest.mark.whoscored)])
def test_bands_escalated(gsax_real: dict[str, Any], p: str) -> None:
    """PR1 keeper-fixes §9 / PM E1 (a): on the season-out predictions GSAx reads, every judged
    band outside 2 SD is in ``ESCALATED_BANDS[provider]`` (WhoScored: the two ``holdout_fit``
    ``mouth_y_abs`` bands; StatsBomb: none), and the card's band table has one row per
    ``placement_bands`` row with its counts. The gate quotes this test's exit with E1's ruling."""
    from football_shot_models import gsax, gsax_cards

    r = _run(gsax_real, p)
    t = gsax.placement_bands(r["run"])
    failing = {(f, b, s) for f, b, s in t.filter(pl.col("judged") & ~pl.col("within"))
               .select("feature", "band", "source").rows()}  # fmt: skip
    assert failing <= gsax.ESCALATED_BANDS[p], sorted(failing - gsax.ESCALATED_BANDS[p])
    card = (r["docs"] / f"{gsax_cards.CARD_PREFIX[p]}gsax.md").read_text(encoding="utf-8")
    for row in t.iter_rows(named=True):
        assert (f"| {row['feature']} | {row['band']} | {row['source']} | {row['shots']:,} | "
                f"{row['goals']:,} |") in card, row  # fmt: skip
    assert gsax_cards.CARD_PREFIX[p] + "gsax" in r["run"].cards
