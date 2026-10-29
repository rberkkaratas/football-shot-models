"""``make reproduce``: WhoScored xG v2 on the private snapshot (spec ``runs/PR1/xg-context`` §3.4,
§6.3; marked ``slow`` and ``whoscored``). The regenerated card and README block are byte-equal to
the committed ones, hold no id, and v1's card is untouched."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from football_shot_models import data

REPO = Path(__file__).resolve().parents[1]
START, END = "<!-- v2:summary:start -->", "<!-- v2:summary:end -->"


@pytest.fixture(scope="module")
def real_v2() -> dict[str, Any]:
    manifest = REPO / "manifests" / "whoscored-v1.json"
    if not ((REPO / "data" / "snapshots" / "whoscored-v1").is_dir() and manifest.is_file()):
        pytest.skip("needs the WhoScored snapshot (make snapshot-whoscored)")
    from football_shot_models import cache, v2, v2_cards

    fit = v2.fit(manifest, cache=cache.cache_root(REPO / "data" / "cache"))
    xg = data.load_xg_data(manifest, builder="whoscored")
    return {
        "fit": fit,
        "card": v2_cards.render(fit),
        "block": v2_cards.summary_block(fit),
        "xg": xg,
    }


@pytest.mark.slow
@pytest.mark.whoscored
def test_v2_card_reproduces(real_v2: dict[str, Any]) -> None:
    """§3.4: ``docs/models/xg-v2.md`` is the render of the fit, byte for byte."""
    committed = (REPO / "docs" / "models" / "xg-v2.md").read_text(encoding="utf-8")
    assert real_v2["card"] == committed


@pytest.mark.slow
@pytest.mark.whoscored
def test_v2_readme_block(real_v2: dict[str, Any]) -> None:
    """§3.4: the README holds ``summary_block`` between its markers, once, after the v1 block."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert text.count(START) == 1 and text.count(END) == 1
    assert text.split(START)[1].split(END)[0].strip("\n") == real_v2["block"].strip("\n")
    assert text.index(START) > text.index("<!-- v1:summary:end -->")


@pytest.mark.slow
@pytest.mark.whoscored
def test_v2_card_holds_no_ids(real_v2: dict[str, Any]) -> None:
    """§4 data policy: neither the card nor the block holds a ``ws:`` id, a snapshot match id
    (7 digits) or player id (5-6 digits); every v2 check row 1-6 is computed and printed."""
    xg = real_v2["xg"]
    matches = {m.removeprefix("ws:") for m in xg.matches["match_id"]}
    players = {p.removeprefix("ws:") for p in xg.shots["player_id"]}
    for text in (real_v2["card"], real_v2["block"]):
        assert re.search(r"\bws:", text) is None
        assert not set(re.findall(r"\b\d{7}\b", text)) & matches
        assert not set(re.findall(r"\b\d{5,6}\b", text)) & players
    checks = real_v2["fit"].checks
    assert len(checks) >= 6 and all(c.result in ("met", "missed") for c in checks[:6])
    for c in checks:
        assert c.target in real_v2["card"]
