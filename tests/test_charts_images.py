"""Image baselines of the charts on synthetic fixtures (spec ``runs/CH1/style-shots`` §6; CH1
keeper-model-charts §8: the keeper chart and the model-card visuals from hand-built views):
Inter, the synthetic logo, never StatsBomb data or the StatsBomb logo. To regenerate after an
intended change: ``FSC_UPDATE_BASELINES=1 uv run pytest tests/test_charts_images.py``, then
look at the PNGs in ``tests/baseline_images/charts/``. ``test_glyphs_in_font``: every character
of every text of every chart is in the drawing font's character map (Inter and the fallback)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import charts_cases as cc
import keeper_cases as kc
import polars as pl
import pytest
from football_core import viz
from matplotlib import font_manager, ft2font
from matplotlib.testing.compare import compare_images

from football_shot_charts import shots

BASELINE_DIR = Path(__file__).parent / "baseline_images" / "charts"
IMAGE_TOL = 0.5  # as core's viz baselines


def _frames() -> tuple[object, object]:
    rows = (
        cc.block(14, xg=0.38, goals=6, zone="six_yard", tag="i1")
        + cc.block(52, xg=0.12, goals=8, zone="rest_of_box", body_part="head", tag="i2")
        + cc.block(48, xg=0.10, goals=5, zone="rest_of_box", tag="i3")
        + cc.block(30, xg=0.04, goals=1, zone="outside", body_part="left_foot", tag="i4")
        + [cc.at(102.0, 33.0, xg=0.006, goal=True, sid="i5")]
    )
    other = cc.block(120, xg=0.12, goals=13, team=cc.OTHER, player="sb:9001", tag="i6")
    return cc.team_league(rows, other)


def keeper_view(named: bool = True) -> Any:
    """A hand-built ``KeeperView``: 9 keeper-seasons (invented names), 3 clear, one
    team-selected row shown at the top."""
    from football_shot_charts import keepers

    shrunk = [2.1, 1.6, 1.2, 0.4, 0.1, -0.3, -0.9, -1.3, -1.6]
    half = [1.9, 2.4, 2.5, 2.6, 2.6, 2.6, 2.5, 2.4, 1.5]
    raw = [7.9, 5.0, 4.1, 1.0, 0.2, -0.8, -3.0, -4.2, -6.1]
    shots_ = [140, 118, 125, 130, 127, 110, 102, 160, 185]
    team = [False, True, False, False, False, False, False, False, False]
    comp = ["Alpha League 2015/2016", "Tango Cup 2018", "Bravo Cup 2019"] * 3
    rank = list(range(1, 10))
    flag = f" {keepers.FLAG}"
    if named:
        label = [f"Keeper {chr(65 + i)}{flag if t else ''}" for i, t in enumerate(team)]
        sub = comp
    else:
        label = [f"#{r} · {c}{flag if t else ''}" for r, c, t in zip(rank, comp, team, strict=True)]
        sub = [f"{n} shots" for n in shots_]
    low = [s - h for s, h in zip(shrunk, half, strict=True)]
    high = [s + h for s, h in zip(shrunk, half, strict=True)]
    rows = pl.DataFrame({
        "rank": pl.Series(rank, dtype=pl.Int64), "label": label, "sublabel": sub,
        "team_selected": team, "shots": pl.Series(shots_, dtype=pl.Int64),
        "shrunk": shrunk, "low": low, "high": high, "unshrunk": raw,
        "clear": [lo > 0 or hi < 0 for lo, hi in zip(low, high, strict=True)],
    })  # fmt: skip
    k = int(rows["clear"].sum())
    k_no_team = int((rows["clear"] & ~rows["team_selected"]).sum())
    return keepers.KeeperView(
        rows=rows, n=9, k=k, k_no_team=k_no_team, min_shots=100, level=0.9, spread_100=3.0,
        spread_100_no_team=2.2, at_bound=False, median_width_100=5.0,
        caveat="Not distinguishable from noise at 100 shots: stepped-up split-half r +0.32 "
        "[+0.01, +0.54] on all shots of 111 keepers.",
        team_gap_100=4.8, named=named, mode="flag", left_out=0,
    )  # fmt: skip


def _render(case: str, logo: Path, theme_obj: viz.Theme | None = None) -> viz.Canvas:
    """The chart of one baseline case (``chart-theme[-size]``)."""
    from football_shot_charts import diagnostics, keepers

    chart, theme, *size = case.split("-")
    t: str | viz.Theme = theme_obj if theme_obj is not None else theme
    if chart in ("team", "player"):
        s, league = _frames()
        kw = {"context": cc.CONTEXT, "theme": t, "size": size[0], "footer": cc.FOOTER,
              "excluded": ("1 penalty (1 scored)",), "logo": logo}  # fmt: skip
        if chart == "team":
            return shots.team_chart(s, league, team="Synthetic FC", **kw)
        return shots.player_chart(s, league, player="Test Player", team="Synthetic FC", **kw)
    if chart == "keepers":
        return keepers.keeper_chart(keeper_view(), theme=t,
                                    footer=viz.Footer(("statsbomb",)), logo=logo)  # fmt: skip
    if chart == "keepersws":
        return keepers.keeper_chart(keeper_view(named=False), theme=t,
                                    footer=viz.Footer(("whoscored",)), logo=None)  # fmt: skip
    footer = viz.Footer(("statsbomb",), model="xG: own model, not trained on Test League 2015/2016")
    if chart == "calibration":
        return diagnostics.calibration_chart(kc.model_view(), theme=t, footer=footer, logo=logo)
    if chart == "comparison":
        return diagnostics.comparison_chart(kc.model_view(), theme=t, footer=footer, logo=logo,
                                            size="portrait")  # fmt: skip
    if chart == "xgotcomparison":
        return diagnostics.comparison_chart(kc.xgot_view(), theme=t, footer=footer, logo=logo,
                                            size="square")  # fmt: skip
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    ["team-light-portrait", "team-dark-square", "player-light-square", "player-dark-portrait",
     "keepers-light", "keepers-dark", "calibration-light", "comparison-dark"],
)  # fmt: skip
def test_chart_image(case: str, tmp_path: Path) -> None:
    logo = cc.write_logo(tmp_path / "logo.png")
    c = _render(case, logo)
    assert viz.check(c) == []
    actual = viz.save(c, tmp_path / f"{case}.png")
    baseline = BASELINE_DIR / f"{case}.png"
    if os.environ.get("FSC_UPDATE_BASELINES") == "1":
        baseline.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(actual, baseline)
        pytest.skip(f"baseline written; review it: {baseline}")
    assert baseline.is_file(), f"no baseline at {baseline}; run with FSC_UPDATE_BASELINES=1"
    result = compare_images(str(baseline), str(actual), tol=IMAGE_TOL)
    assert result is None, result


def _charmap(t: Any) -> set[int]:
    path = font_manager.findfont(t.get_fontproperties(), fallback_to_default=False)
    return set(ft2font.FT2Font(path).get_charmap())


@pytest.mark.parametrize("family", ["Inter", "fallback"])
@pytest.mark.parametrize(
    "case", ["keepers-light", "keepersws-dark", "calibration-light", "comparison-dark",
             "xgotcomparison-light"],
)  # fmt: skip
def test_glyphs_in_font(case: str, family: str, tmp_path: Path) -> None:
    """Every character of every text (titles, the flag, U+2212, ·, ş in the footer) is in the
    font the text is drawn with: no missing-glyph boxes in Inter or the fallback."""
    logo = cc.write_logo(tmp_path / "logo.png")
    theme = case.split("-")[1]
    t = viz.THEMES[theme] if family == "Inter" else cc.fallback(theme)
    c = _render(case, logo, t)
    seen = ""
    for text in cc.texts(c):
        cmap = _charmap(text)
        missing = {ch for ch in text.get_text() if not ch.isspace() and ord(ch) not in cmap}
        assert missing == set(), (text.get_text(), missing)
        seen += text.get_text()
    assert "ş" in seen and "·" in seen  # precondition: the footer and a separator were checked
    if case.startswith("keepers"):
        from football_shot_charts import keepers

        assert keepers.FLAG in seen
    if case.startswith(("comparison", "keepers")):
        assert "−" in seen
