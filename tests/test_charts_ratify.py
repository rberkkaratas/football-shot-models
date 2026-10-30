"""Pins for the Engineer's ratify items of CH1.style-shots that no earlier test held (``runs/CH1/
style-shots/build.md``, R8-R11, R13-R16, R19), and the fallback-font title gap QA found while
pinning them."""

from __future__ import annotations

import subprocess
from pathlib import Path

import charts_cases as cc
import polars as pl
import pytest
import test_charts_shots as tcs
from football_core import viz

from football_shot_charts import render, shots
from football_shot_models import metrics

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("logo") / "logo.png")


def _panel(c: viz.Canvas) -> object:
    return next(a for a in c.fig.axes if a.get_gid() in ("panel-zones", "panel-body-parts"))


# --- R8: the pitch shrinks to the capped depth --------------------------------------------------


@pytest.mark.parametrize("size", ["portrait", "square"])
def test_pitch_shrinks_to_capped_depth(size: str, logo: Path) -> None:
    """R8: shots 33 m out on a narrow spread: the aspect fill would draw deeper than the 35 m
    cap; the pitch axes shrink so the view stops at the cap (and the 33 m shot is drawn). A wide
    spread the cap cannot reach shrinks the pitch to ``MIN_PITCH_PX``, never under."""
    deep = [cc.at(72.0, 34.0, xg=0.05, goal=False, sid="d0"),
            cc.at(99.0, 30.0, xg=0.3, goal=True, sid="d1"),
            cc.at(99.0, 38.0, xg=0.3, goal=False, sid="d2")]  # fmt: skip
    s, league = cc.team_league(deep, tcs.others(5))
    c = tcs.team(s, league, logo, size=size)
    r = cc.renderer(c)
    (ax,) = cc.by_gid(c, "pitch")
    assert ax.get_ylim()[0] == pytest.approx(105.0 - viz.CROP_MAX_DEPTH)
    assert ax.get_window_extent(r).height >= shots.MIN_PITCH_PX - 0.5
    assert cc.footer_note(c) is None  # every shot drawn
    wide = [cc.at(96.0, y, xg=0.1, goal=i == 0, sid=f"w{i}")
            for i, y in enumerate((2.0, 66.0, 34.0, 20.0, 50.0))]  # fmt: skip
    s, league = cc.team_league(wide, tcs.others(5))
    c = tcs.team(s, league, logo, size=size)
    r = cc.renderer(c)
    (ax,) = cc.by_gid(c, "pitch")
    assert ax.get_window_extent(r).height == pytest.approx(shots.MIN_PITCH_PX, abs=0.5)
    assert viz.check(c) == []


# --- R9, R10, R19: placement, side width, strip dropped ----------------------------------------


def _five_rows(verdict: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``tcs._worst_case`` with the w4 rows' body part null: 5 body-part rows (only a WhoScored
    profile can have a null body part)."""
    s, league = tcs._worst_case(verdict)
    null = pl.when(pl.col("shot_id").str.contains("w4")).then(None).otherwise(pl.col("body_part"))
    s = s.with_columns(null.alias("body_part"))
    rest = league.filter(~pl.col("shot_id").is_in(s["shot_id"].implode()))
    return s, pl.concat([rest, s])


def test_portrait_panel_beside_map_when_under_is_too_tall(logo: Path) -> None:
    """R9: portrait tries the panel under the map first (a short panel goes there) and puts it
    beside the map when under it the pitch would be under ``MIN_PITCH_PX`` (5 body-part rows)."""
    s, league = cc.team_league(cc.block(30, xg=0.2, goals=6), tcs.others(5))
    short = tcs.player(s, league, logo, size="portrait")
    assert {"under", "wide", "strip"} <= tcs._layout_branches(short)
    s, league = _five_rows("zone_bias")
    tall = tcs.player(s, league, logo, size="portrait", player=cc.NAME_22, team=cc.TEAM_22,
                      **tcs.WORST_KW)  # fmt: skip
    assert len(shots.table(s, league, "body_part")) == 5  # precondition
    assert {"side", "strip"} <= tcs._layout_branches(tall)
    assert viz.check(tall) == []


def test_side_panel_takes_the_narrowest_width(logo: Path) -> None:
    """R9: beside the map the panel takes the narrowest inner width (``PANEL_STEP_PX`` steps
    from ``MIN_TRACK_PX``) that holds its heading, its key and its widest label and value blocks
    side by side; the map gets the rest."""
    s, league = cc.team_league(cc.block(30, xg=0.2, goals=6), tcs.others(5))
    c = tcs.team(s, league, logo, size="square")
    assert "side" in tcs._layout_branches(c)  # precondition
    lines = shots.table(s, league, "zone")
    least = metrics.MIN_JUDGED_SHOTS
    labels = [(ln.label, f"{ln.n} shots" + (" · not judged" if ln.n < least else ""))
              for ln in lines]  # fmt: skip
    values = [(f"{ln.goals} goals", f"{ln.xg:.1f} xG") for ln in lines]

    def block(pairs: list[tuple[str, str]]) -> float:
        return max(max(c.measure(a, "label")[0], c.measure(b, "caption")[0]) for a, b in pairs)

    items = shots.key(lines)
    key_w = sum(shots.KEY_SLOT_PX[g] + shots.KEY_TEXT_GAP_PX + c.measure(t, "caption")[0]
                for g, t in items) + shots.KEY_ITEM_GAP_PX * (len(items) - 1)  # fmt: skip
    need = max(c.measure(shots.ZONE_HEADING, "heading")[0], key_w,
               block(labels) + shots.INNER_GAP_PX + block(values), shots.MIN_TRACK_PX)  # fmt: skip
    r = cc.renderer(c)
    inner = _panel(c).get_window_extent(r).width - 2 * shots.PANEL_PAD_PX  # type: ignore[attr-defined]
    steps = (inner - shots.MIN_TRACK_PX) / shots.PANEL_STEP_PX
    assert steps == pytest.approx(round(steps), abs=0.01)  # on the step grid
    assert inner - shots.PANEL_STEP_PX < need <= inner


def test_strip_dropped_when_no_layout_fits(logo: Path) -> None:
    """R10 (and R19, now obsolete): a 5-row zone_bias profile on the square canvas has no layout
    with the stat strip; it is drawn without it (the title states goals, xG and the SD) and
    passes every check. R19's ``NoLayout`` no longer happens since the one-line panel key (A1)."""
    s, league = _five_rows("zone_bias")
    assert shots.verdict(shots.summary(s, league)) == "zone_bias"  # precondition
    for theme in ("light", cc.fallback("dark")):
        c = tcs.player(s, league, logo, size="square", theme=theme, player=cc.NAME_22,
                       team=cc.TEAM_22, **tcs.WORST_KW)  # fmt: skip
        assert "no-strip" in tcs._layout_branches(c)
        assert cc.styled(c, "stat_value") == []
        assert "(SD " in cc.title(c) and " xG" in cc.title(c)
        assert viz.check(c) == []
    portrait = tcs.player(s, league, logo, size="portrait", player=cc.NAME_22, team=cc.TEAM_22,
                          **tcs.WORST_KW)  # fmt: skip
    assert "strip" in tcs._layout_branches(portrait)  # only when nothing fits with it


# --- R11: the note and the crop settle --------------------------------------------------------


def test_note_and_crop_settle_in_three_passes(logo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R11: a note that never matches the crop is retried at most 3 times, then raises; a note
    that changes once (shots outside the drawn area) settles on the second pass."""
    s, league = cc.team_league(cc.block(20, xg=0.1, goals=2), tcs.others(5))
    made: list[int] = []
    canvas = viz.canvas

    def counting(*args: object, **kw: object) -> viz.Canvas:
        made.append(1)
        return canvas(*args, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(viz, "canvas", counting)
    tcs.team(s, league, logo)
    assert len(made) == 1  # no shot outside the view: the first pass settles
    made.clear()
    lob = [cc.at(20.0, 30.0, xg=0.04, goal=True, sid="lob")]
    s2, league2 = cc.team_league(cc.block(20, xg=0.1, goals=2) + lob, tcs.others(5))
    tcs.team(s2, league2, logo)
    assert len(made) == 2  # the counted note differs once
    made.clear()
    turns = iter(range(100))
    monkeypatch.setattr(shots, "_note", lambda *a: f"Not shown: pass {next(turns)}")
    with pytest.raises(ValueError, match="3 passes"):
        tcs.team(s, league, logo)
    assert len(made) == 3


# --- R13: singular counts, a zero gap ---------------------------------------------------------


def test_singular_counts_and_zero_gap(logo: Path) -> None:
    """R13: "1 goal", "1 shot" in the panel and the note; a gap that rounds to zero prints
    ``0.0`` (no sign, no U+2212)."""
    rows = [cc.at(101.0, 34.0, xg=0.34, goal=False, sid="a"),
            cc.at(94.0, 34.0, xg=0.34, goal=False, sid="b"),
            cc.at(20.0, 30.0, xg=0.34, goal=True, sid="c")]  # fmt: skip
    s, league = cc.team_league(rows, tcs.others(5))
    assert shots.summary(s, league).gap == pytest.approx(-0.02)  # precondition: rounds to 0.0
    c = tcs.team(s, league, logo)
    gap = cc.styled(c, "stat_value")[3].get_text()
    assert gap == "0.0"
    labels = [t.get_text() for t in cc.texts(c)]
    assert "1 goal" in labels and "1 shot · not judged" in labels
    assert not any(x.startswith(("1 goals", "1 shots")) for x in labels)
    assert "1 shot outside the drawn area (1 goal, 0.3 xG)" in (cc.footer_note(c) or "")


# --- R14: the track axes ----------------------------------------------------------------------


def test_track_axes_in_goal_units(logo: Path) -> None:
    """R14: one track axes per panel, x in goals from 0 to the largest of goals, xG and the
    league-rate expectation (``adj_xg``, unjudged rows included); one decor rail per row (no
    ``mark-`` gid, so the checks treat it as decoration)."""
    rows = cc.block(10, xg=0.1, goals=0, zone="six_yard") + cc.block(5, xg=0.2, goals=1)
    other = (cc.block(10, xg=0.1, goals=3, team=cc.OTHER, player="sb:91", zone="six_yard")
             + tcs.others(5))  # fmt: skip
    s, league = cc.team_league(rows, other)
    lines = shots.table(s, league, "zone")
    assert [ln.n for ln in lines] == [10, 5] and lines[0].adj_xg == pytest.approx(3.0)
    assert all(ln.n < metrics.MIN_JUDGED_SHOTS for ln in lines)  # precondition: unjudged
    c = tcs.team(s, league, logo)
    (track,) = cc.by_gid(c, "track")
    assert track.get_xlim() == pytest.approx((0.0, 3.0))
    ticks = cc.by_gid(c, "mark-xg")
    assert sorted(t.get_xdata()[0] for t in ticks) == pytest.approx([1.0, 1.0])
    assert all(t.axes is track for t in ticks)
    rails = cc.by_gid(c, "track-rail")
    assert len(rails) == len(lines) and all(r.axes is track for r in rails)


# --- R15, R16: the CLI and write ----------------------------------------------------------------


def test_excluded_note_holdout_keyword() -> None:
    """R15: ``excluded_note(..., holdout=)`` counts the rows of the pair it is given."""
    from test_charts_render import _excl_frames

    shots_table, exclusions = _excl_frames()
    other_pair = ("sb:9", "sb:1")
    assert render.excluded_note(shots_table, exclusions, team=cc.TEAM, holdout=other_pair) == (
        "1 penalty (1 scored)",
    )
    assert len(render.excluded_note(shots_table, exclusions, team=cc.TEAM)) == 5


def test_cli_player_without_make_variable(capsys: pytest.CaptureFixture[str]) -> None:
    """R15 (PM E2): ``--player`` stays on the CLI, the Makefile has no player variable, and the
    StatsBomb cache defaults to core's ``DEFAULT_CACHE`` (no sibling path in the code)."""
    with pytest.raises(SystemExit) as info:
        render.main(["--help"])
    assert info.value.code == 0
    assert "--player" in capsys.readouterr().out
    make = (REPO / "Makefile").read_text()
    assert "CHART_PLAYER" not in make and "--player" not in make
    from football_core.io import statsbomb

    assert render.STATSBOMB_RAW == statsbomb.DEFAULT_CACHE


def test_write_refuses_any_whoscored_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, logo: Path
) -> None:
    """R16: a chart is WhoScored when its text credits WhoScored, even beside StatsBomb."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("data/\noutputs/\n")
    monkeypatch.chdir(tmp_path)
    both = viz.Footer(("statsbomb", "whoscored"), model="xG: own model")
    c = viz.canvas("light", "square", title="T", subtitle="S", footer=both, logo=logo)
    target = tmp_path / "outputs" / "charts" / "mixed.png"
    with pytest.raises(ValueError, match="private"):
        render.write(c, target)
    assert not target.exists()
    sb = viz.canvas("light", "square", title="T", subtitle="S", logo=logo,
                    footer=viz.Footer(("statsbomb",)))  # fmt: skip
    assert render.write(sb, target).is_file()


# --- QA finding: the fallback family and the longest title branches ---------------------------


def _zone_bias_3_digits() -> tuple[pl.DataFrame, pl.DataFrame]:
    """400 shots, 120 goals from 100.0 xG (outside 2 SD 8.7), the rest of the league at 1.2 goals
    per xG: inside the league-rate band. 3-digit goals and xG in the zone_bias title."""
    other = cc.block(100, xg=0.5, goals=60, team=cc.OTHER, player="sb:91")
    return cc.team_league(cc.block(400, xg=0.25, goals=120), other)


@pytest.mark.parametrize("branch", ["team-rank", "player-zone-bias"])
def test_fallback_draws_the_longest_titles(branch: str, logo: Path) -> None:
    """The fallback family must draw every title branch the charts can produce.

    QA major (CH1 ratify), fixed at the gate (G1): DejaVu Sans is ~16% wider than Inter, so the
    box-share title with its rank and a 3-digit zone_bias title needed 3 lines at TITLE_MIN_PT;
    the fallback family alone steps on to ``viz.TITLE_FALLBACK_PT``.
    """
    theme = cc.fallback("light")
    if branch == "team-rank":
        s, league = tcs._rank_league(tcs.RANKS[0][0])
        assert cc.title(tcs.team(s, league, logo)) == tcs.RANKS[0][1]  # precondition: Inter
        c = tcs.team(s, league, logo, theme=theme)
        assert cc.title(c) == tcs.RANKS[0][1]
    else:
        s, league = _zone_bias_3_digits()
        assert shots.verdict(shots.summary(s, league)) == "zone_bias"  # precondition
        tcs.player(s, league, logo)  # precondition: Inter draws it
        c = tcs.player(s, league, logo, theme=theme)
    assert viz.check(c) == []
