"""Team shot chart and player shot profile (spec ``runs/CH1/style-shots`` §3.1, §6): zones,
lines and their SDs, the league-rate adjustment, verdicts, computed titles, the panels, the note,
input refusals and the machine checks on every chart."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

import charts_cases as cc
import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
from football_core import pitch, viz
from hypothesis import given, settings

from football_shot_charts import rows as rw
from football_shot_charts import shots
from football_shot_models import metrics

ZONE_LABELS = ("Six-yard box", "Rest of the box", "Outside the box")
STAT_LABELS = ["Shots", "Goals", "xG", "Goals − xG"]  # then "xG per shot" / "Goals per shot"


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("logo") / "logo.png")


def team(s: pl.DataFrame, league: pl.DataFrame, logo: Path, **kw: Any) -> viz.Canvas:
    kw.setdefault("team", "Synthetic FC")
    kw.setdefault("context", cc.CONTEXT)
    kw.setdefault("theme", "light")
    kw.setdefault("size", "portrait")
    kw.setdefault("footer", cc.FOOTER)
    return shots.team_chart(s, league, logo=logo, **kw)


def player(s: pl.DataFrame, league: pl.DataFrame, logo: Path, **kw: Any) -> viz.Canvas:
    kw.setdefault("player", "Test Player")
    kw.setdefault("team", "Synthetic FC")
    kw.setdefault("context", cc.CONTEXT)
    kw.setdefault("theme", "light")
    kw.setdefault("size", "portrait")
    kw.setdefault("footer", cc.FOOTER)
    return shots.player_chart(s, league, logo=logo, **kw)


def others(rate_goals: int, *, n: int = 10, zone: str = "rest_of_box") -> list[dict[str, Any]]:
    """Another team's ``n`` shots at 0.5 xG in ``zone`` with ``rate_goals`` goals."""
    return cc.block(n, xg=0.5, goals=rate_goals, team=cc.OTHER, player="sb:9001", zone=zone)


# --- Zones and lines --------------------------------------------------------------------------

EPS = 1e-9
ZONE_CASES = [
    ((pitch.METRE_X[7], 34.0), "six_yard"),  # six-yard line belongs to the six-yard box
    ((pitch.METRE_X[7] - EPS, 34.0), "rest_of_box"),
    ((101.0, pitch.METRE_Y[2]), "six_yard"),
    ((101.0, pitch.METRE_Y[2] - EPS), "rest_of_box"),
    ((101.0, pitch.METRE_Y[5]), "six_yard"),
    ((101.0, pitch.METRE_Y[5] + EPS), "rest_of_box"),
    ((pitch.PITCH_LENGTH, 34.0), "six_yard"),  # on the goal line
    ((pitch.METRE_X[5], 34.0), "rest_of_box"),  # box line belongs to the box
    ((pitch.METRE_X[5] - EPS, 34.0), "outside"),
    ((95.0, pitch.METRE_Y[1]), "rest_of_box"),
    ((95.0, pitch.METRE_Y[1] - EPS), "outside"),
    ((95.0, pitch.METRE_Y[6]), "rest_of_box"),
    ((95.0, pitch.METRE_Y[6] + EPS), "outside"),
    ((30.0, 34.0), "outside"),  # own half
]


def test_zone_boundaries() -> None:
    """IFAB: lines belong to the area; each line probed on and just outside, both sides."""
    assert shots.ZONES == ("six_yard", "rest_of_box", "outside")
    df = pl.DataFrame(
        {"x": [p[0][0] for p in ZONE_CASES], "y": [p[0][1] for p in ZONE_CASES]},
        schema={"x": pl.Float64, "y": pl.Float64},
    )
    got = df.select(shots.zone().cast(pl.String).alias("z"))["z"].to_list()
    assert got == [z for _, z in ZONE_CASES]
    for name, (x, y) in ((n, pts[0]) for n, pts in cc.POINTS.items()):  # fixture points
        one = pl.DataFrame({"x": [x], "y": [y]}, schema={"x": pl.Float64, "y": pl.Float64})
        assert one.select(shots.zone().cast(pl.String)).item() == name


def _mixed() -> tuple[pl.DataFrame, pl.DataFrame]:
    """A goal in every zone; three other teams of unequal size, out of key order."""
    rows = (
        cc.block(7, xg=0.40, goals=3, zone="six_yard")
        + cc.block(23, xg=0.11, goals=4, zone="rest_of_box", body_part="head")
        + cc.block(11, xg=0.03, goals=1, zone="outside", body_part="left_foot")
    )
    other = (
        cc.block(9, xg=0.35, goals=2, team="sb:4", player="sb:41", zone="six_yard")
        + cc.block(17, xg=0.12, goals=3, team="sb:100", player="sb:1001")
        + cc.block(5, xg=0.05, goals=1, team="sb:30", player="sb:301", zone="outside")
        + cc.block(4, xg=0.2, goals=0, team="sb:4", player="sb:42", zone="rest_of_box")
    )
    return cc.team_league(rows, other)


def test_table_sums() -> None:
    """Zone and body-part lines sum to the summary; SDs recomputed by hand; gap = goals − xG."""
    s, league = _mixed()
    total = shots.summary(s, league)
    p = s["xg"].to_numpy()
    assert (total.n, total.goals) == (41, 8)
    assert total.xg == pytest.approx(float(p.sum()))
    assert total.sd == pytest.approx(math.sqrt(float((p * (1 - p)).sum())))
    assert total.gap == pytest.approx(total.goals - total.xg)
    assert total.gap > 0  # precondition: the sign is observable (8 goals from 4.6 xG)
    for by, labels in (("zone", ZONE_LABELS), ("body_part", ("Right foot", "Left foot", "Head"))):
        lines = shots.table(s, league, by=by)
        assert tuple(ln.label for ln in lines) == labels
        assert sum(ln.n for ln in lines) == total.n
        assert sum(ln.goals for ln in lines) == total.goals
        assert sum(ln.xg for ln in lines) == pytest.approx(total.xg)
        assert sum(ln.adj_xg for ln in lines) == pytest.approx(total.adj_xg)
    six = shots.table(s, league, by="zone")[0]
    assert (six.n, six.goals) == (7, 3)
    assert six.sd == pytest.approx(math.sqrt(7 * 0.4 * 0.6))
    assert six.sd == pytest.approx(metrics.poisson_binomial(np.ones(7), np.full(7, 0.4))[1])


def test_adjusted_xg_leaves_own_rows_out() -> None:
    """``r_zone`` = goals / xG of league rows *not in shots* in that zone (own rows left out:
    they convert at a different rate here); r = 1 where the others have no xG; clip at 1."""
    rows = cc.block(20, xg=0.10, goals=10, zone="rest_of_box") + cc.block(
        4, xg=0.40, goals=1, zone="six_yard"
    ) + cc.block(6, xg=0.05, goals=0, zone="outside")  # fmt: skip
    other = (
        cc.block(10, xg=0.30, goals=2, team=cc.OTHER, player="sb:9001")  # rest: r = 2 / 3
        + cc.block(2, xg=0.25, goals=2, team=cc.OTHER, player="sb:9002", zone="six_yard")
    )  # six-yard: r = 4, so 0.4 · 4 = 1.6 clips to 1; outside: no other rows, r = 1
    s, league = cc.team_league(rows, other)
    r = {"six_yard": 2 / 0.5, "rest_of_box": 2 / 3.0, "outside": 1.0}
    with_own = (10 + 2) / (2.0 + 3.0)  # the near miss: own rows inside the rate
    assert not math.isclose(with_own, r["rest_of_box"])
    p_adj = {
        "six_yard": np.minimum(np.full(4, 0.40) * r["six_yard"], 1.0),
        "rest_of_box": np.full(20, 0.10) * r["rest_of_box"],
        "outside": np.full(6, 0.05),
    }
    lines = dict(zip(shots.ZONES, shots.table(s, league, by="zone"), strict=True))
    for z, p in p_adj.items():
        assert lines[z].adj_xg == pytest.approx(float(p.sum())), z
        assert lines[z].sd_adj == pytest.approx(math.sqrt(float((p * (1 - p)).sum()))), z
    total = shots.summary(s, league)
    assert total.adj_xg == pytest.approx(sum(float(p.sum()) for p in p_adj.values()))
    assert shots.league_rate(s, league) == pytest.approx(4 / 3.5)
    assert shots.league_rate(s, s) is None  # no other rows: no league rate


@settings(deadline=None, max_examples=40)
@given(
    spec=st.lists(
        st.tuples(
            st.sampled_from(list(cc.POINTS)),
            st.integers(1, 6),
            st.floats(0.01, 0.95),
            st.sampled_from([*cc.schema.BODY_PARTS, None]),
            st.booleans(),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_lines_property(spec: list[tuple[str, int, float, str | None, bool]]) -> None:
    """On any frame: zone and body-part lines partition the summary; 0 <= adj_xg <= n; gap is
    goals − xG; the verdict is ``too_few`` exactly below ``MIN_JUDGED_SHOTS``."""
    own_rows: list[dict[str, Any]] = []
    other_rows: list[dict[str, Any]] = []
    for i, (zone, n, xg, bp, own) in enumerate(spec):
        own = own or i == 0  # at least one block of the team's own shots
        kw: dict[str, Any] = {} if own else {"team": cc.OTHER, "player": "sb:9001"}
        block = cc.block(n, xg=xg, goals=n // 2, zone=zone, body_part=bp, tag=f"b{i}", **kw)
        (own_rows if own else other_rows).extend(block)
    s, league = cc.team_league(own_rows, other_rows)
    total = shots.summary(s, league)
    for by in ("zone", "body_part"):
        lines = shots.table(s, league, by=by)
        assert sum(ln.n for ln in lines) == total.n == s.height
        assert sum(ln.goals for ln in lines) == total.goals
        assert sum(ln.xg for ln in lines) == pytest.approx(total.xg)
    assert 0.0 <= total.adj_xg <= total.n + 1e-9
    assert total.gap == pytest.approx(total.goals - total.xg)
    assert (shots.verdict(total) == "too_few") is (total.n < metrics.MIN_JUDGED_SHOTS)


# --- Panels -----------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [99, 100])
def test_row_judged_at_100(n: int, logo: Path) -> None:
    """A row of 99 shots is "not judged" and has no band; at 100 the band and its caps are drawn
    and the row carries no caveat."""
    rows = cc.block(n, xg=0.2, goals=n // 5)
    s, league = cc.team_league(rows, others(5))
    c = team(s, league, logo)
    text = cc.all_text(c)
    judged = n >= metrics.MIN_JUDGED_SHOTS
    assert (f"{n} shots · not judged" in text) is not judged
    assert f"{n} shots" in text
    assert bool(cc.by_gid(c, "band-league")) is judged
    assert bool(cc.by_gid(c, "mark-band")) is judged
    assert len(cc.by_gid(c, "mark-xg")) == 1 and not cc.by_gid(c, "mark-league")  # audit A1
    assert cc.by_gid(c, "mark-goals-dot")


def test_band_centres_on_league_rate(logo: Path) -> None:
    """Goals outside xG ± 2 SD but inside the league-rate band: the band is centred on Σ p_adj
    (no tick there: audit A1), the xG tick at xG, the dot at goals; no other track artist."""
    rows = cc.block(100, xg=0.5, goals=61)  # xG 50.0, SD 5.0: 61 is outside [40, 60]
    s, league = cc.team_league(rows, others(6))  # others: 6 goals from 5.0 xG, r = 1.2
    line = shots.summary(s, league)
    assert line.adj_xg == pytest.approx(60.0) and line.sd_adj == pytest.approx(math.sqrt(24.0))
    assert not metrics.within(line.gap, line.sd)  # precondition: outside the raw band
    assert metrics.within(line.goals - line.adj_xg, line.sd_adj)
    c = team(s, league, logo)
    r = cc.renderer(c)
    (band,) = cc.by_gid(c, "band-league")
    (xg_tick,) = cc.by_gid(c, "mark-xg")
    (dot,) = cc.by_gid(c, "mark-goals-dot")
    lo = cc.data_x_px(band, max(0.0, line.adj_xg - metrics.BAND_SDS * line.sd_adj))
    hi = cc.data_x_px(band, line.adj_xg + metrics.BAND_SDS * line.sd_adj)
    e = band.get_window_extent(r)
    assert (e.x0, e.x1) == (pytest.approx(lo, abs=2.0), pytest.approx(hi, abs=2.0))
    assert (e.x0 + e.x1) / 2 == pytest.approx(cc.data_x_px(band, 60.0), abs=2.0)
    assert cc.centre_x_px(xg_tick, r) == pytest.approx(cc.data_x_px(band, 50.0), abs=2.0)
    assert cc.centre_x_px(dot, r) == pytest.approx(cc.data_x_px(band, 61.0), abs=2.0)
    track = {"band-league", "mark-band", "mark-xg", "mark-goals-dot"}
    key = {"mark-shots", "mark-goals", "mark-key", rw.KEY_GID, "band-key"}
    drawn = (cc.gids(c, "mark-") | cc.gids(c, "band-")) - key
    assert drawn <= track, drawn - track


@pytest.mark.parametrize("size", ["portrait", "square"])
def test_one_reference_per_row(size: str, logo: Path) -> None:
    """Audit A1: each panel row draws one xG tick and the goals dot, the league rate only as a
    band (judged rows); a one-line key under the heading names the drawn marks (the band only
    when one is drawn) and no caption explains ticks."""
    rows = cc.block(100, xg=0.2, goals=20) + cc.block(12, xg=0.4, goals=3, zone="six_yard")
    s, league = cc.team_league(rows, others(5))
    for chart, n_rows, band in ((team, 2, True), (player, 1, False)):
        if chart is player:  # the profile's body-part rows: under 100 shots, no band
            s, league = cc.team_league(cc.block(30, xg=0.2, goals=6), others(5))
        c = chart(s, league, logo, size=size)
        assert len(cc.by_gid(c, "mark-xg")) == n_rows and not cc.by_gid(c, "mark-league")
        assert len(cc.by_gid(c, "mark-goals-dot")) == n_rows
        assert len(cc.by_gid(c, "band-league")) == (1 if band else 0)
        r = cc.renderer(c)
        labels = cc.by_gid(c, "panel-key")
        assert [t.get_text() for t in labels] == ["Goals", "xG"] + (
            ["League rate ±2 SD"] if band else [])  # fmt: skip
        assert bool(cc.by_gid(c, "band-key")) is band
        centres = [cc.centre_x_px(t, r) for t in labels]
        assert centres == sorted(centres)  # left to right
        ys = {round(float((t.get_window_extent(r).y0 + t.get_window_extent(r).y1) / 2))
              for t in labels}  # fmt: skip
        assert max(ys) - min(ys) <= 1  # one line
        heading = next(t for t in cc.styled(c, "heading") if "Goals and xG" in t.get_text())
        rails = cc.by_gid(c, "track-rail")
        assert heading.get_window_extent(r).y0 > max(ys) > max(
            a.get_window_extent(r).y1 for a in rails)  # fmt: skip
        assert "tick" not in cc.all_text(c)
        assert viz.check(c) == []


def test_stat_strip_per_shot(logo: Path) -> None:
    """Audit A2: the strip is Shots, Goals, xG, Goals − xG (U+2212, note "SD"), then xG per shot
    on the team chart and goals per shot on the profile, at 2 dp; the league's goals per xG is
    no tile."""
    rows = cc.block(30, xg=0.5, goals=11)  # 11 goals from 15.0 xG: gap −4.0, SD 2.7
    other = cc.block(20, xg=0.5, goals=9, team=cc.OTHER, player="sb:9001")  # 9 / 10.0 = 0.90
    s, league = cc.team_league(rows, other)
    for chart, last, value in ((team, "xG per shot", "0.50"), (player, "Goals per shot", "0.37")):
        c = chart(s, league, logo)
        strip = [t.get_text() for t in cc.styled(c, "stat_label")]
        assert strip == [*STAT_LABELS, last]
        assert [t.get_text() for t in cc.styled(c, "stat_value")][-1] == value
        text = cc.all_text(c)
        assert "−4.0" in text and "SD 2.7" in text and "-4.0" not in text  # never a hyphen
        assert "League goals per xG" not in text
    with pytest.raises(ValueError):
        shots.stats(s, league, per="shots")  # type: ignore[arg-type]


LEVELS = [
    # other team's goals of 40 shots at 0.5 xG (Σ 20.0, SD √10 ≈ 3.16): the sentence
    (20, "The model is in line with the rest of the league: 1.00 goals per xG"),
    (13, "The model over-predicts the rest of the league: 0.65 goals per xG"),
    (27, "The model under-predicts the rest of the league: 1.35 goals per xG"),
]


@pytest.mark.parametrize(("goals", "expected"), LEVELS)
def test_model_level_note(goals: int, expected: str, logo: Path) -> None:
    """Audit A2: the model's level on the rest of the league is a footer sentence after the
    model credit on both charts, its verb from ``metrics.within`` (7 goals off: outside 2 SD);
    no rest, no sentence. The not-shown note stays as it was. Its subject is "The model" after
    the credit too (run8-ch1 item 20: "It" could be read as the team)."""
    other = cc.block(40, xg=0.5, goals=goals, team=cc.OTHER, player="sb:9001")
    s, league = cc.team_league(cc.block(10, xg=0.3, goals=3), other)
    assert shots.model_level(s, league) == expected
    for chart in (team, player):
        c = chart(s, league, logo)
        assert credit(c) == f"Data: StatsBomb · {cc.FOOTER.model}. {expected}"
        assert cc.footer_note(c) is None
    assert credit(team(s, league, logo, footer=viz.Footer(("statsbomb",)))) == (
        f"Data: StatsBomb · {expected}"  # no model credit: the sentence names the model
    )
    assert shots.model_level(s, s) is None
    assert credit(team(s, s, logo)) == f"Data: StatsBomb · {cc.FOOTER.model}"


def credit(c: viz.Canvas) -> str:
    return cc.top_down_text(cc.styled(c, "footer"), cc.renderer(c))


def test_panel_layout(logo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Portrait: the panel under the map, full content width; square: the panel right of the
    map; the legend under the pitch; a pitch below ``MIN_PITCH_PX`` raises."""
    s, league = _mixed()
    for size in ("portrait", "square"):
        c = team(s, league, logo, size=size)
        r = cc.renderer(c)
        panels = [a for a in c.fig.axes if str(a.get_gid() or "").startswith("panel-")]
        maps = [a for a in c.fig.axes if any(x.get_gid() == "mark-shots" for x in a.collections)]
        assert len(panels) == 1 and len(maps) == 1, size
        p, m = panels[0].get_window_extent(r), maps[0].get_window_extent(r)
        if size == "portrait":
            assert p.y1 <= m.y0 and p.width == pytest.approx(1080 - 2 * viz.MARGIN_PX, abs=2)
        else:
            assert p.x0 >= m.x1
        keys = cc.by_gid(c, "mark-key")
        assert keys and all(k.get_window_extent(r).y1 <= m.y0 + 0.5 for k in keys)
    monkeypatch.setattr(shots, "MIN_PITCH_PX", 2000)
    with pytest.raises(ValueError):
        team(s, league, logo, size="square")


# --- Titles -----------------------------------------------------------------------------------

TEAM_TITLES = [
    # (team goals of 100 at 0.5 xG, title); others convert at exactly 1 goal per xG
    (61, "61 non-penalty goals, 11.0 more than the model's 50.0 xG (SD 5.0)"),
    (60, "60 of 60 non-penalty goals came from inside the box (rest of league: 100%)"),
    (40, "40 of 40 non-penalty goals came from inside the box (rest of league: 100%)"),
    (39, "39 non-penalty goals, 11.0 fewer than the model's 50.0 xG (SD 5.0)"),
]


@pytest.mark.parametrize(("goals", "expected"), TEAM_TITLES)
def test_team_statements(goals: int, expected: str, logo: Path) -> None:
    """``beyond`` states the gap with its SD ("more" iff gap > 0, ``|gap|`` printed); ``within``
    at its exact boundary (gap = ±2 SD in float) is not beyond; g = 0 has its own sentence."""
    rows = cc.block(100, xg=0.5, goals=goals)
    s, league = cc.team_league(rows, others(5))
    line = shots.summary(s, league)
    assert (line.xg, line.sd) == (50.0, 5.0)  # exact in float: Σ 0.5 and √Σ 0.25
    assert shots.verdict(line) == ("beyond" if abs(goals - 50) > 10 else "noise")
    if abs(goals - 50) == 10:
        assert abs(line.gap) == metrics.BAND_SDS * line.sd  # precondition: on the boundary
    assert cc.title(team(s, league, logo)) == expected


def test_team_no_goal_statement(logo: Path) -> None:
    s, league = cc.team_league(cc.block(5, xg=0.1, goals=0), others(5))
    assert cc.title(team(s, league, logo)) == "5 non-penalty shots, 0.5 xG and no goal"


def _rank_league(target: list[tuple[str, bool]]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Target team's shots (zone, goal) plus three scorers out of key order (box shares 50 %,
    67 %, 100 % ... see below) and a team without a goal (its 0 % must not count)."""
    rows = [
        cc.at(*cc.POINTS[z][0], xg=0.2, goal=g, sid=f"t{i}", team="sb:30", player="sb:301")
        for i, (z, g) in enumerate(target)
    ]
    other = (
        cc.block(1, xg=0.3, goals=1, team="sb:4", player="sb:41", zone="rest_of_box", tag="a")
        + cc.block(1, xg=0.1, goals=1, team="sb:4", player="sb:41", zone="outside", tag="b")
        + cc.block(2, xg=0.3, goals=2, team="sb:100", player="sb:1001", tag="c")
        + cc.block(1, xg=0.1, goals=1, team="sb:100", player="sb:1001", zone="outside", tag="d")
        + cc.block(4, xg=0.1, goals=0, team="sb:7", player="sb:71", zone="outside", tag="e")
    )  # sb:4 1 of 2 (50 %), sb:100 2 of 3 (67 %), sb:7 no goal; rest of league 3 of 5 = 60 %
    return cc.team_league(rows, other)


RANKS = [
    ([("rest_of_box", True)] * 3 + [("outside", False)] * 2,
     "3 of 3 non-penalty goals came from inside the box, the league's highest share "
     "(rest of league: 60%)"),
    ([("six_yard", True), ("outside", True), ("outside", True)],
     "1 of 3 non-penalty goals came from inside the box, the league's lowest share "
     "(rest of league: 60%)"),
    ([("six_yard", True), ("rest_of_box", True), ("outside", True)],
     "2 of 3 non-penalty goals came from inside the box (rest of league: 60%)"),  # ties sb:100
]  # fmt: skip


@pytest.mark.parametrize(("target", "expected"), RANKS, ids=["highest", "lowest", "tied"])
def test_box_share_title_rank(target: list[tuple[str, bool]], expected: str, logo: Path) -> None:
    """Rank words only when strictly highest / lowest among teams with >= 1 goal (computed per
    team_id); the league share is from the other rows."""
    s, league = _rank_league(target)
    assert shots.verdict(shots.summary(s, league)) == "too_few"  # the box-share branch
    assert cc.title(team(s, league, logo)) == expected


def test_box_share_without_other_goals(logo: Path) -> None:
    """No other team scored: the parenthesis is omitted."""
    rows = cc.block(3, xg=0.3, goals=2)
    s, league = cc.team_league(rows, cc.block(4, xg=0.1, goals=0, team=cc.OTHER, player="sb:91"))
    got = cc.title(team(s, league, logo))
    assert got.startswith("2 of 2 non-penalty goals came from inside the box")
    assert "rest of league" not in got


PLAYER_CASES = [
    # (n shots at 0.5 xG, goals, other team's goals from 5.0 xG, verdict, title)
    (99, 50, 5, "too_few", "50 non-penalty goals from 49.5 xG on 99 shots: too few to judge"),
    (100, 52, 5, "noise", "52 non-penalty goals from 50.0 xG: in line with the model (SD 5.0)"),
    (100, 61, 6, "zone_bias", "61 non-penalty goals from 50.0 xG: in line with the league's rate "
     "from the same zones (SD 4.9)"),
    (100, 61, 5, "beyond", "61 non-penalty goals, 11.0 more than the model's 50.0 xG (SD 5.0)"),
]  # fmt: skip


@pytest.mark.parametrize(("n", "goals", "other", "verdict", "expected"), PLAYER_CASES)
def test_player_verdicts(
    n: int, goals: int, other: int, verdict: str, expected: str, logo: Path
) -> None:
    """too_few below 100 shots; noise inside 2 SD; zone_bias inside 2 SD of the league-rate
    expectation (prints ``sd_adj`` and the panel's adjusted sentence); beyond otherwise."""
    s, league = cc.team_league(cc.block(n, xg=0.5, goals=goals), others(other))
    assert shots.verdict(shots.summary(s, league)) == verdict
    c = player(s, league, logo)
    assert cc.title(c) == expected
    sentence = "At the league's rate per zone these shots are worth 60.0 goals."
    assert (sentence in cc.all_text(c)) is (verdict == "zone_bias")


def test_singulars(logo: Path) -> None:
    """R13: counts of 1 are singular ("1 non-penalty goal", "1 shot"). (R12, the league tick's
    shade word, went with the tick: audit A1.)"""
    s, league = cc.team_league(cc.block(1, xg=0.3, goals=1), others(5))
    assert cc.title(player(s, league, logo)) == (
        "1 non-penalty goal from 0.3 xG on 1 shot: too few to judge"
    )


def test_player_panel_body_part(logo: Path) -> None:
    """Body-part rows in ``BODY_PARTS`` order with labels, "Not recorded" for null last; only
    groups present; the "not judged" caveat on rows under 100 shots only."""
    rows = (
        cc.block(2, xg=0.1, goals=1, body_part=None, tag="n")
        + cc.block(3, xg=0.2, goals=1, body_part="head", tag="h")
        + cc.block(100, xg=0.15, goals=15, tag="r")
        + cc.block(2, xg=0.05, goals=0, body_part="other", tag="o")
    )
    s, league = cc.team_league(rows, others(5))
    assert shots.BODY_PART_LABELS == {"right_foot": "Right foot", "left_foot": "Left foot",
                                      "head": "Head", "other": "Other"}  # fmt: skip
    lines = shots.table(s, league, by="body_part")
    assert [ln.label for ln in lines] == ["Right foot", "Head", "Other", "Not recorded"]
    assert [ln.n for ln in lines] == [100, 3, 2, 2]
    text = cc.all_text(player(s, league, logo))
    assert "Goals and xG by body part" in text
    assert "100 shots" in text and "100 shots · not judged" not in text
    assert "3 shots · not judged" in text and "2 shots · not judged" in text
    assert "Left foot" not in text


def test_player_two_teams(logo: Path) -> None:
    """A player with rows for two teams: ``player_chart`` refuses the mix, ``team_chart`` refuses
    two teams."""
    rows = cc.block(5, xg=0.1, goals=1) + cc.block(4, xg=0.1, goals=1, team=cc.OTHER, tag="x")
    s = cc.frame(rows)
    league = cc.frame(rows + others(5))
    with pytest.raises(ValueError):
        player(s, league, logo)
    with pytest.raises(ValueError):
        team(s, league, logo)


def test_no_skill_words(logo: Path) -> None:
    """No title, subtitle, caption or label says skill, clinical, finish, wasteful, lucky or
    perform, over every title branch of both charts; every gap printed carries its SD."""
    frames = [cc.team_league(cc.block(100, xg=0.5, goals=g), others(o))
              for g, o in ((61, 5), (39, 5), (60, 5), (52, 6), (61, 6))]  # fmt: skip
    frames.append(cc.team_league(cc.block(99, xg=0.5, goals=50), others(5)))
    frames.append(cc.team_league(cc.block(5, xg=0.1, goals=0), others(5)))
    for s, league in frames:
        for chart in (team, player):
            text = cc.all_text(chart(s, league, logo))
            assert not cc.BANNED.search(text), cc.BANNED.search(text)
            if " more than the model's " in text or " fewer than the model's " in text:
                assert "(SD " in text
            assert "SD " in text  # the strip's Goals − xG note


# --- Notes, refusals, checks ------------------------------------------------------------------


def test_not_drawn_note(logo: Path) -> None:
    """Shots outside the drawn area (own half here) are counted with goals and xG after the
    caller's exclusion parts; no part, no note."""
    rows = cc.block(20, xg=0.1, goals=2) + [
        cc.at(20.0, 30.0, xg=0.04, goal=True, sid="lob-1"),
        cc.at(40.0, 50.0, xg=0.22, goal=True, sid="lob-2"),
    ]
    s, league = cc.team_league(rows, others(5))
    c = team(s, league, logo, excluded=("3 penalties (2 scored)",))
    assert cc.footer_note(c) == (
        "Not shown: 3 penalties (2 scored) · 2 shots outside the drawn area (2 goals, 0.3 xG)"
    )
    s, league = cc.team_league(cc.block(20, xg=0.1, goals=2), others(5))
    c = team(s, league, logo)
    assert cc.footer_note(c) is None and "Not shown" not in cc.all_text(c)


def _fault(df: pl.DataFrame, fault: str, row: int) -> pl.DataFrame:
    def put(col: str, value: Any, dtype: Any = None) -> pl.DataFrame:
        values = df[col].to_list()
        values[row] = value
        return df.with_columns(pl.Series(col, values, dtype=dtype or df.schema[col]))

    if fault == "penalty":
        return put("is_penalty", True)
    if fault == "period-5":
        return put("period", 5)
    if fault in {"xg-nan", "xg-1.2", "xg-null", "xg-inf"}:
        value = {"xg-nan": float("nan"), "xg-1.2": 1.2, "xg-null": None, "xg-inf": math.inf}
        return put("xg", value[fault])
    if fault.startswith("null-"):
        return put(fault[5:], None)
    if fault == "x-float32":
        return df.with_columns(pl.col("x").cast(pl.Float32))
    if fault == "body-part-string":
        return df.with_columns(pl.col("body_part").cast(pl.String))
    if fault == "missing-xg":
        return df.drop("xg")
    if fault == "x-nan":
        return put("x", float("nan"))
    if fault == "duplicate-id":
        return put("shot_id", df["shot_id"][row - 1])
    if fault == "two-providers":
        return put("provider", "whoscored")
    raise AssertionError(fault)


FAULTS = ["penalty", "period-5", "xg-nan", "xg-1.2", "xg-null", "xg-inf", "null-is_goal",
          "null-is_penalty", "null-period", "null-provider", "null-team_id", "x-float32", "x-nan",
          "body-part-string", "missing-xg", "duplicate-id", "two-providers"]  # fmt: skip


@pytest.mark.parametrize("fault", FAULTS)
@pytest.mark.parametrize("side", ["shots", "league"])
def test_bad_input_raises(side: str, fault: str, logo: Path) -> None:
    """Both frames go through one validation; the bad row in ``league`` is another team's row,
    so the league check alone must catch it."""
    rows = cc.block(10, xg=0.2, goals=2)
    other = others(5)
    s, league = cc.team_league(rows, other)
    assert team(s, league, logo) is not None  # precondition: the clean pair draws
    if side == "shots":
        s = _fault(s, fault, 3)  # the league keeps the clean row: only the shots check can fail
    else:
        assert league["team_id"][3] == cc.OTHER  # row 3 is another team's (others come first)
        league = _fault(league, fault, 3)
    with pytest.raises(ValueError):
        team(s, league, logo)


@pytest.mark.parametrize("case", ["empty-shots", "id-not-in-league", "two-teams"])
def test_bad_pair_raises(case: str, logo: Path) -> None:
    s, league = cc.team_league(cc.block(10, xg=0.2, goals=2), others(5))
    if case == "empty-shots":
        s = s.head(0)
    elif case == "id-not-in-league":
        league = league.filter(pl.col("shot_id") != s["shot_id"][0])
    else:
        s = pl.concat([s, league.filter(pl.col("team_id") == cc.OTHER).head(1)])
    with pytest.raises(ValueError):
        team(s, league, logo)


def _worst_case(verdict: str = "beyond") -> tuple[pl.DataFrame, pl.DataFrame]:
    """150 shots (3-digit counts), 30 goals from 45.0 xG (a negative 2-digit gap), every zone and
    body part with a null, two own-half shots; three other teams. A judged row (>= 100 shots) on
    both panels beside unjudged ones (integrate census: the band and its caps were drawn on no
    worst-case chart). ``verdict="zone_bias"``: the other teams score at about 0.69 goals per xG,
    so the player takes the longest caption (the adjusted sentence) and the team the box-share
    title; its null body parts are headers instead (StatsBomb records a body part on every shot,
    so a publishable profile has <= 4 rows; the 5-row zone_bias profile, which raised
    ``NoLayout`` before the one-line panel key, is pinned in
    ``test_charts_ratify::test_strip_dropped_when_no_layout_fits``)."""
    rows = (
        cc.block(28, xg=0.3, goals=10, zone="six_yard", tag="w1")
        + cc.block(100, xg=0.3, goals=15, zone="rest_of_box", body_part="left_foot", tag="w2")
        + cc.block(10, xg=0.3, goals=3, zone="outside", body_part="head", tag="w3")
        + cc.block(6, xg=0.3, goals=1, zone="outside", body_part=None if verdict == "beyond"
                   else "head", tag="w4")
        + cc.block(4, xg=0.3, goals=1, zone="rest_of_box", body_part="other", tag="w5")
        + [cc.at(20.0, 30.0, xg=0.3, goal=False, sid="w6"),
           cc.at(30.0, 40.0, xg=0.3, goal=False, sid="w7")]
    )  # fmt: skip
    low = verdict == "zone_bias"
    other = (
        cc.block(120, xg=0.12, goals=10 if low else 15, team="sb:4", player="sb:41", tag="o1")
        + cc.block(80, xg=0.2, goals=11 if low else 16, team="sb:100", player="sb:1001",
                   zone="six_yard", tag="o2")
        + cc.block(60, xg=0.04, goals=2, team="sb:30", player="sb:301", zone="outside", tag="o3")
    )  # fmt: skip
    return cc.team_league(rows, other)


WORST_KW: dict[str, Any] = {
    "context": cc.LEAGUE, "excluded": ("13 penalties (10 scored)", "2 direct corners (1 scored)"),
    "footer": viz.Footer(("statsbomb",),
                         model="xG: own model, not trained on Women's Super League 2018/2019"),
}  # fmt: skip


def _worst_chart(chart: str, verdict: str, logo: Path, **kw: Any) -> viz.Canvas:
    s, league = _worst_case(verdict)
    line = shots.summary(s, league)
    assert line.n == 150 and line.gap == pytest.approx(-15.0)  # precondition
    assert shots.verdict(line) == verdict  # precondition: the branch the variant names
    judged = [t for t in shots.table(s, league, "zone" if chart == "team" else "body_part")
              if t.n >= metrics.MIN_JUDGED_SHOTS]  # fmt: skip
    assert len(judged) == 1  # precondition: a judged row beside unjudged ones
    if chart == "team":
        return team(s, league, logo, team=cc.TEAM_22, **WORST_KW, **kw)
    return player(s, league, logo, player=cc.NAME_22, team=cc.TEAM_22, **WORST_KW, **kw)


@pytest.mark.parametrize("verdict", ["beyond", "zone_bias"])
@pytest.mark.parametrize("family", ["inter", "fallback"])
@pytest.mark.parametrize("size", ["portrait", "square"])
@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("chart", ["team", "player"])
def test_every_chart_passes_checks(
    chart: str, theme: str, size: str, family: str, verdict: str, logo: Path
) -> None:
    """Worst-case labels (22-char names, the longest league label, 3-digit counts, a negative
    2-digit gap, a 3-part note, the longest caption), Inter and the fallback family:
    ``viz.check`` finds nothing; the judged row's band is drawn."""
    c = _worst_chart(chart, verdict, logo, theme=theme if family == "inter" else cc.fallback(theme),
                     size=size)  # fmt: skip
    assert viz.check(c) == []
    assert cc.by_gid(c, "band-league") and cc.by_gid(c, "mark-band")  # the judged row drawn
    assert cc.footer_note(c) is not None and "2 shots outside the drawn area" in cc.all_text(c)
    adjusted = "At the league's rate per zone these shots are worth"
    assert (adjusted in cc.all_text(c)) is (chart == "player" and verdict == "zone_bias")
    want = f"{cc.NAME_22} · {cc.TEAM_22} · {cc.LEAGUE}" if chart == "player" else (
        f"{cc.TEAM_22} · {cc.LEAGUE}")  # fmt: skip
    r = cc.renderer(c)
    assert cc.top_down_text(cc.styled(c, "subtitle"), r) == want


def _layout_branches(c: viz.Canvas) -> set[str]:
    """The layout branches a chart took, read from its artists: panel under or beside the map,
    wide or stacked rows, stat strip kept or dropped, a legend label split in two lines."""
    r = cc.renderer(c)
    pitch_ax = cc.by_gid(c, "pitch")[0].get_window_extent(r)
    panel = next(a for a in c.fig.axes if a.get_gid() in ("panel-zones", "panel-body-parts"))
    track = cc.by_gid(c, "track")[0].get_window_extent(r)
    box = panel.get_window_extent(r)
    return {
        "under" if box.y1 <= pitch_ax.y0 else "side",
        "wide" if track.x0 > box.x0 + rw.PANEL_PAD_PX + 1 else "stacked",
        "strip" if cc.styled(c, "stat_value") else "no-strip",
        *(["legend-split"] if any("\n" in t.get_text() for t in cc.styled(c, "legend")) else []),
    }


def test_worst_case_reaches_every_layout_branch(logo: Path) -> None:
    """Integrate census: across the worst-case matrix each layout branch the checks must see is
    taken at least once (else ``test_every_chart_passes_checks`` would pass vacuously on it)."""
    seen: set[str] = set()
    for chart, size, family, verdict in itertools.product(
        ("team", "player"), ("portrait", "square"), ("inter", "fallback"), ("beyond", "zone_bias")
    ):
        theme = "light" if family == "inter" else cc.fallback("light")
        seen |= _layout_branches(_worst_chart(chart, verdict, logo, theme=theme, size=size))
    assert seen == {"under", "side", "wide", "stacked", "strip", "no-strip", "legend-split"}


# ---- run8-ch1 assessment fixes -----------------------------------------------------------------


@pytest.mark.parametrize(("rest_goals", "team_n", "shown"), [(80, 48, True), (98, 48, False),
                                                             (80, 100, False)])  # fmt: skip
def test_unjudged_row_gets_the_league_rate(
    rest_goals: int, team_n: int, shown: bool, logo: Path
) -> None:
    """Item 9: a zone row under ``MIN_JUDGED_SHOTS`` (no band) in a zone the model misprices on
    the rest of the league gets a caption with its shots at the league's rate there
    (``Line.adj_xg``: 15.36 xG x 80/100 = 12.3 goals, Leicester's six-yard case); none when the
    rest is inside its band (98 of 100) or the row is judged."""
    team_rows = cc.block(team_n, xg=0.32, goals=9, zone="six_yard") + cc.block(
        120, xg=0.1, goals=12, zone="rest_of_box")  # fmt: skip
    rest = cc.block(200, xg=0.5, goals=rest_goals, team=cc.OTHER, player="sb:9001",
                    zone="six_yard") + cc.block(100, xg=0.1, goals=10, team=cc.OTHER,
                                               player="sb:9001")  # fmt: skip
    s, league = cc.team_league(team_rows, rest)
    # precondition: the rest's six-yard gap is outside 2 SD at 80 goals (z 2.8), inside at 98
    assert (shots.mispriced_zones(s, league) == ("Six-yard box",)) is (rest_goals == 80)
    six = next(ln for ln in shots.table(s, league, "zone") if ln.label == "Six-yard box")
    c = team(s, league, logo)
    assert viz.check(c) == []
    sentence = f"Six-yard box {six.adj_xg:.1f} goals"
    assert (sentence in cc.all_text(c)) is shown
    if shown:
        assert six.adj_xg == pytest.approx(0.32 * team_n * rest_goals / 100)
        assert "Six-yard box 12.3 goals" in cc.all_text(c)
