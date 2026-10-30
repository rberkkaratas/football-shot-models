"""Synthetic shot frames and artist lookups for the ``football_shot_charts`` tests (spec
``runs/CH1/style-shots`` §3, §6).

Frames are built column by column with the C3 dtypes (``SHOT_SCHEMA``) plus ``xg``; every value is
made up on valid football rows (attacking-half locations, a few own-half lobs). Nothing is copied
from a provider file. Shots of one block cycle through fixed points of their zone, so a block's
zone is known by construction.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import numpy as np
import polars as pl
from football_core import schema, viz
from matplotlib import font_manager
from matplotlib.colors import to_hex
from matplotlib.text import Text
from matplotlib.transforms import Bbox

COLUMNS = ("shot_id", "provider", "team_id", "player_id", "x", "y", "is_goal", "is_penalty",
           "period", "body_part")  # fmt: skip
POINTS = {
    "six_yard": ((101.0, 34.0), (102.5, 30.0), (100.5, 38.0), (103.0, 41.0)),
    "rest_of_box": ((94.0, 34.0), (92.0, 25.0), (96.0, 45.0), (90.0, 30.0), (97.5, 20.0)),
    "outside": ((80.0, 34.0), (84.0, 20.0), (78.0, 50.0), (86.0, 60.0)),
}
TEAM, OTHER = "sb:22", "sb:9"
PLAYER = "sb:3814"
CONTEXT = "Test League 2015/2016"
FOOTER = viz.Footer(("statsbomb",), model="xG: own model, not trained on Test League 2015/2016")
BANNED = re.compile(r"skill|clinical|finish|wasteful|lucky|perform", re.IGNORECASE)
# worst-case labels (spec §6): a 22-char name, the longest league label
NAME_22 = "Alexandre Oxlade-Duval"
TEAM_22 = "Borussia Mönchengladba"
LEAGUE = "1. Bundesliga 2015/2016"


def block(
    n: int,
    *,
    xg: float,
    goals: int,
    team: str = TEAM,
    player: str = PLAYER,
    zone: str = "rest_of_box",
    body_part: str | None = "right_foot",
    tag: str = "",
) -> list[dict[str, Any]]:
    """``n`` shots in ``zone`` at ``xg`` each, the first ``goals`` of them scored."""
    pts = POINTS[zone]
    tag = tag or f"{team[3:]}-{player[3:]}-{zone}-{body_part}"
    return [
        {
            "shot_id": f"sb:{tag}-{i:03d}",
            "provider": "statsbomb",
            "team_id": team,
            "player_id": player,
            "x": pts[i % len(pts)][0],
            "y": pts[i % len(pts)][1],
            "is_goal": i < goals,
            "is_penalty": False,
            "period": 1 + i % 2,
            "body_part": body_part,
            "xg": xg,
        }
        for i in range(n)
    ]


def at(x: float, y: float, *, xg: float, goal: bool, sid: str, **kw: Any) -> dict[str, Any]:
    """One shot at a given point (team, player, body part as ``block``'s defaults)."""
    row = block(1, xg=xg, goals=int(goal), tag=sid)[0]
    return {**row, "shot_id": f"sb:{sid}", "x": x, "y": y, **kw}


def frame(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    rows = list(rows)
    cols = {c: pl.Series(c, [r[c] for r in rows], dtype=schema.SHOT_SCHEMA[c]) for c in COLUMNS}
    return pl.DataFrame({**cols, "xg": pl.Series("xg", [r["xg"] for r in rows], dtype=pl.Float64)})


def team_league(
    team_rows: list[dict[str, Any]], other_rows: list[dict[str, Any]]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """``shots`` (the team's rows) and ``league`` (the other rows first, then the team's)."""
    return frame(team_rows), frame(other_rows + team_rows)


def write_logo(path: Path) -> Path:
    """A synthetic two-colour RGBA block standing in for the provider logo."""
    img = np.zeros((10, 60, 4))
    img[..., 3] = 1.0
    img[:, :30, 0] = 1.0
    img[:, 30:, 2] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(path, img)
    return path


# --- Artist lookups ---------------------------------------------------------------------------


def renderer(c: viz.Canvas) -> Any:
    c.fig.canvas.draw()
    return c.fig.canvas.get_renderer()


def texts(c: viz.Canvas) -> list[Text]:
    found = c.fig.findobj(lambda a: isinstance(a, Text))
    return [t for t in found if t.get_visible() and t.get_text().strip()]


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).replace("- ", "-").strip()


def all_text(c: viz.Canvas) -> str:
    return collapse(" ".join(t.get_text() for t in texts(c)))


def theme_of(c: viz.Canvas) -> viz.Theme:
    return c.theme if isinstance(c.theme, viz.Theme) else viz.THEMES[c.theme]


def _weight(w: object) -> int:
    return int(w) if isinstance(w, int | float) else int(font_manager.weight_dict[str(w).lower()])


def styled(c: viz.Canvas, style: str, sizes: Sequence[float] | None = None) -> list[Text]:
    size, weight, token = viz.TEXT_STYLES[style]
    colour = getattr(theme_of(c), token)
    allowed = {float(s) for s in (sizes if sizes is not None else (size,))}
    return [
        t
        for t in texts(c)
        if float(t.get_fontsize()) in allowed
        and to_hex(t.get_color()) == to_hex(colour)
        and _weight(t.get_weight()) == _weight(weight)
    ]


def top_down_text(ts: Sequence[Text], r: Any) -> str:
    ordered = sorted(ts, key=lambda t: -t.get_window_extent(r).y1)
    return collapse(" ".join(t.get_text() for t in ordered))


def title(c: viz.Canvas) -> str:
    r = renderer(c)
    return top_down_text(
        styled(c, "title", (21.0, viz.TITLE_STEP_PT, viz.TITLE_MIN_PT, viz.TITLE_FALLBACK_PT)), r
    )


def body_px(c: viz.Canvas) -> Bbox:
    w, h = (c.fig.get_size_inches() * c.fig.dpi).tolist()
    left, bottom, width, height = c.body
    return Bbox.from_bounds(left * w, bottom * h, width * w, height * h)


def footer_note(c: viz.Canvas) -> str | None:
    """The caption-style text under the body (the footer note), or None when there is none."""
    r = renderer(c)
    body = body_px(c)
    below = [t for t in styled(c, "caption") if t.get_window_extent(r).y1 <= body.y0 + 0.5]
    return top_down_text(below, r) or None


def by_gid(c: viz.Canvas, gid: str) -> list[Any]:
    return c.fig.findobj(lambda a: a.get_gid() == gid)


def gids(c: viz.Canvas, prefix: str) -> set[str]:
    return {
        a.get_gid()
        for a in c.fig.findobj(lambda a: isinstance(a.get_gid(), str))
        if a.get_gid().startswith(prefix)
    }


def centre_x_px(artist: Any, r: Any) -> float:
    """Display x of a marker collection's first offset, or the centre of another artist."""
    if hasattr(artist, "get_offsets") and len(artist.get_offsets()):
        return float(artist.get_offset_transform().transform(artist.get_offsets()[:1])[0, 0])
    e = artist.get_window_extent(r)
    return float((e.x0 + e.x1) / 2)


def data_x_px(artist: Any, value: float) -> float:
    """Display x of the data value ``value`` on the axes the artist is drawn on."""
    return float(artist.axes.transData.transform((value, 0.0))[0])


def fallback(theme: str) -> viz.Theme:
    return dataclasses.replace(viz.THEMES[theme], family=viz.FALLBACK_FAMILY)
