"""Render and write the goalkeeper cards (P3.1-P3.3) and the StatsBomb keeper chart.

:func:`render` turns one provider's :class:`~football_shot_models.gsax.GsaxFit` into its two
cards (``gsax`` and ``gk-model``, ``ws-`` prefixed for WhoScored) as Markdown: no fit, no
write, no print. :func:`chart` draws the StatsBomb keeper forest plot (with the StatsBomb logo)
as SVG text. :func:`run_from` is fit, names, render, write and the printed summary; ``python -m
football_shot_models.gsax_cards --provider both`` (``make gsax``) runs it after ``make xgot``.

Rule 1: the WhoScored cards and the README block hold counts and metrics only (no keeper, no
id); keeper names appear only on the StatsBomb card and chart (``NAMED``), from the lineup
files. The chart embeds the StatsBomb logo, which is never committed, so it is written to the
git-ignored ``CHART_PATH`` (PM E3 (b)) and checked by ``make reproduce``.

Render-only for the prediction cache (``cache.RENDER_ONLY``): no other module imports it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import matplotlib
import numpy as np
import polars as pl
from football_core import viz
from football_core.io import snapshot, statsbomb
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from football_shot_models import gsax, metrics, season_out, xgot, xgot_cards
from football_shot_models.cards import fmt, signed
from football_shot_models.gsax import (
    CI_LEVEL,
    KEEPER_SEASON,
    MIN_TIER_KEEPERS,
    SEASON_KEY,
    TEAM_SELECTED,
    WITHOUT,
    GsaxFit,
    NextSeason,
    Tier,
)
from football_shot_models.gsax import gain_text as _gain
from football_shot_models.metrics import Check, GainCI
from football_shot_models.xgot_cards import split_half_cell

StrPath = str | os.PathLike[str]

CARD_PREFIX: Final[Mapping[str, str]] = MappingProxyType({"statsbomb": "", "whoscored": "ws-"})
NAMED: Final[frozenset[str]] = frozenset({"statsbomb"})  # providers whose cards name keepers
CHART_FILE: Final[str] = "data/charts/gsax-keepers.svg"  # git-ignored: it embeds the logo
CHART_PATH: StrPath = CHART_FILE  # where run_from writes it, read when called (tests move it)
# the open-data cache holding the lineups; this repo's own by default (core's loader default),
# so src assumes no sibling checkout: the Makefile passes football-core's (STATSBOMB_RAW)
STATSBOMB_RAW: Final[str] = statsbomb.DEFAULT_CACHE
VERSION: Final[str] = "v2"
CHANGELOG: Final[tuple[str, ...]] = (
    "v1 (MS6, P3.1-P3.3): first version.",
    "v2 (PR1): WhoScored season-out from competition-season folds; one penalty δ per season-out "
    "source; team-selected competition-seasons at δ = 0 (flagged); goal_z sensitivity "
    "(StatsBomb).",
)
SALT: Final[str] = "gsax-keepers"  # svg.hashsalt: the SVG's element ids, so its bytes repeat
FLAG: Final[str] = "†"  # marks a team-selected keeper-season in the table and the chart


# ---- sentences -----------------------------------------------------------------------------


def _r(s: metrics.SplitHalf) -> str:
    return split_half_cell(s, 2)


def stability_sentence(tier: Tier) -> str:
    """What a reliability tier allows the cards to say; the only place keeper skill is worded.

    Examples
    --------
    >>> none = metrics.SplitHalf(0, None, None, None)
    >>> stability_sentence(Tier(500, 0, none, none, none, none))
    'no keeper reaches 500 shots (empty tier)'
    """
    t, k, a, b = tier.tier, tier.keepers, tier.first_sb, tier.all_shots_sb
    if k == 0:
        return f"no keeper reaches {t} shots (empty tier)"
    if tier.first.r is None:
        return f"{k} keepers reach {t} shots: too few to estimate"
    if k < MIN_TIER_KEEPERS:
        return f"{k} keepers reach {t} shots: too few to judge (r {_r(tier.first)})"
    if a.low is not None and b.low is not None and a.low > 0 and b.low > 0:
        return (
            f"keeper records are repeatable beyond chance at {t} shots; this measures "
            "the keeper and the defence in front of him, not the keeper alone"
        )
    return f"not distinguishable from noise at {t} shots"


def _above(c: GainCI | None) -> str:
    return "n/a" if c is None else "above 0" if c.low > 0 else "not above 0"


def next_season_sentence(ns: NextSeason, no_team: NextSeason | None = None) -> str:
    """The next-season result in words; each clause holds in its own branch.

    With ``no_team`` (the same test without team-selected competition-seasons) a second
    sentence gives its gains and whether each interval is above 0 (the small-margin rule).

    Examples
    --------
    >>> none = NextSeason(0, 0, 0, 0, (None, None, None), None, None, None, None, 0, "no pair")
    >>> next_season_sentence(none)
    'Untestable: no pair.'
    """
    if ns.untestable is not None or ns.gain is None or ns.gain_none is None:
        return f"Untestable: {ns.untestable or 'no gain computed'}."
    first = (
        "Shrunken estimates predict next season better than raw GSAx, because raw GSAx "
        "overfits small samples"
        if ns.gain.low > 0
        else "Shrunken estimates do not predict next season better than raw GSAx"
    )
    if ns.gain_none.low > 0:
        second = f"and improve log loss by {ns.gain_none.gain:.5f} over xGOT alone"
    elif ns.gain_none.high >= 0:
        second = "and are not distinguishable from xGOT alone (the league average)"
    else:
        second = "and predict next season worse than xGOT alone (the league average)"
    out = f"{first}, {second}."
    if no_team is None:
        return out
    if no_team.untestable is not None:
        return (
            f"{out} Without {TEAM_SELECTED} competition-seasons: untestable ({no_team.untestable})."
        )
    return (
        f"{out} Without {TEAM_SELECTED} competition-seasons ({no_team.pairs:,} pairs) the gain "
        f"over raw is {_gain(no_team.gain)}, its interval {_above(no_team.gain)}, and over xGOT "
        f"alone {_gain(no_team.gain_none)}, its interval {_above(no_team.gain_none)}."
    )


# ---- shared tables -------------------------------------------------------------------------


def _checks(rows: Sequence[Check]) -> list[str]:
    return [
        "| Target | Measured | Result |",
        "|---|---|---|",
        *(f"| {c.target} | {c.measured} | {c.result} |" for c in rows),
        "",
    ]


def _num(v: float | None, d: int = 2) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:+.{d}f}"


def _source_rows(g: GsaxFit) -> list[str]:
    """Season-out xGOT per source beside the match out-of-fold xGOT on the same rows."""
    oof = g.xgot.predictions.select("shot_id", oof=pl.col("xgot"))
    rows = g.predictions.join(oof, on="shot_id", how="left")
    out = [
        "| Source | Kind | Shots | Goals | Σ xGOT | Log loss | AUC | Match-OOF log loss | "
        "Match-OOF AUC |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (src, kind), part in sorted(rows.group_by("source", "kind"), key=lambda kv: str(kv[0])):
        y = part["is_goal"].to_numpy()
        both = 0 < y.sum() < y.size
        s = metrics.score(y, part["xgot"].to_numpy()) if both else None
        o = metrics.score(y, part["oof"].to_numpy()) if both else None
        out.append(
            f"| {src} | {kind} | {part.height:,} | {int(y.sum()):,} | "
            f"{float(part['xgot'].sum()):,.1f} | {fmt(s.log_loss if s else None)} | "
            f"{fmt(s.auc if s else None)} | {fmt(o.log_loss if o else None)} | "
            f"{fmt(o.auc if o else None)} |"
        )
    return [*out, ""]


def _rating_lines(g: GsaxFit) -> list[str]:
    """One line per kind from ``gsax.rating_groups``: the fitted δ and the groups at δ = 0."""
    groups = gsax.rating_groups(g.rated)
    fitted = groups.filter(~gsax.AT_ZERO)
    zero = groups.filter(gsax.AT_ZERO)
    op = fitted.filter(pl.col("kind") == "open_play")
    fin = op.filter(pl.col("delta").is_finite())["delta"].to_numpy()
    deg = op.filter(pl.col("delta").is_infinite())
    rng = (f"from {fin.min():+.2f} to {fin.max():+.2f} (median {float(np.median(fin)):+.2f})"
           if fin.size else "none finite")  # fmt: skip
    n = op["rows"].to_numpy()
    per = f"min {n.min():,}, median {float(np.median(n)):,.1f}" if n.size else "none"
    lines = [
        f"- open play: δ {rng} over {fin.size:,} competition-seasons; {deg.height:,} with every "
        f"shot scored or none ({int(deg['rows'].sum()):,} shots), where every keeper gets 0 and "
        f"δ is infinite; shots per δ: {per}; "
        f"{zero.filter(pl.col('kind') == 'open_play').height:,} competition-seasons at δ = 0",
    ]
    pens = g.rated.filter(pl.col("kind").cast(pl.String) == "penalty")
    for r in fitted.filter(pl.col("kind") == "penalty").iter_rows(named=True):
        fam = (
            pens.filter(pl.col("group") == r["group"])
            .group_by(f=pl.col("family").cast(pl.String))
            .len()
            .sort("f")
        )
        mix = ", ".join(f"{k} {v:,}" for k, v in fam.iter_rows())
        lines.append(
            f"- penalty, δ per source: {r['group']} {_num(r['delta'])} on {r['rows']:,} kicks "
            f"({r['keepers']:,} keepers; {mix})"
        )
    pz = zero.filter(pl.col("kind") == "penalty")
    lines.append(
        f"- penalty at δ = 0: {int(pz.filter(~pl.col('pool_small'))['rows'].sum()):,} kicks of "
        f"competition-seasons at δ = 0; {pz.filter(pl.col('pool_small')).height:,} source "
        f"pool(s) failing E2's minimum ({int(pz.filter(pl.col('pool_small'))['rows'].sum()):,} "
        "kicks)"
    )
    return lines


def _z(z: float) -> str:
    return f"{z:+.2f}".replace("-", "−")  # a true minus, as ``cards.signed``


def _band_lines(g: GsaxFit) -> list[str]:
    """The rated q by placement band (``gsax.placement_bands``) and its failing judged bands,
    each with its counts, gap, SD and z (true minus signs) and whether it is in
    ``gsax.ESCALATED_BANDS`` (escalated and ruled a listed limitation, PM PR1 E1 (a)); the
    keeper-level effect of the season-out refit against the match out-of-fold
    xGOT."""
    t, least = gsax.placement_bands(g), metrics.MIN_JUDGED_SHOTS
    rows = [
        f"| {r['feature']} | {r['band']} | {r['source']} | {r['shots']:,} | {r['goals']:,} | "
        + (f"{signed(r['gap'])} | {r['sd']:.1f} | " if r["shots"] else "n/a | 0.0 | ")
        + (("yes" if r["within"] else "no") if r["judged"] else f"not judged (n < {least})")
        + (f" | {signed(r['gap_oof'])} ({r['sd_oof']:.1f}) |" if r["shots"] else " | n/a |")
        for r in t.iter_rows(named=True)
    ]
    bad = t.filter(pl.col("judged") & ~pl.col("within"))
    pinned = gsax.ESCALATED_BANDS.get(g.xgot.provider, frozenset())
    listed = "; ".join(
        f"{r['feature']} {r['band']} ({r['source']}): {r['shots']:,} shots, {r['goals']:,} "
        f"goals, Σ q − goals {signed(r['gap'])} (SD {r['sd']:.1f}, z "
        f"{_z(r['gap'] / r['sd'])}; "
        f"match out of fold {signed(r['gap_oof'])}), "
        + ("escalated, ruled a listed limitation (PM PR1 E1 (a))"
           if (r["feature"], r["band"], r["source"]) in pinned else "not escalated")
        for r in bad.iter_rows(named=True)
    )  # fmt: skip
    lines = [
        "### Calibration of the rated q by placement band",
        "",
        f"Open play; `mouth_y_abs` and `mouth_z` (m) as on the xGOT card, `angle_rad` the "
        f"opening angle; source `{gsax.POOLED}` pools the rows. A band is judged with at least "
        f"{least} shots, within {metrics.BAND_SDS:g} SD (√Σ q(1 − q)). The last column rates the "
        "same rows from the xGOT card's match out-of-fold xGOT (diagnostic: it saw the rated "
        "season).",
        "",
        "| Feature | Band | Source | Shots | Goals | Σ q − goals | SD | Within "
        f"{metrics.BAND_SDS:g} SD | Match out of fold: Σ q − goals (SD) |",
        "|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Limitation: {bad.height:,} of {int(t['judged'].sum()):,} judged bands are outside "
        f"{metrics.BAND_SDS:g} SD" + (f" ({listed}); the league shift removes a season's level "
                                      "gap, not a misprice of shape, so a keeper whose shots "
                                      "lean to these bands is charged or credited for it."
                                      if listed else "."),
    ]  # fmt: skip
    oof = gsax.keeper_table(gsax.variant(gsax.oof_rated(g), WITHOUT)).select(
        *KEEPER_SEASON, oof=pl.col("gsax")
    )
    big = g.keepers[WITHOUT].filter(pl.col("shots") >= gsax.TABLE_MIN_SHOTS)
    moved = big.join(oof, on=KEEPER_SEASON, how="left")
    d, sd = (moved["gsax"] - moved["oof"]).abs().to_numpy(), moved["sd"].to_numpy()
    if d.size:
        lines += [
            "",
            f"Keeper-level effect: rated from the match out-of-fold xGOT instead, the {d.size:,} "
            f"keeper-seasons with >= {gsax.TABLE_MIN_SHOTS} shots move by a median of "
            f"{float(np.median(d)):.2f} goals (max {float(d.max()):.2f}; their mean SD "
            f"{float(sd.mean()):.2f}).",
        ]
    return [*lines, ""]


def _abs(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def _line_lines(g: GsaxFit) -> list[str]:
    """The goal_z sensitivities (``gsax.line_summary``, ``gsax.shift_summary`` of
    ``gsax_noz``): summaries only, no per-keeper column; none without them (WhoScored)."""
    line, noz = gsax.line_summary(g), gsax.shift_summary(g, "gsax_noz")
    if line is None:
        return []
    kept = line.saves - line.far
    share = f"{line.far / line.saves:.1%}" if line.saves else "n/a"
    out = [
        "### goal_z sensitivity",
        "",
        "A projected save's `goal_z` is where the keeper met the ball, not its height at the "
        f"line. Re-rated on the goals and the saves stopped within {gsax.LINE_YD:g} yd of the "
        f"line (`end_x` >= {gsax.LINE_X:.4f} m; δ and E2's minimum re-fitted on that subset): "
        f"{kept:,} of {line.saves:,} open-play saves are kept and {line.far:,} ({share}) left "
        f"out; only saves are left out, every goal stays. Over the {line.keeper_seasons:,} "
        f"keeper-seasons with >= {gsax.TABLE_MIN_SHOTS} shots in competition-seasons with a δ "
        f"fitted on that subset, GSAx moves by a median of "
        f"{_abs(line.median_abs)} goals (90th percentile {_abs(line.p90_abs)}, max "
        f"{_abs(line.max_abs)}; their mean SD {_abs(line.mean_sd)}); Spearman with the headline "
        f"{_num(line.rank_r)}. The shift tracks the share of saves stopped off the line "
        f"(Spearman {_num(line.far_share_r)} with each keeper-season's far saves per shot). "
        "Open play only; a sensitivity beside the headline, not a rating.",
        "",
        f"Left out of both summaries: {line.at_zero:,} keeper-seasons with >= "
        f"{gsax.TABLE_MIN_SHOTS} shots in competition-seasons at δ = 0 on the subset "
        f"(team-selected, or E2-small there; median |shift| {_abs(line.at_zero_median_abs)}). "
        "A δ = 0 season cannot re-level a subset selected on the outcome, so their shift is "
        "that selection, not `goal_z`.",
        "",
    ]
    if noz is not None:
        out += [
            f"Without `{'`, `'.join(gsax.NOZ_FEATURES)}`: the headline xGOT refitted season-out "
            "without it (no selection on the outcome; open play only, the penalty xGOT reads it "
            f"too) moves the same {noz.keeper_seasons:,} keeper-seasons by a median of "
            f"{_abs(noz.median_abs)} goals (90th percentile {_abs(noz.p90_abs)}, max "
            f"{_abs(noz.max_abs)}; their mean SD {_abs(noz.mean_sd)}); Spearman with the "
            f"headline {_num(noz.rank_r)}, and its shift with the far-save share "
            f"{_num(noz.far_share_r)}. A sensitivity, not the headline.",
            "",
        ]
    return out


def _team_lines(g: GsaxFit) -> list[str]:
    """The team-selected competition-seasons with a fitted δ before PR1 (team-selected, not
    E2-small) by side (``gsax.team_selected_split``), counts only; none without such rows."""
    groups = gsax.rating_groups(gsax.variant(g.rated, WITHOUT))
    seasons = groups.filter(pl.col("zero_family") & ~pl.col("small"))
    keys = (
        gsax.variant(g.rated, WITHOUT)
        .filter(pl.col("group").is_in(seasons["group"].implode()))
        .select(SEASON_KEY)
        .unique()
    )
    split = gsax.team_selected_split(g.keepers[WITHOUT].join(keys, on=SEASON_KEY, how="semi"))
    if split.height == 0:
        return []
    sides = "; ".join(
        f"{r['side']}: {r['keeper_seasons']:,} keeper-seasons, {r['shots']:,} shots, GSAx "
        f"{signed(r['gsax'])} (SD {r['sd']:.1f})"
        for r in split.iter_rows(named=True)
    )
    return [
        f"Team-selected competition-seasons ({FLAG}) not E2-small: {seasons.height:,}, at δ = 0 "
        f"(PM 2026-10-28). By side (the selected side's keeper faces at least "
        f"{gsax.SELECTED_SIDE} shots that season): {sides}; the seasons' Σ (q − goal) "
        f"{signed(float(seasons['gap'].sum()))}. δ = 0 removes the league-sum coupling, not the "
        "gap between the sides.",
        "",
    ]


def _tier_rows(tiers: Sequence[Tier]) -> list[str]:
    ci = f"{metrics.LEVEL:.0%} CI"
    return [
        f"| Shots faced (tier) | Keepers | First T shots: r [{ci}] | Spearman-Brown | All their "
        f"shots: r [{ci}] | Spearman-Brown | Reading |",
        "|---|---|---|---|---|---|---|",
        *(
            f"| {t.tier} | {t.keepers:,} | {_r(t.first)} | {_r(t.first_sb)} | {_r(t.all_shots)} "
            f"| {_r(t.all_shots_sb)} | {stability_sentence(t)} |"
            for t in tiers
        ),
        "",
    ]


def _overview(g: GsaxFit, name: str, task: str) -> list[str]:
    f = g.xgot
    m = f.xg.manifest
    return [
        f"- **Name / version:** `{name}` {VERSION}",
        f"- **Task:** {task}",
        "- **Status:** experimental",
        f"- **Source(s):** {f.setup.text.source}; snapshot `{m.name}`, data digest "
        f"`{m.data_digest}`; xGOT: the headline {f.head.spec.title}, refitted season-out",
        "",
    ]


# ---- the GSAx card -------------------------------------------------------------------------


def _population(g: GsaxFit) -> list[str]:
    f, open_ = g.xgot, gsax.variant(g.rated, WITHOUT)
    recon = {r: (n, k) for r, n, k in f.recon.rows}
    og = open_.filter(pl.col("open_goal").fill_null(False))
    lg = g.leagues[WITHOUT]
    small = lg.filter(pl.col("small"))
    small_ks = g.keepers[WITHOUT].join(small.select(SEASON_KEY), on=SEASON_KEY, how="semi")
    team = (
        open_.group_by(SEASON_KEY)
        .agg(pl.col("small", "zero_family").first())
        .filter(pl.col("zero_family"))
    )
    team_ks = g.keepers[WITHOUT].join(team.select(SEASON_KEY), on=SEASON_KEY, how="semi")
    pens = g.rated.filter(pl.col("kind").cast(pl.String) == "penalty")
    lines = [
        f"- **Rated rows:** {open_.height:,} on-target open-play shots "
        f"({int(open_['is_goal'].sum()):,} goals) and {pens.height:,} in-game penalties "
        f"({int(pens['is_goal'].sum()):,} goals), "
        f"faced by {g.rated['gk_id'].n_unique():,} keepers in {g.keepers[WITHOUT].height:,} "
        f"keeper-seasons and {lg.height:,} competition-seasons.",
        f"- **Not rated:** {recon['shootout'][0]:,} shoot-out kicks (reported apart by the "
        f"penalty xGOT); {recon['direct_corner'][0]:,} on-target direct corners; own goals are "
        "never on target, so they are in no keeper's record. Goals in the tables are goals from "
        "rated shots only, not goals conceded.",
        "- **Open goals** (`open_goal`, kept in the population, PM X1): "
        + (f"{og.height:,} shots, Σ GSAx {_og_gsax(og):+.1f}."
           if open_["open_goal"].null_count() < open_.height else "not recorded."),
        f"- **Small competition-seasons** (at most {_small_text()}): {small.height:,} rated at "
        f"δ = 0 and flagged (PM E2 (b)), with {small_ks.height:,} keeper-seasons; "
        f"{lg.filter(pl.col('keepers') <= 2).height:,} competition-seasons have at most 2 "
        "keepers.",
    ]  # fmt: skip
    if team.height:
        lines.append(
            f"- **Team-selected competition-seasons:** {team.height:,} rated at δ = 0 and flagged "
            f"(PM 2026-10-28), {team.filter(pl.col('small')).height:,} also E2-small, with "
            f"{team_ks.height:,} keeper-seasons."
        )
    if recon["coarse_grid"][0]:
        lines.append(
            f"- **Coarse-grid matches:** {recon['coarse_grid'][0]:,} on-target shots of matches "
            "recorded on the 1 yd grid are left out (PM E1 (a)); by competition-season below."
        )
    return [*lines, ""]


def _og_gsax(og: pl.DataFrame) -> float:
    return float((og["q"] - og["is_goal"].cast(pl.Float64)).sum())


def _small_text() -> str:
    s = gsax.SMALL_LEAGUE
    return "no rule" if s is None else f"{s.max_keepers} keepers or under {s.min_shots} shots"


def _ci_lines(g: GsaxFit) -> list[str]:
    c = g.ci
    return [
        "## Uncertainty",
        "",
        f"`sd` is √Σ q(1 − q): the spread of GSAx for a league-average keeper facing these "
        f"shots, not the uncertainty of this keeper's estimate. The {CI_LEVEL:.0%} interval "
        f"resamples the keeper-season's matches ({metrics.N_BOOT:,} draws, seed {metrics.SEED}) "
        "with the rating held fixed; with few matches it undercovers. Against a null with goals "
        "drawn once from q (no keeper differences): "
        f"{c.table_observed:,} of {c.table_rows:,} keeper-seasons with >= {gsax.TABLE_MIN_SHOTS} "
        f"shots have a {CI_LEVEL:.0%} CI excluding 0; {c.table_null:,} would with no keeper "
        f"differences. Over the {c.multi_rows:,} keeper-seasons with >= 2 matches: "
        f"{c.multi_observed:,} observed, {c.multi_null:,} under the null.",
        "",
    ]


def _keeper_table(g: GsaxFit, names: pl.DataFrame, chart_sha256: str | None) -> list[str]:
    rows = gsax.keeper_rows(g)
    named = rows.join(names.select(gk_id="player_id", name="player_name"), on="gk_id",
                      how="left")  # fmt: skip
    if missing := named.filter(pl.col("name").is_null()).height:
        raise ValueError(f"render: {missing} table keeper(s) are missing from names")
    conflicts = names.join(rows.select(player_id="gk_id"), on="player_id", how="semi")
    out = [
        f"## Keeper-seasons with >= {gsax.TABLE_MIN_SHOTS} shots ({WITHOUT}, sorted by GSAx)",
        "",
        f"{rows.height:,} keeper-seasons. Names are the lineup display names (the nickname when "
        f"StatsBomb gives one); {int((conflicts['names'] > 1).sum()):,} table keeper(s) appear "
        "under two spellings, resolved to the latest match. The chart is written by `make "
        f"gsax` to `{CHART_FILE}` (git-ignored: it embeds the StatsBomb logo, PM E3)"
        + (
            "."
            if chart_sha256 is None
            else f"; its SHA-256 is `{chart_sha256}`, so `make "
            "reproduce` checks the chart through this card."
        ),
        "",
        "| # | Keeper | Family | Competition season | Shots | Goals | Open goals | Open-goal "
        f"GSAx | GSAx raw | GSAx | SD | {CI_LEVEL:.0%} CI | Shrunken GSAx | GSAx with penalties |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(named.iter_rows(named=True), 1):
        ci = "n/a" if r["ci_low"] is None else f"[{r['ci_low']:+.1f}, {r['ci_high']:+.1f}]"
        og = "n/a" if r["open_goal"] is None else f"{r['open_goal']:,}"
        out.append(
            f"| {i} | {r['name']} | {_family(r['family'])} | {r['competition_name']} "
            f"{r['season_name']} | "
            f"{r['shots']:,} | {r['goals']:,} | {og} | {_num(r['open_goal_gsax'], 1)} | "
            f"{r['gsax_raw']:+.1f} | {r['gsax']:+.1f} | {r['sd']:.1f} | {ci} | "
            f"{_num(r['gsax_shrunk'], 1)} | {_num(r['gsax_with'], 1)} |"
        )
    return [*out, ""]


def _family(family: str) -> str:
    """The family cell or label, flagged when team-selected (``_team_lines``)."""
    return f"{family} {FLAG}" if family == TEAM_SELECTED else family


def _distribution(g: GsaxFit) -> list[str]:
    # the criterion reports GSAx with and without penalties: both variants over the same
    # keeper-seasons (>= TABLE_MIN_SHOTS shots without penalties), as the StatsBomb table does
    keys = g.keepers[WITHOUT].filter(pl.col("shots") >= gsax.TABLE_MIN_SHOTS).select(KEEPER_SEASON)
    qs = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    out = [
        f"## Distribution over keeper-seasons with >= {gsax.TABLE_MIN_SHOTS} shots ({WITHOUT})",
        "",
        f"{keys.height:,} keeper-seasons; no row per keeper (counts and metrics only). The second "
        "row rates the same keeper-seasons with their in-game penalties.",
        "",
        "| GSAx | Min | 10% | 25% | Median | 75% | 90% | Max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for variant in gsax.VARIANTS:
        rows = keys.join(g.keepers[variant], on=KEEPER_SEASON, how="left")["gsax"]
        vals = [_num(rows.quantile(q, "nearest"), 1) if rows.len() else "n/a" for q in qs]
        out.append(f"| {variant} | " + " | ".join(vals) + " |")
    return [*out, ""]


def _gsax_card(g: GsaxFit, names: pl.DataFrame | None, name: str, chart_sha256: str | None) -> str:
    f = g.xgot
    text = f.setup.text
    lines = [
        "# Model card: Goals saved above expected (GSAx)",
        "",
        "## Overview",
        *_overview(g, name, "a keeper's record of on-target shots faced: Σ q − goals, q the "
                   "xGOT rated against the league sum (P3.1)"),
        "## Intended use",
        "A record of shots faced, rated against the league, not a skill rating: GSAx says how "
        "many fewer goals a keeper conceded than a league-average keeper facing the same "
        "on-target shots, given where they crossed the line (positive is better). It credits "
        "the keeper and the defence in front of him; the stability section says what it allows "
        "beyond that. For research; not for other providers' data without re-fitting.",
        "",
        "## Data",
        *_population(g),
        *xgot_cards.grid_lines(f.grid),
        "## Method",
        "",
        "Every row's xGOT comes from fits (xG input and xGOT) that never saw its season "
        f"(`season_out.season_out`, source `{season_out.SEASON_OUT[f.provider]}`: folds grouped "
        "by competition-season, the test season from the refit's dev fit, `holdout_fit`); the "
        "match out-of-fold xGOT of the xGOT card is beside it on the same rows.",
        "",
        *_source_rows(g),
        "Rating against the league sum: one logit shift δ per rating group makes Σ q = goals, "
        "q = expit(logit xGOT + δ), so the xGOT level gap is not charged to keepers (PM after "
        "MS5). Open play has one δ per competition-season, penalties one δ per season-out source "
        f"(PM 2026-10-28 (a)); a group with at most {_small_text()} is at δ = 0 (E2's "
        "minimum, every grouping), and so are team-selected competition-seasons. "
        "Each row is rated once, so GSAx with penalties minus without is exactly the keeper's "
        "penalty GSAx.",
        "",
        *_rating_lines(g),
        "",
        *_band_lines(g),
        *_line_lines(g),
        "## Checks",
        "",
        *_checks(g.checks),
        *_ci_lines(g),
        "## Stability",
        "",
        f"Split-half reliability of q − goal per shot ({WITHOUT}), shots alternated within a "
        "keeper by match date, period, time and shot id (PM X5), over the keepers whose total "
        "shots faced (all seasons) reach the tier: first on each keeper's first T shots, then "
        f"on all their shots; Spearman-Brown steps r up to the whole. {metrics.LEVEL:.0%} "
        "intervals by Fisher z. "
        f"A tier is judged only with at least {MIN_TIER_KEEPERS} keepers.",
        "",
        *_tier_rows(g.tiers),
        *_team_lines(g),
        *(_keeper_table(g, names, chart_sha256) if names is not None else _distribution(g)),
        "## Columns",
        "",
        "- `goals`: goals from rated shots only (own goals, direct corners and shoot-outs are "
        "not in it), so not goals conceded.",
        "- `GSAx raw`: Σ xGOT − goals, before the league rating; `GSAx`: Σ q − goals.",
        f"- `SD`: the null SD √Σ q(1 − q); `{CI_LEVEL:.0%} CI`: the match bootstrap above.",
        "- `family`: women; team-selected (one team in at least half of a competition-season's "
        "matches; rated at δ = 0); league or tournament.",
        "",
        "## Data licensing",
        text.licensing,
        "",
        "## Changelog",
        *(f"- {entry}" for entry in CHANGELOG),
        "",
    ]  # fmt: skip
    return "\n".join(lines)


# ---- the keeper-model card -----------------------------------------------------------------


def _next_rows(label: str, ns: NextSeason) -> str:
    ll = " | ".join(fmt(v) for v in ns.log_loss)
    return (
        f"| {label} | {ns.pairs:,} | {ns.left_out:,} | {ns.test_shots:,} / {ns.test_matches:,} | "
        f"{ll} | {_gain(ns.gain)} | {_gain(ns.gain_none)} | {_gain(ns.gain_loo)} | "
        f"{_gain(ns.gain_none_loo)} |"
    )


def _model_card(g: GsaxFit, name: str) -> str:
    f, m, c = g.xgot, g.model, g.career
    lvl = f"{metrics.GAIN_LEVEL:.0%}"
    rows = [("Keeper-season effect", g.next_season)]
    if g.next_no_team is not None:
        rows.append((f"… without {TEAM_SELECTED} competition-seasons", g.next_no_team))
    rows.append(("Career effect (the literal formula)", g.next_career))
    loo = [
        f"Leave-out columns of *{label}*: n/a, because {ns.loo_missing:,} test season(s) have "
        "no shot faced by an unpaired keeper, or no finite δ without the paired ones."
        if ns.gain_loo is None
        else f"Leave-out columns of *{label}*: {ns.loo_missing:,} test season(s) with no shot "
        "faced by an unpaired keeper, or no finite δ without the paired ones, are left out "
        f"({ns.loo_shots:,} test shots, {ns.loo_pairs:,} pairs)."
        for label, ns in rows
        if ns.loo_missing
    ]
    loo += [
        f"Leave-out δ per test season, E2's minimum (at most {_small_text()}: δ = 0) applied "
        "to the unpaired keepers: "
        + "; ".join(f"*{label}* {ns.loo_missing:,} left out, {ns.loo_small:,} at δ = 0"
                    for label, ns in rows) + ".",
        "",
    ]  # fmt: skip
    nt = g.model_no_team
    no_team = (
        []
        if nt is None
        else [f"| keeper-season without {TEAM_SELECTED} competition-seasons | {nt.tau:.4f} | "
              f"{'yes' if nt.at_bound else 'no'} | {nt.per_100:.2f} |"]
    )  # fmt: skip
    lines = [
        "# Model card: Hierarchical keeper model",
        "",
        "## Overview",
        *_overview(g, name, "goal ~ offset(logit q) + (1 | keeper-season), no fixed intercept: a "
                   "shrunken GSAx per keeper-season (P3.2)"),
        "## Intended use",
        "Shrinks each keeper-season's GSAx towards the league according to how many shots it "
        "rests on, and asks whether that predicts the next season better. For research; the "
        "stability section of the GSAx card decides what may be said about keeper skill.",
        "",
        "## Model",
        "",
        "Each on-target open-play shot's goal is Bernoulli with logit = logit q + u, q the rated "
        "xGOT (so the league level is fixed per season and there is no intercept), u ~ N(0, τ²) "
        "per group. τ̂ maximises the Laplace approximation of the marginal likelihood (bounded "
        f"search on log τ in [{gsax.TAU_BOUNDS[0]:g}, {gsax.TAU_BOUNDS[1]:g}]); u is the "
        "posterior mode at τ̂, by Newton steps. The Laplace τ̂ is biased low when the true τ is "
        "small, so the shrinkage is on the strong side. No PyMC or Stan: one random effect needs "
        "only the closed-form Laplace fit.",
        "",
        "| Grouping | τ̂ (logit) | At a bound | One SD in goals per 100 shots |",
        "|---|---|---|---|",
        f"| keeper-season | {m.tau:.4f} | {'yes' if m.at_bound else 'no'} | {m.per_100:.2f} |",
        *no_team,
        f"| keeper career | {c.tau:.4f} | {'yes' if c.at_bound else 'no'} | {c.per_100:.2f} |",
        "",
        "## Next-season test",
        "",
        "Pairs: the same keeper in the same competition in consecutive seasons (by start year); "
        "a pair whose rated-season estimate is infinite (0 or every shot a goal) is left out. "
        "For each test year the model is fitted on every rated row of earlier seasons; raw is "
        "the keeper's own unpooled shift, none is xGOT alone (the league average). Log loss on "
        f"the next season's on-target shots; gains with {lvl} match-bootstrap intervals; the "
        "leave-out columns rate the test season without the paired keepers' own shots.",
        "",
        "| Model | Pairs | Left out | Test shots / matches | Log loss none | Log loss raw | "
        "Log loss shrunk | Gain over raw | Gain over none | Gain over raw (leave-out δ) | Gain "
        "over none (leave-out δ) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *(_next_rows(label, ns) for label, ns in rows),
        "",
        *loo,
        next_season_sentence(g.next_season, g.next_no_team),
        "",
        "Raw GSAx is the unpooled estimate on small samples, so beating it says raw overfits; "
        "keeper differences are read from the gain over none.",
        "",
        "Disclosures: (1) the test offsets use the test season's δ, which the paired keepers' "
        "own goals feed; the leave-out columns show the result without them. (2) "
        + _saw_test_text(g.saw_test),
        "",
        "## Checks",
        "",
        *_checks(g.checks[gsax.NEXT_SEASON_CHECK - 1 : gsax.NEXT_SEASON_CHECK]),
        "## Data licensing",
        f.setup.text.licensing,
        "",
        "## Changelog",
        *(f"- {entry}" for entry in CHANGELOG),
        "",
    ]  # fmt: skip
    return "\n".join(lines)


def _saw_test_text(saw: tuple[int, int, int]) -> str:
    """Disclosure (2) from ``GsaxFit.saw_test`` (S, P, L), computed (PM E4).

    Examples
    --------
    >>> _saw_test_text((0, 4, 1))
    "No pair's rated-season xGOT was fitted on its test season (4 pairs scored, 1 left out)."
    """
    seen, pairs, left = saw
    if seen == 0:
        return (
            f"No pair's rated-season xGOT was fitted on its test season ({pairs:,} pairs scored, "
            f"{left:,} left out)."
        )
    return (
        f"For {seen:,} of the {pairs:,} pairs the test scores, the rated season's xGOT comes from "
        "fits that trained on the test season (season-out folds train on later seasons too), the "
        "same for all three predictors; a rolling xGOT is a later item (PM E4); "
        f"{left:,} more pairs are left out by the test (infinite raw estimate)."
    )


def render(
    g: GsaxFit, *, names: pl.DataFrame | None, chart_sha256: str | None = None
) -> Mapping[str, str]:
    """The two cards of one provider, card name → Markdown; pure.

    ``chart_sha256`` (StatsBomb) is the SHA-256 of the keeper chart's bytes, printed beside
    ``CHART_FILE``: the chart is git-ignored (PM E3 (b)), so the committed card carries its
    digest and the card's byte check covers the chart.

    Raises
    ------
    ValueError
        StatsBomb without ``names``; WhoScored with names; a table keeper missing from
        ``names``; an unknown provider.

    Examples
    --------
    >>> sorted(render(g_ws, names=None))  # doctest: +SKIP
    ['ws-gk-model', 'ws-gsax']
    """
    p = g.xgot.provider
    if p not in CARD_PREFIX:
        raise ValueError(f"render: unknown provider {p!r}")
    if (p in NAMED) != (names is not None):
        raise ValueError(f"render: {p} cards {'need' if p in NAMED else 'take no'} keeper names")
    pre = CARD_PREFIX[p]
    return MappingProxyType({
        f"{pre}gsax": _gsax_card(g, names, f"{pre}gsax", chart_sha256),
        f"{pre}gk-model": _model_card(g, f"{pre}gk-model"),
    })  # fmt: skip


# ---- chart ---------------------------------------------------------------------------------


def chart(table: pl.DataFrame, *, names: pl.DataFrame, logo: StrPath) -> str:
    """The StatsBomb keeper forest plot as SVG text: GSAx with its CI, the shrunken GSAx hollow.

    ``table`` is :func:`gsax.keeper_rows` output. Deterministic bytes (``svg.hashsalt``, no
    date).

    Raises
    ------
    ValueError
        A ``gk_id`` that is not ``sb:`` (Rule 1) or is missing from ``names``.
    FileNotFoundError
        No logo file (``viz`` refuses an uncredited StatsBomb chart).

    Examples
    --------
    >>> svg = chart(gsax.keeper_rows(g), names=names, logo=statsbomb.logo_file())
    ... # doctest: +SKIP
    >>> viz.LOGO_GID in svg  # doctest: +SKIP
    True
    """
    if bad := table.filter(~pl.col("gk_id").str.starts_with("sb:")).height:
        raise ValueError(f"chart: {bad} keeper id(s) are not StatsBomb ('sb:'); Rule 1")
    rows = table.join(names.select(gk_id="player_id", name="player_name"), on="gk_id", how="left")
    if missing := rows.filter(pl.col("name").is_null()).height:
        raise ValueError(f"chart: {missing} keeper(s) are missing from names")
    n = rows.height
    w, h = 9.0, 1.6 + 0.22 * max(n, 1)
    fig = Figure(figsize=(w, h), dpi=100, facecolor=viz.SURFACE)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((3.6 / w, 1.0 / h, 1 - 3.9 / w, 1 - 1.3 / h))
    if n == 0:
        ax.text(0.5, 0.5, f"no keeper-season reaches {gsax.TABLE_MIN_SHOTS} shots", ha="center",
                va="center", transform=ax.transAxes)  # fmt: skip
        ax.set_yticks([])
    else:
        y = np.arange(n)[::-1].astype(np.float64)
        lo, hi = rows["ci_low"].fill_null(np.nan), rows["ci_high"].fill_null(np.nan)
        gs = rows["gsax"].to_numpy()
        ax.hlines(y, lo.to_numpy(), hi.to_numpy(), color=viz.LINE_COLOUR, lw=1.2)
        ax.plot(gs, y, "o", color=viz.GOAL_COLOUR, ms=4, label="GSAx")
        ax.plot(rows["gsax_shrunk"].fill_null(np.nan).to_numpy(), y, "o", mfc="none",
                color=viz.SHOT_COLOUR, ms=5, label="shrunken GSAx")  # fmt: skip
        labels = [
            f"{r['name']} · {r['competition_name']} {r['season_name']}"
            + (f" · {_family(r['family'])}" if r["family"] == TEAM_SELECTED else "")
            for r in rows.iter_rows(named=True)
        ]
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_ylim(-1, n)
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    ax.axvline(0.0, color=viz.INK, lw=0.6)
    note = (f"\n{FLAG} {TEAM_SELECTED}: rated at δ = 0, no league shift (PM 2026-10-28; see "
            "the card)"
            if (rows["family"] == TEAM_SELECTED).any() else "")  # fmt: skip
    ax.set_xlabel(f"Goals saved above expected ({WITHOUT}; {CI_LEVEL:.0%} match-bootstrap CI)"
                  + note, fontsize=8)  # fmt: skip
    ax.set_title(f"Keeper-seasons with >= {gsax.TABLE_MIN_SHOTS} on-target shots faced", fontsize=9)
    viz.add_attribution(ax, "statsbomb", logo=logo)
    buf = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": SALT}):
        fig.savefig(buf, format="svg", metadata={"Date": None})
    return buf.getvalue()


# ---- README block, write and run -----------------------------------------------------------


def summary_block(fits: Sequence[GsaxFit]) -> str:
    """The README block ``gsax:summary``: counts and metrics only, no keeper name or id.

    Examples
    --------
    >>> print(summary_block([g_sb, g_ws]))  # doctest: +SKIP
    | Provider | Keeper-seasons | ...
    """
    lines = [
        "| Provider | Rated shots | Keeper-seasons | Keeper-seasons >= "
        f"{gsax.TABLE_MIN_SHOTS} shots | Checks met | τ̂ keeper-season | Next season: gain over raw "
        "| Gain over xGOT alone |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes = []
    for g in fits:
        src, ns, ks = g.xgot.setup.text.source, g.next_season, g.keepers[WITHOUT]
        met = sum(c.result == "met" for c in g.checks)
        lines.append(
            f"| {src} | {g.rated.height:,} | {ks.height:,} | "
            f"{ks.filter(pl.col('shots') >= gsax.TABLE_MIN_SHOTS).height:,} | {met} of "
            f"{len(g.checks)} | {g.model.tau:.4f} | {_gain(ns.gain)} | {_gain(ns.gain_none)} |"
        )
        notes += [f"{src}: {next_season_sentence(ns, g.next_no_team)} Stability: "
                  + "; ".join(stability_sentence(t) for t in g.tiers) + ".", ""]  # fmt: skip
    return "\n".join([*lines, "", *notes[:-1]]) + "\n"


@dataclass(frozen=True, eq=False)
class GsaxRun(GsaxFit):
    """A :class:`~football_shot_models.gsax.GsaxFit` plus what :func:`run_from` wrote: ``cards``
    (card name → Markdown), the StatsBomb chart ``svg`` and its ``chart_path`` (None for
    WhoScored)."""

    cards: Mapping[str, str]
    svg: str | None
    chart_path: Path | None


def write(
    written: Mapping[str, str], svg: str | None, *, docs_dir: StrPath, chart_path: StrPath
) -> Path | None:
    """Write each card to ``docs_dir/<name>.md`` and the SVG (if any) to ``chart_path``, which
    must be git-ignored when it lies in a work tree (fail closed).

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     write({"gsax": "# x\\n"}, None, docs_dir=tmp, chart_path=Path(tmp) / "c.svg") is None
    True
    """
    out = Path(docs_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, card in written.items():
        (out / f"{name}.md").write_text(card, encoding="utf-8", newline="\n")
    if svg is None:
        return None
    path = Path(chart_path)
    snapshot.check_ignored(
        Path(os.path.abspath(path)),
        "keeper chart",
        "keep it under the git-ignored data/ (it embeds the StatsBomb logo)",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")
    return path


def run_from(f: xgot.XgotFit, *, docs_dir: StrPath, raw: StrPath) -> GsaxRun:
    """Fit, name (StatsBomb), render, write and print one provider's goalkeeper results.

    The chart goes to ``CHART_PATH``, read when called, and its SHA-256 onto the StatsBomb GSAx
    card; ``raw`` is the StatsBomb open-data cache whose ``<commit>`` checkout holds the lineups
    (read, never synced: a missing file is an error, not a download).

    Examples
    --------
    >>> run_from(xgot.fit("whoscored", cache="data/cache"), docs_dir="docs/models",
    ...          raw=STATSBOMB_RAW).cards.keys()  # doctest: +SKIP
    dict_keys(['ws-gsax', 'ws-gk-model'])
    """
    g = gsax.fit(f)
    names, svg = None, None
    if f.provider in NAMED:
        ids = f.xg.matches.sort("match_date", "match_id")["match_id"].to_list()
        table = gsax.keeper_rows(g)
        # the synced checkout itself, never statsbomb.sync: a missing file must fail, not download
        checkout = Path(raw) / statsbomb.OPEN_DATA_COMMIT
        names = statsbomb.player_names(checkout, ids,
                                       player_ids=table["gk_id"].to_list())  # fmt: skip
        svg = chart(table, names=names, logo=statsbomb.logo_file())
    sha = None if svg is None else hashlib.sha256(svg.encode("utf-8")).hexdigest()
    written = render(g, names=names, chart_sha256=sha)
    path = write(written, svg, docs_dir=docs_dir, chart_path=CHART_PATH)
    print(*(f"check | {c.target} | {c.measured} | {c.result}" for c in g.checks), sep="\n")
    print(f"cards: {Path(docs_dir)}" + ("" if path is None else f"; chart: {path}"))
    fields = {field.name: getattr(g, field.name) for field in dataclasses.fields(g)}
    return GsaxRun(**fields, cards=written, svg=svg, chart_path=path)


def run(provider: str, *, docs_dir: StrPath, cache: StrPath | None, raw: StrPath) -> GsaxRun:
    """``run_from(xgot.fit(provider, cache=cache), …)``.

    Examples
    --------
    >>> run("statsbomb", docs_dir="docs/models", cache="data/cache", raw=STATSBOMB_RAW)
    ... # doctest: +SKIP
    """
    return run_from(xgot.fit(provider, cache=cache), docs_dir=docs_dir, raw=raw)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; exit 0 even when a check is missed (it reports), 1 on an error
    (a StatsBomb run without ``tree.json`` in ``--statsbomb-raw`` fails before any fit), 2 on
    bad arguments (argparse's ``SystemExit``).

    Examples
    --------
    >>> main(["--provider", "nobody"])
    Traceback (most recent call last):
    SystemExit: 2
    """
    parser = argparse.ArgumentParser(
        prog="python -m football_shot_models.gsax_cards",
        description="Rate the goalkeepers (MS6 P3.1-P3.3) and write their cards and chart.",
    )
    providers = tuple(xgot.PROVIDERS)
    parser.add_argument("--provider", choices=(*providers, "both"), default="both")
    parser.add_argument("--docs-dir", default="docs/models", help="where the cards go")
    parser.add_argument("--statsbomb-raw", default=STATSBOMB_RAW,
                        help="the StatsBomb open-data cache (lineups)")  # fmt: skip
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--cache", default="data/cache", help="prediction cache root (git-ignored)")
    where.add_argument("--no-cache", action="store_true", help="fit everything, write no cache")
    args = parser.parse_args(argv)
    keys = providers if args.provider == "both" else (args.provider,)
    cache = None if args.no_cache else args.cache
    tree = Path(args.statsbomb_raw) / statsbomb.OPEN_DATA_COMMIT / "tree.json"
    if any(k in NAMED for k in keys) and not tree.is_file():
        print(f"fsm: error: no StatsBomb checkout at {tree} (the lineups name the keepers); "
              "pass --statsbomb-raw", file=sys.stderr)  # fmt: skip
        return 1
    try:
        runs = [run(k, docs_dir=args.docs_dir, cache=cache, raw=args.statsbomb_raw) for k in keys]
    except (ValueError, OSError, snapshot.SnapshotError) as e:
        print(f"fsm: error: {e}", file=sys.stderr)
        return 1
    print(summary_block(runs), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
