"""M1.5: the penalty model, one probability for every in-game penalty, and its card.

**Penalties** (``xg-penalty``) get one probability: the posterior mean of a Beta-binomial
with the Jeffreys prior over every in-game penalty (:func:`penalty_rows`); shoot-out kicks are
reported in their own rows and never pooled. Match xG = Σ non-penalty xG + P(goal | in-game
penalty) × in-game penalties.

:func:`exchangeability` reports how far the kicks are from one pooled rate (Pearson dispersion
across competition-seasons, and the most frequent taker), and :func:`penalty_card` writes the
card. The run that calls them is :func:`football_shot_models.models.run`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import polars as pl
from scipy import stats

from football_shot_models import data
from football_shot_models.cards import LICENSING, CardContext, fmt
from football_shot_models.metrics import JEFFREYS, LEVEL, Rate, beta_interval

PENALTY_NAME: Final[str] = "xg-penalty"
PENALTY_TITLE: Final[str] = "Penalty conversion (Beta-binomial)"
PENALTY_RANGE: Final[tuple[float, float]] = (0.76, 0.80)
PENALTY_CHANGELOG: Final[tuple[str, ...]] = (
    "v1 (MS3, M1.5): first version.",
    "v2 (MS3 assessment): the milestone check adds the pooled estimate without the team-selected "
    "seasons.",
)
IN_GAME, SHOOTOUT = "In-game", "Shoot-out"
MATCH_XG: Final[str] = (
    "Match xG = Σ non-penalty xG (headline model) + P(goal | in-game penalty) × in-game "
    "penalties; shoot-out kicks are not part of match xG, and own goals and direct corners "
    "carry no xG."
)


@dataclass(frozen=True)
class PenaltyRow:
    """One group of penalty kicks: its Beta posterior, matches and the provider mean xG.

    ``matches`` counts distinct ``match_id``; ``provider_xg`` is ``None`` when any kick in
    the group has no provider xG.
    """

    label: str
    rate: Rate
    matches: int
    provider_xg: float | None


def penalty_rows(
    penalties: pl.DataFrame, *, holdout: data.Holdout | pl.Expr | None = None
) -> tuple[PenaltyRow, ...]:
    """The penalty model (the first row) and its breakdown; shoot-outs in their own rows.

    Rows, in order, each skipped when it has no kick: in-game all (the model), dev, hold-out,
    men, women, then each family except ``women`` (that family is the women's row); then
    shoot-out all, men, women.

    Parameters
    ----------
    penalties : polars.DataFrame
        ``XgData.penalties``: ``reason``, ``is_goal``, ``gender``, ``family``, ``match_id``,
        ``provider_xg`` and the columns ``holdout`` reads.
    holdout : data.Holdout or polars.Expr, optional
        True for hold-out kicks; default ``data.is_holdout()`` (the full competition-season
        pair, so shots and penalties share one key). A run passes the same
        :class:`~football_shot_models.data.Holdout` it split the shots with.

    Returns
    -------
    tuple of PenaltyRow

    Raises
    ------
    ValueError
        A ``reason`` other than ``penalty`` / ``shootout``; a null in ``reason``,
        ``is_goal``, ``gender``, ``family``, ``match_id`` or the hold-out mask; a hold-out
        that fails on this frame (:func:`~football_shot_models.data.holdout_mask`); no in-game
        penalty.

    Examples
    --------
    >>> kicks = pl.DataFrame({
    ...     "reason": ["penalty", "penalty", "penalty", "shootout"],
    ...     "is_goal": [True, True, False, True],
    ...     "gender": ["male"] * 4,
    ...     "family": ["league or tournament"] * 4,
    ...     "match_id": ["sb:1", "sb:2", "sb:2", "sb:3"],
    ...     "competition_id": ["sb:2", "sb:11", "sb:11", "sb:11"],
    ...     "season_id": ["sb:27"] * 4,
    ...     "provider_xg": [0.78, 0.78, 0.78, None]})
    >>> [(r.label, r.rate.n, r.matches) for r in penalty_rows(kicks)][:3]
    [('In-game: all', 3, 2), ('In-game: dev', 2, 1), ('In-game: hold-out', 1, 1)]
    >>> penalty_rows(kicks)[-1].label, penalty_rows(kicks)[-1].provider_xg
    ('Shoot-out: men', None)
    """
    for col in ("reason", "is_goal", "gender", "family", "match_id"):
        if nulls := penalties[col].null_count():
            raise ValueError(f"penalties: {col} has {nulls} null value(s)")
    if other := sorted(set(penalties["reason"].cast(pl.String)) - {"penalty", "shootout"}):
        raise ValueError(f"penalties: reason {other} is not 'penalty' or 'shootout'")
    held = data.holdout_mask(
        penalties, data.is_holdout() if holdout is None else holdout, "penalties"
    )
    frame = penalties.with_columns(_held=held)
    game, shoot = (frame.filter(pl.col("reason") == r) for r in ("penalty", "shootout"))
    if game.height == 0:
        raise ValueError("penalties: no in-game penalty to pool")
    men, women, is_held = pl.col("gender") == "male", pl.col("gender") == "female", pl.col("_held")
    groups = [
        *((f"{IN_GAME}: {k}", game.filter(e)) for k, e in (("all", True), ("dev", ~is_held))),
        *((f"{IN_GAME}: {k}", game.filter(e)) for k, e in (("hold-out", is_held), ("men", men))),
        (f"{IN_GAME}: women", game.filter(women)),
        *((f"{IN_GAME}: {f}", game.filter(pl.col("family") == f)) for f in data.FAMILIES[:-1]),
        *((f"{SHOOTOUT}: {k}", shoot.filter(e)) for k, e in (("all", True), ("men", men))),
        (f"{SHOOTOUT}: women", shoot.filter(women)),
    ]
    rows = []
    for label, g in (pair for pair in groups if pair[1].height):
        xg = g["provider_xg"].cast(pl.Float64)
        mean = None if xg.null_count() else float(xg.sum()) / g.height
        rate = beta_interval(int(g["is_goal"].sum()), g.height)
        rows.append(PenaltyRow(label, rate, g["match_id"].n_unique(), mean))
    return tuple(rows)


@dataclass(frozen=True)
class Spread:
    """How exchangeable the in-game kicks are: dispersion across competition-seasons, takers.

    ``dispersion`` (Pearson chi2 / df) and ``p_value`` are ``None`` with one group or a
    pooled rate of 0 or 1.
    """

    groups: int
    dispersion: float | None
    df: int
    p_value: float | None
    top_taker_kicks: int
    kicks: int


def exchangeability(penalties: pl.DataFrame) -> Spread:
    """Pearson dispersion of in-game conversion across competition-seasons, and the top taker.

    chi2 = Σ over competition-seasons of (goals - n p)² / (n p (1 - p)), with p the pooled
    rate, on ``groups - 1`` df: a dispersion near 1 is what binomial noise alone gives.

    Raises
    ------
    ValueError
        No in-game penalty; a null ``player_id`` or ``is_goal`` among them.

    Examples
    --------
    >>> kicks = pl.DataFrame({"reason": ["penalty"] * 4, "is_goal": [True, False, True, True],
    ...                       "competition_id": ["sb:1", "sb:1", "sb:2", "sb:2"],
    ...                       "season_id": ["sb:9"] * 4,
    ...                       "player_id": ["sb:p1", "sb:p1", "sb:p1", "sb:p2"]})
    >>> s = exchangeability(kicks)
    >>> s.groups, s.df, round(s.dispersion, 4), s.top_taker_kicks, s.kicks
    (2, 1, 1.3333, 3, 4)
    """
    game = penalties.filter(pl.col("reason") == "penalty")
    if game.height == 0:
        raise ValueError("penalties: no in-game penalty")
    for col in ("player_id", "is_goal"):
        if nulls := game[col].null_count():
            raise ValueError(f"penalties: {col} has {nulls} null value(s) among in-game kicks")
    by = game.group_by("competition_id", "season_id").agg(n=pl.len(), goals=pl.col("is_goal").sum())
    n, k = by["n"].to_numpy().astype(np.float64), by["goals"].to_numpy().astype(np.float64)
    p, df = float(k.sum() / n.sum()), by.height - 1
    dispersion = p_value = None
    if df > 0 and 0.0 < p < 1.0:
        chi2 = float(np.sum((k - n * p) ** 2 / (n * p * (1.0 - p))))
        dispersion, p_value = chi2 / df, float(stats.chi2.sf(chi2, df))
    top = int(game.group_by("player_id").len()["len"].to_numpy().max())
    return Spread(by.height, dispersion, df, p_value, top, game.height)


def _interval(r: Rate) -> str:
    return f"{r.mean:.4f} [{r.low:.4f}, {r.high:.4f}]"


def _constant_loss(r: Rate, q: float | None) -> str:
    if q is None or not 0.0 < q < 1.0:
        return "n/a"
    return f"{-(r.goals * math.log(q) + (r.n - r.goals) * math.log1p(-q)) / r.n:.4f}"


def _penalty_check(r: Rate, team: Rate | None = None) -> list[str]:
    lo, hi = PENALTY_RANGE
    bound, name = (lo, "lower") if r.mean - lo <= hi - r.mean else (hi, "upper")
    # a mean on a bound is inside the window: "above the lower", "below the upper"
    where = "above" if r.mean > bound or (name == "lower" and r.mean == bound) else "below"
    result = "met" if lo <= r.mean <= hi else "missed"
    lines = [
        "| Target | Measured | Result |",
        "|---|---|---|",
        f"| posterior mean in [{lo:.2f}, {hi:.2f}] (milestone wording) | {_interval(r)}; "
        f"{abs(r.mean - bound):.4f} {where} the {name} bound | {result} |",
    ]
    if outside := [w for w, out in (("below", r.low < lo), ("above", r.high > hi)) if out]:
        width, window = r.high - r.low, hi - lo
        lines += [
            "",
            f"The 95% interval is {width:.3f} wide, {'wider than' if width > window else 'against'}"
            f" the {window:.2f} window, and extends {' and '.join(outside)} it: the check is on "
            "the point estimate.",
        ]
    if team is not None and team.n < r.n:
        rest = beta_interval(r.goals - team.goals, r.n - team.n)
        where = "inside" if lo <= rest.mean <= hi else "outside"
        lines += [
            "",
            f"Without the team-selected seasons ({team.n:,} kicks, conversion {team.rate:.3f}) "
            f"the pooled estimate is {_interval(rest)} from {rest.n:,} kicks, {where} the "
            "window.",
        ]
    return lines


def _spread_line(s: Spread) -> str:
    disp = f"the Pearson dispersion is not estimable ({s.df} df)"
    if s.dispersion is not None and s.p_value is not None:
        disp = f"the Pearson dispersion is {s.dispersion:.2f} on {s.df} df (p {s.p_value:.2f})"
    return (
        f"Kicks are pooled as exchangeable: across {s.groups:,} competition-seasons {disp}, and "
        f"the most frequent taker has {s.top_taker_kicks:,} of {s.kicks:,} kicks."
    )


def penalty_card(rows: Sequence[PenaltyRow], spread: Spread, context: CardContext) -> str:
    """The ``xg-penalty`` model card, from :func:`penalty_rows` and :func:`exchangeability`.

    Raises
    ------
    ValueError
        ``rows`` does not start with the in-game all row.

    Examples
    --------
    >>> card = penalty_card(penalty_rows(xg.penalties), exchangeability(xg.penalties),
    ...                     card_context(xg, dev, held))  # doctest: +SKIP
    >>> card.splitlines()[0]  # doctest: +SKIP
    '# Model card: Penalty conversion (Beta-binomial)'
    """
    if not rows or rows[0].label != f"{IN_GAME}: all":
        raise ValueError(f"penalty_card: the first row must be '{IN_GAME}: all'")
    by, c, r, (a0, b0) = {row.label: row for row in rows}, context, rows[0].rate, JEFFREYS
    dev, held = by.get(f"{IN_GAME}: dev"), by.get(f"{IN_GAME}: hold-out")
    shoot, team = by.get(f"{SHOOTOUT}: all"), by.get(f"{IN_GAME}: team-selected")
    validation = benchmarks = [
        f"The {'hold-out' if held is None else 'dev'} has no in-game penalties."
    ]
    if dev is not None and held is not None:
        h, inside = held.rate, held.rate.low <= dev.rate.mean <= held.rate.high
        validation = [
            f"The hold-out ({c.holdout_label}) has {h.n:,} in-game kicks, so its 95% interval "
            f"[{h.low:.4f}, {h.high:.4f}] is {h.high - h.low:.3f} wide and this is a weak test: "
            f"the dev estimate {dev.rate.mean:.4f} lies {'inside' if inside else 'outside'} it."
        ]
        benchmarks = [
            f"Each model as a constant on the {h.n:,} hold-out kicks:",
            "",
            "| Model | Kicks | Log loss |",
            "|---|---|---|",
            f"| Dev posterior mean ({dev.rate.mean:.4f}) | {h.n:,} | "
            f"{_constant_loss(h, dev.rate.mean)} |",
            f"| StatsBomb mean xG ({fmt(held.provider_xg)}) | {h.n:,} | "
            f"{_constant_loss(h, held.provider_xg)} |",
        ]
    families = (*data.FAMILIES[:-1], "women")
    spread_by_family = "; ".join(
        f"{f} {by[k].rate.rate:.3f} ({by[k].rate.n:,} kicks)"
        for f in families
        if (k := f"{IN_GAME}: {f}") in by
    )
    shootouts = (
        f"{shoot.rate.n:,} shoot-out kicks (period 5) cluster in {shoot.matches:,} matches; they "
        "are reported apart, never pooled, and their interval treats kicks of one shoot-out as "
        "independent."
        if shoot
        else "There are no shoot-out kicks."
    )
    metrics = [
        f"| {row.label} | {row.rate.n:,} | {row.matches:,} | {row.rate.goals:,} | "
        f"{row.rate.rate:.4f} | {_interval(row.rate)} | {fmt(row.provider_xg)} |"
        for row in rows
    ]
    sections: list[tuple[str, list[str]]] = [
        ("Overview", [
            f"- **Name / version:** `{PENALTY_NAME}` v{len(PENALTY_CHANGELOG)}",
            "- **Task:** P(goal | in-game penalty), periods 1–4; shoot-out kicks reported apart",
            "- **Status:** experimental",
            f"- M1.5: pooled conversion of in-game penalties. P(goal) = {_interval(r)} (95% "
            f"interval) from {r.n:,} kicks in {rows[0].matches:,} matches.",
        ]),
        ("Intended use", [
            f"The penalty term of match xG. {MATCH_XG} Not for rating takers or keepers: every "
            "in-game penalty gets the same probability."
        ]),
        ("Data", [
            f"- **Source(s):** StatsBomb open data, revision `{c.source_revision}`; snapshot "
            f"`{c.manifest_name}`, data digest `{c.data_digest}`",
            "- **Rows:** the shots the xG population excludes as in-game penalties, dev and "
            f"hold-out ({c.holdout_label}) together, from {spread.groups:,} competition-seasons.",
        ]),
        ("Target", ["`is_goal`: the penalty kick went in."]),
        ("Features", ["None: one probability for every in-game penalty."]),
        ("Model", [
            f"Beta-binomial with the Jeffreys prior Beta({a0:g}, {b0:g}): the posterior is "
            f"Beta({a0:g} + goals, {b0:g} + misses); P(goal) is its mean and the interval its "
            f"{LEVEL:.0%} equal-tailed interval. Nothing is fitted per taker, keeper or "
            "competition."
        ]),
        ("Validation design", ["Nothing is tuned; the model pools all kicks.", *validation]),
        ("Metrics", [
            "| Group | Kicks | Matches | Goals | Conversion | P(goal) [95% interval] "
            "| StatsBomb mean xG |",
            "|---|---|---|---|---|---|---|",
            *metrics, "", "### Milestone check", "",
            *_penalty_check(r, None if team is None else team.rate),
        ]),
        ("Benchmarks", benchmarks),
        ("Limitations & known biases", [
            "- No taker, keeper or foot: the model cannot tell a specialist from anyone else.",
            f"- {_spread_line(spread)}",
            f"- In-game conversion by family: {spread_by_family}.",
            f"- {shootouts}",
        ]),
        ("Data licensing", [LICENSING]),
        ("Changelog", [f"- {item}" for item in PENALTY_CHANGELOG]),
    ]  # fmt: skip
    body = "\n\n".join(f"## {title}\n" + "\n".join(text) for title, text in sections)
    return f"# Model card: {PENALTY_TITLE}\n\n{body}\n"
