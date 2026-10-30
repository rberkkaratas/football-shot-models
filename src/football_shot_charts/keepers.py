"""The keeper (GSAx) chart: which keeper-seasons are distinguishable from noise.

It replaces the earlier forest plot of every keeper-season. A ranked list of 97 seasons invites
reading the order as skill, and the reliability table says that at 100 shots a keeper is not
distinguishable from noise. So the chart shows only the extremes (the highest and lowest
``TOP_N``), and on each row what the keeper model believes:

- **dot**: shrunk GSAx per 100 on-target shots, the keeper model's posterior mode
  (``gsax_shrunk``); per 100 because shots run 102-185, so a season total partly measures
  workload;
- **band**: its ``gsax.CI_LEVEL`` posterior interval at the estimated keeper spread τ̂ (Laplace:
  ``u ± z·s``, ``s = (h + 1/τ̂²)^−½``, mapped through ``u ↦ Σ q − Σ expit(logit q + u)``); τ̂'s
  own uncertainty is not included, and the chart says so;
- **tick**: the unshrunk (rated) GSAx per 100, the noisy value shrinkage pulls in; no number
  is printed for it.

The title counts the rows whose interval excludes 0 over *all* keeper-seasons, not only those
drawn; the subtitle is the card's reliability verdict (``gsax_cards.caveat``); the strip sets
the keeper spread beside the median interval width, which is why nothing clears 0.

Team-selected seasons are rated against the model, not their league (δ = 0: their league sum is
one team's keeper against its opponents):
they are flagged (``team_selected="flag"``, with ``FLAG``: the card's † reads as "+" at chart
size, beside signed values) with the card's selected-side gap in the footer,
or left out and rated by the keeper model fitted without them (``"drop"``, Berk's publishing
switch). Keeper names appear only on StatsBomb charts (the private archive publishes counts and
metrics only): a WhoScored view is unnamed, its labels rank-prefixed competition-seasons.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from statistics import NormalDist
from typing import Final, Literal

import numpy as np
import polars as pl
from football_core import viz
from scipy.special import expit

from football_shot_charts import rows
from football_shot_models import gsax, metrics
from football_shot_models.gsax import KEEPER_SEASON, TEAM_SELECTED, GsaxFit, KeeperModel

Mode = Literal["flag", "drop"]

TOP_N: Final = 3  # highest and lowest shown (a3: 4 overflows the portrait body)
RECON_TOL: Final = 1e-9  # goals: the reconstructed shrunk total against the model's
SIZE: Final = "portrait"
MODEL_TEXT: Final = "xGOT: own model, fitted without the rated season"
GID: Final = "keepers"
FLAG: Final = "*"  # a team-selected row; never "†", which reads as "+" beside signed values
# the one chart writer (``rows.write_all``), here so gsax_cards imports this module only
write_all = rows.write_all
_ROW_SCHEMA: Final = pl.Schema({
    "rank": pl.Int64(), "label": pl.String(), "sublabel": pl.String(),
    "team_selected": pl.Boolean(), "shots": pl.Int64(), "shrunk": pl.Float64(),
    "low": pl.Float64(), "high": pl.Float64(), "unshrunk": pl.Float64(), "clear": pl.Boolean(),
})  # fmt: skip


@dataclass(frozen=True, eq=False)
class KeeperView:
    """What the keeper chart draws, computed from a :class:`~football_shot_models.gsax.GsaxFit`.

    Attributes
    ----------
    rows : polars.DataFrame
        Every keeper-season with at least ``min_shots`` shots (columns in ``_ROW_SCHEMA``, per
        100 shots), sorted by ``shrunk`` descending, then ``KEEPER_SEASON``.
    n, k, k_no_team : int
        Rows, rows whose interval excludes 0 (``clear``), and those not team-selected.
    min_shots : int
    level : float
        The interval level (``gsax.CI_LEVEL``).
    spread_100 : float
        The keeper model's SD per 100 shots (``KeeperModel.per_100``).
    spread_100_no_team : float or None
        The same without team-selected rows (None without that model).
    at_bound : bool
        τ̂ at its optimiser bound.
    median_width_100 : float
        Median ``high − low`` over all rows.
    caveat : str
        The subtitle: the reliability verdict.
    team_gap_100 : float or None
        The team-selected selected side's GSAx per 100 shots (the card's row).
    named : bool
    mode : {"flag", "drop"}
    left_out : int
        Team-selected rows left out (drop mode).
    flag : str
        The mark of a team-selected row (``FLAG``).
    """

    rows: pl.DataFrame
    n: int
    k: int
    k_no_team: int
    min_shots: int
    level: float
    spread_100: float
    spread_100_no_team: float | None
    at_bound: bool
    median_width_100: float
    caveat: str
    team_gap_100: float | None
    named: bool
    mode: Mode
    left_out: int
    flag: str = FLAG


def shrunk_interval(model: KeeperModel, rated: pl.DataFrame) -> pl.DataFrame:
    """Each keeper-season's shrunk GSAx and its ``gsax.CI_LEVEL`` posterior interval, in goals.

    The model's groups are rebuilt from the "without penalties" rows of ``rated`` (the rows it
    was fitted on), ``u`` and ``h = Σ p(1 − p)`` come from ``gsax.keeper_modes`` at τ̂, and the
    interval is ``u ± z·s`` mapped through the decreasing ``u ↦ Σ q − Σ expit(logit q + u)``
    (so ``low`` comes from ``u + z·s``). Arithmetic only, no refit.

    Returns
    -------
    polars.DataFrame
        ``KEEPER_SEASON``, ``gsax_shrunk``, ``low``, ``high`` (Float64), one row per
        ``model.estimates`` row, in its order.

    Raises
    ------
    ValueError
        ``model.by`` is not ``KEEPER_SEASON``; ``rated`` rows outside the model's groups or a
        group without rows; a reconstructed shrunk total off ``estimates["gsax_shrunk"]`` by
        more than ``RECON_TOL`` goals.

    Examples
    --------
    >>> r = _example_rated()
    >>> m = gsax.fit_keeper_model(r, by=KEEPER_SEASON)
    >>> iv = shrunk_interval(m, r)
    >>> bool(((iv["low"] <= iv["gsax_shrunk"]) & (iv["gsax_shrunk"] <= iv["high"])).all())
    True
    """
    if model.by != KEEPER_SEASON:
        raise ValueError(f"shrunk_interval: the model is by {model.by}, not {KEEPER_SEASON}")
    est = model.estimates.with_row_index("code")
    open_ = gsax.variant(rated, gsax.WITHOUT).join(
        est.select(*KEEPER_SEASON, "code"), on=list(KEEPER_SEASON), how="left",
        maintain_order="left",
    )  # fmt: skip
    if n := open_["code"].null_count():
        raise ValueError(f"shrunk_interval: {n} rated row(s) are outside the model's groups")
    grp = open_["code"].to_numpy().astype(np.intp)
    g = est.height
    if (counts := np.bincount(grp, minlength=g)).min() == 0:
        raise ValueError(f"shrunk_interval: {int((counts == 0).sum())} group(s) have no row")
    q = open_["q"].to_numpy().astype(np.float64)
    off, y = metrics.logit(q), open_["is_goal"].cast(pl.Float64).to_numpy()
    u, h = gsax.keeper_modes(off, y, grp, tau=model.tau)
    sum_q = np.bincount(grp, q, g)

    def gsax_at(shift: np.ndarray) -> np.ndarray:
        out: np.ndarray = sum_q - np.bincount(grp, expit(off + shift[grp]), g)
        return out

    shrunk = gsax_at(u)
    if (gap := float(np.abs(shrunk - est["gsax_shrunk"].to_numpy()).max())) > RECON_TOL:
        raise ValueError(f"shrunk_interval: the shrunk totals differ from the model by {gap:.3g}")
    zs = NormalDist().inv_cdf((1 + gsax.CI_LEVEL) / 2) / np.sqrt(h + 1 / model.tau**2)
    return est.select(KEEPER_SEASON).with_columns(
        gsax_shrunk=pl.Series(shrunk), low=pl.Series(gsax_at(u + zs)),
        high=pl.Series(gsax_at(u - zs)),
    )  # fmt: skip


def keeper_view(
    g: GsaxFit,
    *,
    names: pl.DataFrame | None,
    caveat: str,
    flag: str,
    team_gap_100: float | None,
    team_selected: Mode = "flag",
) -> KeeperView:
    """The keeper chart's rows and numbers: ``gsax.keeper_rows`` joined to
    :func:`shrunk_interval` of ``g.model`` (``"flag"``) or of ``g.model_no_team`` with the
    team-selected rows left out (``"drop"``), and, when ``names`` is given, to the names.

    Raises
    ------
    ValueError
        No row; a named view with a ``gk_id`` that is not ``sb:`` (names are StatsBomb only) or a
        missing name;
        a null shrunk value; ``keeper_rows`` shots differing from the model's on any row;
        ``"drop"`` without ``g.model_no_team``; an unknown ``team_selected``.

    Examples
    --------
    >>> v = keeper_view(g, names=names, caveat=c, flag="†", team_gap_100=None)  # doctest: +SKIP
    >>> v.rows.columns  # doctest: +SKIP
    ['rank', 'label', 'sublabel', 'team_selected', 'shots', 'shrunk', 'low', 'high', ...]
    """
    if team_selected not in ("flag", "drop"):
        raise ValueError(f"team_selected must be 'flag' or 'drop', not {team_selected!r}")
    if g.model.by != KEEPER_SEASON:
        raise ValueError(f"the keeper model is by {g.model.by}, not {KEEPER_SEASON}")
    table = gsax.keeper_rows(g).with_columns(
        team_selected=pl.col("family").cast(pl.String) == TEAM_SELECTED
    )
    if table.height == 0:
        raise ValueError(f"no keeper-season reaches {gsax.TABLE_MIN_SHOTS} shots: nothing to chart")
    model, rated, left_out = g.model, g.rated, 0
    if team_selected == "drop":
        if g.model_no_team is None:
            raise ValueError("drop mode needs the keeper model without team-selected rows")
        model = g.model_no_team
        rated = rated.filter(pl.col("family").cast(pl.String) != TEAM_SELECTED)
        left_out = int(table["team_selected"].sum())
        table = table.filter(~pl.col("team_selected"))
        if table.height == 0:
            raise ValueError("no keeper-season is left without the team-selected ones")
    if names is not None:
        if bad := table.filter(~pl.col("gk_id").str.starts_with("sb:")).height:
            raise ValueError(f"{bad} keeper id(s) are not StatsBomb ('sb:'): names are Rule 1")
        table = table.join(names.select(gk_id="player_id", name="player_name").unique("gk_id"),
                           on="gk_id", how="left")  # fmt: skip
        if missing := table.filter(pl.col("name").is_null()).height:
            raise ValueError(f"{missing} keeper(s) are missing from names")
    iv = shrunk_interval(model, rated).join(
        model.estimates.select(*KEEPER_SEASON, model_shots="shots"), on=list(KEEPER_SEASON)
    )
    table = table.drop("gsax_shrunk").join(iv, on=list(KEEPER_SEASON), how="left")
    if n := table["gsax_shrunk"].null_count():
        raise ValueError(f"{n} keeper-season(s) have no shrunk value")
    if n := table.filter(pl.col("shots") != pl.col("model_shots")).height:
        raise ValueError(f"{n} keeper-season(s): keeper_rows and the model disagree on shots")
    per = 100 / pl.col("shots")
    out = (
        table.with_columns(shrunk=pl.col("gsax_shrunk") * per, low=pl.col("low") * per,
                           high=pl.col("high") * per, unshrunk=pl.col("gsax") * per)
        .sort(["shrunk", *KEEPER_SEASON], descending=[True, False, False, False])
        .with_row_index("rank", offset=1)
        .with_columns(rank=pl.col("rank").cast(pl.Int64),
                      clear=(pl.col("low") > 0) | (pl.col("high") < 0))
    )  # fmt: skip
    mark = pl.when(pl.col("team_selected")).then(pl.lit(f" {flag}")).otherwise(pl.lit(""))
    season = pl.concat_str("competition_name", pl.lit(" "), "season_name")
    if names is not None:
        label, sub = pl.concat_str("name", mark), season
    else:
        label = pl.concat_str(pl.lit("#"), pl.col("rank").cast(pl.String), pl.lit(" · "), season,
                              mark)  # fmt: skip
        sub = pl.concat_str(pl.col("shots").cast(pl.String), pl.lit(" shots"))
    out = out.with_columns(label=label, sublabel=sub).select(list(_ROW_SCHEMA))
    clear = out["clear"]
    nt = g.model_no_team
    return KeeperView(
        rows=out, n=out.height, k=int(clear.sum()),
        k_no_team=int((clear & ~out["team_selected"]).sum()), min_shots=gsax.TABLE_MIN_SHOTS,
        level=gsax.CI_LEVEL, spread_100=model.per_100,
        spread_100_no_team=None if nt is None else nt.per_100, at_bound=model.at_bound,
        median_width_100=float(np.median((out["high"] - out["low"]).to_numpy())), caveat=caveat,
        team_gap_100=team_gap_100, named=names is not None, mode=team_selected,
        left_out=left_out, flag=flag,
    )  # fmt: skip


# --- Words ------------------------------------------------------------------------------------


def _verb(k: int) -> str:
    return "differs" if k == 1 else "differ"


def title(view: KeeperView) -> str:
    """The statement: how many keeper-seasons (of all, not only those drawn) have an interval
    excluding 0, and how many without the flagged ones when a clear row is flagged.

    Examples
    --------
    >>> title(_example_view())
    'After shrinkage, 1 of 4 keeper-seasons differs clearly from average (0 without *)'
    """
    if view.k == 0:
        return f"After shrinkage, none of {view.n} keeper-seasons differs clearly from average"
    out = (f"After shrinkage, {view.k} of {view.n} keeper-seasons {_verb(view.k)} clearly from "
           "average")  # fmt: skip
    flagged = bool((view.rows["clear"] & view.rows["team_selected"]).any())
    return out + (f" ({view.k_no_team} without {view.flag})" if flagged else "")


def strip(view: KeeperView) -> tuple[viz.Stat, ...]:
    """Keeper spread beside the median interval width (both per 100 shots).

    Examples
    --------
    >>> [s.note for s in strip(_example_view())]
    ['model estimate, SD per 100 · n/a without *', '90% width, per 100 shots']
    """
    # at the bound "SD per 100" gives way: the note must fit its column in both fonts
    est = "model estimate at its bound" if view.at_bound else "model estimate, SD per 100"
    if view.mode == "drop":
        note = f"{est}, without {view.flag}"
    else:
        other = "n/a" if view.spread_100_no_team is None else f"{view.spread_100_no_team:.1f}"
        note = f"{est} · {other} without {view.flag}"
    return (
        viz.Stat(f"{view.spread_100:.1f}", "Keeper spread", note),
        viz.Stat(f"{view.median_width_100:.1f}", "Median interval",
                 f"{view.level:.0%} width, per 100 shots"),
    )  # fmt: skip


def footer_note(view: KeeperView) -> str | None:
    """The flag's sentence when a drawn row is team-selected (flag mode), the left-out count
    (drop mode), else none. The gap is the selected sides' keepers over every team-selected
    season (``gsax_cards.team_gap_100``), so the sentence names that group, not the drawn rows.

    Examples
    --------
    >>> footer_note(_example_view())
    "* team-selected season: rated against the model, not its league; the selected sides' keepers sit 1.2 per 100 above it"
    """  # noqa: E501
    if view.mode == "drop":
        return f"{view.left_out} team-selected keeper-seasons left out"
    if not bool(shown(view)["team_selected"].any()):
        return None
    out = f"{view.flag} team-selected season: rated against the model, not its league"
    if view.team_gap_100 is None:
        return out
    side = "above" if view.team_gap_100 >= 0 else "below"
    return (f"{out}; the selected sides' keepers sit {abs(view.team_gap_100):.1f} per 100 "
            f"{side} it")  # fmt: skip


def shown(view: KeeperView) -> pl.DataFrame:
    """The drawn rows: all when ``n <= 2 * TOP_N``, else the highest and lowest ``TOP_N``."""
    r = view.rows
    return r if r.height <= 2 * TOP_N else pl.concat([r.head(TOP_N), r.tail(TOP_N)])


def _divider(hidden: pl.DataFrame) -> rows.Divider:
    m, j = hidden.height, int(hidden["clear"].sum())
    what = "1 keeper-season" if m == 1 else f"{m} keeper-seasons"
    verdict = "none differs clearly" if j == 0 else f"{j} {_verb(j)} clearly"
    return rows.Divider(f"{what} in between, {verdict}")


def panel(view: KeeperView) -> tuple[rows.Panel, tuple[float, float]]:
    """The row panel and its scale (``rows.scale_of`` over the drawn low, high and unshrunk).

    Examples
    --------
    >>> p, scale = panel(_example_view())
    >>> [type(r).__name__ for r in p.rows], p.rows[0].value
    (['Row', 'Row', 'Row', 'Row'], '+1.5')
    """

    cols = ("label", "sublabel", "shrunk", "low", "high", "unshrunk")
    items: list[rows.Row | rows.Divider] = [
        rows.Row(label=str(label), sublabel=str(sub), value=rows.signed(float(shrunk)),
                 point=float(shrunk), low=float(low), high=float(high), tick=float(raw))
        for label, sub, shrunk, low, high, raw in shown(view).select(cols).iter_rows()
    ]  # fmt: skip
    r = view.rows
    if r.height > 2 * TOP_N:
        items.insert(TOP_N, _divider(r.slice(TOP_N, r.height - 2 * TOP_N)))
    drawn = shown(view)
    scale = rows.scale_of([*drawn["low"], *drawn["high"], *drawn["unshrunk"]])
    p = rows.Panel(
        f"GSAx per 100 on-target shots · {view.min_shots}+ shots, penalties excluded",
        (("dot", "After shrinkage"), ("tick", "Before shrinkage"),
         ("band", f"{view.level:.0%} interval")),
        tuple(items),
        caption="Right of 0: fewer open-play goals than the model expects of an average keeper "
        "facing the same on-target shots.\nBands: keeper-model posterior at the estimated keeper "
        "spread (its own uncertainty not included).",
        scale_row=True,
    )  # fmt: skip
    return p, scale


def keeper_chart(
    view: KeeperView, *, theme: str | viz.Theme, footer: viz.Footer, logo: viz.StrPath | None
) -> viz.Canvas:
    """The keeper chart (portrait): statement title, the caveat as subtitle, the 2-stat strip,
    the row panel top-aligned in the body; the footer credits ``MODEL_TEXT`` (unless
    ``footer.model`` is set) and carries :func:`footer_note`.

    Raises
    ------
    ValueError
        From ``viz.canvas`` (a part over its line limit); ``rows.NoLayout`` when the panel does
        not fit the body.

    Examples
    --------
    >>> c = keeper_chart(_example_view(), theme="dark", footer=viz.Footer(("whoscored",)),
    ...                  logo=None)
    >>> viz.check(c)
    []
    """
    foot = dataclasses.replace(footer, model=footer.model or MODEL_TEXT, note=footer_note(view))
    c = viz.canvas(theme, SIZE, title=title(view), subtitle=view.caveat, stats=strip(view),
                   footer=foot, logo=logo)  # fmt: skip
    bw, bh = rows.body_px(c)
    p, scale = panel(view)
    m = rows.place(c, p, (0.0, 0.0, bw, bh))
    rows.draw_panel(c, (0.0, bh - m.height, bw, m.height), m, scale=scale, gid=GID)
    return c


# --- Doctest fixtures -------------------------------------------------------------------------


def _example_rated() -> pl.DataFrame:
    """Made-up open-play rows of three keeper-seasons (no provider row)."""
    n = 40
    keys = [("sb:1", "c", "s"), ("sb:2", "c", "s"), ("sb:3", "c", "s")]
    rng = np.random.default_rng(0)
    rows_ = [(k, float(q), bool(rng.random() < q)) for k in keys
             for q in rng.uniform(0.1, 0.5, n)]  # fmt: skip
    return pl.DataFrame({
        "gk_id": [r[0][0] for r in rows_], "competition_id": [r[0][1] for r in rows_],
        "season_id": [r[0][2] for r in rows_], "kind": ["open_play"] * len(rows_),
        "q": [r[1] for r in rows_], "is_goal": [r[2] for r in rows_],
    })  # fmt: skip


def _example_view() -> KeeperView:
    """A made-up four-row view, one team-selected row clear of 0."""
    r = pl.DataFrame(
        {
            "rank": [1, 2, 3, 4], "label": ["#1 · L 2015/2016 †", "#2 · L 2015/2016",
                                            "#3 · L 2015/2016", "#4 · L 2015/2016"],
            "sublabel": ["120 shots", "130 shots", "110 shots", "140 shots"],
            "team_selected": [True, False, False, False], "shots": [120, 130, 110, 140],
            "shrunk": [1.5, 0.4, -0.2, -0.9], "low": [0.1, -2.0, -2.6, -3.1],
            "high": [3.0, 2.7, 2.2, 1.4], "unshrunk": [4.2, 1.1, -0.8, -2.5],
            "clear": [True, False, False, False],
        },
        schema=_ROW_SCHEMA,
    )  # fmt: skip
    return KeeperView(
        r,
        4,
        1,
        0,
        100,
        0.9,
        3.0,
        None,
        False,
        4.8,
        "Not distinguishable from noise at 100 shots.",
        1.2,
        False,
        "flag",
        0,
    )
