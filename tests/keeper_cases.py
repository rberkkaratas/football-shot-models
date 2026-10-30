"""Hand-built keeper fits and model views for the CH1 keeper and model-card chart tests (spec
``runs/CH1/keeper-model-charts`` §3, §4, §8).

Every id, name and number is invented. A keeper fit is the session's synthetic ``gsax_fits``
fit (``conftest``) with its rated rows, keeper tables and keeper-season models replaced by
rows built here, so ``keeper_rows`` and the keeper charts read controlled values. The keeper
model is built at a *fixed* τ (:func:`model_at`, the modes from ``gsax.keeper_modes``), not by
the optimiser, so a test chooses how far the shrinkage pulls and which intervals clear 0.

``q`` is ``xgot`` scaled by ``Q_SCALE`` (a league offset δ ≠ 0), so the rated GSAx (Σ q − goals)
differs from the raw one (Σ xgot − goals): a chart that marks ``gsax_raw`` instead of ``gsax``
is caught. Team-selected rows sit in the ``T18`` / ``T19`` competition-seasons (family and
``zero_family`` set together, season by season).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import gsax_cases as gc
import numpy as np
import polars as pl
from scipy.special import expit

XGOT = 0.3
Q_SCALE = 0.9  # q = 0.27: an average keeper concedes 40.5 of 150
SHOTS = 150
TAU = 1.0  # logit SD: wide enough that the extremes clear 0 at 150 shots
TEAM_SEASONS = frozenset({"T18", "T19"})

# (gk_id, season, shots, goals) out of key order: ranks 1..12 by shrunk GSAx are the list order.
# Rank 3 / 4 is a tie (identical rows in one season): the smaller key (sb:g33) ranks first and
# is shown, sb:g55 is hidden yet clear of 0. sb:g12 (rank 2, T18) is a team-selected row inside
# the shown 2N; sb:g61 (rank 7, T19) is one outside it.
TWELVE: tuple[tuple[str, str, int, int], ...] = (
    ("sb:g71", "A18", SHOTS, 18),
    ("sb:g12", "T18", SHOTS, 22),
    ("sb:g33", "C18", SHOTS, 26),
    ("sb:g55", "C18", SHOTS, 26),
    ("sb:g40", "A20", SHOTS, 36),
    ("sb:g08", "B19", SHOTS, 39),
    ("sb:g61", "T19", SHOTS, 41),
    ("sb:g09", "A19", SHOTS, 43),
    ("sb:g27", "F1", SHOTS, 46),
    ("sb:g19", "H15", SHOTS, 48),
    ("sb:g84", "A18", SHOTS, 53),
    ("sb:g02", "B18", SHOTS, 62),
)
# worst-case strings (spec §8, a3): an invented 34-char name and a 33-char competition + season
NAME_34 = "Abcdefghij Klmnopqrstu Vwxyzabcdef"
COMP_33 = ("sb:777", "sb:778", "Abcdef Ghijk Lmnop Qrst", "2018/2019")
BANNED = r"best|worst|skill|elite|clinical|lucky|perform"


def rated(
    specs: Sequence[tuple[str, str, int, int]],
    *,
    penalties: Sequence[tuple[str, str, int, int]] = (),
    q_scale: float = Q_SCALE,
) -> pl.DataFrame:
    """A rated frame (``gsax.rate`` output shape, δ = 0 columns from ``gsax_cases.rated``) with
    one block per spec: ``shots`` rows at ``XGOT`` in 3 matches, the first ``goals`` scored, and
    ``q = xgot · q_scale``. ``penalties`` adds penalty rows (kind ``penalty``, xGOT 0.75)."""
    parts = []
    for gk, season, n, goals in specs:
        tag = f"{gk[3:]}{season.lower()}"
        parts.append(gc.block(gk, season, goals=[i < goals for i in range(n)], xgot=XGOT,
                              matches=[f"sb:m{tag}{j}" for j in range(3)]))  # fmt: skip
    for gk, season, n, goals in penalties:
        tag = f"{gk[3:]}{season.lower()}p"
        parts.append(gc.block(gk, season, goals=[i < goals for i in range(n)], xgot=0.75,
                              matches=[f"sb:m{tag}{j}" for j in range(2)],
                              kind="penalty"))  # fmt: skip
    seasons = [s for part in parts for s in part["season"]]
    frame = gc.blocks(*parts, family=[gc.TEAM if s in TEAM_SEASONS else gc.LEAGUE for s in seasons])
    team = pl.Series([s in TEAM_SEASONS for s in seasons], dtype=pl.Boolean())
    return gc.rated(frame).with_columns(
        q=pl.col("xgot") * q_scale,
        zero_family=team,
        delta=pl.lit(0.05, dtype=pl.Float64()),
        group=pl.concat_str("competition_id", "season_id", separator=" "),
    )


def model_at(open_rows: pl.DataFrame, tau: float, *, by: Sequence[str] | None = None) -> Any:
    """A ``gsax.KeeperModel`` of open-play rated rows at a fixed τ: codes over the sorted keys,
    modes from ``gsax.keeper_modes``, ``gsax_shrunk = Σ q − Σ expit(logit q + u)``."""
    from football_shot_models import gsax, metrics

    by = tuple(by or gsax.KEEPER_SEASON)
    keys = open_rows.select(by).unique().sort(by).with_row_index("code")
    codes = (
        open_rows.select(by).join(keys, on=list(by), how="left", maintain_order="left")["code"]
        .to_numpy().astype(np.intp)
    )  # fmt: skip
    q = open_rows["q"].to_numpy()
    off, y = metrics.logit(q), open_rows["is_goal"].cast(pl.Float64).to_numpy()
    u, _ = gsax.keeper_modes(off, y, codes, tau=tau)
    g = len(u)
    shrunk = np.bincount(codes, q, g) - np.bincount(codes, expit(off + u[codes]), g)
    qbar = float(q.mean())
    est = keys.drop("code").with_columns(
        shots=pl.Series(np.bincount(codes, minlength=g), dtype=pl.Int64()),
        u=pl.Series(u, dtype=pl.Float64()),
        gsax_shrunk=pl.Series(shrunk, dtype=pl.Float64()),
    )
    return gsax.KeeperModel(by, tau, False, 100 * tau * qbar * (1 - qbar), est)


def fit_from(
    base: Any, rows: pl.DataFrame, *, tau: float = TAU, tau_no_team: float | None = TAU
) -> Any:
    """``base`` (a ``GsaxFit``) with ``rated``, ``keepers``, ``model`` and ``model_no_team``
    rebuilt from ``rows``; the goal_z sensitivities cleared. ``model_no_team`` is None when
    ``tau_no_team`` is None or no row is team-selected."""
    from football_shot_models import gsax

    open_ = gsax.variant(rows, gsax.WITHOUT)
    team = open_["family"].cast(pl.String) == gsax.TEAM_SELECTED
    no_team = None
    if tau_no_team is not None and team.any():
        no_team = model_at(open_.filter(~team), tau_no_team)
    keepers = {name: gsax.keeper_table(gsax.variant(rows, name)) for name in gsax.VARIANTS}
    return dataclasses.replace(
        base, rated=rows, keepers=keepers, model=model_at(open_, tau), model_no_team=no_team,
        line=None, noz=None, line_seasons=None,
    )  # fmt: skip


def names(ids: Sequence[str], *, long: str | None = None) -> pl.DataFrame:
    """A ``statsbomb.player_names`` frame: ``Keeper <id digits>`` per id (``long`` replaces the
    first one's name)."""
    ids = sorted(set(ids))
    out = [f"Keeper {i[4:]}" for i in ids]
    if long is not None:
        out[0] = long
    return pl.DataFrame({"player_id": ids, "player_name": out,
                         "names": pl.Series([1] * len(ids), dtype=pl.Int64())})  # fmt: skip


def name_of(frame: pl.DataFrame, gk: str) -> str:
    return str(frame.filter(pl.col("player_id") == gk)["player_name"][0])


# ---- model views (diagnostics) -----------------------------------------------------------


def bins(
    mean_p: Sequence[float],
    rate: Sequence[float],
    low: Sequence[float],
    high: Sequence[float],
    *,
    shots: int = 900,
    clipped: Sequence[bool] | None = None,
) -> pl.DataFrame:
    """A ``render.calibration_bins`` frame; ``inside`` computed as ``low <= mean_p <= high``."""
    n = len(mean_p)
    return pl.DataFrame({
        "bin": pl.Series(range(n), dtype=pl.Int64()),
        "shots": pl.Series([shots] * n, dtype=pl.Int64()),
        "mean_p": pl.Series(mean_p, dtype=pl.Float64()),
        "rate": pl.Series(rate, dtype=pl.Float64()),
        "low": pl.Series(low, dtype=pl.Float64()),
        "high": pl.Series(high, dtype=pl.Float64()),
        "inside": pl.Series([lo <= m <= hi for m, lo, hi in zip(mean_p, low, high, strict=True)],
                            dtype=pl.Boolean()),
        "clipped": pl.Series(list(clipped) if clipped is not None else [False] * n,
                             dtype=pl.Boolean()),
    })  # fmt: skip


XG_MEANS = (0.006, 0.021, 0.028, 0.036, 0.046, 0.060, 0.081, 0.117, 0.192, 0.427)
XGOT_MEANS = (0.003, 0.05, 0.12, 0.2, 0.3, 0.42, 0.55, 0.68, 0.79, 0.905)


def band_bins(means: Sequence[float], *, outside: Sequence[int] = (), shots: int = 900) -> Any:
    """Bins around ``means`` (rate = mean · 1.02, band ± 12 % of the mean); each index in
    ``outside`` gets a rate and band above the mean (``inside`` False)."""
    rate, low, high = [], [], []
    for i, m in enumerate(means):
        r = min(m * (1.25 if i in outside else 1.02), 0.98)
        half = 0.12 * m if i not in outside else 0.05 * m
        rate.append(r)
        low.append(max(r - half, 1e-4) if i not in outside else m * 1.1)
        high.append(min(r + half, 0.995))
    return bins(means, rate, low, high, shots=shots)


def model_view(**over: Any) -> Any:
    """A StatsBomb-like xG ``diagnostics.ModelView`` (hold-out split, three references, no
    extras); keywords override fields."""
    from football_shot_charts import diagnostics as dg

    rows = (
        dg.GainRow(role="Best logistic regression", title="LR splines + context + freeze frames",
                   cv=(0.0032, 0.0025, 0.0040), held=(0.0015, -0.0007, 0.0035)),
        dg.GainRow(role="Standard baseline (S0)", title="LR distance + angle + body part",
                   cv=(0.0231, 0.0216, 0.0246), held=(0.0188, 0.0144, 0.0235)),
        dg.GainRow(role="StatsBomb's xG", title="provider model, freeze-frame inputs",
                   cv=(0.0018, 0.0008, 0.0029), held=(-0.0013, -0.0041, 0.0017)),
    )  # fmt: skip
    base: dict[str, Any] = {
        "quantity": "xG", "outcome": "goal", "unit": "shots",
        "split": ("Hold-out", "the hold-out", "hold-out"),
        "contexts": ("39 competition-seasons", "Test League 2015/2016"),
        "cv_bins": band_bins(XG_MEANS), "held_bins": band_bins(XG_MEANS, outside=(9,)),
        "cv_n": 89_895, "held_n": 9_816, "cv_slope": 1.0086, "held_gap": -50.0,
        "held_sd": 26.4, "cv_log_loss": 0.2641, "held_log_loss": 0.2702, "rows": rows,
        "extras": (), "level": 0.95, "n_boot": 2000, "notes": (),
    }  # fmt: skip
    return dg.ModelView(**{**base, **over})


def xgot_view(**over: Any) -> Any:
    """A WhoScored-like xGOT view: test-season split, one reference, two strip extras."""
    from football_shot_charts import diagnostics as dg

    base: dict[str, Any] = {
        "quantity": "xGOT", "unit": "on-target shots",
        "split": ("Test season", "the test season", "test-season"),
        "contexts": ("12 competition-seasons", "Abcdef Ghijk Lmnop Qrst 2018/2019"),
        "cv_bins": band_bins(XGOT_MEANS), "held_bins": band_bins(XGOT_MEANS),
        "cv_n": 29_350, "held_n": 5_102, "cv_slope": None, "held_gap": 12.5, "held_sd": 18.0,
        "cv_log_loss": 0.4030, "held_log_loss": 0.3997,
        "rows": (dg.GainRow(role="Logistic regression", title="LR xG + placement",
                            cv=(0.0097, 0.0078, 0.0116), held=(0.0083, 0.0022, 0.0142)),),
        "extras": (dg.Extra(label="xG only", cv=0.108, held=0.113),
                   dg.Extra(label="base rate", cv=0.202, held=0.210)),
    }  # fmt: skip
    return model_view(**{**base, **over})
