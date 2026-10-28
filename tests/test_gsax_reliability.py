"""Split-half reliability tiers (spec ``runs/MS6/gsax`` §4.6, PM ruling X5).

Tiers count each keeper's total open-play rated shots over all seasons; every tier in ``TIERS``
is returned, an empty one with 0 keepers. ``first``: ``split_half`` on each tier keeper's first T
rows in X5 order (``match_date``, ``period``, ``seconds_in_period``, ``shot_id``), value
``q − goal``, ``unit="row"``, ``min_rows=T``; ``all_shots``: the same on all their rows; ``*_sb``
the Spearman-Brown step-up of r and its interval.

The fixture's rows are shuffled, ``shot_id`` order is not time order, and period-2 shots have
smaller ``seconds_in_period`` than period-1 shots, so any other order picks other rows.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import gsax_cases as gc
import polars as pl

from football_shot_models import metrics

ORDER = list(metrics.SPLIT_ORDER["row"])
SIZES = {  # keeper -> open-play rows (r6 over two seasons)
    "sb:r1": 300, "sb:r2": 300, "sb:r3": 300, "sb:r4": 300, "sb:r5": 300, "sb:r6": 260,
    "sb:r7": 249, "sb:r8": 120, "sb:r9": 99, "sb:r10": 100,
}  # fmt: skip


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _tier_frame() -> pl.DataFrame:
    cols: dict[str, list[Any]] = {
        k: [] for k in ("gk", "season", "match", "goal", "xgot", "shot_id", "date", "period",
                        "seconds")
    }  # fmt: skip
    for k, (gk, n) in enumerate(SIZES.items()):
        for j in range(n):
            mi = j // 30
            second = gk == "sb:r6" and j >= 140
            cols["gk"].append(gk)
            cols["season"].append("B18" if second else "A18")
            cols["match"].append(f"sb:{gk[3:]}m{mi:02d}")
            cols["date"].append(dt.date(2019 if second else 2018, 9, 1) + dt.timedelta(days=7 * mi))
            cols["period"].append(1 if j % 30 < 15 else 2)
            cols["seconds"].append(1000.0 + 10 * (j % 15) if j % 30 < 15 else 10.0 * (j % 15))
            early = (j + k) % (3 + k % 4) == 0
            cols["goal"].append(early if j < 250 else (j + k) % 2 == 0)
            cols["xgot"].append(0.2 + 0.05 * (k % 5) + 0.1 * ((j * 3) % 7) / 6)
            cols["shot_id"].append(f"sb:{gk[3:]}-{(j * 7919) % 1000:04d}")
    f = gc.frame(gk=cols["gk"], season=cols["season"], match=cols["match"], goal=cols["goal"],
                 xgot=cols["xgot"], shot_id=cols["shot_id"], date=cols["date"],
                 period=cols["period"], seconds=cols["seconds"])  # fmt: skip
    perm = [(i * 7) % f.height for i in range(f.height)]
    assert sorted(perm) == list(range(f.height))
    return gc.rated(f[perm])


def _triple(x: Any) -> tuple[float | None, float | None, float | None]:
    if hasattr(x, "r"):
        return x.r, x.low, x.high
    a, b, c = x
    return a, b, c


def _close(a: tuple[Any, ...], b: tuple[Any, ...]) -> bool:
    return all(
        (x is None and y is None) or (x is not None and y is not None and abs(x - y) <= 1e-12)
        for x, y in zip(a, b, strict=True)
    )


def _split(rows: pl.DataFrame, t: int) -> metrics.SplitHalf:
    v = rows.with_columns(v=pl.col("q") - pl.col("is_goal").cast(pl.Float64))
    return metrics.split_half(v, by="gk_id", value="v", min_rows=t, unit="row")


def test_tiers_first_t() -> None:
    """Tier keepers by total shots over seasons (249 out of tier 250, 100 in tier 100, 99 out;
    a keeper reaching 250 only over two seasons in); ``first`` is ``split_half`` on each one's
    first T rows in X5 order, ``all_shots`` on all their rows; ``*_sb`` is ``spearman_brown``
    of r, low and high."""
    gsax = _gsax()
    rated = _tier_frame()
    tiers = gsax.reliability(rated)
    assert [t.tier for t in tiers] == [100, 250, 500]
    want_keepers = {100: 9, 250: 6, 500: 0}
    for tier in tiers:
        t = tier.tier
        assert tier.keepers == want_keepers[t], t
        members = [gk for gk, n in SIZES.items() if n >= t]
        rows = rated.filter(pl.col("gk_id").is_in(members))
        first = rows.sort(["gk_id", *ORDER]).filter(pl.int_range(pl.len()).over("gk_id") < t)
        exp_first, exp_all = _split(first, t), _split(rows, t)
        for got, exp in ((tier.first, exp_first), (tier.all_shots, exp_all)):
            assert got.keys == exp.keys, t
            assert _close((got.r, got.low, got.high), (exp.r, exp.low, exp.high)), (t, got, exp)
        for got_sb, base in ((tier.first_sb, tier.first), (tier.all_shots_sb, tier.all_shots)):
            want = tuple(metrics.spearman_brown(x) for x in (base.r, base.low, base.high))
            assert _close(_triple(got_sb), want), t
        if t in (100, 250):
            assert exp_first.r is not None and exp_all.r is not None  # precondition
            wrong = rows.sort(["gk_id", "shot_id"]).filter(
                pl.int_range(pl.len()).over("gk_id") < t
            )  # precondition: another order picks other rows and another r
            assert abs((_split(wrong, t).r or 0.0) - exp_first.r) > 1e-6, t
    assert abs((tiers[1].first.r or 0.0) - (tiers[1].all_shots.r or 0.0)) > 1e-6


def test_empty_tier_reported() -> None:
    """Every tier is returned, in ``TIERS`` order, whether or not a keeper reaches it: an empty
    tier has 0 keepers and r None; an empty frame gives three empty tiers."""
    gsax = _gsax()
    tiers = gsax.reliability(_tier_frame())
    assert len(tiers) == len(gsax.TIERS)
    empty = tiers[2]
    assert empty.tier == 500 and empty.keepers == 0
    assert empty.first.r is None and empty.all_shots.r is None
    assert _triple(empty.first_sb) == (None, None, None)
    none = gsax.reliability(_tier_frame().clear())
    assert [(t.tier, t.keepers, t.first.r) for t in none] == [(100, 0, None), (250, 0, None),
                                                              (500, 0, None)]  # fmt: skip
