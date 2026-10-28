"""The hierarchical keeper model: ``keeper_modes`` and ``fit_keeper_model`` (spec
``runs/MS6/gsax`` §4.4, §6).

``goal ~ offset(logit q) + (1 | group)``, no fixed intercept, fitted by a Laplace approximation:
per-group posterior modes by Newton, τ̂ maximising the Laplace marginal on log τ within
``TAU_BOUNDS``. The model, next-season, reliability and ``ci_counts`` functions take "without
penalties" rows only and raise on any other kind.
"""

from __future__ import annotations

import math
from typing import Any

import gsax_cases as gc
import numpy as np
import polars as pl
import pytest
from scipy import integrate, optimize
from scipy.special import expit

KEEPER_SEASON = ("gk_id", "competition_id", "season_id")


def _gsax() -> Any:
    from football_shot_models import gsax

    return gsax


def _loglik(u: float, o: np.ndarray, y: np.ndarray) -> float:
    eta = o + u
    return float(np.sum(y * eta - np.logaddexp(0.0, eta)))


def _mode(o: np.ndarray, y: np.ndarray, tau: float) -> float:
    """The posterior mode of one group's u under N(0, τ²), by scipy (an independent route)."""
    res = optimize.minimize_scalar(
        lambda u: -_loglik(u, o, y) + u * u / (2 * tau * tau), bounds=(-30, 30), method="bounded",
        options={"xatol": 1e-12},
    )  # fmt: skip
    return float(res.x)


def test_constants() -> None:
    gsax = _gsax()
    assert gsax.KEEPER_SEASON == KEEPER_SEASON and gsax.CAREER == ("gk_id",)
    assert gsax.TAU_BOUNDS == (1e-3, 3.0)
    assert (gsax.NEWTON_TOL, gsax.NEWTON_MAX, gsax.MAX_STEP) == (1e-10, 100, 5.0)
    assert gsax.TIERS == (100, 250, 500) and gsax.MIN_TIER_KEEPERS == 30
    assert gsax.TABLE_MIN_SHOTS == 100 and gsax.SUM_TOL == 1e-9
    assert gsax.TEAM_SELECTED == "team-selected"


def test_modes_match_scipy() -> None:
    """Per group, ``keeper_modes`` returns the posterior mode of u (scipy within 1e-7) and
    ``h = Σ p(1−p)`` at it; groups of 9, 4 and 15 rows with codes out of order, one of them all
    goals (finite only through the prior)."""
    gsax = _gsax()
    sizes, codes = (9, 4, 15), []
    for g, n in zip((2, 0, 1), sizes, strict=True):
        codes += [g] * n
    n = len(codes)
    group = np.array(codes, dtype=np.intp)[np.argsort([(i * 11) % n for i in range(n)])]
    o = gc.logit(np.array(gc.spread(n, 0.1, 0.8)))
    y = np.array([(i % 3 == 0) for i in range(n)], dtype=np.float64)
    y[group == 0] = 1.0
    tau = 0.4
    u, h = gsax.keeper_modes(o, y, group, tau=tau)
    assert u.shape == (3,) and h.shape == (3,)
    for g in range(3):
        m = group == g
        want = _mode(o[m], y[m], tau)
        assert u[g] == pytest.approx(want, abs=1e-7), g
        p = expit(o[m] + u[g])
        assert h[g] == pytest.approx(float(np.sum(p * (1 - p))), rel=1e-9), g
    assert u[0] > 0 and math.isfinite(u[0])


def _quadrature_tau(o: np.ndarray, y: np.ndarray, bounds: tuple[float, float]) -> float:
    """τ̂ maximising the exact one-group marginal ∫ L(u) N(u; 0, τ²) du (adaptive quadrature)."""

    def log_marginal(tau: float) -> float:
        um = _mode(o, y, tau)
        c = _loglik(um, o, y) - um * um / (2 * tau * tau)
        val, _ = integrate.quad(
            lambda u: math.exp(_loglik(u, o, y) - u * u / (2 * tau * tau) - c),
            um - 12, um + 12, limit=200, points=[um],
        )  # fmt: skip
        return c + math.log(val) - math.log(tau) - 0.5 * math.log(2 * math.pi)

    res = optimize.minimize_scalar(
        lambda lt: -log_marginal(math.exp(lt)), bounds=tuple(math.log(b) for b in bounds),
        method="bounded", options={"xatol": 1e-6},
    )  # fmt: skip
    return math.exp(float(res.x))


def test_laplace_matches_quadrature() -> None:
    """One group of 200 shots conceding 50 goals against Σ q = 80: the Laplace τ̂ of
    ``fit_keeper_model`` is the exact-marginal τ̂ within 1e-3 (interior, ``at_bound`` False);
    ``u`` is the posterior mode at τ̂ and ``gsax_shrunk`` = Σ q − Σ expit(logit q + u)."""
    gsax = _gsax()
    n = 200
    q = [0.2 + 0.4 * ((i * 7) % 20) / 19 for i in range(n)]
    goals = [i % 4 == 0 for i in range(n)]
    matches = [f"sb:m{i % 10}" for i in range(n)]
    f = gc.rated(gc.frame(gk=["sb:g01"] * n, season=["A18"] * n, match=matches, goal=goals, xgot=q))
    model = gsax.fit_keeper_model(f, by=gsax.CAREER)
    o, y = gc.logit(np.array(q)), np.array(goals, dtype=np.float64)
    want = _quadrature_tau(o, y, gsax.TAU_BOUNDS)
    assert 0.1 < want < 2.0  # precondition: interior
    assert model.tau == pytest.approx(want, abs=1e-3)
    assert model.at_bound is False
    assert tuple(model.by) == ("gk_id",)
    est = model.estimates
    assert est.height == 1 and est["shots"].to_list() == [200]
    u = float(est["u"][0])
    assert u == pytest.approx(_mode(o, y, model.tau), abs=1e-6)
    assert u < 0
    shrunk = float(np.sum(q) - np.sum(expit(o + u)))
    assert float(est["gsax_shrunk"][0]) == pytest.approx(shrunk, abs=1e-9)
    assert model.per_100 == pytest.approx(100 * model.tau * 0.4 * 0.6, rel=1e-9)  # q̄ = 0.4


def test_tau_at_bound_without_variance() -> None:
    """Every keeper concedes exactly Σ q (pairs at q 0.3 / 0.7 with one goal): û = 0 at any τ,
    so τ̂ sits at the lower bound and ``at_bound`` is True; ``per_100`` = 100 τ̂ q̄(1 − q̄) with
    q̄ the mean q (0.5 here, not the mean of q(1 − q), 0.21)."""
    gsax = _gsax()
    gks, q, goals, matches = [], [], [], []
    for k in range(12):
        for j in range(4):
            gks.append(f"sb:g{k:02d}")
            q.append(0.3 if j % 2 == 0 else 0.7)
            goals.append(j in (0, 3))  # Σ q = 2 = goals per keeper
            matches.append(f"sb:m{k}")
    f = gc.rated(gc.frame(gk=gks, season=["A18"] * len(gks), match=matches, goal=goals, xgot=q))
    model = gsax.fit_keeper_model(f, by=gsax.KEEPER_SEASON)
    assert model.at_bound is True
    assert model.tau == pytest.approx(gsax.TAU_BOUNDS[0], rel=1e-3)
    assert model.per_100 == pytest.approx(100 * model.tau * 0.25, rel=1e-9)
    assert np.allclose(model.estimates["u"].to_numpy(), 0.0, atol=1e-6)


def _three_keepers() -> pl.DataFrame:
    """sb:g30 concedes far fewer than q, sb:g07 far more, sb:g12 both (A18 fewer, B18 more);
    40 shots each, rows interleaved (keys unsorted)."""
    parts = [
        gc.block("sb:g30", "A18", goals=gc.pattern(40, 8), xgot=gc.spread(40, 0.3, 0.6),
                 matches=["sb:m1", "sb:m2"]),
        gc.block("sb:g07", "A18", goals=gc.pattern(40, 4, 1), xgot=gc.spread(40, 0.05, 0.2),
                 matches=["sb:m3", "sb:m4"]),
        gc.block("sb:g12", "B18", goals=[True] * 30 + [False] * 10, xgot=gc.spread(40, 0.2, 0.5),
                 matches=["sb:m5", "sb:m6"]),
        gc.block("sb:g12", "A18", goals=gc.pattern(40, 10), xgot=gc.spread(40, 0.3, 0.6),
                 matches=["sb:m7", "sb:m8"]),
        gc.block("sb:g44", "A18", goals=gc.pattern(40, 3), xgot=gc.spread(40, 0.2, 0.5),
                 matches=["sb:m9", "sb:m10"]),
    ]  # fmt: skip
    f = gc.rated(gc.blocks(*parts))
    order = [(i * 37) % f.height for i in range(f.height)]
    assert sorted(order) == list(range(f.height))
    return f[order]


@pytest.mark.parametrize("by", ["career", "keeper_season"])
def test_codes_map_back(by: str) -> None:
    """Group codes are the row numbers of the sorted unique ``by`` keys: each ``estimates`` row's
    ``u`` is ``keeper_modes`` at τ̂ on codes built here from the sorted keys; ``shots`` and
    ``gsax_shrunk`` are recomputed per key; signs follow each keeper's record (sb:g12's two
    seasons apart under ``KEEPER_SEASON``). A bare ``str`` is refused."""
    gsax = _gsax()
    cols = gsax.CAREER if by == "career" else gsax.KEEPER_SEASON
    f = _three_keepers()
    model = gsax.fit_keeper_model(f, by=cols)
    assert tuple(model.by) == tuple(cols)
    assert model.at_bound is False  # precondition: real between-keeper variance
    keys = sorted(set(f.select(cols).rows()))
    index = {k: i for i, k in enumerate(keys)}
    codes = np.array([index[k] for k in f.select(cols).rows()], dtype=np.intp)
    o, y = gc.logit(f["q"].to_numpy()), f["is_goal"].to_numpy().astype(np.float64)
    u, _ = gsax.keeper_modes(o, y, codes, tau=model.tau)
    est = model.estimates
    assert est.height == len(keys) and set(cols) <= set(est.columns)
    assert est.schema["shots"] == pl.Int64
    for r in est.iter_rows(named=True):
        k = tuple(r[c] for c in cols)
        g = index[k]
        assert r["u"] == pytest.approx(float(u[g]), abs=1e-9), k
        m = codes == g
        assert r["shots"] == int(m.sum()), k
        q = f["q"].to_numpy()[m]
        shrunk = float(q.sum() - expit(gc.logit(q) + u[g]).sum())
        assert r["gsax_shrunk"] == pytest.approx(shrunk, abs=1e-9), k
    by_gk = {r["gk_id"] if by == "career" else (r["gk_id"], r["competition_id"]): r["gsax_shrunk"]
             for r in est.iter_rows(named=True)}  # fmt: skip
    if by == "career":
        assert by_gk["sb:g30"] > 0 > by_gk["sb:g07"]
    else:
        assert by_gk[("sb:g30", "sb:9")] > 0 > by_gk[("sb:g07", "sb:9")]
        assert by_gk[("sb:g12", "sb:9")] > 0 > by_gk[("sb:g12", "sb:72")]
    with pytest.raises((TypeError, ValueError)):
        gsax.fit_keeper_model(f, by="gk_id")


def test_modes_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty input, a non-finite offset, a goal not 0 / 1, a negative code and no convergence
    within ``NEWTON_MAX`` (read when called) raise ``ValueError``."""
    gsax = _gsax()
    o = gc.logit(np.array([0.2, 0.4, 0.6, 0.3]))
    y = np.array([1.0, 0.0, 0.0, 1.0])
    g = np.array([0, 1, 0, 1], dtype=np.intp)
    gsax.keeper_modes(o, y, g, tau=0.5)  # valid
    cases = [
        (np.array([]), np.array([]), np.array([], dtype=np.intp)),
        (np.array([0.1, np.nan, 0.2, 0.3]), y, g),
        (np.array([0.1, np.inf, 0.2, 0.3]), y, g),
        (o, np.array([1.0, 0.5, 0.0, 1.0]), g),
        (o, y, np.array([0, -1, 0, 1], dtype=np.intp)),
    ]
    for args in cases:
        with pytest.raises(ValueError):
            gsax.keeper_modes(*args, tau=0.5)
    monkeypatch.setattr(gsax, "NEWTON_MAX", 1)
    with pytest.raises(ValueError):
        gsax.keeper_modes(o, y, g, tau=2.0)


def test_open_play_only() -> None:
    """``fit_keeper_model``, ``next_season``, ``reliability`` and ``ci_counts`` take "without
    penalties" rows: one penalty row raises ``ValueError`` in each (the variant pinned by
    code)."""
    gsax = _gsax()
    f = _three_keepers()
    pen = f.with_columns(
        kind=pl.when(pl.int_range(pl.len()) == 5)
        .then(pl.lit("penalty", dtype=f.schema["kind"]))
        .otherwise(pl.col("kind"))
    )  # fmt: skip
    gsax.fit_keeper_model(f, by=gsax.KEEPER_SEASON)  # valid
    calls = (
        lambda r: gsax.fit_keeper_model(r, by=gsax.KEEPER_SEASON),
        lambda r: gsax.next_season(r, by=gsax.KEEPER_SEASON),
        gsax.reliability,
        lambda r: gsax.ci_counts(r, seed=0),
    )
    for call in calls:
        with pytest.raises(ValueError):
            call(pen)
