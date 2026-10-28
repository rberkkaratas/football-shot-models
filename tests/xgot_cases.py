"""Hand-built synthetic fixtures for the MS5.xgot tests (spec ``runs/MS5/xgot`` §11) and the MS6
xgot-ready tests (spec ``runs/MS6/xgot-ready`` §4, §8).

Every id, name and number is invented; nothing is copied from an open-data or archive file.
Frames are built column by column with the C3 dtypes. Nothing here imports the new module
``football_shot_models.xgot`` at module level, so a missing module fails the tests that need it,
not the collection.

- ``mouth_frame`` / ``placement``: a few on-target shots and the placement features
  ``design_matrix`` computes from them through the xGOT guard.
- ``xgot_tables`` / ``write_xgot_snapshot``: one snapshot per provider with goal-mouth points.
  StatsBomb: 20 dev matches in two competition-seasons and 5 hold-out matches (PL 2015/16).
  WhoScored: 20 train matches (``2025/2026`` and the calendar season ``2026``) and 6 test
  matches (``2026/2027``); its match, team, player and keeper ids are 6-digit sentinels
  (``ws:830101``...) that a Rule 1 test looks for in every output. Each match has 36 open-play
  shots (blocked, on target, post, off target), 3 in-game penalties on target (one saved, placed
  so that no placement cell separates goals from saves); two matches have an extra missed
  penalty, two have a shoot-out (2 kicks on target, 1 off). Excluded on purpose: an own goal, a
  direct corner on target, and per provider a no-frame save (StatsBomb) or an unresolved-assist
  save (WhoScored).
- ``check_inputs``: stand-in evaluations for ``xgot.checks`` with chosen scores, and a dev frame
  with goal-mouth points for the MS6 band row.
- ``patched_providers``: MS6 §4 fixture rewrite. ``Provider.candidates`` is gone, so the fixture
  patches the module globals that ``Provider.candidate_specs`` reads (``ablation.STEPS``,
  ``models.GBM``, ``v1.STEPS``, ``v1.GBM``) and ``baselines.BASELINES`` (to ``()``, so the tiny
  StatsBomb step, which reads ``open_goal``, stays the headline and the keeper-neutral refit runs).

MS6 §2.2: the StatsBomb snapshot's ``goal_y`` values are cell centres of the 0.1 yd goal-mouth
lattice (as the MS6 loader writes them), so ``xgot.fit``'s lattice guard accepts the snapshot.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import functools
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import models_cases as mc
import numpy as np
import numpy.typing as npt
import polars as pl
import xg_cases as xc
from football_core import pitch, schema
from football_core.io import snapshot
from scipy.special import expit

from football_shot_models import ablation, baselines, evaluation, features, gbm, metrics

FloatArray = npt.NDArray[np.float64]
KIND = pl.Enum(["open_play", "penalty", "shootout"])
SPLIT = pl.Enum(["dev", "holdout"])
LR_FEATURES = (
    "xg_logit",
    *("mouth_y_abs", "mouth_y_abs_rcs1", "mouth_y_abs_rcs2", "mouth_y_abs_rcs3"),
    "mouth_y_past",
    *("mouth_z", "mouth_z_rcs1", "mouth_z_rcs2", "mouth_z_rcs3"),
    "angle_x_mouth_y",
    "angle_x_mouth_z",
    "mouth_far",
)
# MS6 xgot-ready §3, candidate (b): placement carries no monotone constraint
GBM_FEATURES = ("xg_logit", "mouth_y_abs", "mouth_out", "mouth_z", "far_side", "angle_rad")
PENALTY_FEATURES = ("mouth_y_abs", "mouth_z", "mouth_yz")
GBM_SIGNS = {"xg_logit": 1}
KEEPER = (
    "gk_x", "gk_y", "gk_in_cone", "gk_absent", "gk_to_goal_m", "gk_to_shot_m", "gk_off_line_m",
    "open_goal",
)  # fmt: skip
PLACEMENT_NAMES = (
    *("mouth_y_abs", "mouth_y_abs_rcs1", "mouth_y_abs_rcs2", "mouth_y_abs_rcs3"),
    *("mouth_z", "mouth_z_rcs1", "mouth_z_rcs2", "mouth_z_rcs3"),
    *("far_side", "mouth_far", "angle_x_mouth_y", "angle_x_mouth_z", "mouth_yz"),
    *("mouth_y_in", "mouth_y_past", "mouth_out"),
)


def lattice_y(goal_y: float) -> float:
    """MS6 §2.2, written out: a pitch-width goal_y (metres) moved to the centre of the StatsBomb
    0.1 yd goal-mouth cell that holds it (``pitch`` is the reference map)."""
    yd = float(np.interp(goal_y, pitch.METRE_MOUTH_Y, pitch.STATSBOMB.mouth_y))
    k = min(max(math.ceil(yd / 0.1 - 1e-4), 1), 800)
    return float(pitch.goal_mouth_to_canonical([(k - 0.5) * 0.1], [0.0], "statsbomb")[0][0])


def placement_names() -> tuple[str, ...]:
    """``features.PLACEMENT``'s names, whether it is a mapping or a tuple of features."""
    return tuple(k if isinstance(k, str) else k.name for k in features.PLACEMENT)


# ---- placement frames --------------------------------------------------------------------------


def mouth_frame(
    goal_y: Sequence[float | None],
    goal_z: Sequence[float | None],
    y: Sequence[float | None],
    *,
    on_target: Sequence[bool] | None = None,
    angle: Sequence[float] | None = None,
) -> pl.DataFrame:
    """On-target shots with the columns the placement features read (plus the harness keys)."""
    n = len(goal_y)
    return pl.DataFrame(
        {
            "shot_id": [f"sb:x{i:03d}" for i in range(n)],
            "match_id": [f"sb:m{i % 5:02d}" for i in range(n)],
            "y": pl.Series(list(y), dtype=pl.Float64()),
            "angle_rad": pl.Series([0.5] * n if angle is None else list(angle), dtype=pl.Float64()),
            "goal_y": pl.Series(list(goal_y), dtype=pl.Float64()),
            "goal_z": pl.Series(list(goal_z), dtype=pl.Float64()),
            "on_target": pl.Series([True] * n if on_target is None else list(on_target),
                                   dtype=pl.Boolean()),
            "is_goal": pl.Series([i % 2 == 0 for i in range(n)], dtype=pl.Boolean()),
        }
    )  # fmt: skip


def placement(frame: pl.DataFrame, names: Sequence[str]) -> dict[str, list[float]]:
    """The design columns of placement features, by name (through the xGOT guard)."""
    matrix, cols = features.design_matrix(
        frame, list(names), derived=features.XGOT_DERIVED, guard=features.check_xgot_features
    )
    return {c: matrix[:, i].tolist() for i, c in enumerate(cols)}


# ---- evaluations built from arrays -------------------------------------------------------------


def folds_from_tests(n: int, tests: Sequence[Sequence[int]]) -> evaluation.Folds:
    """(train, test) pairs from test index lists; train is the sorted complement."""
    out = []
    for t in tests:
        test = np.asarray(t, dtype=np.intp)
        out.append((np.setdiff1d(np.arange(n), test).astype(np.intp), test))
    return tuple(out)


def make_eval(
    spec: evaluation.ModelSpec | evaluation.Benchmark,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    folds: evaluation.Folds,
    oof: FloatArray,
    holdout_p: FloatArray,
) -> evaluation.Evaluation:
    """An Evaluation whose scores are those of the given arrays (as ``evaluate`` would build)."""
    y, yh = evaluation.labels(dev), evaluation.labels(held)
    per_fold = tuple(metrics.score(y[test], oof[test]) for _, test in folds)
    return evaluation.Evaluation(
        spec=spec,
        folds=per_fold,
        cv=metrics.mean_scores(per_fold),
        oof=oof,
        train=metrics.score(y, oof),
        holdout=metrics.score(yh, holdout_p),
        coefficients=(),
        holdout_p=holdout_p,
        fold_test=tuple(test for _, test in folds),
    )


def lr_spec(**over: Any) -> evaluation.ModelSpec:
    """An xGOT LR spec as the spec table writes it (``xgot-lr``)."""
    fields: dict[str, Any] = {
        "name": "xgot-lr",
        "title": "xGOT LR",
        "features": LR_FEATURES,
        "make": ablation.make_lr,
        "coefficients": baselines.lr_coefficients,
        "guard": features.check_xgot_features,
        "derived": features.XGOT_DERIVED,
        "xg_input": "xg-input",
    }
    return evaluation.ModelSpec(**{**fields, **over})


def gbm_spec(**over: Any) -> evaluation.ModelSpec:
    """An xGOT GBM spec (``coefficients=gbm.gbm_settings`` marks a GBM), fixture-sized grid."""
    make = functools.partial(gbm.MonotoneGBM, dict(GBM_SIGNS), **mc.TINY)
    return lr_spec(name="xgot-gbm", title="xGOT GBM", features=GBM_FEATURES, make=make,
                   coefficients=gbm.gbm_settings, **over)  # fmt: skip


# ---- xgot.checks inputs ------------------------------------------------------------------------


def check_inputs(
    *,
    cv_auc: float = 0.85,
    ho_auc: float = 0.85,
    cv_slope: float | None = 1.0,
    ho_slope: float | None = 1.0,
    gbm_head: bool = False,
    low: float = -0.001,
    recon: Any = None,
    head_features: tuple[str, ...] | None = None,
    xg_features: tuple[str, ...] = ("distance_m", "angle_rad"),
    frame_y: Sequence[float] | None = None,
    is_goal: Sequence[bool] | None = None,
    oof: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Keyword arguments of ``xgot.checks``: every row is met unless a parameter moves it.

    The scores are set directly (the checks read ``Scores``); the arrays are valid but their
    ECE is whatever the test patches ``metrics.ece`` to return. MS6 row 12 reads ``dev``'s
    goal-mouth points and ``head.oof``: by default every dev shot is at the goal centre
    (``frame_y`` 0, goal-frame metres) with xGOT 1/3 and 17 goals in 50, so the band row is met
    (|16.67 − 17| ≤ 2 · 3.33). ``frame_y``, ``is_goal`` and ``oof`` (50 values each) move it.
    """
    from football_shot_models import xgot

    n_dev, n_held = 50, 20
    dev = pl.DataFrame({
        "shot_id": [f"sb:d{i:02d}" for i in range(n_dev)],
        "match_id": [f"sb:m{i // 5:02d}" for i in range(n_dev)],
        "is_goal": [i % 3 == 0 for i in range(n_dev)] if is_goal is None else list(is_goal),
        "on_target": [True] * n_dev,
        "goal_y": [pitch.GOAL_CENTRE_Y - v for v in (frame_y or [0.0] * n_dev)],
        "goal_z": [1.0] * n_dev,
    })  # fmt: skip
    assert dev.height == n_dev  # the overrides have 50 values
    held = pl.DataFrame({
        "shot_id": [f"sb:h{i:02d}" for i in range(n_held)],
        "match_id": [f"sb:n{i // 5:02d}" for i in range(n_held)],
        "is_goal": [i % 3 == 0 for i in range(n_held)],
        "on_target": [True] * n_held,
    })  # fmt: skip
    folds = folds_from_tests(n_dev, [list(range(k, n_dev, 5)) for k in range(5)])
    cv = mc.scores(n=n_dev, goals=17, xg=17.0, auc=cv_auc, slope=cv_slope)
    holdout = mc.scores(n=n_held, goals=7, xg=7.0, auc=ho_auc, slope=ho_slope)
    spec = gbm_spec() if gbm_head else lr_spec()
    if head_features is not None:
        spec = dataclasses.replace(spec, features=head_features)
    head = dataclasses.replace(
        mc.fake_eval(spec, cv=cv, holdout=holdout, folds=(cv,) * 5,
                     oof=np.full(n_dev, 1 / 3) if oof is None else np.asarray(oof, np.float64),
                     holdout_p=np.full(n_held, 1 / 3)),
        fold_test=tuple(test for _, test in folds),
    )  # fmt: skip
    gain = evaluation.Comparison(
        "over xGOT LR", mc.gain(low + 0.002, low, low + 0.004), mc.gain(0.0, -0.01, 0.01)
    )
    xg_spec = evaluation.ModelSpec("xg-input", "xG input", xg_features, ablation.make_lr)
    xg_ev = mc.fake_eval(xg_spec)
    if recon is None:
        recon = reconciliation(1000, 300, 1000, 300)
    return {"head": head, "gbm_gain": gain, "recon": recon, "xg_ev": xg_ev, "dev": dev,
            "held": held, "split": "hold-out", "_xgot": xgot}  # fmt: skip


def call_checks(inputs: dict[str, Any]) -> tuple[Any, ...]:
    args = {k: v for k, v in inputs.items() if not k.startswith("_")}
    result: tuple[Any, ...] = inputs["_xgot"].checks(**args)
    return result


def reconciliation(c3_shots: int, c3_goals: int, shots: int, goals: int) -> Any:
    from football_shot_models import xgot

    rows = tuple((r, 0, 0) for r in xgot.RECONCILE_REASONS[:-1])
    rows += ((xgot.RECONCILE_REASONS[-1], c3_shots, c3_goals),)
    return xgot.Reconciliation(c3_shots=c3_shots, c3_goals=c3_goals, shots=shots, goals=goals,
                               unscored=0, rows=rows)  # fmt: skip


# ---- the run snapshots -------------------------------------------------------------------------

SEED = 20261027
SHOTS_PER_MATCH = 36
# kick placements in the goal frame (y, z); a match's save and its two goals use 3 different
# cells, and over the matches every cell holds saves and goals, so no cell separates them
CELLS: tuple[tuple[float, float], ...] = (
    (0.5, 0.3), (-2.0, 0.3), (3.0, 1.5), (-1.0, 2.0), (2.5, 1.0), (-0.2, 1.2),
)  # fmt: skip
SB_BLOCKS = (  # (competition id, season id, competition name, season name), matches
    (("sb:11", "sb:4", "Fixture League", "2018/2019"), 12),
    (("sb:43", "sb:3", "Fixture Cup", "2018"), 8),
    (("sb:2", "sb:27", "Premier League", "2015/2016"), 5),  # the hold-out pair
)
WS_BLOCKS = (
    (("ws:710", "ws:72526", "Fixture League A", "2025/2026"), 12),
    (("ws:730", "ws:72026", "Fixture Cup", "2026"), 8),
    (("ws:710", "ws:72627", "Fixture League A", "2026/2027"), 6),  # the test season
)
WS_TRAIN_LAST = dt.date(2026, 7, 4)
WS_TEST_FIRST = dt.date(2026, 8, 8)
PLAYERS = 16


@dataclasses.dataclass(frozen=True)
class Ids:
    """The provider's id scheme; WhoScored's are bare 6-digit sentinels behind ``ws:``."""

    prefix: str

    def match(self, k: int) -> str:
        return f"sb:m{k:02d}" if self.prefix == "sb:" else f"ws:{830100 + k}"

    def team(self, t: int) -> str:
        return f"sb:t{t:02d}" if self.prefix == "sb:" else f"ws:{840100 + t}"

    def player(self, j: int) -> str:
        return f"sb:p{j:02d}" if self.prefix == "sb:" else f"ws:{850100 + j}"

    def keeper(self, t: int) -> str:
        return f"sb:g{t:02d}" if self.prefix == "sb:" else f"ws:{860100 + t}"

    def keeper_of(self, team_id: str) -> str:
        """The keeper of a team (one per team)."""
        if self.prefix == "sb:":
            return self.keeper(int(team_id.removeprefix("sb:t")))
        return self.keeper(int(team_id.removeprefix("ws:")) - 840100)


def ws_sentinels() -> set[str]:
    """Every bare WhoScored id number the fixture uses (match, team, player, keeper)."""
    ids = Ids("ws:")
    n = sum(k for _, k in WS_BLOCKS)
    out = {ids.match(k) for k in range(1, n + 1)}
    out |= {ids.team(t) for t in range(1, 2 * n + 1)} | {ids.keeper(t) for t in range(1, 2 * n + 1)}
    out |= {ids.player(j) for j in range(PLAYERS)}
    return {s.removeprefix("ws:") for s in out}


def _matches(provider: str) -> list[xc.Match]:
    ids = Ids("sb:" if provider == "statsbomb" else "ws:")
    blocks = SB_BLOCKS if provider == "statsbomb" else WS_BLOCKS
    out: list[xc.Match] = []
    k = 0
    for (comp, season, cname, sname), n in blocks:
        for j in range(n):
            k += 1
            if provider == "statsbomb":
                day = dt.date(2016 if comp == "sb:2" else 2019, 1, 5) + dt.timedelta(days=7 * j)
            elif sname == "2026/2027":
                day = WS_TEST_FIRST + dt.timedelta(days=7 * j)
            elif sname == "2026":
                day = WS_TRAIN_LAST - dt.timedelta(days=14 * (n - 1 - j))
            else:
                day = dt.date(2025, 8, 9) + dt.timedelta(days=14 * j)
            out.append(xc.Match(ids.match(k), comp, season, ids.team(2 * k - 1), ids.team(2 * k),
                                "male", cname, sname, day))  # fmt: skip
    return out


def _shot(ids: Ids, k: int, m: xc.Match, sid: str, home: bool, **kw: Any) -> xc.Shot:
    base = xc.Shot(f"{ids.prefix}s{k:02d}-{sid}", m.match_id, m.competition_id, m.season_id,
                   m.home if home else m.away, m.away if home else m.home, home,
                   match_date=m.match_date, provider_xg=None)  # fmt: skip
    return dataclasses.replace(base, **kw)


def xgot_tables(provider: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``(shots, freeze_frames, matches)`` of one provider's snapshot, shots sorted by id."""
    sb = provider == "statsbomb"
    ids = Ids("sb:" if sb else "ws:")
    rng = np.random.default_rng(SEED + (0 if sb else 1))
    matches = _matches(provider)
    shots: list[xc.Shot] = []
    extra: dict[str, list[Any]] = {
        k: [] for k in ("outcome", "goal_y", "goal_z", "deflected", "open_goal", "assist_type",
                        "player_id", "gk_id")
    }  # fmt: skip

    def add(shot: xc.Shot, outcome: str, mouth: tuple[float, float] | None, *,
            deflected: bool = False, open_goal: bool = False, assist: str | None = "pass",
            player: int = 0) -> None:  # fmt: skip
        shots.append(shot)
        extra["outcome"].append(outcome)
        if mouth is None:
            extra["goal_y"].append(None)
        elif sb:  # MS6 §2.2: StatsBomb goal-mouth y on the 0.1 yd lattice
            extra["goal_y"].append(lattice_y(pitch.GOAL_CENTRE_Y - mouth[0]))
        else:
            extra["goal_y"].append(round(pitch.GOAL_CENTRE_Y - mouth[0], 4))
        extra["goal_z"].append(None if mouth is None else round(mouth[1], 4))
        extra["deflected"].append(deflected if sb else None)
        extra["open_goal"].append(open_goal if sb else None)
        extra["assist_type"].append(assist)
        extra["player_id"].append(ids.player(player % PLAYERS))
        extra["gk_id"].append(None if shot.own_goal else ids.keeper_of(shot.opponent_team_id))

    for k, m in enumerate(matches):
        shot = functools.partial(_shot, ids, k, m)

        for i in range(SHOTS_PER_MATCH):
            x, y = float(rng.uniform(86.0, 103.5)), float(rng.uniform(20.0, 48.0))
            x, y = round(x, 2), round(y, 2)
            d = float(pitch.shot_distance([x], [y])[0])
            xg_logit = 1.6 - 0.22 * d
            minute = 2 + 2 * i if i < 18 else 46 + 2 * (i - 18)
            u = float(rng.uniform())
            flags = {"deflected": bool(rng.uniform() < 0.05),
                     "open_goal": bool(rng.uniform() < 0.06)}  # fmt: skip
            kw = {"x": x, "y": y, "period": 1 if i < 18 else 2, "minute": minute,
                  "provider_xg": round(float(expit(xg_logit)), 4) if sb else None}  # fmt: skip
            player = 7 * k + i
            if u < 0.1:
                add(shot(f"{i:02d}", i % 2 == 0, **kw), "blocked", None, player=player, **flags)
            elif u < 0.6:
                gy, gz = float(rng.uniform(-3.8, 3.8)), float(rng.uniform(0.05, 2.35))
                p = expit(0.8 * xg_logit + 0.4 + 1.1 * (abs(gy) - 1.8) + 0.5 * (gz - 1.2))
                if abs(gy) > features.INNER_Y:  # the ball touches the post: mostly saved
                    p = 0.25
                goal = bool(rng.uniform() < p)
                add(shot(f"{i:02d}", i % 2 == 0, goal=goal, **kw), "goal" if goal else "saved",
                    (gy, gz), player=player, **flags)  # fmt: skip
            elif u < 0.65:
                side = 1.0 if rng.uniform() < 0.5 else -1.0
                add(shot(f"{i:02d}", i % 2 == 0, **kw), "post",
                    (side * 3.7, float(rng.uniform(0.2, 2.3))), player=player, **flags)  # fmt: skip
            else:
                side = 1.0 if rng.uniform() < 0.5 else -1.0
                mouth = (side * float(rng.uniform(3.9, 7.0)), float(rng.uniform(0.0, 3.0)))
                add(shot(f"{i:02d}", i % 2 == 0, **kw), "off_target", mouth, player=player,
                    **flags)  # fmt: skip
        for j in range(3):  # in-game penalties, all on target: kick 0 saved
            gy, gz = CELLS[(k + 2 * j) % len(CELLS)]
            add(shot(f"pen{j}", j % 2 == 0, x=94.0, y=34.0, goal=j != 0, penalty=True, period=2,
                     minute=60 + j), "goal" if j else "saved", (gy, gz), player=k + j)  # fmt: skip

    def special(k: int, sid: str, outcome: str, mouth: tuple[float, float] | None, **kw: Any
                ) -> None:  # fmt: skip
        add(_shot(ids, k, matches[k], sid, True, **kw), outcome, mouth, player=k)

    last = len(matches) - 1
    for k in (1, last):  # a missed in-game penalty: not in the on-target kicks
        special(k, "penx", "off_target", (5.0, 1.0), x=94.0, y=34.0, penalty=True, period=1,
                minute=30)  # fmt: skip
    for k in (2, last - 1):  # shoot-outs: goal, save, miss
        for j, (outcome, mouth) in enumerate(
            (("goal", (2.8, 0.6)), ("saved", (-1.2, 0.4)), ("off_target", (4.5, 2.9)))
        ):
            special(k, f"so{j}", outcome, mouth, x=94.0, y=34.0, penalty=True, period=5,
                    minute=120, goal=outcome == "goal")  # fmt: skip
    special(3, "og", "goal", None, x=101.0, y=36.0, goal=True, own_goal=True, minute=40)
    special(4, "dc", "goal", (3.2, 2.0), x=104.9, y=0.5, goal=True, minute=70,
            situation="direct_corner")  # fmt: skip
    special(5, "odd", "saved", (-0.6, 0.8), x=96.0, y=30.0, minute=41)
    frame = xc.shots_frame(shots)
    n = frame.height
    outcome = pl.Series("outcome", extra["outcome"], dtype=schema.OUTCOME)
    odd = f"{ids.prefix}s05-odd"  # the no-frame save (StatsBomb), the null-assist save (WhoScored)
    assist = [None if (not sb and s.shot_id == odd) else a
              for s, a in zip(shots, extra["assist_type"], strict=True)]  # fmt: skip
    frame = frame.with_columns(
        outcome,
        pl.Series("goal_y", extra["goal_y"], dtype=pl.Float64()),
        pl.Series("goal_z", extra["goal_z"], dtype=pl.Float64()),
        pl.Series("deflected", extra["deflected"], dtype=pl.Boolean()),
        pl.Series("open_goal", extra["open_goal"], dtype=pl.Boolean()),
        pl.Series("assist_type", assist, dtype=schema.ASSIST_TYPE),
        pl.Series("player_id", extra["player_id"], dtype=pl.String()),
        pl.Series("gk_id", extra["gk_id"], dtype=pl.String()),
    ).with_columns(
        on_target=pl.col("outcome").cast(pl.String).is_in(["goal", "saved"])
        & ~pl.col("is_own_goal"),
    )
    assert frame.height == n
    if not sb:
        frame = frame.with_columns(
            pl.lit("whoscored").cast(schema.PROVIDER).alias("provider"),
            *(pl.lit(None).cast(frame.schema[c]).alias(c)
              for c in ("technique", "follows_dribble", "provider_xg")),
            pl.lit("[]").alias("raw_qualifiers"),
        )  # fmt: skip
    frame = frame.sort("shot_id")
    if sb:
        frames = xc.frames_for(frame.filter(pl.col("shot_id") != odd))
    else:
        frames = pl.DataFrame(schema=schema.FREEZE_FRAME_SCHEMA)
    matches_df = xc.matches_frame(matches)
    if not sb:
        matches_df = matches_df.with_columns(
            pl.lit("whoscored").cast(schema.PROVIDER).alias("provider"),
            pl.lit(None, dtype=pl.Int16()).alias("home_score"),
            pl.lit(None, dtype=pl.Int16()).alias("away_score"),
            pl.lit(None, dtype=pl.Boolean()).alias("has_360"),
        )
    return frame, frames, matches_df


def write_xgot_snapshot(
    root: Path, provider: str, *, edit: Callable[[pl.DataFrame], pl.DataFrame] | None = None
) -> Path:
    """Write one provider's fixture under ``root`` (``statsbomb-open-v1`` / ``whoscored-v1``);
    the manifest path. ``edit`` changes the shots table first (a near-miss snapshot)."""
    shots, frames, matches = xgot_tables(provider)
    if edit is not None:
        shots = edit(shots)
    sb = provider == "statsbomb"
    result = snapshot.write_snapshot(
        "statsbomb-open-v1" if sb else "whoscored-v1",
        {"shots": shots, "freeze_frames": frames, "matches": matches},
        xc.SOURCE if sb else snapshot.Source("files", "whoscored-csv-archive", "0" * 64),
        root=root,
        sort_by={"shots": ("shot_id",), "freeze_frames": ("shot_id",), "matches": ("match_id",)},
        builder=provider,
    )
    return result.manifest_path


# ---- a fixture-sized run -----------------------------------------------------------------------


def tiny_candidates(prefix: str, *, keeper: bool) -> tuple[evaluation.ModelSpec, ...]:
    """One LR step and a fixture-sized GBM of the xG run (``(step, gbm)``); with ``keeper`` both
    read ``open_goal`` (a ``KEEPER_POSITION`` member), so the keeper-neutral refit runs."""
    feats = ("distance_m", "angle_rad", *(("open_goal",) if keeper else ()))
    lr = evaluation.ModelSpec(f"{prefix}xg-lr-t", "LR distance + angle", feats,
                              ablation.make_lr, baselines.lr_coefficients)  # fmt: skip
    signs = {"distance_m": -1, "angle_rad": 1}
    g = evaluation.ModelSpec(f"{prefix}xg-gbm-t", "Tiny monotone GBM", feats,
                             functools.partial(gbm.MonotoneGBM, signs, **mc.TINY),
                             gbm.gbm_settings)  # fmt: skip
    return lr, g


def patch_candidates(mp: Any, *, baselines_off: bool = True) -> None:
    """MS6 §4 fixture rewrite: the module globals ``Provider.candidate_specs`` and the xG runs
    read, pointed at the tiny specs (StatsBomb reads ``open_goal``); ``baselines.BASELINES`` is
    emptied unless ``baselines_off`` is False."""
    from football_shot_models import models, v1

    sb_lr, sb_gbm = tiny_candidates("", keeper=True)
    ws_lr, ws_gbm = tiny_candidates("ws-", keeper=False)
    mp.setattr(ablation, "STEPS", (sb_lr,))
    mp.setattr(models, "GBM", sb_gbm)
    mp.setattr(v1, "STEPS", (ws_lr,))
    mp.setattr(v1, "GBM", ws_gbm)
    if baselines_off:
        mp.setattr(baselines, "BASELINES", ())


def tiny_specs(original: Callable[[str, str], tuple[evaluation.ModelSpec, ...]]) -> Any:
    """``xgot.xgot_specs`` with the GBM's grid made fixture-sized; its sign map is kept."""

    def specs(prefix: str, xg_name: str) -> tuple[evaluation.ModelSpec, ...]:
        out = []
        for s in original(prefix, xg_name):
            if s.coefficients is gbm.gbm_settings:
                signs = s.make().get_params()["monotone"]
                s = dataclasses.replace(
                    s, make=functools.partial(gbm.MonotoneGBM, signs, **mc.TINY)
                )
            out.append(s)
        return tuple(out)

    return specs


def patched_providers(xgot: Any, manifests: dict[str, Path], mp: Any) -> Any:
    """``xgot.PROVIDERS`` pointing at the fixture manifests; patches the candidate globals
    (:func:`patch_candidates`) and the xGOT specs (:func:`tiny_specs`) through ``mp``."""
    from types import MappingProxyType

    patch_candidates(mp)
    mp.setattr(xgot, "xgot_specs", tiny_specs(xgot.xgot_specs))
    out = {
        key: dataclasses.replace(p, manifest=str(manifests[key]))
        for key, p in xgot.PROVIDERS.items()
    }
    return MappingProxyType(out)
