"""The StatsBomb hold-out level gap, decomposed by segment, and the rule that adopts a fix.

The xG model's GBM over-predicted the PL 2015/16 hold-out (Σ xG 974.8 for 913 goals, z +2.30). This
module holds everything about that gap that fits nothing: the level test (:func:`level_z`), the
segment decomposition, the adoption rule (:func:`adopt`) and the card text
(:func:`level_lines`). What is fitted (the pooled GBM and the two candidate fixes) and the
adopted index are constants of :mod:`football_shot_models.models`, which evaluates them and
calls :func:`level_probe`. This module reads only ``Evaluation`` arrays and frames, never the
prediction cache, so it is orchestration-only (``cache.ORCHESTRATION_ONLY``): a wording fix
keeps the cache warm.

**The level test.** Goals on ``n`` shots with probabilities ``p`` are a Poisson-binomial count:
its SD is ``√Σ p(1−p)`` (``metrics.poisson_binomial``), and the gap ``Σ p − goals`` is within
the noise band when ``|z| <= metrics.BAND_SDS`` (``metrics.within``, the xGOT and GSAx band rule,
read when called; two-sided, equality is within).

**The rule.** A fix is adoptable iff its hold-out level is within the band, every xG check row that
the pooled GBM meets is met by the fix too, and no xGOT / GSAx check refused it
(``models.DOWNSTREAM_MISSED``). The first adoptable fix in ``models.LEVEL_FIXES`` order is
adopted; ``models.run`` refuses an ``ADOPTED`` constant that the rule does not pick. Once a fix
is adopted, the hold-out chose it, so its level is no longer an independent check.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import polars as pl

from football_shot_models import data, features, metrics
from football_shot_models.cards import fmt, signed
from football_shot_models.evaluation import BENCHMARK, Evaluation, ModelSpec, labels
from football_shot_models.metrics import Check

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

DISTANCE_BANDS: Final = (6.0, 11.0, 16.0, 20.0, 25.0)  # metres, (a, b], via np.searchsorted
HOLDOUT_KINDS: Final = ("phase", "body part", "distance", "assist type", "shot freeze frame",
                        "360 frame")  # fmt: skip
DEV_KINDS: Final = ("family", "shot freeze frame", "360 frame")
SPLITS: Final = ("hold-out", "dev OOF")


def level_z(y: BoolArray, p: FloatArray) -> tuple[float, float, float]:
    """``(Σ p − Σ y, SD, z)`` of a goal count against its Poisson-binomial SD ``√Σ p(1−p)``.

    Raises
    ------
    ValueError
        No shot; ``y`` and ``p`` of different lengths; a NaN or infinite ``p``; SD 0.

    Examples
    --------
    >>> level_z(np.array([True, False, False, False]), np.full(4, 0.5))
    (1.0, 1.0, 1.0)
    """
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    if p.size == 0 or y.shape != p.shape:
        raise ValueError(f"level_z: {y.size} outcome(s) for {p.size} probabilities (need >= 1)")
    if not np.isfinite(p).all():
        raise ValueError("level_z: a probability is NaN or infinite")
    diff, sd = metrics.poisson_binomial(y, p)
    if sd == 0.0:
        raise ValueError("level_z: the Poisson-binomial SD is 0 (every p is 0 or 1)")
    return diff, sd, diff / sd


def within(z: float) -> bool:
    """``|z| <= metrics.BAND_SDS`` (read when called): ``metrics.within`` on a z (SD 1).

    Examples
    --------
    >>> within(-2.0), within(2.0001)
    (True, False)
    """
    return metrics.within(z, 1.0)


@dataclass(frozen=True)
class LevelRow:
    """One fitted GBM on the hold-out level: the level test, its xG check rows and scores.

    ``met`` counts the check rows met and ``judged`` the rows that are not ``n/a``;
    ``adopted`` marks the fix the rule picks and ``refused`` a fix that meets the rule
    (within the band, every check the pooled GBM meets kept) but an xGOT / GSAx check refused
    (``models.DOWNSTREAM_MISSED``): a skipped fix that fails the rule is not ``refused``.
    """

    name: str
    title: str
    xg: float
    goals: int
    sd: float
    z: float
    within: bool
    met: int
    judged: int
    adopted: bool
    refused: bool
    cv_log_loss: float
    holdout_log_loss: float
    fold_xg_mean: float
    fold_xg_max: float


@dataclass(frozen=True, eq=False)
class LevelProbe:
    """The level decomposition and the candidate fixes, chart-ready (no refit needed).

    Attributes
    ----------
    rows : tuple of LevelRow
        The pooled GBM, then each fix in ``models.LEVEL_FIXES`` order.
    segments : polars.DataFrame
        One row per (split, kind, level): ``n``, ``goals``, then ``xg``, ``diff``, ``sd``,
        ``z`` of the pooled GBM (``_pooled``) and of the headline (``_head``), and ``xg_bench``,
        Σ of the provider's own xG (``evaluation.BENCHMARK``; null without that column).
    pooled, head : Evaluation
        The pooled GBM and the headline (the adopted fix, or the pooled GBM without one).
    fixes : tuple of Evaluation
        Each fix, in ``models.LEVEL_FIXES`` order (the adopted one is the headline itself).
    adopted : int or None
        The index into ``fixes`` the rule picks.
    dev, held : polars.DataFrame
        The frames every evaluation was scored on (``oof`` / ``holdout_p`` align with their
        ``shot_id`` order).
    """

    rows: tuple[LevelRow, ...]
    segments: pl.DataFrame
    pooled: Evaluation
    fixes: tuple[Evaluation, ...]
    head: Evaluation
    adopted: int | None
    dev: pl.DataFrame
    held: pl.DataFrame


def adopt(
    pooled_checks: Sequence[Check],
    fix_checks: Sequence[Sequence[Check]],
    rows: Sequence[LevelRow],
    *,
    skip: Sequence[str],
) -> int | None:
    """The rule: the first fix that is within the band, keeps every xG check row the pooled GBM
    meets, and is not in ``skip``; None when no fix qualifies.

    Raises
    ------
    TypeError
        ``skip`` is a bare str.
    ValueError
        ``fix_checks`` and ``rows`` of different lengths; a fix with another number of check
        rows than the pooled GBM.

    Examples
    --------
    >>> ok, bad = Check("t", "m", "met"), Check("t", "m", "missed")
    >>> row = LevelRow("f", "F", 1.0, 1, 1.0, 0.5, True, 1, 1, False, False, *[0.0] * 4)
    >>> adopt([ok], [[bad], [ok]], [row, row], skip=())
    1
    >>> adopt([ok], [[ok]], [row], skip=("f",)) is None
    True
    """
    if isinstance(skip, str):
        raise TypeError("skip must be a sequence of spec names, not a single str")
    if len(fix_checks) != len(rows):
        raise ValueError(f"adopt: {len(fix_checks)} check set(s) for {len(rows)} fix row(s)")
    for i, (checks, row) in enumerate(zip(fix_checks, rows, strict=True)):
        if len(checks) != len(pooled_checks):
            raise ValueError(
                f"adopt: {row.name} has {len(checks)} check row(s), the pooled GBM "
                f"{len(pooled_checks)}"
            )
        kept = all(
            c.result == "met"
            for p, c in zip(pooled_checks, checks, strict=True)
            if p.result == "met"
        )
        if row.within and kept and row.name not in skip:
            return i
    return None


def _row(
    ev: Evaluation, held: pl.DataFrame, checks: Sequence[Check], *, adopted: bool, refused: bool
) -> LevelRow:
    diff, sd, z = level_z(labels(held), ev.holdout_p)
    mean, worst = metrics.fold_xg_error(ev.folds)
    return LevelRow(
        name=ev.spec.name,
        title=ev.spec.title,
        xg=float(ev.holdout_p.sum()),
        goals=ev.holdout.goals,
        sd=sd,
        z=z,
        within=within(z),
        met=sum(c.result == "met" for c in checks),
        judged=sum(c.result != "n/a" for c in checks),
        adopted=adopted,
        refused=refused,
        cv_log_loss=ev.cv.log_loss,
        holdout_log_loss=ev.holdout.log_loss,
        fold_xg_mean=mean,
        fold_xg_max=worst,
    )


def band_labels() -> tuple[str, ...]:
    """The distance band labels, nearest first (``DISTANCE_BANDS`` read when called).

    Examples
    --------
    >>> band_labels()[:2]
    ('<= 6 m', '(6, 11] m')
    """
    edges = DISTANCE_BANDS
    inner = [f"({a:g}, {b:g}] m" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    return (f"<= {edges[0]:g} m", *inner, f"> {edges[-1]:g} m")


def _band(distance: pl.Series) -> pl.Series:
    idx = np.searchsorted(np.asarray(DISTANCE_BANDS), distance.to_numpy(), side="left")
    names = band_labels()
    return pl.Series([names[i] for i in idx], dtype=pl.String)


def _kinds(
    frame: pl.DataFrame, split: str, freeze_frames: pl.DataFrame | None
) -> list[tuple[str, pl.Series]]:
    framed = (
        frame["shot_id"].is_in(freeze_frames["shot_id"].implode()).to_numpy()
        if freeze_frames is not None and freeze_frames.height
        else np.zeros(frame.height, dtype=bool)
    )
    with_frame = {
        "shot freeze frame": pl.Series(np.where(framed, "with", "without")),
        "360 frame": pl.Series(np.where(frame["visible_area"].is_not_null(), "with", "without")),
    }
    if split == SPLITS[1]:
        return [("family", frame["family"].cast(pl.String)), *with_frame.items()]
    phase = features.DERIVED["phase"].fn({"shots": frame}).cast(pl.String)
    return [
        ("phase", phase),
        ("body part", frame["body_part"].cast(pl.String)),
        ("distance", _band(frame["distance_m"])),
        ("assist type", frame["assist_type"].cast(pl.String)),
        *with_frame.items(),
    ]


def segments(
    pooled: Evaluation,
    head: Evaluation,
    *,
    dev: pl.DataFrame,
    held: pl.DataFrame,
    freeze_frames: pl.DataFrame | None,
) -> pl.DataFrame:
    """The decomposition: per (split, kind, level), the level of the pooled GBM and the headline.

    Hold-out: phase, body part, distance band (``DISTANCE_BANDS``), assist type, shot freeze
    frame (its ``shot_id`` in ``freeze_frames``) and 360 frame (``visible_area`` not null). Dev
    out of fold: family, shot freeze frame, 360 frame. A null level prints as ``"null"``.
    ``xg_bench`` is Σ ``BENCHMARK.column`` on the segment (a counterfactual for the card: the
    provider's own xG on the same shots), null when the frame has no such column.

    Raises
    ------
    ValueError
        A kind whose rows do not sum to the split's totals (an internal error); from
        :func:`level_z`.

    Examples
    --------
    >>> segments(pooled, head, dev=dev, held=held, freeze_frames=ff).columns[:5]
    ... # doctest: +SKIP
    ['split', 'kind', 'level', 'n', 'goals']
    """
    rows = []
    for split, frame, p0, p1 in (
        (SPLITS[0], held, pooled.holdout_p, head.holdout_p),
        (SPLITS[1], dev, pooled.oof, head.oof),
    ):
        y = labels(frame)
        col = BENCHMARK.column
        bench = frame[col].cast(pl.Float64).to_numpy() if col in frame.columns else None
        for kind, values in _kinds(frame, split, freeze_frames):
            keys = values.fill_null("null").to_numpy()
            n_sum = goals_sum = 0
            order = {b: i for i, b in enumerate(band_labels())}  # bands nearest first
            for level in sorted(set(keys.tolist()), key=lambda v: (order.get(v, 0), v)):
                m = keys == level
                d0, s0, z0 = level_z(y[m], p0[m])
                d1, s1, z1 = level_z(y[m], p1[m])
                n_sum, goals_sum = n_sum + int(m.sum()), goals_sum + int(y[m].sum())
                x0, x1 = float(p0[m].sum()), float(p1[m].sum())
                rows.append(
                    (
                        split,
                        kind,
                        level,
                        int(m.sum()),
                        int(y[m].sum()),
                        x0,
                        d0,
                        s0,
                        z0,
                        x1,
                        d1,
                        s1,
                        z1,
                        None if bench is None else float(bench[m].sum()),
                    )
                )
            if (n_sum, goals_sum) != (frame.height, int(y.sum())):
                raise ValueError(f"segments: {split} {kind} rows do not sum to the totals")
    names = ("split", "kind", "level", "n", "goals")
    stats = ("xg", "diff", "sd", "z")
    cols = (*names, *(f"{s}_pooled" for s in stats), *(f"{s}_head" for s in stats), "xg_bench")
    schema = {c: pl.String if c in names[:3] else pl.Int64 if c in names else pl.Float64
              for c in cols}  # fmt: skip
    return pl.DataFrame(rows, schema=schema, orient="row")


def level_probe(
    pooled: Evaluation,
    fixes: Sequence[Evaluation],
    head: Evaluation,
    *,
    checks_of: Callable[[Evaluation], tuple[Check, ...]],
    dev: pl.DataFrame,
    held: pl.DataFrame,
    freeze_frames: pl.DataFrame | None,
    skip: Sequence[str] = (),
) -> LevelProbe:
    """The rows, the rule's pick and the decomposition, from evaluations already made (pure).

    ``checks_of(ev)`` gives ``models.milestone_checks`` of ``ev`` as if it were the headline;
    ``skip`` names the evaluations (spec names) of the fixes in ``models.DOWNSTREAM_MISSED``.

    Raises
    ------
    ValueError
        No fix; from :func:`level_z`, :func:`adopt` or :func:`segments`.

    Examples
    --------
    >>> probe = level_probe(pooled, fixes, gbm, checks_of=checks_of, dev=dev, held=held,
    ...                     freeze_frames=ff, skip=())  # doctest: +SKIP
    >>> probe.adopted  # doctest: +SKIP
    1
    """
    if not fixes:
        raise ValueError("level_probe: no fix to probe")
    base = checks_of(pooled)
    fix_checks = [checks_of(ev) for ev in fixes]
    draft = [
        _row(ev, held, c, adopted=False, refused=False)
        for ev, c in zip(fixes, fix_checks, strict=True)
    ]
    picked = adopt(base, fix_checks, draft, skip=skip)
    # refused downstream: in ``skip`` and adoptable by the MS3 rule alone (the card says so)
    refused = [
        r.name in skip and adopt(base, [c], [r], skip=()) == 0
        for r, c in zip(draft, fix_checks, strict=True)
    ]
    rows = (
        _row(pooled, held, base, adopted=False, refused=False),
        *(
            _row(ev, held, c, adopted=i == picked, refused=refused[i])
            for i, (ev, c) in enumerate(zip(fixes, fix_checks, strict=True))
        ),
    )
    table = segments(pooled, head, dev=dev, held=held, freeze_frames=freeze_frames)
    return LevelProbe(rows, table, pooled, tuple(fixes), head, picked, dev, held)


# ---- card text (render-side, but here so the cache stays warm) -------------------------------


def _z(z: float) -> str:
    return f"{z:+.2f}".replace("-", "−")


def _segment_lines(table: pl.DataFrame) -> list[str]:
    lines = [
        "| Split | Segment | Shots | Goals | Pooled Σ xG − goals (z) | Headline Σ xG − goals (z) |",
        "|---|---|---|---|---|---|",
    ]
    for (split, kind), part in table.group_by(["split", "kind"], maintain_order=True):
        if part.height == 1:
            n = int(part["n"][0])
            lines.append(f"| {split} | {kind}: one level only (n = {n:,}) | | | | |")
            continue
        for r in part.iter_rows(named=True):
            lines.append(
                f"| {split} | {kind} = {r['level']} | {r['n']:,} | {r['goals']:,} | "
                f"{signed(r['diff_pooled'])} ({_z(r['z_pooled'])}) | "
                f"{signed(r['diff_head'])} ({_z(r['z_head'])}) |"
            )
    return lines


def _share(gap: float, goals: int) -> str:
    return f"{gap / goals:+.1%}".replace("-", "−") if goals else "n/a"


def _bench_text(part: pl.DataFrame, w: dict[str, Any]) -> str:
    """The provider's own xG on the worst segment and on the whole split (``part``: every level
    of that segment's kind, so its sums are the split's), or "" without it."""
    xb, total = w["xg_bench"], float(part["xg_bench"].sum())
    goals, all_goals = int(w["goals"]), int(part["goals"].sum())
    if not isinstance(xb, float) or not math.isfinite(xb) or not math.isfinite(total):
        return ""
    return (
        f" {BENCHMARK.title} on the same shots: Σ {xb:.1f} for {goals:,} goals "
        f"({signed(xb - goals)}, {_share(xb - goals, goals)}); on the whole hold-out "
        f"{_share(total - all_goals, all_goals)} (a computed counterfactual, not a causal claim)."
    )


def _family_diff(table: pl.DataFrame, family: str, column: str) -> str:
    part = table.filter(
        (pl.col("split") == SPLITS[1]) & (pl.col("kind") == "family") & (pl.col("level") == family)
    )
    if part.height == 0:
        return "n/a (no such shots)"
    r = part.row(0, named=True)
    xg = r[column.replace("diff", "xg")]
    rel = r[column] / r["goals"] if r["goals"] else float("nan")
    share = f"{rel:+.1%}".replace("-", "−")
    z = f", z {_z(r[column.replace('diff', 'z')])}" if column.endswith("_head") else ""
    return f"{signed(r[column])} ({share} of {r['goals']:,} goals; Σ xG {xg:.1f}{z})"


def level_lines(probe: LevelProbe, *, steps_train_exclude: tuple[str, ...]) -> tuple[str, ...]:
    """The xg-gbm card's hold-out level section: decomposition, probe table, computed lines.

    ``steps_train_exclude`` is the LR steps' ``train_exclude`` (all of them share one); when it
    differs from the headline's, a line says the GBM-over-LR row mixes training population with
    model class.

    Examples
    --------
    >>> lines = level_lines(probe, steps_train_exclude=())  # doctest: +SKIP
    >>> lines[0].startswith("### Hold-out level")  # doctest: +SKIP
    True
    """
    least = metrics.MIN_JUDGED_SHOTS  # read when called
    rows, t = probe.rows, probe.segments
    pooled, head = rows[0], next((r for r in rows[1:] if r.adopted), rows[0])
    probe_rows = [
        f"| {r.title}{' (adopted)' * r.adopted} | {r.xg:.1f} | {r.goals:,} | "
        f"{signed(r.xg - r.goals)} | {r.sd:.1f} | {_z(r.z)} | {'yes' if r.within else 'no'} | "
        f"{r.met} of {r.judged} | "
        f"{fmt(r.cv_log_loss)} | {fmt(r.holdout_log_loss)} | {r.fold_xg_mean:.2%} | "
        f"{r.fold_xg_max:.2%} |"
        for r in rows
    ]
    held = t.filter(pl.col("split") == SPLITS[0])
    judged = held.filter(pl.col("n") >= least)
    lines = [
        "### Hold-out level (PR1)",
        "",
        f"The pooled GBM's hold-out Σ xG is {pooled.xg:.1f} for {pooled.goals:,} goals "
        f"({signed(pooled.xg - pooled.goals)}, Poisson-binomial SD {pooled.sd:.1f}, z "
        f"{_z(pooled.z)}). Decomposition by segment (Σ xG − goals, z = gap / √Σ p(1−p)); the dev "
        "rows are out of fold:",
        "",
        *_segment_lines(t),
        "",
        f"At most two candidate fixes, one refit each. A fix is adopted iff its hold-out Σ xG is "
        f"within {metrics.BAND_SDS:g} SD of goals and every MS3 check the pooled GBM meets stays "
        "met; the first such fix in this order is adopted.",
        "",
        "| Model | Hold-out Σ xG | Goals | Gap | SD | z | Within | MS3 checks met | CV log loss "
        "| Hold-out log loss | Fold Σ xG error mean | max |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *probe_rows,
        "",
    ]
    refused = [r for r in rows[1:] if r.refused]
    if head is pooled:
        lines.append(
            "- Adopted fix: none: the decomposition is a limitation of this model (the table "
            "above)."
            + "".join(
                f" {r.title} met the MS3 rule but an xGOT / GSAx check refused it." for r in refused
            )
        )
    else:
        removed = (pooled.xg - head.xg) / (pooled.xg - pooled.goals)
        lines.append(
            f"- Adopted: {head.title}. The hold-out chose it, so its hold-out level is no longer "
            f"an independent check. It removes {removed:.0%} of the pooled hold-out excess "
            f"({signed(pooled.xg - pooled.goals)} → {signed(head.xg - head.goals)})."
        )
    if judged.height:
        w = judged.row(int(judged["z_head"].abs().arg_max() or 0), named=True)
        lines.append(
            f"- Largest headline hold-out segment gap (segments of {least}+ shots): {w['kind']} = "
            f"{w['level']}, {signed(w['diff_head'])} on {w['n']:,} shots (z {_z(w['z_head'])})."
            + _bench_text(held.filter(pl.col("kind") == w["kind"]), w)
        )
    league, team = data.FAMILIES[0], data.FAMILIES[1]
    lines += [
        f"- Dev out of fold, {league}: pooled {_family_diff(t, league, 'diff_pooled')}; headline "
        f"{_family_diff(t, league, 'diff_head')}; this is support the hold-out did not choose.",
        f"- Dev out of fold, {team}, under pooled training: "
        f"{_family_diff(t, team, 'diff_pooled')}"
        + (
            f"; under the headline: {_family_diff(t, team, 'diff_head')}."
            if head is not pooled
            else "."
        ),
    ]
    exclude = head_exclude(probe.head)
    if exclude != steps_train_exclude:
        lines.append(
            f"- The LR steps are trained on {_population(steps_train_exclude)} and the headline on "
            f"{_population(exclude)}, so the GBM-over-LR gain mixes training population with "
            "model class."
        )
    return (*lines, "")


def head_exclude(ev: Evaluation) -> tuple[str, ...]:
    """The ``train_exclude`` of an evaluation's spec (``()`` for a benchmark).

    Examples
    --------
    >>> head_exclude(gbm)  # doctest: +SKIP
    ('team-selected',)
    """
    return ev.spec.train_exclude if isinstance(ev.spec, ModelSpec) else ()


def fit_population(exclude: tuple[str, ...]) -> str:
    """The training rows of a fit with ``train_exclude`` = ``exclude``, as the cards name them.

    Examples
    --------
    >>> fit_population(()), fit_population(("team-selected",))
    ('all of dev', 'dev without team-selected seasons')
    """
    return "all of dev" if not exclude else f"dev without {', '.join(exclude)} seasons"


def _population(exclude: tuple[str, ...]) -> str:
    return f"{fit_population(exclude)}{'' if exclude else ' (pooled)'}"
