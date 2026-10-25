"""M1.5 penalty model (spec MS3.models §1.7, §1.8, §1.11, §3.3, §7, §9): the shared hold-out
key, ``XgData.penalties``, ``penalty_rows``, ``exchangeability`` and the penalty card.

The table is ``models_cases.penalties_frame`` (10 in-game kicks in 6 matches, 5 shoot-out kicks
in 2), built column by column from ``xg_cases.shots_frame``.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from football_core import schema
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats
from sklearn.metrics import log_loss
from test_baselines_cards import _header, _rows, _section

from football_shot_models import cards, data, metrics

TEMPLATE = Path(__file__).resolve().parents[1] / "docs" / "model-card-template.md"
CHECK_TARGET = "posterior mean in [0.76, 0.80] (milestone wording)"


def _key(rows: Any) -> list[tuple[int, int, int]]:
    return [(r.rate.n, r.rate.goals, r.matches) for r in rows]


# ---- hold-out key -----------------------------------------------------------------------------


def test_is_holdout() -> None:
    """§7, §9: the hold-out is the (competition, season) pair, never one half of it;
    ``split_holdout`` uses the same mask."""
    frame = pl.DataFrame(
        {
            "competition_id": ["sb:2", "sb:11", "sb:2", "sb:37", "sb:2"],
            "season_id": ["sb:27", "sb:27", "sb:44", "sb:90", "sb:27"],
            "row": [0, 1, 2, 3, 4],
        }
    )
    mask = frame.select(data.is_holdout()).to_series()
    assert mask.to_list() == [True, False, False, False, True]
    other = frame.select(data.is_holdout(("sb:11", "sb:27"))).to_series()
    assert other.to_list() == [False, True, False, False, False]
    dev, held = data.split_holdout(frame)
    assert dev.equals(frame.filter(~mask)) and held.equals(frame.filter(mask))


# ---- XgData.penalties -------------------------------------------------------------------------


def test_load_penalties(xg_data: Any, models_run: dict[str, Any], tmp_path: Path) -> None:
    """§7, §9: the in-game penalty and the shoot-out kick of the MS2 fixture, with all C3 columns,
    then ``reason``, the match columns and ``family``; the own goal, the frameless shot and a
    direct corner with frame rows (ablation fixture) stay out, as does every kept shot. A
    snapshot without penalties gives an empty table of the same schema."""
    pens = xg_data.penalties
    assert pens.columns == [
        *schema.column_names("shots"), "reason", *data.MATCH_FEATURES, "family",
    ]  # fmt: skip
    assert pens["shot_id"].to_list() == ["sb:s-m04-pen", "sb:s-m07-so"]
    assert pens["reason"].to_list() == ["penalty", "shootout"]
    assert pens.schema["reason"] == data.EXCLUSION_REASON
    assert pens.schema["family"] == data.FAMILY
    assert pens.schema["gender"] == schema.GENDER
    assert pens["family"].to_list() == ["league or tournament", "team-selected"]
    assert pens["player_id"].null_count() == 0
    assert not set(pens["shot_id"]) & set(xg_data.shots["shot_id"])
    shots, frames, matches = xc.fixture_tables()
    none = xc.write_fixture_snapshot(
        tmp_path, shots=shots.filter(~pl.col("is_penalty")), freeze_frames=frames, matches=matches
    )
    empty = data.load_xg_data(none).penalties
    assert empty.height == 0 and empty.schema == pens.schema
    import ablation_cases as ac

    more = models_run["xg"].penalties
    assert set(more["reason"].to_list()) == {"penalty", "shootout"} and more.height == 7
    assert ac.DIRECT_CORNER_ID not in more["shot_id"].to_list()


# ---- penalty_rows -----------------------------------------------------------------------------


def test_penalty_rows() -> None:
    """§3.3: rows in order (in-game all, dev, hold-out, men, women, families except `women`,
    shoot-out all, men, women), each a Jeffreys ``beta_interval`` with distinct matches; the
    hold-out is the pair (kicks in (sb:11, sb:27) and (sb:2, sb:44) are dev); a null StatsBomb
    xG makes that group's ``provider_xg`` None."""
    from football_shot_models import models

    rows = models.penalty_rows(mc.penalties_frame())
    assert _key(rows) == list(mc.PEN_ROWS)
    for row, (n, goals, _) in zip(rows, mc.PEN_ROWS, strict=True):
        assert row.rate == metrics.beta_interval(goals, n)
        assert isinstance(row.label, str) and row.label
    assert len({r.label for r in rows}) == len(rows)
    sb = [r.provider_xg for r in rows]
    assert all(v == pytest.approx(0.7835) for v in sb[:7])
    assert sb[7:] == [None, pytest.approx(0.8), None]
    # a custom hold-out expression (MS4's season split) picks its own rows
    custom = models.penalty_rows(mc.penalties_frame(), holdout=pl.col("season_id") == "sb:44")
    assert _key(custom)[1:3] == [(9, 6, 5), (1, 1, 1)]


def test_penalty_rows_skips_empty_groups() -> None:
    """§3.3: a group with no kick has no row (no women, no hold-out, no shoot-out), and a 0-kick
    hold-out does not raise as ``split_holdout`` would."""
    from football_shot_models import models

    men = tuple(k for k in mc.PEN_KICKS if k[1] != "sb:m05")
    assert _key(models.penalty_rows(mc.penalties_frame(men))) == [
        (7, 5, 5), (4, 3, 3), (3, 2, 2), (7, 5, 5), (5, 4, 4), (2, 1, 1), (3, 2, 1), (3, 2, 1),
    ]  # fmt: skip
    dev_only = tuple(k for k in mc.PEN_KICKS if k[1] not in ("sb:m01", "sb:m02"))
    rows = models.penalty_rows(mc.penalties_frame(dev_only))
    assert _key(rows)[:3] == [(7, 5, 4), (7, 5, 4), (4, 3, 3)]
    in_game = tuple(k for k in mc.PEN_KICKS if k[2] != 5)
    assert len(models.penalty_rows(mc.penalties_frame(in_game))) == 7


def test_penalty_rows_shootout_apart() -> None:
    """§1.7, §9: a shoot-out kick's outcome changes only the shoot-out rows."""
    from football_shot_models import models

    frame = mc.penalties_frame()
    flipped = mc.set_value(frame, "is_goal", 14, True)  # sb:s05, a women's shoot-out miss
    before, after = models.penalty_rows(frame), models.penalty_rows(flipped)
    assert before[:7] == after[:7]
    assert _key(after)[7:] == [(5, 4, 2), (3, 2, 1), (2, 2, 1)]


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda f: mc.set_value(f, "reason", 0, "own_goal"), "own_goal"),
        (lambda f: mc.set_value(f, "reason", 3, "no_freeze_frame"), "no_freeze_frame"),
        (lambda f: mc.with_null(f, "is_goal", 2), "is_goal"),
        (lambda f: mc.with_null(f, "gender", 4), "gender"),
        (lambda f: mc.with_null(f, "family", 5), "family"),
        (lambda f: mc.with_null(f, "match_id", 12), "match_id"),
        (lambda f: f.filter(pl.col("reason") == "shootout"), "penalt"),
    ],
    ids=[
        "own goal",
        "no freeze frame",
        "null is_goal",
        "null gender",
        "null family",
        "null match_id (shoot-out row)",
        "no in-game penalty",
    ],
)
def test_penalty_rows_raise(edit: Any, match: str) -> None:
    """§3.3: another exclusion reason, a null key and no in-game penalty are ValueErrors."""
    from football_shot_models import models

    with pytest.raises(ValueError, match=match):
        models.penalty_rows(edit(mc.penalties_frame()))


@settings(deadline=None, max_examples=40)
@given(
    st.lists(
        st.tuples(st.sampled_from(sorted(mc.PEN_MATCHES)), st.sampled_from([1, 2, 5]),
                  st.booleans()),
        min_size=1, max_size=25,
    ).filter(lambda ks: any(p != 5 for _, p, _ in ks))
)  # fmt: skip
def test_penalty_rows_property(kicks: list[tuple[str, int, bool]]) -> None:
    """Property: the first row pools every in-game kick (never a shoot-out kick), and flipping
    every shoot-out outcome leaves all in-game rows unchanged."""
    from football_shot_models import models

    table = tuple(
        (f"sb:k{i:02d}", m, p, g, f"sb:p-{i % 3}", 0.7835) for i, (m, p, g) in enumerate(kicks)
    )
    flipped = tuple((s, m, p, (not g) if p == 5 else g, t, x) for s, m, p, g, t, x in table)
    rows = models.penalty_rows(mc.penalties_frame(table))
    in_game = [(m, g) for m, p, g in kicks if p != 5]
    first = rows[0]
    assert first.rate == metrics.beta_interval(sum(g for _, g in in_game), len(in_game))
    assert first.matches == len({m for m, _ in in_game})
    shootout_rows = 0
    if any(p == 5 for _, p, _ in kicks):
        genders = {mc.PEN_MATCHES[m][2] for m, p, _ in kicks if p == 5}
        shootout_rows = 1 + len(genders)
    again = models.penalty_rows(mc.penalties_frame(flipped))
    k = len(rows) - shootout_rows
    assert again[:k] == rows[:k]


# ---- exchangeability --------------------------------------------------------------------------


def test_exchangeability() -> None:
    """§3.3: Pearson chi2 over the in-game kicks by (competition, season): 5 groups (by season
    alone 4, by competition alone 3), p = 7/10, chi2 = 80/63 by hand; the top taker has 3
    in-game kicks (6 with his shoot-out kicks, which do not count). One group or a pooled rate
    of 0 or 1 gives None; a null ``player_id`` on an in-game kick raises."""
    from football_shot_models import models

    s = models.exchangeability(mc.penalties_frame())
    assert (s.groups, s.df, s.top_taker_kicks, s.kicks) == (5, 4, 3, 10)
    assert s.dispersion == pytest.approx(80 / 63 / 4, abs=1e-12)
    assert s.p_value == pytest.approx(float(stats.chi2.sf(80 / 63, 4)), abs=1e-12)
    # one competition-season (df 0) or a pooled rate of 0 or 1: not estimable (§9)
    one = tuple(k for k in mc.PEN_KICKS if k[1] in ("sb:m01", "sb:m02") or k[2] == 5)
    s = models.exchangeability(mc.penalties_frame(one))
    assert (s.groups, s.df, s.dispersion, s.p_value) == (1, 0, None, None)
    all_goals = tuple((a, b, p, True if p != 5 else g, t, x) for a, b, p, g, t, x in mc.PEN_KICKS)
    s = models.exchangeability(mc.penalties_frame(all_goals))
    assert (s.groups, s.dispersion, s.p_value) == (5, None, None)
    with pytest.raises(ValueError, match="player_id"):
        models.exchangeability(mc.with_null(mc.penalties_frame(), "player_id", 6))


# ---- penalty card -----------------------------------------------------------------------------


def _card(rows: Any = None, frame: pl.DataFrame | None = None) -> str:
    from football_shot_models import models

    frame = mc.penalties_frame() if frame is None else frame
    rows = models.penalty_rows(frame) if rows is None else rows
    return str(models.penalty_card(rows, models.exchangeability(frame), mc.card_context()))


def _check_row(card: str) -> list[str]:
    (row,) = [r for r in _rows(card) if CHECK_TARGET in r[0]]
    return row


def test_penalty_card() -> None:
    """§3.3: the template headings, name / title / version, features, the Metrics table, the
    validation and Benchmarks sentences for a hold-out row, the exchangeability, shoot-out and
    match-xG text, licensing; no date and no "CI inside"."""
    from football_shot_models import models

    rows = models.penalty_rows(mc.penalties_frame())
    card = _card()
    lines = card.splitlines()
    template = [x for x in TEMPLATE.read_text(encoding="utf-8").splitlines() if x.startswith("## ")]
    assert [x for x in lines if x.startswith("## ")] == template
    assert lines[0] == "# Model card: Penalty conversion (Beta-binomial)"
    assert re.search(r"`xg-penalty` v\d+", _section(card, "## Overview"))
    assert "None: one probability for every in-game penalty" in _section(card, "## Features")
    use = _section(card, "## Intended use")
    assert "match xG" in use and "shoot-out kicks are not part of match xG" in use
    block = _section(card, "## Metrics")
    table = block[: block.index(CHECK_TARGET)] if CHECK_TARGET in block else block
    assert _header(table) == [
        "Group", "Kicks", "Matches", "Goals", "Conversion", "P(goal) [95% interval]",
        "StatsBomb mean xG",
    ]  # fmt: skip
    body = [r for r in _rows(table) if len(r) == 7]
    assert [(r[1], r[2], r[3]) for r in body] == [
        (str(n), str(m), str(g)) for n, g, m in mc.PEN_ROWS
    ]
    assert [r[6] for r in body[7:]] == ["n/a", "0.8000", "n/a"]
    assert body[0][6] == "0.7835"
    held, dev = rows[2].rate, rows[1].rate
    validation = _section(card, "## Validation design")
    assert "has 3 in-game kicks" in validation and "this is a weak test" in validation
    assert f"{held.high - held.low:.3f} wide" in validation
    assert "lies inside it" in validation  # dev 0.6875 inside the hold-out's wide interval
    bench = _section(card, "## Benchmarks")
    assert _header(bench) == ["Model", "Kicks", "Log loss"]
    y = np.array([True, True, False])
    want = {f"{log_loss(y, np.full(3, q), labels=[False, True]):.4f}" for q in (dev.mean, 0.7835)}
    assert {r[2] for r in _rows(bench)} == want and {r[1] for r in _rows(bench)} == {"3"}
    limits = _section(card, "## Limitations & known biases")
    p = float(stats.chi2.sf(80 / 63, 4))
    assert (
        "Kicks are pooled as exchangeable: across 5 competition-seasons the Pearson dispersion "
        f"is 0.32 on 4 df (p {p:.2f}), and the most frequent taker has 3 of 10 kicks."
    ) in limits
    assert "cluster in 2 matches" in limits
    assert cards.LICENSING in _section(card, "## Data licensing")
    assert not re.search(r"\d{4}-\d{2}-\d{2}", card)
    assert "CI inside" not in card and "sb:" not in card


def test_penalty_card_no_holdout_and_na() -> None:
    """§3.3: without a hold-out row both sections say so; a dev estimate outside the hold-out
    interval says ``outside``; a null StatsBomb xG prints ``n/a``; an inestimable dispersion
    prints ``not estimable``."""
    from football_shot_models import models

    dev_only = tuple(k for k in mc.PEN_KICKS if k[1] not in ("sb:m01", "sb:m02"))
    card = _card(frame=mc.penalties_frame(dev_only))
    sentence = "The hold-out has no in-game penalties."
    assert sentence in _section(card, "## Validation design")
    assert sentence in _section(card, "## Benchmarks")
    frame = mc.with_null(mc.penalties_frame(), "provider_xg", 2)  # a hold-out kick
    rows = models.penalty_rows(frame)
    assert rows[2].provider_xg is None
    narrow = dataclasses.replace(rows[2], rate=metrics.Rate(3, 2, 2 / 3, 0.9, 0.8, 0.95))
    card = _card(rows=(*rows[:2], narrow, *rows[3:]), frame=frame)
    assert "lies outside it" in _section(card, "## Validation design")
    assert "n/a" in [r[2] for r in _rows(_section(card, "## Benchmarks"))]
    one = tuple(k for k in mc.PEN_KICKS if k[1] in ("sb:m01", "sb:m02") or k[2] == 5)
    assert "not estimable" in _section(
        _card(frame=mc.penalties_frame(one)), "## Limitations & known biases"
    )


@pytest.mark.parametrize(
    ("mean", "low", "high", "result", "margin", "extends"),
    [
        (0.7599, 0.735, 0.785, "missed", None, "below"),
        (0.76, 0.735, 0.785, "met", "0.0000 above the lower bound", "below"),
        (0.7627, 0.7374, 0.7872, "met", "0.0027 above the lower bound", "below"),
        (0.775, 0.765, 0.795, "met", "0.0150 above the lower bound", None),
        (0.79, 0.765, 0.815, "met", "0.0100 below the upper bound", "above"),
        (0.80, 0.775, 0.825, "met", "0.0000 below the upper bound", "above"),
        (0.8001, 0.775, 0.825, "missed", None, "above"),
    ],
)
def test_penalty_card_check_row(
    mean: float, low: float, high: float, result: str, margin: str | None, extends: str | None
) -> None:
    """§1.8, §3.3: the milestone checks the point estimate in [0.76, 0.80] (bounds inclusive),
    prints the interval and the margin to the nearer bound, and says when the interval extends
    below or above the 0.04 window."""
    from football_shot_models import models

    assert models.PENALTY_RANGE == (0.76, 0.80)
    rows = models.penalty_rows(mc.penalties_frame())
    pooled = dataclasses.replace(rows[0], rate=metrics.Rate(1118, 853, 853 / 1118, mean, low, high))
    card = _card(rows=(pooled, *rows[1:]))
    row = _check_row(card)
    assert row[-1] == result
    measured = " | ".join(row[1:-1])
    assert measured.startswith(f"{mean:.4f} [{low:.4f}, {high:.4f}]")
    if margin is not None:
        assert measured == f"{mean:.4f} [{low:.4f}, {high:.4f}]; {margin}"
    sentence = f"The 95% interval is {high - low:.3f} wide, wider than the 0.04 window, and extends"
    if extends is None:
        assert "and extends" not in card
    else:
        assert f"{sentence} {extends} it" in card


def test_penalty_card_without_team_selected() -> None:
    """run3-ms3 (football 10): under the milestone check, the pooled estimate without the
    team-selected seasons (all in-game kicks minus that family's) and whether it is inside the
    window; no team-selected row -> no line."""
    from football_shot_models import models

    rows = models.penalty_rows(mc.penalties_frame())
    team = next(r for r in rows if r.label == "In-game: team-selected")
    assert (team.rate.n, team.rate.goals) == (2, 1)
    pooled = dataclasses.replace(rows[0], rate=metrics.Rate(1118, 853, 853 / 1118, 0.7627, 0, 1))
    team_ts = dataclasses.replace(team, rate=metrics.beta_interval(171, 209))
    by = [pooled if r is rows[0] else team_ts if r is team else r for r in rows]
    rest = metrics.beta_interval(853 - 171, 1118 - 209)
    card = _card(rows=tuple(by))
    check = _section(card, "### Milestone check")
    assert (
        f"Without the team-selected seasons (209 kicks, conversion {171 / 209:.3f}) the pooled "
        f"estimate is {rest.mean:.4f} [{rest.low:.4f}, {rest.high:.4f}] from 909 kicks, outside "
        "the window."
    ) in check
    assert rest.mean < models.PENALTY_RANGE[0]
    inside = dataclasses.replace(team, rate=metrics.beta_interval(59, 100))
    by = [pooled if r is rows[0] else inside if r is team else r for r in rows]
    assert "from 1,018 kicks, inside the window." in _section(
        _card(rows=tuple(by)), "### Milestone check"
    )
    no_team = tuple(r for r in rows if r is not team)
    assert "Without the team-selected" not in _card(rows=no_team)


def test_exchangeability_null_is_goal() -> None:
    """run3-ms3 (QA 15): a null ``is_goal`` among in-game kicks raises (it would otherwise count as
    a miss in the dispersion)."""
    from football_shot_models import models

    with pytest.raises(ValueError, match="is_goal has 1 null"):
        models.exchangeability(mc.with_null(mc.penalties_frame(), "is_goal", 6))
