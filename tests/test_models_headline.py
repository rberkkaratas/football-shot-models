"""The headline rule and the sentences and checks that report it (spec MS3.models §1.6, §1.9,
§1.12, §3.2, §3.4, §6): ``headline``, ``side``, ``gain_cell``, ``verdict``,
``statsbomb_sentence``, ``milestone_checks``, the GBM card's Platt line and ``summary_table``.

Evaluations are stand-ins with chosen scores (``models_cases.fake_eval``); ``headline`` is also
run on real fixture evaluations.
"""

from __future__ import annotations

from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
from football_core import schema
from hypothesis import given
from hypothesis import strategies as st
from test_baselines_cards import _header, _rows

from football_shot_models import ablation, baselines, cards, evaluation, metrics

S0, S1, S3 = baselines.BASELINES[2], ablation.STEPS[0], ablation.STEPS[2]
LR_GAIN = evaluation.Comparison(
    f"over {S3.title}", mc.gain(0.0035, 0.0028, 0.0043), mc.gain(0.0014, -0.0005, 0.0034)
)


def _gbm(**kwargs: Any) -> evaluation.Evaluation:
    from football_shot_models import models

    return mc.fake_eval(models.GBM, **kwargs)


# ---- headline ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cv_low", "ho_low", "gbm_is_head"),
    [(1e-6, 0.001, True), (0.0, 0.001, False), (-1e-6, 0.001, False), (0.001, -0.002, True)],
    ids=["cv low +1e-6", "cv low 0.0", "cv low -1e-6", "cv above, hold-out below"],
)
def test_headline_rule(
    monkeypatch: pytest.MonkeyPatch, cv_low: float, ho_low: float, gbm_is_head: bool
) -> None:
    """§1.6, §3.2: the GBM is headline iff its CV gain over the best LR has ``low > 0`` (0.0 is
    not); the hold-out does not decide. ``headline`` calls ``compare`` itself, with the LR as the
    reference, and returns that comparison."""
    from football_shot_models import models

    lr, gbm = mc.fake_eval(S3), _gbm()
    dev, held = pl.DataFrame({"k": [1]}), pl.DataFrame({"k": [2]})
    row = evaluation.Comparison(
        "stub", mc.gain(cv_low + 0.001, cv_low, cv_low + 0.002), mc.gain(0.0, ho_low, 0.003)
    )
    seen: list[tuple[Any, ...]] = []

    def fake(*args: Any) -> evaluation.Comparison:
        seen.append(args)
        return row

    monkeypatch.setattr(models, "compare", fake, raising=False)
    monkeypatch.setattr(evaluation, "compare", fake)
    head, got = models.headline(lr, gbm, dev, held)
    assert got is row
    assert head is (gbm if gbm_is_head else lr)
    assert len(seen) == 1
    label, ref, new, d, h = seen[0]
    assert ref is lr and new is gbm and d is dev and h is held
    assert label == f"over {S3.title}"


def test_headline_compares_real_evaluations(
    run_result: tuple[Any, Any], dev_held: tuple[pl.DataFrame, pl.DataFrame]
) -> None:
    """On real fixture evaluations: the returned comparison is ``compare(best_lr, gbm)``, so
    swapping the arguments flips the gain's sign."""
    from football_shot_models import models

    evaluations, _ = run_result
    dev, held = dev_held
    a, b = evaluations[1], evaluations[2]
    head, got = models.headline(a, b, dev, held)
    want = evaluation.compare(f"over {a.spec.title}", a, b, dev, held)
    assert (got.label, got.cv, got.holdout) == (want.label, want.cv, want.holdout)
    assert head is (b if want.cv.low > 0 else a)
    _, swapped = models.headline(b, a, dev, held)
    assert swapped.cv.gain == pytest.approx(-want.cv.gain, abs=1e-12)


# ---- side, gain_cell --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [
        (1e-9, 0.002, "lies above 0"),
        (0.0, 0.002, "includes 0"),
        (-0.001, 0.0, "includes 0"),
        (-0.001, 0.001, "includes 0"),
        (-0.002, -1e-9, "lies below 0"),
        (float("nan"), 0.1, None),
        (-0.1, float("nan"), None),
    ],
)
def test_side(low: float, high: float, expected: str | None) -> None:
    """§3.2: above iff ``low > 0``, below iff ``high < 0``, else includes; a NaN bound raises."""
    g = mc.gain(0.0, low, high)
    if expected is None:
        with pytest.raises(ValueError):
            cards.side(g)
    else:
        assert cards.side(g) == expected


def test_gain_cell_signed() -> None:
    """§6: the sign is kept unless the value is exactly 0, so a tiny negative prints
    ``-0.0000`` (MS3.features printed it as ``+0.0000``)."""
    assert cards.gain_cell(mc.gain(-4e-6, -6e-5, 0.0)) == "-0.0000 [-0.0001, +0.0000]"
    assert cards.gain_cell(mc.gain(0.0, -0.0, 4e-6)) == "+0.0000 [+0.0000, +0.0000]"
    assert cards.gain_cell(mc.gain(0.0035, -0.0005, 0.0123)) == "+0.0035 [-0.0005, +0.0123]"


@given(st.floats(-1.0, 1.0, allow_nan=False).filter(lambda v: v != 0.0))
def test_signed_sign(x: float) -> None:
    """Property: a non-zero gain prints with its own sign, however small."""
    cell = cards.gain_cell(mc.gain(x, x, x))
    assert cell[0] == ("-" if x < 0 else "+")
    assert cell.split(" [")[1][0] == cell[0]


# ---- verdict, StatsBomb sentence --------------------------------------------------------------

V_HEAD_GBM = (
    "This is the headline model: its CV log-loss gain over LR splines + context + freeze frames "
    "is +0.0035 [+0.0028, +0.0043], a 95% interval that lies above 0; on the hold-out it is "
)
LR_STAYS = evaluation.Comparison(
    f"over {S3.title}", mc.gain(0.0004, -0.0003, 0.0011), mc.gain(0.0001, -0.0010, 0.0012)
)


@pytest.mark.parametrize(
    ("ev", "head", "gain", "expected"),
    [
        (
            "gbm",
            "gbm",
            LR_GAIN,
            V_HEAD_GBM + "+0.0014 [-0.0005, +0.0034], an interval that includes 0.",
        ),
        (
            "gbm",
            "gbm",
            evaluation.Comparison(LR_GAIN.label, LR_GAIN.cv, mc.gain(-0.002, -0.004, -0.0001)),
            V_HEAD_GBM + "-0.0020 [-0.0040, -0.0001], an interval that lies below 0.",
        ),
        (
            "s3",
            "s3",
            LR_STAYS,
            "This is the headline model: the GBM's CV gain over it is +0.0004 "
            "[-0.0003, +0.0011], an interval that does not lie above 0, so the GBM does not ship "
            "as headline.",
        ),
        (
            "gbm",
            "s3",
            LR_STAYS,
            "Not the headline model: its CV gain over LR splines + context + "
            "freeze frames is +0.0004 [-0.0003, +0.0011], an interval that does not lie above 0, "
            "so LR splines + context + freeze frames stays headline.",
        ),
        ("s3", "gbm", LR_GAIN, "Not the headline model; that is Monotone GBM + Platt."),
        (
            "s1",
            "s3",
            LR_STAYS,
            "Not the headline model; that is LR splines + context + freeze frames.",
        ),
    ],  # fmt: skip
    ids=[
        "GBM head, hold-out includes 0",
        "GBM head, hold-out below 0",
        "LR head on its card",
        "LR head on the GBM card",
        "GBM head on an LR card",
        "LR head on another LR card",
    ],
)
def test_verdict_sentences(ev: str, head: str, gain: Any, expected: str) -> None:
    """§3.2: the four overview sentences, with cells from ``gain_cell`` and ``side``."""
    from football_shot_models import models

    evs = {"gbm": _gbm(), "s3": mc.fake_eval(S3), "s1": mc.fake_eval(S1)}
    assert models.verdict(evs[ev], evs[head], gain) == expected


@pytest.mark.parametrize(
    ("sb_gain", "cv_side", "ho_side"),
    [
        (
            evaluation.Comparison(
                "over StatsBomb xG",
                mc.gain(0.0021, 0.0012, 0.0031),
                mc.gain(-0.0013, -0.004, 0.0016),
            ),
            "+0.0021 [+0.0012, +0.0031], an interval that lies above 0",
            "-0.0013 [-0.0040, +0.0016], an interval that includes 0",
        ),
        (
            evaluation.Comparison(
                "over StatsBomb xG",
                mc.gain(0.0001, -0.0002, 0.0004),
                mc.gain(-0.003, -0.005, -0.001),
            ),
            "+0.0001 [-0.0002, +0.0004], an interval that includes 0",
            "-0.0030 [-0.0050, -0.0010], an interval that lies below 0",
        ),
    ],
)
def test_statsbomb_sentence(sb_gain: Any, cv_side: str, ho_side: str) -> None:
    """§1.12, §3.2: both intervals with their side, and goals predicted against goals scored
    with the sign, for the model and for StatsBomb; never "beats" or "matches"."""
    from football_shot_models import models

    ev = _gbm(holdout=mc.scores(n=9816, goals=913, xg=975.2))
    bench = mc.fake_eval(evaluation.BENCHMARK, holdout=mc.scores(n=9816, goals=913, xg=900.4))
    got = mc.statsbomb_sentence(models, ev, sb_gain, bench)
    assert got == (
        f"Against StatsBomb's own xG: CV gain {cv_side}; hold-out gain {ho_side}. On the hold-out "
        f"it predicts 975 goals for 913 scored ({62.2 / 913:+.2%}); StatsBomb predicts 900 "
        f"({-12.6 / 913:+.2%})."
    )
    assert "+6.81%" in got and "-1.38%" in got
    assert "beat" not in got and "match" not in got


# ---- milestone checks -------------------------------------------------------------------------


def _bucket(n_low: int, g_low: int, n_high: int, g_high: int) -> tuple[np.ndarray, np.ndarray]:
    """Shots at p = 0.1 and p = 0.5 with the given goals: two groups, so the calibration slope
    is exactly logit(rate_high) - logit(rate_low) over logit(0.5) - logit(0.1)."""
    p = np.concatenate([np.full(n_low, 0.1), np.full(n_high, 0.5)])
    y = np.concatenate([np.arange(n_low) < g_low, np.arange(n_high) < g_high])
    return p, y


def _frame(parts: list[tuple[str, np.ndarray]]) -> pl.DataFrame:
    gender = [g for g, y in parts for _ in range(y.size)]
    y = np.concatenate([y for _, y in parts]) if parts else np.empty(0, dtype=bool)
    return pl.DataFrame(
        {
            "match_id": pl.Series([f"sb:m{i // 20:03d}" for i in range(y.size)], dtype=pl.String()),
            "is_goal": pl.Series(y, dtype=pl.Boolean()),
            "gender": pl.Series(gender, dtype=schema.GENDER),
        }
    )


def _checks(
    *, head: str = "gbm", women: int = 500, ho_slope: float | None = 0.953,
    cv_ll: float = 0.2638, fold_xg: tuple[float, ...] = (101.0, 98.0, 103.0, 95.0, 105.0),
    gbm_low: float = 0.0028,
) -> tuple[Any, ...]:  # fmt: skip
    from football_shot_models import models

    men_p, men_y = _bucket(500, 50, 500, 250)  # slope 1.0
    wom_p, wom_y = _bucket(500, 25, women, women // 2)  # rates 0.05 / 0.5: slope 1.34
    dev = _frame([("male", men_y), ("female", wom_y)])
    held = _frame([("male", men_y)])
    folds = tuple(mc.scores(n=400, goals=100, xg=x, log_loss=cv_ll, slope=1.01) for x in fold_xg)
    spec = models.GBM if head == "gbm" else S3
    ev = mc.fake_eval(
        spec,
        folds=folds,
        holdout=mc.scores(n=1000, goals=300, xg=310.0, slope=ho_slope),
        oof=np.concatenate([men_p, wom_p]),
        holdout_p=men_p,
    )
    bench = mc.fake_eval(evaluation.BENCHMARK, cv=mc.scores(log_loss=0.2659))
    s0 = evaluation.Comparison(
        f"over {S0.title}", mc.gain(0.0234, 0.0219, 0.0249), mc.gain(0.0188, 0.0145, 0.0233)
    )
    sb = evaluation.Comparison(
        "over StatsBomb xG", mc.gain(0.0021, 0.0012, 0.0031), mc.gain(-0.0013, -0.004, 0.0016)
    )
    gbm = evaluation.Comparison(
        LR_GAIN.label, mc.gain(gbm_low + 0.0007, gbm_low, gbm_low + 0.0015), LR_GAIN.holdout
    )
    s0_ev = mc.fake_eval(S0, cv=mc.scores(log_loss=0.2872))
    return models.milestone_checks(
        ev, bench, s0_gain=s0, sb_gain=sb, gbm_gain=gbm, dev=dev, held=held, s0=s0_ev
    )


def test_milestone_checks() -> None:
    """§3.2: eleven rows in order (S0 gain, StatsBomb gap, six slopes, fold Σ xG mean and max,
    GBM over the best LR). Near misses: the women's pooled slope (1.34) misses while the men's
    meets; fold Σ xG errors 1, 2, 3, 5, 5 % have a pooled error of 0.4 % and a median of 3 %
    but a mean of 3.2 % (missed), and a max of exactly 5 % (met, inclusive)."""
    from football_shot_models import models

    assert models.SLOPE_RANGE == (0.9, 1.1)
    assert (models.SB_GAP, models.SB_MS2_CV, models.MS2_BEST_CV) == (0.006, 0.2674, 0.2880)
    assert (models.FOLD_XG_MEAN, models.FOLD_XG_MAX, models.MIN_SEGMENT_SHOTS) == (0.03, 0.05, 1000)
    checks = _checks()
    assert all(isinstance(c, metrics.Check) for c in checks)
    assert [c.result for c in checks] == [
        "met", "met",
        "met", "met", "met", "missed", "met", "n/a",
        "missed", "met",
        "met",
    ]  # fmt: skip
    assert "0.2880" in checks[0].target and "Monotone GBM + Platt" in checks[0].measured
    assert "0.2872" in checks[0].target  # S0's CV on these folds (``s0=``, Engineer's seam)
    assert "0.2659" in checks[1].target and "0.2674" in checks[1].target
    assert "0.2638" in checks[1].measured and "0.2659" in checks[1].measured
    assert "0 women's shots in the hold-out" in checks[7].measured
    assert any("0.40%" in c.measured for c in checks[8:10])  # the pooled error is shown
    last = checks[10].measured
    assert cards.gain_cell(mc.gain(0.0035, 0.0028, 0.0043)) in last
    assert cards.gain_cell(LR_GAIN.holdout) in last and cards.side(LR_GAIN.holdout) in last
    with pytest.raises(AttributeError):
        checks[0].result = "missed"  # type: ignore[misc]  # frozen


def test_milestone_checks_lr_head_and_na() -> None:
    """§3.2: with an LR headline, checks 1 and 2 cite it, not the GBM; the StatsBomb gap of
    0.0061 misses (0.0059 meets); an inestimable hold-out slope and a 999-shot gender group are
    ``n/a`` with the reason; one fold at 6 % misses the max but not the mean; a GBM whose CV gain
    does not lie above 0 misses check 5."""
    checks = _checks(
        head="lr", women=499, ho_slope=None, cv_ll=0.2720,
        fold_xg=(100.0, 100.0, 100.0, 100.0, 106.0), gbm_low=-0.0001,
    )  # fmt: skip
    assert [c.result for c in checks] == [
        "met", "missed",
        "met", "n/a", "met", "n/a", "met", "n/a",
        "met", "missed",
        "missed",
    ]  # fmt: skip
    for c in checks[:2]:
        assert "Monotone GBM" not in c.measured
    assert S3.title in checks[0].measured
    assert "0.2720" in checks[1].measured
    assert "not estimable" in checks[3].measured
    assert "999" in checks[5].measured
    assert [c.result for c in _checks(cv_ll=0.2718)][1] == "met"


# ---- GBM card text ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "line"),
    [
        (
            -0.0122,
            0.9923,
            "Calibration is close to identity (Platt slope 0.992, intercept -0.012): "
            "the boosted model is already calibrated and Platt is a safeguard.",
        ),
        (
            -0.049,
            0.951,
            "Calibration is close to identity (Platt slope 0.951, intercept -0.049): "
            "the boosted model is already calibrated and Platt is a safeguard.",
        ),
        (0.0, 1.06, "Platt moves the scores: slope 1.060, intercept +0.000."),
        (0.051, 1.0, "Platt moves the scores: slope 1.000, intercept +0.051."),
    ],  # fmt: skip
)
def test_gbm_model_section_platt(a: float, b: float, line: str) -> None:
    """§3.4: the Platt sentence is computed from the dev fit's settings rows with
    ``NEAR_IDENTITY`` on both the slope and the intercept; the no-peeking sentence and the
    scikit-learn version are always there."""
    from football_shot_models import models

    assert models.NEAR_IDENTITY == 0.05
    coefficients = (
        ("max_leaf_nodes", 16, 16), ("iterations", 200, 200), ("platt_intercept", a, a),
        ("platt_slope", b, b), ("monotone:distance_m", -1, -1),
    )  # fmt: skip
    ev = _gbm(coefficients=coefficients)
    text = "\n".join(models.CARD.model(ev, mc.card_context()))
    assert line in text
    other = "Platt moves" if "identity" in line else "close to identity"
    assert other not in text
    assert (
        "Tuning and calibration see only the rows of the fit they belong to; each outer test "
        "fold is scored by a model that never saw it."
    ) in text
    assert "scikit-learn 1.9.1" in text


# ---- summary table ----------------------------------------------------------------------------


def test_summary_table() -> None:
    """§3.4: every model then every benchmark, the headline in bold, calibration slopes (``n/a``
    for None) and the signed hold-out Σ xG error; then the StatsBomb sentence, the penalty line
    and the match-xG sentence; nothing claims a CI inside the window or a win over StatsBomb."""
    from football_shot_models import models

    s0 = mc.fake_eval(S0, holdout=mc.scores(goals=913, xg=975.0, log_loss=0.2721, slope=0.965))
    s3 = mc.fake_eval(S3, holdout=mc.scores(goals=913, xg=960.2, log_loss=0.2547, slope=None))
    gbm = _gbm(holdout=mc.scores(goals=913, xg=975.2, log_loss=0.2533, slope=0.953))
    sb = mc.fake_eval(
        evaluation.BENCHMARK, holdout=mc.scores(goals=913, xg=900.4, log_loss=0.252, slope=0.992)
    )
    sb_gain = evaluation.Comparison(
        "over StatsBomb xG", mc.gain(0.0021, 0.0012, 0.0031), mc.gain(-0.0013, -0.004, 0.0016)
    )
    rate = metrics.beta_interval(7, 10)
    penalty = models.PenaltyRow("in-game", rate, 6, 0.7835)
    text = models.summary_table((s0, s3, gbm), (sb,), gbm, penalty, sb_gain)
    assert _header(text) == [
        "Model", "CV log loss", "CV AUC", "CV cal. slope", "Hold-out log loss",
        "Hold-out cal. slope", "Hold-out Σ xG error (signed)",
    ]  # fmt: skip
    expected = []
    for ev in (s0, s3, gbm, sb):
        title = f"**{ev.spec.title}**" if ev is gbm else ev.spec.title
        signed = (ev.holdout.xg - ev.holdout.goals) / ev.holdout.goals
        expected.append([
            title, cards.fmt(ev.cv.log_loss), cards.fmt(ev.cv.auc), cards.fmt(ev.cv.cal_slope),
            cards.fmt(ev.holdout.log_loss), cards.fmt(ev.holdout.cal_slope), f"{signed:+.2%}",
        ])  # fmt: skip
    assert _rows(text) == expected
    assert expected[1][5] == "n/a" and expected[2][6] == "+6.81%" and expected[3][6] == "-1.38%"
    assert mc.statsbomb_sentence(models, gbm, sb_gain, sb) in text
    penalty_line = (
        f"Penalties (in-game): P(goal) point estimate {rate.mean:.4f} [95% interval "
        f"{rate.low:.4f}, {rate.high:.4f}], 10 kicks; shoot-outs reported apart."
    )
    assert penalty_line in text
    assert "match xG" in text
    assert "CI inside" not in text and "beats" not in text.lower()
