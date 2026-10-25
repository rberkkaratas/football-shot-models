"""M0: the legacy v0 xG as a test-only benchmark (spec MS4.v1 §3).

v0's inputs are read through ``score_v0`` with hand-made coefficient vectors: with one
coefficient set to ``c`` and the intercept 0, ``logit(p) / c`` is that input, so a distance, an
angle or a flag is checked without re-implementing v0. The fixture v0 file holds invented
coefficients (``v1_cases.V0_COEF``), never the legacy file's.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import v1_cases as vc
from football_core import pitch, schema
from hypothesis import given, settings
from hypothesis import strategies as st

V0_FEATURES = (
    "log_dist", "dist_m", "angle_rad", "header", "big_chance", "one_on_one", "fast_break",
    "from_corner", "set_piece", "direct_freekick",
)  # fmt: skip
OPTA = {"x": [88.5, 100.0], "y": [50.0, 50.0]}


def _legacy() -> Any:
    from football_shot_models import legacy

    return legacy


def _frame(
    x: list[float], y: list[float], *, body: str = "right_foot", situation: str = "open_play",
    big: bool | None = False, one: bool | None = False,
) -> pl.DataFrame:  # fmt: skip
    n = len(x)
    return pl.DataFrame(
        {
            "x": pl.Series(x, dtype=pl.Float64()),
            "y": pl.Series(y, dtype=pl.Float64()),
            "body_part": pl.Series([body] * n, dtype=schema.BODY_PART),
            "situation": pl.Series([situation] * n, dtype=schema.SITUATION),
            "big_chance": pl.Series([big] * n, dtype=pl.Boolean()),
            "one_on_one": pl.Series([one] * n, dtype=pl.Boolean()),
        }
    )


def _opta_frame(**kw: Any) -> pl.DataFrame:
    cx, cy = pitch.to_canonical(OPTA["x"], OPTA["y"], "opta")
    return _frame([float(v) for v in cx], [float(v) for v in cy], **kw)


def _logit(p: pl.Series) -> np.ndarray[Any, Any]:
    q = p.to_numpy()
    return np.log(q / (1.0 - q))  # type: ignore[no-any-return]


def test_v0_known_cases() -> None:
    """§3, evidence: Opta (88.5, 50) is 12.075 m and 0.588607 rad from goal in v0's own linear
    0-100 -> 105 x 68 frame; Opta x = 100 clips the distance to 0.5 m (finite p); the output is
    ``v0_xg``, Float64; the full invented model gives the hand-computed probability."""
    legacy = _legacy()
    shots = _opta_frame()
    for name, value in (("dist_m", 12.075), ("log_dist", math.log(12.075)),
                        ("angle_rad", 0.588607)):  # fmt: skip
        out = legacy.score_v0(shots, vc.legacy_model({name: 0.1}))
        assert out.name == "v0_xg" and out.dtype == pl.Float64
        assert _logit(out)[0] / 0.1 == pytest.approx(value, abs=1e-6), name
    clipped = legacy.score_v0(shots, vc.legacy_model({"dist_m": 0.1}))
    assert _logit(clipped)[1] / 0.1 == pytest.approx(0.5, abs=1e-9)
    full = legacy.score_v0(shots, vc.legacy_model(vc.V0_COEF, vc.V0_INTERCEPT)).to_list()
    c = vc.V0_COEF
    angle = math.atan2(7.32 * 12.075, 12.075**2 - 3.66**2)  # posts 7.32 m apart, dy = 0
    assert angle == pytest.approx(0.588607, abs=1e-6)
    z = (vc.V0_INTERCEPT + c["log_dist"] * math.log(12.075) + c["dist_m"] * 12.075
         + c["angle_rad"] * angle)  # fmt: skip
    assert full[0] == pytest.approx(1.0 / (1.0 + math.exp(-z)), rel=1e-6)
    assert 0.0 < full[1] < 1.0 and math.isfinite(full[1])
    empty = legacy.score_v0(shots.head(0), vc.legacy_model(vc.V0_COEF))
    assert empty.len() == 0 and empty.dtype == pl.Float64


# flag weights: powers of two (in eighths), so logit(p) * 8 is the set of flags that fire
BITS = {"fast_break": 1, "from_corner": 2, "set_piece": 4, "direct_freekick": 8, "header": 16,
        "big_chance": 32, "one_on_one": 64}  # fmt: skip
FLAGS = {
    "open_play": (), "fast_break": ("fast_break",), "from_corner": ("from_corner",),
    "from_free_kick": ("set_piece",), "from_throw_in": ("set_piece",),
    "direct_free_kick": ("direct_freekick",), "direct_corner": (), "penalty": (), "kick_off": (),
}  # fmt: skip


@pytest.mark.parametrize("situation", schema.SITUATIONS)
@pytest.mark.parametrize("body", ["head", "right_foot", "other"])
def test_v0_flags(situation: str, body: str) -> None:
    """§3: v0's flags from C5 columns: ``header`` iff ``body_part == head``, ``set_piece`` iff
    the situation is a free kick or a throw-in, ``fast_break``, ``from_corner`` and
    ``direct_freekick`` from their own situation only (a direct corner fires none);
    ``big_chance`` and ``one_on_one`` read their columns."""
    legacy = _legacy()
    model = vc.legacy_model({k: v / 8 for k, v in BITS.items()})
    for big, one in ((False, False), (True, False), (False, True)):
        shots = _opta_frame(body=body, situation=situation, big=big, one=one)
        fired = round(float(_logit(legacy.score_v0(shots, model))[0]) * 8)
        expected = sum(BITS[f] for f in FLAGS[situation])
        expected += BITS["header"] * (body == "head") + BITS["big_chance"] * big
        expected += BITS["one_on_one"] * one
        assert fired == expected, (situation, body, big, one)


def test_v0_uncovered_null() -> None:
    """§3: ``covered`` is True iff x, y, body_part, situation, big_chance and one_on_one are all
    non-null (each element of the location pair on its own row); an uncovered shot scores null,
    never a fill."""
    legacy = _legacy()
    full = _frame([94.0] * 7, [34.0] * 7)
    rows = []
    for i, col in enumerate(("x", "y", "body_part", "situation", "big_chance", "one_on_one")):
        values = full[col].to_list()
        values[i] = None
        rows.append(pl.Series(col, values, dtype=full.schema[col]))
    shots = full.with_columns(rows)
    cov = legacy.covered(shots)
    assert cov.dtype == pl.Boolean and cov.to_list() == [False] * 6 + [True]
    out = legacy.score_v0(shots, vc.legacy_model(vc.V0_COEF, vc.V0_INTERCEPT))
    assert out.is_null().to_list() == [True] * 6 + [False]


READ_CASES: dict[str, tuple[dict[str, Any], str]] = {
    "features out of order": (
        vc.v0_doc(features=[*V0_FEATURES[1:], V0_FEATURES[0]]), "features"),
    "a feature missing": (vc.v0_doc(features=V0_FEATURES[:-1]), "features"),
    "coefficient keys missing one": (
        vc.v0_doc(coefficients={k: v for k, v in vc.V0_COEF.items() if k != "header"}),
        "features"),
    "an extra coefficient": (vc.v0_doc(coefficients={**vc.V0_COEF, "q215": 0.1}), "features"),
    "date without dashes": (vc.v0_doc(when="20260706"), "'20260706'"),
    "month 13": (vc.v0_doc(when="2026-13-01"), "'2026-13-01'"),
    "date as an int": (vc.v0_doc(when=20260706), "20260706"),
    "date with a time": (vc.v0_doc(when="2026-07-06T10:00"), "2026-07-06T10:00"),
}  # fmt: skip


@pytest.mark.parametrize("case", list(READ_CASES))
def test_read_v0(case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """§3: the SHA-256 pin is read from the module at call time; a file that does not hash as
    pinned is refused naming both digests; behind the pin, the features and coefficient keys
    must equal ``V0_FEATURES`` and ``trained_on.date`` must be a real ``YYYY-MM-DD`` date; a
    missing file is the OS's ``FileNotFoundError``."""
    legacy = _legacy()
    assert legacy.V0_FEATURES == V0_FEATURES
    path = tmp_path / "xg_model.json"
    sha = vc.write_v0(path)
    pinned = legacy.V0_SHA256
    with pytest.raises(ValueError, match=f"(?s)(?=.*{sha})(?=.*{pinned})"):
        legacy.read_v0(path)
    monkeypatch.setattr(legacy, "V0_SHA256", sha)
    model = legacy.read_v0(path)
    assert model.trained_on == dt.date(2026, 7, 6) and model.trained_shots == vc.V0_SHOTS
    assert model.sha256 == sha and model.penalty_xg == vc.V0_PENALTY
    assert model.intercept == vc.V0_INTERCEPT and dict(model.coefficients) == dict(vc.V0_COEF)
    doc, message = READ_CASES[case]
    monkeypatch.setattr(legacy, "V0_SHA256", vc.write_v0(path, doc))
    with pytest.raises(ValueError, match=message):
        legacy.read_v0(path)
    with pytest.raises(FileNotFoundError):
        legacy.read_v0(tmp_path / "missing.json")


@settings(deadline=None, max_examples=60)
@given(
    st.floats(0.0, 105.0), st.floats(0.0, 68.0), st.sampled_from(schema.BODY_PARTS),
    st.sampled_from(schema.SITUATIONS), st.booleans(), st.booleans(),
    st.sampled_from([1.0, 40.0, -40.0]),
)  # fmt: skip
def test_v0_proba_property(
    x: float, y: float, body: str, situation: str, big: bool, one: bool, scale: float
) -> None:
    """Property: any canonical point gives a finite p strictly inside (0, 1), extreme
    coefficients included (the linear predictor is clipped)."""
    legacy = _legacy()
    model = vc.legacy_model({k: v * scale for k, v in vc.V0_COEF.items()}, vc.V0_INTERCEPT)
    p = legacy.score_v0(_frame([x], [y], body=body, situation=situation, big=big, one=one), model)
    value = p.item()
    assert value is not None and math.isfinite(value) and 0.0 < value < 1.0


def test_refit_v0() -> None:
    """§3: the remove counterfactual fits v0's own columns minus ``drop`` on train rows only:
    one probability per held row; the held labels are never read; a dropped column no longer
    moves the predictions (it does when kept); an unknown name, a bare str or an uncovered row
    in either frame is a ``ValueError``."""
    legacy = _legacy()
    shots, _, matches = vc.v1_tables()
    kept = vc.kept_open_play(shots)
    test = pl.col("match_id").is_in(
        matches.filter(pl.col("season_name").is_in(vc.TEST_SEASONS))["match_id"].implode()
    )
    dev, held = kept.filter(~test), kept.filter(test)
    p = legacy.refit_v0(dev, held, ("big_chance",))
    assert p.shape == (held.height,) and bool(np.all((p > 0) & (p < 1)))
    flipped_y = held.with_columns(~pl.col("is_goal"))
    assert np.array_equal(legacy.refit_v0(dev, flipped_y, ("big_chance",)), p)
    flipped_big = held.with_columns(~pl.col("big_chance"))
    assert np.array_equal(legacy.refit_v0(dev, flipped_big, ("big_chance",)), p)
    kept_p = legacy.refit_v0(dev, held, ())
    assert not np.allclose(legacy.refit_v0(dev, flipped_big, ()), kept_p)  # it matters if kept
    assert not np.allclose(legacy.refit_v0(dev.with_columns(~pl.col("is_goal")), held, ()),
                           kept_p)  # fmt: skip  # the train labels are read
    no_one = dev.with_columns(
        pl.when(pl.int_range(pl.len()) == 3).then(None).otherwise(pl.col("one_on_one"))
        .alias("one_on_one")
    )  # fmt: skip
    for bad_dev, bad_held, drop in (
        (dev, held, ("distance_m",)),
        (dev, held, "big_chance"),
        (no_one, held, ("big_chance",)),
        (dev, no_one.head(held.height), ("big_chance",)),
    ):
        with pytest.raises(ValueError):
            legacy.refit_v0(bad_dev, bad_held, drop)
