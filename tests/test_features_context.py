"""M1.2 context (spec MS3.features §1.3-§1.4, §3.1, §3.3): ``phase`` and ``contact`` are the only
readers of the restricted ``situation`` and ``technique`` columns."""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import polars as pl
import pytest
import xg_cases as xc
from football_core import schema
from hypothesis import given, settings

from football_shot_models import features

SITUATION_REASON = (
    "F5 (direct_corner / direct_free_kick shots are logged on outcome); the derived feature "
    "'phase' reads it with those levels handled"
)
TECHNIQUE_REASON = (
    "its 'lob' level is split off (lob is labelled from the ball's flight, post-shot); the "
    "derived feature 'contact' reads it with lob merged into normal"
)


def _shots(situations: list[str], ys: list[float], x: float = 85.0) -> pl.DataFrame:
    """C3 shots (distance and angle recomputed) with the given situations and y."""
    shots = [
        xc.Shot(f"sb:s{i}", x=x, y=y, situation=s, penalty=s == "penalty")
        for i, (s, y) in enumerate(zip(situations, ys, strict=True))
    ]
    return xc.shots_frame(shots)


def _levels(shots: pl.DataFrame, name: str, levels: tuple[str, ...]) -> list[str]:
    """Decode the indicator columns of one Enum feature back to its level per shot."""
    matrix, names = features.design_matrix(shots, [name])
    assert names == tuple(f"{name}={level}" for level in levels[1:])
    assert set(np.unique(matrix)) <= {0.0, 1.0} and np.all(matrix.sum(axis=1) <= 1.0)
    return [levels[1 + int(np.argmax(r))] if r.any() else levels[0] for r in matrix]


@pytest.mark.parametrize(
    ("situation", "y", "phase"),
    [
        ("open_play", 34.0, "open_play"),
        ("open_play", 5.0, "open_play"),  # the wide rule is for direct free kicks only
        ("kick_off", 34.0, "open_play"),
        ("fast_break", 30.0, "fast_break"),
        ("from_corner", 40.0, "corner"),
        ("from_corner", 5.0, "corner"),
        ("from_free_kick", 34.0, "free_kick"),
        ("from_throw_in", 34.0, "throw_in"),
        ("direct_free_kick", 34.0, "direct_free_kick"),
        ("direct_free_kick", 13.84, "direct_free_kick"),  # the box edge is central
        ("direct_free_kick", 13.84 - 1e-9, "free_kick"),
        ("direct_free_kick", 54.16, "direct_free_kick"),
        ("direct_free_kick", 54.16 + 1e-9, "free_kick"),
        ("direct_free_kick", 3.0, "free_kick"),
        ("direct_free_kick", 64.0, "free_kick"),
    ],
)
def test_phase_mapping(situation: str, y: float, phase: str) -> None:
    assert features.PHASES == (
        "open_play", "fast_break", "corner", "free_kick", "throw_in", "direct_free_kick",
    )  # fmt: skip
    assert pl.Enum(features.PHASES) == features.PHASE
    assert (features.BOX_RIGHT_Y, features.BOX_LEFT_Y) == (13.84, 54.16)
    # a second, central open-play shot keeps the frame two-class and is a near miss for order
    shots = _shots([situation, "open_play"], [y, 34.0])
    assert _levels(shots, "phase", features.PHASES) == [phase, "open_play"]


@pytest.mark.parametrize("situation", ["penalty", "direct_corner"])
def test_phase_excluded_levels_raise(situation: str) -> None:
    """Penalties and direct corners are not in the xG population: phase is null and
    design_matrix refuses it rather than filling a level."""
    shots = _shots([situation, "open_play"], [34.0 if situation == "penalty" else 0.5, 34.0])
    with pytest.raises(ValueError, match=r"feature 'phase' has 1 null value"):
        features.design_matrix(shots, ["phase"])


@settings(deadline=None, max_examples=60)
@given(st.lists(st.floats(0.0, 68.0), min_size=1, max_size=8))
def test_property_direct_free_kick_phase_by_y(ys: list[float]) -> None:
    shots = _shots(["direct_free_kick"] * len(ys), ys)
    expected = ["direct_free_kick" if 13.84 <= y <= 54.16 else "free_kick" for y in ys]
    assert _levels(shots, "phase", features.PHASES) == expected


def test_contact_merges_lob() -> None:
    """``contact`` is ``technique`` with lob read as normal: no lob level, so no indicator
    column that is always zero; every other level is kept as it is."""
    assert tuple(t for t in schema.TECHNIQUES if t != "lob") == features.CONTACTS
    assert pl.Enum(features.CONTACTS) == features.CONTACT
    techniques = list(schema.TECHNIQUES)
    shots = _shots(["open_play"] * len(techniques), [34.0] * len(techniques)).with_columns(
        technique=pl.Series(techniques, dtype=schema.TECHNIQUE)
    )
    expected = ["normal" if t == "lob" else t for t in techniques]
    assert _levels(shots, "contact", features.CONTACTS) == expected
    assert features.DERIVED["contact"].sources == (("shots", "technique"),)
    unknown = shots.with_columns(
        technique=pl.when(pl.col("shot_id") == "sb:s2").then(None).otherwise("technique")
    )
    with pytest.raises(ValueError, match=r"feature 'contact' has 1 null value"):
        features.design_matrix(unknown, ["contact"])


def test_restricted_readers() -> None:
    """``situation`` and ``technique`` are refused as names and as sources, with the reason;
    only the registered ``phase`` / ``contact`` objects read them, each only its own column."""
    assert dict(features.RESTRICTED_UNTIL) == {
        "situation": SITUATION_REASON,
        "technique": TECHNIQUE_REASON,
    }
    assert dict(features.RESTRICTED_READERS) == {
        "situation": frozenset({"phase"}),
        "technique": frozenset({"contact"}),
    }
    for column, reason in (("situation", SITUATION_REASON), ("technique", TECHNIQUE_REASON)):
        with pytest.raises(ValueError) as err:
            features.check_xg_features(["distance_m", column])
        assert str(err.value) == f"column {column!r} must not be an xG feature until {reason}"
        with pytest.raises(ValueError, match="must not be an xG feature until"):
            features.DerivedFeature("raw", (("shots", column),), lambda t: t["shots"]["y"])
    # the reader name is per column: contact may not read situation, phase may not read technique
    for name, column in (("contact", "situation"), ("phase", "technique")):
        with pytest.raises(ValueError, match="must not be an xG feature until"):
            features.DerivedFeature(name, (("shots", column),), lambda t: t["shots"]["y"])
    # a custom registry cannot pass raw situation through a feature named phase
    raw = features.DerivedFeature(
        "phase", (("shots", "situation"), ("shots", "y")), lambda t: t["shots"]["situation"]
    )
    shots = _shots(["direct_free_kick", "open_play"], [5.0, 34.0])
    needle = (
        r"derived feature 'phase' reads restricted column\(s\) .*situation.*; only "
        r"features\.DERIVED\['phase'\] may read them"
    )
    with pytest.raises(ValueError, match=needle):
        features.check_xg_features(["phase"], {"phase": raw})
    with pytest.raises(ValueError, match=needle):
        features.design_matrix(shots, ["phase"], derived={"phase": raw})
    # the registered objects pass, also from a copy of the registry
    names = ("phase", "contact")
    assert features.check_xg_features(names, dict(features.DERIVED)) == names
