"""M1.0 leakage guard: the xG allow-list is the C3 ``pre_shot`` role minus the columns restricted
until a follow-up lands, for C3 columns and for the sources of derived features (spec §3)."""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
import xg_cases as xc
from football_core import schema
from hypothesis import given

from football_shot_models import baselines, features

ROLE = {c.name: c.role for c in schema.SHOT_COLUMNS}
NOT_PRE_SHOT = tuple(name for name, role in ROLE.items() if role != "pre_shot")
FRAME_ROLE = {c.name: c.role for c in schema.FREEZE_FRAME_COLUMNS}
RESTRICTED = ("situation",)  # pre_shot in C3, but recorded on outcome for direct set pieces


def test_baseline_features_are_pre_shot() -> None:
    """The literal criterion: every feature of every baseline has C3 role ``pre_shot``."""
    assert [b.name for b in baselines.BASELINES] == [
        "xg-base-rate",
        "xg-lr-distance-angle",
        "xg-lr-distance-angle-body-part",
    ]
    for baseline in baselines.BASELINES:
        for name in baseline.features:
            assert ROLE[name] == "pre_shot", (baseline.name, name, ROLE.get(name))
        assert features.check_xg_features(baseline.features) == tuple(baseline.features)


def test_allow_list_is_c3_pre_shot() -> None:
    expected = {name for name, role in ROLE.items() if role == "pre_shot"} - set(RESTRICTED)
    assert frozenset(expected) == features.XG_ALLOWED
    assert len(features.XG_ALLOWED) == 20
    assert tuple(features.RESTRICTED_UNTIL) == RESTRICTED
    assert "big_chance" not in features.XG_ALLOWED  # restricted: ablation only, null for SB


@pytest.mark.parametrize("column", NOT_PRE_SHOT)
def test_non_pre_shot_column_rejected(column: str) -> None:
    """Parametrised from C3 itself: a new non-pre_shot column is covered automatically."""
    shots = xc.shots_frame([xc.Shot("sb:s1"), xc.Shot("sb:s2", goal=True)])
    for call in (
        lambda: features.check_xg_features(["distance_m", column]),
        lambda: features.design_matrix(shots, ["distance_m", column]),
    ):
        with pytest.raises(ValueError, match="pre_shot") as err:
            call()
        assert repr(column) in str(err.value)
        assert repr(ROLE[column]) in str(err.value)


@pytest.mark.parametrize("column", RESTRICTED)
def test_restricted_column_rejected(column: str) -> None:
    """``situation`` is refused with its reason until F5, although C3 calls it pre_shot."""
    shots = xc.shots_frame([xc.Shot("sb:s1"), xc.Shot("sb:s2", goal=True)])
    for call in (
        lambda: features.check_xg_features(["distance_m", column]),
        lambda: features.design_matrix(shots, ["distance_m", column]),
    ):
        with pytest.raises(ValueError, match=rf"'{column}' must not be an xG feature until F5"):
            call()


def _derived(*sources: tuple[str, str]) -> features.DerivedFeature:
    return features.DerivedFeature("d", sources, lambda t: t["shots"]["shot_id"])


@pytest.mark.parametrize(
    ("source", "needle"),
    [
        *((("shots", c), f"'{c}' has role '{ROLE[c]}'") for c in NOT_PRE_SHOT),
        *(
            (("freeze_frames", c), f"freeze_frames column '{c}' has role '{r}'")
            for c, r in FRAME_ROLE.items()
            if r != "pre_shot"
        ),
        *((("shots", c), "must not be an xG feature until") for c in RESTRICTED),
        (("shots", "gender"), "'gender' is not a column of the C3 shots table"),
        (("matches", "gender"), "unknown table 'matches'"),
    ],
    ids=lambda v: v if isinstance(v, str) and " " not in v else None,
)
def test_derived_feature_source_rejected(source: tuple[str, str], needle: str) -> None:
    """The allow-list extended to derived features: any source that is not an unrestricted
    pre_shot column of C3 shots or freeze frames is refused when the feature is made."""
    with pytest.raises(ValueError, match="pre_shot" if "role" in needle else ".") as err:
        _derived(("shots", "distance_m"), source)
    assert needle in str(err.value)


@pytest.mark.parametrize(
    ("bad", "error", "needle"),
    [
        (["distance_m", "gender"], ValueError, "'gender' is not a column of the C3 shots table"),
        (["family"], ValueError, "'family' is not a column of the C3 shots table"),
        (["made_up"], ValueError, "'made_up' is not a column"),
        ("distance_m", TypeError, "not a single str"),
        (["distance_m", "distance_m"], ValueError, "duplicate"),
    ],
    ids=["gender", "family", "made_up", "bare str", "duplicate"],
)
def test_bad_feature_lists_rejected(bad: object, error: type[Exception], needle: str) -> None:
    with pytest.raises(error, match=needle):
        features.check_xg_features(bad)  # type: ignore[arg-type]


@given(st.lists(st.sampled_from([*ROLE, "gender", "family", "xg"]), max_size=6))
def test_property_guard_accepts_exactly_unique_pre_shot(names: list[str]) -> None:
    allowed = all(ROLE.get(n) == "pre_shot" and n not in RESTRICTED for n in names)
    ok = allowed and len(set(names)) == len(names)
    if ok:
        assert features.check_xg_features(names) == tuple(names)
    else:
        with pytest.raises(ValueError):
            features.check_xg_features(names)
