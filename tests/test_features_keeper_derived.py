"""MS6 E5 guard: the keeper-derived context columns may enter xG but never the xGOT input chain
(spec ``runs/PR1/xg-context`` §3.1)."""

from __future__ import annotations

import pytest

from football_shot_models import features

GUARDS = {
    "check_keeper_neutral": (lambda names: features.check_keeper_neutral(names), ()),
    "check_xgot_features": (lambda names: features.check_xgot_features(names), ("xg_logit",)),
}


@pytest.mark.parametrize("column", ["prev_action", "rebound"])
@pytest.mark.parametrize("guard", sorted(GUARDS))
def test_keeper_derived_refused_by_xgot_guards(guard: str, column: str) -> None:
    """§3.1: both xGOT guards refuse each ``KEEPER_DERIVED`` column (the whole ``prev_action``
    column, not only its ``opp_keeper`` level), and the message names MS6 E5. Near miss: the
    same names without it pass."""
    assert frozenset({"prev_action", "rebound"}) == features.KEEPER_DERIVED
    check, base = GUARDS[guard]
    ok = (*base, "distance_m", "angle_rad")
    assert check(ok) == ok
    with pytest.raises(ValueError, match="E5") as err:
        check((*ok, column))
    assert column in str(err.value)


def test_keeper_derived_allowed_in_xg() -> None:
    """§3.1: ``check_xg_features`` accepts both columns (role ``pre_shot``), so WhoScored xG v2
    may read them; the keeper-position guard still refuses ``open_goal`` beside them."""
    names = ("distance_m", "angle_rad", "prev_action", "rebound")
    assert features.check_xg_features(names) == names
    with pytest.raises(ValueError, match="keeper-position"):
        features.check_keeper_neutral(("distance_m", "open_goal"))
