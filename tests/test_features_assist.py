"""The derived feature ``assist`` (spec MS4.v1 §2.3): ``assist_type`` without the ``cutback``
level, which the WhoScored archive cannot record (C5 census), reference ``none``."""

from __future__ import annotations

import polars as pl
import pytest
from football_core import schema

from football_shot_models import features

LEVELS = ("none", "pass", "cross", "through_ball")


def _shots(values: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "shot_id": [f"ws:s{i}" for i in range(len(values))],
            "assist_type": pl.Series(values, dtype=schema.ASSIST_TYPE),
        }
    )


def test_assist_levels() -> None:
    """§2.3: one source column (``shots.assist_type``, ``pre_shot``); the Enum levels are
    ``schema.ASSIST_TYPES`` without ``cutback``, in schema order, so ``none`` is the reference
    and every other level gets an indicator (from the dtype: a level with no row still gets its
    column)."""
    feature = features.DERIVED["assist"]
    assert feature.name == "assist" and feature.sources == (("shots", "assist_type"),)
    assert tuple(a for a in schema.ASSIST_TYPES if a != "cutback") == LEVELS
    assert features.check_xg_features(["assist"]) == ("assist",)
    out = feature.fn({"shots": _shots(["cross", "none"])})
    assert isinstance(out.dtype, pl.Enum) and tuple(out.dtype.categories.to_list()) == LEVELS
    x, names = features.design_matrix(_shots(["none", "pass", "cross", "through_ball", "pass"]),
                                      ["assist"])  # fmt: skip
    assert names == ("assist=pass", "assist=cross", "assist=through_ball")
    assert x.tolist() == [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]]
    _, only_none = features.design_matrix(_shots(["none", "none"]), ["assist"])
    assert only_none == names


def test_assist_cutback_and_null() -> None:
    """§2.3: a ``cutback`` value is a ``ValueError`` naming it (never a constant-0 column); a null
    stays null, so ``design_matrix`` refuses it (no fill)."""
    with pytest.raises(ValueError, match="cutback"):
        features.design_matrix(_shots(["pass", "cutback", "none"]), ["assist"])
    out = features.DERIVED["assist"].fn({"shots": _shots(["pass", None, "none"])})
    assert out.to_list() == ["pass", None, "none"]
    with pytest.raises(ValueError, match="'assist' has 1 null"):
        features.design_matrix(_shots(["pass", None, "none"]), ["assist"])
