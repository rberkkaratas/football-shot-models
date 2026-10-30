"""Ratify tests for the docs-sites build decisions (``build.md`` "Architect: ratify" R2-R11 and
the result-audit A2 scope): number-token boundaries, attribute docstrings, the results block's
note rule, the logo box, the theme variables and logo, the ``docs-results`` target, the result
labels, and the site-wide versus card-only CSS rules.

Fixtures are written by hand into ``tmp_path``; committed files only (``pyproject.toml``,
``Makefile``) and the installed Zensical stylesheets are read."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import zensical
from football_core import viz
from football_docs import pages, theme
from PIL import Image

from football_shot_charts import rows

REPO = Path(__file__).resolve().parents[1]


# --- R2: number-token boundaries --------------------------------------------------------------

TOKENS = [
    # (text, tokens unsourced against an empty source)
    ("MS6", []),
    ("q214", []),
    ("3.3m", []),
    ("v1.2", []),
    ("1,118 shots", ["1,118"]),
    ("log loss 0.2641", ["0.2641"]),
    ("ends at 0.2641.", ["0.2641"]),
    ("of 1,118, and", ["1,118"]),  # a trailing comma is punctuation, not part of the number
    ("(12)", ["12"]),
    ("2015/16", ["2015", "16"]),
]


@pytest.mark.parametrize(("text", "expected"), TOKENS, ids=[t for t, _ in TOKENS])
def test_number_token_boundaries(text: str, expected: list[str]) -> None:
    """R2: a number token starts after no word character, ``.`` or ``,`` and ends before no word
    character or ``[.,]digit``: codes and units (``MS6``, ``q214``, ``3.3m``) give no token,
    ``1,118`` and ``0.2641`` give one each, sentence punctuation is not part of a token."""
    assert pages.unsourced_numbers(text, "") == expected
    # a source holding the same number as a token sources it
    assert pages.unsourced_numbers(text, " ".join(expected) or "none") == []


# --- R3: attribute docstrings -----------------------------------------------------------------

ATTRIBUTES = '''\
"""Module."""
import os

X = 1
"""Attribute X: see R14."""
os.getcwd()
"""A string after a call is not a docstring: R15."""
Y = 2
os.getcwd()
"""Two statements after an assignment: R16."""


class C:
    """Class."""

    y: int = 2
    """Annotated attribute: MS6."""


if X:
    W = 1
    """Inside an if body: run8."""


def f() -> None:
    v = 1
    """After a local assignment: A5."""
'''


def test_attribute_docstrings_are_scanned(tmp_path: Path) -> None:
    """R3: ``docstring_codes`` reads every string statement directly after an ``Assign`` or
    ``AnnAssign``, in any body (module, class, ``if``, function); a string after any other
    statement is not a docstring."""
    src = tmp_path / "attrs.py"
    src.write_text(ATTRIBUTES)
    compile(ATTRIBUTES, str(src), "exec")  # precondition: valid Python
    lines = ATTRIBUTES.splitlines()
    got = [(line, code) for line, code, _ in pages.docstring_codes(src)]
    assert got == [(5, "R14"), (17, "MS6"), (22, "run8"), (27, "A5")]
    for line, code in got:
        assert code in lines[line - 1]  # file line numbers point at the code


# --- R4: the results block --------------------------------------------------------------------

README = (
    "<!-- m:summary:start -->\n"
    "| Model | CV log loss |\n|---|---|\n| **GBM** | 0.2641 |\n\n"
    "Shrunken: gain +0.0123 [-0.0045, +0.0301], interval above 0. Another sentence follows.\n"
    "<!-- m:summary:end -->\n"
)


def test_results_block_note_and_label_rules() -> None:
    """R4: the block has no trailing newline; a note sentence runs to the first ``.`` followed
    by whitespace or the line end (decimal points inside it do not end it); a number in a result
    label must occur in the README."""
    pick = pages.Pick("xG, GBM: log loss 0.2641", "m", ("GBM",), ("CV log loss",), ("Shrunken:",))
    block = pages.results_block(README, [pick])
    assert block.endswith("<!-- results:end -->") and not block.endswith("\n")
    note = "Shrunken: gain +0.0123 [-0.0045, +0.0301], interval above 0."
    assert block.splitlines()[-3:-1] == [note, ""]
    assert "Another sentence" not in block
    assert "7" not in pages.unsourced_numbers(README, "none")  # precondition: 7 is not sourced
    bad = pages.Pick("xG on 7 seasons", "m", ("GBM",), ("CV log loss",))
    with pytest.raises(ValueError, match=r"\['7'\] are not in the README"):
        pages.results_block(README, [bad])


# --- R5: the logo box -------------------------------------------------------------------------

H, W = 300, 400
PAD, SIDE = viz.LOGO_PLATE_PAD_PX, viz.LOGO_PX
TOP, LEFT = H - viz.BOTTOM_PX - SIDE - PAD, W - viz.MARGIN_PX - SIDE - PAD
BOTTOM, RIGHT = H - viz.BOTTOM_PX + PAD, W - viz.MARGIN_PX + PAD


def _image(mode: str, path: Path) -> Path:
    """Logo red over the box plus a one-pixel ring outside it, in ``mode``."""
    a = np.full((H, W, 4), (248, 250, 252, 255), dtype=np.uint8)
    a[TOP - 1 : BOTTOM + 1, LEFT - 1 : RIGHT + 1, :3] = rows.LOGO_RGB
    img = Image.fromarray(a, "RGBA")
    (img if mode == "RGBA" else img.convert(mode)).save(path)
    return path


BOX = (SIDE + 2 * PAD) ** 2


@pytest.mark.parametrize(("mode", "expected"), [("RGBA", BOX), ("RGB", BOX), ("L", 0)])
def test_logo_box_geometry(mode: str, expected: int, tmp_path: Path) -> None:
    """R5: the box is the ``LOGO_PX`` square right-aligned at ``MARGIN_PX`` above ``BOTTOM_PX``,
    padded by ``LOGO_PLATE_PAD_PX`` on every side, exactly (the ring outside it is not counted);
    a 3-channel image counts as opaque; an image without 3 or 4 channels (grayscale) gives 0."""
    assert TOP > 0 and LEFT > 0 and BOTTOM < H and RIGHT < W  # precondition: ring fits
    assert rows.logo_pixels(_image(mode, tmp_path / f"{mode}.png")) == expected


# --- R8 / R9: theme variables and logo --------------------------------------------------------


def _zensical_css() -> str:
    folder = Path(zensical.__file__).parent / "templates" / "assets" / "stylesheets" / "modern"
    return " ".join(p.read_text(encoding="utf-8") for p in sorted(folder.glob("*.css")))


def test_theme_variables() -> None:
    """R8: every ``VARS`` name is one Zensical's modern stylesheets read and none is a site-own
    ``--fd-`` name; ``OWN_VARS`` are ``--fd-`` names Zensical never reads, declared after
    ``VARS`` in each scheme block; ``value`` is public, appends an alpha suffix and refuses a
    field that is not a colour."""
    zcss = _zensical_css()
    assert zcss  # precondition: the stylesheets were found
    for name, _ in theme.VARS:
        assert f"var({name})" in zcss or f"var({name}," in zcss, name
    own = [name for name, _ in theme.OWN_VARS]
    assert own and all(n.startswith("--fd-") and n not in zcss for n in own)
    assert not {n for n, _ in theme.VARS} & set(own)
    css = theme.css()
    for scheme, name in theme.SCHEMES:
        block = css.split(f'[data-md-color-scheme="{scheme}"] {{\n', 1)[1].split("}", 1)[0]
        declared = [line.split(":")[0].strip() for line in block.splitlines()]
        assert declared == [n for n, _ in theme.VARS + theme.OWN_VARS], scheme
        assert f"--fd-title: {viz.THEMES[name].title};" in block
    light = viz.THEMES["light"]
    assert theme.value(light, "accent+1a") == light.accent + "1a"
    assert theme.value(light, "title") == light.title
    for bad in ("family", "mark_alpha", "nonexistent"):
        with pytest.raises(ValueError, match="not a colour field"):
            theme.value(light, bad)


def test_logo_geometry() -> None:
    """R9: a wide goal (posts at x 4.5 and 19.5 from the ground line at y 18.5 up to the bar at
    y 7.5) on a ground line across the full 24 px, stroke 3 in the light accent, no fill."""
    svg = theme.logo_svg()
    assert 'viewBox="0 0 24 24"' in svg
    m = re.search(r'd="M([\d.]+) ([\d.]+)V([\d.]+)h([\d.]+)v([\d.]+)M([\d.]+) ([\d.]+)h([\d.]+)"',
                  svg)  # fmt: skip
    assert m is not None
    x0, y0, bar, width, down, gx, gy, glen = (float(v) for v in m.groups())
    assert (x0, x0 + width, bar, y0) == (4.5, 19.5, 7.5, 18.5)
    assert bar + down == y0  # the right post comes back down to the ground line
    assert (gx, gy, gx + glen) == (0.0, y0, 24.0)
    assert f'stroke="{viz.THEMES["light"].accent}"' in svg and 'stroke="#e4002b"' in svg
    assert 'stroke-width="3"' in svg and 'fill="none"' in svg


# --- R10 / R11: the results target and labels -------------------------------------------------


def test_docs_results_target() -> None:
    """R10: ``make docs-results`` rewrites the home block through the pages CLI (``--write``),
    is phony, and the pyproject comment over the picks names it."""
    makefile = (REPO / "Makefile").read_text()
    m = re.search(r"^docs-results:[^\n]*\n((?:\t[^\n]*\n)+)", makefile, re.M)
    assert m is not None
    assert m[1].strip() == "uv run python -m football_docs.pages results docs/index.md --write"
    phony = " ".join(re.findall(r"^\.PHONY:((?:[^\n]*\\\n)*[^\n]*)", makefile, re.M))
    assert "docs-results" in phony.split()
    assert "`make docs-results`" in (REPO / "pyproject.toml").read_text()


SB, WS = "StatsBomb open data", "private WhoScored / Opta archive"
KEEPER = "next-season log-loss gain of the shrunken keeper estimates"
LABELS = (
    f"xG, monotone GBM + Platt trained without team-selected seasons, {SB}: log loss",
    "StatsBomb's own xG on the same shots: log loss",
    f"xG v1, monotone GBM + Platt, {WS}: log loss",
    f"xGOT, monotone GBM + Platt, {SB}: AUC",
    f"xGOT, monotone GBM + Platt, {WS}: AUC",
    f"Keeper model, {SB}: {KEEPER}",
    f"Keeper model, {WS}: {KEEPER}",
)


def test_result_labels() -> None:
    """R11: the seven home-page result labels, word for word, in pick order."""
    got = pages.picks((REPO / "pyproject.toml").read_text())
    assert tuple(p.result for p in got) == LABELS


# --- A2: compact tables site-wide, card layout on card pages at desktop width -----------------


def test_table_rules_site_wide_card_rules_scoped() -> None:
    """A2: the compact table rules apply on every page (top level, no ``:has`` scope); the
    hidden right-hand contents and the 72rem grid apply only on model-card pages and only
    inside the ``min-width: 60em`` media query."""
    css = theme.css()
    media = re.search(r"@media screen and \(min-width: 60em\) \{\n(.*?)\n\}\n", css, re.S)
    assert media is not None
    outside = css.replace(media.group(0), "")
    card = 'h1[id^="model-card"]'
    assert card not in outside and "@media" not in outside
    inner = [line.strip() for line in media[1].splitlines()]
    assert inner == [
        f".md-main:has({card}) .md-sidebar--secondary {{ display: none; }}",
        f".md-main__inner:has({card}) {{ max-width: 72rem; }}",
    ]
    tables = [line for line in outside.splitlines() if "table" in line and "{" in line]
    assert len(tables) == 4  # precondition: font, header, cells, tabular figures
    assert all(line.startswith(".md-typeset table") and ":has(" not in line for line in tables)
    assert ".md-typeset table:not([class]) { font-size: .6rem;" in outside
