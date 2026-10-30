"""The published StatsBomb example charts: the committed set, the logo check and its near misses,
no WhoScored chart under ``docs/``, the keeper charts against the card digests, alt text from the
chart titles (read from rendered HTML), the publish step and the titles file (docs-sites
spec §3, §4.2, §4.4, §6).

Synthetic PNGs are written by numpy and Pillow into ``tmp_path``; committed files are read from
``docs/`` only (never ``outputs/`` or ``data/``)."""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import re
import subprocess
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import charts_cases as cc
import markdown
import numpy as np
import pytest
from football_core import viz
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from football_shot_charts import render, rows

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
PUBLISHED = DOCS / "assets" / "charts"
THEMES = ("light", "dark")
# (pattern with {theme}, kind): spec §3, written out here so a change to PUBLISH is seen
PUBLISH = (
    ("team-*-{theme}-square.png", "Team shot map"),
    ("player-*-{theme}-square.png", "Player shot map"),
    ("gsax-keepers-{theme}.png", "Keeper GSAx interval chart"),
    ("calibration-xg-{theme}.png", "xG calibration chart"),
    ("calibration-xgot-{theme}.png", "xGOT calibration chart"),
    ("comparison-xg-{theme}.png", "xG model comparison chart"),
    ("comparison-xgot-{theme}.png", "xGOT model comparison chart"),
)
SELECTED = [(p, t) for p, _ in PUBLISH for t in THEMES]
LOGO = (226, 12, 22)
H, W = 300, 400  # synthetic chart size: larger than the footer logo box


# --- synthetic PNGs ---------------------------------------------------------------------------


def _logo_rect(h: int, w: int) -> tuple[slice, slice]:
    """20 x 150 px at the bottom of the footer logo (right-aligned at ``MARGIN_PX``, its
    bottom at ``BOTTOM_PX``): inside any box that holds the logo."""
    x1 = w - viz.MARGIN_PX
    return (slice(h - viz.BOTTOM_PX - 20, h - viz.BOTTOM_PX),
            slice(x1 - viz.LOGO_PX + 10, x1 - 10))  # fmt: skip


FILLED = 20 * 150


def _png(
    path: Path,
    *,
    rgb: tuple[int, int, int] | None = LOGO,
    alpha: int = 255,
    where: str = "logo",
    size: tuple[int, int] = (H, W),
    tag: int = 0,
) -> Path:
    """An RGBA PNG on the light background with ``rgb`` at ``alpha`` in the logo rectangle
    (``where="logo"``), in the top band (``"top"``) or over the whole image (``"all"``);
    ``tag`` varies one corner pixel so files differ byte for byte."""
    h, w = size
    a = np.zeros((h, w, 4), dtype=np.uint8)
    a[..., :3] = (248, 250, 252)
    a[..., 3] = 255
    a[0, 0, :3] = (tag % 256, tag // 256, 7)
    if rgb is not None:
        if where == "all":
            region: tuple[slice, slice] = (slice(None), slice(None))
        elif where == "top":
            region = (slice(0, 20), _logo_rect(h, w)[1])
        else:
            region = _logo_rect(h, w)
        a[region[0], region[1], :3] = rgb
        a[region[0], region[1], 3] = alpha
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a, "RGBA").save(path)
    return path


# --- the committed set ------------------------------------------------------------------------


def _committed() -> dict[str, Any]:
    return dict(json.loads((PUBLISHED / "charts.json").read_text()))


def _one(pattern: str, theme: str, names: list[str]) -> str:
    found = fnmatch.filter(names, pattern.format(theme=theme))
    assert len(found) == 1, (pattern, theme, found)
    return found[0]


def test_committed_chart_set() -> None:
    """``docs/assets/charts`` holds exactly the 14 selected PNGs and ``charts.json``, whose keys
    are those PNGs (title and subtitle, one line each), written sorted, indent 1, UTF-8 and a
    trailing newline; ``render.PUBLISH`` is the spec's table; the team and player twins name one
    team and one player."""
    assert tuple(render.PUBLISH) == PUBLISH
    files = sorted(p.name for p in PUBLISHED.iterdir())
    pngs = [f for f in files if f.endswith(".png")]
    assert files == sorted([*pngs, "charts.json"])
    assert sorted(_one(p, t, pngs) for p, t in SELECTED) == pngs and len(pngs) == 14
    entries = _committed()
    assert sorted(entries) == pngs
    for entry in entries.values():
        assert set(entry) == {"title", "subtitle"}
        assert all(isinstance(v, str) and v and "\n" not in v for v in entry.values())
    text = (PUBLISHED / "charts.json").read_text(encoding="utf-8")
    assert text == json.dumps(entries, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    for prefix in ("team-", "player-"):
        light, dark = (_one(f"{prefix}*-{{theme}}-square.png", t, pngs) for t in THEMES)
        assert light.replace("-light-", "-dark-") == dark


@pytest.mark.parametrize(("pattern", "theme"), SELECTED, ids=[f"{p}:{t}" for p, t in SELECTED])
def test_committed_charts_carry_logo(pattern: str, theme: str) -> None:
    """Each committed PNG carries the StatsBomb logo in its footer box (1,512 px measured)."""
    name = _one(pattern, theme, [p.name for p in PUBLISHED.glob("*.png")])
    assert rows.logo_pixels(PUBLISHED / name) >= rows.LOGO_MIN_PX


NEAR = {
    # case: (rgb, alpha, where, size, logo pixels expected)
    "logo": (LOGO, 255, "logo", (H, W), FILLED),
    "within-12": ((238, 0, 34), 255, "logo", (H, W), FILLED),  # every channel 12 off
    "alpha-253": (LOGO, 253, "logo", (H, W), FILLED),  # 0.992 >= 0.99
    "blanked": (None, 255, "logo", (H, W), 0),
    "alpha-0": (LOGO, 0, "logo", (H, W), 0),
    "alpha-252": (LOGO, 252, "logo", (H, W), 0),  # 0.988 < 0.99
    "accent-e4002b": ((0xE4, 0x00, 0x2B), 255, "logo", (H, W), 0),  # 21 off in blue
    "off-13": ((239, 12, 22), 255, "logo", (H, W), 0),
    "outside-box": (LOGO, 255, "top", (H, W), 0),
    "smaller-than-box": (LOGO, 255, "all", (100, 100), 0),
}


@pytest.mark.parametrize("case", list(NEAR))
def test_logo_check_near_misses(case: str, tmp_path: Path) -> None:
    """``logo_pixels`` counts pixels within ``LOGO_TOL`` (inclusive) of ``LOGO_RGB`` in every
    channel at alpha >= 0.99, inside the footer logo box only; an image smaller than the box
    is 0. The real logo passes, every near miss falls below ``LOGO_MIN_PX``."""
    assert (rows.LOGO_RGB, rows.LOGO_TOL, rows.LOGO_MIN_PX) == (LOGO, 12, 1000)
    rgb, alpha, where, size, expected = NEAR[case]
    path = _png(tmp_path / f"{case}.png", rgb=rgb, alpha=alpha, where=where, size=size)
    got = rows.logo_pixels(path)
    assert got == expected
    assert (got >= rows.LOGO_MIN_PX) == (expected > 0)


# The square box ``logo_pixels`` reads: LOGO_PX + pad up from BOTTOM_PX, left from MARGIN_PX
BOX_TOP = H - viz.BOTTOM_PX - viz.LOGO_PX - viz.LOGO_PLATE_PAD_PX
BOX_LEFT = W - viz.MARGIN_PX - viz.LOGO_PX - viz.LOGO_PLATE_PAD_PX
ABOVE = st.tuples(st.integers(0, BOX_TOP - 11), st.integers(0, W - 1), st.integers(1, 10),
                  st.integers(1, 60))  # fmt: skip
LEFT = st.tuples(st.integers(0, H - 1), st.integers(0, BOX_LEFT - 61), st.integers(1, 40),
                 st.integers(1, 60))  # fmt: skip


@settings(max_examples=25, deadline=None)
@given(st.lists(st.one_of(ABOVE, LEFT), max_size=6))
def test_logo_count_ignores_red_outside_the_footer(rects: list[tuple[int, int, int, int]]) -> None:
    """Logo red painted anywhere above or left of the footer logo box never changes the
    count."""
    assert BOX_TOP > 20 and BOX_LEFT > 70  # precondition: room outside the box
    with tempfile.TemporaryDirectory() as d:
        path = _png(Path(d) / "c.png")
        base = rows.logo_pixels(path)
        a = np.asarray(Image.open(path)).copy()
        for y, x, h, w in rects:
            a[y : y + h, x : x + w, :3] = LOGO
        Image.fromarray(a, "RGBA").save(path)
        assert base == FILLED
        assert rows.logo_pixels(path) == base


def test_no_whoscored_image_under_docs() -> None:
    """No file under ``docs/`` is a WhoScored chart: no ``ws-`` or ``whoscored`` name, every
    raster image is a published chart carrying the StatsBomb logo, and no published title or
    subtitle credits WhoScored or Opta."""
    raster = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    for p in DOCS.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in raster | {".svg"}:
            assert not p.name.lower().startswith("ws-") and "whoscored" not in p.name.lower(), p
        if p.suffix.lower() in raster:
            assert p.parent == PUBLISHED, p
            assert rows.logo_pixels(p) >= rows.LOGO_MIN_PX, p
    for entry in _committed().values():
        text = f"{entry['title']} {entry['subtitle']}".lower()
        assert "whoscored" not in text and "opta" not in text


def test_keeper_charts_match_card_digests() -> None:
    """The published keeper charts are the ones the GSAx card pins (the flagged chart, as
    Berk chose to publish it): SHA-256 of each equals the digest printed on
    ``docs/models/gsax.md``."""
    card = (DOCS / "models" / "gsax.md").read_text()
    digests = dict(re.findall(r"`(gsax-keepers-(?:light|dark)\.png)` \(SHA-256 `([0-9a-f]{64})`",
                              card))  # fmt: skip
    assert set(digests) == {"gsax-keepers-light.png", "gsax-keepers-dark.png"}
    for name, digest in digests.items():
        assert hashlib.sha256((PUBLISHED / name).read_bytes()).hexdigest() == digest, name
    assert not list(PUBLISHED.glob("gsax-keepers-no-team-*"))


# --- alt text ---------------------------------------------------------------------------------

EXTENSIONS = ["attr_list", "md_in_html", "tables", "admonition", "toc", "pymdownx.superfences"]


class _Images(HTMLParser):
    """``img`` attributes with the ``href`` of the enclosing link; heading ids."""

    def __init__(self) -> None:
        super().__init__()
        self.images: list[dict[str, str | None]] = []
        self.ids: set[str] = set()
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "a":
            self._href = a.get("href")
        elif tag == "img":
            self.images.append({**a, "_href": self._href})
        if re.fullmatch(r"h[1-6]", tag) and a.get("id"):
            self.ids.add(str(a["id"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._href = None


def _html(text: str) -> _Images:
    if text.startswith("---\n"):
        text = text[4:].partition("\n---\n")[2]
    parser = _Images()
    parser.feed(markdown.Markdown(extensions=EXTENSIONS).convert(text))
    return parser


def _chart_images(page: Path) -> list[tuple[str, str, str | None, str | None]]:
    """``(name, theme fragment, alt, href)`` of each chart image, in page order."""
    out = []
    for img in _html(page.read_text()).images:
        src = str(img.get("src") or "")
        if "assets/charts/" in src:
            name, _, frag = src.split("assets/charts/", 1)[1].partition("#")
            out.append((name, frag, img.get("alt"), img.get("_href")))
    return out


def test_alt_text_matches_titles() -> None:
    """On every page that shows a chart (home and Charts at least), each image's alt (read from
    Python-Markdown's HTML) is ``alt_text(name, charts.json[name])``; each light image is
    followed by its dark twin with the same alt and link; Charts images link to their own PNG
    and show all 14, home images (the team square and the keeper chart) link to a Charts
    section that exists."""
    entries = _committed()
    pages = [p for p in sorted(DOCS.rglob("*.md")) if "assets/charts/" in p.read_text()]
    assert {DOCS / "index.md", DOCS / "charts.md"} <= set(pages)
    charts_ids = _html((DOCS / "charts.md").read_text()).ids
    for page in pages:
        images = _chart_images(page)
        assert images and len(images) % 2 == 0, page
        for (name, frag, alt, href), (name2, frag2, alt2, href2) in zip(
            images[::2], images[1::2], strict=True
        ):
            assert frag == "only-light" and frag2 == "only-dark", (page, name, name2)
            assert "-light" in name and name2 == name.replace("-light", "-dark"), (page, name)
            assert alt == render.alt_text(name, entries[name]), (page, name)
            assert alt2 == alt and href2 is not None and href is not None
            if page.name == "charts.md":
                assert (href, href2) == (f"assets/charts/{name}", f"assets/charts/{name2}")
            else:
                anchor = href.partition("#")[2]
                assert href.startswith("charts.md#") and anchor in charts_ids, (page, href)
                assert href2 == href
        if page.name == "charts.md":
            assert sorted(n for n, *_ in images) == sorted(entries)
    home = {n for n, *_ in _chart_images(DOCS / "index.md")}
    names = list(entries)
    shown = ("team-*-{theme}-square.png", "gsax-keepers-{theme}.png")
    assert home == {_one(p, t, names) for p in shown for t in THEMES}


def _rendered_alt(md: str) -> list[str | None]:
    return [img.get("alt") for img in _html(md).images]


HOSTILE = (
    ("Unbalanced [bracket (x)", "Premier League 2015/2016"),
    ("Interval [-0.28, +0.40] *starred* and _under_", "sub `code`"),
    ("Back\\slash ] alone", "Karataş · 2015/2016 †"),
)


@pytest.mark.parametrize(("title", "subtitle"), HOSTILE)
def test_alt_text_escaping(title: str, subtitle: str) -> None:
    """``alt_text`` = ``kind: title (subtitle)``; ``image_markdown`` escapes the Markdown
    metacharacters so the rendered alt is ``alt_text`` exactly (unbalanced bracket, ``*``,
    ``_``, backtick, backslash), with the theme fragment, ``loading=lazy`` and the link."""
    name = "gsax-keepers-dark.png"
    entry = {"title": title, "subtitle": subtitle}
    assert render.alt_text(name, entry) == f"Keeper GSAx interval chart: {title} ({subtitle})"
    md = render.image_markdown(name, entry, "charts.md#keeper")
    (img,) = _html(md).images
    assert img.get("alt") == render.alt_text(name, entry)
    assert img.get("src") == "assets/charts/gsax-keepers-dark.png#only-dark"
    assert img.get("loading") == "lazy" and img.get("_href") == "charts.md#keeper"


WORD = st.text(alphabet=st.sampled_from(list("abXZ09[]*_`\\()-:,.+/%&'\"·ş−")),
               min_size=1, max_size=8)  # fmt: skip


@settings(max_examples=60, deadline=None)
@given(st.lists(WORD, min_size=1, max_size=6), st.lists(WORD, min_size=1, max_size=3))
def test_alt_text_round_trip(title: list[str], subtitle: list[str]) -> None:
    """Any title and subtitle (single-spaced words over the metacharacters) come back from
    Python-Markdown as ``alt_text`` exactly."""
    entry = {"title": " ".join(title), "subtitle": " ".join(subtitle)}
    name = "team-team-one-light-square.png"
    md = render.image_markdown(name, entry, "assets/charts/" + name)
    assert _rendered_alt(md) == [render.alt_text(name, entry)]


@settings(max_examples=60, deadline=None)
@given(st.from_regex(r"[a-z0-9]+(-[a-z0-9]+){0,3}", fullmatch=True), st.sampled_from(THEMES))
def test_kind_by_pattern(slug: str, theme: str) -> None:
    """Names a ``PUBLISH`` pattern matches get its kind; the private prefix, the portrait
    twins and the drop-team-selected keeper chart are outside ``PUBLISH``."""
    assert render.kind(f"team-{slug}-{theme}-square.png") == "Team shot map"
    assert render.kind(f"player-{slug}-{theme}-square.png") == "Player shot map"
    for bad in (f"ws-team-{slug}-{theme}-square.png", f"team-{slug}-{theme}-portrait.png",
                f"gsax-keepers-no-team-{theme}.png", f"ws-calibration-xg-{theme}.png"):  # fmt: skip
        with pytest.raises(ValueError):
            render.kind(bad)


# --- publish ----------------------------------------------------------------------------------

SOURCE = [
    *(f"team-team-one-{t}-{s}.png" for t in THEMES for s in ("square", "portrait")),
    *(f"player-player-7-{t}-{s}.png" for t in THEMES for s in ("square", "portrait")),
    *(f"{f}-{t}.png" for f in ("gsax-keepers", "calibration-xg", "calibration-xgot",
                               "comparison-xg", "comparison-xgot") for t in THEMES),
]  # fmt: skip
EXPECTED = sorted(n for n in SOURCE if "portrait" not in n)


def _source(
    root: Path, names: list[str] | None = None, *, json_names: list[str] | None = None
) -> tuple[Path, dict[str, dict[str, str]]]:
    """``make charts`` output: one synthetic logo PNG per name (distinct bytes) and
    ``charts.json`` for ``json_names`` (default: the same names)."""
    src = root / "outputs" / "charts"
    names = SOURCE if names is None else names
    for i, n in enumerate(names):
        _png(src / n, tag=i + 1)
    entries = {n: {"title": f"Title of {n} · Karataş", "subtitle": "League 2015/2016"}
               for n in (names if json_names is None else json_names)}  # fmt: skip
    (src / "charts.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False) + "\n")
    return src, entries


def _pngs(d: Path) -> list[str]:
    return sorted(p.name for p in d.glob("*.png")) if d.is_dir() else []


def test_publish_selects_and_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``publish`` copies exactly the 14 selected PNGs byte for byte (no portrait, no
    ``gsax-keepers-no-team``), writes their entries to ``dest/charts.json`` (sorted, indent 1,
    UTF-8, trailing newline), prints one ``image_markdown`` line per file linking to its PNG,
    leaves the source untouched and can be re-run."""
    src, entries = _source(tmp_path, [*SOURCE, "gsax-keepers-no-team-light.png",
                                      "gsax-keepers-no-team-dark.png"])  # fmt: skip
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    dest = tmp_path / "docs" / "assets" / "charts"
    got = render.publish(src, dest)
    assert sorted(Path(p).name for p in got) == EXPECTED
    assert {Path(p).resolve() for p in got} == {(dest / n).resolve() for n in EXPECTED}
    assert _pngs(dest) == EXPECTED
    for n in EXPECTED:
        assert (dest / n).read_bytes() == (src / n).read_bytes(), n
    selected = {n: entries[n] for n in EXPECTED}
    text = (dest / "charts.json").read_text(encoding="utf-8")
    assert text == json.dumps(selected, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[![")]
    expected_lines = {render.image_markdown(n, entries[n], f"assets/charts/{n}") for n in EXPECTED}
    assert len(lines) == 14 and set(lines) == expected_lines
    assert {p.name: p.read_bytes() for p in src.iterdir()} == before
    again = render.publish(src, dest)  # the selection already in dest is not "outside" it
    assert sorted(Path(p).name for p in again) == EXPECTED and _pngs(dest) == EXPECTED


def _without(names: list[str], *drop: str) -> list[str]:
    return [n for n in names if n not in drop]


REFUSE = {
    # case: (PNG files, charts.json names, PNG in dest beforehand, logo-less file, message part)
    "no-charts-json": (SOURCE, None, None, None, "make charts"),
    "ws-name": ([*SOURCE, "ws-calibration-xg-light.png"],
                [*SOURCE, "ws-calibration-xg-light.png"], None, None, "ws-calibration-xg-light"),
    "logo-less": (SOURCE, SOURCE, None, "comparison-xgot-dark.png", "comparison-xgot-dark.png"),
    "extra-in-dest": (SOURCE, SOURCE, "old-chart.png", None, "old-chart.png"),
    "two-team-squares": ([*SOURCE, "team-team-two-light-square.png"],
                         [*SOURCE, "team-team-two-light-square.png"], None, None,
                         "team-team-two-light-square.png"),
    "keeper-exact-name": (
        [*_without(SOURCE, "gsax-keepers-light.png"), "gsax-keepers-no-team-light.png"],
        [*_without(SOURCE, "gsax-keepers-light.png"), "gsax-keepers-no-team-light.png"],
        None, None, "gsax-keepers"),
    "selected-file-missing": (_without(SOURCE, "calibration-xg-dark.png"), SOURCE, None, None,
                              "calibration-xg-dark.png"),
}  # fmt: skip


@pytest.mark.parametrize("case", list(REFUSE))
def test_publish_refuses(case: str, tmp_path: Path) -> None:
    """Each refusal raises ``ValueError`` naming its cause and copies nothing: no
    ``charts.json`` (run make charts first), a ``ws-`` name, a selected PNG without the logo, a
    PNG in ``dest`` outside the selection (kept, never deleted), two team squares in one theme,
    the exact keeper name missing while the drop-team-selected one is there, a selected file
    missing."""
    files, json_names, extra, logo_less, message = REFUSE[case]
    src, _ = _source(tmp_path, list(files), json_names=None if json_names is None
                     else list(json_names))  # fmt: skip
    if json_names is None:
        (src / "charts.json").unlink()
    if logo_less:
        _png(src / logo_less, rgb=None, tag=999)
    dest = tmp_path / "docs" / "assets" / "charts"
    if extra:
        _png(dest / extra, tag=500)
    with pytest.raises(ValueError, match=re.escape(message)):  # the error names its cause
        render.publish(src, dest)
    assert _pngs(dest) == ([extra] if extra else [])
    assert not (dest / "charts.json").exists()


def _cli(argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            rc = render.main(argv)
        except SystemExit as e:  # argparse (--help)
            rc = int(e.code or 0)
    return rc, out.getvalue(), err.getvalue()


def test_publish_cli(tmp_path: Path) -> None:
    """``--publish <dest> --out <src>`` publishes; with ``--private`` it exits 1, no traceback,
    nothing copied; a refusal exits 1 the same way; the help text carries no internal code."""
    from football_docs import pages

    src, _ = _source(tmp_path)
    dest = tmp_path / "published"
    rc, _, err = _cli(["--publish", str(dest), "--out", str(src), "--private"])
    assert rc == 1 and "Traceback" not in err and _pngs(dest) == []
    (src / "charts.json").rename(src / "titles.json")
    rc, _, err = _cli(["--publish", str(dest), "--out", str(src)])
    assert rc == 1 and "Traceback" not in err and "make charts" in err and _pngs(dest) == []
    (src / "titles.json").rename(src / "charts.json")
    rc, out, err = _cli(["--publish", str(dest), "--out", str(src)])
    assert rc == 0, err
    assert _pngs(dest) == EXPECTED
    rc, out, _ = _cli(["--help"])
    assert rc == 0 and "--publish" in out
    assert pages.find_codes(out) == []


# --- the titles file (make charts) ------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway git repo (cwd) ignoring ``data/`` and ``outputs/``."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("data/\noutputs/\n")
    monkeypatch.chdir(root)
    return root


@pytest.fixture(scope="module")
def logo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return cc.write_logo(tmp_path_factory.mktemp("logo") / "logo.png")


def _canvas(logo: Path, provider: str = "statsbomb", *, drop: str | None = None) -> viz.Canvas:
    c = viz.canvas("light", "square", title="T\nU", subtitle="S",
                   footer=viz.Footer((provider,)), logo=logo)  # fmt: skip
    if drop:
        for t in c.fig.findobj(lambda a: a.get_gid() == drop):
            t.set_gid(f"not-{drop}")
    return c


def test_titles_json_from_canvas(
    repo: Path, logo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``write_charts`` writes ``charts.json`` from each canvas's title and subtitle texts (``\\n``
    read as a space; sorted, indent 1, UTF-8) after the charts and prints ``titles: <path>``; a
    canvas without the title or the subtitle text, or a ``charts.json`` git would track, writes
    nothing; the private run writes no ``charts.json``."""
    out = repo / "outputs" / "charts"
    render.write_charts({"team-x-light-square.png": _canvas(logo)}, out, private=False)
    entries = {"team-x-light-square.png": {"title": "T U", "subtitle": "S"}}
    text = (out / "charts.json").read_text(encoding="utf-8")
    assert text == json.dumps(entries, sort_keys=True, indent=1, ensure_ascii=False) + "\n"
    assert _pngs(out) == ["team-x-light-square.png"]
    assert any(line.startswith("titles: ") and line.endswith("charts.json")
               for line in capsys.readouterr().out.splitlines())  # fmt: skip
    for gid in ("title", "subtitle"):
        bad = repo / "outputs" / f"no-{gid}"
        with pytest.raises(ValueError):
            render.write_charts({"a.png": _canvas(logo, drop=gid)}, bad, private=False)
        assert not bad.exists() or not list(bad.iterdir())
    # PNGs ignored, charts.json not: the titles file's own git check refuses, nothing written
    (repo / ".gitignore").write_text("data/\nshown/*.png\n")
    shown = repo / "shown"
    with pytest.raises(ValueError):
        render.write_charts({"team-x-light-square.png": _canvas(logo)}, shown, private=False)
    assert not shown.exists() or not list(shown.iterdir())
    private = repo / "data" / "charts" / "private"
    render.write_charts({"ws-x.png": _canvas(logo, "whoscored")}, private, private=True)
    assert sorted(p.name for p in private.iterdir()) == ["ws-x.png"]
