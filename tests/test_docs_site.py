"""The public docs site of football-shot-models: no internal process code on any page, README or
unhashed docstring; the home page's statement, sourced results block and install steps; the nav;
the theme and handler configuration; relative links (docs-sites spec §2.2, §2.3,
§4.1-§4.3, §6).

Every test reads committed files only (no ``data/``, no ``outputs/``, no built ``site/``)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from football_docs import meta, pages, theme

from football_shot_models import cache

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
README = REPO / "README.md"
INDEX = DOCS / "index.md"
CHARTS_PAGE = DOCS / "charts.md"
CHARTS_JSON = DOCS / "assets" / "charts" / "charts.json"

# The README's two card blocks keep their codes this round (the cards stay byte-identical);
# the pattern list itself, with the brief's example codes, is pinned in football-core's tests
CARD_BLOCKS = ("xgot", "v2")
CARD_BLOCK_CODES = 9
# The only generated blocks the scan exempts: the README's card blocks and the home results
GENERATED = {
    "README.md": ["models:summary", "v1:summary", "v2:summary", "xgot:summary", "gsax:summary"],
    "docs/index.md": ["results"],
}


def _front_matter(text: str) -> tuple[dict[str, Any], str]:
    """``(front matter, body)``; ``({}, text)`` when the page has none."""
    if not text.startswith("---\n"):
        return {}, text
    head, _, body = text[4:].partition("\n---\n")
    return yaml.safe_load(head) or {}, body


def _flat(text: str) -> str:
    """Lower case, whitespace collapsed, comment and quote marks dropped: a phrase wrapped over
    lines (a docstring or a ``#`` comment) still reads as one phrase."""
    lines = (line.strip().lstrip("#").strip() for line in text.splitlines())
    return " ".join(" ".join(lines).split()).lower()


def _scanned_pages() -> list[Path]:
    """Committed READMEs and docs pages in the clean scope: cards (``docs/models``) are fixed
    this round and ``docs/notebooks`` is generated, never committed."""
    readmes = [README, *sorted(REPO.glob("*/README*.md"))]
    readmes = [
        p
        for p in readmes
        if p.is_file() and not p.parts[-2].startswith(".") and p.parts[-2] != "site"
    ]
    docs = [
        p
        for p in sorted(DOCS.rglob("*.md"))
        if not p.is_relative_to(DOCS / "models") and not p.is_relative_to(DOCS / "notebooks")
    ]
    return readmes + docs


def _unhashed_sources() -> list[Path]:
    """The render-only and orchestration-only files (read from ``cache``, so core ``viz.py``
    comes too) and every module of ``football_shot_charts``."""
    roots = {r.name: r for r in cache.ROOTS}
    files = []
    for name in sorted(cache.RENDER_ONLY | cache.ORCHESTRATION_ONLY):
        root, _, rest = name.partition("/")
        files.append(roots[root] / rest)
    files += sorted((REPO / "src" / "football_shot_charts").glob("*.py"))
    return files


# --- internal codes ---------------------------------------------------------------------------


def test_pages_and_unhashed_docstrings_clean() -> None:
    """No ``CODES`` hit and no ``STALE`` phrase on any README or docs page outside the cards and
    the generated blocks, nor in any unhashed docstring (doctest examples included: they show on
    the API pages); the scan reads what it claims to (the new pages, ``viz.py``, the chart
    package); the exempt blocks are exactly the README's card blocks and the home results
    block, and the README's codes sit only in its two card blocks (9 codes)."""
    scanned = _scanned_pages()
    assert {README, INDEX, CHARTS_PAGE, DOCS / "roadmap.md"} <= set(scanned)
    blocks = {
        str(p.relative_to(REPO)): [name for name, _ in pages.generated_blocks(p.read_text())]
        for p in scanned
    }
    assert {page: names for page, names in blocks.items() if names} == GENERATED
    hits = [
        (str(p.relative_to(REPO)), line, code)
        for p in scanned
        for line, code, _ in pages.find_codes(pages.strip_generated(p.read_text()))
    ]
    assert hits == []
    # the home page's generated results block carries no code either (it copies cells only)
    assert pages.find_codes(INDEX.read_text()) == []
    sources = _unhashed_sources()
    names = {p.name for p in sources}
    assert {"viz.py", "gsax.py", "level.py", "gsax_cards.py", "render.py", "rows.py"} <= names
    doc_hits = [
        (str(p), line, code)
        for p in sources
        for line, code, _ in pages.docstring_codes(p, doctests=True)
    ]
    assert doc_hits == []
    for path in [*scanned, *sources]:
        flat = _flat(path.read_text())
        for phrase in pages.STALE:
            assert phrase.lower() not in flat, (path, phrase)
    text = README.read_text()
    inner = dict(pages.generated_blocks(text))
    in_blocks = [pages.find_codes(inner[f"{b}:summary"]) for b in CARD_BLOCKS]
    assert sum(len(h) for h in in_blocks) == CARD_BLOCK_CODES
    assert len(pages.find_codes(text)) == CARD_BLOCK_CODES  # none outside the two blocks


# --- home page --------------------------------------------------------------------------------

PICKS = (
    ("models", ("Monotone GBM + Platt, trained without team-selected seasons",),
     ("CV log loss", "Hold-out log loss"), ()),
    ("models", ("StatsBomb xG",), ("CV log loss", "Hold-out log loss"), ()),
    ("v1", ("Monotone GBM + Platt",), ("CV log loss", "Test log loss"), ()),
    ("xgot", ("StatsBomb open data", "Monotone GBM + Platt"), ("CV AUC", "Hold-out / test AUC"),
     ()),
    ("xgot", ("private WhoScored / Opta event archive", "Monotone GBM + Platt"),
     ("CV AUC", "Hold-out / test AUC"), ()),
    ("gsax", ("StatsBomb open data",), ("Next season: gain over raw", "Gain over xGOT alone"),
     ("StatsBomb open data: Shrunken", "Without team-selected")),
    ("gsax", ("private WhoScored / Opta event archive",),
     ("Next season: gain over raw", "Gain over xGOT alone"),
     ("private WhoScored / Opta event archive: Shrunken",)),
)  # fmt: skip
# Known cells of the committed README card blocks (cards are byte-identical this round)
KNOWN = (
    "CV log loss: 0.2641<br>Hold-out log loss: 0.2533",
    "CV log loss: 0.2659<br>Hold-out log loss: 0.2520",
    "CV log loss: 0.2724<br>Test log loss: 0.2771",
    "CV AUC: 0.8672<br>Hold-out / test AUC: 0.8685",
    "CV AUC: 0.8435<br>Hold-out / test AUC: 0.8475",
    "Next season: gain over raw: +0.03759 [+0.02901, +0.04714]<br>"
    "Gain over xGOT alone: +0.00097 [-0.00018, +0.00214]",
    "Next season: gain over raw: +0.01034 [+0.00502, +0.01619]<br>"
    "Gain over xGOT alone: +0.00003 [-0.00010, +0.00017]",
    "Without team-selected competition-seasons (22 pairs) the gain over raw is +0.01194 "
    "[-0.00019, +0.02517], its interval not above 0, and over xGOT alone +0.00050 "
    "[-0.00047, +0.00148], its interval not above 0.",
)


def test_home_picks() -> None:
    """``[[tool.football_docs.results]]`` holds the spec's seven picks in order (§4.2.3); the
    keeper labels name the measure."""
    got = pages.picks((REPO / "pyproject.toml").read_text())
    assert [(p.block, tuple(p.row), tuple(p.columns), tuple(p.notes)) for p in got] == list(PICKS)
    for p in got[5:]:
        assert p.result.startswith("Keeper model, ")
        assert p.result.endswith("next-season log-loss gain of the shrunken keeper estimates")
    assert len({p.result for p in got}) == len(got)


def test_home_results_block_current(monkeypatch: pytest.MonkeyPatch) -> None:
    """The home page's ``results`` block equals ``results_block(README, picks)`` (the CLI's
    check exits 0 too), and holds the README's own cells and qualifier sentence verbatim."""
    readme = README.read_text()
    page = INDEX.read_text()
    expected = pages.results_block(readme, pages.picks((REPO / "pyproject.toml").read_text()))
    found = re.findall(r"<!-- results:start -->.*?<!-- results:end -->\n?", page, re.S)
    assert len(found) == 1
    assert found[0].rstrip("\n") == expected.rstrip("\n")
    for cell in KNOWN:
        assert all(n in readme for n in re.findall(r"\d+\.\d+", cell))  # precondition
        assert cell in expected, cell
    monkeypatch.chdir(REPO)
    assert pages.main(["results", "docs/index.md"]) == 0


@pytest.mark.parametrize("page", ["index.md", "charts.md"])
def test_numbers_sourced(page: str) -> None:
    """Every number on the home and Charts pages outside the generated block (alt text
    included) occurs in the README or the published ``charts.json``."""
    sources = (README.read_text(), CHARTS_JSON.read_text())
    assert pages.unsourced_numbers((DOCS / page).read_text(), *sources) == []


def test_published_size_stated() -> None:
    """The README and the Charts page state the published charts' count and total size, read
    from the committed PNGs (MB = 10^6 bytes, two decimals), so a re-publish that changes them
    fails here rather than leaving both pages stale."""
    pngs = sorted(CHARTS_JSON.parent.glob("*.png"))
    size = f"{sum(p.stat().st_size for p in pngs) / 1e6:.2f} MB"
    assert len(pngs) >= 2  # precondition: the published set is committed
    readme = " ".join(README.read_text().split())
    charts = " ".join(CHARTS_PAGE.read_text().split())
    assert f"copy the {len(pngs)} published charts ({size})" in readme
    assert f"The {len(pngs)} published PNGs" in charts and f"together they are {size}" in charts


def test_home_statement() -> None:
    """Home: front matter hides nav and toc; H1 is the repo name; the statement names both data
    sources (WhoScored private); the chart grid's caption line; install steps; about 70 lines."""
    text = INDEX.read_text()
    meta, body = _front_matter(text)
    assert {"navigation", "toc"} <= set(meta.get("hide", []))
    first = next(line for line in body.splitlines() if line.strip())
    assert first == "# football-shot-models"
    flat = " ".join(body.split())
    assert "StatsBomb open data" in flat and "WhoScored" in flat and "private" in flat
    assert "Data: StatsBomb open data; charts drawn from cached predictions" in flat
    assert "uv sync --all-groups" in flat and "make check" in flat
    assert len(text.splitlines()) <= 80


def _make_targets() -> set[str]:
    text = (REPO / "Makefile").read_text()
    return set(re.findall(r"^([A-Za-z0-9][\w.-]*):(?!=)", text, re.M))


def _recipe(target: str) -> str:
    text = (REPO / "Makefile").read_text()
    m = re.search(rf"^{re.escape(target)}:[^\n]*\n((?:\t[^\n]*\n)+)", text, re.M)
    assert m, target
    return m.group(1)


def test_make_targets_on_home_exist() -> None:
    """Every ``make <target>`` on the home page is a Makefile target; the public chain is
    there; ``docs`` runs the raw-role site check after the strict build and checks the results
    block and theme are current (Pages deploys from it); ``docs-charts``
    publishes through the one ``CHARTS`` module path; ``docs-theme`` regenerates the theme."""
    targets = _make_targets()
    named = set(re.findall(r"\bmake ([a-z][\w-]*)", INDEX.read_text()))
    assert named - targets == set()
    assert {"check", "snapshot", "statsbomb-logo", "reproduce-public", "charts",
            "docs-charts"} <= named  # fmt: skip
    docs = _recipe("docs")
    assert "zensical build --strict" in docs
    assert docs.index("zensical build --strict") < docs.index("football_docs.pages site site")
    # the Pages workflow runs only ``make docs``: it must refuse a stale results block or theme
    assert "football_docs.pages results docs/index.md\n" in docs
    assert "football_docs.theme docs --check\n" in docs
    assert "--publish docs/assets/charts" in _recipe("docs-charts")
    makefile = (REPO / "Makefile").read_text()
    assert re.search(r"^CHARTS :?= uv run python -m football_shot_charts\.render$", makefile, re.M)
    assert makefile.count("football_shot_charts.render") == 1
    assert "football_docs.theme docs" in _recipe("docs-theme")


# --- site configuration -----------------------------------------------------------------------


class _Loader(yaml.SafeLoader):
    """Safe loading that tolerates ``!!python/name:`` style tags (read as ``None``)."""


_Loader.add_multi_constructor("", lambda loader, suffix, node: None)


def _config() -> dict[str, Any]:
    return dict(yaml.load((REPO / "mkdocs.yml").read_text(), Loader=_Loader))


def _nav(items: list[Any]) -> list[tuple[str, Any]]:
    out = []
    for item in items:
        ((title, value),) = item.items()
        out.append((title, _nav(value) if isinstance(value, list) else value))
    return out


CARDS = (
    "xg-base-rate", "xg-lr-distance-angle", "xg-lr-distance-angle-body-part", "xg-lr-splines",
    "xg-lr-context", "xg-lr-frames", "xg-gbm", "xg-penalty", "xg-v1", "xg-v2", "xgot-lr",
    "xgot-gbm", "xgot-penalty", "ws-xgot-lr", "ws-xgot-gbm", "ws-xgot-penalty", "gsax",
    "gk-model", "ws-gsax", "ws-gk-model",
)  # fmt: skip
API = (
    "data", "features", "metrics", "evaluation", "cards", "baselines", "ablation", "models",
    "level", "gbm", "penalties", "cache", "legacy", "v1", "release", "v2", "v2_cards", "xgot",
    "xgot_cards", "season_out", "gsax", "gsax_cards", "charts.shots", "charts.render",
    "charts.rows", "charts.keepers", "charts.diagnostics",
)  # fmt: skip


def test_nav_order() -> None:
    """Home, Charts, Data, Methodology, Model cards (the 20 cards in today's order, the template
    last), Modules, Roadmap, References, API reference (today's 27 pages in order)."""
    nav = _nav(_config()["nav"])
    assert [t for t, _ in nav] == ["Home", "Charts", "Data", "Methodology", "Model cards",
                                   "Modules", "Roadmap", "References", "API reference"]  # fmt: skip
    pages_ = dict(nav)
    assert pages_["Home"] == "index.md" and pages_["Charts"] == "charts.md"
    cards = [f for _, f in pages_["Model cards"]]
    assert cards == [f"models/{c}.md" for c in CARDS] + ["model-card-template.md"]
    api = pages_["API reference"]
    assert [t for t, _ in api] == list(API)
    assert [f for _, f in api] == [f"api/{a.replace('.', '_')}.md" for a in API]


def _handler_options(cfg: dict[str, Any]) -> dict[str, Any]:
    for plugin in cfg["plugins"]:
        if isinstance(plugin, dict) and "mkdocstrings" in plugin:
            return dict(plugin["mkdocstrings"]["handlers"]["python"]["options"])
    raise AssertionError("no mkdocstrings plugin")


def _names(entries: list[Any]) -> set[str]:
    """Names of a YAML list whose entries are a bare name or a one-key mapping."""
    return {e if isinstance(e, str) else next(iter(e)) for e in entries}


def test_site_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """mkdocs.yml: author with the s-cedilla, copyright, logo and favicon, Inter, custom palette
    in the automatic entry and both toggled schemes, the generated theme.css, attr_list and
    md_in_html, the roles extension and ``show_source: false``; the committed theme.css and
    logo.svg are current; the home grid class is used on the home page and defined in the
    theme."""
    cfg = _config()
    assert cfg["site_author"] == meta.AUTHOR
    assert cfg["copyright"] == f"© 2026 {meta.AUTHOR} · MIT licence"
    th = cfg["theme"]
    assert th["logo"] == "assets/logo.svg" and th["favicon"] == "assets/logo.svg"
    assert th["font"] == {"text": "Inter", "code": "JetBrains Mono"}
    # automatic entry first (follows the device on every visit), then the two toggled schemes
    schemes = [(p["media"], p.get("scheme")) for p in th["palette"]]
    assert schemes == [
        ("(prefers-color-scheme)", None),
        ("(prefers-color-scheme: light)", "default"),
        ("(prefers-color-scheme: dark)", "slate"),
    ]
    for p in th["palette"]:
        assert p["primary"] == "custom" and p["accent"] == "custom" and "toggle" in p
    assert "stylesheets/theme.css" in cfg["extra_css"]
    assert any("katex" in c for c in cfg["extra_css"])
    assert {"attr_list", "md_in_html"} <= _names(cfg["markdown_extensions"])
    options = _handler_options(cfg)
    assert "football_docs.roles" in _names(options["extensions"])
    assert options["show_source"] is False
    assert (DOCS / "stylesheets" / "theme.css").read_text() == theme.css()
    assert (DOCS / "assets" / "logo.svg").read_text() == theme.logo_svg()
    monkeypatch.chdir(REPO)
    assert theme.main(["docs", "--check"]) == 0
    assert 'class="fd-home-grid"' in INDEX.read_text()
    assert ".fd-home-grid" in theme.css()


def test_card_tables_fit() -> None:
    """Result audit A2: wide card tables are compact and say they scroll, and card pages drop
    the right-hand contents for a wider column, by CSS alone (the cards stay byte-identical).

    Table headers stay on one line (no blank band under a wrapped header), the table font is
    below Zensical's .64rem with compact cells, a table that scrolls shows an edge shadow
    (covers attached ``local``, shadows ``scroll``), and the card-page rule is keyed on an
    ``h1`` id every card page has: each card's first line is ``# Model card: ...``."""
    from markdown.extensions.toc import slugify

    rules = " ".join(theme.css().split())
    table = re.search(r"\.md-typeset table:not\(\[class\]\) \{ font-size: \.(\d+)rem;", rules)
    assert table is not None and float(f"0.{table[1]}") < 0.64
    assert ".md-typeset table:not([class]) th { white-space: nowrap; min-width: 0; }" in rules
    assert "td { padding: .45em .8em; }" in rules
    wrap = rules.split(".md-typeset__scrollwrap {", 1)[1].split("}", 1)[0]
    assert wrap.count("no-repeat local") == 2 and wrap.count("no-repeat scroll") == 2
    assert "radial-gradient" not in wrap  # a radial shadow fades out along a tall table
    card = 'h1[id^="model-card"]'
    assert f".md-main:has({card}) .md-sidebar--secondary {{ display: none; }}" in rules
    assert f".md-main__inner:has({card}) {{ max-width: 72rem; }}" in rules
    cards = sorted((DOCS / "models").glob("*.md"))
    assert len(cards) == 20
    for path in cards:
        first = path.read_text(encoding="utf-8").splitlines()[0]
        assert first.startswith("# Model card: "), path.name
        assert slugify(first[2:], "-").startswith("model-card"), path.name
    other = [p for p in sorted(DOCS.glob("*.md")) if p.name != "model-card-template.md"]
    assert not any(p.read_text().startswith("# Model card") for p in other)


def test_links_relative() -> None:
    """No docs page links or points an image at the site root (a GitHub Pages project site is
    served under ``/<repo>/``)."""
    bad = [
        (str(p.relative_to(REPO)), token)
        for p in sorted(DOCS.rglob("*.md"))
        for token in ("](/", 'src="/', 'href="/')
        if token in p.read_text()
    ]
    assert bad == []


def test_docstring_scan_reads_docstrings_only(tmp_path: Path) -> None:
    """Near misses for the docstring scan the clean test relies on: a code in a docstring is
    found at its file line, one in a doctest example is skipped (and found with
    ``doctests=True``), one in a plain string or comment is not a docstring."""
    src = tmp_path / "mod.py"
    src.write_text(
        '"""Module.\n\nDecided in MS6.\n"""\n\n'
        "X = 'run8'  # PM E1\n\n\n"
        "def f() -> int:\n"
        '    """Do.\n\n    >>> "R14"\n    \'R14\'\n    """\n'
        "    return 1\n"
    )
    ast.parse(src.read_text())  # precondition: valid Python
    assert [(line, code) for line, code, _ in pages.docstring_codes(src)] == [(3, "MS6")]
    with_doctests = {code for _, code, _ in pages.docstring_codes(src, doctests=True)}
    assert {"MS6", "R14"} <= with_doctests and "run8" not in with_doctests
