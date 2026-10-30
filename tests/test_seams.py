"""Seams for the polish round (spec ``runs/MS6/gsax`` §10): the git dependency switch; the
season-out move out of xgot.py (``runs/PR1/keeper-fixes`` §2, §9).

After the switch, football-core is installed from git, not an editable sibling checkout, so no
``src`` module may locate data or code through core's install path or a sibling path.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "football_shot_models"


def test_no_sibling_paths() -> None:
    """No ``src`` module contains ``football_core.__file__`` or ``../football-core`` (the
    detector is checked on hand-written text first)."""
    banned = ("football_core.__file__", "../football-core")
    assert any(b in "root = Path(football_core.__file__).parent" for b in banned)
    assert any(b in 'RAW = "../football-core/data/raw"' for b in banned)
    charts = SRC.parent / "football_shot_charts"  # CH1.style-shots §9: the charts package too
    files = sorted(SRC.rglob("*.py")) + sorted(charts.rglob("*.py"))
    assert files and any(p.parent == charts for p in files)  # precondition: both packages
    for path in files:
        text = path.read_text(encoding="utf-8")
        found = [b for b in banned if b in text]
        assert found == [], (path.name, found)


# ---- PR1 keeper-fixes §2, §9: the season-out block leaves xgot.py ---------------------------

XGOT_CAP = 720  # code lines (PM decision 2026-10-28: "xgot.py then sits under its cap")
MOVED = ("SEASON_KEY", "SOURCE", "SEASON_OUT", "SUFFIX", "_renamed", "refit", "_seasons",
         "_check_season_folds", "_disjoint", "season_out")  # fmt: skip


def _doc_lines(tree: ast.AST) -> set[int]:
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            first = body[0] if body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):  # fmt: skip
                out |= set(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return out


def code_lines(text: str) -> int:
    """Lines holding a token other than COMMENT / NL / NEWLINE / INDENT / DEDENT / ENDMARKER,
    minus module, class and function docstring lines (the spec's counter, ``probe_code_lines``)."""
    docs = _doc_lines(ast.parse(text))
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT,
            tokenize.ENDMARKER}  # fmt: skip
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type not in skip:
            lines |= {n for n in range(tok.start[0], tok.end[0] + 1) if n not in docs}
    return len(lines)


def test_xgot_cap() -> None:
    """PM 2026-10-28: with the season-out block moved to ``season_out.py``, xgot.py is at or under
    its 720 code-line cap (MS6: 743). The counter is checked on hand-written source first: a
    docstring, comments, blank lines and a function docstring do not count; a multi-line call
    counts each line; a string that is not a docstring counts."""
    src = '''"""Module doc.

more doc
"""
# a comment

X = 1  # trailing comment


def f(a: int) -> int:
    """Doc."""
    y = g(
        a,
    )
    s = """not a
docstring"""
    return y
'''
    assert code_lines(src) == 8  # X, def, y = g(, a, ), s = (2 lines), return
    n = code_lines((SRC / "xgot.py").read_text(encoding="utf-8"))
    assert n <= XGOT_CAP, n


def test_no_season_out_in_xgot() -> None:
    """§2: the moved names live in ``season_out`` and xgot.py keeps no alias of them;
    ``season_out`` imports ``xgot``, never the reverse (no cycle)."""
    from football_shot_models import season_out, xgot

    assert [n for n in MOVED if hasattr(xgot, n)] == []
    assert all(hasattr(season_out, n) for n in MOVED)
    tree = ast.parse((SRC / "xgot.py").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported |= {f"{node.module}.{a.name}" for a in node.names} | {node.module or ""}
        elif isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
    assert not [m for m in imported if "season_out" in m], imported
    from football_shot_models import evaluation

    assert season_out.evaluate is evaluation.evaluate  # the name the conftest spy patches
