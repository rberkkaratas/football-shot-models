"""The prediction cache (spec MS4.prep §5): key, entry round trip, misses, refusals, the rows
digest, ``FSM_CACHE``, ``evaluation.evaluate(cache=)`` and ``models.run`` read warm.

Arrays and ids are invented. Everything written goes to tmp paths; ``source=`` is passed
explicitly wherever the source hash is not under test. The cache module is imported by the
``fsm_cache`` fixture (not at module level, so a missing module fails these tests only; pytest
already has a fixture called ``cache``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib
import importlib.metadata
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import models_cases as mc
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from football_shot_models import ablation, baselines, evaluation, features

DIGEST = "0123456789abcdef" + "a" * 48
SOURCE = "fedcba9876543210" + "b" * 48
NAME = "xg-lr-distance-angle"
SMALLEST = float(np.nextafter(0.0, 1.0))  # the smallest subnormal


@pytest.fixture(scope="module")
def fsm_cache() -> ModuleType:
    return importlib.import_module("football_shot_models.cache")


def _folds(n: int, k: int) -> tuple[tuple[Any, Any], ...]:
    """``k`` folds over ``n`` rows (``n >= k``): fold ``i`` tests rows ``i, i + k, ...``."""
    out = []
    for i in range(k):
        test = np.arange(i, n, k, dtype=np.intp)
        out.append((np.setdiff1d(np.arange(n), test).astype(np.intp), test))
    return tuple(out)


def _predictions(fsm_cache: ModuleType, n: int = 10, coefficients: Any = ()) -> Any:
    oof = np.linspace(0.0, 1.0, n)
    return fsm_cache.Predictions(oof, oof[::-1].copy(), np.full(3, 0.25), coefficients)


def _bits(values: Any) -> bytes:
    return np.ascontiguousarray(values, dtype=np.float64).tobytes()


# ---- key ----------------------------------------------------------------------------------------


def test_source_hash(
    fsm_cache: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§5.1: the source hash is stable; one changed byte, a renamed ``.py``, a renamed root and a
    library version each change it; a non-``.py`` file does not count. The default roots are the
    two package directories."""
    a, b = tmp_path / "pkg_a", tmp_path / "pkg_b"
    (a / "sub").mkdir(parents=True)
    b.mkdir()
    (a / "one.py").write_text("x = 1\n")
    (a / "sub" / "two.py").write_text("y = 2\n")
    (a / "notes.txt").write_text("not code\n")
    (b / "three.py").write_text("z = 3\n")
    first = fsm_cache.source_hash([a, b])
    assert len(first) == 64 and int(first, 16) >= 0
    assert fsm_cache.source_hash([a, b]) == first
    (a / "notes.txt").write_text("changed, still not code\n")
    assert fsm_cache.source_hash([a, b]) == first
    (a / "sub" / "two.py").write_text("y = 3\n")
    assert fsm_cache.source_hash([a, b]) != first
    (a / "sub" / "two.py").write_text("y = 2\n")
    assert fsm_cache.source_hash([a, b]) == first  # content, not mtime
    (a / "one.py").rename(a / "uno.py")
    assert fsm_cache.source_hash([a, b]) != first
    (a / "uno.py").rename(a / "one.py")
    shutil.copytree(a, tmp_path / "pkg_z")
    assert fsm_cache.source_hash([tmp_path / "pkg_z", b]) != first  # the root name counts
    real = importlib.metadata.version

    def bumped(lib: str) -> str:
        return real(lib) + ".post999" if lib == "polars" else real(lib)

    monkeypatch.setattr(importlib.metadata, "version", bumped)
    assert fsm_cache.source_hash([a, b]) != first
    monkeypatch.undo()
    assert fsm_cache.LIBRARIES == ("numpy", "scipy", "scikit-learn", "polars")
    assert {r.name for r in fsm_cache.ROOTS} == {"football_shot_models", "football_core"}
    assert fsm_cache.source_hash() == fsm_cache.source_hash(fsm_cache.ROOTS)


_UNIT = st.one_of(st.sampled_from([0.0, 1.0, SMALLEST]), st.floats(0.0, 1.0))
_COEF = st.one_of(
    st.sampled_from([0.0, -0.0, 1e308, -1e308, 1e-300, -SMALLEST]),
    st.floats(allow_nan=False, allow_infinity=False),
)
_TERM = st.from_regex(r"[a-z][a-z_:]{0,19}", fullmatch=True)


@settings(deadline=None, max_examples=40)
@given(
    data=st.data(),
    n=st.integers(2, 30),
    k=st.integers(2, 5),
    coefficients=st.lists(st.tuples(_TERM, _COEF, _COEF), max_size=6).map(tuple),
)
def test_cache_round_trip(
    fsm_cache: ModuleType, data: st.DataObject, n: int, k: int, coefficients: Any
) -> None:
    """§5.1, §6: arrays in [0, 1] (0, 1 and the smallest subnormal included) and coefficient rows
    (±0.0, ±1e308, ``()`` included) come back bit for bit; the write goes through a fresh tmp
    file, so a stale ``<name>.npz.tmp`` planted in the dir is still there, unchanged, after the
    save, and no other ``.tmp`` is left."""
    k = min(k, n)
    oof = np.array(data.draw(st.lists(_UNIT, min_size=n, max_size=n)), dtype=np.float64)
    train = np.array(data.draw(st.lists(_UNIT, min_size=n, max_size=n)), dtype=np.float64)
    held = np.array(data.draw(st.lists(_UNIT, max_size=12)), dtype=np.float64)
    folds = _folds(n, k)
    with tempfile.TemporaryDirectory() as tmp:
        c = fsm_cache.PredictionCache(tmp, DIGEST, SOURCE)
        assert c.key == (DIGEST, SOURCE)
        assert c.dir == Path(tmp) / f"{DIGEST[:16]}-{SOURCE[:16]}"
        c.dir.mkdir(parents=True)
        stale = c.dir / f"{NAME}.npz.tmp"
        stale.write_bytes(b"stale")
        p = fsm_cache.Predictions(oof, train, held, coefficients)
        c.save(NAME, "id-1", folds, p)
        assert c.written == [NAME] and c.hits == []
        assert stale.read_bytes() == b"stale"
        assert sorted(f.name for f in c.dir.iterdir()) == [f"{NAME}.npz", f"{NAME}.npz.tmp"]
        back = c.load(NAME, "id-1", folds)
        assert back is not None and c.hits == [NAME]
        for got, want in ((back.oof, oof), (back.train, train), (back.holdout, held)):
            assert got.dtype == np.float64 and got.shape == want.shape
            assert _bits(got) == _bits(want)
        assert [t for t, _, _ in back.coefficients] == [t for t, _, _ in coefficients]
        got_coef = [(float(a).hex(), float(b).hex()) for _, a, b in back.coefficients]
        assert got_coef == [(float(a).hex(), float(b).hex()) for _, a, b in coefficients]


@pytest.mark.parametrize("case", ["other digest", "other source", "other identity", "fold order"])
def test_cache_misses(fsm_cache: ModuleType, tmp_path: Path, case: str) -> None:
    """§5.1: a key differing only after the 16 characters of the dir name (so the same file is
    read and the stored key is what differs), another identity, or the same folds in another
    order (equal sizes) is a miss: ``None``, nothing appended to ``hits``, the entry kept."""
    folds = _folds(10, 5)
    c = fsm_cache.PredictionCache(tmp_path, DIGEST, SOURCE)
    c.save(NAME, "id-1", folds, _predictions(fsm_cache))
    assert fsm_cache.PredictionCache(tmp_path, DIGEST, SOURCE).load(NAME, "id-1", folds) is not None
    digest, source, identity, order = DIGEST, SOURCE, "id-1", folds
    if case == "other digest":
        digest = DIGEST[:16] + "c" * 48
    elif case == "other source":
        source = SOURCE[:16] + "d" * 48
    elif case == "other identity":
        identity = "id-2"
    else:
        order = (folds[1], folds[0], *folds[2:])
        assert {f[1].size for f in folds} == {2}  # equal sizes: only the labels move
    reader = fsm_cache.PredictionCache(tmp_path, digest, source)
    assert reader.dir == c.dir  # the same entry file is read
    assert reader.load(NAME, identity, order) is None
    assert reader.hits == []
    assert (c.dir / f"{NAME}.npz").is_file()


def _entry_without_coef(fsm_cache: ModuleType, folds: Any) -> bytes:
    buf = io.BytesIO()
    np.savez(
        buf, key=np.array((DIGEST, SOURCE)), identity=np.array("id-1"),
        fold=fsm_cache.fold_ids(folds), oof=np.zeros(10), train=np.zeros(10),
        holdout=np.zeros(3), terms=np.array([], dtype=str),
    )  # fmt: skip
    return buf.getvalue()


@pytest.mark.parametrize(
    "case", ["not a zip", "truncated", "empty", "missing field", "../x", "a/b", ".x", "x\n"]
)
def test_cache_unreadable(fsm_cache: ModuleType, tmp_path: Path, case: str) -> None:
    """§5.1, Decision 5: an entry that is not a zip, truncated, empty or missing a field is a
    ``ValueError`` naming the file (fail closed, never a silent miss); a spec name that is not a
    plain file name is a ``ValueError`` naming it, on load and on save."""
    folds = _folds(10, 5)
    c = fsm_cache.PredictionCache(tmp_path, DIGEST, SOURCE)
    if case in ("../x", "a/b", ".x", "x\n"):
        for call in (
            lambda: c.load(case, "id-1", folds),
            lambda: c.save(case, "id-1", folds, _predictions(fsm_cache)),
        ):
            with pytest.raises(ValueError) as info:
                call()
            assert repr(case) in str(info.value) or case in str(info.value)
        assert not (tmp_path / "x.npz").exists() and not list(tmp_path.rglob("*.npz"))
        return
    c.save(NAME, "id-1", folds, _predictions(fsm_cache))
    path = c.dir / f"{NAME}.npz"
    good = path.read_bytes()
    blob = {
        "not a zip": b"not a zip",
        "truncated": good[: len(good) // 2],
        "empty": b"",
        "missing field": _entry_without_coef(fsm_cache, folds),
    }[case]
    path.write_bytes(blob)
    with pytest.raises(ValueError) as info:
        c.load(NAME, "id-1", folds)
    assert str(path) in str(info.value)
    assert c.hits == []


def _git_repo(root: Path) -> Path:
    root.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def test_cache_refuses_tracked_path(fsm_cache: ModuleType, tmp_path: Path) -> None:
    """§5.1, §5.4, hard rule 1: in a git work tree without a ``data/*`` ignore, ``save`` raises
    and writes nothing; with the ignore it writes; with ``data/`` a symlink out of the repo the
    unresolved path is checked, so git cannot answer and the save fails closed."""
    folds, p = _folds(10, 5), _predictions(fsm_cache)
    repo = _git_repo(tmp_path / "repo")
    c = fsm_cache.PredictionCache(repo / "data" / "cache", DIGEST, SOURCE)
    with pytest.raises(ValueError, match="ignore"):
        c.save(NAME, "id-1", folds, p)
    assert not [f for f in repo.rglob("*") if f.is_file() and ".git" not in f.parts]
    assert c.written == []
    (repo / ".gitignore").write_text("data/*\n")
    c.save(NAME, "id-1", folds, p)
    assert (c.dir / f"{NAME}.npz").is_file() and c.written == [NAME]
    linked = _git_repo(tmp_path / "linked")
    (linked / ".gitignore").write_text("data/*\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (linked / "data").symlink_to(outside, target_is_directory=True)
    via_link = fsm_cache.PredictionCache(linked / "data" / "cache", DIGEST, SOURCE)
    with pytest.raises(ValueError):
        via_link.save(NAME, "id-1", folds, p)
    assert not list(outside.rglob("*.npz"))


def test_cache_identity(fsm_cache: ModuleType) -> None:
    """§5.1: the rows digest is the pinned encoding (dev ids, a blank line, held ids, in frame
    order); a null, empty or multi-line ``shot_id`` on either side is a ``ValueError``; the
    identity is ``name|features|params!r|rows digest``; a memory address in it (a lambda in the
    params) is a ``ValueError``; every spec ``models.run`` evaluates has an address-free
    identity."""
    dev, held = pl.Series(["sb:1", "sb:2"]), pl.Series(["sb:3"])
    want = hashlib.sha256(b"sb:1\nsb:2\n\nsb:3").hexdigest()
    assert fsm_cache.rows_digest(dev, held) == want
    assert fsm_cache.rows_digest(dev[::-1], held) != want
    assert fsm_cache.rows_digest(pl.Series(["sb:1", "sb:3"]), pl.Series(["sb:2"])) != want
    assert fsm_cache.rows_digest(pl.Series(["ab"]), pl.Series(["c"])) != fsm_cache.rows_digest(
        pl.Series(["a"]), pl.Series(["bc"])
    )
    bad = pl.Series(["sb:1", None], dtype=pl.String())
    for ids in (bad, pl.Series(["sb:1", ""]), pl.Series(["sb:1", "sb:2\nsb:3"])):
        with pytest.raises(ValueError):
            fsm_cache.rows_digest(ids, held)
        with pytest.raises(ValueError):
            fsm_cache.rows_digest(dev, ids)
    params = {"C": 1.0, "max_iter": 100}
    text = fsm_cache.identity(NAME, ("distance_m", "angle_rad"), params, dev, held)
    assert text == f"{NAME}|('distance_m', 'angle_rad')|{params!r}|rows {want}"
    with pytest.raises(ValueError, match="address"):
        fsm_cache.identity(NAME, ("distance_m",), {"f": lambda v: v}, dev, held)
    from football_shot_models import models

    variants = tuple(v.spec() for v in ablation.VARIANTS)
    for spec in (*baselines.BASELINES, *ablation.STEPS, *variants, models.GBM):
        fsm_cache.identity(spec.name, spec.features, spec.make().get_params(), dev, held)


def test_cache_root(fsm_cache: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """§5.1: ``FSM_CACHE`` unset or "" -> the default, "off" -> None, anything else -> a
    ``ValueError`` naming the value and the allowed one."""
    default = Path("data") / "cache"
    assert fsm_cache.ENV == "FSM_CACHE"
    monkeypatch.delenv("FSM_CACHE", raising=False)
    assert fsm_cache.cache_root(default) == default
    monkeypatch.setenv("FSM_CACHE", "")
    assert fsm_cache.cache_root(default) == default
    monkeypatch.setenv("FSM_CACHE", "off")
    assert fsm_cache.cache_root(default) is None
    monkeypatch.setenv("FSM_CACHE", "no")
    with pytest.raises(ValueError, match="(?=.*'no')(?=.*off)"):
        fsm_cache.cache_root(default)


# ---- evaluate(cache=) ---------------------------------------------------------------------------


def _same(a: evaluation.Evaluation, b: evaluation.Evaluation) -> None:
    assert a.spec is b.spec
    assert (a.folds, a.cv, a.train, a.holdout) == (b.folds, b.cv, b.train, b.holdout)
    assert a.coefficients == b.coefficients
    assert np.array_equal(a.oof, b.oof) and np.array_equal(a.holdout_p, b.holdout_p)
    assert len(a.fold_test) == len(b.fold_test)
    assert all(np.array_equal(x, y) for x, y in zip(a.fold_test, b.fold_test, strict=True))


def _refuse_fit(*args: Any, **kwargs: Any) -> Any:
    raise ValueError("fitted on a cache hit")  # a ValueError, so models.main reports it


@pytest.mark.parametrize(
    "case", ["hit", "frames", "no shot_id", "row moved", "dev permuted", "swap at equal heights"]
)
def test_evaluate_cached(
    fsm_cache: ModuleType,
    xg_data: Any,
    dev_held: tuple[pl.DataFrame, pl.DataFrame],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """§5.2: a hit rebuilds the miss's Evaluation field by field without a fit (``make()`` once,
    for the identity) and still runs the leakage guard; a frame-feature spec read with
    ``freeze_frames=None`` raises ``_derive``'s error on a hit as it does uncached; a cache needs
    ``shot_id``; a row moved to the hold-out, dev permuted inside one fold, or a dev id swapped
    with a held id at equal heights is a miss (the last two with the cached fold ids, so only
    the rows digest can make them miss)."""
    dev, held = dev_held
    folds = evaluation.cv_folds(dev)
    calls = {"make": 0}
    base = baselines.BASELINES[1]  # LR distance + angle: coefficient rows to round-trip

    def make() -> Any:
        calls["make"] += 1
        return base.make()

    spec = dataclasses.replace(base, make=make)
    c = fsm_cache.PredictionCache(tmp_path, DIGEST, SOURCE)
    if case == "no shot_id":
        with pytest.raises(ValueError, match="shot_id"):
            evaluation.evaluate(spec, dev.drop("shot_id"), held, folds, cache=c)
        return
    if case == "frames":
        frame_spec = evaluation.ModelSpec(
            "xg-frame-base", "Frame base rate", ("nearest_opponent_m",), evaluation.BaseRate
        )
        assert "freeze_frames" in {t for t, _ in features.DERIVED["nearest_opponent_m"].sources}
        message = "reads freeze_frames; pass freeze_frames="
        with pytest.raises(ValueError, match=message):
            evaluation.evaluate(frame_spec, dev, held, folds)  # uncached
        evaluation.evaluate(frame_spec, dev, held, folds, freeze_frames=xg_data.freeze_frames,
                            cache=c)  # fmt: skip
        assert c.written == [frame_spec.name]
        with pytest.raises(ValueError, match=message):
            evaluation.evaluate(frame_spec, dev, held, folds, cache=c)
        return
    first = evaluation.evaluate(spec, dev, held, folds, cache=c)
    assert c.written == [spec.name] and c.hits == []
    _same(first, evaluation.evaluate(spec, dev, held, folds))  # a miss changes nothing
    assert first.coefficients  # the LR's rows are part of the round trip
    stored = np.load(c.dir / f"{spec.name}.npz", allow_pickle=False)["fold"]
    if case == "hit":
        guard = features.check_xg_features
        seen: list[tuple[str, ...]] = []

        def spy(names: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(tuple(names))
            return guard(names, *args, **kwargs)

        monkeypatch.setattr(features, "check_xg_features", spy)
        monkeypatch.setattr(evaluation, "_fit", _refuse_fit)
        calls["make"] = 0
        again = evaluation.evaluate(spec, dev, held, folds, cache=c)
        _same(again, first)
        assert c.hits == [spec.name] and calls["make"] == 1
        assert spec.features in seen
        return
    if case == "row moved":
        dev2, held2 = dev.slice(0, dev.height - 1), pl.concat([held, dev.tail(1)])
        folds2 = evaluation.cv_folds(dev2)
    elif case == "dev permuted":
        test = folds[0][1]
        assert test.size >= 2
        order = np.arange(dev.height)
        order[test] = test[::-1]
        dev2, held2, folds2 = dev[order], held, folds
        assert dev2["shot_id"].to_list() != dev["shot_id"].to_list()
    else:
        i = int(folds[0][1][0])
        dev2 = pl.concat([dev.slice(0, i), held.head(1), dev.slice(i + 1)])
        held2 = pl.concat([dev.slice(i, 1), held.slice(1)])
        folds2 = folds
        assert (dev2.height, held2.height) == (dev.height, held.height)
    if case != "row moved":
        assert np.array_equal(fsm_cache.fold_ids(folds2), stored)  # only the rows digest differs
    evaluation.evaluate(spec, dev2, held2, folds2, cache=c)
    assert c.hits == [] and c.written == [spec.name, spec.name]


# ---- models.run with a cache -----------------------------------------------------------------


def _run_main(models: Any, argv: list[str]) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = models.main(argv)
    return code, out.getvalue(), err.getvalue()


def test_models_run_cached(
    fsm_cache: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§5.3, Decision 4: ``main --cache DIR`` fills the cache (17 written); a second
    ``models.run(cache=DIR)`` reads all 20 specs (17, plus POOLED and the two level fixes of
    PR1.xg-context §3.3, all evaluated when ``ADOPTED`` is None), fits nothing, and writes the
    same 5 card bytes and the same stdout but for the last line ``cache: {dir} (20 read, 0
    written)``; ``main``
    without flags reads ``data/cache`` under the working directory; ``--no-cache`` reads nothing
    (so the fit runs) and leaves the cache as it was."""
    from football_shot_models import data, models

    manifest = mc.write_models_snapshot(tmp_path / "fixture")
    mc.patch_level(models, monkeypatch)
    root, docs = tmp_path / "cache", tmp_path / "docs"
    common = ["--manifest", str(manifest), "--docs-dir", str(docs)]
    code, first_out, err = _run_main(models, ["--cache", str(root), *common])
    assert code == 0, err
    n_specs = len(baselines.BASELINES) + len(ablation.STEPS) + len(ablation.VARIANTS) + 1
    assert n_specs == 17
    n_specs += 1 + len(models.LEVEL_FIXES)  # PR1.xg-context: POOLED and both fixes (ADOPTED None)
    key_dir = fsm_cache.PredictionCache(root, data.load_xg_data(manifest).manifest.data_digest).dir
    assert first_out.splitlines()[-1] == f"cache: {key_dir} (0 read, {n_specs} written)"
    assert len(list(key_dir.glob("*.npz"))) == n_specs
    names = sorted(p.name for p in docs.iterdir())
    assert len(names) == 5
    cold = {name: (docs / name).read_bytes() for name in names}
    monkeypatch.setattr(evaluation, "_fit", _refuse_fit)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        models.run(manifest, docs_dir=docs, cache=root)
    warm_out = out.getvalue()
    assert {name: (docs / name).read_bytes() for name in names} == cold
    assert warm_out.splitlines()[-1] == f"cache: {key_dir} ({n_specs} read, 0 written)"
    assert warm_out.splitlines()[:-1] == first_out.splitlines()[:-1]
    work = tmp_path / "work"
    shutil.copytree(root, work / "data" / "cache")
    monkeypatch.chdir(work)
    code, default_out, err = _run_main(models, common)
    assert code == 0, err
    assert default_out.splitlines()[-1].endswith(f"({n_specs} read, 0 written)")
    before = sorted((work / "data" / "cache").rglob("*"))
    code, no_cache_out, err = _run_main(models, ["--no-cache", *common])
    assert code != 0 and "fitted on a cache hit" in err  # it had to fit: nothing was read
    assert sorted((work / "data" / "cache").rglob("*")) == before
    assert "cache:" not in no_cache_out
