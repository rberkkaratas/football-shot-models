"""A prediction cache: a rerun on the same data and code reads its fits instead of refitting.

``make xg-models`` fits 17 specs, about 8 minutes. Their cards are computed from three arrays
per spec (out-of-fold, in-sample and hold-out probabilities) plus the coefficient rows, so
those are what is cached, never a fitted model or a card: scores, comparisons, bootstraps and
the cards are recomputed on every run, from the cached arrays.

**The key** is the snapshot's data digest and :func:`source_hash`, a hash of every ``.py`` of
this package and of ``football_core`` plus the Python and library versions: any code edit
makes the next run cold, except in the render-only modules (:data:`RENDER_ONLY`: card and
chart rendering), which no fit-side module imports, so they cannot change a prediction, and
in :data:`ORCHESTRATION_ONLY` (``gsax.py``), which fits only through a hashed module.
**The identity** of one entry (:func:`identity`) is the spec's name, features and
``get_params()``, and a digest of the ``shot_id`` values of dev and the hold-out in frame
order, so a cached prediction can never serve another split or rows that no longer line up.
A spec that reads a computed input (the xGOT's cross-fitted ``xg_logit``) adds the input's
source name and :func:`column_digest` of its values, so a changed input is a miss. The fold
assignment is stored and compared too. Any difference is a miss that refits and overwrites;
an entry that cannot be read is an error naming the file.

Entries are written only where git ignores them (``data/cache``), through a temporary file
and an atomic rename, so a killed run never leaves half an entry. The cache is opt-in: only
the command line and the reproduction pass one. The cold run (``make reproduce-cold``, which
sets ``FSM_CACHE=off``) is the reproduction of record.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import football_core
import numpy as np
import numpy.typing as npt
import polars as pl
from football_core.io import snapshot

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]
Folds = Sequence[tuple[IntArray, IntArray]]
Coefficients = tuple[tuple[str, float, float], ...]
StrPath = str | os.PathLike[str]

LIBRARIES: Final[tuple[str, ...]] = ("numpy", "scipy", "scikit-learn", "polars")
ROOTS: Final[tuple[Path, ...]] = (
    Path(__file__).parent,
    Path(*football_core.__path__),
)
ENV: Final[str] = "FSM_CACHE"
# "<root name>/<relative posix path>": rendering only, imported by no other module under ROOTS
RENDER_ONLY: Final[frozenset[str]] = frozenset(
    {
        "football_shot_models/xgot_cards.py",
        "football_shot_models/gsax_cards.py",
        "football_core/viz.py",
    }
)
# orchestration that fits nothing itself: gsax.py calls xgot.season_out (hashed) for every fit,
# never the cache or a fitting module, so its arithmetic cannot change a cached prediction
ORCHESTRATION_ONLY: Final[frozenset[str]] = frozenset({"football_shot_models/gsax.py"})
FIELDS: Final[tuple[str, ...]] = (
    "key",
    "identity",
    "fold",
    "oof",
    "train",
    "holdout",
    "terms",
    "coef",
)
_NAME: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_ADDRESS: Final = re.compile(r" at 0x[0-9a-f]+")


def source_hash(roots: Sequence[Path] = ROOTS) -> str:
    """SHA-256 over every ``*.py`` under ``roots``, then the Python and library versions.

    Each file adds ``"<root name>/<relative posix path>\\0"`` and its own SHA-256, in sorted
    order, so a renamed or edited file changes the hash and a non-``.py`` file does not. A
    :data:`RENDER_ONLY` or :data:`ORCHESTRATION_ONLY` file is skipped, so editing card or chart
    rendering, or the goalkeeper tables computed from cached predictions, keeps the cache warm.

    Examples
    --------
    >>> len(source_hash()), source_hash() == source_hash()
    (64, True)
    """
    h = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            name = f"{root.name}/{path.relative_to(root).as_posix()}"
            if name in RENDER_ONLY or name in ORCHESTRATION_ONLY:
                continue
            h.update(f"{name}\0".encode())
            h.update(hashlib.sha256(path.read_bytes()).digest())
    versions = [f"python {sys.version_info[0]}.{sys.version_info[1]}"]
    versions += [f"{lib} {importlib.metadata.version(lib)}" for lib in LIBRARIES]
    h.update("\n".join(versions).encode())
    return h.hexdigest()


def cache_root(default: Path) -> Path | None:
    """The cache root from ``FSM_CACHE``: unset or ``""`` gives ``default``, ``"off"`` none.

    Raises
    ------
    ValueError
        Any other value.

    Examples
    --------
    >>> from unittest import mock
    >>> with mock.patch.dict(os.environ, {ENV: "off"}):
    ...     cache_root(Path("data/cache")) is None
    True
    """
    value = os.environ.get(ENV, "")
    if value == "":
        return default
    if value == "off":
        return None
    raise ValueError(f"{ENV}={value!r}; expected 'off' or unset (or empty)")


def rows_digest(dev_ids: pl.Series, held_ids: pl.Series) -> str:
    """SHA-256 of the dev ids, one per line, a blank line, then the hold-out ids, in order.

    Raises
    ------
    ValueError
        A null, empty or newline-holding ``shot_id`` on either side (the encoding would be
        ambiguous).

    Examples
    --------
    >>> rows_digest(pl.Series(["sb:1", "sb:2"]), pl.Series(["sb:3"])) == hashlib.sha256(
    ...     b"sb:1\\nsb:2\\n\\nsb:3").hexdigest()
    True
    """
    parts = []
    for side, ids in (("dev", dev_ids), ("held", held_ids)):
        bad = ids.is_null() | (ids.str.len_chars() == 0) | ids.str.contains("\n", literal=True)
        if n := int(bad.sum()):
            raise ValueError(f"cache: {n} null, empty or multi-line shot_id value(s) in {side}")
        parts.append("\n".join(ids.to_list()))
    return hashlib.sha256("\n\n".join(parts).encode()).hexdigest()


def column_digest(dev: pl.DataFrame, held: pl.DataFrame, column: str) -> str:
    """SHA-256 of the Float64 values of ``column`` in ``dev``, then in ``held``.

    Raises
    ------
    ValueError
        The column is missing from either frame or holds a null.

    Examples
    --------
    >>> a, b = pl.DataFrame({"x": [0.5, 1.0]}), pl.DataFrame({"x": [2.0]})
    >>> column_digest(a, b, "x") == column_digest(a, b, "x"), column_digest(a, b, "x") == (
    ...     column_digest(b, a, "x"))
    (True, False)
    """
    h = hashlib.sha256()
    for side, frame in (("dev", dev), ("held", held)):
        if column not in frame.columns:
            raise ValueError(f"cache: {side} has no {column!r} column")
        values = frame[column].cast(pl.Float64)
        if n := values.null_count():
            raise ValueError(f"cache: {column!r} has {n} null value(s) in {side}")
        h.update(values.to_numpy().astype("<f8").tobytes() + side.encode())
    return h.hexdigest()


def identity(
    name: str,
    features: tuple[str, ...],
    params: Mapping[str, object],
    dev_ids: pl.Series,
    held_ids: pl.Series,
    inputs: str = "",
) -> str:
    """What one cache entry was fitted on: spec name, features, settings, rows and inputs.

    ``inputs`` (a computed input's source and :func:`column_digest`) is appended as
    ``|inputs <inputs>`` when not empty; empty keeps the text of a spec without inputs.

    Raises
    ------
    ValueError
        The text holds a memory address (a function in ``params``, different in every
        process, so the entry would never hit); or from :func:`rows_digest`.

    Examples
    --------
    >>> ids = pl.Series(["sb:1"]), pl.Series(["sb:2"])
    >>> identity("xg-x", ("distance_m",), {"C": 1.0}, *ids).rsplit("|", 1)[0]
    "xg-x|('distance_m',)|{'C': 1.0}"
    """
    text = f"{name}|{features}|{params!r}|rows {rows_digest(dev_ids, held_ids)}"
    text += f"|inputs {inputs}" if inputs else ""
    if _ADDRESS.search(text):
        raise ValueError(f"cache: the identity of {name!r} holds a memory address: {text[:200]}")
    return text


@dataclass(frozen=True)
class Predictions:
    """The arrays a card needs from one spec's fits.

    Attributes
    ----------
    oof : numpy.ndarray
        Out-of-fold probabilities, in dev order.
    train, holdout : numpy.ndarray
        The fit on all of dev, on dev (in-sample) and on the hold-out.
    coefficients : tuple of (str, float, float)
        The card rows of that fit.
    """

    oof: FloatArray
    train: FloatArray
    holdout: FloatArray
    coefficients: Coefficients


def fold_ids(folds: Folds) -> npt.NDArray[np.int64]:
    """The fold of every dev row, from (train, test) index pairs.

    Examples
    --------
    >>> fold_ids(((np.array([1]), np.array([0])), (np.array([0]), np.array([1])))).tolist()
    [0, 1]
    """
    out = np.full(sum(test.size for _, test in folds), -1, dtype=np.int64)
    for i, (_, test) in enumerate(folds):
        out[test] = i
    return out


class PredictionCache:
    """Per-spec predictions under ``root``, keyed by (data digest, source hash).

    Parameters
    ----------
    root : str or path
        The cache root (``data/cache``); entries go to ``root/<digest[:16]>-<source[:16]>/``.
    data_digest : str
        The snapshot manifest's ``data_digest``.
    source : str, optional
        :func:`source_hash` (computed when omitted).

    Attributes
    ----------
    dir : pathlib.Path
        This key's directory.
    key : tuple of str
        ``(data_digest, source)``, stored in every entry and compared on read.
    hits, written : list of str
        Spec names read and written by this object, in order.

    Examples
    --------
    >>> folds = ((np.array([1, 2]), np.array([0])), (np.array([0]), np.array([1, 2])))
    >>> p = Predictions(np.array([0.1, 0.2, 0.3]), np.array([0.2, 0.2, 0.2]), np.array([0.4]),
    ...                 (("intercept", -1.5, -1.5),))
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     c = PredictionCache(tmp, "d" * 64, "s" * 64)
    ...     c.save("xg-x", "id", folds, p)
    ...     back = c.load("xg-x", "id", folds)
    ...     (back.coefficients, c.load("xg-x", "other id", folds), c.hits, c.written)
    ((('intercept', -1.5, -1.5),), None, ['xg-x'], ['xg-x'])
    """

    def __init__(self, root: StrPath, data_digest: str, source: str | None = None) -> None:
        source = source_hash() if source is None else source
        self.dir = Path(root) / f"{data_digest[:16]}-{source[:16]}"
        self.key = (data_digest, source)
        self.hits: list[str] = []
        self.written: list[str] = []

    def _path(self, name: str) -> Path:
        if not _NAME.fullmatch(name):
            raise ValueError(f"cache: spec name {name!r} is not a file name")
        return self.dir / f"{name}.npz"

    def load(self, name: str, identity: str, folds: Folds) -> Predictions | None:
        """The entry of ``name`` if its key, identity and folds match; ``None`` otherwise.

        Raises
        ------
        ValueError
            A name that is not a file name; an entry that cannot be read (the message names
            the file: delete it).
        """
        path = self._path(name)
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                e = {k: z[k] for k in FIELDS}
            fits = (
                tuple(e["key"].tolist()) == self.key
                and str(e["identity"]) == identity
                and np.array_equal(e["fold"], fold_ids(folds))
            )
            coef = tuple(
                (str(t), float(a), float(b))
                for t, (a, b) in zip(e["terms"].tolist(), e["coef"].tolist(), strict=True)
            )
        except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as err:
            raise ValueError(f"cache: cannot read {path} ({err}); delete it") from None
        if not fits:
            return None
        self.hits.append(name)
        return Predictions(e["oof"], e["train"], e["holdout"], coef)

    def save(self, name: str, identity: str, folds: Folds, p: Predictions) -> None:
        """Write the entry of ``name`` atomically, only where git ignores it.

        Raises
        ------
        ValueError
            A name that is not a file name; a path git would track, or git not answering
            (from ``football_core.io.snapshot``).
        OSError
            On a filesystem error.
        """
        path = self._path(name)
        # unresolved on purpose: a data/ symlinked out of the repo must fail closed
        snapshot._check_ignored(Path(os.path.abspath(path)))
        self.dir.mkdir(parents=True, exist_ok=True)
        # a unique tmp name: two runs writing one key never share or truncate a tmp file
        fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f"{name}.", suffix=".npz.tmp")
        with os.fdopen(fd, "wb") as f:
            np.savez(
                f, key=np.array(self.key), identity=np.array(identity), fold=fold_ids(folds),
                oof=p.oof, train=p.train, holdout=p.holdout,
                terms=np.array([t for t, _, _ in p.coefficients], dtype=str),
                coef=np.array([c[1:] for c in p.coefficients], dtype=np.float64).reshape(-1, 2),
            )  # fmt: skip
        os.replace(tmp, path)
        self.written.append(name)
