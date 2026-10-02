"""Scoring functions, all with the convention **higher = more similar**.

Every scorer is implemented from first principles rather than delegated to
``sklearn.metrics``, for the same reason the metrics are: a benchmark whose
numbers move when a dependency is upgraded is not a benchmark, and
``scipy.spatial.distance.cosine`` / ``euclidean`` have changed their reduction
order across releases.

Sign conventions, which are the usual source of silent bugs here:

* ``euclidean`` returns the **negated** distance, so that "more similar" is
  "smaller distance" without every caller having to remember to flip it;
* ``cosine`` assumes L2-normalised inputs but normalises defensively anyway;
* ``plda_llr`` delegates to :func:`voiceforge.sv.plda.plda_score`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..core.errors import ShapeError
from ..core.types import FLOAT_DTYPE

__all__ = [
    "cosine_similarity",
    "euclidean_score",
    "negative_euclidean",
    "zscore",
    "fuse_scores",
    "Scorer",
    "COSINE",
    "EUCLIDEAN",
    "PLDA_LLR",
    "SCORERS",
]


def _as_vector(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=FLOAT_DTYPE)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 1:
        raise ShapeError(f"{name} must be a 1-D vector", name=name, shape=list(arr.shape))
    return arr


def cosine_similarity(a: np.ndarray, b: np.ndarray, *, assume_normalised: bool = False) -> float:
    """Cosine similarity in ``[-1, 1]``; higher means more similar.

    Normalises defensively even when ``assume_normalised`` is set: the i-vector
    path already L2-normalises, but a zero or non-unit input would otherwise
    produce a silent NaN that only shows up much later as an unexplained EER.
    """
    x = _as_vector(a, "a")
    y = _as_vector(b, "b")
    if x.shape[0] != y.shape[0]:
        raise ShapeError("cosine input dimension mismatch", a=int(x.shape[0]), b=int(y.shape[0]))
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    if not assume_normalised:
        x = x / nx
        y = y / ny
    return float(np.dot(x, y))


def negative_euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Negative Euclidean distance; higher means more similar."""
    x = _as_vector(a, "a")
    y = _as_vector(b, "b")
    if x.shape[0] != y.shape[0]:
        raise ShapeError("euclidean input dimension mismatch", a=int(x.shape[0]), b=int(y.shape[0]))
    return float(-np.linalg.norm(x - y))


#: Alias kept for readability at call sites.
euclidean_score = negative_euclidean


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardise scores to zero mean and unit variance.

    A constant input (every trial scored identically) has zero variance; it is
    mapped to all-zeros rather than producing NaN, because a degenerate score
    distribution is a legitimate outcome that the fusion stage must survive.
    """
    arr = np.asarray(values, dtype=FLOAT_DTYPE).ravel()
    if arr.size == 0:
        return arr
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if sd < 1e-12:
        return np.zeros_like(arr)
    return (arr - mu) / sd


def fuse_scores(
    score_a: np.ndarray, score_b: np.ndarray, weight: float = 0.5, *, calibrate: str = "zscore"
) -> np.ndarray:
    """Linearly fuse two score vectors; higher means more similar.

    The two score sets live on different scales (a cosine in ``[-1, 1]`` and a
    PLDA LLR in nats), so they are z-scored before the weighted sum.  Without
    that step the fusion weight is meaningless -- it would be expressing a
    ratio of two arbitrary units rather than a preference between two detectors.
    """
    a = np.asarray(score_a, dtype=FLOAT_DTYPE).ravel()
    b = np.asarray(score_b, dtype=FLOAT_DTYPE).ravel()
    if a.shape[0] != b.shape[0]:
        raise ShapeError("fusion input length mismatch", a=int(a.shape[0]), b=int(b.shape[0]))
    if not 0.0 <= weight <= 1.0:
        raise ShapeError("fusion weight must lie in [0, 1]", weight=weight)
    if calibrate == "zscore":
        return weight * zscore(a) + (1.0 - weight) * zscore(b)
    if calibrate == "none":
        return weight * a + (1.0 - weight) * b
    raise ShapeError(f"unknown calibration {calibrate!r}", calibrate=calibrate, known=["zscore", "none"])


@dataclass
class Scorer:
    """A named similarity function with a vectorised trial interface."""

    name: str
    fn: Callable[..., float]

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(self.fn(a, b))

    def score_trials(self, enrolls: list[np.ndarray], tests: list[np.ndarray]) -> np.ndarray:
        """Score aligned enrollment/test lists.

        The identical pair is scored once and written to both positions, which
        halves the work for the (very common) symmetric case and guarantees the
        two off-diagonal entries are bit-identical.
        """
        if len(enrolls) != len(tests):
            raise ShapeError("trial count mismatch", n_enroll=len(enrolls), n_test=len(tests))
        n = len(enrolls)
        out = np.empty(n, dtype=FLOAT_DTYPE)
        for i in range(n):
            out[i] = self.fn(enrolls[i], tests[i])
        return out

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return self.fn(a, b)


def _plda_scorer(a: np.ndarray, b: np.ndarray) -> float:  # pragma: no cover - re-export shim
    from .plda import plda_score

    raise RuntimeError("PLDA_LLR requires a fitted model; use voiceforge.sv.plda.plda_score directly")


COSINE = Scorer("cosine", cosine_similarity)
EUCLIDEAN = Scorer("euclidean", negative_euclidean)

#: Registry of the pure-vector scorers.  PLDA is absent because it needs a
#: fitted model rather than a closure over two vectors.
SCORERS: dict[str, Scorer] = {
    "cosine": COSINE,
    "euclidean": EUCLIDEAN,
    "plda_llr": Scorer("plda_llr", _plda_scorer),
}


def scorer_table() -> list[dict[str, Any]]:
    """Machine-readable description of the scoring functions (for the report)."""
    return [
        {"name": "cosine", "higher_is_better": True, "range": [-1.0, 1.0], "requires_model": False},
        {"name": "euclidean", "higher_is_better": True, "range": "(-inf, 0]", "requires_model": False},
        {"name": "plda_llr", "higher_is_better": True, "range": "(-inf, inf)", "requires_model": True},
    ]
