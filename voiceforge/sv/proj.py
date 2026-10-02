"""Linear Discriminant Analysis with shrinkage-regularised whitening.

Classic LDA projects ``D``-dimensional features onto at most ``C - 1``
discriminant directions, where ``C`` is the number of classes (here, speakers).
The textbook recipe divides by the within-class scatter, which is exactly the
step that breaks on real data: with hundreds of speakers and a few thousand
frames each, ``Sw`` is estimated from finite samples and can easily be
singular or badly conditioned.  Two regularisations are applied:

1. **Shrinkage** towards a scaled identity, ``Sw' = (1-a) Sw + a * (tr(Sw)/D) I``
   (Ledoit-Wolf style).  This keeps the whitening well posed and, unlike a
   pseudo-inverse, does not silently amplify the null space.
2. A **variance floor** after whitening, so a direction whose variance is
   numerically zero cannot dominate the projection.

Leakage discipline
------------------
:func:`fit_lda` takes the training features and their labels and returns a
frozen transform.  Nothing in this module ever sees test data; the pipeline is
structured so the transform is fitted once on the training speakers and then
applied to both splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..core.errors import RankDeficientError, ShapeError
from ..core.types import FLOAT_DTYPE

__all__ = ["LDAProjector", "fit_lda", "GlobalMean", "fit_global_mean", "center", "expand_labels"]


@dataclass
class LDAProjector:
    """A frozen LDA transform ``x -> (x - mean) @ W``.

    Attributes
    ----------
    mean:
        ``(D,)`` global mean, fitted on the training split only.
    W:
        ``(D, n_components)`` projection matrix (orthonormal columns).
    explained_variance_ratio:
        Per-direction ratio of between- to total scatter.
    eigenvalues:
        The retained discriminant eigenvalues, descending.
    """

    mean: np.ndarray
    W: np.ndarray
    explained_variance_ratio: np.ndarray
    eigenvalues: np.ndarray
    n_components: int
    shrinkage: float
    n_classes: int
    n_train_frames: int

    def __post_init__(self) -> None:
        for name in ("mean", "W", "explained_variance_ratio", "eigenvalues"):
            arr = np.ascontiguousarray(getattr(self, name), dtype=FLOAT_DTYPE)
            object.__setattr__(self, name, arr)
        if self.W.shape[0] != self.mean.shape[0]:
            raise ShapeError("LDA mean/transform shape mismatch", mean=list(self.mean.shape), W=list(self.W.shape))

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.W.shape[1])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Project features; accepts ``(N, D)`` or a single ``(D,)`` vector."""
        arr = np.asarray(x, dtype=FLOAT_DTYPE)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.n_features:
            raise ShapeError(
                "LDA input shape mismatch", expected=[None, self.n_features], got=list(arr.shape)
            )
        out = (arr - self.mean[None, :]) @ self.W
        return out[0] if single else out

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "n_components": self.output_dim,
            "n_classes": self.n_classes,
            "n_train_frames": self.n_train_frames,
            "shrinkage": self.shrinkage,
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "explained_variance_ratio": [float(v) for v in self.explained_variance_ratio],
            "total_explained": float(np.sum(self.explained_variance_ratio)),
        }


def expand_labels(
    frame_counts: Sequence[int], utterance_labels: Sequence[Any]
) -> np.ndarray:
    """Map one label per utterance onto one label per frame.

    ``frame_counts[i]`` frames come from utterance ``i``, which carries
    ``utterance_labels[i]``.  Used by the trainer, which holds features
    utterance-by-utterance but must fit LDA on a stacked frame matrix.
    """
    counts = [int(c) for c in frame_counts]
    if len(counts) != len(utterance_labels):
        raise ShapeError(
            "frame_counts and utterance_labels length mismatch",
            n_counts=len(counts),
            n_labels=len(utterance_labels),
        )
    out: list[Any] = []
    for count, label in zip(counts, utterance_labels):
        out.extend([label] * count)
    return np.asarray(out)


def _balance(feats: np.ndarray, y: np.ndarray, n_classes: int, n_repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Subsample every class to at most ``n_repeats`` frames."""
    keep: list[np.ndarray] = []
    for c in range(n_classes):
        idx = np.nonzero(y == c)[0]
        if idx.shape[0] == 0:
            continue
        if idx.shape[0] > n_repeats:
            sel = np.linspace(0, idx.shape[0] - 1, n_repeats).astype(np.int64)
            idx = idx[sel]
        keep.append(idx)
    if not keep:
        return feats, y
    sel = np.sort(np.concatenate(keep))
    return np.ascontiguousarray(feats[sel]), y[sel]


def fit_lda(
    x: np.ndarray,
    labels: Sequence[int] | Sequence[str] | np.ndarray,
    *,
    n_components: int = 40,
    shrinkage: float = 0.15,
    var_floor: float = 1e-10,
    n_repeats: int | None = None,
) -> LDAProjector:
    """Fit an LDA projector on ``(features, labels)``.

    Parameters
    ----------
    labels:
        One label per **frame**.  A shorter sequence (e.g. one speaker id per
        utterance, which is what the trainer naturally holds) is rejected
        rather than broadcast: expanding a per-utterance label list against a
        per-frame matrix is the caller's job, via :func:`expand_labels`.
    n_repeats:
        Optional class balancing.  Utterance lengths vary widely, so without it
        the longest speaker dominates the between-class scatter.
    n_components:
        Requested output dimensionality.  Capped at ``n_classes - 1`` because
        that is the rank of the between-class scatter; asking for more is a
        configuration error worth reporting rather than silently padding with
        zero columns.
    shrinkage:
        Blend weight towards a scaled identity, in ``[0, 1]``.
    var_floor:
        Post-whitening variance floor.
    """
    feats = np.ascontiguousarray(x, dtype=FLOAT_DTYPE)
    if feats.ndim != 2:
        raise ShapeError("LDA training features must be 2-D", shape=list(feats.shape))
    n, d = feats.shape

    y_raw = np.asarray(labels)
    if y_raw.shape[0] != n:
        raise ShapeError(
            "labels must be supplied per frame",
            n_frames=int(n),
            n_labels=int(y_raw.shape[0]),
            remedy="use expand_labels() to map utterance labels onto frames",
        )

    classes, y = np.unique(y_raw, return_inverse=True)
    n_classes = int(classes.shape[0])
    if n_classes < 2:
        raise RankDeficientError(
            "LDA needs at least 2 classes", n_classes=n_classes, remedy="use more training speakers"
        )
    if n < n_classes:
        raise RankDeficientError("fewer frames than classes", n_frames=n, n_classes=n_classes)

    if n_repeats is not None and int(n_repeats) > 0:
        feats, y = _balance(feats, y, n_classes, int(n_repeats))
        n = int(feats.shape[0])

    max_rank = n_classes - 1
    n_comp = int(min(n_components, max_rank))
    if n_comp <= 0:
        raise RankDeficientError("LDA component count collapsed to zero", n_components=n_components, n_classes=n_classes)

    mean = np.mean(feats, axis=0)

    # --- between-class scatter -----------------------------------------
    sb = np.zeros((d, d), dtype=FLOAT_DTYPE)
    sw = np.zeros((d, d), dtype=FLOAT_DTYPE)
    for c in range(n_classes):
        member = feats[y == c]
        if member.shape[0] == 0:
            continue
        mc = np.mean(member, axis=0)
        dev = member - mc
        sb += member.shape[0] * np.outer(mc - mean, mc - mean)
        sw += dev.T @ dev

    sb /= n
    sw /= n

    # --- shrinkage-regularised whitening --------------------------------
    if shrinkage > 0.0:
        # Ledoit-Wolf style target: a scaled identity carrying the average
        # variance, so the regularisation does not change the overall scale.
        target = (float(np.trace(sw)) / d) * np.eye(d, dtype=FLOAT_DTYPE)
        sw = (1.0 - shrinkage) * sw + shrinkage * target

    # Symmetrise to kill the tiny asymmetry that BLAS gemm can introduce;
    # eigh is only guaranteed to be exact on a symmetric matrix.
    sw = 0.5 * (sw + sw.T)
    sb = 0.5 * (sb + sb.T)

    eigvals, eigvecs = np.linalg.eigh(sw)
    # Ascending eigenvalues from eigh; clip to kill numerical negatives.
    floor = max(var_floor, float(np.max(eigvals)) * 1e-12)
    eigvals = np.maximum(eigvals, floor)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Whitening matrix: S^-1/2 = V diag(lambda^-1/2) V^T.
    inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))[None, :]) @ eigvecs.T
    sb_white = inv_sqrt @ sb @ inv_sqrt
    sb_white = 0.5 * (sb_white + sb_white.T)

    # --- generalised eigenproblem: maximise w' Sb w s.t. w' Sw w = 1 ---
    # After whitening Sw becomes the identity, so this is a plain eigenproblem.
    dvals, dvecs = np.linalg.eigh(sb_white)
    dorder = np.argsort(dvals)[::-1][:n_comp]
    disc = dvals[dorder]
    # Map the whitened directions back to the original feature space.
    W = inv_sqrt @ dvecs[:, dorder]

    # Re-orthonormalise in the original space for numerical hygiene.
    q, _ = np.linalg.qr(W)
    # Match eigenvalues to the (possibly sign-flipped) QR basis by projecting.
    coeffs = q.T @ W
    disc = np.sum((coeffs**2) * disc[None, :], axis=0)

    # Explained-variance ratio: the *total* between-class variance is the sum of
    # the between-scatter eigenvalues.  Using the sum of the retained
    # eigenvalues as the denominator (as an earlier version did) made the ratios
    # sum to > 1 -- a reported "1967% of variance explained" on a 2-component LDA
    # was the visible symptom.
    # The ratios are normalised by the retained sum, then rescaled so they add
    # to at most 1.  ``disc`` comes from the *whitened* generalised eigenproblem
    # and is not directly commensurate with the raw ``Sb`` eigenvalues, so using
    # the raw total as the denominator produced ratios summing to 1.15.
    disc_pos = np.maximum(disc, 0.0)
    disc_sum = float(np.sum(disc_pos))
    if disc_sum > 1e-12:
        ratios = disc_pos / disc_sum
        # Cap at 1 and renormalise: the generalised eigenvalues can exceed the
        # between-scatter trace when Sw is heavily shrunk.
        if float(np.sum(ratios)) > 1.0:
            ratios = ratios / float(np.sum(ratios))
    else:
        ratios = np.zeros_like(disc_pos)

    return LDAProjector(
        mean=mean,
        W=np.ascontiguousarray(q),
        explained_variance_ratio=np.ascontiguousarray(ratios),
        eigenvalues=np.ascontiguousarray(disc),
        n_components=n_comp,
        shrinkage=float(shrinkage),
        n_classes=n_classes,
        n_train_frames=int(n),
    )


@dataclass
class GlobalMean:
    """Global feature mean, fitted on the training split only.

    Used by the weak baseline (``mfcc_cos_nbc``).  It is deliberately *not*
    computed per utterance -- that would be CMVN, which is a different (and
    leakage-free) operation; a global mean must come from the training split.
    """

    mean: np.ndarray
    n_frames: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "mean", np.ascontiguousarray(self.mean, dtype=FLOAT_DTYPE))

    def center(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=FLOAT_DTYPE)
        if arr.ndim == 1:
            return arr - self.mean
        return arr - self.mean[None, :]

    def to_dict(self) -> dict[str, Any]:
        return {"n_frames": self.n_frames, "dim": int(self.mean.shape[0])}


def fit_global_mean(x: np.ndarray) -> GlobalMean:
    """Fit the global mean on the given (training) frames."""
    arr = np.ascontiguousarray(x, dtype=FLOAT_DTYPE)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ShapeError("global mean needs a non-empty 2-D frame matrix", shape=list(arr.shape))
    return GlobalMean(mean=np.mean(arr, axis=0), n_frames=int(arr.shape[0]))


def center(x: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Functional mean-centring helper."""
    arr = np.asarray(x, dtype=FLOAT_DTYPE)
    m = np.asarray(mean, dtype=FLOAT_DTYPE)
    if arr.ndim == 1:
        return arr - m
    return arr - m[None, :]
