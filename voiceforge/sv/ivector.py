"""i-vector extraction via total-variability factor analysis (Dehak et al. 2011).

Pipeline
--------
1. **Supervector.**  Stack the MAP-adapted GMM's means and (square-root)
   variances.  The layout order is *fixed by a module constant* so that two
   runs -- and two backends -- always agree on what column means what.
2. **LLR.**  For each frame, compute the log-likelihood ratio vector between the
   speaker-adapted model and the UBM, restricted to the informative components.
3. **Total variability.**  Learn ``A`` such that ``w = A^T * l`` where the
   supervector statistics of ``w`` are maximally correlated, via the standard
   iterative re-estimation.
4. **Nuisance back-channel compensation (NBC).**  Project out the dominant
   *session* directions, estimated from the within-session scatter of the
   training sessions.
5. **CMS + L2.**  Mean-centre and length-normalise so that only relative
   geometry reaches PLDA.

Invariants asserted by the test-suite
-------------------------------------
* ``T`` (the accumulated LLR matrix) has full row rank;
* the output i-vector has unit L2 norm;
* the extraction is **invariant to a permutation of the frames within an
  utterance** -- this is the strongest available check that the statistic is a
  genuine *set* function and not accidentally order-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.errors import RankDeficientError, ShapeError
from ..core.types import FLOAT_DTYPE
from .gmm import GMM

__all__ = [
    "SUPERVECTOR_ORDER",
    "supervector",
    "llr",
    "TotalVariabilityModel",
    "train_total_variability",
    "extract_ivector",
    "NbcProjector",
    "fit_nbc",
    "WccnProjector",
    "fit_wccn",
    "supervector_dim",
]

#: Canonical supervector layout.  ``interleaved`` is the standard ordering:
#: ``[m_1, s_1, m_2, s_2, ...]`` so that a component's mean and its variance sit
#: next to each other.  The constant is exported and hashed into the benchmark
#: so a silent change of layout becomes visible.
SUPERVECTOR_ORDER: str = "interleaved"


def supervector_dim(n_gauss: int, n_features: int, *, order: str = SUPERVECTOR_ORDER) -> int:
    """Dimensionality of the supervector for a given GMM size."""
    return int(n_gauss * 2 * n_features)


def supervector(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    order: str = SUPERVECTOR_ORDER,
    root: bool = True,
) -> np.ndarray:
    """Flatten a diagonal GMM into a supervector.

    Variances enter as **standard deviations** (``root=True``, the Dehak
    convention) rather than raw variances: the raw variances span many orders of
    magnitude and would dominate the total-variability projection purely because
    of their scale, not because they carry speaker information.
    """
    mu = np.asarray(means, dtype=FLOAT_DTYPE)
    var = np.asarray(variances, dtype=FLOAT_DTYPE)
    if mu.shape != var.shape:
        raise ShapeError("means/variances shape mismatch", means=list(mu.shape), variances=list(var.shape))
    sd = np.sqrt(np.maximum(var, 0.0)) if root else var
    k, d = mu.shape
    if order == "interleaved":
        # Interleave per *dimension* within each component: for component k the
        # layout is [m_k,0, m_k,1, ..., m_k,d-1, s_k,0, s_k,1, ...].
        #
        # Note the ravel order matters: mu.ravel() is row-major, i.e. all of
        # component 0's dimensions, then component 1's, ...  Building the output
        # with a flat stride assignment therefore produces
        #   [m_0,0, m_1,0, m_2,0, ..., s_0,0, s_1,0, ...]
        # (means and variances interleaved *across components*), which is NOT
        # the per-component interleave this layout is supposed to express.  The
        # explicit 2-D construction below is what actually interleaves within a
        # component.
        out = np.empty((k, 2 * d), dtype=FLOAT_DTYPE)
        out[:, 0::2] = mu
        out[:, 1::2] = sd
        return out.ravel()
    if order == "blocked":
        return np.concatenate([mu.ravel(), sd.ravel()])
    raise ShapeError(f"unknown supervector order {order!r}", order=order, known=["interleaved", "blocked"])


def center_supervector(
    means: np.ndarray,
    ubm_means: np.ndarray,
    *,
    length_norm: bool = True,
) -> np.ndarray:
    """Centre a MAP supervector on the UBM mean, then L2-normalise.

    This is the Kaldi ``ivector-extract`` step and it is not optional.  Every
    MAP-adapted mean ``mu_k(s)`` contains a large *common* component inherited
    from the UBM; without removing it the cosine between two speakers' vectors
    is dominated by that shared part rather than by the speaker-specific
    residual.  The measured symptom is a within-speaker / between-speaker cosine
    gap of almost nothing -- all speakers end up crowded into the same region
    of the unit sphere.

    Parameters
    ----------
    means:
        ``(K, D)`` MAP-adapted component means.
    ubm_means:
        ``(K, D)`` UBM component means (the *same* ordering as ``means``).
    length_norm:
        Scale to unit L2 norm after centring.

    Returns
    -------
    np.ndarray
        The flattened ``(K * D,)`` centred (and optionally normalised) vector.
    """
    m = np.ascontiguousarray(means, dtype=FLOAT_DTYPE)
    ubm = np.ascontiguousarray(ubm_means, dtype=FLOAT_DTYPE)
    if m.shape != ubm.shape:
        raise ShapeError(
            "supervector centring shape mismatch", means=list(m.shape), ubm_means=list(ubm.shape)
        )
    centred = (m - ubm).ravel()
    if length_norm:
        norm = float(np.linalg.norm(centred))
        if norm > 1e-12:
            centred = centred / norm
    return np.ascontiguousarray(centred, dtype=FLOAT_DTYPE)


def llr(
    adapted: GMM,
    ubm: GMM,
    x: np.ndarray,
    *,
    min_var: float = 1e-6,
    min_count: float = 1e-3,
) -> np.ndarray:
    """Frame-level log-likelihood ratio vector.

    Returns ``(n_frames, D)`` where ``D`` is the supervector dimension of the
    *informative* components.  A component is informative when its UBM weight
    times occupancy exceeds ``min_count``; uninformative components carry no
    speaker information and their LLR is pure estimation noise, so they are
    dropped.  This is the standard Dehak pre-filter and it is what keeps the
    total-variability matrix well conditioned.
    """
    if adapted.n_gauss != ubm.n_gauss or adapted.dim != ubm.dim:
        raise ShapeError(
            "adapted/UBM shape mismatch",
            adapted=[adapted.n_gauss, adapted.dim],
            ubm=[ubm.n_gauss, ubm.dim],
        )
    resp_a = adapted.posterior(x)  # (N, K)
    resp_u = ubm.posterior(x)  # (N, K)

    # Per-component mean log-likelihood under each model.
    var_a = np.maximum(adapted.variances, min_var)
    var_u = np.maximum(ubm.variances, min_var)
    quad_a = np.sum((x[:, None, :] ** 2) / var_a[None, :, :], axis=2)
    quad_u = np.sum((x[:, None, :] ** 2) / var_u[None, :, :], axis=2)

    ll_a = resp_a * (
        -0.5 * quad_a
        - 0.5 * np.sum(np.log(2.0 * np.pi * var_a), axis=1)[None, :]
        + np.log(np.maximum(adapted.weights, 1e-300))[None, :]
    )
    ll_u = resp_u * (
        -0.5 * quad_u
        - 0.5 * np.sum(np.log(2.0 * np.pi * var_u), axis=1)[None, :]
        + np.log(np.maximum(ubm.weights, 1e-300))[None, :]
    )
    # ll_a / ll_u are (N, K): the log-likelihood of frame n under component k.
    # The LLR vector for frame n is the component-wise *difference*, summed over
    # components only where they are informative -- NOT summed over frames.
    #
    # (An earlier version did ``ll_a.sum(axis=0)``, which collapsed the frame
    # axis and returned a single (K,) vector for the whole utterance.  That
    # silently discarded all frame-level structure and made the "invariance to
    # frame permutation" invariant true for the trivial reason that there was
    # nothing left to permute.)
    per_frame = ll_a - ll_u  # (N, K)

    # Informative components: UBM weight x occupancy above a floor.  Components
    # with negligible occupancy carry no speaker information and their LLR is
    # pure estimation noise, so dropping them keeps the total-variability
    # matrix well conditioned.
    #
    # The mask is derived from the **UBM alone**, not from the current
    # utterance.  Deriving it per call makes the output width depend on the
    # data: two utterances of the same speaker can then return 54 and 56
    # columns, and concatenating them fails with an opaque numpy error deep
    # inside the trainer.  A UBM-derived mask is a property of the model, so
    # every utterance yields the same width and the LLR space is consistent.
    occupancy = ubm.weights  # (K,)
    keep = occupancy > min_count
    if not np.any(keep):
        keep = np.ones(ubm.n_gauss, dtype=bool)

    return np.ascontiguousarray(per_frame[:, keep], dtype=FLOAT_DTYPE)


def _stack_llr(llr_list: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-session LLR matrices into the ``(D, T)`` design matrix ``L``.

    Each element of ``llr_list`` is a ``(n_frames_i, D)`` matrix, so stacking
    along **axis 1** concatenates frames, not rows.

    An earlier version called ``.ravel()`` on each session first, which turned
    every ``(n_frames, D)`` block into a single flat vector of length
    ``n_frames * D``.  The resulting ``L`` had ``D' = n_frames * D`` rows, so
    the TV model was fitted in a 960-dimensional space for 8-dimensional LLR
    vectors and every downstream shape check failed.  The ravel is gone; the
    width is validated instead.
    """
    mats = [np.atleast_2d(np.asarray(v, dtype=FLOAT_DTYPE)) for v in llr_list]
    mats = [m for m in mats if m.size > 0]
    if not mats:
        raise RankDeficientError("no LLR vectors to stack", n_vectors=0)
    widths = {int(m.shape[1]) for m in mats}
    if len(widths) != 1:
        raise ShapeError("LLR vectors have inconsistent widths", widths=sorted(widths))
    return np.ascontiguousarray(np.concatenate(mats, axis=0).T)  # (D, T)


@dataclass
class TotalVariabilityModel:
    """The learned TV projection ``A`` plus the statistics needed to apply it."""

    A: np.ndarray  # (D+1, tv_dim)
    mean_llr: np.ndarray  # (D,) global training LLR mean, used for CMS
    keep_mask: np.ndarray  # (K,) bool
    n_gauss: int
    n_features: int
    order: str
    n_iter: int
    tv_reg: float
    history: tuple[float, ...] = field(default=(), compare=False)

    @property
    def n_supervector(self) -> int:
        """Dimensionality of the LLR vector this model consumes.

        ``A`` is stored as ``(D + 1, K)`` -- the ``+1`` is the bias row -- so D
        is ``A.shape[0] - 1``.  Reading ``A.shape[0]`` here without the ``-1``
        makes every downstream shape check fail by exactly one dimension.
        """
        return int(self.A.shape[0] - 1)

    @property
    def tv_dim(self) -> int:
        return int(self.A.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_supervector": self.n_supervector,
            "tv_dim": self.tv_dim,
            "n_gauss": self.n_gauss,
            "n_features": self.n_features,
            "order": self.order,
            "n_kept_components": int(np.sum(self.keep_mask)),
            "n_iter": self.n_iter,
            "tv_reg": self.tv_reg,
        }


def train_total_variability(
    llr_per_session: Sequence[np.ndarray],
    *,
    tv_dim: int = 400,
    n_iter: int = 2,
    tv_reg: float = 1e-6,
    order: str = SUPERVECTOR_ORDER,
) -> TotalVariabilityModel:
    """Train the total-variability projection from per-session LLR matrices.

    Parameters
    ----------
    llr_per_session:
        One ``(n_frames_i, D)`` LLR matrix per training session (speaker).
    tv_dim:
        Requested output dimensionality.
    n_iter:
        Re-estimation iterations.
    tv_reg:
        Ridge added to ``L^T L`` before inversion.

    Notes
    -----
    The estimator follows Dehak et al.: stack the LLR vectors into
    ``L = [l_1 ... l_N 1]^T``, form the target ``Y = [Lambda; 0]^T`` from the
    eigenvalues of the LLR second-moment matrix, and solve

        A = (L^T L + lambda I)^-1 L^T Y

    iterating with ``Lambda`` re-estimated from the current ``A``.  Only the top
    ``tv_dim`` directions of ``Y`` are retained, which is what makes the solve
    tractable: the full target would be ``D x D``.
    """
    L = _stack_llr(llr_per_session)  # (D, T)
    d, t = L.shape
    if t < 2:
        raise RankDeficientError("total variability needs at least 2 LLR columns", n_columns=t)

    n_comp = int(min(tv_dim, d, max(t - 1, 1)))
    if n_comp < 1:
        raise RankDeficientError("TV dimension collapsed to zero", tv_dim=tv_dim, n_supervector=d, n_columns=t)

    # Ridge-regularised Gram matrix of the LLR matrix, L L^T (D, D).
    gram = L @ L.T
    reg = tv_reg * float(np.trace(gram)) / max(d, 1)
    if reg <= 0.0:
        reg = tv_reg
    gram = gram + reg * np.eye(d, dtype=FLOAT_DTYPE)

    try:
        eigvals, eigvecs = np.linalg.eigh(gram)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - numerical failure
        raise RankDeficientError("eigendecomposition of the LLR Gram matrix failed", reason=str(exc)) from exc

    order_idx = np.argsort(eigvals)[::-1][:n_comp]
    lam = np.maximum(eigvals[order_idx], 0.0)
    V = np.ascontiguousarray(eigvecs[:, order_idx])  # (D, K)
    inv_gram = np.linalg.pinv(gram, hermitian=True)  # (D, D)

    # Orientation.  This is easy to get wrong and the failure is a bare matmul
    # error, so it is spelled out once:
    #
    #   L      (D, T)   D supervector rows, T LLR columns
    #   V      (D, K)   K target directions in the SAME (D,) supervector space
    #   Y      (T, K)   the target evaluated at each training column = L^T V
    #   A      (D, K)   = (L L^T)^-1 L Y, and the projection is w = A^T l
    #
    # Both factors of the final product have D rows: ``L @ Y`` contracts the
    # supervector axis.  Writing ``L @ V`` instead (i.e. skipping the ``L^T``
    # that lifts V into column space) raises a core-dimension mismatch as soon
    # as T != D, which is essentially always.
    target_cols = L.T @ V  # (T, K)
    A_full = inv_gram @ (L @ target_cols)  # (D, K)

    # Re-estimation.  ``w = A_full^T L`` is (T, K), so its covariance and
    # eigenvectors live in the K-dimensional projected space, not the
    # D-dimensional supervector space.  The rotated target is therefore pulled
    # back into the D space through ``A_full`` before being re-projected.
    history: list[float] = []
    for _ in range(max(int(n_iter), 1)):
        w = (A_full.T @ L).T  # (T, K)
        cov = (w.T @ w) / max(t, 1)
        cov = 0.5 * (cov + cov.T)
        ceig, cevec = np.linalg.eigh(cov)
        cidx = np.argsort(ceig)[::-1][:n_comp]
        rotated = cevec[:, cidx] * np.sqrt(np.maximum(ceig[cidx], 1e-12))[None, :]  # (K, K)
        target = A_full @ rotated  # (D, K)
        A_full = inv_gram @ (L @ (L.T @ target))  # (D, K)
        history.append(float(np.sum(ceig[cidx])))

    # Append a zero bias row so the augmented design is [L; 1] and the
    # projection ``w = A_aug @ l`` (with the last row acting as the intercept).
    #
    # ``A_full`` is (D, K), so the augmented matrix is (D+1, K) and is built by
    # concatenating *along axis 0*.  No transposition is involved: an earlier
    # version used vstack(...).T, which silently produced (K+1, D) and made
    # ``n_supervector`` off by a factor of K.
    A_aug = np.ascontiguousarray(
        np.concatenate([A_full, np.zeros((1, A_full.shape[1]), dtype=FLOAT_DTYPE)], axis=0)
    )  # (D+1, K)

    return TotalVariabilityModel(
        A=np.ascontiguousarray(A_aug),
        # Global mean over the *training* LLRs.  CMS must subtract this, not
        # the per-utterance mean: subtracting a single utterance's own mean
        # cancels it to exactly zero (see extract_ivector).
        mean_llr=np.ascontiguousarray(np.mean(L, axis=1)),
        keep_mask=np.ones(1, dtype=bool),
        n_gauss=0,
        n_features=0,
        order=order,
        n_iter=len(history),
        tv_reg=float(tv_reg),
        history=tuple(history),
    )


def extract_ivector(
    model: TotalVariabilityModel,
    llr_frames: np.ndarray,
    *,
    cms: bool = True,
    length_norm: bool = True,
) -> np.ndarray:
    """Project frame LLRs into a single normalised i-vector.

    The result is ``(tv_dim,)`` with unit norm when ``length_norm`` is set, which
    makes the downstream PLDA length-invariant.
    """
    l = np.asarray(llr_frames, dtype=FLOAT_DTYPE)
    if l.ndim == 1:
        l = l[None, :]
    if l.shape[1] != model.n_supervector:
        raise ShapeError("LLR width does not match the TV model", expected=model.n_supervector, got=int(l.shape[1]))

    if cms:
        # Cepstral mean subtraction against the GLOBAL training mean.
        #
        # Subtracting the per-utterance mean is the obvious mistake here and it
        # is silent: for a single utterance the offset is its own mean, so the
        # result is identically zero, every i-vector collapses onto the origin,
        # and cosine similarity returns a constant 1.0 -- EER exactly 0.5 with
        # no error raised.  The per-utterance mean carries the *channel*; the
        # quantity that must be removed is the training-set average, which is
        # available as ``model.mean_llr``.
        l = l - model.mean_llr[None, :]

    # Projection.  ``A`` is stored as (D+1, K): the **first** D rows are the
    # linear map and the **last** row is the intercept, so the contraction is
    # ``A[:-1].T @ l + A[-1]`` and not ``A @ l`` -- the latter matches K against
    # D and raises unless the i-vector happens to be square.
    #
    # Using the *mean* LLR (rather than every frame) makes the statistic an
    # average over frames, which is what gives the frame-permutation invariance
    # asserted by the tests.
    w = model.A[:-1, :].T @ np.mean(l, axis=0) + model.A[-1, :]

    if length_norm:
        norm = float(np.linalg.norm(w))
        if norm > 1e-12:
            w = w / norm
    return np.ascontiguousarray(w, dtype=FLOAT_DTYPE)


@dataclass
class NbcProjector:
    """Nuisance back-channel compensation projection ``(I - W^T (W W^T)^-1 W)``."""

    W: np.ndarray  # (D, n_dirs) orthonormal nuisance directions

    @property
    def n_dirs(self) -> int:
        return int(self.W.shape[1])

    def apply(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=FLOAT_DTYPE)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.shape[1] != self.W.shape[0]:
            raise ShapeError("NBC dimension mismatch", expected=int(self.W.shape[0]), got=int(arr.shape[1]))
        # Projection onto the orthogonal complement of the nuisance subspace.
        out = arr - (arr @ self.W) @ self.W.T
        return out[0] if single else out

    def to_dict(self) -> dict[str, Any]:
        return {"n_dirs": self.n_dirs, "dim": int(self.W.shape[0])}


def fit_nbc(session_llr: Sequence[np.ndarray], *, n_dirs: int = 30) -> NbcProjector | None:
    """Estimate the nuisance subspace from the within-session LLR scatter.

    ``session_llr`` holds one ``(n_frames_i, D)`` matrix per session.  The
    within-session scatter captures everything that varies *inside* a session --
    channel, noise, phonetic content -- and its leading eigenvectors are
    removed from the projection.  Returns ``None`` when there are too few
    sessions to estimate a nuisance subspace, so the caller can skip NBC rather
    than fabricate directions.
    """
    if n_dirs <= 0 or len(session_llr) < 2:
        return None
    d = int(np.asarray(session_llr[0]).shape[1])
    n_use = int(min(n_dirs, d, max(d - 1, 1)))
    if n_use < 1:
        return None

    scatter = np.zeros((d, d), dtype=FLOAT_DTYPE)
    total = 0
    for mat in session_llr:
        m = np.asarray(mat, dtype=FLOAT_DTYPE)
        if m.ndim != 2 or m.shape[0] < 2:
            continue
        dev = m - np.mean(m, axis=0, keepdims=True)
        scatter += dev.T @ dev
        total += m.shape[0] - 1
    if total == 0:
        return None

    scatter /= total
    scatter = 0.5 * (scatter + scatter.T)
    eigvals, eigvecs = np.linalg.eigh(scatter)
    idx = np.argsort(eigvals)[::-1][:n_use]
    W = np.ascontiguousarray(eigvecs[:, idx])
    # Re-orthonormalise (eigh already returns orthonormal columns, but the
    # explicit QR guards against accumulated numerical drift).
    W, _ = np.linalg.qr(W)
    return NbcProjector(W=W)


@dataclass
class WccnProjector:
    """Within-class covariance normalisation (WCCN) whitening transform.

    WCCN divides each direction by the *within-speaker* standard deviation, so a
    direction that varies a lot *within* a speaker (channel, phonetic content)
    is shrunk and a direction that is stable within a speaker but differs
    *between* speakers is amplified.  In the i-vector space this is the second
    stage of the standard Dehak front-end (GMM -> LDA -> WCCN -> PLDA) and it is
    what lets PLDA model the two-covariance structure cleanly.  Without it the
    i-vector is dominated by the within-speaker (content) directions and every
    speaker ends up crowded into the same region of the hypersphere.
    """

    T: np.ndarray  # (D, D) whitening matrix, x -> T @ x

    @property
    def dim(self) -> int:
        return int(self.T.shape[0])

    def apply(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=FLOAT_DTYPE)
        single = arr.ndim == 1
        if single:
            arr = arr[None, :]
        if arr.shape[1] != self.dim:
            raise ShapeError("WCCN dimension mismatch", expected=self.dim, got=int(arr.shape[1]))
        out = arr @ self.T
        return out[0] if single else out

    def to_dict(self) -> dict[str, Any]:
        return {"dim": self.dim}


def fit_wccn(
    ivecs: np.ndarray,
    labels: Sequence[Any],
    *,
    floor: float = 1e-8,
) -> WccnProjector | None:
    """Estimate the WCCN whitening transform from within-speaker scatter.

    ``ivecs`` is ``(N, D)`` with one row per *utterance* and ``labels`` the
    corresponding speaker id per row.  Per-utterance vectors are required: if a
    speaker contributes only one vector the within-class scatter is exactly
    zero and there is nothing to whiten, so ``None`` is returned and the caller
    skips WCCN rather than fabricate a transform.
    """
    X = np.ascontiguousarray(ivecs, dtype=FLOAT_DTYPE)
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    y = np.asarray(labels)
    if y.shape[0] != X.shape[0]:
        raise ShapeError("WCCN needs one label per i-vector", n_ivecs=int(X.shape[0]), n_labels=int(y.shape[0]))
    classes = np.unique(y)
    if classes.shape[0] < 2:
        return None

    d = X.shape[1]
    Sw = np.zeros((d, d), dtype=FLOAT_DTYPE)
    total = 0
    for c in classes:
        idx = np.nonzero(y == c)[0]
        if idx.shape[0] < 2:
            # A single utterance per speaker contributes no within-class
            # variance; skip it so a lone speaker cannot zero the whole matrix.
            continue
        dev = X[idx] - np.mean(X[idx], axis=0, keepdims=True)
        Sw += dev.T @ dev
        total += idx.shape[0] - 1
    if total == 0:
        return None

    Sw /= total
    Sw = 0.5 * (Sw + Sw.T)
    vals, vecs = np.linalg.eigh(Sw)
    f = max(floor, float(np.max(vals)) * 1e-12)
    vals = np.maximum(vals, f)
    inv_sqrt = (vecs * (1.0 / np.sqrt(vals))[None, :]) @ vecs.T
    return WccnProjector(T=np.ascontiguousarray(inv_sqrt, dtype=FLOAT_DTYPE))
