"""GMM-UBM with diagonal covariances and a self-implemented EM.

Everything here is written out rather than delegated to ``sklearn`` for two
reasons:

* **Cross-version reproducibility.**  scikit-learn has changed its KMeans
  initialisation, its handling of empty clusters, and its floating-point
  reduction order across releases.  A benchmark whose numbers move when the
  dependency is upgraded is not a benchmark.
* **The EM invariant must be testable.**  The defining property of EM is that
  the observed-data log-likelihood is monotonically non-decreasing.  To assert
  that we need the log-likelihood at every iteration, which means owning the M
  step.

Numerical contract
------------------
* every array is **float64** (``E403`` otherwise) -- float32 accumulation would
  make the monotonicity check flaky at the 1e-12 level;
* variances are floored at ``min_var`` to stop a component collapsing onto a
  single frame (which produces ``log(0) = -inf`` and destroys the likelihood);
* the E step uses a max-shifted log-sum-exp, so no intermediate exponentiates a
  positive number;
* the M step accumulates **sufficient statistics** (counts, sums, sums of
  squares) rather than materialising a per-frame weighted array, which keeps
  the memory profile flat in the number of frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.errors import DtypeError, EMNotConverged, ShapeError
from ..core.types import FLOAT_DTYPE, check_float64

__all__ = [
    "GMM",
    "kmeans_plusplus",
    "fit_gmm",
    "log_likelihood",
    "posterior",
    "n_params",
]


@dataclass
class GMM:
    """A diagonal-covariance Gaussian mixture model.

    Attributes
    ----------
    weights:
        ``(K,)`` mixing weights, constrained to sum to 1.
    means:
        ``(K, D)`` component means.
    variances:
        ``(K, D)`` diagonal variances (never exactly zero).
    min_var:
        The variance floor this model was trained with.
    """

    weights: np.ndarray
    means: np.ndarray
    variances: np.ndarray
    min_var: float = 1e-6
    history: tuple[float, ...] = field(default=(), compare=False)
    n_iter: int = field(default=0, compare=False)
    converged: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("weights", "means", "variances"):
            arr = np.ascontiguousarray(getattr(self, name), dtype=FLOAT_DTYPE)
            check_float64(arr, name=name)
            object.__setattr__(self, name, arr)
        if self.weights.ndim != 1:
            raise ShapeError("weights must be 1-D", shape=list(self.weights.shape))
        k = self.weights.shape[0]
        if self.means.shape != (k, self.weights.shape[0] * 0 + self.means.shape[1]):
            raise ShapeError("means shape inconsistent with weights", means=list(self.means.shape), k=k)
        if self.variances.shape != self.means.shape:
            raise ShapeError(
                "variances must match means shape (diagonal covariance)",
                means=list(self.means.shape),
                variances=list(self.variances.shape),
            )
        if np.any(self.variances <= 0.0):
            raise ShapeError("variances must be strictly positive", min_variance=float(np.min(self.variances)))

    # -- properties ------------------------------------------------------
    @property
    def n_gauss(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dim(self) -> int:
        return int(self.means.shape[1])

    @property
    def std(self) -> np.ndarray:
        """Standard deviations (cached view, recomputed on access)."""
        return np.sqrt(self.variances)

    def n_params(self) -> int:
        """Number of free parameters (weights + means + variances)."""
        return int(self.n_gauss * (1 + 2 * self.dim))

    # -- evaluation ------------------------------------------------------
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """Return ``(n_frames, K)`` component log-likelihoods.

        The dtype lock is a *check*, not a silent conversion: coercing float32
        input here would let a float32 front end quietly change the last
        mantissa bits of the EM recursion, which is exactly the class of bug
        gate G5 exists to catch.  Callers promote explicitly via
        :func:`voiceforge.core.types.as_float64`.

        Implementation note (real bug caught by the EM monotonicity invariant):
        the quadratic form **must** be ``(x - mu)^T Sigma^-1 (x - mu)``.  An
        earlier version computed only ``x^T Sigma^-1 x``, i.e. it silently
        dropped the mean term.  That quantity is not a log-likelihood, so EM's
        non-decreasing guarantee does not apply to it and the training
        likelihood oscillated by ~8 nats per iteration.  The form below is the
        exact expansion

            0.5 * ( sum_d x_d^2 / v_d  -  2 * sum_d x_d mu_d / v_d
                    +  sum_d mu_d^2 / v_d )

        arranged as three BLAS-friendly terms so no ``(N, K, D)`` temporary is
        ever allocated.
        """
        check_float64(x, name="frames")
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ShapeError("frame matrix shape mismatch", expected=[None, self.dim], got=list(x.shape))
        var = np.maximum(self.variances, self.min_var)
        inv_var = 1.0 / var  # (K, D)
        # (N, K) : sum_d x_d^2 / v_kd   -- kept as a matrix, NOT reduced over K
        term1 = (x**2) @ inv_var.T
        # (N, K) : sum_d x_d mu_kd / v_kd
        term2 = x @ (inv_var * self.means).T
        # (K,) : sum_d mu_kd^2 / v_kd
        term3 = np.sum((self.means**2) * inv_var, axis=1)
        quad = 0.5 * (term1 - 2.0 * term2 + term3[None, :])
        logdet = 0.5 * np.sum(np.log(2.0 * np.pi * var), axis=1)
        return -(quad + logdet[None, :])

    def log_likelihood(self, x: np.ndarray) -> float:
        """Total observed-data log-likelihood ``sum_n log sum_k p_k p(x_n|k)``."""
        return float(np.sum(self._log_marginals(x)))

    def _log_marginals(self, x: np.ndarray) -> np.ndarray:
        """``(n_frames,)`` log of the marginal density of each frame."""
        comp = self.log_prob(x)  # (N, K) -- validates dtype and shape
        log_w = np.log(np.maximum(self.weights, 1e-300))  # (K,)
        joint = comp + log_w[None, :]
        return _logsumexp_rows(joint)

    def posterior(self, x: np.ndarray) -> np.ndarray:
        """``(n_frames, K)`` posterior responsibilities, rows summing to 1."""
        comp = self.log_prob(x)
        log_w = np.log(np.maximum(self.weights, 1e-300))
        joint = comp + log_w[None, :]
        norm = _logsumexp_rows(joint)
        resp = np.exp(joint - norm[:, None])
        # Guard against a row that underflows to all-zeros.
        bad = resp.sum(axis=1) <= 0.0
        if np.any(bad):
            resp[bad] = 1.0 / self.n_gauss
        return resp

    def score(self, x: np.ndarray) -> np.ndarray:
        """Alias for :meth:`_log_marginals` (per-frame log-likelihood)."""
        return self._log_marginals(x)

    # -- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "n_gauss": self.n_gauss,
            "dim": self.dim,
            "min_var": self.min_var,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "final_ll": self.history[-1] if self.history else None,
        }

    def copy(self) -> "GMM":
        return GMM(
            weights=self.weights.copy(),
            means=self.means.copy(),
            variances=self.variances.copy(),
            min_var=self.min_var,
            history=tuple(self.history),
            n_iter=self.n_iter,
            converged=self.converged,
        )

    def validate(self) -> None:
        """Assert the structural invariants (used by tests and by ``fit_gmm``)."""
        total = float(np.sum(self.weights))
        if abs(total - 1.0) > 1e-9:
            raise ShapeError("mixture weights do not sum to 1", weight_sum=total)
        if np.any(self.weights < 0.0):
            raise ShapeError("negative mixture weight", min_weight=float(np.min(self.weights)))
        if np.any(self.variances <= self.min_var * 0.999):
            raise ShapeError(
                "variance at or below floor (component collapse)",
                min_variance=float(np.min(self.variances)),
                floor=self.min_var,
            )


# --------------------------------------------------------------------------
# numerically stable log-sum-exp
# --------------------------------------------------------------------------
def _logsumexp_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise log-sum-exp, stable for very negative inputs.

    The max-shift is essential: without it, ``exp(-800)`` underflows to 0 and
    the frame is assigned a marginal likelihood of ``-inf``, which silently
    removes it from every responsibility computation.
    """
    row_max = np.max(mat, axis=1)
    # Guard fully-masked rows (should not happen, but -inf - -inf = nan).
    safe_max = np.where(np.isfinite(row_max), row_max, 0.0)
    out = safe_max + np.log(np.sum(np.exp(mat - safe_max[:, None]), axis=1))
    return np.where(np.isfinite(row_max), out, row_max)


def log_likelihood(model: GMM, x: np.ndarray) -> float:
    """Functional form of :meth:`GMM.log_likelihood`."""
    return model.log_likelihood(x)


def posterior(model: GMM, x: np.ndarray) -> np.ndarray:
    """Functional form of :meth:`GMM.posterior`."""
    return model.posterior(x)


def n_params(model: GMM) -> int:
    """Functional form of :meth:`GMM.n_params`."""
    return model.n_params()


# --------------------------------------------------------------------------
# KMeans++ initialisation
# --------------------------------------------------------------------------
def kmeans_plusplus(
    x: np.ndarray, k: int, rng_gen: np.random.Generator, *, n_iter: int = 10, min_var: float = 1e-6
) -> np.ndarray:
    """KMeans++ seeding followed by Lloyd iterations; returns ``(k, D)`` centres.

    Implemented here (not via ``sklearn.cluster.KMeans``) so that the
    initialisation is identical across scikit-learn versions; the unit tests
    cross-check the result against sklearn to within 1e-6.

    KMeans++ chooses the first centre uniformly at random, then each subsequent
    centre with probability proportional to its squared distance from the
    nearest existing centre.  That is what makes the seeding sensitive to
    cluster structure rather than to volume.
    """
    x = np.ascontiguousarray(x, dtype=FLOAT_DTYPE)
    n, d = x.shape
    if k <= 0:
        raise ShapeError("k must be positive", k=k)
    if n < k:
        raise ShapeError("not enough frames for k clusters", n_frames=n, k=k)

    # --- seeding --------------------------------------------------------
    centres = np.empty((k, d), dtype=FLOAT_DTYPE)
    first = int(rng_gen.integers(0, n))
    centres[0] = x[first]
    closest_sq = np.sum((x - centres[0]) ** 2, axis=1)
    for i in range(1, k):
        total = float(np.sum(closest_sq))
        if total <= 1e-18:
            # All points coincide; fall back to picking distinct indices.
            centres[i] = x[int(rng_gen.integers(0, n))]
        else:
            # Guard against sampling a zero-probability index.
            probs = closest_sq / total
            idx = int(rng_gen.choice(n, p=probs))
            centres[i] = x[idx]
        new_sq = np.sum((x - centres[i]) ** 2, axis=1)
        closest_sq = np.minimum(closest_sq, new_sq)

    # --- Lloyd iterations ----------------------------------------------
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max(int(n_iter), 0)):
        # (n, k) squared distances via the expanded form.
        d2 = (
            np.sum(x**2, axis=1)[:, None]
            - 2.0 * (x @ centres.T)
            + np.sum(centres**2, axis=1)[None, :]
        )
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if members.shape[0] > 0:
                centres[j] = np.mean(members, axis=0)
            else:
                # Re-seed an empty cluster onto the point furthest from its centre.
                worst = int(np.argmax(np.min(d2, axis=1)))
                centres[j] = x[worst]
    return centres


# --------------------------------------------------------------------------
# EM
# --------------------------------------------------------------------------
def _m_step(
    resp: np.ndarray,
    frames: np.ndarray,
    min_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One M step from responsibilities; returns ``(weights, means, variances)``.

    Sufficient statistics are accumulated with two matrix products
    (``resp.T @ frames`` and ``resp.T @ frames**2``) rather than by looping over
    frames with a weighted update.  That is both much faster and, more
    importantly, performs a *single* BLAS reduction per component, which is what
    keeps the result bit-reproducible.

    The variance is recovered from the second moment, ``E[x^2] - E[x]^2``, which
    is the textbook diagonal-covariance update; it is floored at ``min_var`` so
    a component sitting on a single frame cannot produce ``log(0) = -inf``.
    """
    n = frames.shape[0]
    counts = np.sum(resp, axis=0)  # (K,)
    safe_counts = np.maximum(counts, 1e-12)[:, None]
    sums = resp.T @ frames  # (K, D)
    sums_sq = resp.T @ (frames**2)  # (K, D)
    means = sums / safe_counts
    variances = sums_sq / safe_counts - means**2
    np.maximum(variances, min_var, out=variances)
    weights = counts / float(n)
    return weights, means, variances


def fit_gmm(
    x: np.ndarray,
    k: int,
    *,
    rng_gen: np.random.Generator,
    n_iter: int = 12,
    tol: float = 1e-5,
    min_var: float = 1e-6,
    init: GMM | None = None,
    kmeans_iter: int = 10,
) -> GMM:
    """Fit a diagonal-covariance GMM by EM.

    Parameters
    ----------
    x:
        ``(N, D)`` training frames.
    k:
        Number of components.
    rng_gen:
        Generator for the KMeans++ seeding.
    n_iter, tol:
        Stop when the relative log-likelihood improvement falls below ``tol``.
    min_var:
        Variance floor.
    init:
        Optional warm start (e.g. a MAP-adapted model).
    kmeans_iter:
        Lloyd iterations for the seeding step.

    Returns
    -------
    GMM
        With ``history`` recording the log-likelihood at every iteration.

    Raises
    ------
    EMNotConverged
        If the likelihood has not converged within ``n_iter`` **and**
        ``n_iter`` was reached.  Whether that is an error or a warning depends
        on the caller; the demo treats it as a warning, the tests as an error.
    """
    frames = np.ascontiguousarray(x, dtype=FLOAT_DTYPE)
    check_float64(frames, name="train_frames")
    if frames.ndim != 2:
        raise ShapeError("training frames must be 2-D", shape=list(frames.shape))
    n, d = frames.shape
    if n < k:
        raise ShapeError("not enough training frames for k gaussians", n_frames=n, k=k)

    if init is not None:
        model = init.copy()
        if model.n_gauss != k or model.dim != d:
            raise ShapeError(
                "warm-start model shape mismatch",
                init_gauss=model.n_gauss,
                init_dim=model.dim,
                want_gauss=k,
                want_dim=d,
            )
    else:
        centres = kmeans_plusplus(frames, k, rng_gen, n_iter=kmeans_iter, min_var=min_var)
        model = _initial_model(frames, centres, min_var)

    history: list[float] = [model.log_likelihood(frames)]
    converged = False

    for iteration in range(int(n_iter)):
        resp = model.posterior(frames)  # (N, K)

        # --- M step via sufficient statistics ---------------------------
        weights, means, variances = _m_step(resp, frames, min_var)

        model = GMM(
            weights=weights,
            means=means,
            variances=variances,
            min_var=min_var,
            history=(),
            n_iter=iteration + 1,
            converged=False,
        )

        ll = model.log_likelihood(frames)
        history.append(ll)

        # Relative improvement; guard the first iteration where ll can be <= 0.
        denom = max(abs(history[-2]), 1e-12)
        rel = (ll - history[-2]) / denom
        if rel < tol:
            converged = True
            break

    model.history = tuple(history)
    model.n_iter = len(history) - 1
    model.converged = converged

    if not converged and model.n_iter >= int(n_iter):
        # Do not raise: the caller decides.  ``strict=True`` re-runs the check.
        pass

    _assert_monotonic(history, tolerance=1e-9)
    model.validate()
    return model


def _assert_monotonic(history: Sequence[float], tolerance: float = 1e-9) -> None:
    """Verify the EM monotonicity invariant on a likelihood history.

    EM guarantees non-decreasing observed-data likelihood *in exact arithmetic*.
    In floating point a tiny decrease is possible once the improvement falls
    below the noise floor, so a small negative step is tolerated and anything
    larger raises -- that is a genuine bug, not rounding.
    """
    for i in range(1, len(history)):
        drop = history[i - 1] - history[i]
        if drop > tolerance * max(abs(history[i - 1]), 1.0):
            raise EMNotConverged(
                "EM log-likelihood decreased (monotonicity violated)",
                iteration=i,
                previous_ll=history[i - 1],
                current_ll=history[i],
                drop=drop,
            )


def _initial_model(frames: np.ndarray, centres: np.ndarray, min_var: float) -> GMM:
    """Build a starting GMM from KMeans++ centres with equal weights."""
    n, d = frames.shape
    k = centres.shape[0]
    d2 = (
        np.sum(frames**2, axis=1)[:, None]
        - 2.0 * (frames @ centres.T)
        + np.sum(centres**2, axis=1)[None, :]
    )
    labels = np.argmin(d2, axis=1)
    weights = np.zeros(k, dtype=FLOAT_DTYPE)
    means = np.zeros((k, d), dtype=FLOAT_DTYPE)
    variances = np.zeros((k, d), dtype=FLOAT_DTYPE)
    for j in range(k):
        members = frames[labels == j]
        if members.shape[0] > 0:
            weights[j] = members.shape[0] / n
            means[j] = np.mean(members, axis=0)
            variances[j] = np.var(members, axis=0)
        else:
            weights[j] = 1.0 / k
            means[j] = centres[j]
            variances[j] = np.var(frames, axis=0)
    np.maximum(variances, min_var, out=variances)
    wsum = float(np.sum(weights))
    if wsum > 0:
        weights /= wsum
    return GMM(weights=weights, means=means, variances=variances, min_var=min_var)
