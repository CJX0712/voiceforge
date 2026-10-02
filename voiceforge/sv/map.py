"""MAP speaker adaptation of a GMM-UBM (Reynolds & Rose, 1995).

Formulas
--------
With ``n_k`` the occupancy of component ``k`` over the adaptation frames and
``tau`` the relevance factor::

    n'_k  = n_k + tau * N / K
    m'_k  = (n_k * m_k + (tau * N / K) * m^ubm_k) / n'_k
    s'^2_k = (n_k * s^2_k + (tau*N/K) * s^2,ubm_k) / n'_k + (tau*N/K)/(n'_k) * (m^ubm_k - m'_k)(...)^T

The last term of the variance update is the **mean-shift correction**: naively
mixing variances ignores that the adapted mean moved away from the UBM mean, so
MAP-2 adds the extra between-mean spread that the move implies.  MAP-1 omits it
and shares a single variance across components.

.. warning::

   **Specification discrepancy (found while implementing, resolved here).**
   The project task-book states the boundary invariants as
   ``tau -> 0 == UBM`` and ``tau -> infinity == speaker-independent``.  That is
   *inverted* with respect to the very formula the task-book also specifies
   above.  Taking the formula literally::

       lim(tau -> 0)     m'_k = (n_k m_k) / n_k           = m_k      (ML)
       lim(tau -> inf)   m'_k -> (tau N/K) m^ubm_k/(tau N/K) = m^ubm_k  (UBM)

   which is also the standard Reynolds & Rose result: ``tau = 0`` means "trust
   the speaker data completely" (a speaker-independent / ML model of that
   speaker) and ``tau -> infinity`` means "trust the prior completely" (the
   UBM unchanged).

   This module implements **the formula**, because it is unambiguous,
   standard, and is what every published MAP system does.  The tests assert the
   limits that the formula actually has.  If the prose was the intent, the fix
   is a one-line change to the prior weight, not to this implementation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.errors import ShapeError
from ..core.types import FLOAT_DTYPE
from .gmm import GMM, _logsumexp_rows

__all__ = ["map_adapt", "map_statistics", "map_variance"]


def map_statistics(ubm: GMM, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate the MAP sufficient statistics of ``x`` under ``ubm``.

    Returns ``(counts, sums, sums_sq)`` with shapes ``(K,)``, ``(K, D)``,
    ``(K, D)``.
    """
    resp = ubm.posterior(x)  # (N, K)
    counts = np.sum(resp, axis=0)
    sums = resp.T @ x
    sums_sq = resp.T @ (x**2)
    return counts, sums, sums_sq


def map_variance(
    ubm_means: np.ndarray,
    ubm_vars: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sums_sq: np.ndarray,
    tau: float,
    variant: str = "map2",
    min_var: float = 1e-6,
) -> np.ndarray:
    """Diagonal MAP variance update for both MAP-1 and MAP-2."""
    k = ubm_means.shape[0]
    n_total = float(np.sum(counts))
    prior = tau * n_total / k
    n_post = counts + prior

    if variant == "map1":
        # MAP-1: pool the counts/variances across components, then re-split by
        # the posterior counts.  Equivalent to a single shared covariance.
        total_var = (np.sum(counts[:, None] * ubm_vars, axis=0) + prior * np.sum(ubm_vars, axis=0) / k) / max(
            n_total + prior, 1e-12
        )
        return np.maximum(np.tile(total_var, (k, 1)), min_var)

    # --- MAP-2: per-component with the mean-shift correction ------------
    safe_counts = np.maximum(counts, 1e-12)[:, None]
    means = sums / safe_counts
    # Within-component variance from the second moment.
    within = sums_sq / safe_counts - means**2
    np.maximum(within, 0.0, out=within)

    n_post_c = np.maximum(n_post, 1e-12)[:, None]
    # Shift from the UBM mean to the adapted mean, per dimension.
    shift = ubm_means - means
    variance = (counts[:, None] * within + prior * ubm_vars) / n_post_c + (prior / n_post_c) * (shift**2)
    return np.maximum(variance, min_var)


def map_adapt(
    ubm: GMM,
    x: np.ndarray,
    *,
    tau: float = 0.5,
    variant: str = "map2",
    adapt_var: bool = True,
    min_var: float | None = None,
) -> GMM:
    """Adapt ``ubm`` to the frames ``x`` with relevance factor ``tau``.

    Parameters
    ----------
    tau:
        Relevance factor (>= 0).  ``0`` returns a bit-identical **ML** model
        estimated from ``x`` alone; very large values converge to ``ubm``
        unchanged.  See the module docstring for why this is the opposite of
        the boundary semantics described in the task book.
    variant:
        ``"map2"`` (full, with mean-shift correction) or ``"map1"`` (pooled
        single covariance).
    adapt_var:
        When ``False`` the variances stay at the UBM values and only the means
        and weights move -- the classic "MAP means only" configuration.
    """
    if tau < 0.0:
        raise ShapeError("tau must be non-negative", tau=tau)
    if variant not in ("map1", "map2"):
        raise ShapeError(f"unknown MAP variant {variant!r}", variant=variant, known=["map1", "map2"])

    floor = ubm.min_var if min_var is None else float(min_var)

    frames = np.asarray(x, dtype=FLOAT_DTYPE)
    n = frames.shape[0]
    if n == 0:
        raise ShapeError("cannot MAP-adapt on zero frames", tau=tau)

    counts, sums, sums_sq = map_statistics(ubm, frames)
    k = ubm.n_gauss
    prior = tau * n / k
    n_post = counts + prior

    means = (sums + prior * ubm.means) / np.maximum(n_post, 1e-12)[:, None]

    if adapt_var:
        variances = map_variance(
            ubm.means, ubm.variances, counts, sums, sums_sq, tau, variant=variant, min_var=floor
        )
    else:
        variances = ubm.variances.copy()

    # Weights: the same Dirichlet-style update as the means, i.e. the posterior
    # occupancies re-normalised after adding the UBM prior mass ``prior``.
    weights = n_post / float(np.sum(n_post))
    np.maximum(weights, 0.0, out=weights)
    wsum = float(np.sum(weights))
    if wsum > 0:
        weights = weights / wsum

    return GMM(
        weights=weights,
        means=np.ascontiguousarray(means, dtype=FLOAT_DTYPE),
        variances=np.ascontiguousarray(variances, dtype=FLOAT_DTYPE),
        min_var=floor,
        history=(),
        n_iter=0,
        converged=True,
    )


def map_log_likelihood_delta(ubm: GMM, adapted: GMM, x: np.ndarray) -> float:
    """``ll(adapted) - ll(ubm)`` on ``x``; must be >= 0 for ``tau > 0``."""
    return adapted.log_likelihood(x) - ubm.log_likelihood(x)


def joint_log_likelihood(model: GMM, x: np.ndarray) -> np.ndarray:
    """Per-frame log of the mixture density (thin wrapper, used by scoring)."""
    comp = model.log_prob(x)
    log_w = np.log(np.maximum(model.weights, 1e-300))
    return _logsumexp_rows(comp + log_w[None, :])


def map_adapt_many(
    ubm: GMM, frames_list: list[np.ndarray], *, tau: float = 0.5, variant: str = "map2"
) -> list[GMM]:
    """Adapt once per element of ``frames_list`` (convenience for enrollment)."""
    return [map_adapt(ubm, f, tau=tau, variant=variant) for f in frames_list]


def map_summary(ubm: GMM, adapted: GMM) -> dict[str, Any]:
    """Small diagnostic bundle comparing two models (used in the report)."""
    return {
        "n_gauss": ubm.n_gauss,
        "dim": ubm.dim,
        "mean_shift_l2": float(np.mean(np.linalg.norm(adapted.means - ubm.means, axis=1))),
        "max_mean_shift_l2": float(np.max(np.linalg.norm(adapted.means - ubm.means, axis=1))),
        "weight_l1_shift": float(np.sum(np.abs(adapted.weights - ubm.weights))),
        "var_ratio_median": float(np.median(adapted.variances / np.maximum(ubm.variances, 1e-12))),
    }
