"""Probabilistic Linear Discriminant Analysis (single Gaussian, two factors).

The generative model for an i-vector ``x`` is

    x = mu + a * w + b
    w ~ N(0, Sigma_w)        (between / speaker variability)
    b ~ N(0, Sigma_n)        (within / residual variability)
    a ~ N(mu_b, Sigma_b)     (the speaker mean distribution)

Estimation is by EM.  The two posterior moments needed are available in closed
form because the model is a two-factor Gaussian:

    E[w | a] = Sigma_b (Sigma_b + Sigma_n)^-1 a
    E[b | a] = Sigma_n (Sigma_b + Sigma_n)^-1 a

and the M step updates

    Sigma_w = (1/N) sum_i E[w_i w_i^T] + Psi
    Sigma_b = (1/N) sum_i (a_i - mu_b)(a_i - mu_b)^T - Sigma_w
    Sigma_n = (1/N) sum_i E[b_i b_i^T] + Delta

The two additive terms ``Psi``/``Delta`` are the standard regularisers: without
them the EM update can collapse a covariance to a singular matrix and the
inverse in the E step then produces NaNs.

Scoring is a log-likelihood ratio between the "same speaker" hypothesis (the
pair shares a latent ``w``) and the "different speaker" hypothesis:

    LLR = log N(x_e; mu_b, Sigma_b + Sigma_n)
        - log N(x_e; mu_b, Sigma_w + Sigma_b + Sigma_n)
        + 0.5 log|Sigma_n| - 0.5 log|Sigma_w + Sigma_n|

Length normalisation (SLV, Vogt & Kamm 2005) rescales each vector by the
standard deviation along the between-class direction, so that a short
utterance is not penalised for having fewer frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import ModelError, RankDeficientError, ShapeError
from ..core.types import FLOAT_DTYPE

__all__ = ["PldaModel", "fit_plda", "plda_score", "slv_length_norm", "SLV_WEIGHT"]


@dataclass
class PldaModel:
    """A fitted two-factor PLDA model.

    Attributes
    ----------
    mu_b:
        ``(D,)`` mean of the speaker distribution.
    sigma_w:
        ``(D, D)`` between-speaker (total) covariance.
    sigma_n:
        ``(D, D)`` within-speaker (residual) covariance.
    psi, delta:
        Regularisation added to ``sigma_w`` / ``sigma_n`` during EM.
    """

    mu_b: np.ndarray
    sigma_w: np.ndarray
    sigma_n: np.ndarray
    psi: np.ndarray
    delta: np.ndarray
    n_iter: int = 0
    converged: bool = False
    history: tuple[float, ...] = field(default=(), compare=False)
    slv: bool = True
    slv_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mu_b", "sigma_w", "sigma_n", "psi", "delta"):
            arr = np.ascontiguousarray(getattr(self, name), dtype=FLOAT_DTYPE)
            object.__setattr__(self, name, arr)
        d = self.mu_b.shape[0]
        for name in ("sigma_w", "sigma_n", "psi", "delta"):
            arr = getattr(self, name)
            if arr.shape != (d, d):
                raise ShapeError(f"{name} must be ({d}, {d})", shape=list(arr.shape), expected=[d, d])

    @property
    def dim(self) -> int:
        return int(self.mu_b.shape[0])

    def covariances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(sigma_b, sigma_w + sigma_n, sigma_w + sigma_b + sigma_n)``."""
        sigma_b = _safe_psd(self.sigma_w - self.psi - self.delta)
        s_wn = self.sigma_w + self.sigma_n
        s_wbn = sigma_b + s_wn
        return sigma_b, s_wn, s_wbn

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "n_iter": self.n_iter,
            "converged": self.converged,
            "slv": self.slv,
            "slv_weight": self.slv_weight,
            "trace_sigma_w": float(np.trace(self.sigma_w)),
            "trace_sigma_n": float(np.trace(self.sigma_n)),
        }


def _safe_psd(mat: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Project a matrix onto the PSD cone and floor its eigenvalues.

    The EM update ``Sigma_b = Sigma_w - Psi - Delta`` is not guaranteed to stay
    positive definite in finite precision, and an indefinite ``Sigma_b`` makes
    the E step's inverse produce nonsense rather than raise.
    """
    sym = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(sym)
    vals = np.maximum(vals, floor)
    return np.ascontiguousarray((vecs * vals) @ vecs.T)


def _sym_inv(mat: np.ndarray) -> np.ndarray:
    """Inverse of a symmetric positive-definite matrix, with a floor."""
    vals, vecs = np.linalg.eigh(0.5 * (mat + mat.T))
    vals = np.maximum(vals, 1e-10)
    return np.ascontiguousarray((vecs * (1.0 / vals)) @ vecs.T)


def _log_det(mat: np.ndarray) -> float:
    """``log|det|`` of a symmetric PD matrix via its eigenvalues."""
    vals = np.linalg.eigvalsh(0.5 * (mat + mat.T))
    return float(np.sum(np.log(np.maximum(vals, 1e-300))))


def _log_normal_pdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Diagonal of ``log N(x_i; mean, cov)`` for every row of ``x``."""
    inv = _sym_inv(cov)
    d = x - mean[None, :]
    quad = np.einsum("ij,jk,ik->i", d, inv, d)
    return -0.5 * (quad + _log_det(cov) + x.shape[1] * np.log(2.0 * np.pi))


def fit_plda(
    x: np.ndarray,
    *,
    n_iter: int = 40,
    tol: float = 1e-6,
    var_floor: float = 1e-6,
    psi_scale: float = 1e-2,
    delta_scale: float = 1e-2,
    slv: bool = True,
    slv_weight: float = 0.0,
) -> PldaModel:
    """Fit a single-Gaussian two-factor PLDA model by EM.

    Parameters
    ----------
    x:
        ``(N, D)`` training i-vectors.  ``N`` should be comfortably larger than
        ``D``; the between-scatter is a rank-``N-1`` estimate and with ``N <= D``
        it is singular by construction.
    psi_scale, delta_scale:
        Regularisation strength, expressed as a fraction of the initial data
        variance so it scales with the input.
    """
    vecs = np.ascontiguousarray(x, dtype=FLOAT_DTYPE)
    if vecs.ndim != 2:
        raise ShapeError("PLDA training data must be 2-D", shape=list(vecs.shape))
    n, d = vecs.shape
    if n < 2:
        raise ModelError("PLDA needs at least 2 training vectors", n_vectors=n)
    if n <= d:
        raise RankDeficientError(
            "PLDA between-scatter is rank n-1; need more vectors than dimensions",
            n_vectors=n,
            dim=d,
            remedy="reduce plda.dim or provide more training vectors",
        )

    mu_b = np.mean(vecs, axis=0)
    centred = vecs - mu_b[None, :]

    # Initialise: a small fraction of the total scatter for both factors.  A
    # symmetric split is the standard starting point and converges reliably.
    scatter = centred.T @ centred / max(n - 1, 1)
    total = float(np.trace(scatter)) / max(d, 1)
    psi = (psi_scale * total) * np.eye(d, dtype=FLOAT_DTYPE)
    delta = (delta_scale * total) * np.eye(d, dtype=FLOAT_DTYPE)
    sigma_w = _safe_psd(scatter + psi)
    sigma_n = _safe_psd(scatter * 0.5 + delta)

    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, max(int(n_iter), 1) + 1):
        # --- E step: posterior moments of (w, b) given a ------------------
        s_wn = _safe_psd(sigma_w + sigma_n)
        inv = _sym_inv(s_wn)
        # E[w|a] = sigma_w sigma_wn^-1 a ; E[b|a] = sigma_n sigma_wn^-1 a
        m_w = sigma_w @ inv
        m_b = sigma_n @ inv
        e_ww = centred @ m_w.T
        e_bb = centred @ m_b.T
        # Second moments: E[w w^T] = M_w a a^T M_w^T + (sigma_w - M_w sigma_w)
        # with M_w = sigma_w sigma_wn^-1.  Because M_w is symmetric here the
        # term reduces to sigma_w - sigma_w sigma_wn^-1 sigma_w.
        cov_w = _safe_psd(sigma_w - m_w @ sigma_w)
        cov_b = _safe_psd(sigma_n - m_b @ sigma_n)

        # --- M step -------------------------------------------------------
        sigma_w_new = (e_ww.T @ e_ww) / n + cov_w + psi
        sigma_n_new = (e_bb.T @ e_bb) / n + cov_b + delta
        sigma_b = _safe_psd((centred.T @ centred) / n - (sigma_w_new - psi - delta))

        sigma_w = _safe_psd(sigma_w_new, floor=var_floor)
        sigma_n = _safe_psd(sigma_n_new, floor=var_floor)
        mu_b = np.mean(vecs, axis=0)

        s_wn = _safe_psd(sigma_w + sigma_n)
        s_wbn = _safe_psd(sigma_b + s_wn)
        ll = float(
            np.sum(_log_normal_pdf(vecs, mu_b, s_wbn) - _log_normal_pdf(vecs, mu_b, s_wn))
        )
        history.append(ll)
        if len(history) >= 2 and abs(history[-1] - history[-2]) <= tol * max(abs(history[-2]), 1.0):
            converged = True
            break

    model = PldaModel(
        mu_b=mu_b,
        sigma_w=sigma_w,
        sigma_n=sigma_n,
        psi=psi,
        delta=delta,
        n_iter=it,
        converged=converged,
        history=tuple(history),
        slv=slv,
        slv_weight=slv_weight,
    )
    if slv:
        model.slv_weight = slv_length_norm(model)
    return model


def slv_length_norm(model: PldaModel) -> float:
    """Short-length variability normalisation factor (Vogt & Kamm 2005).

    The returned scalar is the standard deviation of the data projected on the
    between-class direction.  Dividing an i-vector by it removes the systematic
    length difference between short and long utterances before scoring.
    """
    sigma_b, _, _ = model.covariances()
    w, vecs = np.linalg.eigh(_safe_psd(sigma_b))
    direction = vecs[:, int(np.argmax(w))]
    return float(np.linalg.norm(direction)) or 1.0


def _apply_slv(v: np.ndarray, weight: float) -> np.ndarray:
    return v / weight if weight > 1e-12 else v


def plda_score(model: PldaModel, enroll: np.ndarray, test: np.ndarray) -> float:
    """Log-likelihood ratio for a single enrollment/test pair.

    Higher means more similar.  The score is antisymmetric up to the constant
    ``0.5 log|Sigma_n| - 0.5 log|Sigma_w + Sigma_n|`` by the symmetry of the
    two-hypothesis formulation.
    """
    a = np.asarray(enroll, dtype=FLOAT_DTYPE).ravel()
    b = np.asarray(test, dtype=FLOAT_DTYPE).ravel()
    if a.shape[0] != model.dim or b.shape[0] != model.dim:
        raise ShapeError(
            "PLDA score input dimension mismatch", expected=model.dim, got=[int(a.shape[0]), int(b.shape[0])]
        )

    if model.slv:
        w = model.slv_weight
        a = _apply_slv(a, w)
        b = _apply_slv(b, w)

    mu = model.mu_b
    sigma_b, s_wn, s_wbn = model.covariances()

    def pair_logpdf(x, y) -> float:
        """log N([x;y]; [mu;mu], Sigma) under a 2D block covariance."""
        d = 2 * model.dim
        mean = np.concatenate([x, y])
        cov = np.zeros((d, d), dtype=FLOAT_DTYPE)
        cov[: model.dim, : model.dim] = s_wn
        cov[model.dim :, model.dim :] = s_wn
        cov[: model.dim, model.dim :] = sigma_b
        cov[model.dim :, : model.dim] = sigma_b
        return float(_log_normal_pdf(mean[None, :], np.concatenate([mu, mu]), cov)[0])

    same = pair_logpdf(a, b)
    diff = float(_log_normal_pdf(a[None, :], mu, s_wbn)[0]) + float(_log_normal_pdf(b[None, :], mu, s_wbn)[0])
    const = 0.5 * _log_det(model.sigma_n) - 0.5 * _log_det(s_wn)
    return float(same - diff + const)
