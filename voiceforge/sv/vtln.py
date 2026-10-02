"""Vocal Tract Length Normalisation (VTLN).

VTLN compensates for the fact that two speakers with identical vocal tract
*shape* can have different vocal tract *length*, which stretches or compresses
the whole formant pattern along the frequency axis.  A single scalar warp
factor ``w`` models that::

    f' = f / w

Estimating ``w`` is a maximum-likelihood problem: choose the ``w`` that
maximises the likelihood of the (warped) features under a **frozen** UBM.  The
search is a coarse grid followed by golden-section refinement -- robust, and
(unlike a gradient method) unable to escape the admissible range or stall in a
local optimum.

The UBM used for the search must be a *speaker-independent* model fitted on
training speakers only; using a speaker-adapted model would make the criterion
circular.

Warping without recomputing the FFT
----------------------------------
The expensive part of VTLN is the STFT, and the whole point is to try dozens of
warp factors.  This implementation warps the **log-mel vector** directly in the
filterbank domain by linear interpolation along the mel axis.  The per-channel
displacement is derived from the *centre frequencies* of the filterbank (so
the warp follows the mel scale rather than being a blind index shift), and it
is hard-clamped so an extreme ``w`` cannot shift a channel off the end of the
basis.  Cost is ``O(n_frames * n_mels)`` per candidate instead of a fresh FFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.errors import FeatureError, ShapeError
from ..core.types import FLOAT_DTYPE

__all__ = [
    "VtlnResult",
    "warp_mel",
    "find_warp_factor",
    "VtlnNormaliser",
    "VTLN_MIN",
    "VTLN_MAX",
]

VTLN_MIN = 0.85
VTLN_MAX = 1.15


@dataclass(frozen=True)
class VtlnResult:
    """Outcome of a VTLN search."""

    warp_factor: float
    log_likelihood: float
    n_grid: int
    n_refine: int
    converged: bool
    grid_best: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "warp_factor": self.warp_factor,
            "log_likelihood": self.log_likelihood,
            "n_grid": self.n_grid,
            "n_refine": self.n_refine,
            "converged": self.converged,
            "grid_best": self.grid_best,
        }


def _mel_centres(bank: Any) -> np.ndarray:
    """Centre frequency (Hz) of every mel channel of ``bank``."""
    freqs = np.linspace(0.0, bank.sample_rate / 2.0, bank.n_bins, dtype=FLOAT_DTYPE)
    return freqs[np.asarray(bank.matrix, dtype=FLOAT_DTYPE).argmax(axis=1)]


def warp_mel(log_mel: np.ndarray, warp: float, bank: Any, *, max_shift: int = 40) -> np.ndarray:
    """Warp a log-mel matrix along the frequency axis by ``1/warp``.

    Parameters
    ----------
    log_mel:
        ``(n_frames, n_mels)`` log-mel energies.
    warp:
        Warp factor ``w``; frequencies map as ``f -> f / w``.
    bank:
        The :class:`~voiceforge.sv.frontend.MelBank` that produced ``log_mel``.
    max_shift:
        Hard cap on the per-channel displacement, in mel channels.
    """
    if warp <= 0.0:
        raise FeatureError("warp factor must be positive", warp=warp)
    mel = np.asarray(log_mel, dtype=FLOAT_DTYPE)
    if mel.ndim != 2:
        raise ShapeError("log_mel must be 2-D", shape=list(mel.shape))
    if abs(warp - 1.0) < 1e-12:
        return mel

    centres = _mel_centres(bank)
    n_mels = int(centres.shape[0])
    idx = np.arange(n_mels, dtype=FLOAT_DTYPE)

    # A channel at frequency f reads from f/w in the original spectrum, so the
    # source position is the fractional mel index of f/w.
    src = np.interp(centres / warp, centres, idx)
    shift = np.clip(src - idx, -float(max_shift), float(max_shift))
    lo = np.clip(idx + np.floor(shift), 0, n_mels - 1).astype(np.int64)
    hi = np.clip(lo + 1, 0, n_mels - 1)
    frac = (shift - np.floor(shift))[None, :]

    return np.ascontiguousarray(mel[:, lo] * (1.0 - frac) + mel[:, hi] * frac, dtype=FLOAT_DTYPE)


def find_warp_factor(
    log_mel: np.ndarray,
    ubm: Any,
    bank: Any,
    *,
    n_grid: int = 29,
    w_min: float = VTLN_MIN,
    w_max: float = VTLN_MAX,
    n_refine: int = 12,
) -> VtlnResult:
    """Search the warp factor maximising the UBM log-likelihood.

    The objective is ``sum_n log p_ubm(warp(x_n))`` with the UBM held fixed.
    A coarse grid is evaluated first (so refinement cannot be captured by a
    distant local maximum), then golden-section search refines inside the
    bracket around the grid winner.
    """
    mel = np.asarray(log_mel, dtype=FLOAT_DTYPE)
    if mel.ndim != 2 or mel.shape[0] == 0:
        raise FeatureError("VTLN needs a non-empty 2-D log-mel matrix", shape=list(mel.shape))
    if n_grid < 3:
        raise FeatureError("VTLN grid must have at least 3 points", n_grid=n_grid)
    if not 0.0 < w_min < w_max:
        raise FeatureError("invalid VTLN warp range", w_min=w_min, w_max=w_max)

    def objective(w: float) -> float:
        warped = warp_mel(mel, w, bank)
        value = ubm.log_likelihood(warped)
        return float(value) if np.isfinite(value) else -np.inf

    # --- coarse grid ----------------------------------------------------
    grid = np.linspace(w_min, w_max, int(n_grid))
    lls = np.array([objective(float(w)) for w in grid])
    best_idx = int(np.argmax(lls))
    grid_best = float(grid[best_idx])
    best_ll = float(lls[best_idx])

    # --- golden-section refinement in the bracketing interval -----------
    a = float(grid[max(best_idx - 1, 0)])
    b = float(grid[min(best_idx + 1, grid.shape[0] - 1)])
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = objective(c), objective(d)
    for _ in range(int(n_refine)):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = objective(d)
        if abs(b - a) < 1e-4:
            break
    w_ref = 0.5 * (a + b)
    ll_ref = objective(w_ref)

    if ll_ref > best_ll:
        return VtlnResult(
            warp_factor=float(w_ref), log_likelihood=ll_ref, n_grid=int(n_grid),
            n_refine=int(n_refine), converged=True, grid_best=grid_best,
        )
    return VtlnResult(
        warp_factor=grid_best, log_likelihood=best_ll, n_grid=int(n_grid),
        n_refine=int(n_refine), converged=False, grid_best=grid_best,
    )


class VtlnNormaliser:
    """Stateful VTLN wrapper caching the per-utterance warp factor."""

    def __init__(
        self,
        ubm: Any,
        bank: Any,
        *,
        n_grid: int = 29,
        w_min: float = VTLN_MIN,
        w_max: float = VTLN_MAX,
        n_refine: int = 12,
        enabled: bool = True,
    ) -> None:
        self.ubm = ubm
        self.bank = bank
        self.n_grid = int(n_grid)
        self.w_min = float(w_min)
        self.w_max = float(w_max)
        self.n_refine = int(n_refine)
        self.enabled = bool(enabled)
        self._cache: dict[str, float] = {}

    def warp_for(self, utterance_id: str, log_mel: np.ndarray) -> float:
        """Return (and cache) the warp factor for one utterance."""
        if not self.enabled:
            return 1.0
        if utterance_id in self._cache:
            return self._cache[utterance_id]
        res = find_warp_factor(
            log_mel, self.ubm, self.bank,
            n_grid=self.n_grid, w_min=self.w_min, w_max=self.w_max, n_refine=self.n_refine,
        )
        self._cache[utterance_id] = res.warp_factor
        return res.warp_factor

    def normalise(self, log_mel: np.ndarray, warp: float) -> np.ndarray:
        """Apply a previously computed warp factor."""
        if not self.enabled or abs(warp - 1.0) < 1e-12:
            return np.asarray(log_mel, dtype=FLOAT_DTYPE)
        return warp_mel(log_mel, warp, self.bank)

    def process(self, utterance_id: str, log_mel: np.ndarray) -> np.ndarray:
        """Convenience: search then apply."""
        return self.normalise(log_mel, self.warp_for(utterance_id, log_mel))

    def stats(self) -> dict[str, float]:
        if not self._cache:
            return {"n": 0, "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
        vals = np.array(list(self._cache.values()), dtype=FLOAT_DTYPE)
        return {
            "n": int(vals.shape[0]),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
