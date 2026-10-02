"""Shared pytest fixtures and invariant helpers.

The invariants asserted across this suite are the "hard gold standard" of the
project: each one caught a real defect during development, so none of them may
be weakened or skipped.  The helpers live here so that a test file states
*what* it checks rather than *how*.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make the package importable when running from the repository root without
# an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voiceforge.core.config import load_config  # noqa: E402
from voiceforge.core.seed import rng, set_all  # noqa: E402
from voiceforge.sv.gmm import GMM, fit_gmm  # noqa: E402
from voiceforge.sv.map import map_adapt  # noqa: E402


@pytest.fixture(autouse=True)
def _deterministic_env(monkeypatch):
    """Pin the thread environment for every test.

    Without this, a test that runs after a multi-threaded one can inherit a
    different BLAS reduction order and fail on the last mantissa bit.
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("MKL_CBWR", "COMPATIBLE")
    set_all(12345)


@pytest.fixture
def rng_gen():
    """A named RNG substream, reproducible across runs."""
    return rng(12345, "pytest")


@pytest.fixture
def smoke_config():
    """The cheapest valid configuration, for fast invariant checks."""
    return load_config("smoke", use_env=False)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def brute_force_eer(scores, labels) -> float:
    """Reference EER by exhaustive enumeration of the FPR == FNR crossing.

    Deliberately written as plain Python loops with no vectorisation: its value
    is that it shares no code with the implementation under test, so agreement
    between the two is real evidence rather than a tautology.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    n_gen = int(np.sum(y))
    n_imp = int(np.sum(~y))
    points: list[tuple[float, float]] = []
    for t in sorted(set(s.tolist())):
        acc = s >= t
        fpr = float(np.sum(acc & ~y)) / n_imp
        fnr = float(np.sum(~acc & y)) / n_gen
        points.append((fpr, fnr))
    points.append((1.0, 0.0))  # accept everything
    points.sort()
    best = 1.0
    for i in range(len(points) - 1):
        f1, n1 = points[i]
        f2, n2 = points[i + 1]
        g1, g2 = f1 - n1, f2 - n2
        if g1 <= 0.0 <= g2:
            alpha = 0.0 if g2 == g1 else (0.0 - g1) / (g2 - g1)
            best = min(best, f1 + alpha * (f2 - f1))
    return best


def make_gmm_fixture(k: int = 6, d: int = 5, n_per: int = 150, seed: int = 0):
    """Return ``(frames, model)`` for a well-separated synthetic mixture."""
    r = rng(seed, "gmm-fixture")
    centres = r.normal(0.0, 8.0, size=(k, d))
    frames = np.concatenate([c + r.normal(0.0, 1.0, size=(n_per, d)) for c in centres])
    model = fit_gmm(frames, k, rng_gen=rng(seed + 1, "gmm-init"), n_iter=20, tol=1e-7)
    return frames, model


def make_speaker_sessions(k: int = 6, d: int = 5, n_spk: int = 8, n_per: int = 100, seed: int = 0):
    """Return ``(ubm, sessions)`` where each session is a speaker's frames."""
    r = rng(seed, "sessions")
    ubm_frames = r.normal(0.0, 2.0, size=(400, d))
    ubm = fit_gmm(ubm_frames, k, rng_gen=rng(seed + 1, "ubm"), n_iter=15)
    sessions: list[np.ndarray] = []
    for _ in range(n_spk):
        centre = r.normal(0.0, 0.8, size=d)
        sessions.append(centre + r.normal(0.0, 1.0, size=(n_per, d)))
    return ubm, sessions


__all__ = ["brute_force_eer", "make_gmm_fixture", "make_speaker_sessions"]
