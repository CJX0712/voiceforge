"""Reference implementations shared by the test-suite.

These deliberately share no code with the modules under test: the value of a
cross-check is that agreement is evidence rather than a tautology.
"""

from __future__ import annotations

import numpy as np

from voiceforge.core.seed import rng
from voiceforge.sv.gmm import fit_gmm


def brute_force_eer(scores, labels) -> float:
    """Reference EER: minimum FPR over all points where FPR == FNR.

    Plain Python loops sharing no code with ``voiceforge.sv.metrics``, so
    agreement between the two is evidence rather than a tautology.

    Operating points are enumerated in **threshold order** (accept nothing
    first, accept everything last) and consecutive pairs are checked for a sign
    change of ``g = FPR - FNR``.

    Two subtleties, both of which produced wrong reference values while this
    helper was being written:

    * The points must **not** be re-sorted by FPR.  A real ROC contains
      vertical segments (FPR pinned while TPR climbs), so sorting by FPR
      interleaves points that were never adjacent and manufactures crossings
      that no threshold realises -- the reference then disagrees with the
      implementation for the wrong reason.
    * On a vertical segment FPR is pinned and FNR slides past it, so the
      equal-error value is the pinned FPR itself; there is nothing to
      interpolate.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels).astype(bool)
    n_gen = int(np.sum(y))
    n_imp = int(np.sum(~y))

    # Traverse thresholds in DESCENDING order.  A trial is accepted when
    # score >= t, so lowering t admits progressively more trials: the first
    # (highest) threshold accepts almost nothing and the last accepts almost
    # everything.  Iterating ascending (the intuitive `sorted()`) walks the
    # curve backwards -- from accept-everything to accept-nothing -- and makes
    # FPR *decrease*, which fabricates a single spurious crossing and yields a
    # reference EER of 0.498 where the truth is 0.205.
    points: list[tuple[float, float]] = []  # (FPR, FNR) in threshold order
    for t in sorted(set(s.tolist()), reverse=True):
        acc = s >= t
        points.append((float(np.sum(acc & ~y)) / n_imp, float(np.sum(~acc & y)) / n_gen))

    best = 1.0
    for i in range(len(points) - 1):
        f1, n1 = points[i]
        f2, n2 = points[i + 1]
        g1, g2 = f1 - n1, f2 - n2
        if g1 > 0.0 or g2 < 0.0:
            continue  # no sign change on this segment
        if g1 == 0.0:
            best = min(best, f1)
            continue
        if abs(f2 - f1) < 1e-15:
            best = min(best, f2)  # vertical segment: FPR pinned
            continue
        alpha = (0.0 - g1) / (g2 - g1)
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
    """Return ``(ubm, sessions)`` where each session is one speaker's frames."""
    r = rng(seed, "sessions")
    ubm = fit_gmm(r.normal(0.0, 2.0, size=(400, d)), k, rng_gen=rng(seed + 1, "ubm"), n_iter=15)
    sessions = [r.normal(0.0, 0.8, size=d) + r.normal(0.0, 1.0, size=(n_per, d)) for _ in range(n_spk)]
    return ubm, sessions
