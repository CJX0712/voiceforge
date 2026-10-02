"""Detection metrics: ROC, EER, AUC, normalised minDCF.

All four are implemented from first principles.  The reason is
reproducibility, not novelty:

* ``sklearn.metrics.roc_auc_score`` changed its tie handling between versions
  (1.9 in particular), so a benchmark pinned to it silently moves when the
  dependency is bumped;
* ``sklearn.metrics.roc_curve`` returns points on a piecewise-linear ROC, and
  linear interpolation across the *flat* (TPR-constant) segment biases the EER
  -- the operating point we care about sits precisely in that segment;
* this project needs the EER to be **exactly** invariant under any monotone
  transform of the scores, which only a rank-based construction guarantees.

Constructions
-------------
**ROC / EER.**  Scores are ranked once; thresholds are inserted at each distinct
score value, so the curve has one operating point per distinct score and never
an interpolated one.  The EER is the point where ``FPR = 1 - TPR``, located by
``searchsorted`` on the (already sorted) false-alarm and true-positive rates.

**AUC.**  Computed from the Mann-Whitney U statistic with *average ranks for
ties*, which is the only tie convention that keeps AUC invariant under
monotone transforms of the scores (a strictly-increasing transform never creates
ties, so AUC is unchanged; a non-strict one can create them, and average ranks
are the unique tie-consistent choice).

**minDCF.**  SALT 2015 (Martin et al.) normalisation::

    C_norm = (C_min - C_default) / (C_min - C_ideal)

with ``C_min = min_t [ P_miss(target) P_fa(t) + (1 - P_target) P_det(t) ]``
and the ideal detector cost equal to ``P_target``.  The result is comparable
across different trial-set sizes and impostor counts, which a raw minDCF is
not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..core.errors import EmptyTrialSet, MetricError
from ..core.types import FLOAT_DTYPE

__all__ = [
    "ROCCurve",
    "eer_from_roc",
    "roc_curve",
    "compute_eer",
    "compute_auc",
    "compute_min_dcf",
    "detection_error_curve",
    "equal_error_threshold",
    "sigmoid",
]


def _as_arrays(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray):
    """Validate and split scores/labels, raising on the degenerate cases."""
    s = np.asarray(scores, dtype=FLOAT_DTYPE).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    if s.shape[0] != y.shape[0]:
        raise MetricError("scores/labels length mismatch", n_scores=int(s.shape[0]), n_labels=int(y.shape[0]))
    if s.shape[0] == 0:
        raise EmptyTrialSet("cannot compute a metric on zero trials", n_trials=0)
    if not np.all(np.isfinite(s)):
        raise MetricError("non-finite scores present", n_nonfinite=int(np.sum(~np.isfinite(s))))
    n_genuine = int(np.sum(y))
    n_impostor = int(np.sum(~y))
    if n_genuine == 0 or n_impostor == 0:
        raise EmptyTrialSet(
            "both classes must be present", n_genuine=n_genuine, n_impostor=n_impostor, n_trials=int(s.shape[0])
        )
    return s, y, n_genuine, n_impostor


@dataclass(frozen=True)
class ROCCurve:
    """A ROC curve with one operating point per distinct score value."""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    n_genuine: int
    n_impostor: int

    @property
    def eer(self) -> float:
        return eer_from_roc(self.fpr, self.tpr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eer": self.eer,
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "n_points": int(self.fpr.shape[0]),
        }


def roc_curve(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> ROCCurve:
    """Build the ROC curve with an exact operating point per distinct score.

    Convention: a trial is counted as *accepted* when ``score >= threshold``.
    Points are returned in increasing threshold order starting from
    "accept nothing" (FPR = TPR = 0), which is the orientation
    :func:`eer_from_roc` expects.
    """
    s, y, n_genuine, n_impostor = _as_arrays(scores, labels)

    # Sort by descending score; ties are kept adjacent.
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    y_sorted = y[order]

    # Cumulative TP/FP at every position, then keep only the *last* index of each
    # run of equal scores -- that is exactly one point per distinct threshold.
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(~y_sorted)
    distinct = np.nonzero(np.diff(s_sorted))[0]
    idx = np.concatenate([distinct, [s_sorted.shape[0] - 1]])

    tpr = np.concatenate([[0.0], tps[idx] / n_genuine])
    fpr = np.concatenate([[0.0], fps[idx] / n_impostor])
    # Threshold that realises each point: the score at that position.
    thresholds = np.concatenate([[np.inf], s_sorted[idx]])

    return ROCCurve(fpr=fpr, tpr=tpr, thresholds=thresholds, n_genuine=n_genuine, n_impostor=n_impostor)


def eer_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Equal error rate: the common value of FPR and FNR where they are equal.

    Definition
    ----------
    EER is the operating point at which the false acceptance rate equals the
    false rejection rate, ``FPR == FNR``; the reported value is that common
    rate.  Every crossing of the ROC with the diagonal is located and the
    **minimum** over all of them is reported.

    Why the minimum
    ---------------
    A real ROC can cross the diagonal more than once -- the empirical curve
    wiggles, and a vertical segment (several operating points sharing one FPR
    while TPR climbs) is common whenever scores are tied.  Reporting the first
    crossing instead of the best one makes the EER depend on the arbitrary
    ordering of tied scores, which breaks both the ranking invariance and
    agreement with exhaustive enumeration.

    Why not ``min(0.5 * (FPR + FNR))``
    ---------------------------------
    That expression is **wrong** and was the previous implementation here.  Its
    minimum always lands on a *trivial endpoint* -- "accept everything" gives
    ``(1, 0)`` and "reject everything" gives ``(0, 1)``, both scoring ``0.5`` --
    so any system no better than chance reports 0.5 and the metric loses all
    discrimination exactly where it matters.  For 3 genuine / 3 impostor with
    *inverted* scores the only point with FPR == FNR is ``(1, 1)``, so the EER
    is 1.0, but ``min(0.5*(FPR+FNR))`` returns 0.5 from the accept-all endpoint.
    """
    fpr = np.asarray(fpr, dtype=FLOAT_DTYPE)
    tpr = np.asarray(tpr, dtype=FLOAT_DTYPE)
    if fpr.shape != tpr.shape or fpr.size < 2:
        raise MetricError("fpr/tpr shape mismatch or too short", n_fpr=int(fpr.size), n_tpr=int(tpr.size))

    fnr = 1.0 - tpr

    # Traverse in *threshold* order, which is the order roc_curve emits: FPR
    # rises, TPR falls.  Do NOT re-sort by FPR here -- a real ROC contains
    # vertical segments (FPR pinned while TPR climbs), and sorting by FPR
    # interleaves points that were never adjacent, manufacturing crossings
    # that no threshold can realise.  Consecutive points in this order are
    # exactly the pairs of operating points a single threshold move connects.
    f = fpr
    n = fnr
    g = f - n  # <= 0 means FPR < FNR (TPR > 0.5 side)

    best = 1.0
    for i in range(g.shape[0] - 1):
        g_lo, g_hi = float(g[i]), float(g[i + 1])
        if g_lo > 0.0 or g_hi < 0.0:
            continue  # no sign change on this segment
        f_lo, f_hi = float(f[i]), float(f[i + 1])

        if g_lo == 0.0:
            best = min(best, f_lo)
            continue

        if abs(f_hi - f_lo) < 1e-15:
            # Vertical segment: FPR is pinned and FNR slides past it.  The
            # equal-error FPR is the pinned value itself (there is no
            # intermediate FPR to interpolate to).
            best = min(best, f_hi)
            continue

        alpha = (0.0 - g_lo) / (g_hi - g_lo)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        best = min(best, f_lo + alpha * (f_hi - f_lo))

    # A curve that never reaches the diagonal (pathological) reports 1.0.
    return float(min(1.0, max(0.0, best)))


def equal_error_threshold(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> float:
    """Score threshold realising the EER operating point.

    Uses the same crossing bracket as :func:`eer_from_roc`, so the reported
    threshold and the reported EER always describe the same point.
    """
    s, y, _, _ = _as_arrays(scores, labels)
    curve = roc_curve(s, y)
    order = np.argsort(curve.fpr, kind="stable")
    f = curve.fpr[order]
    n = 1.0 - curve.tpr[order]
    thr = curve.thresholds[order]
    g = f - n
    idx = int(np.searchsorted(g, 0.0, side="left"))
    idx = int(np.clip(idx - 1, 0, thr.shape[0] - 1))
    return float(thr[idx])


def compute_eer(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> float:
    """Equal error rate in ``[0, 1]``; lower is better."""
    curve = roc_curve(scores, labels)
    return float(eer_from_roc(curve.fpr, curve.tpr))


def compute_auc(scores: Sequence[float] | np.ndarray, labels: Sequence[bool] | np.ndarray) -> float:
    """Area under the ROC via the tie-corrected Mann-Whitney U statistic.

    ``AUC = (R_positive_sum - n_pos(n_pos+1)/2) / (n_pos * n_neg)`` where the
    rank sum uses **average ranks for ties**.  The result is exactly the
    probability that a randomly chosen genuine outscores a randomly chosen
    impostor, with ties counted as half.
    """
    s, y, n_genuine, n_impostor = _as_arrays(scores, labels)

    order = np.argsort(s, kind="stable")
    s_sorted = s[order]
    ranks = np.empty(s.shape[0], dtype=FLOAT_DTYPE)
    i = 0
    n = s.shape[0]
    while i < n:
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        # Average rank over the tie block [i, j] (1-based ranks).
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1

    rank_sum_positive = float(np.sum(ranks[y[order]]))
    u = rank_sum_positive - n_genuine * (n_genuine + 1) / 2.0
    return float(u / (n_genuine * n_impostor))


def detection_error_curve(
    p_target: float,
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(p_miss, p_fa, c_norm)`` over all thresholds (SALT 2015).

    The prior weights **must** appear inside the detection cost:

        C_det(t) = P_FA(t) * P_NT + P_miss(t) * P_T

    Leaving them out makes every cost smaller by a factor of ``P_NT ~= 1``, and
    since the result is a ratio that factor does not cancel -- it divides the
    final number by ~100 for the default ``P_T = 0.01``.  The reference (best
    trivial detector) cost is ``C_ref = min(P_T, P_NT) = P_T`` for ``P_T < 0.5``.
    """
    if not 0.0 < p_target < 1.0:
        raise MetricError("p_target must be in (0, 1)", p_target=p_target)
    p_nt = 1.0 - p_target
    curve = roc_curve(scores, labels)
    p_miss = 1.0 - curve.tpr
    p_fa = curve.fpr
    c_det = p_fa * p_nt + p_miss * p_target
    c_ref = min(p_target, p_nt)
    c_norm = c_det / c_ref
    return p_miss, p_fa, float(np.min(c_norm))


def compute_min_dcf(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    p_target: float = 0.01,
) -> float:
    """SALT 2015 normalised minimum detection cost.

    ``C_min / C_ref`` with ``C_ref = min(P_T, P_NT)``.  Range: 0 (a perfect
    detector) to 1 (no better than the best trivial detector).  A system at
    chance level scores ~1.0, which is the semantic check the tests assert.
    """
    if not 0.0 < p_target < 1.0:
        raise MetricError("p_target must be in (0, 1)", p_target=p_target)
    _, _, c_norm = detection_error_curve(p_target, scores, labels)
    return float(min(1.0, c_norm))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function (used for score calibration)."""
    x = np.asarray(x, dtype=FLOAT_DTYPE)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def compute_all(
    scores: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    *,
    p_target: float = 0.01,
) -> dict[str, float]:
    """Convenience bundle: EER, AUC, minDCF and the EER threshold in one pass."""
    s, y, n_genuine, n_impostor = _as_arrays(scores, labels)
    curve = roc_curve(s, y)
    eer = float(eer_from_roc(curve.fpr, curve.tpr))
    return {
        "eer": eer,
        "eer_pct": eer * 100.0,
        "auc": compute_auc(s, y),
        "min_dcf": compute_min_dcf(s, y, p_target),
        "threshold": equal_error_threshold(s, y),
        "n_genuine": n_genuine,
        "n_impostor": n_impostor,
        "n_trials": int(s.shape[0]),
    }
