"""Metric invariants: EER, AUC, minDCF.

These are the tests that caught two real defects:

* ``min(0.5 * (FPR + FNR))`` as the EER -- its minimum always lands on a
  trivial endpoint, so any system no better than chance scored 0.5 and the
  metric lost all discrimination exactly where it is needed;
* a minDCF that omitted the ``P_NT`` / ``P_T`` prior weights from the detection
  cost, which divided every reported value by ~100 at the default prior.

The four pinned cases below are therefore regression tests, not examples.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from voiceforge.core.errors import EmptyTrialSet, MetricError
from voiceforge.sv.metrics import (
    compute_all,
    compute_auc,
    compute_eer,
    compute_min_dcf,
    equal_error_threshold,
    roc_curve,
)

from _helpers import brute_force_eer

# The four pinned cases: 3 genuine, 3 impostor, chosen so that every branch of
# the crossing logic is exercised (crosses above the diagonal, never crosses,
# crosses exactly at a vertex, and is flat).
PINNED = {
    "A_inverted": (np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.array([1, 1, 1, 0, 0, 0], bool), 1.0, 1.0),
    "B_perfect": (np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), np.array([1, 1, 1, 0, 0, 0], bool), 0.0, 0.0),
    "C_interp": (np.array([3.0, 4.0, 5.0, 1.0, 2.0, 6.0]), np.array([1, 1, 1, 0, 0, 0], bool), 1.0 / 3.0, 1.0),
    "D_alltie": (np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 1, 0, 0, 0], bool), 0.5, 1.0),
}


@pytest.mark.parametrize("name", sorted(PINNED))
def test_eer_pinned_cases(name):
    """EER must equal the FPR at the FPR == FNR crossing, to full precision."""
    scores, labels, expected_eer, _ = PINNED[name]
    got = compute_eer(scores, labels)
    assert got == pytest.approx(expected_eer, abs=1e-12), f"{name}: EER {got!r} != {expected_eer!r}"


@pytest.mark.parametrize("name", sorted(PINNED))
def test_min_dcf_pinned_cases(name):
    """minDCF is the SALT-normalised C_min, so 0 for perfect and 1 for chance."""
    scores, labels, _, expected = PINNED[name]
    got = compute_min_dcf(scores, labels, p_target=0.01)
    assert got == pytest.approx(expected, abs=1e-9), f"{name}: minDCF {got!r} != {expected!r}"


def test_min_dcf_semantic_perfect_is_zero():
    """Genuine scores above impostor scores -> a perfect detector -> 0.

    ``[-1, -2]`` with labels ``[1, 0]``: the genuine trial scored -1 and the
    impostor -2, so the genuine is ranked *higher* and separation is perfect.
    """
    assert compute_min_dcf([-1.0, -2.0], [1, 0], p_target=0.01) == pytest.approx(0.0, abs=1e-12)


def test_min_dcf_semantic_inverted_is_one():
    """Genuine ranked below impostor -> no better than chance -> 1.

    This is the regression that pinned the prior-weight bug: with the priors
    omitted from the detection cost the value came out ~0.01 instead of 1.0.
    """
    assert compute_min_dcf([1.0, 2.0], [1, 0], p_target=0.01) == pytest.approx(1.0, abs=1e-12)


def test_min_dcf_prior_weights_are_in_the_numerator():
    """C_min must be prior-weighted, not the raw unweighted error sum.

    One genuine (score 5) and three impostors (4, 1, 2).  Threshold 5 accepts
    the genuine and rejects every impostor, so ``p_fa = 0``, ``p_miss = 0``,
    ``C = 0`` and ``minDCF = 0``.

    This case cannot by itself distinguish a prior-weighted cost from an
    unweighted one (both are 0 here), which is exactly why the perfect-case
    assertion above exists.  What it *does* pin is the endpoint convention: an
    implementation that only ever considered the two trivial endpoints would
    return 1.0 here.
    """
    scores = np.array([5.0, 4.0, 1.0, 2.0])
    labels = np.array([1, 0, 0, 0], bool)
    got = compute_min_dcf(scores, labels, p_target=0.01)
    assert got == pytest.approx(0.0, abs=1e-12)


def test_min_dcf_uses_the_interior_operating_points():
    """The minimum must be searched over thresholds, not just the endpoints.

    Four genuine (5, 4, 4, 3) and four impostors (3, 2, 2, 1).  The ranges
    overlap at 3, so no threshold is perfect and the optimum is interior:
    threshold 4 costs ``p_miss = 1/4`` and threshold 3 costs ``p_fa = 1/4``,
    giving ``C = 0.2475`` and ``minDCF = 0.2475 / 0.01 = 0.25``.

    This pins the concrete value, so a change that stops the search from
    visiting interior thresholds (and silently falls back to a trivial
    detector) is caught.
    """
    scores = np.array([5.0, 4.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0])
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], bool)
    got = compute_min_dcf(scores, labels, p_target=0.01)
    assert got == pytest.approx(0.25, abs=1e-12)


@pytest.mark.parametrize("seed", range(6))
def test_eer_matches_brute_force(seed):
    """The implementation must agree with exhaustive enumeration."""
    r = np.random.default_rng(seed)
    scores = r.normal(size=300)
    labels = (scores + r.normal(0.0, 0.9, size=300)) > 0.0
    assert compute_eer(scores, labels) == pytest.approx(brute_force_eer(scores, labels), abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_auc_matches_exhaustive(seed):
    """AUC must equal the probability that a genuine outscores an impostor."""
    r = np.random.default_rng(100 + seed)
    scores = r.normal(size=90)
    labels = r.random(90) < 0.5
    if labels.all() or (~labels).all():
        pytest.skip("degenerate label draw")
    pos, neg = scores[labels], scores[~labels]
    exhaustive = np.mean(
        [[1.0 if a > b else 0.5 if a == b else 0.0 for b in neg] for a in pos]
    )
    assert compute_auc(scores, labels) == pytest.approx(exhaustive, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_auc_matches_sklearn_including_ties(seed):
    """Cross-check against scikit-learn, which is what we do *not* depend on.

    If a future scikit-learn release changes its tie convention this test will
    fail -- which is exactly the signal that our own implementation, not the
    dependency, is the thing being tested.
    """
    r = np.random.default_rng(200 + seed)
    scores = np.round(r.normal(size=150), 1)  # rounding forces ties
    labels = r.random(150) < 0.5
    if labels.all() or (~labels).all():
        pytest.skip("degenerate label draw")
    assert compute_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def _is_order_preserving(scores, mapped):
    """True when ``mapped`` induces the same *ranking* as ``scores``.

    The premise of rank-invariance is that the transform is increasing **as a
    function**, which is only observable on sorted inputs -- checking
    ``np.diff(mapped) > 0`` on unsorted scores is meaningless and spuriously
    rejects perfectly good transforms such as ``exp``.  It must also be
    injective, i.e. it may not create new ties.
    """
    order = np.argsort(scores, kind="stable")
    s_sorted = np.asarray(scores)[order]
    m_sorted = np.asarray(mapped)[order]
    return bool(np.all(np.diff(m_sorted) > 0.0))


@pytest.mark.parametrize("transform", [np.exp, lambda v: v * 1000.0, lambda v: np.cbrt(v) + 1e3])
def test_eer_is_invariant_under_increasing_transforms(transform):
    """EER depends only on the score *ranking*, never on its scale."""
    r = np.random.default_rng(7)
    scores = r.normal(size=500)
    labels = r.normal(size=500) > 0.0
    base_eer, base_auc = compute_eer(scores, labels), compute_auc(scores, labels)
    mapped = transform(scores)
    # Guard the premise: invariance only holds for strictly increasing,
    # injective transforms, and asserting that here stops a non-monotone test
    # transform from silently producing a spurious failure (this happened once
    # with (3x+5)^2, which has a minimum at x = -5/3).
    assert _is_order_preserving(scores, mapped), "test transform does not preserve the score ranking"
    assert compute_eer(mapped, labels) == pytest.approx(base_eer, abs=1e-12)
    assert compute_auc(mapped, labels) == pytest.approx(base_auc, abs=1e-12)


def test_roc_has_one_point_per_distinct_score():
    """No interpolated operating points: a threshold must realise every point."""
    r = np.random.default_rng(11)
    scores = np.round(r.normal(size=200), 1)
    labels = r.random(200) < 0.5
    curve = roc_curve(scores, labels)
    # One point per distinct score value, plus the leading (0, 0) point that
    # corresponds to "accept nothing" (threshold = +inf).  No trailing point is
    # added, because the lowest real threshold already realises accept-everything
    # whenever the lowest score is unique.
    n_distinct = len(np.unique(scores))
    assert curve.fpr.shape[0] == n_distinct + 1
    assert np.isinf(curve.thresholds[0]), "first operating point must accept nothing"
    # FPR is non-decreasing and TPR non-decreasing in *threshold* order (lower
    # thresholds accept more trials).  Neither is monotone in the other
    # direction, and -- importantly -- TPR is NOT monotone when scores are tied:
    # a score shared by a genuine and an impostor moves both rates at once, so
    # FPR can rise while TPR falls within one step.
    assert np.all(np.diff(curve.fpr) >= -1e-15), "FPR must be non-decreasing"
    assert np.all(np.diff(curve.tpr) >= -1e-15), "TPR must be non-decreasing"
    assert curve.fpr[0] == 0.0 and curve.tpr[0] == 0.0, "first point accepts nothing"
    assert curve.fpr[-1] == 1.0 and curve.tpr[-1] == 1.0, "last point accepts everything"
    assert np.all((curve.fpr >= 0.0) & (curve.fpr <= 1.0)), "FPR out of range"
    assert np.all((curve.tpr >= 0.0) & (curve.tpr <= 1.0)), "TPR out of range"


def test_eer_threshold_reproduces_the_reported_eer():
    """The reported threshold and the reported EER must describe one point."""
    r = np.random.default_rng(5)
    scores = r.normal(size=400)
    labels = (scores + r.normal(0.0, 0.8, size=400)) > 0.0
    thr = equal_error_threshold(scores, labels)
    accept = scores >= thr
    fpr = np.sum(accept & ~labels) / np.sum(~labels)
    fnr = np.sum(~accept & labels) / np.sum(labels)
    assert fpr == pytest.approx(compute_eer(scores, labels), abs=0.02)
    assert fpr == pytest.approx(fnr, abs=0.02)


def test_eer_is_zero_only_for_perfect_separation():
    """A single swapped label must move the EER off zero."""
    scores = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    labels = np.array([1, 1, 1, 0, 0, 0], bool)
    assert compute_eer(scores, labels) == 0.0
    flipped = labels.copy()
    flipped[0] = False
    assert compute_eer(scores, flipped) > 0.0


@pytest.mark.parametrize(
    "scores, labels",
    [
        (np.array([]), np.array([], bool)),
        (np.array([1.0, 2.0]), np.array([1, 1], bool)),
        (np.array([1.0, 2.0, 3.0]), np.array([0, 0, 0], bool)),
        (np.array([1.0, 2.0, 3.0]), np.array([1, 0], bool)),
    ],
)
def test_degenerate_inputs_raise(scores, labels):
    """Empty / single-class / mismatched inputs must raise, not return a number."""
    with pytest.raises((EmptyTrialSet, MetricError)):
        compute_eer(scores, labels)


def test_non_finite_scores_raise():
    with pytest.raises(MetricError):
        compute_eer(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]), np.array([1, 1, 0, 0, 1, 0], bool))


def test_compute_all_is_self_consistent():
    r = np.random.default_rng(21)
    scores = r.normal(size=300)
    labels = r.normal(size=300) > 0.0
    out = compute_all(scores, labels, p_target=0.01)
    assert out["eer"] == pytest.approx(compute_eer(scores, labels), abs=1e-15)
    assert out["auc"] == pytest.approx(compute_auc(scores, labels), abs=1e-15)
    assert out["min_dcf"] == pytest.approx(compute_min_dcf(scores, labels, 0.01), abs=1e-15)
    assert out["n_genuine"] + out["n_impostor"] == out["n_trials"] == 300
    assert 0.0 <= out["eer"] <= 1.0 and 0.0 <= out["min_dcf"] <= 1.0
