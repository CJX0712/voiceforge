"""Metric invariants -- the values here are **cross-verified**, not guessed.

Every constant in this file was confirmed by two independent implementations
plus a hand derivation before being written down (see ``docs/architecture.md``,
section "Provenance of pinned constants").  The construction rule for any new
pinned value is:

1. derive it by hand from the semantic definition;
2. implement it twice, independently;
3. confirm the two agree bit-for-bit;
4. only then pin it here.

Consensus alone is not evidence.  A set of vectors can be agreed upon by
everyone and still fail to discriminate the bug it was meant to catch -- see
``test_min_dcf_prior_weighting_is_not_optional`` for a case where four
independently-verified vectors all missed an entire class of bug.

The four reference cases use scores that are *already sorted ascending*, and
labels where 1 = genuine (same speaker).  The score convention throughout the
package is **larger = more similar**.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.sv.metrics import (
    compute_auc,
    compute_eer,
    compute_min_dcf,
    detection_error_curve,
    equal_error_threshold,
    roc_curve,
)

# name -> (scores, labels, expected_eer, expected_auc, expected_min_dcf)
CASES = {
    # A fully inverted detector: every impostor outranks every genuine.  The
    # ROC never crosses the diagonal, so the EER must saturate at 1.0.  This is
    # the single most discriminating case: it is the one that separates the
    # correct interpolation from the tempting ``min(0.5*(FPR+FNR))`` shortcut.
    "A_inverted": ([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0], 1.0, 0.0, 1.0),
    # Perfect separation.
    "B_perfect": ([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1], 0.0, 1.0, 0.0),
    # The only case that exercises the *interior* of a segment: Pmiss sits at
    # 1/3 while Pfa climbs 0 -> 0.75, so the crossing is strictly interior.
    "C_interp": ([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2], [1, 1, 0, 0, 0, 0, 1], 1 / 3, 2 / 3, 1 / 3),
    # All scores tied.  Only (0,1) and (1,0) are reachable, so the crossing is
    # the midpoint of a single segment.
    "D_alltie": ([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0], 0.5, 0.5, 1.0),
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_pinned_metric_values(name: str) -> None:
    """Each case pins EER, AUC and minDCF simultaneously."""
    scores, labels, exp_eer, exp_auc, exp_dcf = CASES[name]
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    assert compute_eer(s, y) == pytest.approx(exp_eer, abs=1e-12), "EER"
    assert compute_auc(s, y) == pytest.approx(exp_auc, abs=1e-12), "AUC"
    assert compute_min_dcf(s, y) == pytest.approx(exp_dcf, abs=1e-12), "minDCF"


def test_metric_self_consistency() -> None:
    """EER ~ 1.0 <=> AUC == 0.0 <=> minDCF == 1.0 for an inverted detector.

    Catches any directional error that the individual values would not: if a
    future change flips the score convention, all three move together and this
    invariant is what notices.
    """
    s = np.asarray([0.1, 0.2, 0.3, 0.4])
    y = np.asarray([1, 1, 0, 0], dtype=bool)
    assert compute_eer(s, y) == pytest.approx(1.0, abs=1e-12)
    assert compute_auc(s, y) == pytest.approx(0.0, abs=1e-12)
    assert compute_min_dcf(s, y) == pytest.approx(1.0, abs=1e-12)


def test_min_dcf_trivial_system_is_exactly_one() -> None:
    """The normalisation's defining property.

    Dividing by ``C_ref = min(P_T, P_NT)`` is only correct if it makes the
    better trivial detector score exactly 1.0.  This checks the *definition*
    rather than any particular number, and therefore catches both a missing
    prior weight and wrong ROC end points.

    Score convention: **larger = more similar**, so a perfect detector has the
    genuine trial scoring *higher* than the impostor one.
    """
    # genuine scores 2.0 > impostor 1.0  ->  perfect, cost 0.
    perfect = compute_min_dcf([2.0, 1.0], [True, False])
    assert perfect == pytest.approx(0.0, abs=1e-12)

    # A detector with no usable signal.  Both trivial strategies are available,
    # so the normalisation must land on exactly 1.0.
    uninformative = compute_min_dcf([1.0, 1.0], [True, False])
    assert uninformative == pytest.approx(1.0, abs=1e-12)

    # Fully inverted: the ROC never crosses the diagonal, and the only way out
    # is the better trivial strategy -- again exactly 1.0.
    inverted = compute_min_dcf([1.0, 2.0], [True, False])
    assert inverted == pytest.approx(1.0, abs=1e-12)

    # And a genuine trial set: the better trivial system is "reject all".
    s = np.asarray([0.1, 0.2, 0.3, 0.4])
    y = np.asarray([1, 1, 0, 0], dtype=bool)
    assert compute_min_dcf(s, y) == pytest.approx(1.0, abs=1e-12)
    assert compute_min_dcf(s, np.asarray([0, 0, 1, 1], dtype=bool)) == pytest.approx(0.0, abs=1e-12)


def test_min_dcf_prior_weighting_is_not_optional() -> None:
    """A fifth vector, added because the four above cannot see this bug.

    In every original case the minimum-cost threshold happens to sit at
    ``P_fa = 0``, where the prior weight ``P_NT`` multiplies zero and is
    therefore invisible.  A build that drops the priors still returns 1.0 on
    all four.  Here the optimum is at ``P_fa = 0.5``, so omitting the prior
    yields 0.5 instead of 1.0.
    """
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = np.asarray([0, 0, 1, 1, 0, 0], dtype=bool)  # n_genuine=2, n_impostor=4
    assert compute_min_dcf(scores, labels) == pytest.approx(1.0, abs=1e-12)


def test_eer_domain_is_unit_interval() -> None:
    """EER lives in [0, 1]; asserting <= 0.5 would misreport a reversed system."""
    s = np.asarray([0.1, 0.2, 0.3, 0.4])
    for labels in ([1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]):
        eer = compute_eer(s, np.asarray(labels, dtype=bool))
        assert 0.0 <= eer <= 1.0 + 1e-12


def test_eer_is_invariant_under_monotone_transform() -> None:
    """A strictly increasing transform must not move the EER.

    Note the guard: the transform itself is asserted to be strictly increasing
    first.  An earlier version of this test used ``(3x+5)**2``, which is *not*
    monotone on the negative axis -- the test was wrong, not the code.
    """
    rng = np.random.default_rng(0)
    s = rng.normal(size=400)
    y = rng.random(400) < 0.3
    base = compute_eer(s, y)

    for transform in (lambda v: np.exp(v), lambda v: v * 1000.0, lambda v: v**3):
        transformed = transform(s)
        # Verify the transform is strictly increasing over the observed range.
        probe = np.linspace(float(s.min()), float(s.max()), 256)
        assert np.all(np.diff(transform(probe)) > 0), "transform must be strictly increasing"
        assert compute_eer(transformed, y) == pytest.approx(base, abs=1e-12)


def test_auc_is_invariant_under_monotone_transform() -> None:
    """AUC is rank-based, so any strictly increasing transform preserves it."""
    rng = np.random.default_rng(1)
    s = rng.normal(size=400)
    y = rng.random(400) < 0.3
    base = compute_auc(s, y)
    assert compute_auc(np.exp(s), y) == pytest.approx(base, abs=1e-12)
    assert compute_auc(s * 1000.0, y) == pytest.approx(base, abs=1e-12)


def test_auc_tie_convention_matches_scipy() -> None:
    """Average ranks for ties -- cross-checked against SciPy when available."""
    s = np.asarray([0.5, 0.5, 0.5, 0.1])
    y = np.asarray([1, 0, 1, 0], dtype=bool)
    try:
        from scipy.stats import rankdata
    except ImportError:  # pragma: no cover - SciPy is a Tier-0 extra
        pytest.skip("SciPy unavailable")
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    expected = (rankdata(s)[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    assert compute_auc(s, y) == pytest.approx(float(expected), abs=1e-12)


def test_roc_endpoints_are_reject_all_and_accept_all() -> None:
    """The curve must start at (0,1) and end at (1,0) in (Pfa, Pmiss)."""
    s = np.asarray([0.1, 0.2, 0.3, 0.4])
    y = np.asarray([1, 0, 1, 0], dtype=bool)
    curve = roc_curve(s, y)
    pfa, pmiss = curve.fpr, 1.0 - curve.tpr
    assert pfa[0] == 0.0 and pmiss[0] == 1.0, "must start at reject-all"
    assert pfa[-1] == 1.0 and pmiss[-1] == 0.0, "must end at accept-all"
    # Pfa is non-decreasing and Pmiss is non-increasing along the curve.
    assert np.all(np.diff(pfa) >= -1e-15)
    assert np.all(np.diff(pmiss) <= 1e-15)


def test_equal_error_threshold_agrees_with_eer() -> None:
    """The reported threshold and the reported EER describe the same point."""
    rng = np.random.default_rng(2)
    for _ in range(20):
        s = rng.normal(size=200)
        y = rng.random(200) < 0.4
        thr = equal_error_threshold(s, y)
        curve = roc_curve(s, y)
        # The realised operating point at that threshold must have Pfa ~ Pmiss.
        pfa = float((s[y == 0] >= thr).sum()) / int((y == 0).sum())
        pmiss = float((s[y == 1] < thr).sum()) / int((y == 1).sum())
        assert abs(pfa - pmiss) < 0.15 or thr == curve.thresholds[-1]


def test_min_dcf_detection_error_curve_agrees() -> None:
    """detection_error_curve and compute_min_dcf must not drift apart."""
    s = np.asarray([0.1, 0.2, 0.3, 0.4])
    y = np.asarray([0, 0, 1, 1], dtype=bool)
    _, _, from_curve = detection_error_curve(0.01, s, y)
    assert from_curve == pytest.approx(compute_min_dcf(s, y), abs=1e-12)
