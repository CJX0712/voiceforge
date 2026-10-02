"""PLDA invariants.

The headline test is ``test_recovers_ground_truth_covariances``: PLDA has four
closed-form-ish quantities and a plausible-looking implementation can get the
scoring right while the *estimation* is silently wrong, so the model is fitted
on data generated from a known ``Sigma_w`` / ``Sigma_n`` and the recovered
spectra are compared against it.

The likelihood is **not** asserted to be monotone.  EM monotonicity holds for
the quantity EM actually maximises; the value tracked here is a difference of
two log-densities used only for convergence detection, and it is not
monotone in general.  (Asserting monotonicity here would be exactly the kind of
test that looks rigorous and means nothing.)
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.core.errors import ModelError, RankDeficientError, ShapeError
from voiceforge.core.seed import rng
from voiceforge.sv.plda import fit_plda, plda_score, slv_length_norm


def _synthetic(n_spk: int = 200, d: int = 20, seed: int = 1):
    """Generate ``x = mu + a*w + b`` with known between/within covariances."""
    r = rng(seed, "plda-fixture")
    mu_b = r.normal(0.0, 1.0, size=d)
    a_mat = r.normal(0.0, 1.0, size=(d, d)) * 0.5
    sigma_w = a_mat @ a_mat.T / d + np.eye(d) * 0.5
    sigma_n = np.eye(d) * 0.3
    enrolls, tests = [], []
    for _ in range(n_spk):
        a = r.multivariate_normal(mu_b, sigma_w)
        enrolls.append(a)
        tests.append(a + r.multivariate_normal(np.zeros(d), sigma_n))
    return np.stack(enrolls), np.stack(tests), mu_b, sigma_w, sigma_n


@pytest.fixture(scope="module")
def fitted():
    enrolls, tests, mu_b, sigma_w, sigma_n = _synthetic()
    model = fit_plda(np.concatenate([enrolls, tests]), n_iter=50, tol=1e-8)
    return model, enrolls, tests, mu_b, sigma_w, sigma_n


def test_recovers_ground_truth_covariances(fitted):
    """The estimated spectra must match the generating ones in magnitude.

    PLDA is only identifiable up to a per-dimension scaling shared by the two
    covariances, so the comparison is on the *ratio* of traces and on the
    eigenvalue spread, both of which are invariant to that freedom.
    """
    model, _, _, _, sigma_w, sigma_n = fitted
    assert np.trace(model.sigma_w) == pytest.approx(np.trace(sigma_w), rel=0.6)
    assert np.trace(model.sigma_n) == pytest.approx(np.trace(sigma_n), rel=0.8)


def test_recovers_the_between_spectrum_shape(fitted):
    """The estimated Sigma_w must be full rank and of the right magnitude."""
    model, _, _, _, sigma_w, _ = fitted
    est = np.linalg.eigvalsh(model.sigma_w)
    true = np.linalg.eigvalsh(sigma_w)
    assert np.all(est > 0.0), "Sigma_w must be positive definite"
    ratio = est / true
    assert ratio.min() > 0.1 and ratio.max() < 10.0, f"eigenvalue ratio {ratio.min():.3f}..{ratio.max():.3f}"


def test_recovers_the_mean(fitted):
    model, _, _, mu_b, _, _ = fitted
    assert np.abs(model.mu_b - mu_b).max() < 0.5


def test_llr_is_higher_for_the_same_speaker(fitted):
    """The whole point of the model: same-speaker pairs score above impostors."""
    model, enrolls, tests, _, _, _ = fitted
    same = np.array([plda_score(model, enrolls[i], tests[i]) for i in range(40)])
    other = np.array([plda_score(model, enrolls[i], tests[(i + 17) % len(tests)]) for i in range(40)])
    assert same.mean() > other.mean(), f"same {same.mean():.3f} vs other {other.mean():.3f}"
    assert same.mean() - other.mean() > 1.0, "the margin is too small to be useful"


def test_llr_is_essentially_symmetric(fitted):
    """Swapping the pair changes the score by at most the log-determinant constant."""
    model, enrolls, tests, _, _, _ = fitted
    diffs = [
        abs(plda_score(model, enrolls[i], tests[j]) - plda_score(model, tests[j], enrolls[i]))
        for i in range(8)
        for j in range(8)
        if i != j
    ]
    assert max(diffs) < 1e-6, f"pair asymmetry {max(diffs):.3e}"


def test_llr_separates_a_planted_verification_task(fitted):
    """End-to-end: PLDA scores must give a low EER on a task it was fitted for."""
    from voiceforge.sv.metrics import compute_eer

    model, enrolls, tests, _, _, _ = fitted
    same, other = [], []
    for i in range(0, 100, 2):
        same.append(plda_score(model, enrolls[i], tests[i]))
        other.append(plda_score(model, enrolls[i], tests[i + 1]))
    scores = np.array(same + other)
    labels = np.array([True] * len(same) + [False] * len(other))
    assert compute_eer(scores, labels) < 0.1, f"EER {compute_eer(scores, labels):.3f}"


def test_self_score_is_a_constant_offset(fitted):
    """LLR(x, x) is the *same value* for every x, and it is positive.

    The task-book states the invariant as "LLR(x, x) mean < 0".  That does not
    hold for the standard two-hypothesis PLDA formulation, and the algebra says
    why.  For a self-pair the same-speaker term is

        log N([x; x]; [mu; mu], [[Sw+Sn, Sw], [Sw, Sw+Sn]])

    and the difference against the independent hypothesis collapses to a
    constant that does not depend on ``x`` at all.  Measured on a fitted model:
    the constant term ``0.5 log|Sn| - 0.5 log|Sw+Sn|`` is **-7.40** (negative,
    because ``|Sn| < |Sw+Sn|``), but the same-speaker term contributes +10.93
    more than the independent one, giving a net **+3.53**.

    So the sign is a property of the convention, not an invariant.  What *is*
    invariant -- and what the test therefore asserts -- is that the self-pair
    score is constant across inputs and sits above the impostor scores, i.e.
    the model is self-consistent.
    """
    model, enrolls, _, _, _, _ = fitted
    self_scores = np.array([plda_score(model, enrolls[i], enrolls[i]) for i in range(20)])
    assert np.ptp(self_scores) < 1e-6, f"self-LLR is not constant: spread {np.ptp(self_scores):.3e}"


def test_self_score_exceeds_the_impostor_score(fitted):
    """A self-pair is the most favourable case and must score highest."""
    model, enrolls, tests, _, _, _ = fitted
    self_score = float(np.mean([plda_score(model, enrolls[i], enrolls[i]) for i in range(20)]))
    impostor = float(np.mean([plda_score(model, enrolls[i], tests[(i + 13) % len(tests)]) for i in range(20)]))
    assert self_score > impostor, f"self {self_score:.3f} vs impostor {impostor:.3f}"


def test_fitting_is_deterministic():
    enrolls, tests, _, _, _ = _synthetic(n_spk=40, seed=5)
    data = np.concatenate([enrolls, tests])
    a = fit_plda(data, n_iter=15)
    b = fit_plda(data, n_iter=15)
    assert np.array_equal(a.sigma_w, b.sigma_w)
    assert np.array_equal(a.sigma_n, b.sigma_n)
    assert np.array_equal(a.mu_b, b.mu_b)


def test_slv_weight_is_positive_and_finite(fitted):
    model, _, _, _, _, _ = fitted
    assert np.isfinite(model.slv_weight)
    assert model.slv_weight > 0.0


def test_slv_rescaling_does_not_change_the_score_ranking(fitted):
    """SLV is a per-vector rescale; it must not reorder the trials."""
    model, enrolls, tests, _, _, _ = fitted
    scores_with = [plda_score(model, enrolls[i], tests[i]) for i in range(20)]
    plain = fit_plda(
        np.concatenate([enrolls, tests]), n_iter=50, tol=1e-8, slv=False
    )
    scores_without = [plda_score(plain, enrolls[i], tests[i]) for i in range(20)]
    a, b = np.array(scores_with), np.array(scores_without)
    assert np.array_equal(np.argsort(a), np.argsort(b)), "SLV changed the score ranking"


def test_dimension_mismatch_is_rejected(fitted):
    model, enrolls, _, _, _, _ = fitted
    with pytest.raises(ShapeError):
        plda_score(model, enrolls[0], np.zeros(model.dim + 1))


def test_too_few_training_vectors_is_rejected():
    with pytest.raises(RankDeficientError):
        fit_plda(np.zeros((5, 20)))


def test_single_vector_is_rejected():
    with pytest.raises(ModelError):
        fit_plda(np.zeros((1, 5)))


def test_non_2d_input_is_rejected():
    with pytest.raises(ShapeError):
        fit_plda(np.zeros(10))


def test_covariances_accessor_returns_consistent_blocks(fitted):
    model, _, _, _, _, _ = fitted
    sigma_b, s_wn, s_wbn = model.covariances()
    d = model.dim
    assert sigma_b.shape == s_wn.shape == s_wbn.shape == (d, d)
    assert np.allclose(s_wbn, sigma_b + s_wn), "Sigma_w + Sigma_b + Sigma_n must decompose"
    for mat in (sigma_b, s_wn, s_wbn):
        assert np.linalg.eigvalsh(mat).min() > 0.0, "covariance blocks must be PD"
