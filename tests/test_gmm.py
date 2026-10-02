"""GMM-UBM invariants.

The EM monotonicity assertion in this file is not decoration: it caught a real
defect where ``GMM.log_prob`` computed ``x^T Sigma^-1 x`` instead of
``(x - mu)^T Sigma^-1 (x - mu)``.  That quantity is not a log-likelihood, so
EM's non-decreasing guarantee does not apply to it and the training likelihood
oscillated by ~8 nats per iteration.  Monotonicity alone cannot catch an
algebraic error that happens to be monotone, which is why
``test_log_prob_matches_scipy`` exists alongside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.core.errors import DtypeError, EMNotConverged, ShapeError
from voiceforge.core.seed import rng
from voiceforge.sv.gmm import GMM, _logsumexp_rows, fit_gmm, kmeans_plusplus

from _helpers import make_gmm_fixture


def test_log_prob_matches_scipy():
    """Component log-likelihoods must equal scipy's diagonal-normal logpdf."""
    from scipy.stats import multivariate_normal

    weights = np.array([0.3, 0.7])
    means = np.array([[1.0, 2.0], [-1.0, 0.5]])
    variances = np.array([[0.5, 1.5], [2.0, 0.25]])
    model = GMM(weights=weights, means=means, variances=variances)
    x = np.array([[0.5, 1.0], [2.0, -1.0]])

    mine = model.log_prob(x)
    ref = np.array(
        [
            [multivariate_normal.logpdf(x[i], mean=means[k], cov=np.diag(variances[k])) for k in range(2)]
            for i in range(x.shape[0])
        ]
    )
    assert np.abs(mine - ref).max() == pytest.approx(0.0, abs=1e-12)


def test_log_likelihood_matches_manual_mixture():
    """The marginal likelihood must be the log of the weighted mixture."""
    model = GMM(
        weights=np.array([0.4, 0.6]),
        means=np.array([[0.0, 0.0], [1.0, 1.0]]),
        variances=np.array([[1.0, 1.0], [0.5, 0.5]]),
    )
    x = np.array([[0.2, 0.3], [0.9, 1.1], [-0.4, 0.1]])
    expected = float(
        np.sum(
            [
                np.log(sum(model.weights[k] * np.exp(model.log_prob(x)[i, k]) for k in range(2)))
                for i in range(x.shape[0])
            ]
        )
    )
    assert model.log_likelihood(x) == pytest.approx(expected, abs=1e-12)


def test_em_log_likelihood_is_monotonically_non_decreasing():
    """The defining property of EM: the observed likelihood never decreases."""
    frames, model = make_gmm_fixture(k=6, d=5, n_per=120, seed=3)
    history = np.asarray(model.history)
    assert history.shape[0] >= 2, "history must record at least the initial likelihood"
    steps = np.diff(history)
    assert np.all(steps >= -1e-9), f"EM likelihood decreased: min step {steps.min():.3e}"
    assert history[-1] >= history[0], "final likelihood must not be below the initial one"


def test_em_is_deterministic():
    """Same seed, same frames -> bit-identical parameters."""
    frames, _ = make_gmm_fixture(seed=4)
    a = fit_gmm(frames, 5, rng_gen=rng(99, "k"), n_iter=8, tol=1e-6)
    b = fit_gmm(frames, 5, rng_gen=rng(99, "k"), n_iter=8, tol=1e-6)
    assert np.array_equal(a.means, b.means)
    assert np.array_equal(a.variances, b.variances)
    assert np.array_equal(a.weights, b.weights)


def test_weights_sum_to_one_exactly():
    _, model = make_gmm_fixture(seed=5)
    assert float(np.sum(model.weights)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(model.weights >= 0.0)


def test_responsibilities_are_a_valid_distribution():
    frames, model = make_gmm_fixture(seed=6)
    resp = model.posterior(frames)
    assert resp.shape == (frames.shape[0], model.n_gauss)
    assert np.abs(resp.sum(axis=1) - 1.0).max() < 1e-12
    assert np.all(resp >= 0.0)


def test_variances_never_collapse_below_the_floor():
    """A component sitting on one frame would otherwise give log(0) = -inf."""
    r = rng(7, "collapse")
    frames = np.concatenate([r.normal(0.0, 1.0, size=(100, 3)), r.normal(50.0, 1.0, size=(3, 3))])
    model = fit_gmm(frames, 12, rng_gen=rng(8, "k"), n_iter=15, min_var=1e-6)
    assert np.all(model.variances >= 1e-6)
    assert np.all(np.isfinite(model.log_likelihood(frames)))


def test_float32_input_is_rejected():
    """The dtype lock is a check, not a silent conversion."""
    frames, model = make_gmm_fixture(seed=9)
    with pytest.raises(DtypeError):
        model.log_prob(frames.astype(np.float32))
    with pytest.raises(DtypeError):
        model.log_likelihood(frames.astype(np.float32))


def test_kmeans_plusplus_agrees_with_sklearn():
    """Our own KMeans++ must find the same clusters scikit-learn does.

    scikit-learn is used only as a cross-check, never at runtime: its
    initialisation has changed across releases, and a benchmark that moves when
    a dependency is upgraded is not a benchmark.
    """
    from sklearn.cluster import KMeans

    r = rng(10, "km")
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [-8.0, 6.0], [5.0, -8.0]])
    data = np.concatenate([c + r.normal(0.0, 1.0, size=(200, 2)) for c in centres])

    mine = kmeans_plusplus(data, 4, rng(1, "kmeans"), n_iter=10)
    theirs = KMeans(n_clusters=4, init="k-means++", n_init=1, random_state=0, max_iter=10).fit(data).cluster_centers_

    # Compare as sets: cluster labels are arbitrary, so match each reference
    # centre to its nearest counterpart.
    worst = max(
        float(min(np.linalg.norm(theirs[i] - mine[j]) for j in range(4))) for i in range(4)
    )
    assert worst < 1e-6, f"nearest-centre mismatch {worst:.3e}"


def test_kmeans_inertia_is_optimal():
    """Lloyd refinement must not increase the within-cluster sum of squares."""
    r = rng(11, "km2")
    data = np.concatenate([c + r.normal(0.0, 1.0, size=(150, 3)) for c in np.eye(3) * 6.0])

    def inertia(c):
        labels = np.argmin(((data[:, None, :] - c[None, :, :]) ** 2).sum(axis=2), axis=1)
        return float(((data - c[labels]) ** 2).sum())

    initial = data[r.choice(data.shape[0], size=3, replace=False)]
    assert inertia(kmeans_plusplus(data, 3, rng(2, "k"), n_iter=10)) <= inertia(initial) + 1e-9


def test_logsumexp_is_stable_for_extreme_values():
    """Without the max-shift, exp(-800) underflows and the frame is lost."""
    mat = np.array([[-800.0, -801.0], [-1e5, -1e5], [-1e300, -1e300]])
    got = _logsumexp_rows(mat)
    assert np.all(np.isfinite(got))
    expected = np.array([np.log1p(np.exp(-1.0)) - 800.0, np.log(2.0) - 1e5, np.log(2.0) - 1e300])
    assert np.allclose(got, expected, rtol=1e-12, atol=0.0)


def test_shape_errors_are_raised():
    model = GMM(weights=np.array([1.0]), means=np.zeros((1, 3)), variances=np.ones((1, 3)))
    with pytest.raises(ShapeError):
        model.log_prob(np.zeros((5, 4)))
    with pytest.raises(ShapeError):
        fit_gmm(np.zeros((2, 3)), 5, rng_gen=rng(1, "k"))  # fewer frames than components


def test_model_reports_its_parameter_count():
    model = GMM(weights=np.array([0.5, 0.5]), means=np.zeros((2, 4)), variances=np.ones((2, 4)))
    assert model.n_params() == 2 * (1 + 2 * 4)
    assert model.n_gauss == 2 and model.dim == 4


def test_gmm_fit_history_is_monotone_even_with_bad_init():
    """A deliberately poor initialisation must still converge monotonically."""
    r = rng(12, "bad-init")
    frames = np.concatenate([c + r.normal(0.0, 0.5, size=(80, 4)) for c in np.eye(4) * 10.0])
    # Start from a single tight cluster so several components begin degenerate.
    model = fit_gmm(frames, 8, rng_gen=rng(13, "k"), n_iter=20, tol=1e-7, min_var=1e-4)
    steps = np.diff(np.asarray(model.history))
    assert np.all(steps >= -1e-9), f"monotonicity violated: {steps.min():.3e}"
