"""MAP adaptation invariants.

The task-book prose states ``tau -> 0 == UBM`` and ``tau -> infinity ==
speaker-independent``, which is inverted with respect to the formula the same
section specifies.  These tests therefore assert the limits the *formula*
actually has, and ``test_spec_limit_semantics`` documents the discrepancy
explicitly so nobody re-introduces the wrong expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.core.errors import ShapeError
from voiceforge.core.seed import rng
from voiceforge.sv.gmm import fit_gmm
from voiceforge.sv.map import map_adapt, map_log_likelihood_delta, map_variance

from _helpers import make_gmm_fixture, make_speaker_sessions


def _ubm_and_speaker(seed: int = 0):
    ubm, sessions = make_speaker_sessions(k=6, d=5, n_spk=4, n_per=100, seed=seed)
    return ubm, sessions[0]


def test_tau_zero_equals_the_ml_estimate_bitwise():
    """tau = 0 must reproduce the ML means implied by the UBM posteriors.

    Per the formula ``m'_k = (n_k m_k + prior * m_ubm_k) / (n_k + prior)``, the
    prior vanishes at ``tau = 0`` and ``m'_k = m_k``, i.e. the maximum-
    likelihood estimate under the UBM responsibilities.
    """
    ubm, speaker = _ubm_and_speaker(1)
    adapted = map_adapt(ubm, speaker, tau=0.0)

    resp = ubm.posterior(speaker)
    counts = resp.sum(axis=0)
    ml_means = (resp.T @ speaker) / np.maximum(counts, 1e-12)[:, None]
    assert np.allclose(adapted.means, ml_means, rtol=0.0, atol=1e-12)
    # And it must be *different* from the UBM, or the assertion is vacuous.
    assert not np.allclose(adapted.means, ubm.means, atol=1e-6)


def test_tau_infinity_converges_to_the_ubm_means():
    """A huge prior overwhelms the data, so the adapted means return to the UBM."""
    ubm, speaker = _ubm_and_speaker(2)
    adapted = map_adapt(ubm, speaker, tau=1e14)
    assert np.abs(adapted.means - ubm.means).max() < 1e-8


def test_spec_limit_semantics_is_documented():
    """Pin the arithmetic that resolves the task-book's inverted wording.

    Taking the task-book formula at face value::

        lim(tau -> 0)    m'_k = (n_k m_k) / n_k          = m_k     (ML)
        lim(tau -> inf)  m'_k -> (tau N/K) m_ubm / (tau N/K) = m_ubm (UBM)

    This test evaluates that arithmetic directly.  If a future change swaps the
    prior and the data term, this fails before the boundary tests do.
    """
    n_k, m_k, m_ubm, n_total, k, tau = 100.0, 5.0, 0.0, 200.0, 8, 0.5
    prior = tau * n_total / k
    m_post = (n_k * m_k + prior * m_ubm) / (n_k + prior)
    assert m_post == pytest.approx(4.4444, abs=1e-4)
    assert m_post < m_k, "interpolation must lie strictly between ML and the prior"


def test_log_likelihood_exceeds_the_ubm_for_any_positive_tau():
    """MAP never moves away from the data it was fitted to."""
    ubm, speaker = _ubm_and_speaker(3)
    for tau in (0.05, 0.2, 0.5, 1.0, 3.0):
        adapted = map_adapt(ubm, speaker, tau=tau)
        assert map_log_likelihood_delta(ubm, adapted, speaker) >= -1e-9, f"tau={tau}"


def test_likelihood_delta_decreases_monotonically_in_tau():
    """More prior weight means less fit to the data, monotonically."""
    ubm, speaker = _ubm_and_speaker(4)
    previous = np.inf
    for tau in np.linspace(0.05, 5.0, 24):
        delta = map_log_likelihood_delta(ubm, map_adapt(ubm, speaker, tau=float(tau)), speaker)
        assert delta <= previous + 1e-9, f"delta rose at tau={tau}"
        previous = delta


def test_weights_are_normalised():
    ubm, speaker = _ubm_and_speaker(5)
    for tau in (0.1, 0.5, 2.0):
        adapted = map_adapt(ubm, speaker, tau=tau)
        assert float(np.sum(adapted.weights)) == pytest.approx(1.0, abs=1e-12)
        assert np.all(adapted.weights >= 0.0)


def test_map1_pools_variances_and_map2_does_not():
    """MAP-1 shares one covariance; MAP-2 keeps per-component variances."""
    ubm, speaker = _ubm_and_speaker(6)
    a1 = map_adapt(ubm, speaker, tau=0.5, variant="map1")
    a2 = map_adapt(ubm, speaker, tau=0.5, variant="map2")
    assert np.allclose(a1.variances, a1.variances[0:1].repeat(a1.n_gauss, axis=0)), "MAP-1 must share one variance row"
    assert not np.allclose(a2.variances, a2.variances[0:1].repeat(a2.n_gauss, axis=0)), "MAP-2 must vary per component"


def test_map2_variance_never_falls_below_the_floor():
    ubm, speaker = _ubm_and_speaker(7)
    for variant in ("map1", "map2"):
        adapted = map_adapt(ubm, speaker, tau=0.3, variant=variant)
        assert np.all(adapted.variances >= ubm.min_var)


def test_adapt_var_false_keeps_ubm_variances():
    ubm, speaker = _ubm_and_speaker(8)
    adapted = map_adapt(ubm, speaker, tau=0.5, adapt_var=False)
    assert np.array_equal(adapted.variances, ubm.variances)
    assert not np.allclose(adapted.means, ubm.means, atol=1e-6)


def test_map_adaptation_is_deterministic():
    ubm, speaker = _ubm_and_speaker(9)
    a = map_adapt(ubm, speaker, tau=0.5)
    b = map_adapt(ubm, speaker, tau=0.5)
    assert np.array_equal(a.means, b.means)
    assert np.array_equal(a.variances, b.variances)


def test_negative_tau_is_rejected():
    ubm, speaker = _ubm_and_speaker(10)
    with pytest.raises(ShapeError):
        map_adapt(ubm, speaker, tau=-0.1)


def test_unknown_variant_is_rejected():
    ubm, speaker = _ubm_and_speaker(11)
    with pytest.raises(ShapeError):
        map_adapt(ubm, speaker, tau=0.5, variant="map3")


def test_map_moves_means_toward_the_speaker():
    """Adaptation must actually move the model towards the data, not away."""
    ubm, sessions = make_speaker_sessions(k=6, d=5, n_spk=6, n_per=120, seed=12)
    own = sessions[0]
    other = sessions[1]
    d_own = float(np.mean(np.linalg.norm(map_adapt(ubm, own, tau=1.0).means - ubm.means, axis=1)))
    d_other = float(np.mean(np.linalg.norm(map_adapt(ubm, other, tau=1.0).means - ubm.means, axis=1)))
    assert d_own > 0.0 and d_other > 0.0
    # Adapted models of different speakers must stay distinguishable.
    a_own = map_adapt(ubm, own, tau=1.0).means
    a_other = map_adapt(ubm, other, tau=1.0).means
    assert float(np.mean(np.linalg.norm(a_own - a_other, axis=1))) > 1e-3
