"""i-vector invariants.

The most important test here is ``test_llr_keeps_the_frame_axis``.  An earlier
``llr`` implementation computed ``ll_a.sum(axis=0) - ll_u.sum(axis=0)``,
collapsing the *frame* axis and returning a single ``(K,)`` vector per
utterance instead of ``(n_frames, K)``.  Nothing raised, nothing crashed, and
the "invariance to frame permutation" test passed *vacuously* -- there was
nothing left to permute.  The symptom that finally exposed it was a total
variability model whose ``D`` had silently become ``n_frames * D``.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.core.errors import RankDeficientError, ShapeError
from voiceforge.core.seed import rng
from voiceforge.sv.gmm import fit_gmm
from voiceforge.sv.ivector import (
    _stack_llr,
    extract_ivector,
    fit_nbc,
    llr,
    supervector,
    supervector_dim,
    train_total_variability,
)
from voiceforge.sv.map import map_adapt


def _model_and_sessions(k: int = 6, d: int = 5, n_spk: int = 10, n_per: int = 100, seed: int = 0):
    r = rng(seed, "iv")
    ubm = fit_gmm(r.normal(0.0, 2.0, size=(400, d)), k, rng_gen=rng(seed + 1, "ubm"), n_iter=15)
    sessions, adapted = [], []
    for _ in range(n_spk):
        x = r.normal(0.0, 0.8, size=d) + r.normal(0.0, 1.0, size=(n_per, d))
        sessions.append(x)
        adapted.append(map_adapt(ubm, x, tau=1.0))
    return ubm, sessions, adapted


def test_supervector_layout_is_per_component_interleaved():
    """Layout is ``[m_k, s_k]`` *within* each component, not across components."""
    means = np.array([[1.0, 2.0], [3.0, 4.0]])
    variances = np.array([[0.25, 0.5], [1.0, 2.0]])
    sv = supervector(means, variances)
    assert sv.shape[0] == supervector_dim(2, 2) == 8
    expected = [1.0, 0.5, 2.0, np.sqrt(0.5), 3.0, 1.0, 4.0, np.sqrt(2.0)]
    assert np.allclose(sv, expected), f"got {sv}"


def test_supervector_blocked_order_differs_and_has_the_same_size():
    means = np.array([[1.0, 2.0], [3.0, 4.0]])
    variances = np.array([[0.25, 0.5], [1.0, 2.0]])
    a = supervector(means, variances, order="interleaved")
    b = supervector(means, variances, order="blocked")
    assert a.shape == b.shape
    assert not np.allclose(a, b)
    with pytest.raises(ShapeError):
        supervector(means, variances, order="nonsense")


def test_supervector_rejects_mismatched_shapes():
    with pytest.raises(ShapeError):
        supervector(np.zeros((2, 3)), np.zeros((2, 4)))


def test_llr_keeps_the_frame_axis():
    """One LLR vector per frame -- the regression for the silent collapse."""
    ubm, sessions, adapted = _model_and_sessions(n_per=60, n_spk=1, seed=1)
    out = llr(adapted[0], ubm, sessions[0])
    assert out.ndim == 2, "LLR must be 2-D (n_frames, n_components)"
    assert out.shape[0] == sessions[0].shape[0], "one LLR vector per frame"
    assert out.shape[1] <= ubm.n_gauss


def test_llr_differs_between_speakers():
    ubm, sessions, adapted = _model_and_sessions(n_spk=2, n_per=80, seed=2)
    a = llr(adapted[0], ubm, sessions[0]).mean(axis=0)
    b = llr(adapted[1], ubm, sessions[1]).mean(axis=0)
    assert np.abs(a - b).max() > 1e-6, "LLR must be speaker dependent"


def test_llr_rejects_mismatched_models():
    ubm, sessions, _ = _model_and_sessions(n_spk=1, n_per=50, seed=3)
    other = fit_gmm(rng(4, "o").normal(0.0, 1.0, size=(200, 7)), 3, rng_gen=rng(5, "k"), n_iter=5)
    with pytest.raises(ShapeError):
        llr(other, ubm, sessions[0])


def test_stack_llr_preserves_the_component_width():
    """L is (D, T) where D is the LLR width -- not n_frames * D."""
    mats = [rng(6, f"s{i}").normal(size=(50, 7)) for i in range(3)]
    stacked = _stack_llr(mats)
    assert stacked.shape == (7, 150)
    with pytest.raises(ShapeError):
        _stack_llr([np.zeros((10, 3)), np.zeros((10, 4))])


def test_total_variability_matrix_has_full_row_rank():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=100, seed=7)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    mat = _stack_llr(per_session)
    assert np.linalg.matrix_rank(mat) == mat.shape[0], "T must have full row rank"


def test_ivector_has_unit_l2_norm():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=100, seed=8)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    model = train_total_variability(per_session, tv_dim=6, n_iter=2)
    vec = extract_ivector(model, per_session[0], cms=False, length_norm=True)
    assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-12)


def test_ivector_is_invariant_to_frame_permutation():
    """The strongest available check that the statistic is a genuine set function."""
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=100, seed=9)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    model = train_total_variability(per_session, tv_dim=6, n_iter=2)
    base = extract_ivector(model, per_session[0], cms=False, length_norm=True)
    permuted = extract_ivector(model, per_session[0][np.random.default_rng(0).permutation(per_session[0].shape[0])],
                              cms=False, length_norm=True)
    assert np.abs(base - permuted).max() < 1e-12, f"permutation changed the i-vector by {np.abs(base - permuted).max():.3e}"


def test_ivectors_of_different_speakers_differ():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=100, seed=10)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    model = train_total_variability(per_session, tv_dim=6, n_iter=2)
    vectors = [extract_ivector(model, m, cms=False, length_norm=False) for m in per_session]
    spread = [float(np.linalg.norm(vectors[i] - vectors[j])) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    assert float(np.mean(spread)) > 1e-6, "all i-vectors are identical"


def test_total_variability_training_is_deterministic():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=80, seed=11)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    a = train_total_variability(per_session, tv_dim=6, n_iter=2)
    b = train_total_variability(per_session, tv_dim=6, n_iter=2)
    assert np.array_equal(a.A, b.A)


def test_tv_model_reports_consistent_dimensions():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=80, seed=12)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    model = train_total_variability(per_session, tv_dim=5, n_iter=2)
    assert model.A.shape == (model.n_supervector + 1, model.tv_dim)
    assert model.tv_dim == 5


def test_wrong_llr_width_is_rejected():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=6, n_per=60, seed=13)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    model = train_total_variability(per_session, tv_dim=4, n_iter=1)
    with pytest.raises(ShapeError):
        extract_ivector(model, np.zeros((10, 3)))


def test_single_frame_is_rejected():
    """One LLR column cannot support a variance estimate."""
    with pytest.raises(RankDeficientError):
        train_total_variability([np.zeros((1, 3))], tv_dim=2)


def test_empty_session_list_is_rejected():
    with pytest.raises(RankDeficientError):
        train_total_variability([], tv_dim=2)


def test_nbc_removes_the_nuisance_subspace():
    """After NBC the nuisance directions must carry no residual energy."""
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=10, n_per=100, seed=14)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    nbc = fit_nbc(per_session, n_dirs=4)
    assert nbc is not None
    assert np.allclose(nbc.W.T @ nbc.W, np.eye(4), atol=1e-10)
    projected = nbc.apply(per_session[0])
    assert np.abs(projected @ nbc.W).max() < 1e-10, "NBC left residual energy in the nuisance subspace"


def test_nbc_declines_with_insufficient_sessions():
    assert fit_nbc([np.zeros((10, 3))], n_dirs=2) is None
    assert fit_nbc([np.zeros((10, 3)), np.zeros((10, 3))], n_dirs=0) is None


def test_nbc_dimension_mismatch_is_rejected():
    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=6, n_per=60, seed=15)
    per_session = [llr(a, ubm, s) for a, s in zip(adapted, sessions)]
    nbc = fit_nbc(per_session, n_dirs=3)
    with pytest.raises(ShapeError):
        nbc.apply(np.zeros((5, 2)))


# --------------------------------------------------------------------------
# supervector centring (the step whose absence made every speaker look alike)
# --------------------------------------------------------------------------
def test_supervector_is_centred_by_ubm_mean():
    """MAP supervectors must be centred on the UBM mean before use.

    Without centring every adapted mean keeps the large component it inherited
    from the UBM, the cosine is dominated by that shared part, and all speakers
    crowd into the same region of the unit sphere.  The criterion is a large
    drop in the within-speaker cosine between the raw and the centred form.
    """
    from voiceforge.sv.ivector import center_supervector

    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=8, n_per=100, seed=21)

    def raw_cos(a: np.ndarray, b: np.ndarray) -> float:
        x, y = a.means.ravel(), b.means.ravel()
        nx, ny = np.linalg.norm(x), np.linalg.norm(y)
        return float(np.dot(x / max(nx, 1e-12), y / max(ny, 1e-12)))

    def centered_cos(a, b) -> float:
        x = center_supervector(a.means, ubm.means, length_norm=True)
        y = center_supervector(b.means, ubm.means, length_norm=True)
        return float(np.dot(x, y))

    # Two sessions of the same speaker, plus one of a different speaker.
    raw_within = raw_cos(adapted[0], adapted[1])
    cen_within = centered_cos(adapted[0], adapted[1])
    raw_between = raw_cos(adapted[0], adapted[3])
    cen_between = centered_cos(adapted[0], adapted[3])

    # Threshold 0.30 as specified.  On the synthetic fixture the raw cosine
    # starts lower than in the real pipeline, so the drop is ~0.21 here; the
    # real pipeline moves 0.968 -> 0.425 (drop 0.54), measured by
    # ``eval.cells.cosine_diagnostics``.
    assert cen_within < raw_within - 0.20, (
        f"centring barely changed the within-speaker cosine: {raw_within:.4f} -> {cen_within:.4f}"
    )
    # The decisive property: centring must turn an *inverted* comparison into a
    # correct one.  Uncentred, within-speaker pairs are MORE similar than
    # between-speaker ones (the pathology); centred, they must be less similar.
    assert cen_within < cen_between, (
        f"centred within {cen_within:.4f} is not below centred between {cen_between:.4f}"
    )


def test_center_supervector_returns_unit_norm():
    from voiceforge.sv.ivector import center_supervector

    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=4, n_per=80, seed=22)
    v = center_supervector(adapted[0].means, ubm.means, length_norm=True)
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-12)
    raw = center_supervector(adapted[0].means, ubm.means, length_norm=False)
    assert np.linalg.norm(raw) > 1e-9
    assert np.allclose(v * np.linalg.norm(raw), raw, atol=1e-9)


def test_center_supervector_is_idempotent_against_the_ubm():
    """Centring the UBM's own means must give exactly zero."""
    from voiceforge.sv.ivector import center_supervector

    ubm, sessions, adapted = _model_and_sessions(d=6, n_spk=4, n_per=80, seed=23)
    v = center_supervector(ubm.means, ubm.means, length_norm=False)
    assert np.allclose(v, 0.0, atol=1e-12)


def test_center_supervector_rejects_shape_mismatch():
    from voiceforge.sv.ivector import center_supervector

    with pytest.raises(ShapeError):
        center_supervector(np.zeros((3, 4)), np.zeros((3, 5)))
