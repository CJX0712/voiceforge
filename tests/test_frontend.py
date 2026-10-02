"""Front-end invariants: framing, mel filterbank, DCT, CMVN, tier parity."""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.core.errors import FeatureError
from voiceforge.core.seed import rng
from voiceforge.sv.frontend import (
    Frontend,
    FrontendConfig,
    cmvn,
    dct_matrix,
    hz_to_mel,
    lifter,
    mel_filterbank,
    mel_to_hz,
    n_frames_for,
    regression_deltas,
    preemphasis,
)

SR = 16000
FRAME, HOP = 400, 160


@pytest.mark.parametrize("n_samples", [400, 401, 1000, 16000, 48000, 47999])
def test_frame_count_matches_the_declared_formula(n_samples):
    """``1 + floor((n - frame_len) / hop)``; an off-by-one shifts every index."""
    expected = 0 if n_samples < FRAME else 1 + (n_samples - FRAME) // HOP
    assert n_frames_for(n_samples, FRAME, HOP) == expected


def test_frame_count_formula_on_the_real_frontend():
    fe = Frontend(FrontendConfig(backend="tier1"))
    for n in (16000, 32000, 48000):
        assert fe.n_frames(n) == 1 + (n - fe.config.frame_length) // fe.config.hop_length


def test_signal_shorter_than_a_frame_yields_no_frames():
    fe = Frontend(FrontendConfig(backend="tier1"))
    out = fe.process(np.zeros(100, dtype=np.float64))
    assert out.shape == (0, fe.n_features)


@pytest.mark.parametrize("n_utt", [5, 40])
def test_emitted_feature_rows_equal_the_frame_count(n_utt):
    fe = Frontend(FrontendConfig(backend="tier1"))
    wave = rng(1, "w").normal(0.0, 0.1, size=16000 + 137)
    out = fe.process(wave)
    assert out.shape[0] == n_frames_for(wave.shape[0], FRAME, HOP)
    assert out.shape[1] == fe.n_features == 20 * 3


def test_mel_scale_round_trips():
    hz = np.array([20.0, 300.0, 1000.0, 3000.0, 7600.0])
    assert np.abs(mel_to_hz(hz_to_mel(hz)) - hz).max() < 1e-8


def test_tier1_mel_matches_librosa():
    """The pure-numpy Slaney filterbank must agree with librosa's to ~1e-9.

    This is what licenses reporting tier0 and tier1 results in the same table:
    they are two implementations of one definition, not two different features.
    """
    t0 = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier0")
    t1 = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier1")
    assert t0.backend_tag == "librosa" and t1.backend_tag == "numpy"
    assert t0.matrix.shape == t1.matrix.shape == (40, 257)
    assert np.abs(t0.matrix - t1.matrix).max() < 1e-7


def test_tier_fingerprints_differ_but_are_stable():
    a = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier0")
    b = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier0")
    c = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier1")
    assert a.fingerprint() == b.fingerprint(), "same backend must give the same fingerprint"
    assert a.fingerprint() != c.fingerprint(), "different implementations must be distinguishable"


def test_mel_filterbank_is_ascending_and_overlapping():
    """Channel centres must ascend and adjacent triangles must overlap."""
    fb = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier1")
    centres = fb.matrix.argmax(axis=1)
    assert np.all(np.diff(centres) > 0), "mel channel centres must strictly ascend"
    overlapping = sum(
        1 for i in range(fb.matrix.shape[0] - 1) if bool(fb.matrix[i + 1][fb.matrix[i] > 0].any())
    )
    assert overlapping >= fb.matrix.shape[0] - 2, f"only {overlapping} overlapping pairs"


def test_mel_filterbank_has_a_slope_normalisation():
    """Slaney normalisation keeps every channel's peak comparable."""
    fb = mel_filterbank(SR, 512, 40, 20.0, 7600.0, backend="tier1")
    peaks = fb.matrix.max(axis=1)
    assert np.all(peaks > 0.0)
    assert peaks.max() / peaks.min() < 10.0, "channel gains are wildly unbalanced"


def test_dct_matrix_is_orthonormal():
    d = dct_matrix(20, 40)
    assert np.abs(d @ d.T - np.eye(20)).max() < 1e-12


def test_dct_matches_scipy_ortho():
    from scipy.fftpack import dct

    mine = dct_matrix(20, 40)
    ref = dct(np.eye(40), type=2, axis=0, norm="ortho")[:20]
    assert np.abs(mine - ref).max() < 1e-12


def test_cmvn_normalises_every_coefficient():
    r = rng(2, "cmvn")
    feats = r.normal(3.0, 2.0, size=(300, 20)) + np.arange(20)
    out = cmvn(feats)
    assert np.abs(out.mean(axis=0)).max() < 1e-9
    assert np.abs(out.std(axis=0) - 1.0).max() < 1e-9


def test_cmvn_handles_constant_columns_without_dividing_by_zero():
    feats = np.ones((50, 4))
    out = cmvn(feats)
    assert np.all(np.isfinite(out))


def test_cmvn_is_per_utterance_and_therefore_leakage_free():
    """Two utterances with the same shape must normalise identically."""
    base = rng(3, "u").normal(0.0, 1.0, size=(200, 8))
    a = cmvn(base)
    b = cmvn(base * 3.0 + 100.0)  # scaled and shifted
    assert np.abs(a - b).max() < 1e-9


@pytest.mark.parametrize("order", [1, 2])
def test_deltas_preserve_shape_and_vanish_on_constants(order):
    x = rng(4, "d").normal(size=(120, 6))
    for order_ in (1, 2):
        out = regression_deltas(x, 2, order=order_)
        assert out.shape == x.shape
    assert np.allclose(regression_deltas(np.ones((40, 3)), 2, order=1), 0.0)
    assert np.allclose(regression_deltas(np.ones((40, 3)), 2, order=2), 0.0)


def test_deltas_of_a_ramp_recover_its_slope():
    """A linear trend must give a constant first-order delta."""
    t = np.arange(60, dtype=np.float64)
    ramp = np.stack([2.0 * t + c for c in (0.0, 5.0, -3.0)], axis=1)
    d1 = regression_deltas(ramp, 2, order=1)
    interior = d1[4:-4]
    assert np.abs(interior - 2.0).max() < 1e-9


def test_lifter_increases_high_order_coefficients():
    x = np.tile(np.arange(20, dtype=np.float64), (10, 1))
    lifted = lifter(x, 22.0)
    assert lifted.shape == x.shape
    assert lifted[0, 10] > x[0, 10], "liftering must boost higher cepstral orders"
    assert lifter(x, 0.0).tolist() == x.tolist()


def test_preemphasis_shapes_the_signal():
    x = np.ones(100, dtype=np.float64)
    assert np.allclose(preemphasis(x, 0.0), x)
    out = preemphasis(x, 0.97)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(1.0 - 0.97)


def test_frontend_is_deterministic():
    fe = Frontend(FrontendConfig(backend="tier1"))
    wave = rng(5, "w").normal(0.0, 0.1, size=16000)
    assert np.array_equal(fe.process(wave), fe.process(wave))


def test_keep_c0_changes_the_features_but_not_the_dimension():
    a = Frontend(FrontendConfig(backend="tier1", keep_c0=False)).process(rng(6, "w").normal(0, 0.1, 16000))
    b = Frontend(FrontendConfig(backend="tier1", keep_c0=True)).process(rng(6, "w").normal(0, 0.1, 16000))
    assert a.shape == b.shape
    assert not np.allclose(a, b)


def test_sample_rate_mismatch_is_rejected():
    fe = Frontend(FrontendConfig(backend="tier1", sample_rate=16000))
    with pytest.raises(FeatureError):
        fe.process(np.zeros(16000, dtype=np.float64), sample_rate=8000)


def test_wrong_width_log_mel_is_rejected():
    fe = Frontend(FrontendConfig(backend="tier1"))
    with pytest.raises(FeatureError):
        fe.mfcc_from_log_mel(np.zeros((10, 13)))


def test_tier1_frontend_matches_tier0():
    """End-to-end: the two tiers must produce numerically equivalent features."""
    wave = rng(7, "w").normal(0.0, 0.1, size=24000)
    a = Frontend(FrontendConfig(backend="tier0")).process(wave)
    b = Frontend(FrontendConfig(backend="tier1")).process(wave)
    assert a.shape == b.shape
    assert np.abs(a - b).max() < 1e-6, f"tier0/tier1 feature mismatch {np.abs(a - b).max():.3e}"
