"""Synthesis invariants, above all the separability regression.

The separability test is the single most important check in this suite.  An
earlier version of the generator drew fricative bursts at a *random position
per utterance*; the resulting broadband transients dominated the frame-level
log-mel covariance and the intra/inter distance ratio was **0.915** -- worse
than chance, i.e. the corpus carried no speaker identity at all.  Making the
frication pattern a property of the *speaker* moved the ratio to **~2.6**.

If this test fails, every downstream EER is meaningless, so it is worth far
more than its runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

from voiceforge.core.errors import SynthesisError
from voiceforge.core.seed import rng
from voiceforge.data.synthesis import (
    FORMANT_JITTER_HZ,
    MIN_SEPARATION_HZ,
    SynthesisParams,
    apply_channel,
    apply_noise,
    build_speaker_profiles,
    measure_snr_db,
    separation_margin,
    synthesize_utterance,
)

PRM = SynthesisParams(sample_rate=16000)
SR = 16000


@pytest.fixture(scope="module")
def profiles():
    return build_speaker_profiles(12, seed=17, sample_rate=SR, params=PRM)


def test_speaker_layout_meets_the_separation_contract(profiles):
    """Distinct speakers must be far apart in the (F1, F2) plane."""
    f12 = np.array([[p.formants[0], p.formants[1]] for p in profiles])
    dist = np.linalg.norm(f12[:, None, :] - f12[None, :, :], axis=2) + np.eye(len(profiles)) * 1e9
    assert float(np.min(dist)) >= MIN_SEPARATION_HZ


def test_within_speaker_jitter_stays_inside_the_budget(profiles):
    """Per-utterance formant movement must not close the inter-speaker gap."""
    p = profiles[0]
    _, meta = synthesize_utterance(p, 2.0, seed=17, params=PRM)
    assert max(meta["formant_jitter_hz"]) <= FORMANT_JITTER_HZ + 1e-9


def test_synthesis_is_bit_exact_across_runs(profiles):
    """Same (profile, duration, seed) -> byte-identical waveform."""
    a, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    b, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    assert np.array_equal(a, b)
    assert a.dtype == np.float64


def test_different_seeds_give_different_waveforms(profiles):
    a, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    b, _ = synthesize_utterance(profiles[0], 2.0, seed=29, params=PRM)
    assert not np.array_equal(a, b)


def test_waveform_shape_and_level_are_valid(profiles):
    wave, meta = synthesize_utterance(profiles[0], 3.0, seed=17, params=PRM)
    assert wave.shape == (3 * SR,)
    assert wave.ndim == 1 and wave.dtype == np.float64
    assert np.all(np.isfinite(wave))
    assert float(np.max(np.abs(wave))) <= 1.0 + 1e-9
    assert meta["duration_sec"] == pytest.approx(3.0, abs=1e-12)


def test_measured_f0_tracks_the_speaker_target(profiles):
    """The F0 bug this catches: a cumsum random walk drifted the mean by 260%."""
    for p in profiles[:5]:
        _, meta = synthesize_utterance(p, 2.0, seed=17, params=PRM)
        mean_f0 = meta["f0_measured_mean"]
        assert mean_f0 == pytest.approx(p.f0_median, rel=0.25), f"{p.speaker_id}: {mean_f0} vs {p.f0_median}"
        assert 50.0 < mean_f0 < 500.0


def test_separability_ratio_is_well_above_one(profiles):
    """THE regression: intra-speaker distance must be well below inter-speaker.

    A ratio at or below 1 means the corpus carries no usable speaker identity
    and every EER measured on it is noise.
    """
    margin = separation_margin(profiles, n_utts=3, seed=17, sample_rate=SR, prm=PRM)
    assert margin["ratio"] > 1.5, (
        f"separability ratio {margin['ratio']:.3f} <= 1.5 "
        f"(intra {margin['mean_intra_distance']:.4f} vs inter {margin['mean_inter_distance']:.4f})"
    )


def test_separability_holds_without_frication():
    """Isolates the frication fix: a fricative-free corpus is still separable."""
    prm = SynthesisParams(sample_rate=SR, fricative_prob=0.0)
    profs = build_speaker_profiles(10, seed=17, sample_rate=SR, params=prm)
    for p in profs:
        object.__setattr__(p, "fricative_pattern", ())
    margin = separation_margin(profs, n_utts=3, seed=17, sample_rate=SR, prm=prm)
    assert margin["ratio"] > 1.5, f"ratio {margin['ratio']:.3f}"


def test_fricative_pattern_is_a_speaker_property(profiles):
    """Frication must be consistent across a speaker's utterances."""
    p = profiles[0]
    assert len(p.fricative_pattern) >= 1
    _, m1 = synthesize_utterance(p, 2.0, seed=17, params=PRM)
    _, m2 = synthesize_utterance(p, 2.0, seed=99, params=PRM)
    assert m1["n_fricatives"] == m2["n_fricatives"] == len(p.fricative_pattern)


def test_too_short_utterance_is_rejected(profiles):
    with pytest.raises(SynthesisError):
        synthesize_utterance(profiles[0], 0.01, seed=17, params=PRM)


@pytest.mark.parametrize("channel, param", [("clean", None), ("telephone", None), ("reverb", 0.3)])
def test_channels_preserve_length_and_finiteness(profiles, channel, param):
    wave, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    out, meta = apply_channel(wave, channel, SR, param=param, rng_gen=rng(1, "ch"))
    assert out.shape == wave.shape
    assert np.all(np.isfinite(out))
    assert meta["channel"] == channel


def test_telephone_channel_removes_high_frequencies(profiles):
    """Narrowband coding must actually band-limit the signal."""
    wave, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    out, _ = apply_channel(wave, "telephone", SR, rng_gen=rng(1, "ch"))
    spec_before = np.abs(np.fft.rfft(wave))
    spec_after = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(wave.shape[0], 1.0 / SR)
    high = freqs > 5000.0
    assert spec_after[high].max() < spec_before[high].max() * 0.5


def test_reverb_channel_reduces_crest_factor(profiles):
    """Convolution with a decaying IR smears energy, flattening the waveform."""
    wave, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    out, _ = apply_channel(wave, "reverb", SR, param=0.4, rng_gen=rng(1, "ch"))
    crest_before = float(np.max(np.abs(wave)) / np.mean(np.abs(wave)))
    crest_after = float(np.max(np.abs(out)) / np.mean(np.abs(out)))
    assert crest_after < crest_before


def test_unknown_channel_is_rejected(profiles):
    wave, _ = synthesize_utterance(profiles[0], 1.0, seed=17, params=PRM)
    with pytest.raises(SynthesisError):
        apply_channel(wave, "telepathy", SR)


@pytest.mark.parametrize("snr", [20.0, 5.0, 0.0, -3.0])
def test_white_noise_snr_is_exact(profiles, snr):
    """The measured SNR must match the target, not merely approach it."""
    wave, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    noisy, meta = apply_noise(wave, "white", snr, SR, rng_gen=rng(2, "n"))
    assert meta["measured_snr_db"] == pytest.approx(snr, abs=1e-6)
    assert measure_snr_db(wave, noisy) == pytest.approx(snr, abs=1e-6)


def test_babble_snr_is_exact(profiles):
    """Babble is built from other speakers' speech, so it is a realistic confuser."""
    wave, _ = synthesize_utterance(profiles[0], 2.0, seed=17, params=PRM)
    pool = [synthesize_utterance(p, 2.0, seed=17, params=PRM)[0] for p in profiles[1:5]]
    noisy, meta = apply_noise(wave, "babble", 5.0, SR, rng_gen=rng(3, "n"), speech_pool=pool)
    assert meta["measured_snr_db"] == pytest.approx(5.0, abs=1e-6)


def test_babble_without_pool_is_rejected(profiles):
    wave, _ = synthesize_utterance(profiles[0], 1.0, seed=17, params=PRM)
    with pytest.raises(SynthesisError):
        apply_noise(wave, "babble", 5.0, SR, speech_pool=[])


def test_clean_conditions_add_nothing(profiles):
    wave, _ = synthesize_utterance(profiles[0], 1.5, seed=17, params=PRM)
    out, meta = apply_noise(wave, "none", None, SR)
    assert np.array_equal(out, wave)
    assert meta["measured_snr_db"] is None


def test_noise_is_reproducible(profiles):
    wave, _ = synthesize_utterance(profiles[0], 1.5, seed=17, params=PRM)
    a, _ = apply_noise(wave, "white", 5.0, SR, rng_gen=rng(4, "n"))
    b, _ = apply_noise(wave, "white", 5.0, SR, rng_gen=rng(4, "n"))
    assert np.array_equal(a, b)


def test_corpus_build_covers_every_speaker(profiles):
    """A corpus must contain the requested number of distinct identities."""
    corpus_speakers = build_speaker_profiles(8, seed=41, sample_rate=SR, params=PRM)
    assert len({p.speaker_id for p in corpus_speakers}) == 8
    assert len({round(p.f0_median, 6) for p in corpus_speakers}) > 1
