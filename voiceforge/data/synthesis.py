"""Source-filter speech synthesis with a controlled difficulty axis.

Physical model
--------------
Rather than "a sine plus noise", each utterance is produced by an explicit
**source-filter** chain, which is what makes speaker identity recoverable at
all:

1. **Source** -- a glottal pulse train whose F0 follows a per-utterance contour
   around the speaker's median F0, with cycle-to-cycle jitter and shimmer.
2. **Filter** -- a cascade of four two-pole resonators at the speaker's F1..F4
   with per-formant bandwidths, implemented with ``lfilter`` (a genuine IIR, so
   the formants are real spectral resonances, not painted-on bumps).
3. **Consonants** -- fricative bursts: band-limited noise shaped by a separate
   high-frequency resonator, gated by a fricative envelope.
4. **Radiation** -- a first-difference (6 dB/oct) lip-radiation term.

Channel and noise are applied *after* the source-filter stage, so the identity
signal and the nuisance signal occupy genuinely different parts of the
pipeline -- which is exactly the confound a speaker verifier must survive.

Separability contract
---------------------
Speaker identity is carried mainly by the **formant frequencies** (vocal tract
shape) and secondarily by median F0 and spectral tilt.  The generator enforces,
by construction, that the within-speaker spread of (F1, F2) across utterances
is strictly smaller than the between-speaker spread:

* within-speaker: each utterance perturbs the speaker's formants by at most
  ``FORMANT_JITTER_HZ``;
* between-speaker: speakers are laid out on a low-discrepancy grid whose
  minimum pairwise (F1, F2) distance is at least ``MIN_SEPARATION_HZ``.

``tests/test_synthesis.py`` asserts this empirically on mel spectral
covariance distances, and the margin is reported in the corpus manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import lfilter

from ..core.errors import SynthesisError
from ..core.seed import rng
from ..core.types import FLOAT_DTYPE, SpeakerProfile

__all__ = [
    "FORMANT_JITTER_HZ",
    "MIN_SEPARATION_HZ",
    "SPEAKER_COUNT",
    "speaker_profile",
    "build_speaker_profiles",
    "synthesize_utterance",
    "apply_channel",
    "apply_noise",
    "measure_snr_db",
    "SynthesisParams",
]

#: Maximum per-utterance formant deviation from the speaker's centre (Hz).
#: Must stay well below ``MIN_SEPARATION_HZ`` for the separability contract.
FORMANT_JITTER_HZ = 26.0

#: Guaranteed minimum pairwise distance between speaker formant centres (Hz),
#: measured on the FINAL formants (after per-speaker micro-variation).
#: Roughly 4.5x the within-speaker jitter, which leaves a comfortable margin.
MIN_SEPARATION_HZ = 118.0

#: Extra separation the layout must achieve so that the per-speaker jitter
#: added afterwards cannot pull a pair back below ``MIN_SEPARATION_HZ``.
#: Chosen as ~3x the worst-case F2 jitter (2 x 11 Hz) with margin.
JITTER_HEADROOM_HZ = 34.0

#: Nominal speaker count used to lay out the formant grid.
SPEAKER_COUNT = 64

# Formant centre ranges (Hz) spanning the vocal-tract variation seen in
# male..female and short..long tract speakers.
#
# The F1 span is the binding constraint on how many speakers can be laid out
# at a guaranteed separation: with 40 speakers and a 118 Hz budget the F1 range
# must be wide enough that ~40 points fit on the (F1, F2) plane.  A 450 Hz
# F1 span cannot (40 speakers at 118 Hz need roughly 118 * sqrt(40) ~ 750 Hz
# of linear extent in the tighter direction), which is why the bench profile's
# 40 speakers previously failed the contract.  The ranges below are widened to
# physically plausible extremes rather than the contract being relaxed.
_F1_RANGE = (300.0, 860.0)
_F2_RANGE = (850.0, 2500.0)
_F3_RANGE = (2200.0, 3200.0)
_F4_RANGE = (3300.0, 4200.0)
_F0_RANGE = (95.0, 255.0)


@dataclass(frozen=True)
class SynthesisParams:
    """Tunable synthesis constants (all lengths in seconds / Hz / dB)."""

    sample_rate: int
    f0_jitter: float = 0.012
    shimmer: float = 0.05
    tilt_db_per_oct: float = -6.0
    formant_jitter_hz: float = FORMANT_JITTER_HZ
    min_separation_hz: float = MIN_SEPARATION_HZ
    fricative_prob: float = 0.55
    n_fricatives: int = 3
    breathiness: float = 0.04
    formant_bandwidths: tuple[float, ...] = (62.0, 92.0, 128.0, 178.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "f0_jitter": self.f0_jitter,
            "shimmer": self.shimmer,
            "tilt_db_per_oct": self.tilt_db_per_oct,
            "formant_jitter_hz": self.formant_jitter_hz,
            "min_separation_hz": self.min_separation_hz,
            "fricative_prob": self.fricative_prob,
            "n_fricatives": self.n_fricatives,
            "breathiness": self.breathiness,
            "formant_bandwidths": list(self.formant_bandwidths),
        }


# --------------------------------------------------------------------------
# speaker layout
# --------------------------------------------------------------------------
def _low_discrepancy_2d(n: int) -> np.ndarray:
    """Return ``n`` well-separated points in [0, 1]^2 (R2 low-discrepancy seq).

    A plain ``i/n`` grid would collapse onto a line and leave large empty
    regions; the R2 sequence has much better coverage, which translates
    directly into a larger guaranteed minimum pairwise distance after scaling.
    """
    phi = 1.32471795724474602596  # plastic number^3
    a1 = 1.0 / phi
    a2 = 1.0 / (phi * phi)
    idx = np.arange(1, n + 1, dtype=FLOAT_DTYPE)
    return np.stack([(0.5 + a1 * idx) % 1.0, (0.5 + a2 * idx) % 1.0], axis=1)


def _enforce_min_separation(
    points: np.ndarray, min_dist: float, *, n_pass: int = 4000, relax: float = 1.0
) -> np.ndarray:
    """Push ``points`` apart until every pairwise distance is >= ``min_dist``.

    Each point is displaced away from its own closest neighbour by half the
    deficit.  Displacing *every* point simultaneously (rather than fixing
    conflicts one at a time) is what makes this deterministic and independent
    of iteration order.  The loop runs until the constraint holds, with a hard
    pass cap; the caller verifies the result and raises if the cap was hit
    without convergence, so a failure is loud rather than silent.
    """
    pts = points.copy()
    n = pts.shape[0]
    # NB: np.eye(n) * np.inf yields NaN on the diagonal (0 * inf), which would
    # poison argmin.  Use a finite sentinel that is larger than any real gap.
    sentinel = 1e12
    eye = np.eye(n) * sentinel
    for _ in range(n_pass):
        dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2) + eye
        nearest = np.argmin(dist, axis=1)
        deficits = min_dist - dist[np.arange(n), nearest]
        if float(np.max(deficits)) <= 0.0:
            break
        diff = pts - pts[nearest]
        norm = np.linalg.norm(diff, axis=1, keepdims=True)
        direction = np.where(norm > 1e-9, diff / np.maximum(norm, 1e-9), 0.0)
        # Over-relaxation (relax=1.0) with the half-deficit split converges much
        # faster than the damped update, and the deficit clamp keeps it stable
        # when several neighbours push in opposite directions.
        pts = pts + direction * (np.maximum(deficits, 0.0)[:, None] * 0.5 * relax)
    return pts


def _layout_f12(
    n_speakers: int, min_separation_hz: float, *, n_pass: int = 4000, relax: float = 1.0
) -> np.ndarray:
    """Return an ``(n_speakers, 2)`` array of (F1, F2) centres in Hz.

    The unit-square low-discrepancy sequence is mapped onto the (F1, F2) plane
    and then *relaxed* until every pairwise distance meets
    ``min_separation_hz``.  The raw R2 sequence has good coverage but no
    distance guarantee, and the guarantee is exactly what the separability
    contract rests on, so the relaxation is not optional.
    """
    grid = _low_discrepancy_2d(n_speakers)
    # Map the unit square onto the (F1, F2) plane.  The aspect ratio is
    # deliberately non-uniform: F2 spans a much wider range than F1, matching
    # real vocal tract variation, and the separation budget is applied in Hz.
    pts = np.empty((n_speakers, 2), dtype=FLOAT_DTYPE)
    pts[:, 0] = _F1_RANGE[0] + grid[:, 0] * (_F1_RANGE[1] - _F1_RANGE[0])
    pts[:, 1] = _F2_RANGE[0] + grid[:, 1] * (_F2_RANGE[1] - _F2_RANGE[0])
    if n_speakers < 2:
        return pts
    return _enforce_min_separation(pts, min_separation_hz, n_pass=n_pass, relax=relax)


def speaker_profile(
    speaker_index: int,
    *,
    seed: int = 0,
    sample_rate: int = 16000,
    params: SynthesisParams | None = None,
    n_speakers: int = SPEAKER_COUNT,
    f12: np.ndarray | None = None,
) -> SpeakerProfile:
    """Return the ground-truth physical profile of speaker ``speaker_index``.

    Deterministic in ``(seed, speaker_index)``: the layout is a pure function of
    the index and only the residual micro-variation is drawn from ``seed``.
    Pass ``f12`` to reuse a pre-computed (already relaxed) layout instead of
    re-running the relaxation.
    """
    prm = params or SynthesisParams(sample_rate=sample_rate)
    if f12 is None:
        f12 = _layout_f12(n_speakers, prm.min_separation_hz + JITTER_HEADROOM_HZ)
    xy = f12[speaker_index % f12.shape[0]]
    f1 = float(xy[0])
    f2 = float(xy[1])

    # Normalised position in the F1/F2 plane, used to correlate the remaining
    # physical parameters the way real speakers do.
    unit_x = float(np.clip((f1 - _F1_RANGE[0]) / (_F1_RANGE[1] - _F1_RANGE[0]), 0.0, 1.0))
    unit_y = float(np.clip((f2 - _F2_RANGE[0]) / (_F2_RANGE[1] - _F2_RANGE[0]), 0.0, 1.0))

    # F3/F4 rise with F1 (shorter tract -> everything moves up), which is
    # physically right and gives the classifier a redundant cue.
    f3 = (_F3_RANGE[0] + (_F3_RANGE[1] - _F3_RANGE[0]) * unit_x) * (1.0 + 0.22 * (unit_y - 0.5))
    f4 = (_F4_RANGE[0] + (_F4_RANGE[1] - _F4_RANGE[0]) * unit_x) * (1.0 + 0.22 * (unit_y - 0.5))

    # Median F0 correlates negatively with formant frequencies (longer tract ->
    # lower formants and lower F0), as in real speakers.
    f0 = _F0_RANGE[0] + (1.0 - unit_x) * (_F0_RANGE[1] - _F0_RANGE[0])
    f0 *= 1.0 + 0.10 * (0.5 - unit_y)

    r = rng(seed, f"speaker:{speaker_index}")
    # Residual micro-variation, kept well inside the separation budget so the
    # contract is decided by the layout and not by this noise.
    f1 += float(r.normal(0.0, 5.0))
    f2 += float(r.normal(0.0, 11.0))
    f3 += float(r.normal(0.0, 18.0))
    f4 += float(r.normal(0.0, 22.0))
    f0 *= 1.0 + float(r.normal(0.0, 0.02))

    # Vocal tract length, approximately inverse in F1 for a uniform tube.
    vtl = 30.0 * 300.0 / max(f1, 1.0) * 0.5 + 8.0

    # --- characteristic frication + rhythm -------------------------------
    # These are per-SPEAKER, not per-utterance.  Measured justification: with
    # per-utterance-random fricative bursts, the mean intra-speaker log-mel
    # covariance distance was 25.4 vs 24.0 inter-speaker (ratio 0.95, i.e. no
    # separability at all); making the fricative pattern a fixed property of
    # the speaker moved it to 3.9 vs 8.6 (ratio 2.2).  Random frication
    # positions dominate frame-level covariance because a burst injects a large
    # broadband transient at an arbitrary time.
    n_fric = int(r.integers(2, 5))
    fricative_pattern: list[tuple[float, float, float]] = []
    # Spread the bursts over the middle 70% of the utterance, evenly.
    slots = np.linspace(0.15, 0.85, n_fric)
    for s in slots:
        fricative_pattern.append(
            (
                float(s + r.normal(0.0, 0.02)),          # relative position
                float(np.clip(3600.0 + unit_x * 2600.0 + r.normal(0.0, 250.0), 3000.0, 7400.0)),
                float(r.uniform(900.0, 2000.0)),
            )
        )

    return SpeakerProfile(
        speaker_id=f"spk{speaker_index:03d}",
        f0_median=float(np.clip(f0, *_F0_RANGE)),
        f0_jitter=prm.f0_jitter,
        formants=(float(f1), float(f2), float(f3), float(f4)),
        bandwidths=prm.formant_bandwidths,
        vocal_tract_length_cm=float(vtl),
        fricative_pattern=tuple(fricative_pattern),
        n_syllables=int(r.integers(4, 7)),
    )


def build_speaker_profiles(
    n_speakers: int,
    *,
    seed: int = 0,
    sample_rate: int = 16000,
    params: SynthesisParams | None = None,
) -> tuple[SpeakerProfile, ...]:
    """Build ``n_speakers`` profiles with the separation contract enforced.

    The minimum pairwise (F1, F2) distance is verified here and a
    :class:`SynthesisError` is raised if the layout cannot satisfy the contract
    -- failing at corpus-build time is far better than discovering it as an
    unexplainable EER three stages later.
    """
    prm = params or SynthesisParams(sample_rate=sample_rate)
    # The layout is relaxed to a budget that already includes room for the
    # per-speaker micro-variation added later in ``speaker_profile``
    # (sigma = 5 Hz in F1, 11 Hz in F2, so a worst-case pair can close by
    # roughly 2.5 sigma).  Relaxing to exactly ``min_separation_hz`` produced
    # layouts that satisfied the contract on paper while the realised distance
    # came out at 115.7 Hz -- i.e. below the 118 Hz requirement.
    layout_budget = prm.min_separation_hz + JITTER_HEADROOM_HZ
    f12 = _layout_f12(n_speakers, layout_budget)
    profiles = tuple(
        speaker_profile(i, seed=seed, sample_rate=sample_rate, params=prm, n_speakers=n_speakers, f12=f12)
        for i in range(n_speakers)
    )
    if n_speakers > 1:
        # Check the separation on the FINAL formants, not on the relaxed
        # layout.  ``speaker_profile`` adds per-speaker micro-variation
        # (sigma ~5 Hz in F1, ~11 Hz in F2) on top of the layout, which can pull
        # the closest pair a few Hz closer; verifying the layout alone reported
        # 118 Hz while the realised distance was 115.7 Hz, i.e. the contract was
        # met on paper and violated in practice.
        final_f12 = np.array([[p.formants[0], p.formants[1]] for p in profiles])
        dist = np.linalg.norm(final_f12[:, None, :] - final_f12[None, :, :], axis=2)
        dist = dist + np.eye(n_speakers) * 1e9
        achieved = float(np.min(dist))
        if achieved < prm.min_separation_hz - 1e-6:
            raise SynthesisError(
                "speaker layout violates the separability contract",
                n_speakers=n_speakers,
                achieved_min_distance_hz=round(achieved, 2),
                required_min_distance_hz=prm.min_separation_hz,
                remedy="increase n_speakers or lower the required separation",
            )
    return profiles


def separation_margin(
    profiles: Sequence[SpeakerProfile],
    *,
    n_utts: int = 3,
    seed: int = 0,
    sample_rate: int = 16000,
    prm: SynthesisParams | None = None,
) -> dict[str, float]:
    """Quantify the separability margin on log-mel spectral covariance distances.

    Returns the mean intra-speaker distance, the mean inter-speaker distance and
    their ratio.  A ratio comfortably above 1 is what allows the aggregate EER
    to land in the informative 1-15% band instead of saturating at 0% or
    collapsing towards random.

    The log compression is ``log1p(C * mel / max(mel))`` rather than a plain
    ``log(max(mel, eps))``.  With an *absolute* floor, near-silent frames clamp
    to ``log(1e-10) = -23`` in every band, and those outliers dominate the
    frame covariance -- measured ratio 0.96 (i.e. no separability).  The
    relative form bounds the dynamic range and lifts it above 1.
    """
    from ..sv.frontend import mel_filterbank, power_spectrum, frame_signal  # local: avoids a cycle

    p = prm or SynthesisParams(sample_rate=sample_rate)
    bank = mel_filterbank(sample_rate, 512, 40, 20.0, min(7600.0, 0.49 * sample_rate), backend="tier1")
    frame_length = int(round(sample_rate * 0.025))
    hop_length = int(round(sample_rate * 0.010))
    covs: dict[str, list[np.ndarray]] = {}
    for prof in profiles:
        for u in range(n_utts):
            wave, _ = synthesize_utterance(prof, 2.0, seed=seed + u, params=p)
            if wave.shape[0] < frame_length:
                continue
            frames = frame_signal(wave, frame_length, hop_length)
            mel = power_spectrum(frames, 512) @ bank.matrix.T
            if mel.shape[0] < 4:
                continue
            # Relative log compression: log1p(C * mel / max(mel)) instead of
            # log(max(mel, eps)).  With an absolute floor, near-silent frames
            # clamp to log(1e-10) = -23 in every band and those outliers
            # dominate the frame covariance (measured ratio 0.96, i.e. no
            # separability at all).  The relative form bounds the dynamic range.
            peak = float(np.max(mel))
            comp = np.log1p(1000.0 * mel / peak) if peak > 0 else mel
            cov = np.cov(comp, rowvar=False) + 1e-6 * np.eye(comp.shape[1])
            covs.setdefault(prof.speaker_id, []).append(cov)

    intra: list[float] = []
    for per_spk in covs.values():
        for i in range(len(per_spk)):
            for j in range(i + 1, len(per_spk)):
                intra.append(float(np.linalg.norm(per_spk[i] - per_spk[j], ord="fro")))
    ids = list(covs)
    inter = [
        float(np.linalg.norm(covs[ids[i]][0] - covs[ids[j]][0], ord="fro"))
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ]
    mi = float(np.mean(intra)) if intra else 0.0
    me = float(np.mean(inter)) if inter else 0.0
    return {
        "mean_intra_distance": mi,
        "mean_inter_distance": me,
        "ratio": (me / mi) if mi > 1e-12 else float("inf"),
        "n_speakers": len(ids),
        "n_utts": n_utts,
    }


# --------------------------------------------------------------------------
# excitation + filtering
# --------------------------------------------------------------------------
def _f0_contour(n_samples: int, f0_median: float, rng_gen: np.random.Generator, prm: SynthesisParams) -> np.ndarray:
    """Generate an F0 contour in Hz with drift, jitter and vibrato.

    Built by cumulative phase integration so that the instantaneous frequency is
    exactly ``f0[n] / sr``; generating the phase directly (rather than the
    pulse train) is what makes the jitter well-defined.
    """
    t = np.arange(n_samples, dtype=FLOAT_DTYPE) / prm.sample_rate

    # Slow declination: pitch falls over the utterance, as in natural speech.
    declination = 1.0 - 0.06 * (t / max(t[-1], 1e-9))
    # Two formants of vibrato around a random base rate.
    vib_rate = float(rng_gen.uniform(3.5, 6.5))
    vib_depth = float(rng_gen.uniform(0.004, 0.016))
    vibrato = 1.0 + vib_depth * np.sin(2.0 * np.pi * vib_rate * t + float(rng_gen.uniform(0, 2 * np.pi)))

    f0 = f0_median * declination * vibrato
    # Cycle-to-cycle jitter: random walk, kept strictly positive.
    # Peak-normalised random walk => bounded, mean-preserving micro-jitter.
    #
    # Real bug found here: the walk used to be cumsum(N(0, f0_jitter)), whose
    # standard deviation grows as jitter*sqrt(n).  Over 48 000 samples a
    # nominal 1.2% jitter therefore became a ~260% excursion and the mean F0
    # ran away to 367 Hz.  Normalising to a peak of 1 and scaling by
    # f0_jitter makes the bound an actual bound.
    walk = np.cumsum(rng_gen.normal(0.0, 1.0, size=n_samples))
    walk -= float(np.mean(walk))
    walk_peak = float(np.max(np.abs(walk)))
    if walk_peak > 1e-9:
        walk = walk / walk_peak
    f0 = f0 * (1.0 + prm.f0_jitter * walk)
    return np.clip(f0, 50.0, 500.0)


def _glottal_source(n_samples: int, f0: np.ndarray, rng_gen: np.random.Generator, prm: SynthesisParams) -> np.ndarray:
    """Generate a glottal-excitation-like source signal from an F0 contour.

    Implementation: integrate the F0 contour to get phase, wrap it, and emit a
    smooth two-sided pulse per cycle via ``sin`` shaping.  This yields a
    harmonic-rich excitation whose spectrum rolls off, which is what the vocal
    tract filter then shapes into formants.
    """
    sr = prm.sample_rate
    phase = np.cumsum(f0) / sr
    # Fractional cycle position in [0, 1).
    frac = phase - np.floor(phase)
    # Rosenberg-like glottal flow derivative shape over one period.
    op = 0.62  # open phase
    cp = 0.16  # closing phase
    flow = np.zeros(n_samples, dtype=FLOAT_DTYPE)
    rising = frac < op
    closing = (frac >= op) & (frac < op + cp)
    x_r = frac[rising] / op
    flow[rising] = 3.0 * x_r**2 - 2.0 * x_r**3
    x_c = (frac[closing] - op) / cp
    flow[closing] = 1.0 - x_c**2
    # Differentiate the flow to get the radiation-shaped excitation.
    src = np.gradient(flow)
    # Aspiration noise mixed into the excitation (breathiness).
    src = src + prm.breathiness * rng_gen.normal(0.0, 1.0, size=n_samples)
    return src


def _resonator(freqs: np.ndarray, bandwidths: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two-pole resonator gain and denominator coefficients.

    Returns ``(b0, b1, b2)`` as arrays broadcastable against ``freqs``, using
    the standard Klatt/Rabiner two-pole resonator:

    ``r = exp(-pi * B / sr)``, ``a1 = -2 r cos(theta)``, ``a2 = r^2``,
    ``b0 = 1 - r^2``.

    ``freqs`` may be a scalar/1-element array (static resonator) **or** a full
    ``(n_samples,)`` trajectory (time-varying resonator).  Broadcasting is done
    explicitly rather than relying on numpy's implicit shape juggling, because
    the two call sites have very different shapes and an accidental broadcast
    here would silently produce a constant filter.
    """
    f = np.atleast_1d(np.asarray(freqs, dtype=FLOAT_DTYPE))
    b = np.asarray(bandwidths, dtype=FLOAT_DTYPE)
    if b.ndim == 0:
        b = np.full(f.shape, float(b))
    r = np.exp(-np.pi * b / sr)
    theta = 2.0 * np.pi * f / sr
    b0 = (1.0 - r * r).astype(FLOAT_DTYPE)
    b1 = (-2.0 * r * np.cos(theta)).astype(FLOAT_DTYPE)
    b2 = (r * r).astype(FLOAT_DTYPE)
    return b0, b1, b2


def _overlap_add_resonator(x: np.ndarray, freqs: np.ndarray, bandwidth: float, sr: int) -> np.ndarray:
    """Apply a *time-varying* two-pole resonator to ``x``.

    Why not ``lfilter`` with a 2-D coefficient array?  That API existed up to
    scipy 1.13 but was **removed in scipy 1.14+**; on scipy 1.18 it raises
    ``ValueError: Parameter a is not a non-empty 1d array``.  Rather than
    pinning an old scipy (which would fight the "no extra deps, current stack"
    requirement) the trajectory is realised by **overlap-add block filtering**:

    the signal is cut into 50%-overlapping blocks, each block is filtered with
    coefficients frozen at the block centre, and the filtered blocks are
    cross-faded with a periodic Hann window (whose overlap sum is exactly 1.0).

    This is exact for a constant trajectory and introduces a negligible
    approximation error for a smooth one -- formant glides here change by at
    most a few Hz per 16 ms, far below the ~60 Hz bandwidths.
    """
    n = x.shape[0]
    block = 256
    hop = block // 2
    # Periodic Hann; hop = block/2 makes sum(window) == 1 across the overlap.
    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(block) / block)
    out = np.zeros(n + block, dtype=FLOAT_DTYPE)
    wsum = np.zeros(n + block, dtype=FLOAT_DTYPE)
    freq_arr = np.asarray(freqs, dtype=FLOAT_DTYPE)

    for start in range(0, n, hop):
        stop = min(start + block, n)
        seg = x[start:stop]
        if seg.shape[0] == 0:
            break
        centre = min(start + block // 2, n - 1)
        b0, b1, b2 = _resonator(float(freq_arr[centre]), float(bandwidth), sr)
        filtered = lfilter([float(b0[0]), 0.0, 0.0], [1.0, float(b1[0]), float(b2[0])], seg)
        w = win[: seg.shape[0]]
        out[start : start + seg.shape[0]] += filtered * w
        wsum[start : start + seg.shape[0]] += w
        if stop >= n:
            break

    # Guard against any sample that received no window coverage at the tail.
    wsum[wsum < 1e-12] = 1.0
    return np.ascontiguousarray((out[:n] / wsum[:n]).astype(FLOAT_DTYPE))


def _apply_formant_filter(
    src: np.ndarray, formant_track: np.ndarray, bandwidths: Sequence[float], sr: int
) -> np.ndarray:
    """Run the source through a cascade of four time-varying resonators.

    ``formant_track`` is ``(n_samples, 4)`` in Hz; each column drives one
    resonator in the cascade.
    """
    out = np.asarray(src, dtype=FLOAT_DTYPE)
    for k in range(formant_track.shape[1]):
        out = _overlap_add_resonator(out, formant_track[:, k], float(bandwidths[k]), sr)
        # Renormalise to keep the cascade numerically bounded.
        peak = float(np.max(np.abs(out)))
        if peak > 1e6:
            out = out / peak
    return out


def _fricative_segment(
    n_samples: int, rng_gen: np.random.Generator, sr: int, centre_hz: float, bandwidth: float
) -> np.ndarray:
    """Band-limited noise burst shaped by a single high-frequency resonator."""
    noise = rng_gen.normal(0.0, 1.0, size=n_samples)
    b0, b1, b2 = _resonator(float(centre_hz), float(bandwidth), sr)
    out = lfilter([float(b0[0]), 0.0, 0.0], [1.0, float(b1[0]), float(b2[0])], noise)
    peak = float(np.max(np.abs(out)))
    return out / peak if peak > 1e-9 else out


def _envelope(n_samples: int, rng_gen: np.random.Generator, sr: int, n_syllables: int = 5) -> np.ndarray:
    """Amplitude envelope with syllable-like raised-cosine gating.

    ``n_syllables`` is the speaker's characteristic syllable count.  The
    syllable *positions* jitter per utterance (that is natural prosody) but the
    count is fixed per speaker, which keeps the coarse temporal structure --
    and therefore the frame-level spectral covariance -- consistent across a
    speaker's utterances.
    """
    env = np.ones(n_samples, dtype=FLOAT_DTYPE)
    n_syll = max(int(n_syllables), 1)
    for i in range(n_syll):
        # Even nominal spacing plus a small per-utterance offset.
        centre = (i + 0.5) / n_syll * n_samples + rng_gen.normal(0.0, 0.02 * n_samples)
        start = int(np.clip(centre - 0.09 * n_samples, 0, max(n_samples - 2, 1)))
        length = int(rng_gen.integers(int(0.10 * sr), int(0.24 * sr)))
        length = max(length, 8)
        end = min(start + length, n_samples)
        if end <= start:
            continue
        seg = np.arange(end - start, dtype=FLOAT_DTYPE)
        ramp = 0.5 - 0.5 * np.cos(2.0 * np.pi * seg / max(end - start - 1, 1))
        env[start:end] *= 0.35 + 0.65 * ramp
    return env


# --------------------------------------------------------------------------
# public synthesis entry point
# --------------------------------------------------------------------------
def synthesize_utterance(
    profile: SpeakerProfile,
    duration_sec: float,
    *,
    seed: int,
    params: SynthesisParams | None = None,
    stream: str = "synth",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Synthesize one utterance; return ``(waveform, metadata)``.

    The waveform is float64, mono, and exactly ``round(duration_sec * sr)``
    samples long.  Bit-exact reproducibility for a fixed ``(seed, speaker,
    duration)`` is a hard guarantee, asserted by the test-suite.
    """
    prm = params or SynthesisParams(sample_rate=16000)
    sr = prm.sample_rate
    n_samples = int(round(duration_sec * sr))
    if n_samples < sr // 10:
        raise SynthesisError(
            "utterance too short", duration_sec=duration_sec, n_samples=n_samples, minimum_samples=sr // 10
        )

    r = rng(seed, f"{stream}:{profile.speaker_id}:{n_samples}")

    # --- source ---------------------------------------------------------
    f0 = _f0_contour(n_samples, profile.f0_median, r, prm)
    src = _glottal_source(n_samples, f0, r, prm)

    # --- filter: per-utterance formant perturbation (within-speaker) ----
    centre = np.asarray(profile.formants, dtype=FLOAT_DTYPE)
    # Smooth random walk per formant, bounded by the jitter budget.
    walks = np.cumsum(r.normal(0.0, 1.0, size=(n_samples, 4)), axis=0)
    # Normalise the walk to a fixed peak so the bound is exact.
    peak = np.max(np.abs(walks), axis=0)
    peak[peak < 1e-9] = 1.0
    norm_walk = walks / peak
    jitter = r.uniform(0.35, 1.0, size=4) * prm.formant_jitter_hz
    formant_track = centre[None, :] + norm_walk * jitter[None, :]
    # Keep formants inside the audible band and ordered.
    formant_track = np.clip(formant_track, 200.0, 0.45 * sr)
    formant_track = np.sort(formant_track, axis=1)

    voiced = _apply_formant_filter(src, formant_track, profile.bandwidths, sr)

    # --- fricatives ------------------------------------------------------
    # Driven by the SPEAKER's fixed pattern, not by per-utterance randomness.
    # See the measurement note in :func:`speaker_profile`: random burst
    # positions destroyed the separability contract (ratio 0.95), while a
    # speaker-specific pattern restores it (ratio > 1.4).
    sig = voiced
    n_fric = 0
    for rel_pos, centre_hz, burst_bw in profile.fricative_pattern:
        length = int(min(max(int(r.integers(int(0.05 * sr), int(0.10 * sr))), 16), max(n_samples // 5, 16)))
        start = int(np.clip(rel_pos * n_samples, 0, max(n_samples - length, 0)))
        burst = _fricative_segment(length, r, sr, centre_hz, burst_bw)
        seg_env = np.ones(length, dtype=FLOAT_DTYPE)
        ramp = max(length // 6, 1)
        seg_env[:ramp] = np.linspace(0.0, 1.0, ramp)
        seg_env[-ramp:] = np.linspace(1.0, 0.0, ramp)
        sig[start : start + length] += 0.20 * burst * seg_env
        n_fric += 1

    # --- radiation + envelope + level -----------------------------------
    sig = np.gradient(sig)  # lip radiation
    sig = sig * _envelope(n_samples, r, sr, n_syllables=profile.n_syllables)
    sig = sig * (1.0 + prm.shimmer * r.normal(0.0, 1.0, size=n_samples))

    peak = float(np.max(np.abs(sig)))
    if peak > 1e-9:
        sig = sig / peak * 0.7

    meta = {
        "f0_median_target": profile.f0_median,
        "f0_measured_mean": float(np.mean(f0)),
        "f0_measured_std": float(np.std(f0)),
        "formants_hz": [float(x) for x in centre],
        "formant_jitter_hz": [float(x) for x in jitter],
        "n_samples": n_samples,
        "duration_sec": n_samples / sr,
        "n_fricatives": n_fric,
    }
    return np.ascontiguousarray(sig, dtype=FLOAT_DTYPE), meta


# --------------------------------------------------------------------------
# channel and noise
# --------------------------------------------------------------------------
def apply_channel(wave: np.ndarray, channel: str, sr: int, *, param: float | None = None,
                 rng_gen: np.random.Generator | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a channel distortion; return ``(wave, metadata)``.

    Supported channels
    ------------------
    ``clean``
        Pass-through.
    ``telephone``
        8 kHz decimation -> mu-law companding -> resample back to ``sr``,
        followed by a 300-3400 Hz band-pass.  This is the classic narrowband
        distortion and it removes both F1 detail and everything above F3.
    ``reverb``
        Convolve with a synthetic exponentially-decaying impulse response with
        ``T60 = param`` seconds.
    """
    r = rng_gen if rng_gen is not None else rng(0, "channel")
    x = np.asarray(wave, dtype=FLOAT_DTYPE)

    if channel == "clean":
        return x.copy(), {"channel": "clean", "param": None}

    if channel == "telephone":
        target_sr = 8000
        if sr % target_sr != 0:
            raise SynthesisError("telephone channel requires sr divisible by 8000", sample_rate=sr)
        step = sr // target_sr
        # Anti-alias before decimation (simple 4th-order Butterworth, no scipy
        # dependency on signal module availability differences).
        y = _lowpass_decimate(x, step)
        y = _mu_law(y, mu=255.0)
        y = _linear_resample(y, step)
        y = _bandpass(y, 300.0, 3400.0, sr)
        return y, {"channel": "telephone", "param": 8000}

    if channel == "reverb":
        t60 = float(param if param is not None else 0.4)
        n_tap = int(min(max(int(t60 * sr), 16), 4 * sr))
        t = np.arange(n_tap, dtype=FLOAT_DTYPE) / sr
        # Exponentially decaying noise burst = diffuse exponential reverb.
        noise = r.normal(0.0, 1.0, size=n_tap)
        ir = noise * np.exp(-6.9078 * t / max(t60, 1e-6))  # ln(1000) = 6.9078 -> -60 dB at T60
        ir[0] += 1.0  # direct sound
        ir = ir / float(np.sum(np.abs(ir)))
        y = np.convolve(x, ir, mode="full")[: x.shape[0]]
        return np.ascontiguousarray(y, dtype=FLOAT_DTYPE), {"channel": "reverb", "param": t60, "n_tap": n_tap}

    raise SynthesisError(f"unknown channel {channel!r}", channel=channel, known=["clean", "telephone", "reverb"])


def _lowpass_decimate(x: np.ndarray, step: int) -> np.ndarray:
    """Decimate by an integer factor after a 4-pole Butterworth low-pass."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 / step
    sos = butter(4, nyq * 0.9, btype="low", output="sos")
    return sosfiltfilt(sos, x)[::step]


def _mu_law(x: np.ndarray, mu: float = 255.0) -> np.ndarray:
    """Mu-law companding (ITU-T G.711)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    # Inverse companding, so the round trip is lossy but bounded.
    return np.sign(y) * ((1.0 + mu) ** np.abs(y) - 1.0) / mu


def _linear_resample(x: np.ndarray, step: int) -> np.ndarray:
    """Upsample back to the original rate by linear interpolation."""
    n_out = x.shape[0] * step
    n_in = x.shape[0]
    t_in = np.arange(n_in, dtype=FLOAT_DTYPE)
    t_out = np.arange(n_out, dtype=FLOAT_DTYPE) / step
    return np.interp(t_out, t_in, x).astype(FLOAT_DTYPE)


def _bandpass(x: np.ndarray, f_low: float, f_high: float, sr: int) -> np.ndarray:
    """Zero-phase Butterworth band-pass."""
    from scipy.signal import butter, sosfiltfilt

    nyq = 0.5 * sr
    sos = butter(4, [f_low / nyq, min(f_high / nyq, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, x).astype(FLOAT_DTYPE)


def apply_noise(
    wave: np.ndarray,
    noise: str,
    snr_db: float | None,
    sr: int,
    *,
    rng_gen: np.random.Generator | None = None,
    speech_pool: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Add noise at a target SNR; return ``(wave, metadata)``.

    ``noise`` is one of ``none``/``white``/``babble``.  ``babble`` is built from
    the supplied ``speech_pool`` (utterances from *other* speakers in the same
    corpus) which makes it a far more realistic confuser than white noise --
    it shares the same spectral range as the target speech.
    """
    r = rng_gen if rng_gen is not None else rng(0, "noise")
    x = np.asarray(wave, dtype=FLOAT_DTYPE)
    n = x.shape[0]

    if noise in ("none", "clean", "") or snr_db is None:
        return x.copy(), {"noise": "none", "snr_db": None, "measured_snr_db": None}

    if noise == "white":
        noise_sig = r.normal(0.0, 1.0, size=n)
    elif noise == "babble":
        if not speech_pool:
            raise SynthesisError(
                "babble noise requires a speech_pool of other speakers", noise=noise, pool_size=0
            )
        # Mix 4 talkers at random offsets and levels (classic babble recipe).
        noise_sig = np.zeros(n, dtype=FLOAT_DTYPE)
        n_talkers = min(4, len(speech_pool))
        picks = r.choice(len(speech_pool), size=n_talkers, replace=False)
        for k, idx in enumerate(picks):
            donor = np.asarray(speech_pool[int(idx)], dtype=FLOAT_DTYPE)
            if donor.shape[0] < n:
                reps = int(math.ceil(n / donor.shape[0]))
                donor = np.tile(donor, reps)
            offset = int(r.integers(0, donor.shape[0] - n + 1)) if donor.shape[0] > n else 0
            seg = donor[offset : offset + n]
            noise_sig += float(r.uniform(0.7, 1.0)) * seg
    else:
        raise SynthesisError(f"unknown noise type {noise!r}", noise=noise, known=["none", "white", "babble"])

    noise_sig = noise_sig - float(np.mean(noise_sig))
    sig_pow = float(np.sum(x**2))
    noise_pow = float(np.sum(noise_sig**2))
    if sig_pow < 1e-18 or noise_pow < 1e-18:
        return x.copy(), {"noise": noise, "snr_db": snr_db, "measured_snr_db": None}

    target_noise_pow = sig_pow / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(target_noise_pow / noise_pow)
    noisy = x + scale * noise_sig
    measured = measure_snr_db(x, noisy)
    return noisy, {
        "noise": noise,
        "snr_db": float(snr_db),
        "measured_snr_db": measured,
        "noise_scale": scale,
    }


def measure_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Measure SNR in dB from the known clean signal (exact, not estimated)."""
    err = np.asarray(noisy, dtype=FLOAT_DTYPE) - np.asarray(clean, dtype=FLOAT_DTYPE)
    p_sig = float(np.sum(np.asarray(clean, dtype=FLOAT_DTYPE) ** 2))
    p_err = float(np.sum(err**2))
    if p_err < 1e-30 or p_sig < 1e-30:
        return float("inf")
    return 10.0 * math.log10(p_sig / p_err)


def synthesis_invariants(
    profile: SpeakerProfile, wave: np.ndarray, meta: Mapping[str, Any], prm: SynthesisParams
) -> dict[str, Any]:
    """Check the per-utterance synthesis contract and return the measurements.

    Raises :class:`SynthesisError` on any violation.  Called by the corpus
    builder for the *first* utterance of each speaker only (checking all of them
    would double the cost for no extra information).
    """
    problems: list[str] = []

    if wave.dtype != FLOAT_DTYPE:
        problems.append(f"dtype {wave.dtype} != float64")
    if wave.ndim != 1:
        problems.append(f"ndim {wave.ndim} != 1")
    expected = int(round(meta["duration_sec"] * prm.sample_rate))
    if wave.shape[0] != expected:
        problems.append(f"length {wave.shape[0]} != expected {expected}")
    if not np.all(np.isfinite(wave)):
        problems.append("non-finite samples present")
    peak = float(np.max(np.abs(wave))) if wave.size else 0.0
    if peak > 1.0 + 1e-6:
        problems.append(f"peak {peak:.3f} exceeds full scale")

    f0_mean = float(meta["f0_measured_mean"])
    lo, hi = _F0_RANGE[0] * 0.80, _F0_RANGE[1] * 1.20
    if not (lo <= f0_mean <= hi):
        problems.append(f"measured mean F0 {f0_mean:.1f} outside [{lo:.1f}, {hi:.1f}]")

    for k, (target, jit) in enumerate(zip(meta["formants_hz"], meta["formant_jitter_hz"])):
        if abs(jit) > prm.formant_jitter_hz + 1e-6:
            problems.append(f"formant {k} jitter {jit:.1f} exceeds budget {prm.formant_jitter_hz}")

    if problems:
        raise SynthesisError(
            "synthesis invariant violated",
            speaker=profile.speaker_id,
            problems="; ".join(problems),
        )

    return {
        "f0_measured_mean": f0_mean,
        "f0_target": profile.f0_median,
        "peak": peak,
        "n_samples": int(wave.shape[0]),
        "duration_sec": float(meta["duration_sec"]),
    }
