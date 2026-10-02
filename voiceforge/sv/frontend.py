"""Feature front end: framing, mel filterbank, MFCC, deltas, CMVN.

Two DSP tiers are implemented behind one interface:

* **tier0** -- delegates the mel filterbank to
  ``librosa.filters.mel(..., htk=False, norm="slaney")``;
* **tier1** -- a self-contained numpy implementation of the *same* Slaney-normalised
  triangular filterbank.

Both produce a matrix that is fingerprinted with :func:`mel_fingerprint` and the
fingerprint is embedded in ``benchmark.json``, so a reader can verify that two
runs really used the same filterbank rather than trusting the label.

The tier1 implementation is not a "close enough" approximation: it follows the
Slaney definition (linear below 1 kHz, logarithmic above; each triangle scaled
by ``2 / (f_{i+2} - f_i)``) so that the two tiers agree to within floating-point
noise on the parameters this project uses.  The measured deviation is reported by
``tests/test_frontend.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.backend import BackendInfo, mel_fingerprint, require_librosa, resolve_dsp
from ..core.errors import FeatureError
from ..core.types import FLOAT_DTYPE

__all__ = [
    "FrontendConfig",
    "MelBank",
    "mel_filterbank",
    "hz_to_mel",
    "mel_to_hz",
    "dct_matrix",
    "mfcc",
    "regression_deltas",
    "cmvn",
    "lifter",
    "preemphasis",
    "frame_signal",
    "n_frames_for",
    "power_spectrum",
    "log_mel_spectrogram",
    "Frontend",
]


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FrontendConfig:
    """Immutable front-end parameters (mirrors the ``audio`` config group)."""

    sample_rate: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    fmin: float = 20.0
    fmax: float = 7600.0
    n_mfcc: int = 20
    keep_c0: bool = False
    lifter: float = 22.0
    preemph: float = 0.97
    n_deltas: int = 2
    delta_width: int = 2
    cmvn: bool = True
    backend: str = "auto"
    log_floor: float = 1e-10

    @property
    def frame_length(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def hop_length(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))

    @property
    def n_out(self) -> int:
        """Feature dimension after delta expansion."""
        return self.n_mfcc * (1 + self.n_deltas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "n_fft": self.n_fft,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "n_mfcc": self.n_mfcc,
            "keep_c0": self.keep_c0,
            "lifter": self.lifter,
            "preemph": self.preemph,
            "n_deltas": self.n_deltas,
            "delta_width": self.delta_width,
            "cmvn": self.cmvn,
            "backend": self.backend,
        }


# --------------------------------------------------------------------------
# mel scale + filterbank
# --------------------------------------------------------------------------
def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """Convert Hz to the Slaney mel scale (linear below 1000 Hz, log above)."""
    hz = np.asarray(hz, dtype=FLOAT_DTYPE)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mel = np.where(
        hz >= min_log_hz,
        min_log_mel + np.log(np.maximum(hz, 1e-10) / min_log_hz) / logstep,
        hz / f_sp,
    )
    return mel


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`hz_to_mel`."""
    mel = np.asarray(mel, dtype=FLOAT_DTYPE)
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(
        mel >= min_log_mel,
        min_log_hz * np.exp(logstep * (mel - min_log_mel)),
        mel * f_sp,
    )


def _mel_filterbank_numpy(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """Tier-1 Slaney-normalised triangular mel filterbank (pure numpy).

    Triangles are built on the mel scale, converted back to Hz, and each row is
    scaled by ``2 / (f[i+2] - f[i])`` -- the Slaney normalisation that keeps
    constant-energy response across the filterbank.  Rows whose support is
    entirely below ``fmin`` are dropped and the count is padded back with zeros
    so the output is always exactly ``(n_mels, n_fft//2 + 1)``.
    """
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_bins, dtype=FLOAT_DTYPE)

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2, dtype=FLOAT_DTYPE)
    hz = mel_to_hz(mels)

    fb = np.zeros((n_mels, n_bins), dtype=FLOAT_DTYPE)
    for i in range(n_mels):
        left, centre, right = hz[i], hz[i + 1], hz[i + 2]
        if right <= left:
            continue
        # Rising side.
        up = (fft_freqs - left) / max(centre - left, 1e-9)
        # Falling side.
        down = (right - fft_freqs) / max(right - centre, 1e-9)
        triangle = np.maximum(0.0, np.minimum(up, down))
        # Slaney area normalisation.
        fb[i] = triangle * (2.0 / (right - left))
    return fb


@dataclass(frozen=True)
class MelBank:
    """A mel filterbank matrix plus its provenance and fingerprint."""

    matrix: np.ndarray
    backend_tag: str
    n_mels: int
    sample_rate: int
    n_fft: int
    fmin: float
    fmax: float

    def fingerprint(self) -> str:
        """Short hash of the exact filterbank realisation."""
        return mel_fingerprint(self.matrix)

    @property
    def n_bins(self) -> int:
        return int(self.matrix.shape[1])

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_tag": self.backend_tag,
            "n_mels": self.n_mels,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "fingerprint": self.fingerprint(),
            "shape": list(self.matrix.shape),
        }

    def apply(self, power: np.ndarray) -> np.ndarray:
        """Project a ``(n_frames, n_bins)`` power spectrum onto the filterbank."""
        return np.asarray(power, dtype=FLOAT_DTYPE) @ self.matrix.T


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float,
    *,
    backend: str = "auto",
    info: BackendInfo | None = None,
) -> MelBank:
    """Build a mel filterbank using the requested backend.

    ``backend="tier1"`` (or ``force_tier1``) never touches librosa, which is
    what ``tests/test_backend.py`` asserts by inspecting ``sys.modules``.
    """
    resolved = info or resolve_dsp(backend)
    if resolved.resolved == "tier0":
        librosa = require_librosa()
        try:
            matrix = librosa.filters.mel(
                sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=False, norm="slaney"
            )
        except (ImportError, OSError, ValueError) as exc:
            raise FeatureError(
                "librosa mel filterbank construction failed",
                reason=f"{type(exc).__name__}: {exc}",
                remedy="use backend='tier1'",
            ) from exc
        matrix = np.ascontiguousarray(np.asarray(matrix, dtype=FLOAT_DTYPE))
        tag = "librosa"
    else:
        matrix = _mel_filterbank_numpy(sample_rate, n_fft, n_mels, fmin, fmax)
        tag = "numpy"

    if matrix.shape != (n_mels, n_fft // 2 + 1):
        raise FeatureError(
            "unexpected mel filterbank shape",
            expected=[n_mels, n_fft // 2 + 1],
            got=list(matrix.shape),
            backend_tag=tag,
        )
    return MelBank(
        matrix=matrix,
        backend_tag=tag,
        n_mels=n_mels,
        sample_rate=sample_rate,
        n_fft=n_fft,
        fmin=fmin,
        fmax=fmax,
    )


# --------------------------------------------------------------------------
# framing + spectrum
# --------------------------------------------------------------------------
def n_frames_for(n_samples: int, frame_length: int, hop_length: int) -> int:
    """Number of frames, matching the "no padding, no centring" convention.

    ``1 + floor((n_samples - frame_length) / hop)``, floored at 0.  This exact
    formula is asserted in ``tests/test_frontend.py`` because an off-by-one
    here silently shifts every downstream frame index.
    """
    if frame_length <= 0 or hop_length <= 0:
        raise FeatureError("frame_length and hop_length must be positive", frame_length=frame_length, hop_length=hop_length)
    if n_samples < frame_length:
        return 0
    return 1 + (n_samples - frame_length) // hop_length


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split into overlapping frames via stride tricks (no copy).

    Returns ``(n_frames, frame_length)``.  Raises when the signal is shorter
    than one frame, which is the caller's cue to lengthen the utterance rather
    than silently emitting a single padded frame.
    """
    x = np.asarray(x, dtype=FLOAT_DTYPE)
    n = n_frames_for(x.shape[0], frame_length, hop_length)
    if n <= 0:
        raise FeatureError(
            "signal shorter than one analysis frame",
            n_samples=int(x.shape[0]),
            frame_length=frame_length,
            required_samples=frame_length,
        )
    strides = (x.strides[0] * hop_length, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=(n, frame_length), strides=strides, writeable=False)


def preemphasis(x: np.ndarray, coeff: float) -> np.ndarray:
    """First-order pre-emphasis filter ``y[n] = x[n] - coeff * x[n-1]``."""
    if coeff == 0.0:
        return np.asarray(x, dtype=FLOAT_DTYPE).copy()
    x = np.asarray(x, dtype=FLOAT_DTYPE)
    out = np.empty_like(x)
    out[0] = x[0]
    out[1:] = x[1:] - coeff * x[:-1]
    return out


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Hamming-windowed real FFT power spectrum.

    ``frames`` is ``(n_frames, frame_length)``; if ``frame_length < n_fft`` the
    frames are zero-padded (and the window is truncated accordingly) so that the
    declared ``n_fft`` is always honoured.
    """
    frames = np.asarray(frames, dtype=FLOAT_DTYPE)
    n_frames, frame_length = frames.shape
    if frame_length > n_fft:
        raise FeatureError("frame_length exceeds n_fft", frame_length=frame_length, n_fft=n_fft)
    window = np.hamming(frame_length).astype(FLOAT_DTYPE)
    windowed = frames * window[None, :]
    if frame_length < n_fft:
        padded = np.zeros((n_frames, n_fft), dtype=FLOAT_DTYPE)
        padded[:, :frame_length] = windowed
        windowed = padded
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    return (np.abs(spec) ** 2).astype(FLOAT_DTYPE)


def log_mel_spectrogram(
    x: np.ndarray,
    sample_rate: int,
    *,
    n_fft: int = 512,
    n_mels: int = 40,
    fmin: float = 20.0,
    fmax: float = 7600.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    backend: str = "auto",
    bank: MelBank | None = None,
    log_floor: float = 1e-10,
) -> np.ndarray:
    """Convenience helper: waveform -> ``(n_frames, n_mels)`` log-mel matrix.

    Used by the synthesis separability check and by diagnostics; the main
    feature path goes through :class:`Frontend` so the filterbank is built once.
    """
    frame_length = int(round(sample_rate * frame_ms / 1000.0))
    hop_length = int(round(sample_rate * hop_ms / 1000.0))
    if x.shape[0] < frame_length:
        return np.zeros((0, n_mels), dtype=FLOAT_DTYPE)
    frames = frame_signal(x, frame_length, hop_length)
    power = power_spectrum(frames, n_fft)
    fb = bank or mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, backend=backend)
    mel = power @ fb.matrix.T
    return np.log(np.maximum(mel, log_floor)).astype(FLOAT_DTYPE)


# --------------------------------------------------------------------------
# DCT + MFCC
# --------------------------------------------------------------------------
def dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape ``(n_out, n_in)``.

    Implements the textbook orthonormal scaling so that ``D @ D.T == I``
    (verified in the tests).  ``scipy.fft.dct(type=2, norm="ortho")`` is
    deliberately *not* used: we need the matrix itself to project arbitrary
    sub-bands (VTLN) without re-running a transform.
    """
    if n_out <= 0 or n_in <= 0:
        raise FeatureError("dct dimensions must be positive", n_out=n_out, n_in=n_in)
    n = np.arange(n_in, dtype=FLOAT_DTYPE)
    k = np.arange(n_out, dtype=FLOAT_DTYPE)[:, None]
    mat = np.cos(np.pi * (n + 0.5) * k / n_in)
    mat *= math.sqrt(2.0 / n_in)
    # DC row is not doubled.
    mat[0] *= 1.0 / math.sqrt(2.0)
    return np.ascontiguousarray(mat, dtype=FLOAT_DTYPE)


def lifter(cepstra: np.ndarray, l: float) -> np.ndarray:
    """Cephalometric liftering: ``1 + (L/2) sin^2(pi k / L)``."""
    if l <= 0.0:
        return cepstra
    n = cepstra.shape[1]
    k = np.arange(n, dtype=FLOAT_DTYPE)
    weights = 1.0 + (l / 2.0) * np.sin(np.pi * k / l) ** 2
    return (cepstra * weights[None, :]).astype(FLOAT_DTYPE)


def mfcc(
    log_mel: np.ndarray, n_mfcc: int, *, keep_c0: bool = False, lifter_l: float = 0.0
) -> np.ndarray:
    """Project log-mel energies onto the first ``n_mfcc`` DCT-II coefficients.

    With ``keep_c0=False`` the c0 coefficient (overall log energy) is computed
    and then **dropped**, because it encodes recording loudness rather than
    vocal tract shape and would otherwise dominate every distance.
    """
    if log_mel.ndim != 2:
        raise FeatureError("log_mel must be 2-D", shape=list(log_mel.shape))
    n_bands = log_mel.shape[1]
    n_take = n_mfcc if keep_c0 else min(n_mfcc + 1, n_bands)
    dct = dct_matrix(n_take, n_bands)
    out = log_mel @ dct.T
    if not keep_c0 and n_take > 1:
        out = out[:, 1:]
    return np.ascontiguousarray(out, dtype=FLOAT_DTYPE)


def regression_deltas(feat: np.ndarray, width: int = 2, order: int = 1) -> np.ndarray:
    """Regression coefficients over ``+/-width`` frames.

    Implemented as the least-squares slope over the window, matching the HTK
    convention.  Edges are handled by *replicating* the boundary frame, which
    keeps the output length equal to the input length (a length change here
    would break the frame-count invariant).
    """
    if width < 1:
        raise FeatureError("delta width must be >= 1", width=width)
    if feat.shape[0] <= 1:
        return np.zeros_like(feat)
    offsets = np.arange(-width, width + 1, dtype=FLOAT_DTYPE)
    denom = float(np.sum(offsets**2))
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    out = np.zeros_like(feat)
    for i, off in enumerate(offsets.astype(int)):
        out += off * padded[i : i + feat.shape[0]]
    out /= denom
    if order == 2:
        # Second order: regress on the centred quadratic basis.
        centred = offsets - offsets.mean()
        basis = np.stack([centred, centred**2 - np.mean(centred**2)], axis=1)
        # Solve the 2x2 normal equations analytically.
        a11 = float(np.sum(basis[:, 0] ** 2))
        a12 = float(np.sum(basis[:, 0] * basis[:, 1]))
        a22 = float(np.sum(basis[:, 1] ** 2))
        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-12:
            return np.zeros_like(feat)
        w1 = np.array([a22, -a12]) / det
        w2 = np.array([-a12, a11]) / det
        out2 = np.zeros_like(feat)
        for i in range(basis.shape[0]):
            out2 += basis[i, 0] * w1[0] * padded[i : i + feat.shape[0]] + basis[i, 1] * w2[1] * padded[
                i : i + feat.shape[0]
            ]
        return np.ascontiguousarray(out2, dtype=FLOAT_DTYPE)
    return np.ascontiguousarray(out, dtype=FLOAT_DTYPE)


def cmvn(feat: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    """Per-utterance, per-coefficient mean/variance normalisation.

    This is **not** data leakage: the statistics come from the utterance being
    normalised, exactly as in per-utterance CMVN in an online recogniser.
    Global statistics (e.g. the global mean used by the weak baseline) are a
    different matter and are fitted on the training split only.
    """
    if feat.shape[0] == 0:
        return feat.copy()
    mu = np.mean(feat, axis=0, keepdims=True)
    sd = np.std(feat, axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return np.ascontiguousarray((feat - mu) / sd, dtype=FLOAT_DTYPE)


# --------------------------------------------------------------------------
# the front end
# --------------------------------------------------------------------------
class Frontend:
    """Waveform -> feature matrix, with a cached mel filterbank.

    The filterbank is built once per instance and reused across utterances,
    which is what keeps the demo inside its time budget.
    """

    def __init__(self, config: FrontendConfig | None = None, *, backend: str | None = None) -> None:
        cfg = config or FrontendConfig()
        if backend is not None:
            cfg = FrontendConfig(**{**cfg.to_dict(), "backend": backend})
        self.config = cfg
        self.backend_info = resolve_dsp(cfg.backend)
        self.bank = mel_filterbank(
            cfg.sample_rate, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax, info=self.backend_info
        )
        self._dct = dct_matrix(cfg.n_mfcc + (0 if cfg.keep_c0 else 1), cfg.n_mels)

    # -- properties ------------------------------------------------------
    @property
    def name(self) -> str:
        return f"mfcc{self.config.n_mfcc}x{self.config.n_deltas}_{self.bank.backend_tag}"

    @property
    def n_features(self) -> int:
        return self.config.n_out

    def fingerprint(self) -> str:
        """Fingerprint of the active mel filterbank (goes into benchmark.json)."""
        return self.bank.fingerprint()

    # -- pipeline --------------------------------------------------------
    def log_mel(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """Waveform -> ``(n_frames, n_mels)`` log-mel matrix (pre-DCT).

        This is the representation VTLN operates on.  VTLN warps the *frequency*
        axis, so it must be applied here, **before** the DCT-II: cepstral
        coefficients have no frequency interpretation and warping them is
        meaningless.
        """
        cfg = self.config
        sr = int(sample_rate if sample_rate is not None else cfg.sample_rate)
        x = np.asarray(audio, dtype=FLOAT_DTYPE)
        if x.ndim != 1:
            raise FeatureError("expected mono 1-D audio", shape=list(x.shape))
        if x.shape[0] < cfg.frame_length:
            return np.zeros((0, cfg.n_mels), dtype=FLOAT_DTYPE)
        x = preemphasis(x, cfg.preemph)
        frames = frame_signal(x, cfg.frame_length, cfg.hop_length)
        power = power_spectrum(frames, cfg.n_fft)
        return np.log(np.maximum(power @ self.bank.matrix.T, cfg.log_floor)).astype(FLOAT_DTYPE)

    def mfcc_from_log_mel(self, log_mel: np.ndarray) -> np.ndarray:
        """log-mel -> liftered MFCC, i.e. the post-DCT part of the front end."""
        cfg = self.config
        if log_mel.ndim != 2 or log_mel.shape[1] != cfg.n_mels:
            raise FeatureError("log_mel shape mismatch", expected=[None, cfg.n_mels], got=list(log_mel.shape))
        feats = log_mel @ self._dct.T
        if not cfg.keep_c0:
            feats = feats[:, 1:]
        if cfg.lifter > 0.0:
            feats = lifter(feats, cfg.lifter)
        return np.ascontiguousarray(feats, dtype=FLOAT_DTYPE)

    def expand(self, feats: np.ndarray) -> np.ndarray:
        """Apply CMVN and delta expansion to an MFCC matrix."""
        cfg = self.config
        if feats.shape[0] == 0:
            return np.zeros((0, cfg.n_out), dtype=FLOAT_DTYPE)
        if cfg.cmvn:
            feats = cmvn(feats)
        if cfg.n_deltas > 0:
            parts = [feats]
            for order in range(1, cfg.n_deltas + 1):
                parts.append(regression_deltas(feats, cfg.delta_width, order=order))
            feats = np.concatenate(parts, axis=1)
        return np.ascontiguousarray(feats, dtype=FLOAT_DTYPE)

    def process(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """Full front end: waveform -> ``(n_frames, n_features)`` float64.

        Returns an empty ``(0, n_features)`` matrix when the signal is shorter
        than one frame, so callers can filter utterances without special cases.
        """
        cfg = self.config
        sr = int(sample_rate if sample_rate is not None else cfg.sample_rate)
        if sr != cfg.sample_rate:
            raise FeatureError("sample rate mismatch", expected=cfg.sample_rate, got=sr)
        x = np.asarray(audio, dtype=FLOAT_DTYPE)
        if x.ndim != 1:
            raise FeatureError("expected mono 1-D audio", shape=list(x.shape))
        if x.shape[0] < cfg.frame_length:
            return np.zeros((0, cfg.n_out), dtype=FLOAT_DTYPE)
        return self.expand(self.mfcc_from_log_mel(self.log_mel(x, sr)))

    def process_log_mel(self, audio: np.ndarray, sample_rate: int | None = None) -> np.ndarray:
        """Alias of :meth:`log_mel` (used by VTLN and by diagnostics)."""
        return self.log_mel(audio, sample_rate)
        frames = frame_signal(x, cfg.frame_length, cfg.hop_length)
        power = power_spectrum(frames, cfg.n_fft)
        return np.log(np.maximum(power @ self.bank.matrix.T, cfg.log_floor)).astype(FLOAT_DTYPE)

    def n_frames(self, n_samples: int) -> int:
        """Number of frames the front end will emit for ``n_samples``."""
        return n_frames_for(n_samples, self.config.frame_length, self.config.hop_length)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "mel_bank": self.bank.to_dict(),
            "n_features": self.n_features,
            "name": self.name,
            "backend": self.backend_info.to_dict(),
        }
