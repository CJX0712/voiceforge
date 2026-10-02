"""Immutable data containers shared across layers.

Every dataclass here is ``frozen=True`` and uses ``tuple`` / ``Mapping``
instead of ``list`` / ``dict`` for containers, so a value can be shared across
modules without any risk of aliasing mutation.  Arrays are exposed through
:meth:`FrozenArray.ro` which flips ``writeable=False``; that converts a whole
class of "someone modified the UBM in place" bugs into loud exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .errors import DtypeError

__all__ = [
    "FLOAT_DTYPE",
    "FrozenArray",
    "AudioClip",
    "SpeakerProfile",
    "Utterance",
    "Corpus",
    "Trial",
    "SystemScore",
    "MetricResult",
    "DatasetSpec",
    "SystemSpec",
    "StageTiming",
    "as_float64",
    "check_float64",
]

#: The one and only floating point dtype used for model maths.
FLOAT_DTYPE = np.float64


def as_float64(arr: Any, *, name: str = "array") -> np.ndarray:
    """Return ``arr`` as a contiguous float64 array (copying if needed).

    The dtype lock is deliberately a *conversion* helper rather than an
    assertion: callers that legitimately produce float32 (e.g. STFT output)
    get promoted once, at the boundary, instead of leaking float32 into the
    EM recursion where it would silently change the last mantissa bits.
    """
    out = np.asarray(arr, dtype=FLOAT_DTYPE)
    if out.dtype != FLOAT_DTYPE:  # pragma: no cover - defensive
        raise DtypeError(f"{name} must be {FLOAT_DTYPE}", name=name, dtype=str(out.dtype))
    return np.ascontiguousarray(out)


def check_float64(arr: np.ndarray, *, name: str = "array") -> None:
    """Raise :class:`DtypeError` unless ``arr`` is float64.

    Note: ``np.dtype.__name__`` does not exist on a dtype *instance* in numpy
    2.x (``np.float64.__name__`` does, but ``np.dtype('float64').__name__``
    raises).  ``str(dtype)`` is the portable spelling and is what we use.
    """
    if not isinstance(arr, np.ndarray):
        raise DtypeError(f"{name} must be an ndarray, got {type(arr).__name__}", name=name)
    if arr.dtype != FLOAT_DTYPE:
        raise DtypeError(
            f"{name} must be {np.dtype(FLOAT_DTYPE).name}, got {arr.dtype.name}", name=name, dtype=arr.dtype.name
        )


def _ro(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    view = arr.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class FrozenArray:
    """Read-only ndarray wrapper with a stable ``content_hash``."""

    data: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", np.ascontiguousarray(self.data))

    def ro(self) -> np.ndarray:
        """Return a non-writeable view of the wrapped array."""
        return _ro(self.data)  # type: ignore[return-value]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return int(self.data.shape[0]) if self.data.ndim else 0

    def stats(self) -> dict[str, float]:
        """Cheap summary used by the debug log and the report."""
        d = self.data
        if d.size == 0:
            return {"size": 0}
        return {
            "size": int(d.size),
            "mean": float(np.mean(d)),
            "std": float(np.std(d)),
            "min": float(np.min(d)),
            "max": float(np.max(d)),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"FrozenArray(shape={self.shape}, dtype={self.dtype.name})"


@dataclass(frozen=True)
class AudioClip:
    """A mono waveform at a known sample rate."""

    samples: FrozenArray
    sample_rate: int
    utterance_id: str
    speaker_id: str

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.samples.data.ndim != 1:
            raise ValueError(f"expected mono 1-D audio, got shape {self.samples.data.shape}")

    @property
    def duration(self) -> float:
        return float(self.samples.data.shape[0]) / float(self.sample_rate)

    def ro(self) -> np.ndarray:
        return self.samples.ro()  # type: ignore[return-value]


@dataclass(frozen=True)
class SpeakerProfile:
    """Ground-truth physical parameters of a synthetic speaker.

    These are the *latent* source/filter parameters, not measurements.  They
    exist so the corpus builder can report the intended F0/formant layout and
    so tests can assert separability against known ground truth.

    ``fricative_pattern`` is the speaker's *characteristic* frication
    signature -- a fixed tuple of ``(relative_position, centre_hz, bandwidth)``
    triples -- and ``n_syllables`` fixes their rhythm.  Both are properties of
    the **speaker**, not of each utterance, which is what keeps the
    within-speaker spectral covariance tight.  See the note in
    ``data.synthesis.synthesize_utterance`` for the measurement that
    motivated this.
    """

    speaker_id: str
    f0_median: float
    f0_jitter: float
    formants: tuple[float, ...]
    bandwidths: tuple[float, ...]
    vocal_tract_length_cm: float
    fricative_pattern: tuple[tuple[float, float, float], ...] = ()
    n_syllables: int = 5

    @property
    def n_formants(self) -> int:
        return len(self.formants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "f0_median": self.f0_median,
            "f0_jitter": self.f0_jitter,
            "formants": list(self.formants),
            "bandwidths": list(self.bandwidths),
            "vocal_tract_length_cm": self.vocal_tract_length_cm,
            "fricative_pattern": [list(p) for p in self.fricative_pattern],
            "n_syllables": self.n_syllables,
        }


@dataclass(frozen=True)
class Utterance:
    """One enrolled/test utterance: audio plus its speaker and conditions."""

    utterance_id: str
    speaker_id: str
    audio: AudioClip
    dataset_id: str
    seed: int
    profile: SpeakerProfile | None = None
    conditions: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.audio.duration

    def with_audio(self, audio: AudioClip) -> "Utterance":
        return Utterance(
            utterance_id=self.utterance_id,
            speaker_id=self.speaker_id,
            audio=audio,
            dataset_id=self.dataset_id,
            seed=self.seed,
            profile=self.profile,
            conditions=self.conditions,
        )


@dataclass(frozen=True)
class Corpus:
    """A cached, reproducible set of utterances for one (dataset, seed)."""

    dataset_id: str
    seed: int
    utterances: tuple[Utterance, ...]
    speakers: tuple[SpeakerProfile, ...]
    content_hash: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.utterances)

    def __iter__(self) -> Iterator[Utterance]:
        return iter(self.utterances)

    def by_speaker(self) -> dict[str, tuple[Utterance, ...]]:
        """Group utterances by speaker id, preserving order."""
        out: dict[str, list[Utterance]] = {}
        for utt in self.utterances:
            out.setdefault(utt.speaker_id, []).append(utt)
        return {k: tuple(v) for k, v in out.items()}

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(u.speaker_id for u in self.utterances))


@dataclass(frozen=True)
class Trial:
    """A single enrollment/test pair.

    ``label`` is ``True`` for genuine (same speaker) and ``False`` for impostor.
    """

    trial_id: int
    enroll_id: str
    test_id: str
    enroll_speaker: str
    test_speaker: str
    label: bool

    @property
    def is_genuine(self) -> bool:
        return self.label


@dataclass(frozen=True)
class SystemScore:
    """Scores for one system on one (dataset, seed), aligned with trials."""

    system_id: str
    dataset_id: str
    seed: int
    scores: FrozenArray
    labels: tuple[bool, ...]
    trial_ids: tuple[int, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.labels)
        if len(self.scores) != n:
            raise ValueError(f"scores/trials length mismatch: {len(self.scores)} vs {n}")
        if len(self.trial_ids) != n:
            raise ValueError(f"trial_ids length mismatch: {len(self.trial_ids)} vs {n}")

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return self.scores.ro(), np.asarray(self.labels, dtype=bool)  # type: ignore[return-value]


@dataclass(frozen=True)
class MetricResult:
    """A metric bundle for one system on one (dataset, seed)."""

    system_id: str
    dataset_id: str
    seed: int
    eer: float
    min_dcf: float
    auc: float
    n_genuine: int
    n_impostor: int
    threshold: float
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "eer": self.eer,
            "eer_pct": round(self.eer * 100.0, 6),
            "min_dcf": self.min_dcf,
            "auc": self.auc,
            "n_genuine": self.n_genuine,
            "n_impostor": self.n_impostor,
            "threshold": self.threshold,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of one point on the difficulty axis."""

    dataset_id: str
    snr_db: float | None
    noise: str
    channel: str
    enroll_sec: float
    test_sec: float
    expected_eer_range: tuple[float, float]
    description: str = ""
    channel_param: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "snr_db": self.snr_db,
            "noise": self.noise,
            "channel": self.channel,
            "channel_param": self.channel_param,
            "enroll_sec": self.enroll_sec,
            "test_sec": self.test_sec,
            "expected_eer_range": list(self.expected_eer_range),
            "description": self.description,
        }


@dataclass(frozen=True)
class SystemSpec:
    """Declarative description of one speaker-verification system."""

    system_id: str
    family: str
    uses_vtln: bool
    uses_lda: bool
    uses_gmm: bool
    uses_map: bool
    uses_ivector: bool
    uses_nbc: bool
    uses_plda: bool
    backend: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "family": self.family,
            "uses_vtln": self.uses_vtln,
            "uses_lda": self.uses_lda,
            "uses_gmm": self.uses_gmm,
            "uses_map": self.uses_map,
            "uses_ivector": self.uses_ivector,
            "uses_nbc": self.uses_nbc,
            "uses_plda": self.uses_plda,
            "backend": self.backend,
            "description": self.description,
        }


@dataclass(frozen=True)
class StageTiming:
    """Wall-clock record for one pipeline stage."""

    name: str
    seconds: float
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "seconds": round(self.seconds, 6), "extra": dict(self.extra)}


def sequence_to_tuple(values: Sequence[Any]) -> tuple[Any, ...]:
    """Freeze a sequence into a tuple (helper for dataclass construction)."""
    return tuple(values)
