"""Abstract interfaces (structural contracts) for the pipeline stages.

These are ``typing.Protocol`` classes rather than ``abc.ABC`` base classes: the
concrete implementations in ``data/``, ``sv/``, ``eval/`` do not need to import
or inherit from anything here, which keeps the dependency arrows pointing
inwards (``cli -> pipeline -> {data, hpo, training, sv, eval} -> core``) and
makes it trivial to substitute a stub in tests.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import Config
from .types import Corpus, MetricResult, SystemScore, Trial, Utterance

__all__ = [
    "Synthesizer",
    "CorpusBuilder",
    "FeatureExtractor",
    "Trainer",
    "EnrollmentPipeline",
    "Scorer",
    "Evaluator",
    "MelBank",
]


@runtime_checkable
class MelBank(Protocol):
    """A mel filterbank matrix plus its provenance."""

    matrix: np.ndarray
    backend_tag: str

    def fingerprint(self) -> str:
        """Short hash identifying the exact filterbank realisation."""
        ...


@runtime_checkable
class Synthesizer(Protocol):
    """Turns a speaker profile + conditions into a waveform."""

    def synthesize(self, speaker: Any, conditions: Mapping[str, Any], seed: int) -> np.ndarray:
        """Return a mono float64 waveform at the configured sample rate."""
        ...


@runtime_checkable
class CorpusBuilder(Protocol):
    """Produces (and caches) a :class:`Corpus` for one (dataset, seed)."""

    def build(self, dataset_id: str, seed: int) -> Corpus:
        """Return a reproducible corpus; may serve from cache."""
        ...


@runtime_checkable
class FeatureExtractor(Protocol):
    """Waveform -> frame-level feature matrix."""

    name: str

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return an ``(n_frames, n_features)`` float64 matrix."""
        ...

    def process_utterance(self, utterance: Utterance) -> np.ndarray:
        """:meth:`process` bound to an utterance's audio and sample rate."""
        ...


@runtime_checkable
class Trainer(Protocol):
    """Fits a model on training utterances only."""

    def fit(self, features: Sequence[np.ndarray], config: Config) -> Any:
        """Fit and return a model handle."""
        ...


@runtime_checkable
class EnrollmentPipeline(Protocol):
    """Produces a speaker embedding from one or more utterances."""

    name: str

    def enroll(self, features: np.ndarray) -> np.ndarray:
        """Return the speaker representation (embedding / supervector / MAP)."""
        ...


@runtime_checkable
class Scorer(Protocol):
    """Computes a similarity score where **higher means more similar**."""

    name: str

    def score(self, enroll: Any, test: Any) -> float:
        """Return a scalar similarity."""
        ...

    def score_trials(self, enrolls: Sequence[Any], tests: Sequence[Any]) -> np.ndarray:
        """Vectorised :meth:`score` over aligned sequences."""
        ...


@runtime_checkable
class Evaluator(Protocol):
    """Turns raw scores into metrics."""

    def evaluate(self, scores: np.ndarray, labels: np.ndarray) -> MetricResult:
        """Return EER / minDCF / AUC for aligned scores and boolean labels."""
        ...

    def run(self, trials: Sequence[Trial], scores: np.ndarray) -> SystemScore:
        """Bundle scores with their trial metadata."""
        ...
