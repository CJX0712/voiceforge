"""Core layer: errors, config, types, interfaces, determinism, backend, logging.

Import order matters inside this package: :mod:`~voiceforge.core.seed` must be
importable on its own (it is used by ``voiceforge/__init__.py`` before numpy is
loaded), and every other core module is pure standard library.
"""

from __future__ import annotations

from .errors import (
    BackendNotAvailable,
    ConfigError,
    CorpusError,
    DataError,
    DspBackendError,
    DtypeError,
    EMNotConverged,
    EvalError,
    FeatureError,
    MetricError,
    ModelError,
    PipelineError,
    RankDeficientError,
    SchemaValidationError,
    ShapeError,
    SynthesisError,
    UnknownEnvKeyError,
    VoiceForgeError,
)
from .seed import content_hash, preset_threads, rng, set_all

__all__ = [
    "BackendNotAvailable",
    "ConfigError",
    "CorpusError",
    "DataError",
    "DspBackendError",
    "DtypeError",
    "EMNotConverged",
    "EvalError",
    "FeatureError",
    "MetricError",
    "ModelError",
    "PipelineError",
    "RankDeficientError",
    "SchemaValidationError",
    "ShapeError",
    "SynthesisError",
    "UnknownEnvKeyError",
    "VoiceForgeError",
    "content_hash",
    "preset_threads",
    "rng",
    "set_all",
]
