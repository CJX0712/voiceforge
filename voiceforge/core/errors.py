"""Typed error hierarchy with stable numeric codes.

Every error carries a short string ``code`` (e.g. ``"E403"``) and a ``context``
mapping with structured details.  Codes are grouped by layer:

======  ==========================================================
Range   Meaning
======  ==========================================================
E1xx    Configuration / schema violations
E2xx    Data layer (synthesis, corpora, IO)
E3xx    Feature / frontend layer
E4xx    Model / math layer (EM, dtype, shapes, linear algebra)
E5xx    Evaluation layer (metrics, trials, trials-set emptiness)
E6xx    Pipeline / orchestration
E7xx    Backend / environment availability
======  ==========================================================

``core`` must stay free of third-party imports, so this module only uses the
standard library.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "VoiceForgeError",
    "ConfigError",
    "UnknownEnvKeyError",
    "SchemaValidationError",
    "DataError",
    "SynthesisError",
    "CorpusError",
    "FeatureError",
    "ModelError",
    "EMNotConverged",
    "DtypeError",
    "ShapeError",
    "RankDeficientError",
    "EvalError",
    "EmptyTrialSet",
    "MetricError",
    "PipelineError",
    "BackendNotAvailable",
    "DspBackendError",
]


def _render(value: Any, limit: int = 200) -> str:
    """Render ``value`` compactly for error messages."""
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - defensive
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


class VoiceForgeError(Exception):
    """Base class for every error raised by VoiceForge.

    Parameters
    ----------
    message:
        Human readable description.
    context:
        Structured payload attached to the error.  Must be JSON-ish so that it
        can be serialised into ``benchmark.json`` without custom encoders.
    """

    #: Default code for the class; concrete subclasses override.
    code: str = "E000"

    def __init__(self, message: str, /, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the error."""
        return {
            "code": self.code,
            "type": type(self).__name__,
            "message": self.message,
            "context": {k: _render(v) for k, v in sorted(self.context.items())},
        }

    def __str__(self) -> str:
        if not self.context:
            return f"[{self.code}] {self.message}"
        detail = ", ".join(f"{k}={_render(v, 80)}" for k, v in sorted(self.context.items()))
        return f"[{self.code}] {self.message} ({detail})"


# --------------------------------------------------------------------------
# E1xx -- configuration
# --------------------------------------------------------------------------
class ConfigError(VoiceForgeError):
    """Configuration is missing, malformed, or self-contradictory."""

    code = "E100"


class UnknownEnvKeyError(ConfigError):
    """An ``ENV_VOICEFORGE_*`` variable does not map onto the schema.

    Raised eagerly so that a typo such as ``ENV_VOICEFORGE_GMM_N_GSAUSS`` fails
    loudly instead of silently leaving the default in place.
    """

    code = "E101"


class SchemaValidationError(ConfigError):
    """A configuration value failed type / range / exclusivity validation."""

    code = "E102"


# --------------------------------------------------------------------------
# E2xx -- data
# --------------------------------------------------------------------------
class DataError(VoiceForgeError):
    """Base class for data-layer failures."""

    code = "E200"


class SynthesisError(DataError):
    """Waveform synthesis violated one of its own invariants."""

    code = "E201"


class CorpusError(DataError):
    """Corpus construction, caching, or manifest verification failed."""

    code = "E202"


# --------------------------------------------------------------------------
# E3xx -- features
# --------------------------------------------------------------------------
class FeatureError(VoiceForgeError):
    """Base class for frontend failures."""

    code = "E300"


# --------------------------------------------------------------------------
# E4xx -- model / math
# --------------------------------------------------------------------------
class ModelError(VoiceForgeError):
    """Base class for modelling failures."""

    code = "E400"


class EMNotConverged(ModelError):
    """EM failed to reach the requested tolerance within ``max_iter``."""

    code = "E401"


class DtypeError(ModelError):
    """A numeric array violated the hard dtype lock (everything is float64)."""

    code = "E403"


class ShapeError(ModelError):
    """Array shapes are incompatible for the requested operation."""

    code = "E404"


class RankDeficientError(ModelError):
    """A matrix that must be full rank is rank deficient (needs more data)."""

    code = "E405"


# --------------------------------------------------------------------------
# E5xx -- evaluation
# --------------------------------------------------------------------------
class EvalError(VoiceForgeError):
    """Base class for evaluation failures."""

    code = "E500"


class EmptyTrialSet(EvalError):
    """The trial list is empty; EER / ROC are undefined."""

    code = "E501"


class MetricError(EvalError):
    """A metric received degenerate input (all identical scores, NaN, ...)."""

    code = "E502"


# --------------------------------------------------------------------------
# E6xx -- pipeline
# --------------------------------------------------------------------------
class PipelineError(VoiceForgeError):
    """The orchestration layer could not complete a stage."""

    code = "E600"


# --------------------------------------------------------------------------
# E7xx -- backend / environment
# --------------------------------------------------------------------------
class BackendNotAvailable(VoiceForgeError):
    """A requested DSP backend (or one of its dependencies) is unavailable.

    On Windows a missing DLL surfaces as ``ImportError`` *or* ``OSError``
    (``ImportError: DLL load failed``); both are mapped to this error.
    """

    code = "E700"


class DspBackendError(BackendNotAvailable):
    """The DSP backend is present but failed while building a filterbank."""

    code = "E701"


def error_to_dict(exc: BaseException) -> Mapping[str, Any]:
    """Best-effort conversion of any exception into the error envelope."""
    if isinstance(exc, VoiceForgeError):
        return exc.to_dict()
    return {
        "code": "E999",
        "type": type(exc).__name__,
        "message": str(exc),
        "context": {},
    }
