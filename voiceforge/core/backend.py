"""Backend discovery and DSP tier resolution.

Why this module exists
----------------------
``librosa`` pulls in ``numba``, which pulls in a compiled extension.  On a
Windows box without MSVC and on Linux without a toolchain, importing it can
fail in three different ways:

1. ``ModuleNotFoundError`` -- not installed at all;
2. ``ImportError: DLL load failed`` -- the ``.pyd`` exists but a dependency is
   missing (raised as ``OSError`` on some Windows configurations);
3. ``AttributeError`` / ``ValueError`` at *decoration* time, because numba
   cannot find a supported CPU target.

All three must degrade to "tier1 available, tier0 unavailable" rather than
crashing the process at import time.  The detection is therefore split in two:

* :func:`available_librosa` uses :func:`importlib.util.find_spec` only -- it
  answers "could this be importable?" without executing any module code, so it
  is safe to call at import time;
* the actual ``import librosa`` is deferred to the moment a mel filterbank is
  really built (:func:`require_librosa`).

``tests/test_backend.py`` asserts that the tier1 path never touches
``sys.modules['librosa']``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import platform
import sys
from dataclasses import dataclass
from typing import Any

from .errors import BackendNotAvailable, DspBackendError

__all__ = [
    "TIERS",
    "BackendInfo",
    "available_librosa",
    "require_librosa",
    "resolve_dsp",
    "backend_fingerprint",
    "mel_fingerprint",
    "describe_backend",
]

#: Supported DSP tiers.  ``auto`` prefers tier0 and silently falls back.
TIERS: tuple[str, ...] = ("auto", "tier0", "tier1")


@dataclass(frozen=True)
class BackendInfo:
    """Resolved backend description (embedded verbatim in benchmark.json)."""

    requested: str
    resolved: str
    librosa_available: bool
    librosa_version: str
    librosa_error: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "librosa_available": self.librosa_available,
            "librosa_version": self.librosa_version,
            "librosa_error": self.librosa_error,
            "reason": self.reason,
        }


def available_librosa() -> tuple[bool, str]:
    """Probe for librosa *without importing it*.

    Returns ``(available, detail)`` where ``detail`` is either the discovered
    version string or a human-readable reason.  Uses ``find_spec`` so that no
    third-party code executes during the probe.
    """
    try:
        spec = importlib.util.find_spec("librosa")
    except (ImportError, ValueError) as exc:
        # ValueError: __spec__ is None (namespace package edge case).
        return False, f"find_spec failed: {type(exc).__name__}: {exc}"
    if spec is None:
        return False, "librosa not installed"

    # Confirm numba is present too: librosa 0.10+ imports it eagerly, and a
    # half-installed pair is the classic "ImportError: DLL load failed".
    try:
        numba_spec = importlib.util.find_spec("numba")
    except (ImportError, ValueError) as exc:  # pragma: no cover - defensive
        return False, f"numba probe failed: {type(exc).__name__}: {exc}"
    if numba_spec is None:
        return False, "librosa found but numba is missing (librosa>=0.10 requires it)"

    # Read the version without a full import.  importlib.metadata reads the
    # *.dist-info/METADATA file, which lives in site-packages -- NOT inside the
    # package directory that find_spec reports, so going by hand here would
    # always miss and silently return "unknown".
    version = "unknown"
    try:
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        try:
            version = _pkg_version("librosa")
        except PackageNotFoundError:
            version = "unknown"
    except Exception as exc:  # pragma: no cover - defensive
        version = f"unknown ({type(exc).__name__})"
    return True, version


def require_librosa() -> Any:
    """Import and return the ``librosa`` module, or raise ``E700``.

    This is the *only* place in the codebase allowed to ``import librosa``.
    Both ``ImportError`` and ``OSError`` are caught because Windows reports a
    missing dependent DLL as ``ImportError: DLL load failed`` while a missing
    runtime library can surface as a bare ``OSError``.
    """
    available, detail = available_librosa()
    if not available:
        raise BackendNotAvailable(
            "librosa backend requested but unavailable", reason=detail, remedy="use backend='tier1'"
        )
    try:
        import librosa  # noqa: PLC0415 - deliberately deferred
    except (ImportError, OSError, ValueError) as exc:
        raise BackendNotAvailable(
            "librosa found but failed to import",
            reason=f"{type(exc).__name__}: {exc}",
            remedy="use backend='tier1' (pure numpy)",
        ) from exc
    return librosa


def resolve_dsp(requested: str = "auto", *, force_tier1: bool = False) -> BackendInfo:
    """Resolve the requested DSP tier into a concrete tier.

    Parameters
    ----------
    requested:
        ``"auto"`` | ``"tier0"`` | ``"tier1"``.  ``tier0`` *demands* librosa and
        raises ``E700`` when it is missing -- silently downgrading an explicit
        request would make benchmarks incomparable.
    force_tier1:
        Bypass librosa entirely (used by the ablation/tier1 system and by the
        test that proves the tier1 path never imports librosa).
    """
    if requested not in TIERS:
        raise DspBackendError(f"unknown dsp backend {requested!r}", requested=requested, known=list(TIERS))

    available, detail = available_librosa()

    if force_tier1 or requested == "tier1":
        reason = "forced" if force_tier1 else "explicitly requested"
        if not force_tier1 and not available:
            reason = f"explicitly requested; {detail}"
        return BackendInfo(
            requested=requested, resolved="tier1", librosa_available=available,
            librosa_version=detail if available else "", librosa_error="" if available else detail, reason=reason,
        )

    if requested == "tier0":
        if not available:
            raise DspBackendError(
                "tier0 (librosa) requested but unavailable",
                reason=detail,
                remedy="install librosa, or use backend='tier1'",
            )
        return BackendInfo(
            requested=requested, resolved="tier0", librosa_available=True,
            librosa_version=detail, librosa_error="", reason="librosa available",
        )

    # auto
    if available:
        return BackendInfo(
            requested="auto", resolved="tier0", librosa_available=True,
            librosa_version=detail, librosa_error="", reason="librosa available, auto-selected tier0",
        )
    return BackendInfo(
        requested="auto", resolved="tier1", librosa_available=False,
        librosa_version="", librosa_error=detail, reason=f"auto-fallback: {detail}",
    )


def mel_fingerprint(fb: Any) -> str:
    """Return the first 12 hex chars of the sha1 of a mel filterbank matrix.

    The fingerprint is embedded in ``benchmark.json`` so that results obtained
    on tier0 and tier1 can be *proved* to have used the same filterbank (or to
    quantify exactly how much the two implementations differ).  Arrays are
    rounded to 1e-9 before hashing to make the value stable against
    last-bit noise while still catching any real algorithmic change.
    """
    import numpy as np

    arr = np.asarray(fb, dtype=np.float64)
    canonical = np.round(arr, decimals=9)
    payload = canonical.tobytes()
    return hashlib.sha1(payload).hexdigest()[:12]


def backend_fingerprint(info: BackendInfo | None = None) -> str:
    """Stable fingerprint of the runtime environment.

    Includes the resolved tier, the librosa version and the *interpreter*
    implementation, but deliberately excludes the OS release string: the same
    code on Windows and Linux must produce the same numerical results, and the
    OS must therefore not be part of a content hash.
    """
    if info is None:
        info = resolve_dsp("auto")
    payload = "|".join(
        [
            f"tier={info.resolved}",
            f"librosa={info.librosa_version or 'none'}",
            f"py={platform.python_implementation()}",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def describe_backend(info: BackendInfo | None = None) -> dict[str, Any]:
    """Full environment description for ``benchmark.json``'s ``env`` block."""
    import numpy as np

    if info is None:
        info = resolve_dsp("auto")

    def _ver(mod: str) -> str:
        try:
            m = __import__(mod)
            return str(getattr(m, "__version__", "unknown"))
        except (ImportError, OSError):  # pragma: no cover - optional deps
            return "unavailable"

    blas = "unknown"
    try:
        cfg = np.__config__.show(mode="dicts")  # type: ignore[call-arg]
        blas = str(cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"))
    except Exception:  # pragma: no cover - numpy build without config
        blas = "unknown"

    from .seed import current_thread_env

    from ..sv.frontend import mel_filterbank

    # The mel fingerprint is part of the environment block on purpose: it lets
    # a reader verify that two runs used the same filterbank, and quantify the
    # tier0/tier1 difference, instead of trusting the tier label.
    bank = mel_filterbank(16000, 512, 40, 20.0, 7600.0, info=info)

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "scipy": _ver("scipy"),
        "sklearn": _ver("sklearn"),
        "librosa": info.librosa_version if info.librosa_available else "unavailable",
        "soundfile": _ver("soundfile"),
        "blas": blas,
        "threads": current_thread_env(),
        "backend_tier": info.resolved,
        "dsp_backend": info.resolved,
        "backend_fingerprint": backend_fingerprint(info),
        "mel_fingerprint": bank.fingerprint(),
        "mel_backend_tag": bank.backend_tag,
        "platform_system": platform.system(),
        "python_exe": sys.executable,
    }
