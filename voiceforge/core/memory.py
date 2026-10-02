"""Peak resident-set-size measurement (gate G6).

``psutil`` is not a declared dependency, so this module implements the
measurement with the standard library only.  Getting this right on 64-bit
Windows is subtle and cost real debugging time, so the traps are documented:

1. **Handle truncation.**  ``kernel32.GetCurrentProcess`` returns a pseudo
   handle whose value exceeds 32 bits.  If ``restype`` is left at the default
   ``c_int``, ctypes truncates the handle to 32 bits and every subsequent call
   fails with an invalid-handle error -- which surfaces as a silent ``0.0``
   rather than an exception.  ``restype`` **must** be ``wintypes.HANDLE``.
2. **Symbol name.**  Modern Windows exports
   ``K32GetProcessMemoryInfo`` from ``kernel32``; the old ``psapi`` export is a
   forwarder that some SDKs omit entirely.
3. **Unit differences.**  ``ru_maxrss`` is kilobytes on Linux and *bytes* on
   macOS/BSD.  The Linux ``/proc/self/status`` route (``VmHWM``) is preferred
   because it is exact and needs no ctypes at all.

Every path returns MiB, and ``0.0`` only when no strategy works -- callers must
treat ``0.0`` as "unknown", never as "zero megabytes".
"""

from __future__ import annotations

import os
import sys
from typing import Callable

__all__ = ["peak_rss_mb", "current_rss_mb", "rss_backend"]

_MIB = 1024.0 * 1024.0


def _from_proc_status() -> float:
    """Linux: read ``VmHWM`` (peak RSS) and ``VmRSS`` from /proc/self/status."""
    peak = 0.0
    current = 0.0
    with open("/proc/self/status", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                peak = float(line.split()[1]) / 1024.0
            elif line.startswith("VmRSS:"):
                current = float(line.split()[1]) / 1024.0
    if peak <= 0.0 and current <= 0.0:
        raise OSError("no VmHWM/VmRSS in /proc/self/status")
    return peak, current


def _from_windows() -> tuple[float, float]:
    """Windows: GetProcessMemoryInfo via kernel32 (see module docstring)."""
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # Trap 1: without an explicit HANDLE restype the pseudo-handle is truncated.
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []

    handle = kernel32.GetCurrentProcess()
    if not handle:
        raise OSError(f"GetCurrentProcess failed: {ctypes.get_last_error()}")

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)

    # Trap 2: prefer K32*, fall back to the psapi forwarder.
    func = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if func is None:  # pragma: no cover - very old Windows
        func = ctypes.WinDLL("psapi").GetProcessMemoryInfo
    func.restype = wintypes.BOOL
    func.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]

    if not func(handle, ctypes.byref(counters), counters.cb):
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return counters.PeakWorkingSetSize / _MIB, counters.WorkingSetSize / _MIB


def _from_resource() -> tuple[float, float]:
    """POSIX fallback: ``ru_maxrss`` (KiB on Linux, bytes on macOS/BSD)."""
    import resource

    peak_raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return peak_raw / _MIB, peak_raw / _MIB
    return peak_raw / 1024.0, peak_raw / 1024.0


_STRATEGIES: tuple[tuple[str, Callable[[], tuple[float, float]]], ...] = (
    ("proc_status", _from_proc_status),
    ("windows", _from_windows),
    ("resource", _from_resource),
)


def _measure() -> tuple[float, float, str]:
    """Return ``(peak_mb, current_mb, backend_name)`` using the first that works."""
    for name, func in _STRATEGIES:
        try:
            peak, current = func()
        except Exception:
            continue
        if peak > 0.0:
            return peak, current, name
    return 0.0, 0.0, "unavailable"


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB (``0.0`` if unknown)."""
    return _measure()[0]


def current_rss_mb() -> float:
    """Current resident set size of this process, in MiB (``0.0`` if unknown)."""
    return _measure()[1]


def rss_backend() -> str:
    """Name of the strategy that produced the measurement."""
    return _measure()[2]


def format_mb(value: float) -> str:
    """Human-readable MiB rendering that is stable across platforms."""
    if value <= 0.0:
        return "unknown"
    return f"{value:.1f} MiB"


def format_mib(value: float) -> str:  # pragma: no cover - alias
    """Backwards-compatible alias for :func:`format_mb`."""
    return format_mb(value)


def pid() -> int:
    """Current process id (kept here so tests can assert we measure ourselves)."""
    return os.getpid()
