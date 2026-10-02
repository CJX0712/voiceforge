"""Logging helpers that survive a Windows console.

``print()`` of a check-mark or a warning triangle raises
``UnicodeEncodeError`` on a legacy code page (``cp936``/``cp1252``).  The demo
and the CLI both emit such glyphs, so every entry point funnels through
:func:`setup_stdout`, which reconfigures stdout/stderr to UTF-8 and replaces
unsupported characters with ASCII equivalents instead of exploding.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, TextIO

__all__ = [
    "setup_stdout",
    "get_logger",
    "configure_logging",
    "stage_timer",
    "StageTimer",
    "safe_ascii",
    "TableWriter",
]

#: Glyphs used in the console reports and their ASCII fallbacks.
_GLYPH_FALLBACK = {
    "✅": "[OK]",
    "❌": "[FAIL]",
    "⚠️": "[WARN]",
    "⚠": "[WARN]",
    "ℹ": "[INFO]",
    "→": "->",
    "─": "-",
    "│": "|",
    "≥": ">=",
    "≤": "<=",
    "±": "+/-",
    "×": "x",
}


def safe_ascii(text: str) -> str:
    """Replace non-ASCII glyphs and characters the console cannot encode."""
    for glyph, fallback in _GLYPH_FALLBACK.items():
        text = text.replace(glyph, fallback)
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding, errors="strict")
        return text
    except (UnicodeEncodeError, LookupError):
        return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def setup_stdout(stream: TextIO | None = None) -> None:
    """Reconfigure stdout/stderr to UTF-8 so box-drawing glyphs survive.

    Falls back silently on streams that do not support ``reconfigure`` (e.g.
    when output is captured by pytest's ``capsys``).
    """
    for target in (stream,) if stream is not None else (sys.stdout, sys.stderr):
        if target is None:
            continue
        reconfigure = getattr(target, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError, io.UnsupportedOperation):  # pragma: no cover
            pass


def configure_logging(verbose: bool = False, *, stream: TextIO | None = None) -> None:
    """Install a single stderr handler with a compact deterministic format."""
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger("voiceforge")
    root.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``voiceforge`` namespace."""
    return logging.getLogger(f"voiceforge.{name}")


class StageTimer:
    """Accumulate per-stage wall-clock timings for the ``timings`` block."""

    def __init__(self) -> None:
        self._stages: list[tuple[str, float, dict[str, Any]]] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a block and attach arbitrary metadata to it.

        Yields a mutable dict; whatever the caller puts in it lands in the
        timing record (e.g. ``{"n_frames": 1234}``).
        """
        extra: dict[str, Any] = {}
        start = time.perf_counter()
        try:
            yield extra
        finally:
            self._stages.append((name, time.perf_counter() - start, extra))

    def add(self, name: str, seconds: float, **extra: Any) -> None:
        """Record a timing measured elsewhere."""
        self._stages.append((name, float(seconds), extra))

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, Any]:
        """Return ``{"stages": [...], "total_seconds": ...}``."""
        return {
            "stages": [
                {"name": n, "seconds": round(s, 6), "extra": e} for n, s, e in self._stages
            ],
            "total_seconds": round(self.total, 6),
        }

    def peak_rss_mb(self) -> float:
        """Peak resident set size in MiB.

        Delegates to :mod:`voiceforge.core.memory`, which implements the
        measurement with the standard library only (``psutil`` is not a
        dependency).  Returns ``0.0`` when no strategy succeeds, which callers
        must read as "unknown", not "zero".
        """
        from .memory import peak_rss_mb as _peak

        return _peak()


@contextmanager
def stage_timer(timer: StageTimer, name: str) -> Iterator[dict[str, Any]]:
    """Functional alias for ``StageTimer.stage``."""
    with timer.stage(name) as extra:
        yield extra


class TableWriter:
    """Fixed-width console table.

    The widths are fixed by the caller (and by the tests) rather than computed
    from the data, so a column never shifts between two runs of the same report
    -- which matters because the console output is diffed in CI.
    """

    def __init__(self, headers: list[str], widths: list[int], stream: TextIO | None = None) -> None:
        if len(headers) != len(widths):
            raise ValueError("headers and widths must have equal length")
        self.headers = headers
        self.widths = widths
        self.stream = stream or sys.stdout
        self._wrote_header = False

    def header(self) -> None:
        if self._wrote_header:
            return
        cells = " ".join(f"{h:<{w}}" for h, w in zip(self.headers, self.widths))
        rule = "-" * len(cells)
        print(safe_ascii(cells), file=self.stream)
        print(safe_ascii(rule), file=self.stream)
        self._wrote_header = True

    def row(self, cells: list[Any]) -> None:
        self.header()
        rendered = " ".join(f"{str(c):<{w}}" for c, w in zip(cells, self.widths))
        print(safe_ascii(rendered), file=self.stream)

    def rule(self, char: str = "-") -> None:
        total = sum(self.widths) + len(self.widths) - 1
        print(safe_ascii(char * total), file=self.stream)
