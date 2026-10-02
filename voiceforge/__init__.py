"""VoiceForge -- a fully deterministic, self-contained speaker verification system.

Author: 晨星 (Chenxing)
License: MIT

The very first executable statement in this module pins BLAS/OpenMP thread
counts to 1 *before* numpy is imported.  Multi-threaded BLAS reductions are
non-deterministic in the last bits, which would break the content-hash based
determinism guarantee (gate G5) and the EM log-likelihood monotonicity
invariant.  This ordering is therefore a hard requirement, not a style choice.
"""

from __future__ import annotations

from .core.seed import preset_threads as _preset_threads

_preset_threads(1)

__version__ = "0.4.0"
__author__ = "晨星"
__license__ = "MIT"

#: Schema version of the emitted ``benchmark.json`` document.
SCHEMA_VERSION = "0.4.0"

__all__ = ["__version__", "__author__", "__license__", "SCHEMA_VERSION"]
