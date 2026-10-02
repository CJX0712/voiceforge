"""Determinism primitives: thread pinning, named RNG substreams, hashing.

``core`` is forbidden from importing third-party modules at module scope, so
numpy is imported lazily inside the functions that need it.  That is what makes
the ordering constraint in ``voiceforge/__init__.py`` meaningful: the thread
environment variables must be set *before* the BLAS backend initialises, and
importing numpy at the top of this module would break that.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any, Iterable

__all__ = [
    "THREAD_ENV_VARS",
    "preset_threads",
    "set_all",
    "rng",
    "derive_seed",
    "content_hash",
    "canonical_json",
    "digest_obj",
    "current_thread_env",
]

#: Environment variables that control BLAS / OpenMP parallelism.  All of them
#: are forced to a single thread: multi-threaded reductions sum in a
#: non-deterministic order, which perturbs the last mantissa bits.
THREAD_ENV_VARS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def preset_threads(n: int = 1, *, force: bool = True) -> dict[str, str]:
    """Pin every BLAS/OpenMP thread pool to ``n`` threads.

    Also sets ``MKL_CBWR=COMPATIBLE`` which forces MKL to use the plain
    reference code path instead of the CPU-specific ``AVX512`` auto-tuned
    kernels.  Those kernels change the *summation order* between CPU models,
    which would make results irreproducible across machines.

    Parameters
    ----------
    n:
        Thread count.  ``1`` is the only value that yields bit-exact
        reproducibility.
    force:
        When ``True`` (default) existing values are overwritten.

    Returns
    -------
    dict
        The environment variables that were set.
    """
    if n < 1:
        raise ValueError(f"thread count must be >= 1, got {n}")
    applied: dict[str, str] = {}
    for key in THREAD_ENV_VARS:
        value = str(n)
        if force or key not in os.environ:
            os.environ[key] = value
        applied[key] = os.environ[key]
    # Conditional Numerical Reproducibility: consistent, non-crisp summation.
    os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    applied["MKL_CBWR"] = os.environ["MKL_CBWR"]
    applied["PYTHONHASHSEED"] = os.environ["PYTHONHASHSEED"]
    return applied


def current_thread_env() -> dict[str, str]:
    """Return the current values of the thread-pinning variables."""
    keys = (*THREAD_ENV_VARS, "MKL_CBWR", "PYTHONHASHSEED")
    return {k: os.environ.get(k, "<unset>") for k in keys}


def set_all(seed: int) -> None:
    """Seed the global ``random`` and ``numpy.random`` generators.

    Note that VoiceForge code always uses *named substreams* via :func:`rng`
    rather than the global generator.  Seeding the global generator is still
    done so that third-party code called from tests behaves predictably.
    """
    import numpy as np

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))


def derive_seed(seed: int, stream: str) -> int:
    """Derive a stable 64-bit substream seed from ``(seed, stream)``.

    Uses BLAKE2b rather than Python's ``hash()`` because the latter is salted
    per process unless ``PYTHONHASHSEED`` is pinned -- and even then it is an
    implementation detail we refuse to depend on.  The mapping is a pure
    function, so it is stable across runs, machines, and Python versions.
    """
    payload = f"{int(seed)}::{stream}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=False)


def rng(seed: int, stream: str = "default"):
    """Return a ``numpy.random.Generator`` for the named substream.

    Each logical component of the system (synthesis, KMeans init, EM jitter,
    trial sampling, ...) draws from its *own* stream.  Adding a new consumer
    therefore never perturbs the numbers seen by existing ones, which is what
    makes the benchmark reproducible across code revisions.
    """
    import numpy as np

    return np.random.default_rng(derive_seed(seed, stream))


def canonical_json(obj: Any) -> str:
    """Serialise ``obj`` to a canonical JSON string.

    Canonical means: sorted keys, no insignificant whitespace, UTF-8 literals
    preserved, and non-finite floats rejected.  Two structurally equal objects
    always produce byte-identical output.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def _json_default(obj: Any) -> Any:
    """Fallback encoder for numpy scalars/arrays and sets."""
    import numpy as np

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__fspath__"):
        return str(obj)
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def digest_obj(obj: Any, *, size: int = 16) -> str:
    """Return a truncated hex digest of the canonical JSON form of ``obj``."""
    payload = canonical_json(obj).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:size]


def content_hash(obj: Any, *, size: int = 16) -> str:
    """Stable content hash of a JSON-serialisable object.

    This is the backbone of gate G5.  Benchmark documents must exclude
    wall-clock fields (``created_at``, ``run_id``, ``timings``, ``git_sha``)
    before hashing, otherwise no two runs could ever agree.
    """
    return digest_obj(obj, size=size)


def hash_many(parts: Iterable[Any], *, size: int = 16) -> str:
    """Hash an ordered iterable of objects as if they were one structure."""
    return content_hash(list(parts), size=size)
