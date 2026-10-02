"""Configuration, backend resolution and determinism-primitives invariants."""

from __future__ import annotations

import os

import numpy as np
import pytest

from voiceforge.core.backend import (
    available_librosa,
    backend_fingerprint,
    describe_backend,
    mel_fingerprint,
    resolve_dsp,
)
from voiceforge.core.config import DEFAULT_SCHEMA, SCHEMA_GROUPS, load_config
from voiceforge.core.errors import BackendNotAvailable, DspBackendError, SchemaValidationError, UnknownEnvKeyError
from voiceforge.core.memory import current_rss_mb, peak_rss_mb, rss_backend
from voiceforge.core.seed import content_hash, derive_seed, preset_threads, rng, set_all
from voiceforge.core.types import FrozenArray, check_float64
from voiceforge.core.errors import DtypeError


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_all_profiles_load():
    for profile in ("smoke", "demo", "bench"):
        cfg = load_config(profile, use_env=False)
        assert cfg.app.profile == profile
        assert cfg.runtime.threads == 1


def test_schema_declares_twelve_groups():
    assert len(SCHEMA_GROUPS) >= 12
    for group in ("app", "runtime", "audio", "vtln", "proj", "gmm", "map", "ivector", "plda", "score", "data", "bench", "gate"):
        assert group in DEFAULT_SCHEMA
        assert DEFAULT_SCHEMA[group], f"{group} declares no keys"


def test_env_override_takes_effect():
    cfg = load_config("demo", use_env=False, environ={"ENV_VOICEFORGE_AUDIO_N_MFCC": "13"})
    assert cfg.audio.n_mfcc == 13
    assert "ENV_VOICEFORGE_AUDIO_N_MFCC" in cfg.env_applied


def test_env_override_parses_sequences():
    cfg = load_config("demo", use_env=False, environ={"ENV_VOICEFORGE_BENCH_SEEDS": "17,29,41"})
    assert cfg.bench.seeds == (17, 29, 41)


def test_unknown_env_key_raises():
    """A silent typo would leave the default in place and drift the benchmark."""
    with pytest.raises(UnknownEnvKeyError):
        load_config("demo", use_env=False, environ={"ENV_VOICEFORGE_GMM_N_GSAUSS": "32"})


def test_unknown_env_group_raises():
    with pytest.raises(UnknownEnvKeyError):
        load_config("demo", use_env=False, environ={"ENV_VOICEFORGE_NOPE_KEY": "1"})


@pytest.mark.parametrize(
    "overrides",
    [
        {"audio": {"fmax": 9000.0}},          # above Nyquist
        {"audio": {"n_mfcc": 500}},           # out of range
        {"runtime": {"threads": 4}},          # breaks determinism
        {"proj": {"lda_dim": 999}},           # exceeds the feature dimension
        {"ivector": {"nbc_dims": 9999}},      # would annihilate the space
        {"plda": {"dim": 9999}},              # exceeds tv_dim
        {"bench": {"n_genuine_ratio": 2.0}},  # impossible ratio
        {"gmm": {"diag_only": False}},        # unsupported
        {"audio": {"nope": 1}},               # unknown key
    ],
)
def test_invalid_configuration_is_rejected(overrides):
    with pytest.raises(SchemaValidationError):
        load_config("demo", overrides, use_env=False)


def test_unknown_profile_is_rejected():
    with pytest.raises(SchemaValidationError):
        load_config("nonexistent", use_env=False)


def test_config_evolve_and_set_round_trip():
    cfg = load_config("demo", use_env=False)
    assert cfg.evolve(gmm={"n_gauss": 32}).gmm.n_gauss == 32
    assert cfg.set("gmm.n_gauss", 48).gmm.n_gauss == 48
    with pytest.raises(SchemaValidationError):
        cfg.set("gmm.nope", 1)
    with pytest.raises(SchemaValidationError):
        cfg.evolve(nosuchgroup={"x": 1})


# --------------------------------------------------------------------------
# backend
# --------------------------------------------------------------------------
def test_available_librosa_does_not_import_it():
    """find_spec must not execute librosa (it drags in numba)."""
    import sys

    sys.modules.pop("librosa", None)
    ok, detail = available_librosa()
    assert isinstance(ok, bool)
    assert isinstance(detail, str)
    assert "librosa" not in sys.modules, "probing librosa must not import it"


def test_tier1_never_imports_librosa():
    """The tier1 path must be usable on a box where librosa cannot load."""
    import subprocess
    import sys

    code = (
        "import sys;"
        "from voiceforge.sv.frontend import mel_filterbank;"
        "fb = mel_filterbank(16000, 512, 40, 20.0, 7600.0, backend='tier1');"
        "print(fb.backend_tag, 'librosa' in sys.modules)"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": root}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "numpy False", out.stdout


def test_resolve_dsp_tiers():
    assert resolve_dsp("tier1").resolved == "tier1"
    assert resolve_dsp("auto").resolved in ("tier0", "tier1")
    assert resolve_dsp("tier1", force_tier1=True).resolved == "tier1"
    with pytest.raises(DspBackendError):
        resolve_dsp("tier9")


def test_backend_fingerprint_is_stable_and_described():
    info = resolve_dsp("tier1")
    assert backend_fingerprint(info) == backend_fingerprint(info)
    env = describe_backend(info)
    for key in ("python", "numpy", "scipy", "sklearn", "librosa", "blas", "threads", "backend_tier", "mel_fingerprint"):
        assert key in env, f"env block missing {key}"
    assert env["backend_tier"] == "tier1"
    assert "OMP_NUM_THREADS" in env["threads"]


def test_mel_fingerprint_is_content_addressed():
    a = mel_fingerprint(np.eye(3))
    b = mel_fingerprint(np.eye(3))
    c = mel_fingerprint(np.eye(3) * 2.0)
    assert a == b and a != c and len(a) == 12


# --------------------------------------------------------------------------
# determinism primitives
# --------------------------------------------------------------------------
def test_preset_threads_pins_every_pool():
    applied = preset_threads(1)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_CBWR"):
        assert key in applied
        assert os.environ[key] in ("1", "COMPATIBLE")


def test_preset_threads_rejects_zero():
    with pytest.raises(ValueError):
        preset_threads(0)


def test_named_substreams_are_independent_and_reproducible():
    a1 = rng(7, "alpha").normal(size=5)
    a2 = rng(7, "alpha").normal(size=5)
    b = rng(7, "beta").normal(size=5)
    assert np.array_equal(a1, a2), "same (seed, stream) must be reproducible"
    assert not np.array_equal(a1, b), "different streams must differ"
    assert derive_seed(7, "alpha") != derive_seed(7, "beta")
    assert derive_seed(7, "alpha") == derive_seed(7, "alpha")


def test_substreams_do_not_depend_on_consumption_order():
    """Drawing from one stream must not perturb another (G5)."""
    first = rng(3, "x").normal(size=4)
    _ = rng(3, "y").normal(size=1000)
    assert np.array_equal(first, rng(3, "x").normal(size=4))


def test_set_all_makes_global_rngs_reproducible():
    set_all(42)
    a = np.random.rand(3)
    set_all(42)
    assert np.array_equal(a, np.random.rand(3))


def test_content_hash_is_order_independent_for_dicts():
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
    assert content_hash([1, 2, 3]) != content_hash([3, 2, 1])


def test_content_hash_handles_numpy_scalars_and_arrays():
    assert isinstance(content_hash({"x": np.float64(1.5), "arr": np.arange(3)}), str)


# --------------------------------------------------------------------------
# types and memory
# --------------------------------------------------------------------------
def test_frozen_array_is_read_only():
    fa = FrozenArray(np.arange(5.0))
    ro = fa.ro()
    assert ro.flags.writeable is False
    with pytest.raises(ValueError):
        ro[0] = 1.0


def test_check_float64_rejects_float32():
    check_float64(np.zeros(3, dtype=np.float64))
    with pytest.raises(DtypeError):
        check_float64(np.zeros(3, dtype=np.float32))


def test_memory_probe_reports_a_positive_value():
    """Gate G6 needs a real number, not a silent 0.0."""
    assert rss_backend() in ("proc_status", "windows", "resource", "unavailable")
    if rss_backend() != "unavailable":
        assert peak_rss_mb() > 0.0
        assert current_rss_mb() > 0.0


def test_memory_probe_tracks_an_allocation():
    """A ~256 MiB allocation must move the reported RSS."""
    if rss_backend() == "unavailable":
        pytest.skip("no memory probe available on this platform")
    before = peak_rss_mb()
    ballast = [np.zeros((4096, 4096), dtype=np.float64) for _ in range(8)]  # ~1 GiB
    after = peak_rss_mb()
    del ballast
    assert after > before
