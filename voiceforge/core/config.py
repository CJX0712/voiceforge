"""Declarative configuration schema, ENV overrides, and validation.

Design goals
------------
* **Fail fast.** Every key is declared in :data:`DEFAULT_SCHEMA` with a type, a
  range, and optional exclusivity constraints.  An unknown ``ENV_VOICEFORGE_*``
  variable raises :class:`UnknownEnvKeyError` instead of being ignored, because
  a silently-dropped override is the single most common cause of "it works on my
  machine" benchmark drift.
* **No third-party imports.** Validation is hand-rolled so that ``core`` stays
  dependency-free and importable before numpy exists.
* **Immutability.** The resolved config is a frozen dataclass tree; nothing
  downstream can mutate global state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterator, Mapping, Sequence

from .errors import SchemaValidationError, UnknownEnvKeyError

__all__ = [
    "Spec",
    "DEFAULT_SCHEMA",
    "ENV_PREFIX",
    "Config",
    "AppConfig",
    "RuntimeConfig",
    "AudioConfig",
    "VtlnConfig",
    "ProjConfig",
    "GmmConfig",
    "MapConfig",
    "IvectorConfig",
    "PldaConfig",
    "ScoreConfig",
    "DataConfig",
    "BenchConfig",
    "GateConfig",
    "load_config",
    "config_to_dict",
    "apply_env_overrides",
    "SCHEMA_GROUPS",
]

ENV_PREFIX = "ENV_VOICEFORGE_"

_BOOL_TRUE = frozenset({"1", "true", "yes", "on", "t", "y"})
_BOOL_FALSE = frozenset({"0", "false", "no", "off", "f", "n"})


@dataclass(frozen=True)
class Spec:
    """Declaration of a single configuration key.

    Attributes
    ----------
    default:
        Value used when neither the profile nor the environment supplies one.
    type:
        One of ``int``, ``float``, ``bool``, ``str``.
    min / max:
        Inclusive bounds for numeric types.  ``None`` disables the check.
    choices:
        Allowed values for ``str`` (and optionally for ``int``).
    exclusive:
        Keys in the *same group* that may not both be truthy at once.
    doc:
        One-line description surfaced by ``voiceforge config --list``.
    """

    default: Any
    type: str
    min: float | None = None
    max: float | None = None
    choices: tuple[Any, ...] | None = None
    exclusive: tuple[str, ...] = ()
    doc: str = ""


def _s(default: Any, type_: str, doc: str, **kw: Any) -> tuple[str, Spec]:
    return default, Spec(default=default, type=type_, doc=doc, **kw)


#: The complete set of tunables, grouped into 12 sections.
DEFAULT_SCHEMA: dict[str, dict[str, Any]] = {
    "app": {
        "name": _s("voiceforge", "str", "Application name."),
        "profile": _s(
            "demo", "str", "Run profile: smoke | demo | bench.", choices=("smoke", "demo", "bench")
        ),
        "seed": _s(17, "int", "Master seed; all substreams derive from it.", min=0, max=2**31 - 1),
        "verbose": _s(False, "bool", "Verbose stage logging."),
        "out_dir": _s("artifacts", "str", "Root directory for all emitted artifacts."),
    },
    "runtime": {
        "threads": _s(1, "int", "BLAS/OpenMP threads; must be 1 for determinism.", min=1, max=64),
        "float_dtype": _s("float64", "str", "Hard dtype lock for all model arrays.", choices=("float64",)),
        "hash_size": _s(16, "int", "Hex length of content hashes.", min=8, max=64),
    },
    "audio": {
        "sample_rate": _s(16000, "int", "Waveform sample rate (Hz).", min=8000, max=48000),
        "frame_ms": _s(25.0, "float", "Frame length (ms).", min=10.0, max=50.0),
        "hop_ms": _s(10.0, "float", "Frame shift (ms).", min=5.0, max=25.0),
        "n_fft": _s(512, "int", "FFT size.", min=128, max=2048),
        "n_mels": _s(40, "int", "Mel filterbank channels.", min=13, max=128),
        "fmin": _s(20.0, "float", "Lowest mel filter edge (Hz).", min=0.0, max=1000.0),
        "fmax": _s(7600.0, "float", "Highest mel filter edge (Hz).", min=1000.0, max=8000.0),
        "n_mfcc": _s(20, "int", "MFCC coefficients kept (c0 dropped).", min=8, max=64),
        "keep_c0": _s(False, "bool", "Keep the c0 coefficient instead of dropping it."),
        "lifter": _s(22.0, "float", "Cephalometric lifter L (0 disables).", min=0.0, max=100.0),
        "preemph": _s(0.97, "float", "Pre-emphasis coefficient.", min=0.0, max=0.99),
        "n_deltas": _s(2, "int", "Append delta orders 1..n (0 disables).", min=0, max=3),
        "delta_width": _s(2, "int", "Regression half-window for deltas.", min=1, max=8),
        "cmvn": _s(True, "bool", "Per-utterance cepstral mean/variance normalisation."),
    },
    "vtln": {
        "enabled": _s(True, "bool", "Enable VTLN warp-factor search."),
        "n_grid": _s(29, "int", "Coarse warp grid points.", min=3, max=101),
        "w_min": _s(0.85, "float", "Minimum warp factor.", min=0.5, max=0.99),
        "w_max": _s(1.15, "float", "Maximum warp factor.", min=1.01, max=1.5),
        "n_iter_refine": _s(12, "int", "Golden-section refinement iterations.", min=0, max=60),
        "long_win_ms": _s(35.0, "float", "Long analysis window for VTLN frames.", min=25.0, max=60.0),
        "floor": _s(1e-8, "float", "Variance floor inside the UBM likelihood.", min=1e-12, max=1e-3),
    },
    "proj": {
        "lda_dim": _s(
            40, "int", "LDA output dimensionality (0 disables); must be < n_mfcc*(1+n_deltas).", min=0, max=1024
        ),
        "shrinkage": _s(0.15, "float", "Sw shrinkage towards a scaled identity.", min=0.0, max=1.0),
        "fit_max_frames": _s(200000, "int", "Frame cap when fitting LDA.", min=1000, max=5000000),
    },
    "gmm": {
        "n_gauss": _s(512, "int", "UBM mixture size.", min=2, max=4096),
        "n_iter": _s(12, "int", "EM iterations.", min=1, max=200),
        "tol": _s(1e-5, "float", "Relative log-likelihood convergence tolerance.", min=1e-12, max=1e-1),
        "min_var": _s(1e-6, "float", "Variance floor preventing covariance collapse.", min=1e-12, max=1e-2),
        "kmeans_iter": _s(10, "int", "KMeans++ initialisation iterations.", min=1, max=100),
        "train_max_frames": _s(300000, "int", "Frame subsample cap for UBM training.", min=1000, max=5000000),
        "diag_only": _s(True, "bool", "Diagonal covariance (full covariance is unsupported)."),
    },
    "map": {
        "tau": _s(0.5, "float", "MAP relevance factor for means.", min=0.0, max=100.0),
        "variant": _s("map2", "str", "Covariance update variant.", choices=("map1", "map2")),
        "adapt_var": _s(True, "bool", "Adapt diagonal variances (else speaker-independent)."),
    },
    "ivector": {
        "enabled": _s(True, "bool", "Enable i-vector extraction."),
        "tv_dim": _s(400, "int", "Total-variability projection dimensionality.", min=8, max=4000),
        "n_iter": _s(2, "int", "TV re-estimation iterations.", min=1, max=10),
        "nbc": _s(True, "bool", "Nuisance back-channel (session) compensation."),
        "nbc_dims": _s(30, "int", "Nuisance directions removed by NBC.", min=0, max=400),
        "cms": _s(True, "bool", "Cepstral mean subtraction on the supervector."),
        "center_supervector": _s(
            True,
            "bool",
            "Centre the MAP supervector on the UBM mean before use (Kaldi ivector-extract). "
            "Without it every adapted mean keeps a dominant common component.",
        ),
        "length_norm": _s(True, "bool", "L2 length normalisation of the i-vector."),
        "tv_reg": _s(1e-6, "float", "Ridge added to L^T L before inversion.", min=1e-12, max=1e-1),
        "supervector_order": _s(
            "interleaved", "str", "Interleave means/vars or concatenate blocks.", choices=("interleaved", "blocked")
        ),
    },
    "plda": {
        "enabled": _s(True, "bool", "Enable PLDA scoring backend."),
        "dim": _s(200, "int", "i-vector trimming dimension before PLDA.", min=8, max=2000),
        "n_iter": _s(40, "int", "EM iterations.", min=1, max=500),
        "tol": _s(1e-6, "float", "EM convergence tolerance.", min=1e-12, max=1e-1),
        "var_floor": _s(1e-6, "float", "Covariance floor.", min=1e-12, max=1e-2),
        "length_norm": _s(True, "bool", "Short-length variability normalisation (SLV)."),
    },
    "score": {
        "fuse_weight": _s(0.7, "float", "Fusion weight on z-normalised PLDA LLR.", min=0.0, max=1.0),
        "calibrate": _s("zscore", "str", "Score calibration for fusion.", choices=("zscore", "none")),
    },
    "data": {
        "n_speakers": _s(40, "int", "Distinct speaker identities per corpus.", min=2, max=2000),
        "n_utts_per_speaker": _s(6, "int", "Utterances generated per speaker.", min=2, max=100),
        "cache_dir": _s("artifacts/corpus", "str", "Corpus cache root."),
        "use_cache": _s(True, "bool", "Reuse cached corpora when the manifest hash matches."),
        "difficulty_seed_stride": _s(1000, "int", "Seed offset between difficulty axes."),
    },
    "bench": {
        "seeds": _s((17, 29, 41), "int", "Benchmark seeds.", min=0, max=2**31 - 1),
        "datasets": _s(
            ("D1_clean", "D2_white5", "D3_tel8", "D4_rev0", "D5_short1", "D6_mixneg"),
            "str",
            "Dataset ids in the difficulty grid.",
        ),
        "systems": _s(
            (
                "mfcc_cos_nbc",
                "gmmubm_map_cos",
                "ivector_plda_full",
                "ivector_plda_tvnbc_fuse",
                "ivector_plda_full_tier1",
            ),
            "str",
            "System ids to evaluate.",
        ),
        "n_trials": _s(3000, "int", "Trials per (dataset, seed, system).", min=100, max=200000),
        "n_genuine_ratio": _s(0.5, "float", "Fraction of genuine trials.", min=0.05, max=0.95),
        "minDCF_p_target": _s(0.01, "float", "SALT 2015 normalised minDCF prior.", min=1e-4, max=0.5),
        "ablation_datasets": _s(("D3_tel8", "D4_rev0", "D5_short1", "D6_mixneg"), "str", "Datasets used for ablation."),
        "ablation_seeds": _s((17,), "int", "Seeds used for ablation (cost control)."),
        "n_failure_cases": _s(3, "int", "Failure cases mined per (dataset, system).", min=1, max=20),
    },
    "gate": {
        "g1_rel_reduction": _s(0.20, "float", "Flagship must cut flagship-vs-baseline EER by this fraction.", min=0.0, max=1.0),
        "g2_abs_eer": _s(0.05, "float", "Flagship absolute aggregate EER ceiling.", min=0.0, max=1.0),
        "g3_min_dcf": _s(0.35, "float", "Flagship aggregate minDCF ceiling.", min=0.0, max=10.0),
        "g4_tier1_ratio": _s(1.20, "float", "Tier-1 degradation ceiling vs Tier-0.", min=1.0, max=10.0),
        "g6_max_seconds": _s(60.0, "float", "Demo wall-clock budget.", min=1.0, max=100000.0),
        "g6_max_rss_mb": _s(2048.0, "float", "Demo peak-RSS budget (MiB).", min=64.0, max=1048576.0),
        "g7_coverage": _s(80.0, "float", "Minimum line coverage for core modules.", min=0.0, max=100.0),
        "std_tolerance": _s(0.5, "float", "Win requires mean gap > k * (std1 + std2).", min=0.0, max=10.0),
    },
}

#: Canonical group order (also the iteration order of :func:`config_to_dict`).
SCHEMA_GROUPS: tuple[str, ...] = (
    "app",
    "runtime",
    "audio",
    "vtln",
    "proj",
    "gmm",
    "map",
    "ivector",
    "plda",
    "score",
    "data",
    "bench",
    "gate",
)


# --------------------------------------------------------------------------
# typed config tree
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AppConfig:
    name: str
    profile: str
    seed: int
    verbose: bool
    out_dir: str


@dataclass(frozen=True)
class RuntimeConfig:
    threads: int
    float_dtype: str
    hash_size: int


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    frame_ms: float
    hop_ms: float
    n_fft: int
    n_mels: int
    fmin: float
    fmax: float
    n_mfcc: int
    keep_c0: bool
    lifter: float
    preemph: float
    n_deltas: int
    delta_width: int
    cmvn: bool


@dataclass(frozen=True)
class VtlnConfig:
    enabled: bool
    n_grid: int
    w_min: float
    w_max: float
    n_iter_refine: int
    long_win_ms: float
    floor: float


@dataclass(frozen=True)
class ProjConfig:
    lda_dim: int
    shrinkage: float
    fit_max_frames: int


@dataclass(frozen=True)
class GmmConfig:
    n_gauss: int
    n_iter: int
    tol: float
    min_var: float
    kmeans_iter: int
    train_max_frames: int
    diag_only: bool


@dataclass(frozen=True)
class MapConfig:
    tau: float
    variant: str
    adapt_var: bool


@dataclass(frozen=True)
class IvectorConfig:
    enabled: bool
    tv_dim: int
    n_iter: int
    nbc: bool
    nbc_dims: int
    cms: bool
    center_supervector: bool
    length_norm: bool
    tv_reg: float
    supervector_order: str


@dataclass(frozen=True)
class PldaConfig:
    enabled: bool
    dim: int
    n_iter: int
    tol: float
    var_floor: float
    length_norm: bool


@dataclass(frozen=True)
class ScoreConfig:
    fuse_weight: float
    calibrate: str


@dataclass(frozen=True)
class DataConfig:
    n_speakers: int
    n_utts_per_speaker: int
    cache_dir: str
    use_cache: bool
    difficulty_seed_stride: int


@dataclass(frozen=True)
class BenchConfig:
    seeds: tuple[int, ...]
    datasets: tuple[str, ...]
    systems: tuple[str, ...]
    n_trials: int
    n_genuine_ratio: float
    minDCF_p_target: float
    ablation_datasets: tuple[str, ...]
    ablation_seeds: tuple[int, ...]
    n_failure_cases: int


@dataclass(frozen=True)
class GateConfig:
    g1_rel_reduction: float
    g2_abs_eer: float
    g3_min_dcf: float
    g4_tier1_ratio: float
    g6_max_seconds: float
    g6_max_rss_mb: float
    g7_coverage: float
    std_tolerance: float


@dataclass(frozen=True)
class Config:
    """Root configuration object (frozen; use :meth:`evolve` to derive)."""

    app: AppConfig
    runtime: RuntimeConfig
    audio: AudioConfig
    vtln: VtlnConfig
    proj: ProjConfig
    gmm: GmmConfig
    map: MapConfig
    ivector: IvectorConfig
    plda: PldaConfig
    score: ScoreConfig
    data: DataConfig
    bench: BenchConfig
    gate: GateConfig
    env_applied: tuple[str, ...] = field(default=(), compare=True)

    # -- convenience accessors -------------------------------------------
    @property
    def sample_rate(self) -> int:
        return self.audio.sample_rate

    @property
    def seed(self) -> int:
        return self.app.seed

    @property
    def n_features(self) -> int:
        """Dimensionality of one feature frame after delta expansion."""
        base = self.audio.n_mfcc
        return base * (1 + self.audio.n_deltas)

    def evolve(self, **groups: Mapping[str, Any]) -> "Config":
        """Return a copy with the given group(s) replaced field-by-field."""
        updates: dict[str, Any] = {}
        for name, values in groups.items():
            if name not in SCHEMA_GROUPS:
                raise SchemaValidationError(f"unknown config group {name!r}", group=name, known=list(SCHEMA_GROUPS))
            current = getattr(self, name)
            valid = {f.name for f in fields(current)}
            bad = set(values) - valid
            if bad:
                raise SchemaValidationError(
                    f"unknown key(s) in group {name!r}", unknown=sorted(bad), valid=sorted(valid)
                )
            updates[name] = replace(current, **values)
        return replace(self, **updates)

    def get(self, dotted: str) -> Any:
        """Fetch ``"audio.n_mfcc"`` style keys."""
        if "." not in dotted:
            raise SchemaValidationError(f"key must be dotted, got {dotted!r}", key=dotted)
        group, _, key = dotted.partition(".")
        if group not in SCHEMA_GROUPS:
            raise SchemaValidationError(f"unknown group in {dotted!r}", key=dotted)
        if key not in DEFAULT_SCHEMA[group]:
            raise SchemaValidationError(f"unknown key {dotted!r}", key=dotted)
        return getattr(getattr(self, group), key)

    def set(self, dotted: str, value: Any) -> "Config":
        """Return a copy with ``dotted`` set to ``value`` (validated)."""
        group, _, key = dotted.partition(".")
        if group not in SCHEMA_GROUPS:
            raise SchemaValidationError(f"unknown group in {dotted!r}", key=dotted)
        if key not in DEFAULT_SCHEMA[group]:
            raise SchemaValidationError(f"unknown key {dotted!r}", key=dotted)
        return self.evolve(**{group: {key: _coerce(value, DEFAULT_SCHEMA[group][key][1], dotted)}})


# --------------------------------------------------------------------------
# coercion + validation
# --------------------------------------------------------------------------
def _coerce(spec_value: Any, spec: Spec, dotted: str) -> Any:
    """Coerce ``value`` to the declared type of ``spec`` or raise E102."""
    target = spec.type
    if target == "bool":
        if isinstance(spec_value, bool):
            return spec_value
        if isinstance(spec_value, (int, float)):
            return bool(spec_value)
        text = str(spec_value).strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE:
            return False
        raise SchemaValidationError(f"{dotted}: cannot parse bool from {spec_value!r}", key=dotted, value=spec_value)
    if target == "int":
        if isinstance(spec_value, bool):
            raise SchemaValidationError(f"{dotted}: expected int, got bool", key=dotted, value=spec_value)
        try:
            if isinstance(spec_value, float):
                if not float(spec_value).is_integer():
                    raise ValueError("not integral")
                out: Any = int(spec_value)
            else:
                out = int(str(spec_value).strip())
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"{dotted}: cannot parse int from {spec_value!r}", key=dotted, value=spec_value
            ) from exc
    elif target == "float":
        if isinstance(spec_value, bool):
            raise SchemaValidationError(f"{dotted}: expected float, got bool", key=dotted, value=spec_value)
        try:
            out = float(spec_value)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"{dotted}: cannot parse float from {spec_value!r}", key=dotted, value=spec_value
            ) from exc
        if out != out or out in (float("inf"), float("-inf")):
            raise SchemaValidationError(f"{dotted}: non-finite float", key=dotted, value=spec_value)
    elif target == "str":
        if not isinstance(spec_value, str):
            raise SchemaValidationError(
                f"{dotted}: expected str, got {type(spec_value).__name__}", key=dotted, value=spec_value
            )
        out = spec_value
    else:  # pragma: no cover - schema authoring error
        raise SchemaValidationError(f"{dotted}: bad spec type {target!r}", key=dotted)

    if spec.choices is not None and out not in spec.choices:
        raise SchemaValidationError(
            f"{dotted}: {out!r} not in {list(spec.choices)}", key=dotted, value=out, choices=list(spec.choices)
        )
    if target in ("int", "float"):
        if spec.min is not None and out < spec.min:
            raise SchemaValidationError(f"{dotted}: {out} < min {spec.min}", key=dotted, value=out, min=spec.min)
        if spec.max is not None and out > spec.max:
            raise SchemaValidationError(f"{dotted}: {out} > max {spec.max}", key=dotted, value=out, max=spec.max)
    return out


def _coerce_sequence(spec: Spec, raw: str | Sequence[Any], dotted: str) -> tuple[Any, ...]:
    """Parse a sequence-valued config entry and validate every element.

    ``raw`` may be a string such as ``"17,29,41"`` (ENV form) or an already
    split sequence (profile preset / override mapping).  Each element is
    coerced with the *scalar* spec, so range and choice checks apply per item.
    """
    if isinstance(raw, (list, tuple)):
        parts = [str(p).strip() for p in raw]
    else:
        parts = [p.strip() for p in str(raw).replace(";", ",").replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise SchemaValidationError(f"{dotted}: empty sequence", key=dotted, value=raw)
    out = []
    for p in parts:
        sub = Spec(default=spec.default, type=spec.type, min=spec.min, max=spec.max, choices=spec.choices)
        out.append(_coerce(p, sub, dotted))
    return tuple(out)


def _iter_env() -> Iterator[tuple[str, str]]:
    for key in sorted(os.environ):
        if key.startswith(ENV_PREFIX):
            yield key, os.environ[key]


def _env_key_to_dotted(key: str) -> tuple[str, str]:
    """Map ``ENV_VOICEFORGE_AUDIO_N_MFCC`` -> ``("audio", "n_mfcc")``.

    The mapping is purely mechanical: strip the prefix, lower-case, split on
    ``_``.  We deliberately do *not* try to be clever about re-joining segments
    (``n_mfcc`` vs ``nmfcc``); instead the result is validated against the
    schema, which is what turns a typo into a loud error.
    """
    body = key[len(ENV_PREFIX) :].lower()
    if not body:
        raise UnknownEnvKeyError(f"empty env override name: {key}", env_key=key)
    group, _, rest = body.partition("_")
    if not rest:
        raise UnknownEnvKeyError(
            f"env override {key} has no sub-key", env_key=key, expected="ENV_VOICEFORGE_<GROUP>_<KEY>"
        )
    return group, rest


def apply_env_overrides(values: dict[str, dict[str, Any]], environ: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Apply ``ENV_VOICEFORGE_*`` overrides in place; return applied key names.

    Raises
    ------
    UnknownEnvKeyError
        If any override does not resolve to a declared schema key.
    """
    source = os.environ if environ is None else environ
    applied: list[str] = []
    for key in sorted(source):
        if not key.startswith(ENV_PREFIX):
            continue
        group, sub = _env_key_to_dotted(key)
        if group not in DEFAULT_SCHEMA:
            raise UnknownEnvKeyError(
                f"unknown config group in {key}",
                env_key=key,
                group=group,
                known_groups=list(SCHEMA_GROUPS),
            )
        if sub not in DEFAULT_SCHEMA[group]:
            raise UnknownEnvKeyError(
                f"unknown config key {group}.{sub} (from {key})",
                env_key=key,
                dotted=f"{group}.{sub}",
                known_keys=sorted(DEFAULT_SCHEMA[group]),
            )
        _, spec = DEFAULT_SCHEMA[group][sub]
        raw = source[key]
        dotted = f"{group}.{sub}"
        if isinstance(spec.default, tuple):
            values[group][sub] = _coerce_sequence(spec, raw, dotted)
        else:
            values[group][sub] = _coerce(raw, spec, dotted)
        applied.append(key)
    return tuple(applied)


def _validate_exclusive(config: Config) -> None:
    """Enforce the declared mutual-exclusion constraints plus cross-field sanity."""
    for group in SCHEMA_GROUPS:
        for key, (_, spec) in DEFAULT_SCHEMA[group].items():
            if not spec.exclusive:
                continue
            if not getattr(getattr(config, group), key):
                continue
            for other in spec.exclusive:
                if getattr(getattr(config, group), other):
                    raise SchemaValidationError(
                        f"{group}.{key} and {group}.{other} are mutually exclusive",
                        group=group,
                        keys=[key, other],
                    )

    # --- cross-field invariants -----------------------------------------
    audio = config.audio
    if audio.n_deltas > 0 and audio.delta_width < 1:
        raise SchemaValidationError("audio.delta_width must be >= 1 when deltas are on", key="audio.delta_width")
    if audio.fmax >= audio.sample_rate / 2:
        raise SchemaValidationError(
            f"audio.fmax ({audio.fmax}) must be < Nyquist ({audio.sample_rate / 2})",
            key="audio.fmax",
            nyquist=audio.sample_rate / 2,
        )
    if audio.fmin >= audio.fmax:
        raise SchemaValidationError("audio.fmin must be < audio.fmax", key="audio.fmin", fmax=audio.fmax)
    if audio.frame_ms <= audio.hop_ms:
        raise SchemaValidationError(
            "audio.frame_ms must exceed audio.hop_ms (no analysis gap)", key="audio.frame_ms", hop_ms=audio.hop_ms
        )
    if not config.gmm.diag_only:
        raise SchemaValidationError(
            "gmm.diag_only=False is unsupported: full-covariance GMM-UBM is not implemented",
            key="gmm.diag_only",
        )
    if config.ivector.nbc and config.ivector.nbc_dims >= config.ivector.tv_dim:
        raise SchemaValidationError(
            "ivector.nbc_dims must be < ivector.tv_dim (NBC would annihilate the space)",
            key="ivector.nbc_dims",
            tv_dim=config.ivector.tv_dim,
        )
    if config.plda.dim > config.ivector.tv_dim:
        raise SchemaValidationError(
            f"plda.dim ({config.plda.dim}) must be <= ivector.tv_dim ({config.ivector.tv_dim})",
            key="plda.dim",
            tv_dim=config.ivector.tv_dim,
        )
    if config.proj.lda_dim and config.proj.lda_dim >= audio.n_mfcc * (1 + audio.n_deltas):
        raise SchemaValidationError(
            f"proj.lda_dim ({config.proj.lda_dim}) must be < feature dim "
            f"({audio.n_mfcc * (1 + audio.n_deltas)})",
            key="proj.lda_dim",
        )
    if not 0.0 < config.bench.n_genuine_ratio < 1.0:
        raise SchemaValidationError(
            "bench.n_genuine_ratio must be in (0, 1)", key="bench.n_genuine_ratio", value=config.bench.n_genuine_ratio
        )
    if config.bench.n_trials < 100:
        raise SchemaValidationError("bench.n_trials must be >= 100 for a stable EER", key="bench.n_trials")
    if config.runtime.threads != 1:
        raise SchemaValidationError(
            "runtime.threads must be 1: multi-threaded BLAS breaks bit-exact determinism (gate G5)",
            key="runtime.threads",
            value=config.runtime.threads,
        )


#: Profile presets applied on top of :data:`DEFAULT_SCHEMA`.
PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "app": {"profile": "smoke", "out_dir": "artifacts/smoke"},
        "data": {"n_speakers": 6, "n_utts_per_speaker": 3, "cache_dir": "artifacts/corpus_smoke"},
        "gmm": {"n_gauss": 16, "n_iter": 4},
        "ivector": {"tv_dim": 24, "nbc_dims": 4},
        "plda": {"dim": 16, "n_iter": 8},
        "proj": {"lda_dim": 12},
        "vtln": {"n_grid": 5, "n_iter_refine": 2},
        "bench": {"n_trials": 200, "seeds": (17,), "datasets": ("D1_clean",), "systems": ("ivector_plda_full",)},
    },
    "demo": {
        "app": {"profile": "demo", "out_dir": "artifacts/demo"},
        "data": {"n_speakers": 20, "n_utts_per_speaker": 4, "cache_dir": "artifacts/corpus"},
        "gmm": {"n_gauss": 64, "n_iter": 6},
        "ivector": {"tv_dim": 60, "nbc_dims": 8},
        "plda": {"dim": 40, "n_iter": 15},
        "proj": {"lda_dim": 30},
        "vtln": {"n_grid": 7, "n_iter_refine": 3},
        "bench": {
            "n_trials": 800,
            "seeds": (17,),
            "datasets": ("D1_clean", "D5_short1"),
            "systems": ("mfcc_cos_nbc", "gmmubm_map_cos", "ivector_plda_full"),
        },
    },
    "bench": {
        "app": {"profile": "bench", "out_dir": "artifacts/bench"},
        "data": {"cache_dir": "artifacts/corpus"},
    },
}


def load_config(
    profile: str | None = None,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    use_env: bool = True,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Build a validated :class:`Config`.

    Precedence (low to high): :data:`DEFAULT_SCHEMA` < profile preset <
    ``overrides`` mapping < ``ENV_VOICEFORGE_*`` environment.

    Raises
    ------
    SchemaValidationError
        For unknown profiles, bad values, or violated exclusivity.
    UnknownEnvKeyError
        For any ``ENV_VOICEFORGE_*`` variable that does not resolve.
    """
    values: dict[str, dict[str, Any]] = {
        group: {k: v[0] for k, v in table.items()} for group, table in DEFAULT_SCHEMA.items()
    }

    prof = profile if profile is not None else values["app"]["profile"]
    if prof not in PROFILES:
        raise SchemaValidationError(f"unknown profile {prof!r}", profile=prof, known=sorted(PROFILES))
    for group, table in PROFILES[prof].items():
        for key, val in table.items():
            values[group][key] = val
    values["app"]["profile"] = prof

    if overrides:
        for group, table in overrides.items():
            if group not in DEFAULT_SCHEMA:
                raise SchemaValidationError(f"unknown override group {group!r}", group=group)
            for key, val in table.items():
                if key not in DEFAULT_SCHEMA[group]:
                    raise SchemaValidationError(
                        f"unknown override key {group}.{key}", dotted=f"{group}.{key}", group=group
                    )
                values[group][key] = _coerce(val, DEFAULT_SCHEMA[group][key][1], f"{group}.{key}")

    # An explicit ``environ`` mapping is honoured even when ``use_env`` is
    # False: callers pass it precisely to test overrides in isolation, from the
    # real process environment.  ``use_env=False`` only suppresses reading
    # os.environ when no explicit mapping was supplied.
    env_applied = apply_env_overrides(values, environ) if (use_env or environ is not None) else ()

    # Coerce everything through the specs so profiles/overrides are type-checked
    # even when they bypassed _coerce above (e.g. profile presets).
    for group, table in DEFAULT_SCHEMA.items():
        for key, (_, spec) in table.items():
                spec_full = Spec(
                    default=spec.default,
                    type=spec.type,
                    min=spec.min,
                    max=spec.max,
                    choices=spec.choices,
                    exclusive=spec.exclusive,
                    doc=spec.doc,
                )
                raw = values[group][key]
                if isinstance(spec.default, (list, tuple)) or isinstance(raw, (list, tuple)):
                    # Sequence-valued key: validate element-wise.  Accepts both a
                    # raw string ("17,29,41", from ENV) and a python sequence
                    # (from a profile preset or an override mapping).
                    seq_raw = raw if isinstance(raw, (list, tuple)) else str(raw)
                    values[group][key] = _coerce_sequence(spec_full, seq_raw, f"{group}.{key}")
                else:
                    values[group][key] = _coerce(raw, spec_full, f"{group}.{key}")

    config = _build(values, env_applied)
    _validate_exclusive(config)
    return config


def _build(values: Mapping[str, Mapping[str, Any]], env_applied: tuple[str, ...]) -> Config:
    """Instantiate the frozen dataclass tree from validated plain values."""
    return Config(
        app=AppConfig(**values["app"]),
        runtime=RuntimeConfig(**values["runtime"]),
        audio=AudioConfig(**values["audio"]),
        vtln=VtlnConfig(**values["vtln"]),
        proj=ProjConfig(**values["proj"]),
        gmm=GmmConfig(**values["gmm"]),
        map=MapConfig(**values["map"]),
        ivector=IvectorConfig(**values["ivector"]),
        plda=PldaConfig(**values["plda"]),
        score=ScoreConfig(**values["score"]),
        data=DataConfig(**values["data"]),
        bench=BenchConfig(**values["bench"]),
        gate=GateConfig(**values["gate"]),
        env_applied=env_applied,
    )


def config_to_dict(config: Config) -> dict[str, Any]:
    """Plain-dict view of the config (used inside ``benchmark.json``)."""
    out: dict[str, Any] = {}
    for group in SCHEMA_GROUPS:
        sub = getattr(config, group)
        out[group] = {f.name: getattr(sub, f.name) for f in fields(sub)}
    out["app"] = dict(out["app"])
    out["env_applied"] = list(config.env_applied)
    return out


def describe_schema() -> list[dict[str, Any]]:
    """Machine-readable schema description for the CLI and the docs."""
    rows: list[dict[str, Any]] = []
    for group in SCHEMA_GROUPS:
        for key, (default, spec) in DEFAULT_SCHEMA[group].items():
            rows.append(
                {
                    "group": group,
                    "key": key,
                    "default": list(default) if isinstance(default, tuple) else default,
                    "type": spec.type,
                    "min": spec.min,
                    "max": spec.max,
                    "choices": list(spec.choices) if spec.choices else None,
                    "doc": spec.doc,
                }
            )
    return rows
