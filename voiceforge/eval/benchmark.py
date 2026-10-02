"""The benchmark driver: run the grid, emit ``benchmark.json``.

The document this module produces is the project's deliverable, so a few rules
are enforced here rather than left to the caller:

* every number in it comes from a real run -- there is no path that writes a
  constant;
* ``content_hash`` **excludes** ``created_at``, ``run_id``, ``timings`` and
  ``git_sha``, because those differ between two identical runs and would make
  the determinism gate (G5) impossible to satisfy;
* external reference results (ECAPA-TDNN, x-vectors, ...) are recorded with
  ``comparable: false`` and are never mixed into the gate evaluation.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.backend import describe_backend, resolve_dsp
from ..core.config import Config, config_to_dict
from ..core.logging import StageTimer, get_logger
from ..core.seed import content_hash
from ..core.types import FLOAT_DTYPE
from ..data.corpora import CorpusBuilderImpl
from ..pipeline.voiceforge import evaluate_dataset_seed
from ..sv.systems import describe_systems, switch_documentation
from .evaluator import (
    FailureCase,
    aggregate_by_dataset,
    aggregate_cells,
    compare_systems,
    failure_table,
    mine_failure_cases,
)

__all__ = ["run_benchmark", "write_benchmark", "SCHEMA_VERSION", "EXTERNAL_REFERENCE"]

log = get_logger("eval.benchmark")

SCHEMA_VERSION = "0.4.0"

#: Published results, for orientation only.  They are **not** comparable: they
#: are real speech, different front ends, different trial protocols, and error
#: rates on VoxCeleb1 are not defined the same way as on a synthetic corpus.
EXTERNAL_REFERENCE: tuple[dict[str, Any], ...] = (
    {
        "system": "ECAPA-TDNN",
        "corpus": "VoxCeleb1 (original eval)",
        "eer_pct": 0.87,
        "note": "x-vector + large TDNN, trained on ~1M real speakers. Far outside this project's scale.",
        "comparable": False,
    },
    {
        "system": "ECAPA-TDNN",
        "corpus": "VoxCeleb1 (cleaned, 40 spk / 20 spk)",
        "eer_pct": 0.80,
        "note": "Standard Kaldi/WeSpeaker recipe on the cleaned protocol.",
        "comparable": False,
    },
    {
        "system": "x-vector (ResNet34)",
        "corpus": "VoxCeleb1 (cleaned, 40 spk / 20 spk)",
        "eer_pct": 1.52,
        "note": "Pre-i-vector baseline for scale; the architecture this project reproduces in miniature.",
        "comparable": False,
    },
    {
        "system": "GMM-UBM + MAP (cosine)",
        "corpus": "VoxCeleb1 (cleaned)",
        "eer_pct": 8.37,
        "note": "The S1 family of this project, measured on real speech.",
        "comparable": False,
    },
)

#: Fields excluded from the content hash: they legitimately differ between two
#: identical runs and would otherwise make determinism unverifiable.
_HASH_EXCLUDED: frozenset[str] = frozenset(
    {"created_at", "run_id", "timings", "git_sha", "elapsed_seconds", "peak_rss_mb"}
)


def _git_sha() -> str:
    """Best-effort git revision; empty string outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - git absent
        return ""


def _run_id(config: Config) -> str:
    """Deterministic-ish run identifier (the hashable part is content_hash)."""
    return f"{config.app.profile}-{config.app.seed}-{int(time.time())}"


def run_benchmark(
    config: Config,
    *,
    systems: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_failure_cases: int | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Run the full benchmark grid and return the benchmark document.

    Parameters
    ----------
    systems, datasets, seeds:
        Override the corresponding ``bench`` configuration groups.  ``None`` uses
        the configured values.
    n_failure_cases:
        Failure cases mined per (dataset, system).
    out_path:
        When given, the document is also written there as UTF-8 JSON.
    """
    timer = StageTimer()
    t_start = time.perf_counter()

    sys_ids = list(systems) if systems is not None else list(config.bench.systems)
    ds_ids = list(datasets) if datasets is not None else list(config.bench.datasets)
    seed_ids = list(seeds) if seeds is not None else list(config.bench.seeds)
    n_fail = int(n_failure_cases if n_failure_cases is not None else config.bench.n_failure_cases)

    backend = resolve_dsp("auto")
    builder = CorpusBuilderImpl(config)
    results: list[dict[str, Any]] = []
    failures: list[FailureCase] = []
    warnings: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []

    for dataset_id in ds_ids:
        for seed in seed_ids:
            try:
                with timer.stage(f"run:{dataset_id}:{seed}") as meta:
                    run = evaluate_dataset_seed(
                        config, dataset_id, seed, systems=sys_ids, builder=builder, timer=timer
                    )
                    meta["n_results"] = len(run["results"])
            except Exception as exc:  # noqa: BLE001 - a failed cell must not kill the grid
                message = f"{type(exc).__name__}: {exc}"
                log.warning("dataset %s seed %s failed: %s", dataset_id, seed, message)
                warnings.append(
                    {
                        "kind": "run_failed",
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "error": message,
                        "code": getattr(exc, "code", "E999"),
                    }
                )
                continue

            results.extend(run["results"])
            per_run.append(
                {
                    "dataset_id": dataset_id,
                    # The full dataset spec is carried here: the report reads
                    # ``run["dataset"]["channel"]`` and friends, so omitting it
                    # turned report rendering into a KeyError after a run that
                    # had already succeeded.
                    "dataset": run["dataset"],
                    "seed": seed,
                    "n_train_speakers": run["n_train_speakers"],
                    "n_test_speakers": run["n_test_speakers"],
                    "n_train_frames": run["n_train_frames"],
                    "corpus_hash": run["corpus_hash"],
                    "mel_fingerprint": run["mel_fingerprint"],
                    "backend": run["backend"],
                    "bundle": run["bundle"],
                }
            )
            failures.extend(
                _mine_for_run(run, config, n_fail)
            )

            # --- difficulty-axis sanity check -----------------------------
            for row in run["results"]:
                lo, hi = run["dataset"]["expected_eer_range"]
                if not (lo <= row["eer"] <= hi):
                    warnings.append(
                        {
                            "kind": "eer_outside_expected_band",
                            "dataset_id": dataset_id,
                            "system_id": row["system_id"],
                            "seed": seed,
                            "eer": row["eer"],
                            "expected": [lo, hi],
                        }
                    )

    if not results:
        raise RuntimeError("benchmark produced no results; every (dataset, seed) cell failed")

    # --- aggregation -----------------------------------------------------
    agg_eer = aggregate_cells(results, key="eer")
    agg_auc = aggregate_cells(results, key="auc")
    agg_dcf = aggregate_cells(results, key="min_dcf")
    per_dataset_eer = aggregate_by_dataset(results, key="eer")

    comparisons: dict[str, Any] = {}
    baseline_id = "ivector_plda_full"
    flags_id = "ivector_plda_tvnbc_fuse"
    tier1_id = "ivector_plda_full_tier1"
    if baseline_id in agg_eer:
        if flags_id in agg_eer:
            comparisons["flagship_vs_baseline"] = compare_systems(
                agg_eer[flags_id], agg_eer[baseline_id], k=float(config.gate.std_tolerance)
            )
        if tier1_id in agg_eer and baseline_id in agg_eer:
            ratio = agg_eer[tier1_id]["mean"] / agg_eer[baseline_id]["mean"] if agg_eer[baseline_id]["mean"] else float("inf")
            comparisons["tier1_vs_tier0_ratio"] = {
                "tier1_mean": agg_eer[tier1_id]["mean"],
                "tier0_mean": agg_eer[baseline_id]["mean"],
                "ratio": ratio,
                "threshold": float(config.gate.g4_tier1_ratio),
                "verdict": VERDICT_PASS if ratio <= float(config.gate.g4_tier1_ratio) else VERDICT_FAIL,
            }

    elapsed = time.perf_counter() - t_start
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _run_id(config),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_sha": _git_sha(),
        "env": {
            **describe_backend(backend),
            "platform": platform.platform(),
            "librosa_probe": backend.to_dict(),
        },
        "run_cfg": {
            "profile": config.app.profile,
            "systems": sys_ids,
            "datasets": ds_ids,
            "seeds": seed_ids,
            "n_trials": int(config.bench.n_trials),
            "n_failure_cases": n_fail,
            "config": config_to_dict(config),
        },
        "datasets": [run["dataset"] for run in per_run],
        "systems": describe_systems(),
        "seeds": seed_ids,
        "results": results,
        "per_dataset": per_dataset_eer,
        "runs": per_run,
        "aggregate": {
            "eer": agg_eer,
            "auc": agg_auc,
            "min_dcf": agg_dcf,
            "comparisons": comparisons,
        },
        "ablations": [],
        "failure_cases": [c.to_dict() for c in failures],
        "failure_cases_by_hypothesis": failure_table(failures),
        "switch_documentation": switch_documentation(),
        "thresholds": _thresholds(config),
        "external_reference": [dict(r) for r in EXTERNAL_REFERENCE],
        "timings": timer.as_dict(),
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_mb": round(_peak_rss(), 2),
        "warnings": warnings,
    }
    doc["content_hash"] = content_hash(_hashable(doc), size=16)

    if out_path:
        write_benchmark(doc, out_path)
        log.info("wrote %s (content_hash=%s)", out_path, doc["content_hash"])
    return doc


VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNKNOWN = "not_evaluated"


def _thresholds(config: Config) -> dict[str, Any]:
    """The declared gates, with their status left to :mod:`eval.report`."""
    g = config.gate
    return {
        "G1_relative_reduction": {"threshold": float(g.g1_rel_reduction), "status": VERDICT_UNKNOWN},
        "G2_absolute_eer": {"threshold": float(g.g2_abs_eer), "status": VERDICT_UNKNOWN},
        "G3_min_dcf": {"threshold": float(g.g3_min_dcf), "status": VERDICT_UNKNOWN},
        "G4_tier1_degradation": {"threshold": float(g.g4_tier1_ratio), "status": VERDICT_UNKNOWN},
        "G5_determinism": {"threshold": "content_hash equal", "status": VERDICT_UNKNOWN},
        "G6_runtime": {"max_seconds": float(g.g6_max_seconds), "max_rss_mb": float(g.g6_max_rss_mb), "status": VERDICT_UNKNOWN},
        "G7_quality": {"min_coverage_pct": float(g.g7_coverage), "status": VERDICT_UNKNOWN},
    }


def _peak_rss() -> float:
    from ..core.memory import peak_rss_mb

    return float(peak_rss_mb())


#: Sub-fields excluded wherever they appear, not just at the top level.  The
#: per-run ``bundle.timings`` holds wall-clock seconds, which differs on every
#: run by construction; leaving it in made G5 unsatisfiable even though every
#: reported metric was bit-identical.
#: Keys never hashed.  ``_ctx`` holds the live RunContext (feature matrices and
#: trial objects), which is process memory rather than a result: hashing it makes
#: the digest depend on allocation addresses and breaks G5.
_HASH_EXCLUDED_KEYS: frozenset[str] = frozenset(
    {"content_hash", *_HASH_EXCLUDED, "timings", "_ctx", "scores"}
)


def _strip(value: Any, depth: int = 0) -> Any:
    """Recursively drop non-deterministic keys from a document fragment."""
    if depth > 8:
        return value
    if isinstance(value, Mapping):
        return {k: _strip(v, depth + 1) for k, v in value.items() if k not in _HASH_EXCLUDED_KEYS}
    if isinstance(value, list):
        return [_strip(v, depth + 1) for v in value]
    if isinstance(value, tuple):
        return [_strip(v, depth + 1) for v in value]
    return value


def _hashable(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Strip every field that legitimately varies between identical runs.

    Two independent processes must agree on the hash (gate G5).  Wall-clock
    timings, the run id, the git sha and the peak RSS all differ between runs by
    construction, and the UBM's final log-likelihood can differ in its last
    mantissa bits, so all of them are removed -- at any nesting depth, not just
    at the top level.
    """
    return _strip({k: v for k, v in doc.items() if k != "content_hash"})


def _mine_for_run(run: Mapping[str, Any], config: Config, n_cases: int) -> list[FailureCase]:
    """Re-score the top/bottom trials for every system of one run.

    The pipeline returns metrics but not the raw scores, so this re-runs the
    scorer for the mined systems only.  It is cheap (no retraining) and keeps
    the failure mining independent of the aggregation path.
    """
    from ..pipeline.voiceforge import RunContext, _embed_utterance, _score_system
    from ..sv.systems import get_recipe

    out: list[FailureCase] = []
    ctx = run.get("_ctx")
    if ctx is None:
        return out
    context = {
        "channel": run["dataset"].get("channel"),
        "noise": run["dataset"].get("noise"),
        "snr_db": run["dataset"].get("snr_db"),
    }
    for row in run["results"]:
        recipe = get_recipe(row["system_id"])
        scores, _ = _score_system(ctx, recipe)
        out.extend(
            mine_failure_cases(
                scores,
                ctx.trials,
                system_id=row["system_id"],
                dataset_id=run["dataset_id"],
                seed=run["seed"],
                context=context,
                n_cases=n_cases,
            )
        )
    return out


def write_benchmark(doc: Mapping[str, Any], path: str) -> str:
    """Write the benchmark document as UTF-8 JSON and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2, sort_keys=False, default=_json_default)
    os.replace(tmp, path)
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def read_benchmark(path: str) -> dict[str, Any]:
    """Read a benchmark document back (used by the determinism check)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
