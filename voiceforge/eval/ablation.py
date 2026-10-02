"""Ablation study over the reference system.

Each ablation is the *same code path* as the baseline with one flag flipped
(see :mod:`voiceforge.sv.systems`), so the baseline and the ablated variant
cannot drift apart -- which is the usual failure mode of hand-written ablation
scripts.

Every switch has a declared expected direction in ``SWITCH_DOCS``.  A result on
the other side of that expectation is reported as a **warning**, not silently
accepted: on a synthetic corpus a component can legitimately be fitting noise,
but that has to be visible rather than quietly shipped.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.config import Config
from ..core.logging import get_logger
from ..sv.systems import ABLATION_SWITCHES, SWITCH_DOCS, apply_ablation, get_recipe
from .evaluator import compare_systems

__all__ = ["run_ablations", "ablation_row"]

log = get_logger("eval.ablation")

#: The system each ablation is measured against.
BASELINE_SYSTEM = "ivector_plda_full"


def ablation_row(
    switch: str,
    baseline: Mapping[str, float],
    ablated: Mapping[str, float],
    *,
    k: float = 0.5,
) -> dict[str, Any]:
    """Compare one ablated run against the baseline and flag surprises."""
    comparison = compare_systems(ablated, baseline, k=k)
    doc = SWITCH_DOCS.get(switch, {})
    expected = str(doc.get("expected", "worse"))
    delta = ablated["mean"] - baseline["mean"]
    # "worse" means the EER is expected to go UP when the feature is removed.
    matches = (delta > 0.0) if expected == "worse" else (delta < 0.0)
    return {
        "switch": switch,
        "expected_direction": expected,
        "observed_delta_eer": delta,
        "observed_relative": (delta / baseline["mean"]) if baseline["mean"] else float("inf"),
        "direction_as_expected": bool(matches),
        "baseline_mean_eer": baseline["mean"],
        "ablated_mean_eer": ablated["mean"],
        "verdict": comparison["verdict"],
        "hypothesis": doc.get("hypothesis", ""),
        "remedy": doc.get("remedy", ""),
    }


def run_ablations(
    config: Config,
    *,
    datasets: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    switches: Sequence[str] | None = None,
    baseline_system: str = BASELINE_SYSTEM,
) -> dict[str, Any]:
    """Run the baseline plus every requested ablation switch.

    Only the datasets that actually *discriminate* are used: on a clean corpus
    every ablation lands at 0.0% EER and the comparison carries no information.
    """
    from ..data.corpora import CorpusBuilderImpl
    from ..pipeline.voiceforge import evaluate_dataset_seed

    ds_ids = list(datasets) if datasets is not None else list(config.bench.ablation_datasets)
    seed_ids = list(seeds) if seeds is not None else list(config.bench.ablation_seeds)
    sw = list(switches) if switches is not None else sorted(ABLATION_SWITCHES)

    builder = CorpusBuilderImpl(config)
    rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    timings: list[dict[str, Any]] = []

    # --- baseline ---------------------------------------------------------
    t0 = time.perf_counter()
    base_rows: list[dict[str, Any]] = []
    for dataset_id in ds_ids:
        for seed in seed_ids:
            try:
                run = evaluate_dataset_seed(
                    config, dataset_id, seed, systems=[baseline_system], builder=builder
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(
                    {"kind": "ablation_baseline_failed", "dataset_id": dataset_id, "seed": seed,
                     "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            base_rows.extend(run["results"])
    baseline_seconds = time.perf_counter() - t0
    timings.append({"variant": "baseline", "seconds": round(baseline_seconds, 3)})

    if not base_rows:
        return {"rows": [], "warnings": warnings, "timings": timings, "baseline_system": baseline_system}

    from .evaluator import aggregate_cells

    baseline = aggregate_cells(base_rows, key="eer")
    base_stats = baseline[baseline_system]

    # --- ablations --------------------------------------------------------
    base_recipe = get_recipe(baseline_system)
    for switch in sw:
        recipe = apply_ablation(base_recipe, [switch])
        t0 = time.perf_counter()
        rows_for_switch: list[dict[str, Any]] = []
        for dataset_id in ds_ids:
            for seed in seed_ids:
                try:
                    run = evaluate_dataset_seed(
                        config, dataset_id, seed, systems=[recipe.system_id], builder=builder
                    )
                except Exception as exc:  # noqa: BLE001
                    warnings.append(
                        {"kind": "ablation_failed", "switch": switch, "dataset_id": dataset_id,
                         "seed": seed, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                for row in run["results"]:
                    row = {**row, "system_id": recipe.system_id}
                    row.pop("scores", None)
                    rows_for_switch.append(row)
        timings.append({"variant": switch, "seconds": round(time.perf_counter() - t0, 3)})
        if not rows_for_switch:
            warnings.append({"kind": "ablation_no_results", "switch": switch})
            continue

        stats = aggregate_cells(rows_for_switch, key="eer")[recipe.system_id]
        row = ablation_row(switch, base_stats, stats, k=float(config.gate.std_tolerance))
        row["per_cell"] = rows_for_switch
        rows.append(row)
        if not row["direction_as_expected"]:
            warnings.append(
                {
                    "kind": "ablation_unexpected_direction",
                    "switch": switch,
                    "expected": row["expected_direction"],
                    "observed_delta_eer": row["observed_delta_eer"],
                    "note": (
                        "Turning the feature off changed the EER in the opposite direction to the "
                        "registered expectation. On a synthetic corpus this can be legitimate (the "
                        "component may be fitting nuisance structure), but it must be explained, not "
                        "assumed away."
                    ),
                }
            )
        log.info("ablation %-16s EER %.4f -> %.4f (delta %+.4f)", switch, base_stats["mean"], stats["mean"],
                 row["observed_delta_eer"])

    return {
        "baseline_system": baseline_system,
        "baseline": base_stats,
        "datasets": ds_ids,
        "seeds": seed_ids,
        "rows": rows,
        "warnings": warnings,
        "timings": timings,
    }
