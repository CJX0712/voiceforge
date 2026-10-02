"""Evaluation layer: aggregation, comparison, ablation, failure mining, report."""

from __future__ import annotations

from .benchmark import EXTERNAL_REFERENCE, run_benchmark, write_benchmark
from .evaluator import (
    aggregate_by_dataset,
    aggregate_cells,
    compare_systems,
    failure_table,
    mine_failure_cases,
)

__all__ = [
    "EXTERNAL_REFERENCE",
    "aggregate_by_dataset",
    "aggregate_cells",
    "compare_systems",
    "failure_table",
    "mine_failure_cases",
    "run_benchmark",
    "write_benchmark",
]
