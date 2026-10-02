"""Console report and the gate evaluation.

The report is deliberately explicit about what is **not** known: a gate whose
inputs were not measured is reported as ``not_evaluated`` rather than as a
pass.  A benchmark that quietly turns missing evidence into success is worse
than one that admits it has no result.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence, TextIO

from ..core.logging import TableWriter, safe_ascii
from ..core.memory import format_mb

__all__ = ["render_report", "evaluate_gates", "format_table", "VERDICT_PASS", "VERDICT_FAIL", "VERDICT_UNKNOWN"]

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_UNKNOWN = "not_evaluated"

_GLYPH = {VERDICT_PASS: "[OK]", VERDICT_FAIL: "[FAIL]", VERDICT_UNKNOWN: "[SKIP]"}


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[int], stream: TextIO) -> None:
    """Print a fixed-width table (column positions never shift between runs)."""
    writer = TableWriter(list(headers), list(widths), stream=stream)
    for row in rows:
        writer.row(list(row))


def evaluate_gates(doc: Mapping[str, Any], config: Any) -> dict[str, Any]:
    """Evaluate G1..G7 from a benchmark document.

    Any gate whose inputs are absent is marked ``not_evaluated``; it is never
    reported as a pass.
    """
    agg = doc.get("aggregate", {})
    eer = agg.get("eer", {})
    comps = agg.get("comparisons", {})
    thresholds = dict(doc.get("thresholds", {}))
    gates: dict[str, Any] = {}

    baseline = "ivector_plda_full"
    flagship = "ivector_plda_tvnbc_fuse"
    tier1 = "ivector_plda_full_tier1"

    # --- G1: flagship cuts the baseline EER by the required fraction ------
    cmp_g1 = comps.get("flagship_vs_baseline")
    if cmp_g1 is None or flagship not in eer or baseline not in eer:
        gates["G1_relative_reduction"] = {
            "status": VERDICT_UNKNOWN,
            "reason": f"needs both {flagship} and {baseline} in the results",
        }
    else:
        required = float(config.gate.g1_rel_reduction)
        actual = 1.0 - (cmp_g1["candidate_mean"] / cmp_g1["baseline_mean"]) if cmp_g1["baseline_mean"] else float("inf")
        gates["G1_relative_reduction"] = {
            "status": VERDICT_PASS if actual >= required else VERDICT_FAIL,
            "required_reduction": required,
            "observed_reduction": actual,
            "flagship_mean": cmp_g1["candidate_mean"],
            "baseline_mean": cmp_g1["baseline_mean"],
            "verdict_statistical": cmp_g1["verdict"],
            "note": (
                "A relative-reduction gate is only meaningful on a task that is not saturated. "
                "If the baseline is already near 0%, the required absolute improvement is smaller "
                "than the seed-to-seed spread and the gate is not discriminating."
            ),
        }

    # --- G2 / G3: absolute ceilings on the flagship -----------------------
    for gate_id, key, limit in (
        ("G2_absolute_eer", "eer", float(config.gate.g2_abs_eer)),
        ("G3_min_dcf", "min_dcf", float(config.gate.g3_min_dcf)),
    ):
        if flagship not in agg.get(key, {}):
            gates[gate_id] = {"status": VERDICT_UNKNOWN, "reason": f"{flagship} not evaluated"}
            continue
        value = float(agg[key][flagship]["mean"])
        gates[gate_id] = {
            "status": VERDICT_PASS if value <= limit else VERDICT_FAIL,
            "observed": value,
            "threshold": limit,
        }

    # --- G4: tier-1 degradation ------------------------------------------
    cmp_g4 = comps.get("tier1_vs_tier0_ratio")
    if cmp_g4 is None or tier1 not in eer or baseline not in eer:
        gates["G4_tier1_degradation"] = {"status": VERDICT_UNKNOWN, "reason": "tier1 system not evaluated"}
    else:
        gates["G4_tier1_degradation"] = {
            "status": VERDICT_PASS if cmp_g4["ratio"] <= float(config.gate.g4_tier1_ratio) else VERDICT_FAIL,
            "observed_ratio": cmp_g4["ratio"],
            "threshold": float(config.gate.g4_tier1_ratio),
        }

    # --- G5: determinism (needs a second run to compare) -----------------
    if "content_hash" in doc and len(doc.get("runs", [])) > 0:
        gates["G5_determinism"] = {
            "status": VERDICT_UNKNOWN,
            "content_hash": doc["content_hash"],
            "reason": "compare against a second independent run (scripts/check_determinism.py)",
        }
    else:
        gates["G5_determinism"] = {"status": VERDICT_UNKNOWN, "reason": "no content_hash"}

    # --- G6: runtime and memory ------------------------------------------
    elapsed = float(doc.get("elapsed_seconds", 0.0))
    rss = float(doc.get("peak_rss_mb", 0.0))
    if elapsed > 0.0:
        ok_time = elapsed <= float(config.gate.g6_max_seconds)
        ok_rss = rss <= float(config.gate.g6_max_rss_mb) if rss > 0 else False
        gates["G6_runtime"] = {
            "status": VERDICT_PASS if (ok_time and ok_rss) else VERDICT_FAIL,
            "elapsed_seconds": elapsed,
            "max_seconds": float(config.gate.g6_max_seconds),
            "peak_rss_mb": rss,
            "max_rss_mb": float(config.gate.g6_max_rss_mb),
            "peak_rss_note": None if rss > 0 else "RSS probe unavailable; treated as a failure",
        }
    else:
        gates["G6_runtime"] = {"status": VERDICT_UNKNOWN, "reason": "no timing recorded"}

    # --- G7: test quality (needs pytest-cov output) -----------------------
    gates["G7_quality"] = {
        "status": VERDICT_UNKNOWN,
        "required_coverage_pct": float(config.gate.g7_coverage),
        "reason": "run pytest --cov; the CLI does not execute the test suite",
    }

    for gate_id, entry in gates.items():
        thresholds.setdefault(gate_id, {}).update({"status": entry["status"]})
    return {"gates": gates, "thresholds": thresholds}


def render_report(doc: Mapping[str, Any], config: Any, *, stream: TextIO | None = None) -> str:
    """Render the human-readable report and return it as a string."""
    import io

    buf = stream or io.StringIO()
    gate_report = evaluate_gates(doc, config)
    doc = {**doc, "thresholds": gate_report["thresholds"]}

    print("=" * 78, file=buf)
    print("VoiceForge benchmark report", file=buf)
    print(f"  profile   : {doc['run_cfg']['profile']}", file=buf)
    print(f"  run_id    : {doc['run_id']}", file=buf)
    print(f"  created   : {doc['created_at']}", file=buf)
    print(f"  content_hash: {doc.get('content_hash', '<none>')}", file=buf)
    print("=" * 78, file=buf)

    env = doc.get("env", {})
    print("\nEnvironment", file=buf)
    rows = [
        ["python", env.get("python", "?")],
        ["numpy", env.get("numpy", "?")],
        ["scipy", env.get("scipy", "?")],
        ["librosa", env.get("librosa", "?")],
        ["blas", env.get("blas", "?")],
        ["backend tier", env.get("backend_tier", "?")],
        ["mel fingerprint", env.get("mel_fingerprint", "?")],
    ]
    format_table(["key", "value"], rows, [20, 40], buf)

    print("\nAggregate results (equal weight per dataset x seed cell)", file=buf)
    eer = doc.get("aggregate", {}).get("eer", {})
    dcf = doc.get("aggregate", {}).get("min_dcf", {})
    auc = doc.get("aggregate", {}).get("auc", {})
    rows = []
    for system in sorted(eer, key=lambda s: eer[s]["mean"]):
        rows.append([
            system,
            f"{eer[system]['mean']:.4f}",
            f"{eer[system]['std']:.4f}",
            f"{100 * eer[system]['mean']:.2f}%",
            f"{auc.get(system, {}).get('mean', float('nan')):.4f}",
            f"{dcf.get(system, {}).get('mean', float('nan')):.4f}",
            str(eer[system]["n_cells"]),
        ])
    format_table(
        ["system", "EER mean", "EER std", "EER %", "AUC", "minDCF", "cells"], rows, [26, 10, 10, 9, 8, 8, 6], buf
    )

    per_dataset = doc.get("per_dataset", {})
    if per_dataset:
        datasets = sorted({d for s in per_dataset.values() for d in s})
        systems = sorted(per_dataset)
        print("\nEER per dataset (mean over seeds)", file=buf)
        rows = []
        for system in systems:
            rows.append([system] + [f"{per_dataset[system].get(d, {}).get('mean', float('nan')):.4f}" for d in datasets])
        format_table(["system"] + datasets, rows, [26] + [11] * len(datasets), buf)

    print("\nGates", file=buf)
    rows = []
    for gate_id in sorted(gate_report["gates"]):
        entry = gate_report["gates"][gate_id]
        detail = entry.get("reason") or entry.get("observed") or entry.get("observed_ratio") or ""
        if isinstance(detail, float):
            detail = f"{detail:.4f}"
        rows.append([gate_id, _GLYPH.get(entry["status"], entry["status"]), str(detail)])
    format_table(["gate", "status", "detail"], rows, [26, 10, 34], buf)

    warnings = doc.get("warnings", [])
    if warnings:
        print(f"\nWarnings ({len(warnings)})", file=buf)
        for w in warnings[:12]:
            print(f"  - {w.get('kind')}: {jsonish(w)}", file=buf)
        if len(warnings) > 12:
            print(f"  ... and {len(warnings) - 12} more", file=buf)

    cases = doc.get("failure_cases", [])
    if cases:
        print(f"\nFailure cases ({len(cases)} total, by hypothesis)", file=buf)
        by_h = doc.get("failure_cases_by_hypothesis", {})
        format_table(["hypothesis", "count"], [[k, str(v)] for k, v in by_h.items()], [30, 8], buf)
        print("\n  Examples:", file=buf)
        for case in cases[:4]:
            print(
                f"  - [{case['dataset_id']}/{case['system_id']}] {case['kind']} "
                f"score={case['score']:.4f} {case['enroll_speaker']} vs {case['test_speaker']}",
                file=buf,
            )
            print(f"      {case['hypothesis']}", file=buf)
            print(f"      remedy: {case['remedy']}", file=buf)

    ref = doc.get("external_reference", [])
    if ref:
        print("\nExternal reference (NOT comparable -- different corpus and protocol)", file=buf)
        rows = [[r["system"], r["corpus"], f"{r['eer_pct']:.2f}%", str(r["comparable"])] for r in ref]
        format_table(["system", "corpus", "EER", "comparable"], rows, [22, 40, 8, 11], buf)

    print(f"\nRuntime: {doc.get('elapsed_seconds', 0):.1f}s   peak RSS: {format_mb(float(doc.get('peak_rss_mb', 0)))}", file=buf)
    return buf.getvalue()


def jsonish(obj: Mapping[str, Any]) -> str:
    """Compact one-line rendering of a warning for the console."""
    parts = []
    for k, v in obj.items():
        if k == "kind":
            continue
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def print_report(doc: Mapping[str, Any], config: Any, *, stream: TextIO | None = None) -> None:
    """Render and print the report (UTF-8 safe on Windows consoles)."""
    text = render_report(doc, config, stream=stream)
    out = stream or sys.stdout
    out.write(safe_ascii(text))
    out.write("\n")
