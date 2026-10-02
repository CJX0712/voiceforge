"""The 4-cell pipeline ablation and the INV-1..4 acceptance criteria.

Two questions this answers, both of which the current numbers make urgent:

1. Does **supervector centring** (subtract the UBM mean, then L2) account for
   the collapse where every pair of speakers looked alike?
2. Does the **intermediate LDA** that Dehak's pipeline has between the
   supervector and total variability, and which the original specification
   omitted, account for anything?

Cells::

    A  no LDA, no centring      (the state before this change)
    B  no LDA, centring
    C  LDA,    no centring
    D  LDA,    centring          (the full Dehak pipeline)

Acceptance criteria (INV-1..4), replacing ``separation_margin`` as the gate:

======  ====================================================================
INV-1  ``EER(S0) - EER(S1) >= 0.12`` absolute, per configuration
INV-2  ``EER(S1) > EER(S2)`` in at least 4 of 6 configurations
INV-3  ``EER(S0) >= 0.20`` in every configuration
INV-4  within-speaker i-vector cosine < between-speaker cosine + 0.30
======  ====================================================================
"""

from __future__ import annotations

import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.config import Config
from ..core.logging import get_logger

__all__ = [
    "CELLS",
    "INVARIANTS",
    "run_cell_ablation",
    "cosine_diagnostics",
    "evaluate_invariants",
]

log = get_logger("eval.cells")

#: ``(cell_id, use_lda, center)`` for the four pipeline variants.
CELLS: tuple[tuple[str, bool, bool], ...] = (
    ("A_noLDA_noCenter", False, False),
    ("B_noLDA_center", False, True),
    ("C_LDA_noCenter", True, False),
    ("D_LDA_center", True, True),
)


def _configure(config: Config, *, use_lda: bool, center: bool) -> Config:
    """Return a config with LDA and supervector centring switched as requested.

    LDA is *disabled* by setting ``lda_dim = 0`` (the schema documents 0 as
    "off"), and centring by clearing the boolean.  Everything else is left
    untouched so the cells differ in exactly one factor.
    """
    return config.evolve(
        proj={"lda_dim": int(config.proj.lda_dim) if use_lda else 0},
        ivector={"center_supervector": bool(center)},
    )


def cosine_diagnostics(context) -> dict[str, float]:
    """Within- vs between-speaker cosine for the training embeddings.

    This is the direct measurement of whether centring worked.  Before
    centring, every MAP supervector keeps a large common component and the
    within/between cosines are nearly equal; afterwards the within-speaker
    cosine must drop well below the between-speaker one.
    """
    from ..sv.ivector import center_supervector
    from ..sv.map import map_adapt
    from ..sv.scoring import cosine_similarity

    bundle = context.bundle
    vecs: dict[str, list[np.ndarray]] = {}
    for uid, spk in context.labels.items():
        if spk not in bundle.train_speakers:
            continue
        feats = context.features[uid]
        if feats.shape[0] == 0:
            continue
        proj = (
            bundle.lda.transform(feats)
            if (bundle.lda is not None and context.config.proj.lda_dim > 0)
            else feats
        )
        adapted = map_adapt(
            bundle.ubm,
            proj,
            tau=float(context.config.map.tau),
            variant=str(context.config.map.variant),
            adapt_var=bool(context.config.map.adapt_var),
        )
        raw = adapted.means.ravel()
        n = float(np.linalg.norm(raw))
        vecs.setdefault(spk, []).append(raw / n if n > 1e-12 else raw)

    def pairs(groups: Mapping[str, Sequence[np.ndarray]]) -> list[float]:
        out: list[float] = []
        for items in groups.values():
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    out.append(float(np.dot(items[i], items[j])))
        return out

    within = pairs(vecs)
    firsts = [v[0] for v in vecs.values() if v]
    between = [float(np.dot(a, b)) for i, a in enumerate(firsts) for b in firsts[i + 1 :]]

    # And the same statistic after centring, which is what the pipeline uses.
    cvecs: dict[str, list[np.ndarray]] = {}
    for uid, spk in context.labels.items():
        if spk not in bundle.train_speakers:
            continue
        feats = context.features[uid]
        if feats.shape[0] == 0:
            continue
        proj = (
            bundle.lda.transform(feats)
            if (bundle.lda is not None and context.config.proj.lda_dim > 0)
            else feats
        )
        adapted = map_adapt(
            bundle.ubm,
            proj,
            tau=float(context.config.map.tau),
            variant=str(context.config.map.variant),
            adapt_var=bool(context.config.map.adapt_var),
        )
        cvecs.setdefault(spk, []).append(
            center_supervector(adapted.means, bundle.ubm.means, length_norm=True)
        )
    c_within = pairs(cvecs)
    c_firsts = [v[0] for v in cvecs.values() if v]
    c_between = [float(np.dot(a, b)) for i, a in enumerate(c_firsts) for b in c_firsts[i + 1 :]]

    return {
        "raw_within_cosine": float(np.mean(within)) if within else float("nan"),
        "raw_between_cosine": float(np.mean(between)) if between else float("nan"),
        "centered_within_cosine": float(np.mean(c_within)) if c_within else float("nan"),
        "centered_between_cosine": float(np.mean(c_between)) if c_between else float("nan"),
        "raw_gap": (float(np.mean(between)) - float(np.mean(within))) if within and between else float("nan"),
        "centered_gap": (
            (float(np.mean(c_between)) - float(np.mean(c_within))) if c_within and c_between else float("nan")
        ),
        "n_speakers": len(vecs),
    }


def run_cell_ablation(
    config: Config,
    *,
    datasets: Sequence[str] = ("D1_clean", "D2_white5", "D3_tel8", "D4_rev0", "D5_short1", "D6_mixneg"),
    seeds: Sequence[int] = (17, 29, 41),
    systems: Sequence[str] = ("mfcc_cos_nbc", "gmmubm_map_cos", "ivector_plda_full"),
) -> dict[str, Any]:
    """Run the four cells over the dataset x seed grid and collect EERs.

    Returns a document with, per cell: the per-(dataset, seed) EERs, the
    mean/std, the cosine diagnostics, and the model geometry (``tv_dim``,
    ``n_kept_components``, ``trace_sigma_w``, ``trace_sigma_n``).
    """
    from ..data.corpora import CorpusBuilderImpl
    from ..eval.evaluator import aggregate_cells
    from ..pipeline.voiceforge import evaluate_dataset_seed

    # A cache directory of its own: the ablation runs long enough that sharing a
    # cache with a concurrent `cli run` would race on the same corpus files.
    config = config.evolve(data={"cache_dir": os.path.join(config.data.cache_dir, "cells")})

    cells: dict[str, Any] = {}
    for cell_id, use_lda, center in CELLS:
        cfg = _configure(config, use_lda=use_lda, center=center)
        builder = CorpusBuilderImpl(cfg)
        rows: list[dict[str, Any]] = []
        geometry: list[dict[str, Any]] = []
        cosines: list[dict[str, float]] = []
        t0 = time.perf_counter()
        for dataset_id in datasets:
            for seed in seeds:
                try:
                    run = evaluate_dataset_seed(
                        cfg, dataset_id, seed, systems=list(systems), builder=builder
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning("cell %s %s/%s failed: %s", cell_id, dataset_id, seed, exc)
                    continue
                for row in run["results"]:
                    row = {k: v for k, v in row.items() if k != "scores"}
                    rows.append(row)
                b = run["bundle"]
                geometry.append(
                    {
                        "dataset_id": dataset_id,
                        "seed": seed,
                        "lda_dim": (b.get("lda") or {}).get("n_components"),
                        "tv_dim": (b.get("tv") or {}).get("tv_dim"),
                        "n_kept_components": (b.get("tv") or {}).get("n_kept_components"),
                        "plda_dim": (b.get("plda") or {}).get("dim"),
                        "trace_sigma_w": (b.get("plda") or {}).get("trace_sigma_w"),
                        "trace_sigma_n": (b.get("plda") or {}).get("trace_sigma_n"),
                        "ubm_dim": (b.get("ubm") or {}).get("dim"),
                        "ubm_gauss": (b.get("ubm") or {}).get("n_gauss"),
                    }
                )
                ctx = run.get("_ctx")
                if ctx is not None:
                    cosines.append(cosine_diagnostics(ctx))
        agg = aggregate_cells(rows, key="eer") if rows else {}
        cells[cell_id] = {
            "use_lda": use_lda,
            "center": center,
            "seconds": round(time.perf_counter() - t0, 2),
            "aggregate": agg,
            "rows": rows,
            "geometry": geometry,
            "cosines": cosines,
            "cosines_mean": _mean_cosines(cosines),
        }
        log.info("cell %-18s %s", cell_id, {k: round(v["mean"], 4) for k, v in agg.items()})

    return {
        "cells": cells,
        "datasets": list(datasets),
        "seeds": list(seeds),
        "systems": list(systems),
        "invariants": evaluate_invariants(cells, datasets=list(datasets)),
    }


def _mean_cosines(cosines: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average the cosine diagnostics over runs, ignoring non-finite entries."""
    if not cosines:
        return {}
    keys = [k for k in cosines[0] if isinstance(cosines[0][k], (int, float)) and k != "n_speakers"]
    out: dict[str, float] = {}
    for k in keys:
        vals = [float(c[k]) for c in cosines if np.isfinite(float(c.get(k, np.nan)))]
        if vals:
            out[k] = float(np.mean(vals))
    out["n_speakers"] = float(np.mean([float(c.get("n_speakers", 0)) for c in cosines]))
    return out


#: The four acceptance criteria, as machine-readable declarations.
INVARIANTS: tuple[dict[str, Any], ...] = (
    {
        "id": "INV-1",
        "statement": "EER(S0) - EER(S1) >= 0.12 absolute, per configuration",
        "kind": "per_config",
        "threshold": 0.12,
        "compare": ("mfcc_cos_nbc", "gmmubm_map_cos", "subtract"),
    },
    {
        "id": "INV-2",
        "statement": "EER(S1) > EER(S2) in at least 4 of 6 configurations",
        "kind": "count",
        "threshold": 4,
        "total": 6,
        "compare": ("gmmubm_map_cos", "ivector_plda_full", "worse"),
    },
    {
        "id": "INV-3",
        "statement": "EER(S0) >= 0.20 in every configuration",
        "kind": "per_config",
        "threshold": 0.20,
        "compare": ("mfcc_cos_nbc", None, "at_least"),
    },
    {
        "id": "INV-4",
        "statement": "within-speaker i-vector cosine < between-speaker cosine + 0.30",
        "kind": "cosine_gap",
        "threshold": 0.30,
    },
)


def evaluate_invariants(
    cells: Mapping[str, Any], *, datasets: Sequence[str] | None = None
) -> dict[str, Any]:
    """Evaluate INV-1..4 from the per-cell results."""
    # Per-configuration (dataset, seed) EER for each system.
    by_cell: dict[str, dict[tuple[str, int], dict[str, float]]] = {}
    for cell_id, cell in cells.items():
        table: dict[tuple[str, int], dict[str, float]] = {}
        for row in cell.get("rows", []):
            table[(str(row["dataset_id"]), int(row["seed"]))] = {str(row["system_id"]): float(row["eer"])}
        by_cell[cell_id] = table

    configs = sorted({k for t in by_cell.values() for k in t})
    out: dict[str, Any] = {}

    # --- INV-1 / INV-2 / INV-3 ------------------------------------------------
    per_config: list[dict[str, Any]] = []
    inv1_hits = inv2_hits = inv3_hits = 0
    for key in configs:
        entry: dict[str, Any] = {"dataset_id": key[0], "seed": key[1], "cells": {}}
        for cell_id, table in by_cell.items():
            row = table.get(key, {})
            entry["cells"][cell_id] = row
        s0 = entry["cells"].get("A_noLDA_noCenter", {}).get("mfcc_cos_nbc")
        s1 = entry["cells"].get("A_noLDA_noCenter", {}).get("gmmubm_map_cos")
        s2 = entry["cells"].get("A_noLDA_noCenter", {}).get("ivector_plda_full")
        # INV-1 and INV-3 are properties of the *task*, so they are evaluated on
        # the reference cell A; INV-2 too.  They do not depend on the cell under
        # test, which is exactly why they can catch a broken pipeline.
        if s0 is not None and s1 is not None:
            entry["inv1_gap"] = s0 - s1
            entry["inv1_pass"] = bool((s0 - s1) >= 0.12)
            inv1_hits += int(entry["inv1_pass"])
        if s1 is not None and s2 is not None:
            entry["inv2_pass"] = bool(s1 > s2)
            inv2_hits += int(entry["inv2_pass"])
        if s0 is not None:
            entry["inv3_pass"] = bool(s0 >= 0.20)
            inv3_hits += int(entry["inv3_pass"])
        per_config.append(entry)

    n_cfg = max(len(configs), 1)
    out["INV-1"] = {
        "statement": INVARIANTS[0]["statement"],
        "status": "pass" if inv1_hits == n_cfg else "fail",
        "hits": inv1_hits,
        "total": n_cfg,
        "details": [
            {"dataset_id": e["dataset_id"], "seed": e["seed"], "gap": e.get("inv1_gap"), "pass": e.get("inv1_pass")}
            for e in per_config
            if "inv1_gap" in e
        ],
    }
    out["INV-2"] = {
        "statement": INVARIANTS[1]["statement"],
        "status": "pass" if inv2_hits >= 4 else "fail",
        "hits": inv2_hits,
        "total": n_cfg,
        "required": 4,
    }
    out["INV-3"] = {
        "statement": INVARIANTS[2]["statement"],
        "status": "pass" if inv3_hits == n_cfg else "fail",
        "hits": inv3_hits,
        "total": n_cfg,
    }

    # --- INV-4: cosine gap, per cell ----------------------------------------
    cos_rows = []
    inv4_hits = 0
    for cell_id, cell in cells.items():
        c = cell.get("cosines_mean", {})
        gap = c.get("centered_gap")
        entry = {"cell": cell_id, **c}
        if gap is not None and np.isfinite(gap):
            entry["pass"] = bool(gap > 0.30)
            entry["threshold"] = 0.30
            inv4_hits += int(entry["pass"])
        cos_rows.append(entry)
    out["INV-4"] = {
        "statement": INVARIANTS[3]["statement"],
        "status": "pass" if inv4_hits == len(cells) and cells else "fail",
        "hits": inv4_hits,
        "total": len(cells),
        "per_cell": cos_rows,
    }

    out["per_config"] = per_config
    out["overall"] = "pass" if all(out[k]["status"] == "pass" for k in ("INV-1", "INV-2", "INV-3", "INV-4")) else "fail"
    return out
