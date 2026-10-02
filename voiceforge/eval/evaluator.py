"""Aggregation across seeds, statistical comparison, and failure-case mining.

Three responsibilities:

* :func:`aggregate` -- mean / std over seeds, with the **equal weight per
  (dataset, seed) cell** rule.  Weighting by trial count would let the easiest
  dataset dominate the headline number.
* :func:`compare_systems` -- a three-state verdict (better / worse /
  indistinguishable) using the task-book rule "a win requires the mean gap to
  exceed ``k * (std_a + std_b)``".  Reporting only a mean gap invites reading
  noise as an improvement.
* :func:`mine_failure_cases` -- pull out the most informative trials and attach
  a hypothesis and a remedy from a rule table, so a failure is actionable
  rather than just a number.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..core.seed import rng
from ..core.types import FLOAT_DTYPE

__all__ = [
    "aggregate_cells",
    "compare_systems",
    "mine_failure_cases",
    "failure_table",
    "HYPOTHESIS_RULES",
    "VERDICT_BETTER",
    "VERDICT_WORSE",
    "VERDICT_TIE",
]

VERDICT_BETTER = "better"
VERDICT_WORSE = "worse"
VERDICT_TIE = "indistinguishable"


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def aggregate_cells(rows: Sequence[Mapping[str, Any]], *, key: str = "eer") -> dict[str, dict[str, float]]:
    """Aggregate ``(system, dataset, seed)`` rows into ``system -> {mean, std, n}``.

    Every ``(dataset, seed)`` cell contributes equally regardless of how many
    trials it used, so a dataset with more trials cannot dominate.
    """
    by_system: dict[str, list[float]] = {}
    for row in rows:
        by_system.setdefault(str(row["system_id"]), []).append(float(row[key]))

    out: dict[str, dict[str, float]] = {}
    for system, values in by_system.items():
        arr = np.asarray(values, dtype=FLOAT_DTYPE)
        out[system] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
            "n_cells": int(arr.size),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def aggregate_by_dataset(rows: Sequence[Mapping[str, Any]], *, key: str = "eer") -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate into ``system -> dataset -> stats``."""
    by_system: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_system.setdefault(str(row["system_id"]), []).append(row)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for system, group in by_system.items():
        per_dataset: dict[str, list[float]] = {}
        for row in group:
            per_dataset.setdefault(str(row["dataset_id"]), []).append(float(row[key]))
        out[system] = {}
        for dataset, values in per_dataset.items():
            arr = np.asarray(values, dtype=FLOAT_DTYPE)
            out[system][dataset] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "n_seeds": int(arr.size),
            }
    return out


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def compare_systems(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    k: float = 0.5,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Three-state verdict on ``candidate`` versus ``baseline``.

    A verdict of ``better`` requires ``|mean_gap| > k * (std_c + std_b)``.  With
    the default ``k = 0.5`` a gap must exceed half the summed spread, which is
    deliberately conservative: on a saturated task the seed-to-seed spread is
    comparable to the differences being claimed, and a one-sided test would
    report those as improvements.
    """
    c_mean, c_std = float(candidate["mean"]), float(candidate["std"])
    b_mean, b_std = float(baseline["mean"]), float(baseline["std"])
    gap = c_mean - b_mean
    threshold = k * (c_std + b_std)
    rel = gap / b_mean if abs(b_mean) > 1e-12 else float("inf")

    if abs(gap) <= threshold:
        verdict = VERDICT_TIE
    elif (gap < 0) == lower_is_better:
        verdict = VERDICT_BETTER
    else:
        verdict = VERDICT_WORSE

    return {
        "candidate_mean": c_mean,
        "candidate_std": c_std,
        "baseline_mean": b_mean,
        "baseline_std": b_std,
        "gap": gap,
        "relative_gap": rel,
        "threshold": threshold,
        "verdict": verdict,
        "k": k,
    }


# --------------------------------------------------------------------------
# failure mining
# --------------------------------------------------------------------------
#: ``(symptom predicate name) -> (hypothesis, remedy)``.  The predicate is
#: evaluated in order against the mined case; the first match wins.  Keeping
#: this as data rather than ``if`` statements means the report can print the
#: rationale verbatim and a reviewer can check it.
HYPOTHESIS_RULES: tuple[dict[str, Any], ...] = (
    {
        "id": "impostor_squeaks_through",
        "when": "impostor_top",
        "hypothesis": (
            "A high-scoring impostor pair indicates the nuisance subspace is not removed: "
            "session/channel variation still dominates the i-vector."
        ),
        "remedy": "Increase ivector.nbc_dims, or train the UBM on a channel-varied corpus so nuisance directions are populated.",
    },
    {
        "id": "genuine_rejected",
        "when": "genuine_bottom",
        "hypothesis": (
            "A low-scoring genuine pair indicates an under-trained model or too little adaptation "
            "data: the enrollment speaker is not represented well enough to beat impostors."
        ),
        "remedy": "Increase the MAP relevance factor tau, raise gmm.n_gauss, or lengthen the enrollment utterance.",
    },
    {
        "id": "narrowband_collapse",
        "when": "dataset_has_channel",
        "channel": "telephone",
        "hypothesis": (
            "On a narrowband channel the identity cues above F3/4 kHz are gone, so systems that rely on "
            "spectral detail in that band lose their advantage."
        ),
        "remedy": "Add band-limited training augmentation, or use a front end whose mel filterbank stops near 3.4 kHz.",
    },
    {
        "id": "reverb_smeared",
        "when": "dataset_has_channel",
        "channel": "reverb",
        "hypothesis": (
            "Reverberation smears formant transitions and adds temporal variability that a mean-pooled "
            "embedding cannot represent."
        ),
        "remedy": "Increase delta orders, train with reverberant augmentation, or rely on a distribution model (GMM/PLDA).",
    },
    {
        "id": "heavy_babble",
        "when": "noise",
        "noise": "babble",
        "hypothesis": (
            "Babble interference is itself speech and therefore spectrally similar to the target; a "
            "mean-pooled embedding absorbs the interferer."
        ),
        "remedy": "Use CMVN plus mean subtraction of the interfering speech, or move to a discriminatively trained backend.",
    },
    {
        "id": "generic",
        "when": "always",
        "hypothesis": "No specific rule matched; the trial is reported for manual inspection.",
        "remedy": "Inspect the enrollment/test waveforms and the score distribution for this pair.",
    },
)


@dataclass
class FailureCase:
    """One mined trial with an attached hypothesis."""

    system_id: str
    dataset_id: str
    seed: int
    kind: str
    enroll_id: str
    test_id: str
    enroll_speaker: str
    test_speaker: str
    label: bool
    score: float
    rank: int
    hypothesis_id: str
    hypothesis: str
    remedy: str
    context: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "dataset_id": self.dataset_id,
            "seed": self.seed,
            "kind": self.kind,
            "enroll_id": self.enroll_id,
            "test_id": self.test_id,
            "enroll_speaker": self.enroll_speaker,
            "test_speaker": self.test_speaker,
            "label": self.label,
            "score": self.score,
            "rank": self.rank,
            "hypothesis_id": self.hypothesis_id,
            "hypothesis": self.hypothesis,
            "remedy": self.remedy,
            "context": dict(self.context),
        }


def _attribute(case_kind: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the first matching hypothesis rule for a mined case."""
    for rule in HYPOTHESIS_RULES:
        when = rule["when"]
        if when == "always":
            return rule
        if when == case_kind:
            return rule
        if when == "dataset_has_channel" and context.get("channel") == rule.get("channel"):
            return rule
        if when == "noise" and context.get("noise") == rule.get("noise"):
            return rule
    return HYPOTHESIS_RULES[-1]


def mine_failure_cases(
    scores: np.ndarray,
    trials: Sequence[Any],
    *,
    system_id: str,
    dataset_id: str,
    seed: int,
    context: Mapping[str, Any] | None = None,
    n_cases: int = 3,
) -> list[FailureCase]:
    """Mine the most informative trials: top impostors and bottom genuines.

    These are the two populations that dominate the EER, so fixing them is what
    actually moves the metric; the middle of the distribution is uninformative.
    """
    s = np.asarray(scores, dtype=FLOAT_DTYPE).ravel()
    labels = np.array([bool(t.label) for t in trials], dtype=bool)
    ctx = dict(context or {})

    picked: list[FailureCase] = []
    imp_idx = np.nonzero(~labels)[0]
    gen_idx = np.nonzero(labels)[0]

    if imp_idx.size:
        # Highest-scoring impostors: the ones the system should have rejected.
        order = imp_idx[np.argsort(-s[imp_idx], kind="stable")][:n_cases]
        for rank, i in enumerate(order, start=1):
            rule = _attribute("impostor_top", ctx)
            picked.append(_case(system_id, dataset_id, seed, "impostor_top", trials, i, float(s[i]), rank, rule, ctx))

    if gen_idx.size:
        # Lowest-scoring genuines: the ones the system should have accepted.
        order = gen_idx[np.argsort(s[gen_idx], kind="stable")][:n_cases]
        for rank, i in enumerate(order, start=1):
            rule = _attribute("genuine_bottom", ctx)
            picked.append(_case(system_id, dataset_id, seed, "genuine_bottom", trials, i, float(s[i]), rank, rule, ctx))

    return picked


def _case(system_id, dataset_id, seed, kind, trials, i, score, rank, rule, ctx) -> FailureCase:
    t = trials[int(i)]
    return FailureCase(
        system_id=system_id,
        dataset_id=dataset_id,
        seed=seed,
        kind=kind,
        enroll_id=t.enroll_id,
        test_id=t.test_id,
        enroll_speaker=t.enroll_speaker,
        test_speaker=t.test_speaker,
        label=bool(t.label),
        score=score,
        rank=rank,
        hypothesis_id=str(rule["id"]),
        hypothesis=str(rule["hypothesis"]),
        remedy=str(rule["remedy"]),
        context=ctx,
    )


def failure_table(cases: Iterable[FailureCase]) -> dict[str, int]:
    """Count mined cases per hypothesis id."""
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.hypothesis_id] = counts.get(case.hypothesis_id, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
