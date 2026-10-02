"""The system registry: five verification systems of increasing capability.

Each system is a *recipe* -- a declarative description of which components it
uses -- resolved at run time against a fitted :class:`SystemBundle`.  Keeping
the recipes declarative (rather than five hand-written pipelines) is what makes
the ablation honest: a switch such as ``no_nbc`` sets one flag, and the same
code path produces both the baseline and the ablated variant, so there is no
possibility of the two drifting apart.

The ladder
----------
``S0`` ``mfcc_cos_nbc``
    MFCC + CMVN, global mean from the training split, cosine.  No LDA, no GMM,
    no nuisance compensation.  The floor.

``S1`` ``gmmubm_map_cos``
    Full front end + VTLN + LDA + GMM-UBM + MAP, scored by the cosine of the
    MAP means.  A GMM system without the i-vector/PLDA machinery.

``S2`` ``ivector_plda_full``
    The reference system: everything on, PLDA scoring.

``S3`` ``ivector_plda_tvnbc_fuse``
    S2 plus a score-level fusion with the cosine backend, with the weight chosen
    on a *development* half only.

``S4`` ``ivector_plda_full_tier1``
    S2 with the pure-numpy DSP backend, quantifying the cost of having no
    librosa.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.errors import ConfigError
from ..core.types import FLOAT_DTYPE

__all__ = [
    "Recipe",
    "SYSTEMS",
    "SWITCH_DOCS",
    "ABLATION_SWITCHES",
    "get_recipe",
    "system_ids",
    "apply_ablation",
    "describe_systems",
]


@dataclass(frozen=True)
class Recipe:
    """Declarative description of one verification system.

    Every flag maps onto a component of the pipeline, so a system (or an
    ablation of it) is fully described by this record.
    """

    system_id: str
    tier: int
    uses_vtln: bool
    uses_lda: bool
    uses_gmm: bool
    uses_map: bool
    uses_ivector: bool
    uses_nbc: bool
    uses_cms: bool
    uses_length_norm: bool
    uses_plda: bool
    uses_slv: bool
    backend: str
    fuse: bool
    description: str
    ablate: frozenset[str] = field(default_factory=frozenset)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "tier": self.tier,
            "uses_vtln": self.uses_vtln,
            "uses_lda": self.uses_lda,
            "uses_gmm": self.uses_gmm,
            "uses_map": self.uses_map,
            "uses_ivector": self.uses_ivector,
            "uses_nbc": self.uses_nbc,
            "uses_cms": self.uses_cms,
            "uses_length_norm": self.uses_length_norm,
            "uses_plda": self.uses_plda,
            "uses_slv": self.uses_slv,
            "backend": self.backend,
            "fuse": self.fuse,
            "description": self.description,
            "ablate": sorted(self.ablate),
        }


#: The five systems, in increasing capability.
SYSTEMS: dict[str, Recipe] = {
    "mfcc_cos_nbc": Recipe(
        system_id="mfcc_cos_nbc",
        tier=0,
        uses_vtln=False,
        uses_lda=False,
        uses_gmm=False,
        uses_map=False,
        uses_ivector=False,
        uses_nbc=False,
        uses_cms=True,
        uses_length_norm=True,
        uses_plda=False,
        uses_slv=False,
        backend="auto",
        fuse=False,
        description="MFCC+CMVN -> global training mean -> cosine. Weak baseline, no LDA and no nuisance compensation.",
    ),
    "gmmubm_map_cos": Recipe(
        system_id="gmmubm_map_cos",
        tier=1,
        uses_vtln=True,
        uses_lda=True,
        uses_gmm=True,
        uses_map=True,
        uses_ivector=False,
        uses_nbc=False,
        uses_cms=True,
        uses_length_norm=True,
        uses_plda=False,
        uses_slv=False,
        backend="auto",
        fuse=False,
        description="Full front end + VTLN + LDA + GMM-UBM + MAP, scored by cosine on the MAP means.",
    ),
    "ivector_plda_full": Recipe(
        system_id="ivector_plda_full",
        tier=2,
        uses_vtln=True,
        uses_lda=True,
        uses_gmm=True,
        uses_map=True,
        uses_ivector=True,
        uses_nbc=True,
        uses_cms=True,
        uses_length_norm=True,
        uses_plda=True,
        uses_slv=True,
        backend="auto",
        fuse=False,
        description="Reference system: full front end, GMM-UBM + MAP, total variability + NBC + CMS, PLDA scoring.",
    ),
    "ivector_plda_tvnbc_fuse": Recipe(
        system_id="ivector_plda_tvnbc_fuse",
        tier=3,
        uses_vtln=True,
        uses_lda=True,
        uses_gmm=True,
        uses_map=True,
        uses_ivector=True,
        uses_nbc=True,
        uses_cms=True,
        uses_length_norm=True,
        uses_plda=True,
        uses_slv=True,
        backend="auto",
        fuse=True,
        description="Reference system fused with the cosine backend; fusion weight fitted on the dev half only.",
    ),
    "ivector_plda_full_tier1": Recipe(
        system_id="ivector_plda_full_tier1",
        tier=2,
        uses_vtln=True,
        uses_lda=True,
        uses_gmm=True,
        uses_map=True,
        uses_ivector=True,
        uses_nbc=True,
        uses_cms=True,
        uses_length_norm=True,
        uses_plda=True,
        uses_slv=True,
        backend="tier1",
        fuse=False,
        description="Reference system with the pure-numpy DSP backend: quantifies the cost of having no librosa.",
    ),
}

#: Ablation switches and the direction each is *expected* to move the EER.
#:
#: ``expected`` is ``"worse"`` when turning the feature off should make the EER
#: go up.  A measured result on the other side is a warning, not a failure: it
#: can legitimately happen on a synthetic corpus where a component is fitting
#: nuisance structure.  What it must never do is pass unnoticed, because that
#: is how a genuinely broken component gets shipped.
SWITCH_DOCS: dict[str, dict[str, str]] = {
    "no_cmvn": {
        "expected": "worse",
        "hypothesis": "Cepstral mean/variance normalisation removes per-utterance channel gain; without it the global mean absorbs the recording level.",
        "remedy": "Restore per-utterance CMVN, or add a session-level gain normalisation before the front end.",
    },
    "no_vtln": {
        "expected": "worse",
        "hypothesis": "VTLN aligns the vocal tract length; without it speakers with long and short tracts are separated mainly by channel rather than identity.",
        "remedy": "Widen the VTLN search range, or add a cepstral-mean-based tract-length normalisation.",
    },
    "no_lda": {
        "expected": "worse",
        "hypothesis": "LDA projects onto the between-speaker subspace; without it the GMM models nuisance directions as if they were speaker identity.",
        "remedy": "Increase the LDA rank towards n_speakers-1, or add explicit nuisance projection (NBC).",
    },
    "no_nbc": {
        "expected": "worse",
        "hypothesis": "Nuisance back-channel compensation removes the session directions; without it the i-vector absorbs channel and noise variation.",
        "remedy": "Estimate more nuisance directions, or estimate them from a channel-varied training corpus.",
    },
    "no_cms": {
        "expected": "worse",
        "hypothesis": "Cepstral mean subtraction on the supervector removes the session offset; without it the LLR sees a constant shift as speaker evidence.",
        "remedy": "Restore CMS, or subtract a session-level supervector mean estimated on the training split.",
    },
    "no_lengthnorm": {
        "expected": "worse",
        "hypothesis": "L2 length normalisation removes the correlation between i-vector norm and utterance duration; without it short enrollments score low for reasons unrelated to identity.",
        "remedy": "Restore length normalisation, or apply SLV normalisation instead.",
    },
    "plda_to_cosine": {
        "expected": "worse",
        "hypothesis": "PLDA models the two-covariance structure explicitly; a plain cosine treats between- and within-speaker variability identically.",
        "remedy": "Restore PLDA scoring, or whiten by Sigma_n before the cosine.",
    },
    "low_gauss": {
        "expected": "worse",
        "hypothesis": "A 64-component UBM under-resolves the acoustic space, so MAP adaptation cannot place a speaker precisely.",
        "remedy": "Increase n_gauss, or lower the MAP relevance factor so adaptation relies less on component resolution.",
    },
}

#: Switches that are valid for ablation runs.
ABLATION_SWITCHES: frozenset[str] = frozenset(SWITCH_DOCS)


def system_ids() -> tuple[str, ...]:
    """All registered system ids, in tier order."""
    return tuple(sorted(SYSTEMS, key=lambda k: (SYSTEMS[k].tier, k)))


def get_recipe(system_id: str) -> Recipe:
    """Look up a system recipe by id."""
    try:
        return SYSTEMS[system_id]
    except KeyError as exc:
        raise ConfigError(
            f"unknown system id {system_id!r}", system_id=system_id, known=list(system_ids())
        ) from exc


def apply_ablation(recipe: Recipe, switches: Sequence[str] | frozenset[str]) -> Recipe:
    """Return ``recipe`` with the given ablation switches applied.

    Raises
    ------
    ConfigError
        If a switch name is not registered, or if a switch contradicts another
        (e.g. ``no_lengthnorm`` together with ``no_cms`` is fine, but requesting
        a switch twice is not).
    """
    chosen = frozenset(switches)
    unknown = chosen - ABLATION_SWITCHES
    if unknown:
        raise ConfigError(
            f"unknown ablation switch(es): {sorted(unknown)}", unknown=sorted(unknown), known=sorted(ABLATION_SWITCHES)
        )
    if not chosen:
        return recipe

    updates: dict[str, Any] = {}
    if "no_cmvn" in chosen:
        updates["uses_cms"] = False
    if "no_vtln" in chosen:
        updates["uses_vtln"] = False
    if "no_lda" in chosen:
        updates["uses_lda"] = False
    if "no_nbc" in chosen:
        updates["uses_nbc"] = False
    if "no_cms" in chosen:
        updates["uses_cms"] = False
    if "no_lengthnorm" in chosen:
        updates["uses_length_norm"] = False
    if "plda_to_cosine" in chosen:
        updates["uses_plda"] = False
    ablated = recipe.ablate | chosen
    return replace(recipe, system_id=f"{recipe.system_id}[{'+'.join(sorted(chosen))}]", ablate=ablated, **updates)


def describe_systems() -> list[dict[str, Any]]:
    """Machine-readable system table for the report and ``benchmark.json``."""
    return [SYSTEMS[k].to_dict() for k in system_ids()]


def switch_documentation() -> list[dict[str, Any]]:
    """The ablation table, including the expected direction of every switch."""
    return [{"switch": k, **dict(v)} for k, v in sorted(SWITCH_DOCS.items())]
