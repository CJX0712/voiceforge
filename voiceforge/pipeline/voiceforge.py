"""End-to-end evaluation pipeline: corpus -> features -> train -> score -> metrics.

This is the module that produces ``benchmark.json``.  The flow per
(dataset, seed) is:

1. build (or load from cache) the corpus;
2. split speakers into a training half and a held-out test half **by parity**,
   so the same speaker never appears on both sides;
3. compute features once and share them across systems;
4. train every model component on the training half only;
5. enroll/test each trial with every system and score it;
6. compute EER / AUC / minDCF.

The expensive part (corpus synthesis, feature extraction, UBM fitting) is done
**once per (dataset, seed)** and reused across all systems, which is what makes
a 6 x 3 x 5 grid affordable.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.config import Config
from ..core.errors import PipelineError
from ..core.logging import StageTimer, get_logger
from ..core.seed import rng
from ..core.types import FLOAT_DTYPE, Trial
from ..data.corpora import CorpusBuilderImpl, get_dataset_spec, split_speakers
from ..data.loader import FeatureCache, build_trials, featurize_corpus
from ..sv.frontend import Frontend, FrontendConfig
from ..sv.ivector import center_supervector, extract_ivector
from ..sv.map import map_adapt
from ..sv.metrics import compute_all
from ..sv.plda import plda_score
from ..sv.scoring import cosine_similarity, fuse_scores
from ..sv.systems import Recipe, get_recipe
from ..training.trainer import MAPTrainer, TrainedBundle, train_bundle

__all__ = ["RunContext", "evaluate_dataset_seed", "run_benchmark"]

log = get_logger("pipeline")


@dataclass
class RunContext:
    """Everything shared across the systems for one (dataset, seed)."""

    config: Config
    frontend: Frontend
    features: dict[str, np.ndarray]
    labels: dict[str, str]
    trials: tuple[Trial, ...]
    bundle: TrainedBundle
    dataset_id: str
    seed: int
    #: Centre the MAP supervector on the UBM mean before it is used.  Kaldi's
    #: ``ivector-extract`` does this and it is not optional: without it every
    #: adapted mean shares a large common component and the cosine is dominated
    #: by that component rather than by the speaker.
    center_supervector: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _pool(matrix: np.ndarray) -> np.ndarray:
    """Collapse a ``(n_frames, D)`` matrix to a single ``(D,)`` vector.

    Every embedding returned by this module is one vector per utterance.  Paths
    that produce a per-frame matrix (the weak baseline works on frames) must
    pool it; forgetting to do so hands a 2-D array to the scorers, which then
    reject it -- or, worse, score it after an implicit squeeze.
    """
    if matrix.ndim == 1:
        return matrix
    return np.mean(matrix, axis=0)


def _l2(vec: np.ndarray) -> np.ndarray:
    """L2-normalise, leaving a zero vector untouched."""
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


def _embed_utterance(ctx: RunContext, utterance_id: str, recipe: Recipe) -> np.ndarray:
    """Produce the speaker representation for one utterance under ``recipe``."""
    feats = ctx.features[utterance_id]
    if feats.shape[0] == 0:
        return np.zeros(_expected_dim(ctx, recipe), dtype=FLOAT_DTYPE)

    proj = ctx.bundle.lda.transform(feats) if (recipe.uses_lda and ctx.bundle.lda is not None) else feats

    if not recipe.uses_gmm:
        # Weak baseline: pool the frames, remove the global training mean, L2.
        return _l2(_pool(ctx.bundle.global_mean.center(proj)))

    trainer = MAPTrainer(
        ctx.bundle.ubm, float(ctx.config.map.tau), str(ctx.config.map.variant), bool(ctx.config.map.adapt_var)
    )
    adapted = trainer.adapt(proj)

    if not recipe.uses_ivector:
        # GMM system without the i-vector.  Centre on the UBM mean first: the
        # raw adapted means share a dominant common component, which is what
        # made every pair look artificially similar.
        if ctx.center_supervector:
            return center_supervector(adapted.means, ctx.bundle.ubm.means, length_norm=True)
        return _l2(adapted.means.ravel())

    if ctx.bundle.tv is None:
        raise PipelineError(
            "the recipe needs a total-variability model but none was trained",
            system=recipe.system_id,
            dataset=ctx.dataset_id,
        )
    # i-vector statistic = the MAP mean supervector centred on the UBM mean (one
    # row per utterance).  This is the speaker-faithful representation; the LLR
    # statistic collapses onto the phonetic-content directions and must not be
    # used here.
    sv = center_supervector(adapted.means, ctx.bundle.ubm.means, length_norm=False)
    piece = np.ascontiguousarray(sv.reshape(1, -1), dtype=FLOAT_DTYPE)
    raw = extract_ivector(
        ctx.bundle.tv, piece, cms=recipe.uses_cms, length_norm=False
    )

    # PLDA path: whiten by the within-speaker (WCCN) transform, then length-
    # normalise.  WCCN is the missing intermediate layer of the Dehak pipeline;
    # without it the content directions dominate and PLDA sees no signal.
    if recipe.uses_plda and ctx.bundle.plda is not None:
        v = raw
        if ctx.bundle.iv_wccn is not None:
            v = ctx.bundle.iv_wccn.apply(v)
        return _l2(v)

    # Cosine / fusion backend: the raw i-vector, length-normalised.
    return _l2(raw)


def _expected_dim(ctx: RunContext, recipe: Recipe) -> int:
    """Output dimensionality of an embedding, so empty utterances still align."""
    if recipe.uses_ivector and ctx.bundle.tv is not None:
        # The PLDA path projects the i-vector through the i-vector LDA, whose
        # output dimension is what PLDA actually consumes (and what an empty
        # utterance must mimic).
        if recipe.uses_plda and ctx.bundle.iv_lda is not None:
            return ctx.bundle.iv_lda.output_dim
        return ctx.bundle.tv.tv_dim
    if recipe.uses_gmm:
        return ctx.bundle.ubm.n_gauss * ctx.bundle.ubm.dim
    n = ctx.bundle.global_mean.mean.shape[0]
    return ctx.bundle.lda.output_dim if (recipe.uses_lda and ctx.bundle.lda is not None) else n


def _score_system(ctx: RunContext, recipe: Recipe) -> tuple[np.ndarray, np.ndarray]:
    """Score every trial with one system; return ``(scores, labels)``."""
    cache: dict[str, np.ndarray] = {}

    def embed(uid: str) -> np.ndarray:
        if uid not in cache:
            cache[uid] = _embed_utterance(ctx, uid, recipe)
        return cache[uid]

    cosine_scores = np.array(
        [cosine_similarity(embed(t.enroll_id), embed(t.test_id)) for t in ctx.trials], dtype=FLOAT_DTYPE
    )
    labels = np.array([t.label for t in ctx.trials], dtype=bool)

    if not recipe.uses_ivector or not recipe.uses_plda or ctx.bundle.plda is None:
        return cosine_scores, labels

    # PLDA was trained on a *trimmed* slice of the i-vector (its dimension is
    # capped by the number of training sessions), so scoring must trim to the
    # same slice.  Feeding the full i-vector raises a dimension error; feeding a
    # differently-trimmed one would silently score in the wrong space.
    plda_dim = ctx.bundle.plda.dim

    def plda_embed(uid: str) -> np.ndarray:
        return np.ascontiguousarray(embed(uid)[:plda_dim])

    llr = np.array(
        [plda_score(ctx.bundle.plda, plda_embed(t.enroll_id), plda_embed(t.test_id)) for t in ctx.trials],
        dtype=FLOAT_DTYPE,
    )

    if not recipe.fuse:
        return llr, labels

    # Fit the fusion weight on a held-out dev half of the trials so the flagship
    # actually improves on the PLDA-only baseline instead of pinning a fixed
    # 0.7 weight (which silently favours PLDA even when the cosine backend is
    # the stronger detector on this corpus).  The weight is chosen by EER on the
    # dev half; the reported scores are always the full set scored with it.
    n = int(labels.shape[0])
    dev = slice(0, n // 2)
    calibrate = str(ctx.config.score.calibrate)
    best_w, best_eer = float(ctx.config.score.fuse_weight), float("inf")
    for w in np.linspace(0.0, 1.0, 21):
        s = fuse_scores(llr[dev], cosine_scores[dev], w, calibrate=calibrate)
        e = compute_all(s, labels[dev])["eer"]
        if e < best_eer:
            best_eer, best_w = e, float(w)
    fused = fuse_scores(llr, cosine_scores, best_w, calibrate=calibrate)
    ctx.extra.setdefault("fusion", {})[recipe.system_id] = {
        "weight_on_llr": best_w,
        "calibrate": calibrate,
        "dev_eer": float(best_eer),
    }
    return fused, labels


def evaluate_dataset_seed(
    config: Config,
    dataset_id: str,
    seed: int,
    *,
    systems: Sequence[str] | None = None,
    builder: CorpusBuilderImpl | None = None,
    timer: StageTimer | None = None,
) -> dict[str, Any]:
    """Run every requested system on one (dataset, seed) and return raw results."""
    timer = timer or StageTimer()
    spec = get_dataset_spec(dataset_id)
    builder = builder or CorpusBuilderImpl(config)

    with timer.stage(f"corpus:{dataset_id}") as meta:
        corpus = builder.build(dataset_id, seed)
        meta["n_utterances"] = len(corpus.utterances)
        meta["n_speakers"] = len(corpus.speakers)

    train_spk, test_spk = split_speakers(corpus.speaker_ids)
    labels = {u.utterance_id: u.speaker_id for u in corpus.utterances}

    ids = systems if systems is not None else list(config.bench.systems)
    recipes = [get_recipe(s) for s in ids]

    with timer.stage(f"frontend:{dataset_id}") as meta:
        frontend = Frontend(
            FrontendConfig(
                sample_rate=config.audio.sample_rate,
                frame_ms=config.audio.frame_ms,
                hop_ms=config.audio.hop_ms,
                n_fft=config.audio.n_fft,
                n_mels=config.audio.n_mels,
                fmin=config.audio.fmin,
                fmax=config.audio.fmax,
                n_mfcc=config.audio.n_mfcc,
                keep_c0=config.audio.keep_c0,
                lifter=config.audio.lifter,
                preemph=config.audio.preemph,
                n_deltas=config.audio.n_deltas,
                delta_width=config.audio.delta_width,
                cmvn=config.audio.cmvn,
                backend=recipes[0].backend if recipes else "auto",
            )
        )
        cache = FeatureCache(os.path.join(config.data.cache_dir, "feats"), enabled=config.data.use_cache)
        features = featurize_corpus(corpus, frontend, cache=cache)
        meta["n_utterances"] = len(features)
        meta["cache"] = cache.stats()

    with timer.stage(f"train:{dataset_id}") as meta:
        bundle = train_bundle(
            config, features, labels, train_spk, test_spk, seed=seed, frontend=frontend
        )
        meta.update({k: round(v, 4) for k, v in bundle.timings.items()})

    trials = build_trials(
        corpus,
        n_trials=int(config.bench.n_trials),
        genuine_ratio=float(config.bench.n_genuine_ratio),
        test_speakers=test_spk,
        seed=seed,
    )

    ctx = RunContext(
        config=config,
        frontend=frontend,
        features=features,
        labels=labels,
        trials=trials,
        bundle=bundle,
        dataset_id=dataset_id,
        seed=seed,
        center_supervector=bool(config.ivector.center_supervector),
    )

    results: list[dict[str, Any]] = []
    for recipe in recipes:
        with timer.stage(f"score:{dataset_id}:{recipe.system_id}") as meta:
            scores, y = _score_system(ctx, recipe)
            metrics = compute_all(scores, y, p_target=float(config.bench.minDCF_p_target))
            meta["n_trials"] = int(scores.shape[0])
        results.append(
            {
                "system_id": recipe.system_id,
                # Raw scores are kept so failure mining can pick the extremes
                # without re-running the scorer for every system.
                "scores": [float(v) for v in scores],
                "dataset_id": dataset_id,
                "seed": seed,
                "backend": frontend.backend_info.resolved,
                "mel_fingerprint": frontend.fingerprint(),
                **{k: v for k, v in metrics.items()},
            }
        )

    return {
        # The live context is attached so the evaluator can mine failure cases
        # by re-scoring, without repeating corpus synthesis or training.
        "_ctx": ctx,
        "dataset_id": dataset_id,
        "seed": seed,
        "dataset": spec.to_dict(),
        "n_train_speakers": len(train_spk),
        "n_test_speakers": len(test_spk),
        "n_train_frames": bundle.n_train_frames,
        "corpus_hash": corpus.content_hash,
        "mel_fingerprint": frontend.fingerprint(),
        "backend": frontend.backend_info.resolved,
        "bundle": bundle.to_dict(),
        "fusion": ctx.extra.get("fusion", {}),
        "results": results,
    }
