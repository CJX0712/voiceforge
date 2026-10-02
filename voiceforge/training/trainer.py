"""Training of every model component, on the training split only.

Leakage discipline is the whole point of this module.  Each ``fit_*`` function
accepts training utterances and their speaker labels, and each fitted object
exposes only a ``transform``/``score`` method for application.  There is
deliberately no API that takes test data.

The trainer is also where the *expensive* models are fitted, so it records
per-stage timings and keeps the intermediate artefacts (UBM, LDA, TV, PLDA) on
one bundle that the evaluator then reuses for all systems.  Fitting the UBM
once per (dataset, seed) and sharing it across the five systems is the single
biggest saving in the demo budget.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.config import Config
from ..core.errors import ModelError
from ..core.logging import get_logger
from ..core.seed import rng
from ..core.types import FLOAT_DTYPE
from ..sv.gmm import GMM, fit_gmm
from ..sv.ivector import NbcProjector, TotalVariabilityModel, WccnProjector, center_supervector, extract_ivector, fit_nbc, fit_wccn, llr, train_total_variability
from ..sv.map import map_adapt
from ..sv.plda import PldaModel, fit_plda
from ..sv.proj import GlobalMean, LDAProjector, expand_labels, fit_global_mean, fit_lda
from ..sv.vtln import VtlnNormaliser
from ..sv.frontend import Frontend

__all__ = ["TrainedBundle", "train_bundle", "fit_ubm", "MAPTrainer"]


log = get_logger("training.trainer")


def _stack(frames: Sequence[np.ndarray], min_frames: int = 1) -> np.ndarray:
    """Concatenate per-utterance frame matrices, dropping empty ones."""
    usable = [np.ascontiguousarray(f, dtype=FLOAT_DTYPE) for f in frames if f.shape[0] >= min_frames]
    if not usable:
        raise ModelError("no usable frames to train on", n_candidates=len(frames))
    return np.concatenate(usable, axis=0)


def fit_ubm(
    frames: Sequence[np.ndarray], config: Config, *, seed: int = 0, stream: str = "ubm"
) -> GMM:
    """Fit the universal background model on pooled training frames.

    Frames are subsampled to ``gmm.train_max_frames`` before fitting: the
    log-likelihood of an EM iteration is linear in the frame count, so the
    quality plateaus long before the memory does, and the cap is what keeps the
    demo inside its wall-clock budget.
    """
    x = _stack(frames)
    cap = int(config.gmm.train_max_frames)
    if x.shape[0] > cap:
        sel = rng(seed, f"{stream}:subsample").choice(x.shape[0], size=cap, replace=False)
        sel.sort()
        x = np.ascontiguousarray(x[sel])
    log.info("UBM: %d frames, %d gaussians, dim %d", x.shape[0], config.gmm.n_gauss, x.shape[1])
    t0 = time.perf_counter()
    model = fit_gmm(
        x,
        int(config.gmm.n_gauss),
        rng_gen=rng(seed, f"{stream}:init"),
        n_iter=int(config.gmm.n_iter),
        tol=float(config.gmm.tol),
        min_var=float(config.gmm.min_var),
        kmeans_iter=int(config.gmm.kmeans_iter),
    )
    log.info("UBM fitted in %.2fs (%d EM iters, converged=%s)", time.perf_counter() - t0, model.n_iter, model.converged)
    return model


@dataclass
class MAPTrainer:
    """Holds a UBM and produces MAP-adapted models for individual utterances."""

    ubm: GMM
    tau: float
    variant: str
    adapt_var: bool = True

    def adapt(self, frames: np.ndarray) -> GMM:
        if frames.shape[0] == 0:
            raise ModelError("cannot MAP-adapt on an utterance with no frames", n_frames=0)
        return map_adapt(self.ubm, frames, tau=self.tau, variant=self.variant, adapt_var=self.adapt_var)

    def llr(self, adapted: GMM, frames: np.ndarray) -> np.ndarray:
        return llr(adapted, self.ubm, frames)


@dataclass
class TrainedBundle:
    """Every model fitted for one (dataset, seed), shared across systems."""

    ubm: GMM
    global_mean: GlobalMean
    lda: LDAProjector | None
    tv: TotalVariabilityModel | None
    nbc: NbcProjector | None
    iv_lda: LDAProjector | None
    iv_wccn: WccnProjector | None
    plda: PldaModel | None
    vtln: VtlnNormaliser | None
    train_speakers: tuple[str, ...]
    test_speakers: tuple[str, ...]
    n_train_frames: int
    timings: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ubm": self.ubm.to_dict(),
            "global_mean": self.global_mean.to_dict(),
            "lda": self.lda.to_dict() if self.lda else None,
            "tv": self.tv.to_dict() if self.tv else None,
            "nbc": self.nbc.to_dict() if self.nbc else None,
            "iv_lda": self.iv_lda.to_dict() if self.iv_lda else None,
            "iv_wccn": self.iv_wccn.to_dict() if self.iv_wccn else None,
            "plda": self.plda.to_dict() if self.plda else None,
            "vtln": self.vtln.stats() if self.vtln else None,
            "train_speakers": len(self.train_speakers),
            "test_speakers": len(self.test_speakers),
            "n_train_frames": self.n_train_frames,
            "timings": {k: round(v, 4) for k, v in self.timings.items()},
        }


def train_bundle(
    config: Config,
    features: dict[str, np.ndarray],
    labels: dict[str, str],
    train_speakers: Sequence[str],
    test_speakers: Sequence[str],
    *,
    seed: int = 0,
    frontend: Frontend | None = None,
) -> TrainedBundle:
    """Fit every component required by any registered system.

    Parameters
    ----------
    features:
        ``utterance_id -> (n_frames, D)`` feature matrix.
    labels:
        ``utterance_id -> speaker_id``.
    train_speakers:
        Speakers whose utterances are used for fitting.  Everything downstream
        is estimated from these alone.
    """
    timings: dict[str, float] = {}
    train_set = set(train_speakers)
    train_ids = [u for u, s in labels.items() if s in train_set and features.get(u) is not None]
    if not train_ids:
        raise ModelError("no training utterances", n_speakers=len(train_speakers))

    def per_speaker_frames() -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for uid in train_ids:
            out.setdefault(labels[uid], []).append(features[uid])
        return out

    by_spk = per_speaker_frames()
    timings["n_train_utterances"] = float(len(train_ids))

    # --- global mean (weak baseline; training split only) ----------------
    t0 = time.perf_counter()
    global_mean = fit_global_mean(_stack([features[u] for u in train_ids]))
    timings["global_mean"] = time.perf_counter() - t0

    # --- VTLN -----------------------------------------------------------
    # Constructed *after* the UBM (below): the warp factor is estimated per
    # utterance against the frozen UBM, so the normaliser needs both the bank
    # and the model.  The actual searches happen during enrollment and each
    # result is cached by utterance id.
    vtln = None

    # --- LDA ------------------------------------------------------------
    lda = None
    if config.proj.lda_dim > 0:
        t0 = time.perf_counter()
        all_frames = _stack([features[u] for u in train_ids])
        # LDA needs one label per *frame*; the trainer holds one per utterance.
        frame_counts = [int(features[u].shape[0]) for u in train_ids]
        frame_labels = expand_labels(frame_counts, [labels[u] for u in train_ids])
        lda = fit_lda(
            all_frames,
            frame_labels,
            n_components=int(config.proj.lda_dim),
            shrinkage=float(config.proj.shrinkage),
            n_repeats=int(config.proj.fit_max_frames),
        )
        timings["lda"] = time.perf_counter() - t0
        log.info("LDA: %d -> %d dims, %.1f%% variance explained",
                 lda.n_features, lda.output_dim, 100.0 * float(np.sum(lda.explained_variance_ratio)))

    # --- UBM ------------------------------------------------------------
    # Fitted *after* the LDA, and in the LDA space, because that is the space
    # every GMM-based recipe actually operates in (they all project through LDA
    # before MAP adaptation).  Fitting the UBM on the raw 60-dim features and
    # then MAP-adapting the 12-dim LDA output raised a shape error; fitting it
    # in both spaces would double the cost for no benefit.
    ubm_train = (
        [lda.transform(features[u]) for u in train_ids]
        if (lda is not None and config.proj.lda_dim > 0)
        else [features[u] for u in train_ids]
    )
    t0 = time.perf_counter()
    ubm = fit_ubm(ubm_train, config, seed=seed)
    timings["ubm"] = time.perf_counter() - t0

    # --- VTLN normaliser (needs the UBM) --------------------------------
    if config.vtln.enabled and frontend is not None:
        vtln = VtlnNormaliser(
            ubm,
            frontend.bank,
            n_grid=int(config.vtln.n_grid),
            w_min=float(config.vtln.w_min),
            w_max=float(config.vtln.w_max),
            n_refine=int(config.vtln.n_iter_refine),
            enabled=True,
        )

    # --- total variability + WCCN + PLDA on the centred MAP supervector ------
    # The i-vector statistic is the **MAP-adapted mean supervector centred on the
    # UBM mean** (Kaldi's ivector-extract step), one per utterance.  This is the
    # speaker-faithful representation -- the GMM cosine system scores exactly this
    # vector and reaches EER 0.00% on the synthetic corpora -- whereas the
    # earlier per-frame LLR statistic was a PCA of the phonetic *content* scatter
    # and collapsed every speaker into the same cone.  TV then projects the
    # supervector into a compact i-vector; WCCN whitens by the within-speaker
    # scatter so PLDA sees the speaker directions cleanly.
    tv = nbc = iv_lda = iv_wccn = plda = None
    if config.ivector.enabled and lda is not None:
        map_t = MAPTrainer(ubm, float(config.map.tau), str(config.map.variant), bool(config.map.adapt_var))

        # Per-utterance centred MAP supervectors (one (1, D) row per utterance).
        t0 = time.perf_counter()
        utt_sv: list[np.ndarray] = []
        utt_spk: list[str] = []
        for spk, mats in by_spk.items():
            for mat in mats:
                proj = lda.transform(mat)
                if proj.shape[0] == 0:
                    continue
                adapted = map_t.adapt(proj)
                sv = center_supervector(adapted.means, ubm.means, length_norm=False)
                utt_sv.append(np.ascontiguousarray(sv.reshape(1, -1), dtype=FLOAT_DTYPE))
                utt_spk.append(spk)
        timings["supervector"] = time.perf_counter() - t0

        if len(utt_sv) >= 2:
            t0 = time.perf_counter()
            tv = train_total_variability(
                utt_sv,
                tv_dim=int(config.ivector.tv_dim),
                n_iter=int(config.ivector.n_iter),
                tv_reg=float(config.ivector.tv_reg),
            )
            timings["tv"] = time.perf_counter() - t0
            log.info("TV: supervector %d -> %d dims", tv.n_supervector, tv.tv_dim)

            # One i-vector per utterance (no length-norm yet; WCCN / PLDA own it).
            raw_vecs = np.ascontiguousarray(
                [
                    extract_ivector(tv, m, cms=bool(config.ivector.cms), length_norm=False)
                    for m in utt_sv
                ]
            )
            spk_arr = np.asarray(utt_spk)

            # Within-class covariance normalisation (WCCN): whiten by the
            # within-speaker scatter so the residual content/session directions no
            # longer dominate the geometry PLDA models.  Keeps full dimensionality.
            if config.ivector.nbc:
                t0 = time.perf_counter()
                iv_wccn = fit_wccn(raw_vecs, spk_arr)
                timings["iv_wccn"] = time.perf_counter() - t0
                raw_vecs = iv_wccn.apply(raw_vecs) if iv_wccn is not None else raw_vecs
                log.info("i-vector WCCN: %d dims (within-class whitening)", raw_vecs.shape[1])

            # L2 length-normalise before PLDA.
            norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            vecs_plda = np.ascontiguousarray(raw_vecs / norms)

            if config.plda.enabled:
                t0 = time.perf_counter()
                n_vec = vecs_plda.shape[0]
                max_dim = max(n_vec - 3, 1)
                dim = int(min(int(config.plda.dim), vecs_plda.shape[1], max_dim))
                stacked = np.ascontiguousarray(vecs_plda[:, :dim])
                if n_vec > dim + 2:
                    plda = fit_plda(
                        stacked,
                        n_iter=int(config.plda.n_iter),
                        tol=float(config.plda.tol),
                        var_floor=float(config.plda.var_floor),
                        slv=False,
                    )
                    timings["plda"] = time.perf_counter() - t0
                    log.info(
                        "PLDA: dim %d (requested %d, capped by %d vecs), %d vectors, %d iters",
                        plda.dim, int(config.plda.dim), n_vec, n_vec, plda.n_iter,
                    )
                else:
                    timings["plda_skipped"] = 1.0
                    log.warning(
                        "PLDA skipped: %d vectors cannot support dim %d (need > dim+2)",
                        n_vec, dim,
                    )

    return TrainedBundle(
        ubm=ubm,
        global_mean=global_mean,
        lda=lda,
        tv=tv,
        nbc=nbc,
        iv_lda=iv_lda,
        iv_wccn=iv_wccn,
        plda=plda,
        vtln=vtln,
        train_speakers=tuple(sorted(train_speakers)),
        test_speakers=tuple(sorted(test_speakers)),
        n_train_frames=int(sum(features[u].shape[0] for u in train_ids)),
        timings=timings,
    )
