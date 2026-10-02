"""Diagnostic: inspect i-vector geometry vs GMM supervector geometry.

Reproduces the pipeline training for one (dataset, seed), then extracts both the
centered MAP supervector embedding and the i-vector embedding for every TEST
utterance, and reports:
  * within-speaker vs between-speaker cosine distribution (sep = inter - intra)
  * EER of: GMM cosine | i-vector cosine | i-vector + PLDA
so the real cause of the flagship's poor EER is visible rather than guessed.
"""
from __future__ import annotations

import sys
import numpy as np

from voiceforge.core.config import load_config
from voiceforge.data.corpora import CorpusBuilderImpl, get_dataset_spec, split_speakers
from voiceforge.data.loader import FeatureCache, build_trials, featurize_corpus
from voiceforge.sv.frontend import Frontend, FrontendConfig
from voiceforge.sv.ivector import center_supervector, extract_ivector
from voiceforge.sv.map import map_adapt
from voiceforge.sv.metrics import compute_all
from voiceforge.sv.plda import plda_score
from voiceforge.sv.scoring import cosine_similarity
from voiceforge.training.trainer import MAPTrainer, train_bundle

DS = sys.argv[1] if len(sys.argv) > 1 else "D1_clean"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 17

cfg = load_config("demo")

builder = CorpusBuilderImpl(cfg)
corpus = builder.build(DS, SEED)
train_spk, test_spk = split_speakers(corpus.speaker_ids)
labels = {u.utterance_id: u.speaker_id for u in corpus.utterances}

frontend = Frontend(
    FrontendConfig(
        sample_rate=cfg.audio.sample_rate, frame_ms=cfg.audio.frame_ms,
        hop_ms=cfg.audio.hop_ms, n_fft=cfg.audio.n_fft, n_mels=cfg.audio.n_mels,
        fmin=cfg.audio.fmin, fmax=cfg.audio.fmax, n_mfcc=cfg.audio.n_mfcc,
        keep_c0=cfg.audio.keep_c0, lifter=cfg.audio.lifter, preemph=cfg.audio.preemph,
        n_deltas=cfg.audio.n_deltas, delta_width=cfg.audio.delta_width,
        cmvn=cfg.audio.cmvn, backend="auto",
    )
)
cache = FeatureCache("artifacts/diag_feats", enabled=False)
features = featurize_corpus(corpus, frontend, cache=cache)
bundle = train_bundle(cfg, features, labels, train_spk, test_spk, seed=SEED, frontend=frontend)

print(f"=== bundle: tv={'yes' if bundle.tv else 'no'} nbc={'yes' if bundle.nbc else 'no'} "
      f"plda={'yes(dim=%d)' % bundle.plda.dim if bundle.plda else 'no'} lda_dim={bundle.lda.output_dim if bundle.lda else 0}")

trainer = MAPTrainer(bundle.ubm, float(cfg.map.tau), str(cfg.map.variant), bool(cfg.map.adapt_var))

# Extract embeddings for TEST utterances only (held-out speakers).
test_ids = [u for u in corpus.utterances if u.speaker_id in set(test_spk)]
gmm_emb = {}
iv_raw = {}
for u in test_ids:
    feats = features[u.utterance_id]
    proj = bundle.lda.transform(feats)
    adapted = trainer.adapt(proj)
    gmm_emb[u.utterance_id] = center_supervector(adapted.means, bundle.ubm.means, length_norm=True)
    # i-vector statistic = centred MAP supervector
    sv = center_supervector(adapted.means, bundle.ubm.means, length_norm=False).reshape(1, -1)
    raw = extract_ivector(bundle.tv, sv, cms=bool(cfg.ivector.cms), length_norm=False)
    v = raw
    if bundle.iv_wccn is not None:
        v = bundle.iv_wccn.apply(v)
    iv_raw[u.utterance_id] = v / np.linalg.norm(v) if np.linalg.norm(v) > 1e-12 else v


def stats(name, emb):
    ids = list(emb.keys())
    spk = {i: labels[i] for i in ids}
    intra, inter = [], []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            c = float(cosine_similarity(emb[ids[a]], emb[ids[b]]))
            (intra if spk[ids[a]] == spk[ids[b]] else inter).append(c)
    intra = np.array(intra); inter = np.array(inter)
    sep = float(np.mean(inter) - np.mean(intra))
    print(f"[{name}] dim={emb[ids[0]].shape[0]}  intra_cos={np.mean(intra):.3f}±{np.std(intra):.3f}"
          f"  inter_cos={np.mean(inter):.3f}±{np.std(inter):.3f}  sep={sep:+.3f}")
    return emb, spk


def eer_of(name, emb, spk):
    ids = list(emb.keys())
    # genuine = same speaker, different utterance; impostor = different speaker
    sc, lb = [], []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            sc.append(float(cosine_similarity(emb[ids[a]], emb[ids[b]])))
            lb.append(1 if spk[ids[a]] == spk[ids[b]] else 0)
    m = compute_all(np.array(sc), np.array(lb, bool))
    print(f"  EER({name}) = {m['eer']:.4f}  AUC={m['auc']:.4f}")
    return m["eer"]


print("\n=== geometry ===")
g_emb, g_spk = stats("GMM-cos", gmm_emb)
i_emb, i_spk = stats("i-vector", iv_raw)

print("\n=== cosine EER ===")
eer_of("GMM-cos", g_emb, g_spk)
eer_of("i-vector-cos", i_emb, i_spk)

if bundle.plda is not None:
    pdim = bundle.plda.dim
    def plda_emb(name, emb, spk):
        ids = list(emb.keys())
        sc, lb = [], []
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                e1 = np.ascontiguousarray(emb[ids[a]][:pdim])
                e2 = np.ascontiguousarray(emb[ids[b]][:pdim])
                sc.append(plda_score(bundle.plda, e1, e2))
                lb.append(1 if spk[ids[a]] == spk[ids[b]] else 0)
        m = compute_all(np.array(sc), np.array(lb, bool))
        print(f"  EER({name}) = {m['eer']:.4f}  AUC={m['auc']:.4f}")
    print("\n=== PLDA EER ===")
    plda_emb("i-vector-PLDA", i_emb, i_spk)
