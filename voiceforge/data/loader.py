"""Trial construction and the feature-matrix cache.

Trial construction is where a speaker-verification benchmark most easily
cheats itself, so the rules are explicit:

* a genuine trial enrolls and tests on **different utterances of the same
  speaker**, both drawn from the *test* speaker half;
* an impostor trial uses two **different** speakers, both from the test half;
* the trial list is balanced to ``n_genuine_ratio`` and is generated from a
  named RNG substream, so the same seed yields the same trials.

The feature cache exists because a 6x3x5 benchmark re-reads the same utterances
once per system.  It is keyed by ``content_hash`` of the corpus plus the
front-end configuration, so changing a single MFCC parameter invalidates it.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.config import Config
from ..core.errors import DataError
from ..core.logging import get_logger
from ..core.seed import content_hash, rng
from ..core.types import FLOAT_DTYPE, Corpus, Trial, Utterance

__all__ = [
    "build_trials",
    "FeatureCache",
    "featurize_corpus",
    "corpus_feature_hash",
    "GroupedUtterances",
    "group_by_speaker",
]

log = get_logger("data.loader")


@dataclass(frozen=True)
class GroupedUtterances:
    """Utterances grouped by speaker, preserving deterministic order."""

    by_speaker: Mapping[str, tuple[Utterance, ...]]
    order: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.order)

    @property
    def speaker_ids(self) -> tuple[str, ...]:
        return self.order

    def get(self, speaker_id: str) -> tuple[Utterance, ...]:
        return self.by_speaker.get(speaker_id, ())

    def all_utterances(self) -> tuple[Utterance, ...]:
        out: list[Utterance] = []
        for sid in self.order:
            out.extend(self.by_speaker[sid])
        return tuple(out)


def group_by_speaker(utterances: Sequence[Utterance]) -> GroupedUtterances:
    """Group utterances by speaker with a stable (sorted) speaker order."""
    buckets: dict[str, list[Utterance]] = {}
    for utt in utterances:
        buckets.setdefault(utt.speaker_id, []).append(utt)
    order = tuple(sorted(buckets))
    return GroupedUtterances(by_speaker={k: tuple(v) for k, v in buckets.items()}, order=order)


def build_trials(
    corpus: Corpus,
    *,
    n_trials: int,
    genuine_ratio: float = 0.5,
    test_speakers: Sequence[str] | None = None,
    seed: int | None = None,
) -> tuple[Trial, ...]:
    """Build a balanced genuine/impostor trial list for one (dataset, seed).

    Parameters
    ----------
    test_speakers:
        Restrict trials to this speaker subset (the held-out half).  ``None``
        uses every speaker, which is only appropriate for smoke tests.
    n_trials:
        Total number of trials.
    genuine_ratio:
        Fraction of trials labelled genuine.
    """
    if n_trials <= 0:
        raise DataError("n_trials must be positive", n_trials=n_trials)
    if not 0.0 < genuine_ratio < 1.0:
        raise DataError("genuine_ratio must be in (0, 1)", genuine_ratio=genuine_ratio)

    grouped = group_by_speaker(corpus.utterances)
    speakers = tuple(test_speakers) if test_speakers is not None else grouped.order
    speakers = tuple(s for s in speakers if grouped.get(s))
    if len(speakers) < 2:
        raise DataError(
            "need at least two speakers with utterances to build trials",
            n_speakers=len(speakers),
            dataset_id=corpus.dataset_id,
            remedy="increase data.n_speakers or the number of test speakers",
        )

    base_seed = corpus.seed if seed is None else seed
    r = rng(base_seed, f"trials:{corpus.dataset_id}")

    # A speaker needs >= 2 utterances to form a genuine (different-utt) trial.
    usable = tuple(s for s in speakers if len(grouped.get(s)) >= 2)
    if len(usable) < 2:
        raise DataError(
            "need at least two speakers with >= 2 utterances for genuine trials",
            n_usable=len(usable),
            n_utts_per_speaker=len(grouped.get(speakers[0])) if speakers else 0,
        )

    n_genuine = int(round(n_trials * genuine_ratio))
    n_impostor = n_trials - n_genuine

    trials: list[Trial] = []

    # --- genuine ---------------------------------------------------------
    for i in range(n_genuine):
        spk = usable[int(r.integers(0, len(usable)))]
        utts = grouped.get(spk)
        a, b = r.choice(len(utts), size=2, replace=False)
        trials.append(
            Trial(
                trial_id=len(trials),
                enroll_id=utts[int(a)].utterance_id,
                test_id=utts[int(b)].utterance_id,
                enroll_speaker=spk,
                test_speaker=spk,
                label=True,
            )
        )

    # --- impostor --------------------------------------------------------
    for _ in range(n_impostor):
        # Sample a speaker pair uniformly, then re-pick if they collide.
        for _ in range(8):
            ia, ib = r.choice(len(usable), size=2, replace=False)
            if ia != ib:
                break
        else:  # pragma: no cover - only if len(usable) == 1, excluded above
            ia, ib = 0, 1
        s_a, s_b = usable[int(ia)], usable[int(ib)]
        ea = grouped.get(s_a)[int(r.integers(0, len(grouped.get(s_a))))]
        tb = grouped.get(s_b)[int(r.integers(0, len(grouped.get(s_b))))]
        trials.append(
            Trial(
                trial_id=len(trials),
                enroll_id=ea.utterance_id,
                test_id=tb.utterance_id,
                enroll_speaker=s_a,
                test_speaker=s_b,
                label=False,
            )
        )

    # Deterministic shuffle so genuine/impostor are interleaved (a sorted list
    # would make any accidental "first half" logic look plausible).
    perm = r.permutation(len(trials))
    return tuple(
        Trial(
            trial_id=i,
            enroll_id=trials[int(p)].enroll_id,
            test_id=trials[int(p)].test_id,
            enroll_speaker=trials[int(p)].enroll_speaker,
            test_speaker=trials[int(p)].test_speaker,
            label=trials[int(p)].label,
        )
        for i, p in enumerate(perm)
    )


# --------------------------------------------------------------------------
# feature cache
# --------------------------------------------------------------------------
def corpus_feature_hash(corpus: Corpus, frontend_cfg: Mapping[str, Any]) -> str:
    """Hash of (corpus identity + front-end config) for feature-cache keying."""
    return content_hash(
        {"corpus": corpus.content_hash, "dataset_id": corpus.dataset_id, "seed": corpus.seed, "frontend": dict(frontend_cfg)},
        size=16,
    )


class FeatureCache:
    """Disk-backed cache of ``utterance_id -> feature matrix``.

    Stores one ``.pkl`` per (dataset, seed, frontend-hash) holding a dict of
    float64 arrays.  Pickle is used rather than ``.npy`` because the mapping is
    heterogeneous; the file is written atomically (temp + rename) so a crashed
    run cannot leave a half-written cache that later reads as valid.
    """

    def __init__(self, root: str, *, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        self._mem: dict[str, dict[str, np.ndarray]] = {}
        self.hits = 0
        self.misses = 0

    def path_for(self, dataset_id: str, seed: int, fhash: str) -> str:
        return os.path.join(self.root, f"{dataset_id}__seed{seed}__{fhash}.pkl")

    def load(self, dataset_id: str, seed: int, fhash: str) -> dict[str, np.ndarray] | None:
        """Return the cached feature dict, or ``None`` on miss."""
        key = f"{dataset_id}:{seed}:{fhash}"
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        if not self.enabled:
            self.misses += 1
            return None
        path = self.path_for(dataset_id, seed, fhash)
        if not os.path.isfile(path):
            self.misses += 1
            return None
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)  # noqa: S301 - cache written by this process only
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError) as exc:
            log.debug("feature cache unreadable (%s); recomputing", exc)
            self.misses += 1
            return None
        self._mem[key] = data
        self.hits += 1
        return data

    def store(self, dataset_id: str, seed: int, fhash: str, data: Mapping[str, np.ndarray]) -> None:
        """Write the feature dict to disk (best effort)."""
        key = f"{dataset_id}:{seed}:{fhash}"
        self._mem[key] = dict(data)
        if not self.enabled:
            return
        path = self.path_for(dataset_id, seed, fhash)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump(dict(data), fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except (OSError, pickle.PicklingError) as exc:  # pragma: no cover
            log.debug("feature cache write failed (%s)", exc)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._mem)}


def featurize_corpus(
    corpus: Corpus,
    frontend: Any,
    *,
    cache: FeatureCache | None = None,
    fhash: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute (or load) the feature matrix for every utterance in a corpus.

    Utterances too short to yield a single frame are mapped to an empty
    ``(0, D)`` array rather than dropped, so the id space stays aligned with
    the corpus and a downstream lookup can never silently miss.
    """
    key = fhash or corpus_feature_hash(corpus, frontend.config.to_dict())
    if cache is not None:
        cached = cache.load(corpus.dataset_id, corpus.seed, key)
        if cached is not None and len(cached) == len(corpus.utterances):
            return cached

    out: dict[str, np.ndarray] = {}
    empty = 0
    for utt in corpus.utterances:
        feats = frontend.process(utt.audio.samples.data, utt.audio.sample_rate)
        if feats.shape[0] == 0:
            empty += 1
            feats = np.zeros((0, frontend.n_features), dtype=FLOAT_DTYPE)
        out[utt.utterance_id] = np.ascontiguousarray(feats, dtype=FLOAT_DTYPE)
    if empty:
        log.debug("corpus %s: %d utterance(s) shorter than one frame", corpus.dataset_id, empty)
    if cache is not None:
        cache.store(corpus.dataset_id, corpus.seed, key, out)
    return out
