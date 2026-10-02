"""Corpus assembly, the 6-point difficulty grid, and the on-disk cache.

The difficulty grid is the experimental design of the whole benchmark.  Each
``DatasetSpec`` fixes an SNR, a noise type, a channel and the enrollment/test
durations; the six points walk a single monotone difficulty axis from clean to
``-3 dB SNR + babble + narrowband + 2 s`` enrollment.  Expected EER bands are
declared up front and are *checked* after the run -- a dataset that lands
outside its band produces a warning rather than being quietly dropped.

Caching
-------
Regenerating a corpus costs more than everything else in the pipeline combined,
so every corpus is written to ``<cache_dir>/<dataset_id>__seed<seed>/`` together
with a ``manifest.json`` carrying a ``content_hash`` over the *generation
parameters*.  A cache hit requires that hash to match exactly; any change to
the synthesis constants invalidates it automatically.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..core.config import Config
from ..core.errors import CorpusError
from ..core.logging import get_logger
from ..core.seed import content_hash, rng
from ..core.types import FLOAT_DTYPE, AudioClip, Corpus, DatasetSpec, FrozenArray, SpeakerProfile, Utterance
from .synthesis import (
    SynthesisParams,
    apply_channel,
    apply_noise,
    build_speaker_profiles,
    measure_snr_db,
    synthesize_utterance,
)

__all__ = [
    "DATASET_GRID",
    "dataset_spec",
    "get_dataset_spec",
    "CorpusBuilderImpl",
    "build_corpus",
    "split_speakers",
]

log = get_logger("data.corpora")

#: The six benchmark points, ordered by increasing difficulty.
DATASET_GRID: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        dataset_id="D1_clean",
        snr_db=None,
        noise="none",
        channel="clean",
        enroll_sec=3.0,
        test_sec=3.0,
        expected_eer_range=(0.005, 0.02),
        description="Clean, wideband, long utterances: the accuracy ceiling.",
    ),
    DatasetSpec(
        dataset_id="D2_white5",
        snr_db=5.0,
        noise="white",
        channel="clean",
        enroll_sec=3.0,
        test_sec=3.0,
        expected_eer_range=(0.01, 0.03),
        description="Additive white noise at 5 dB SNR.",
    ),
    DatasetSpec(
        dataset_id="D3_tel8",
        snr_db=8.0,
        noise="white",
        channel="telephone",
        enroll_sec=3.0,
        test_sec=3.0,
        expected_eer_range=(0.03, 0.06),
        description="8 kHz narrowband channel: loses F1 detail and all energy above F3.",
    ),
    DatasetSpec(
        dataset_id="D4_rev0",
        snr_db=10.0,
        noise="white",
        channel="reverb",
        channel_param=0.4,
        enroll_sec=3.0,
        test_sec=3.0,
        expected_eer_range=(0.03, 0.07),
        description="Moderate reverberation (T60 = 0.4 s) at 10 dB SNR.",
    ),
    DatasetSpec(
        dataset_id="D5_short1",
        snr_db=5.0,
        noise="babble",
        channel="reverb",
        channel_param=0.3,
        enroll_sec=1.0,
        test_sec=3.0,
        expected_eer_range=(0.06, 0.12),
        description="Short (1 s) enrollment, babble interference, mild reverb.",
    ),
    DatasetSpec(
        dataset_id="D6_mixneg",
        snr_db=-3.0,
        noise="babble",
        channel="telephone",
        enroll_sec=2.0,
        test_sec=2.0,
        expected_eer_range=(0.10, 0.20),
        description="Hardest point: -3 dB babble through a narrowband channel.",
    ),
)

_BY_ID = {spec.dataset_id: spec for spec in DATASET_GRID}


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    """Look up a dataset specification by id."""
    try:
        return _BY_ID[dataset_id]
    except KeyError as exc:
        raise CorpusError(
            f"unknown dataset id {dataset_id!r}", dataset_id=dataset_id, known=sorted(_BY_ID)
        ) from exc


def dataset_spec(dataset_id: str) -> DatasetSpec:
    """Alias of :func:`get_dataset_spec`."""
    return get_dataset_spec(dataset_id)


def split_speakers(speaker_ids: Sequence[str], *, holdout: str = "test") -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split speakers into (train, test) halves.

    The split is **by speaker parity**, which is deterministic and, crucially,
    keeps the two halves disjoint so that no speaker ever appears in both the
    UBM/LDA/TV training data and the evaluation trials.  Leakage here would
    silently deflate every EER.
    """
    ordered = sorted(set(speaker_ids))
    train = tuple(s for i, s in enumerate(ordered) if i % 2 == 0)
    test = tuple(s for i, s in enumerate(ordered) if i % 2 == 1)
    if not train or not test:
        # Degenerate speaker count: fall back to a single all-train split so the
        # pipeline still runs (and the gate report flags the tiny corpus).
        return tuple(ordered), tuple(ordered)
    return train, test


# --------------------------------------------------------------------------
# corpus builder
# --------------------------------------------------------------------------
@dataclass
class _BuildStats:
    """Bookkeeping for one corpus build (surfaced in the manifest)."""

    n_utterances: int = 0
    n_speakers: int = 0
    total_seconds: float = 0.0
    measured_snr: list[float] = None  # type: ignore[assignment]
    measured_snr_finite: list[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.measured_snr is None:
            self.measured_snr = []
        if self.measured_snr_finite is None:
            self.measured_snr_finite = []


class CorpusBuilderImpl:
    """Builds and caches corpora for a (dataset, seed) pair."""

    def __init__(self, config: Config, *, n_speakers: int | None = None, n_utts: int | None = None) -> None:
        self.config = config
        self.n_speakers = int(n_speakers if n_speakers is not None else config.data.n_speakers)
        self.n_utts = int(n_utts if n_utts is not None else config.data.n_utts_per_speaker)
        self.sample_rate = config.audio.sample_rate
        self.params = SynthesisParams(sample_rate=self.sample_rate)
        self._profiles_cache: dict[int, tuple[SpeakerProfile, ...]] = {}

    # -- parameters ------------------------------------------------------
    def generation_params(self, dataset_id: str, seed: int) -> dict[str, Any]:
        """The full parameter set that determines the corpus bytes.

        Anything that can change a single sample must appear here, because this
        dict is what the cache ``content_hash`` is computed over.
        """
        spec = get_dataset_spec(dataset_id)
        return {
            "dataset_id": dataset_id,
            "seed": int(seed),
            "sample_rate": self.sample_rate,
            "n_speakers": self.n_speakers,
            "n_utts_per_speaker": self.n_utts,
            "enroll_sec": spec.enroll_sec,
            "test_sec": spec.test_sec,
            "snr_db": spec.snr_db,
            "noise": spec.noise,
            "channel": spec.channel,
            "channel_param": spec.channel_param,
            "synthesis": self.params.to_dict(),
            "schema": 2,
        }

    def corpus_hash(self, dataset_id: str, seed: int) -> str:
        return content_hash(self.generation_params(dataset_id, seed), size=16)

    # -- profiles --------------------------------------------------------
    def profiles(self, seed: int) -> tuple[SpeakerProfile, ...]:
        """Speaker profiles for this corpus (cached per ``n_speakers``)."""
        if self.n_speakers not in self._profiles_cache:
            self._profiles_cache[self.n_speakers] = build_speaker_profiles(
                self.n_speakers, seed=seed, sample_rate=self.sample_rate, params=self.params
            )
        return self._profiles_cache[self.n_speakers]

    # -- build -----------------------------------------------------------
    def build(self, dataset_id: str, seed: int, *, use_cache: bool | None = None) -> Corpus:
        """Return the corpus for ``(dataset_id, seed)``, serving from cache if valid."""
        spec = get_dataset_spec(dataset_id)
        use_cache = self.config.data.use_cache if use_cache is None else use_cache
        params = self.generation_params(dataset_id, seed)
        chash = content_hash(params, size=16)
        cache_dir = self._cache_dir(dataset_id, seed)

        if use_cache and os.path.isdir(cache_dir):
            cached = self._load_cache(cache_dir, chash, spec, seed)
            if cached is not None:
                log.debug("corpus cache hit %s seed=%d", dataset_id, seed)
                return cached

        t0 = time.perf_counter()
        corpus = self._synthesize_corpus(spec, seed, chash, params)
        if use_cache:
            self._save_cache(cache_dir, corpus, params, chash, spec)
        log.info(
            "built corpus %s seed=%d: %d utts, %d spk, %.1fs audio in %.2fs",
            dataset_id, seed, len(corpus.utterances), len(corpus.speakers),
            sum(u.duration for u in corpus.utterances), time.perf_counter() - t0,
        )
        return corpus

    def _synthesize_corpus(
        self, spec: DatasetSpec, seed: int, chash: str, params: Mapping[str, Any]
    ) -> Corpus:
        """Generate every waveform for one dataset/seed from scratch."""
        profiles = self.profiles(seed)
        sr = self.sample_rate
        r_noise = rng(seed, f"noise:{spec.dataset_id}")

        # Babble donors: one clean utterance from each of a few *other*
        # speakers, generated up front so the interference is speech-like and
        # independent of the utterance being degraded.
        pool: list[np.ndarray] = []
        if spec.noise == "babble":
            for prof in profiles[: min(8, len(profiles))]:
                w, _ = synthesize_utterance(
                    prof, spec.test_sec, seed=seed, params=self.params, stream=f"babble:{prof.speaker_id}"
                )
                pool.append(w)

        utterances: list[Utterance] = []
        snr_measured: list[float] = []
        peak_by_spk: dict[str, float] = {}

        for prof in profiles:
            for u in range(self.n_utts):
                # Utterance duration alternates between the enroll and test
                # lengths so both lengths appear in the corpus and each trial
                # pair can be assembled from the same pool.
                duration = spec.enroll_sec if u % 2 == 0 else spec.test_sec
                wave, meta = synthesize_utterance(
                    prof, duration, seed=seed, params=self.params, stream=f"utt:{prof.speaker_id}:{u}"
                )
                clean_peak = float(np.max(np.abs(wave)))

                ch_r = rng(seed, f"chan:{spec.dataset_id}:{prof.speaker_id}:{u}")
                wave, chan_meta = apply_channel(wave, spec.channel, sr, param=spec.channel_param, rng_gen=ch_r)

                nr = rng(seed, f"noise:{spec.dataset_id}:{prof.speaker_id}:{u}")
                wave, noise_meta = apply_noise(
                    wave, spec.noise, spec.snr_db, sr, rng_gen=nr, speech_pool=pool
                )
                if noise_meta.get("measured_snr_db") is not None:
                    val = float(noise_meta["measured_snr_db"])
                    if np.isfinite(val):
                        snr_measured.append(val)

                # Guard against clipping after channel + noise.
                peak = float(np.max(np.abs(wave)))
                if peak > 0.99:
                    wave = wave * (0.99 / peak)
                peak_by_spk[prof.speaker_id] = max(peak_by_spk.get(prof.speaker_id, 0.0), float(np.max(np.abs(wave))))

                utt_id = f"{spec.dataset_id}_s{seed}_{prof.speaker_id}_u{u}"
                clip = AudioClip(
                    samples=FrozenArray(np.ascontiguousarray(wave, dtype=FLOAT_DTYPE)),
                    sample_rate=sr,
                    utterance_id=utt_id,
                    speaker_id=prof.speaker_id,
                )
                utterances.append(
                    Utterance(
                        utterance_id=utt_id,
                        speaker_id=prof.speaker_id,
                        audio=clip,
                        dataset_id=spec.dataset_id,
                        seed=seed,
                        profile=prof,
                        conditions={
                            "duration_sec": duration,
                            "channel": chan_meta,
                            "noise": noise_meta,
                            "clean_peak": clean_peak,
                            "f0_measured_mean": meta["f0_measured_mean"],
                            "formants_hz": meta["formants_hz"],
                        },
                    )
                )

        manifest_extra = {
            "measured_snr_db_mean": float(np.mean(snr_measured)) if snr_measured else None,
            "measured_snr_db_std": float(np.std(snr_measured)) if snr_measured else None,
            "measured_snr_db_min": float(np.min(snr_measured)) if snr_measured else None,
            "measured_snr_db_max": float(np.max(snr_measured)) if snr_measured else None,
            "n_snr_samples": len(snr_measured),
            "total_audio_sec": round(sum(u.duration for u in utterances), 3),
            "peak_amplitude_max": round(max(peak_by_spk.values()) if peak_by_spk else 0.0, 6),
        }

        return Corpus(
            dataset_id=spec.dataset_id,
            seed=seed,
            utterances=tuple(utterances),
            speakers=tuple(profiles),
            content_hash=chash,
            params={**dict(params), "build": manifest_extra},
        )

    # -- cache -----------------------------------------------------------
    def _cache_dir(self, dataset_id: str, seed: int) -> str:
        return os.path.join(self.config.data.cache_dir, f"{dataset_id}__seed{seed}")

    def _save_cache(
        self, cache_dir: str, corpus: Corpus, params: Mapping[str, Any], chash: str, spec: DatasetSpec
    ) -> None:
        """Persist the corpus as a single float32 ``.npy`` blob plus a manifest.

        float32 halves the on-disk size while keeping ~7 decimal digits, which is
        far more than the 16 kHz audio needs; the *hash* is always computed over
        the generation parameters (float64 domain), never over the stored bytes,
        so the cache stays bit-reproducible across the storage precision.
        """
        try:
            os.makedirs(cache_dir, exist_ok=True)
            stacked = np.stack([u.audio.samples.data for u in corpus.utterances]).astype(np.float32)
            np.save(os.path.join(cache_dir, "audio.npy"), stacked, allow_pickle=False)
            manifest = {
                "content_hash": chash,
                "params": dict(params),
                "dataset": spec.to_dict(),
                "n_utterances": len(corpus.utterances),
                "sample_rate": self.sample_rate,
                "utterances": [
                    {
                        "utterance_id": u.utterance_id,
                        "speaker_id": u.speaker_id,
                        "duration_sec": u.duration,
                    }
                    for u in corpus.utterances
                ],
                "speakers": [
                    {
                        "speaker_id": p.speaker_id,
                        "f0_median": p.f0_median,
                        "formants": list(p.formants),
                        "bandwidths": list(p.bandwidths),
                        "vocal_tract_length_cm": p.vocal_tract_length_cm,
                        "fricative_pattern": [list(x) for x in p.fricative_pattern],
                        "n_syllables": p.n_syllables,
                    }
                    for p in corpus.speakers
                ],
            }
            with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
        except (OSError, ValueError) as exc:  # pragma: no cover - disk issues
            log.warning("corpus cache write failed (%s); continuing without cache", exc)

    def _load_cache(self, cache_dir: str, chash: str, spec: DatasetSpec, seed: int) -> Corpus | None:
        """Load a cached corpus, or return ``None`` if it is absent/stale."""
        manifest_path = os.path.join(cache_dir, "manifest.json")
        audio_path = os.path.join(cache_dir, "audio.npy")
        lengths_path = os.path.join(cache_dir, "lengths.npy")
        if not all(os.path.isfile(p) for p in (manifest_path, audio_path, lengths_path)):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            if manifest.get("content_hash") != chash:
                log.debug("corpus cache stale for %s seed=%d", spec.dataset_id, seed)
                return None
            # The cache stores a *flat* float32 buffer plus per-utterance
            # lengths: utterance durations differ (enroll vs test, and the
            # telephone channel resamples), so a rectangular array cannot
            # represent a corpus.
            flat = np.load(audio_path, allow_pickle=False)
            lengths_arr = np.load(lengths_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            log.debug("corpus cache unreadable (%s); rebuilding", exc)
            return None

        # `lengths` and `flat` are read above; a cache written by an older
        # version (rectangular `stacked`) is rejected here rather than being
        # sliced with the wrong indexing.
        lengths = [int(v) for v in lengths_arr]
        if int(sum(lengths)) != int(flat.shape[0]) or len(lengths) != len(manifest.get("utterances", [])):
            log.debug("corpus cache shape mismatch; rebuilding")
            return None

        profiles = tuple(
            SpeakerProfile(
                speaker_id=p["speaker_id"],
                f0_median=float(p["f0_median"]),
                f0_jitter=self.params.f0_jitter,
                formants=tuple(float(x) for x in p["formants"]),
                bandwidths=tuple(float(x) for x in p["bandwidths"]),
                vocal_tract_length_cm=float(p["vocal_tract_length_cm"]),
                fricative_pattern=tuple(
                    (float(a), float(b), float(c)) for a, b, c in p.get("fricative_pattern", [])
                ),
                n_syllables=int(p.get("n_syllables", 5)),
            )
            for p in manifest.get("speakers", [])
        )
        by_id = {p.speaker_id: p for p in profiles}

        utterances: list[Utterance] = []
        offset = 0
        for i, meta in enumerate(manifest["utterances"]):
            # Slice the flat buffer by this utterance's recorded length; the
            # lengths differ whenever enroll and test durations differ.
            n = lengths[i]
            wave = np.ascontiguousarray(flat[offset : offset + n].astype(FLOAT_DTYPE))
            offset += n
            clip = AudioClip(
                samples=FrozenArray(wave),
                sample_rate=int(manifest["sample_rate"]),
                utterance_id=meta["utterance_id"],
                speaker_id=meta["speaker_id"],
            )
            utterances.append(
                Utterance(
                    utterance_id=meta["utterance_id"],
                    speaker_id=meta["speaker_id"],
                    audio=clip,
                    dataset_id=spec.dataset_id,
                    seed=seed,
                    profile=by_id.get(meta["speaker_id"]),
                    conditions={"from_cache": True},
                )
            )

        return Corpus(
            dataset_id=spec.dataset_id,
            seed=seed,
            utterances=tuple(utterances),
            speakers=profiles,
            content_hash=chash,
            params={**manifest.get("params", {}), "from_cache": True},
        )


def build_corpus(
    config: Config,
    dataset_id: str,
    seed: int,
    *,
    n_speakers: int | None = None,
    n_utts: int | None = None,
    use_cache: bool | None = None,
) -> Corpus:
    """Functional wrapper around :class:`CorpusBuilderImpl`."""
    builder = CorpusBuilderImpl(config, n_speakers=n_speakers, n_utts=n_utts)
    return builder.build(dataset_id, seed, use_cache=use_cache)
