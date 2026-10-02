"""Data layer: waveform synthesis, corpus assembly, and cached loading."""

from __future__ import annotations

from .synthesis import (
    SynthesisParams,
    apply_channel,
    apply_noise,
    build_speaker_profiles,
    measure_snr_db,
    speaker_profile,
    synthesize_utterance,
)

__all__ = [
    "SynthesisParams",
    "apply_channel",
    "apply_noise",
    "build_speaker_profiles",
    "measure_snr_db",
    "speaker_profile",
    "synthesize_utterance",
]
