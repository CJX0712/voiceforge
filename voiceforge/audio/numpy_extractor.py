"""NumpyExtractor：纯 numpy 离线特征提取（默认后端）。"""
import numpy as np

from ..core.config import Config
from ..core.types import AudioSample, Features
from .base import log_mel_features


class NumpyExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def name(self) -> str:
        return "numpy"

    def extract(self, sample: AudioSample) -> Features:
        feat, _logmel = log_mel_features(
            np.asarray(sample.waveform, dtype=np.float32),
            int(sample.sample_rate),
            frame_ms=self.cfg.frame_ms,
            hop_ms=self.cfg.hop_ms,
            n_mels=self.cfg.n_mels,
        )
        frame_rate = sample.sample_rate / (self.cfg.hop_ms / 1000.0)
        return Features(matrix=feat.reshape(1, -1), frame_rate=frame_rate, backend="numpy")
