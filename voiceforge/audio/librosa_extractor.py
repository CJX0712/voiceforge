"""LibrosaExtractor：可选 SOTA 后端。librosa 不可用时 available()=False，pipeline 自动降级。"""
import numpy as np

from ..core.config import Config
from ..core.errors import err_model
from ..core.types import AudioSample, Features


class LibrosaExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def available() -> bool:
        try:
            import librosa  # noqa: F401

            return True
        except Exception:
            return False

    def name(self) -> str:
        return "librosa"

    def extract(self, sample: AudioSample) -> Features:
        try:
            import librosa
        except Exception as e:
            raise err_model("librosa 不可用", cause=e)
        y = np.asarray(sample.waveform, dtype=np.float64)
        hop = int(self.cfg.hop_ms / 1000.0 * sample.sample_rate)
        mfcc = librosa.feature.mfcc(
            y=y, sr=sample.sample_rate, n_mfcc=self.cfg.n_mfcc, n_mels=self.cfg.n_mels, hop_length=hop
        )
        mel = librosa.feature.melspectrogram(
            y=y, sr=sample.sample_rate, n_mels=self.cfg.n_mels, hop_length=hop
        )
        logmel = librosa.power_to_db(mel)
        feat = np.concatenate([mfcc.mean(1), mfcc.std(1), logmel.mean(1), logmel.std(1)]).astype(
            np.float32
        )
        frame_rate = sample.sample_rate / hop if hop > 0 else sample.sample_rate
        return Features(matrix=feat.reshape(1, -1), frame_rate=frame_rate, backend="librosa")


def get_extractor(cfg: Config):
    """按配置挑选后端；librosa 请求但不可用时回退 numpy。"""
    if cfg.feature_backend == "librosa" and LibrosaExtractor.available():
        return LibrosaExtractor(cfg)
    from .numpy_extractor import NumpyExtractor

    return NumpyExtractor(cfg)
