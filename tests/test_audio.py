import numpy as np

from voiceforge.core.config import Config
from voiceforge.core.types import AudioSample
from voiceforge.audio.numpy_extractor import NumpyExtractor
from voiceforge.audio.librosa_extractor import LibrosaExtractor, get_extractor


def _sample(freq=200.0):
    c = Config()
    t = np.linspace(0, c.duration_s, c.sample_rate, endpoint=False)
    w = np.sin(2 * np.pi * freq * t).astype(np.float32)
    return AudioSample(waveform=w, sample_rate=c.sample_rate, label="sine")


def test_numpy_extractor_shape():
    c = Config(n_mels=40)
    ex = NumpyExtractor(c)
    f = ex.extract(_sample())
    assert f.matrix.shape == (1, 80)  # mean + std over 40 mels
    assert f.backend == "numpy"
    assert np.all(np.isfinite(f.matrix))


def test_get_extractor_fallback():
    c = Config(feature_backend="librosa")
    ex = get_extractor(c)
    assert ex.name() in ("librosa", "numpy")
