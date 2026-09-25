import numpy as np
from voiceforge.core.config import Config
from voiceforge.core.errors import VoiceForgeError, err_config
from voiceforge.core.types import AudioSample, Features, Prediction, BenchmarkResult


def test_config_defaults():
    c = Config()
    assert c.sample_rate == 8000
    assert c.feature_backend == "numpy"
    assert c.n_mels == 40


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("VOICEFORGE_SAMPLE_RATE", "16000")
    monkeypatch.setenv("VOICEFORGE_BACKEND", "librosa")
    monkeypatch.setenv("VOICEFORGE_NOISE_STD", "0.1")
    c = Config.from_env()
    assert c.sample_rate == 16000
    assert c.feature_backend == "librosa"
    assert abs(c.noise_std - 0.1) < 1e-9


def test_error_code():
    e = err_config("boom")
    assert isinstance(e, VoiceForgeError)
    assert e.code == "E100"
    assert "E100" in str(e)


def test_types():
    s = AudioSample(waveform=np.zeros(10), sample_rate=8000, label="x", uid="u")
    assert s.label == "x"
    f = Features(matrix=np.zeros((1, 8)), backend="numpy")
    assert f.matrix.shape == (1, 8)
    p = Prediction(label="a", proba={"a": 0.9}, score=0.9)
    assert p.label == "a"
    b = BenchmarkResult(dataset="d", n_samples=1, backend="numpy")
    assert b.dataset == "d"
