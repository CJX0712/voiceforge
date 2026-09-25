import numpy as np
import os
import tempfile

from voiceforge.core.config import Config
from voiceforge.data.synth import generate_sample, generate_dataset, CLASS_WAVEFORMS
from voiceforge.data.loader import load_wav, save_wav


def test_generate_sample_shape():
    c = Config()
    s = generate_sample(c, "sine", 200.0, uid="t")
    assert s.waveform.shape[0] == int(c.duration_s * c.sample_rate)
    assert s.label == "sine"
    assert np.all(np.isfinite(s.waveform))
    assert s.waveform.dtype == np.float32


def test_unknown_waveform_raises():
    c = Config()
    try:
        generate_sample(c, "nope", 200.0)
        assert False, "expected err_data"
    except Exception as e:
        assert "E200" in str(e)


def test_generate_dataset_labels():
    c = Config(n_classes=3, n_per_class=5)
    ds = generate_dataset(c)
    assert len(ds) == 15
    assert len({s.label for s in ds}) == 3


def test_wav_roundtrip():
    c = Config()
    s = generate_sample(c, "sine", 220.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.wav")
        save_wav(s, p)
        s2 = load_wav(p, label="sine")
        assert s2.sample_rate == s.sample_rate
        assert np.abs(s2.waveform).max() <= 1.0 + 1e-6
