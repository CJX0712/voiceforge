import numpy as np

from voiceforge.core.config import Config
from voiceforge.core.types import AudioSample
from voiceforge.pitch.autocorr import estimate_pitch_autocorr


def test_autocorr_near_true():
    c = Config()
    t = np.linspace(0, c.duration_s, c.sample_rate, endpoint=False)
    for f in (150.0, 300.0, 500.0):
        w = np.sin(2 * np.pi * f * t).astype(np.float32)
        s = AudioSample(waveform=w, sample_rate=c.sample_rate)
        est = estimate_pitch_autocorr(s.waveform, s.sample_rate)
        assert abs(est - f) < 15.0, f"est={est} true={f}"
