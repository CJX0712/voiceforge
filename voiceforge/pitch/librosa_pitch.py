"""librosa YIN 基频估计（可选 SOTA 后端）。"""
import numpy as np

from ..core.errors import err_model


def estimate_pitch_librosa(x, sr: int, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    try:
        import librosa
    except Exception as e:
        raise err_model("librosa 不可用", cause=e)
    y = np.asarray(x, dtype=np.float64)
    f0, _voiced, _probs = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr)
    f0 = f0[~np.isnan(f0)]
    return float(np.median(f0)) if len(f0) else 0.0
