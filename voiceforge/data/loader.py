"""音频文件载入/写出（依赖 scipy，离线可用）。"""
import numpy as np

from ..core.errors import err_data
from ..core.types import AudioSample


def load_wav(path: str, label: str = None, uid: str = None) -> AudioSample:
    try:
        from scipy.io import wavfile
    except Exception as e:  # pragma: no cover
        raise err_data(f"scipy 不可用，无法载入 wav: {e}")
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
    else:
        data = data.astype(np.float32)
    return AudioSample(waveform=data, sample_rate=int(sr), label=label, uid=uid or path)


def save_wav(sample: AudioSample, path: str) -> None:
    from scipy.io import wavfile

    data = np.clip(sample.waveform, -1.0, 1.0)
    wavfile.write(path, sample.sample_rate, (data * 32767).astype(np.int16))
