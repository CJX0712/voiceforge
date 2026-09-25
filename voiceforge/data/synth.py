"""合成音频生成：确定性（固定 seed）可复现，零下载。

每条样本由不同波形族(波形类型=类别标签) + 随机基频 + 高斯噪声构成，
用于验证"特征提取 + 分类器"链路在离线环境下即可训练与评测。
"""
import numpy as np

from ..core.config import Config
from ..core.errors import err_data
from ..core.types import AudioSample

# 波形族即分类任务的类别
CLASS_WAVEFORMS = ["sine", "square", "chirp", "noisy_sine", "two_tone"]


def generate_sample(cfg: Config, waveform: str, base_freq: float, uid: str = None) -> AudioSample:
    sr = int(cfg.sample_rate)
    n = int(cfg.duration_s * sr)
    t = np.linspace(0.0, cfg.duration_s, n, endpoint=False)
    if waveform == "sine":
        x = np.sin(2 * np.pi * base_freq * t)
    elif waveform == "square":
        x = np.sign(np.sin(2 * np.pi * base_freq * t))
    elif waveform == "chirp":
        f1, f2 = base_freq, base_freq * 2.5
        x = np.sin(2 * np.pi * (f1 * t + (f2 - f1) / (2 * cfg.duration_s) * t ** 2))
    elif waveform == "noisy_sine":
        x = np.sin(2 * np.pi * base_freq * t) + 0.5 * np.sin(2 * np.pi * base_freq * 3 * t)
    elif waveform == "two_tone":
        x = 0.6 * np.sin(2 * np.pi * base_freq * t) + 0.4 * np.sin(2 * np.pi * (base_freq * 1.5) * t)
    else:
        raise err_data(f"unknown waveform: {waveform}")
    x = x + cfg.noise_std * np.random.randn(n)
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak
    return AudioSample(waveform=x.astype(np.float32), sample_rate=sr, label=waveform, uid=uid)


def generate_dataset(cfg: Config, rng=None):
    rng = rng or np.random.default_rng(cfg.random_seed)
    classes = CLASS_WAVEFORMS[: max(1, cfg.n_classes)]
    samples = []
    for w in classes:
        for i in range(cfg.n_per_class):
            base = float(rng.integers(120, 600))  # 基频 120~600 Hz
            uid = f"{w}_{i:03d}"
            samples.append(generate_sample(cfg, w, base, uid=uid))
    return samples
