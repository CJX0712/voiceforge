"""纯 numpy 音频特征基元：分帧、STFT 幅度、Mel 滤波器组、log-mel。

完全无第三方音频库依赖，作为离线兜底的核心实现。
"""

import numpy as np


def mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """构造 (n_mels, n_fft//2+1) 三角 Mel 滤波器组。"""

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    fmax = sr / 2.0
    mel_pts = np.linspace(hz2mel(0.0), hz2mel(fmax), n_mels + 2)
    hz_pts = mel2hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for m in range(1, n_mels + 1):
        f_left, f_center, f_right = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_left, f_center):
            if f_center > f_left:
                filters[m - 1, k] = (k - f_left) / (f_center - f_left)
        for k in range(f_center, f_right):
            if f_right > f_center:
                filters[m - 1, k] = (f_right - k) / (f_right - f_center)
    return filters


def framing(x: np.ndarray, sr: int, frame_ms: int, hop_ms: int):
    """分帧。返回 (n_frames, frame_len) 与 frame_len。"""
    frame_len = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if hop < 1:
        hop = 1
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    if frame_len < 1:
        raise ValueError("frame_len < 1")
    n_frames = 1 + (len(x) - frame_len) // hop
    if n_frames < 1:
        n_frames = 1
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    idx = np.clip(idx, 0, len(x) - 1)
    return x[idx], frame_len


def log_mel_features(x: np.ndarray, sr: int, frame_ms=25, hop_ms=10, n_mels=40):
    """返回 (聚合特征向量, 逐帧 log-mel)。聚合=各 mel 频带的均值与标准差拼接。"""
    frames, frame_len = framing(x, sr, frame_ms, hop_ms)
    window = np.hanning(frame_len)
    frames = frames * window
    n_fft = frame_len
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2  # 功率谱
    filters = mel_filters(sr, n_fft, n_mels)
    mel = spec @ filters.T  # (n_frames, n_mels)
    mel = np.clip(mel, 1e-10, None)
    logmel = np.log(mel)
    mean = logmel.mean(axis=0)
    std = logmel.std(axis=0)
    feat = np.concatenate([mean, std]).astype(np.float32)
    return feat, logmel.astype(np.float32)
