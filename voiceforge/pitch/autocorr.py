"""自相关基频估计（纯 numpy 离线兜底）。"""
import numpy as np


def estimate_pitch_autocorr(x, sr: int, fmin: float = 50.0, fmax: float = 2000.0) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    seg_len = max(1, int(0.1 * sr))  # 取前 0.1s 估计基频
    seg = x[:seg_len] if len(x) > seg_len else x
    if seg.size < 2:
        return 0.0
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1 :]
    ac[:1] = 0.0
    i_min = max(1, int(sr / fmax))
    i_max = min(int(sr / fmin), len(ac) - 1)
    if i_max <= i_min:
        return 0.0
    peak = i_min + int(np.argmax(ac[i_min : i_max + 1]))
    return float(sr / peak) if peak > 0 else 0.0
