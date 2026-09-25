"""类型定义：音频样本、特征、预测、基准结果。"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AudioSample:
    """单条音频样本。waveform 为单声道 float32 数组。"""

    waveform: Any  # numpy.ndarray (n_samples,)
    sample_rate: int
    label: Optional[str] = None
    uid: Optional[str] = None


@dataclass
class Features:
    """提取出的特征。matrix 形状 (n_frames, n_features)；分类任务下通常为 (1, D)。"""

    matrix: Any  # numpy.ndarray
    frame_rate: Optional[float] = None
    backend: str = "numpy"


@dataclass
class Prediction:
    label: str
    proba: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class BenchmarkResult:
    dataset: str
    n_samples: int
    backend: str
    model: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_ms: float = 0.0
