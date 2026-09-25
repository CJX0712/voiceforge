"""core: 类型 / 错误码 / 配置 / 接口契约（单一职责，契约先行）。"""
from .types import AudioSample, Features, Prediction, BenchmarkResult
from .errors import (
    VoiceForgeError,
    err_config,
    err_data,
    err_model,
    err_eval,
    err_pipeline,
)
from .config import Config
from .interfaces import FeatureExtractor, Classifier, PitchTracker

__all__ = [
    "AudioSample",
    "Features",
    "Prediction",
    "BenchmarkResult",
    "VoiceForgeError",
    "err_config",
    "err_data",
    "err_model",
    "err_eval",
    "err_pipeline",
    "Config",
    "FeatureExtractor",
    "Classifier",
    "PitchTracker",
]
