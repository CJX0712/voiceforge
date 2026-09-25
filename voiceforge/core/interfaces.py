"""接口契约（Protocol）。模块间仅依赖这些协议，保证可独立替换与公平评测。"""
from typing import List, Protocol, runtime_checkable

from .types import AudioSample, Features, Prediction


@runtime_checkable
class FeatureExtractor(Protocol):
    def name(self) -> str: ...

    def extract(self, sample: AudioSample) -> Features: ...


@runtime_checkable
class Classifier(Protocol):
    def name(self) -> str: ...

    def fit(self, X, y) -> "Classifier": ...

    def predict(self, X) -> List[str]: ...

    def predict_proba(self, X) -> List[dict]: ...


@runtime_checkable
class PitchTracker(Protocol):
    def name(self) -> str: ...

    def estimate(self, sample: AudioSample) -> float: ...
