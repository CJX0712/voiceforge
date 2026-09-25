"""配置：支持环境变量 VOICEFORGE_* 覆盖，保证可复现（固定 random_seed）。"""
import os
from dataclasses import dataclass, fields
from typing import Any


@dataclass
class Config:
    sample_rate: int = 8000
    frame_ms: int = 25
    hop_ms: int = 10
    n_mels: int = 40
    n_mfcc: int = 13
    random_seed: int = 42
    feature_backend: str = "numpy"  # numpy | librosa
    n_classes: int = 5
    n_per_class: int = 40
    duration_s: float = 1.0
    test_size: float = 0.3
    noise_std: float = 0.02

    @classmethod
    def from_env(cls) -> "Config":
        c = cls()
        int_fields = {"SAMPLE_RATE", "FRAME_MS", "HOP_MS", "N_MELS", "N_MFCC",
                      "RANDOM_SEED", "N_CLASSES", "N_PER_CLASS"}
        float_fields = {"DURATION_S", "TEST_SIZE", "NOISE_STD"}
        for f in fields(cls):
            key = f.name.upper()
            env_key = "VOICEFORGE_" + key
            val = os.environ.get(env_key)
            if val is None:
                continue
            try:
                if key in int_fields:
                    setattr(c, f.name, int(val))
                elif key in float_fields:
                    setattr(c, f.name, float(val))
                else:
                    setattr(c, f.name, val)
            except (ValueError, TypeError):
                # 类型转换失败则保留默认值，不致命
                pass
        backend = os.environ.get("VOICEFORGE_BACKEND")
        if backend:
            c.feature_backend = backend
        return c

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
