"""VoicePipeline：单向编排 cli -> pipeline -> {data, audio, models, eval} -> core。

全链路：合成/载入音频 -> 特征提取 -> 训练分类器 -> 评测 -> 基频跟踪。
所有可选 SOTA 后端经 available() 探测，缺失则降级，保证零下载 demo 可跑。
"""
import json
import time

import numpy as np

from ..core.config import Config
from ..core.errors import err_pipeline
from ..core.types import AudioSample
from ..data.synth import generate_dataset, generate_sample
from ..eval.metrics import accuracy, f1_macro, rmse
from ..models.sklearn_model import SklearnClassifier
from ..pitch.autocorr import estimate_pitch_autocorr


class VoicePipeline:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config.from_env()

    # ---- 数据 ----
    def prepare_data(self, samples=None):
        if samples is None:
            return generate_dataset(self.cfg)
        return samples

    # ---- 特征 ----
    def extract_features(self, samples, extractor):
        feats = [extractor.extract(s) for s in samples]
        X = np.vstack([f.matrix for f in feats])
        y = [s.label for s in samples]
        return X, y

    # ---- 训练 + 评测 ----
    def train_eval(self, X, y, classifier=None):
        from sklearn.model_selection import train_test_split

        if classifier is None:
            classifier = SklearnClassifier(self.cfg, "rf")
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=self.cfg.test_size, random_state=self.cfg.random_seed, stratify=y
        )
        t0 = time.time()
        classifier.fit(Xtr, ytr)
        fit_ms = (time.time() - t0) * 1000.0
        yp = classifier.predict(Xte)
        return {
            "accuracy": accuracy(yte, yp),
            "f1_macro": f1_macro(yte, yp),
            "n_train": int(len(ytr)),
            "n_test": int(len(yte)),
            "fit_ms": float(fit_ms),
        }, classifier

    # ---- 主流程 ----
    def run(self, samples=None, extractor=None, classifier=None):
        from ..audio.numpy_extractor import NumpyExtractor
        from ..audio.librosa_extractor import LibrosaExtractor

        samples = self.prepare_data(samples)
        if extractor is None:
            if self.cfg.feature_backend == "librosa" and LibrosaExtractor.available():
                extractor = LibrosaExtractor(self.cfg)
            else:
                extractor = NumpyExtractor(self.cfg)
        X, y = self.extract_features(samples, extractor)
        metrics, clf = self.train_eval(X, y, classifier)
        result = {
            "dataset": "synth-audio",
            "n_samples": len(samples),
            "backend": extractor.name(),
            "model": clf.name(),
            "metrics": metrics,
        }
        return result, clf, extractor

    # ---- 基频跟踪基准 ----
    def benchmark_pitch(self, n: int = 20):
        rng = np.random.default_rng(self.cfg.random_seed)
        errs = []
        detail = []
        for i in range(n):
            f = float(rng.integers(120, 600))
            s = generate_sample(self.cfg, "sine", f, uid=f"p{i}")
            est = estimate_pitch_autocorr(s.waveform, s.sample_rate)
            errs.append(abs(est - f))
            detail.append({"true_hz": round(f, 2), "est_hz": round(est, 2)})
        errs = np.array(errs, dtype=np.float64)
        return {
            "pitch_rmse": float(np.sqrt((errs ** 2).mean())),
            "pitch_mae": float(errs.mean()),
            "pitch_max_err": float(errs.max()),
            "n": n,
            "detail": detail,
        }

    # ---- 全量基准（含可选 librosa 对照） ----
    def full_benchmark(self, out_path: str = None, compare_librosa: bool = True):
        report = {"system": "VoiceForge", "config": self.cfg.as_dict(), "runs": []}
        # 主跑（numpy 后端）
        main, _clf, _ex = self.run()
        report["runs"].append({"tag": "primary", **main})
        # 可选 librosa 对照
        if compare_librosa:
            from ..audio.librosa_extractor import LibrosaExtractor

            if LibrosaExtractor.available():
                self.cfg.feature_backend = "librosa"
                try:
                    r2run, _, _ = self.run()
                    report["runs"].append({"tag": "librosa-backend", **r2run})
                finally:
                    self.cfg.feature_backend = "numpy"
            else:
                report["runs"].append(
                    {"tag": "librosa-backend", "skipped": True, "reason": "librosa 不可用（离线降级）"}
                )
        # 基频
        pitch = self.benchmark_pitch()
        report["pitch"] = pitch
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        return report
