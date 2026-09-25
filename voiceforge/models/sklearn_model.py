"""SklearnClassifier：默认离线分类器（RandomForest / LogisticRegression）。"""
import numpy as np

from ..core.config import Config
from ..core.errors import err_model


class SklearnClassifier:
    def __init__(self, cfg: Config, kind: str = "rf"):
        self.cfg = cfg
        self.kind = kind
        self.classes_ = None
        self._clf = None
        self._build()

    def _build(self):
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestClassifier

            self._clf = RandomForestClassifier(
                n_estimators=120, random_state=self.cfg.random_seed, n_jobs=1
            )
        elif self.kind == "lr":
            from sklearn.linear_model import LogisticRegression

            self._clf = LogisticRegression(max_iter=1000, random_state=self.cfg.random_seed)
        else:
            raise err_model(f"未知分类器类型: {self.kind}")

    def name(self) -> str:
        return f"sklearn-{self.kind}"

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        self._clf.fit(X, list(y))
        self.classes_ = [str(c) for c in self._clf.classes_]
        return self

    def predict(self, X):
        return [str(c) for c in self._clf.predict(np.asarray(X, dtype=np.float32))]

    def predict_proba(self, X):
        p = self._clf.predict_proba(np.asarray(X, dtype=np.float32))
        return [dict(zip(self.classes_, row.astype(float))) for row in p]
