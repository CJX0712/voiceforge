"""HpoTuner：Optuna 可选；离线降级为网格交叉验证。"""
import numpy as np

from ..core.config import Config


class HpoTuner:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def available() -> bool:
        try:
            import optuna  # noqa: F401

            return True
        except Exception:
            return False

    def tune(self, X, y, n_trials: int = 20):
        X = np.asarray(X, dtype=np.float32)
        y = list(y)
        if self.available():
            import optuna
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score

            def obj(trial):
                ne = trial.suggest_int("n_estimators", 50, 300)
                md = trial.suggest_int("max_depth", 3, 20)
                clf = RandomForestClassifier(
                    n_estimators=ne, max_depth=md, random_state=self.cfg.random_seed, n_jobs=1
                )
                return float(cross_val_score(clf, X, y, cv=3).mean())

            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=n_trials)
            return dict(study.best_params)
        # 离线降级：网格搜索
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        best, best_score = None, -1.0
        for ne in (80, 120, 160):
            for md in (8, 12, 16):
                clf = RandomForestClassifier(
                    n_estimators=ne, max_depth=md, random_state=self.cfg.random_seed, n_jobs=1
                )
                s = float(cross_val_score(clf, X, y, cv=3).mean())
                if s > best_score:
                    best_score, best = s, {"n_estimators": ne, "max_depth": md}
        return best
