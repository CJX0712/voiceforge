import numpy as np

from voiceforge.core.config import Config
from voiceforge.models.sklearn_model import SklearnClassifier
from voiceforge.models.whisper_asr import WhisperASR
from voiceforge.models.speechbrain_model import SpeechBrainClassifier


def test_sklearn_train_predict():
    c = Config()
    rng = np.random.default_rng(0)
    X = rng.random((60, 20)).astype(np.float32)
    y = ["a"] * 30 + ["b"] * 30
    clf = SklearnClassifier(c, "rf")
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds) <= {"a", "b"}
    proba = clf.predict_proba(X)
    assert abs(sum(proba[0].values()) - 1.0) < 1e-5


def test_sklearn_lr_kind():
    c = Config()
    X = np.random.default_rng(1).random((40, 10)).astype(np.float32)
    y = ["x"] * 20 + ["y"] * 20
    clf = SklearnClassifier(c, "lr")
    clf.fit(X, y)
    assert len(clf.predict(X)) == 40


def test_optional_backends_availability_flag():
    assert isinstance(WhisperASR.available(), bool)
    assert isinstance(SpeechBrainClassifier.available(), bool)
