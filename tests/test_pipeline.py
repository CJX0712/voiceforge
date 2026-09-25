from voiceforge.core.config import Config
from voiceforge.pipeline.pipeline import VoicePipeline


def test_run_basic():
    c = Config(n_classes=3, n_per_class=20)
    pipe = VoicePipeline(c)
    res, clf, ex = pipe.run()
    assert res["backend"] in ("numpy", "librosa")
    assert 0.0 <= res["metrics"]["accuracy"] <= 1.0
    assert 0.0 <= res["metrics"]["f1_macro"] <= 1.0
    assert res["n_samples"] == 60


def test_benchmark_pitch():
    c = Config()
    pipe = VoicePipeline(c)
    pk = pipe.benchmark_pitch(n=10)
    assert pk["pitch_rmse"] >= 0.0
    assert pk["n"] == 10
    assert len(pk["detail"]) == 10


def test_full_benchmark_writes(tmp_path):
    c = Config(n_classes=3, n_per_class=10)
    pipe = VoicePipeline(c)
    out = tmp_path / "bench.json"
    rep = pipe.full_benchmark(out_path=str(out), compare_librosa=False)
    assert out.exists()
    assert "runs" in rep
    assert "pitch" in rep
