"""端到端演示：合成数据 -> 训练 -> 评测 -> 基频跟踪 -> 落盘 benchmark.json。

可直接 `python -m voiceforge.examples.run_demo` 运行；零下载即可跑（numpy 后端）。
"""
import os
import sys

# 允许以脚本方式直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from voiceforge.core.config import Config
from voiceforge.pipeline.pipeline import VoicePipeline


def demo():
    cfg = Config.from_env()
    pipe = VoicePipeline(cfg)
    print("=== VoiceForge 端到端演示（作者：晨星）===")
    print(f"配置: {cfg.as_dict()}")

    print("\n[1/3] 合成音频数据集 + 训练/评测（numpy 特征后端）")
    result, clf, ex = pipe.run()
    m = result["metrics"]
    print(f"  backend={result['backend']} model={result['model']} "
          f"样本数={result['n_samples']}")
    print(f"  accuracy={m['accuracy']:.4f}  f1_macro={m['f1_macro']:.4f}  "
          f"fit={m['fit_ms']:.1f}ms")

    print("\n[2/3] 基频跟踪基准（自相关估计 vs 已知频率）")
    pitch = pipe.benchmark_pitch(n=20)
    print(f"  pitch RMSE={pitch['pitch_rmse']:.3f}Hz  MAE={pitch['pitch_mae']:.3f}Hz  "
          f"max_err={pitch['pitch_max_err']:.3f}Hz")

    print("\n[3/3] 可选 librosa 对照 / 落盘")
    report = pipe.full_benchmark(out_path="benchmark.json", compare_librosa=True)
    for run in report["runs"]:
        if run.get("skipped"):
            print(f"  librosa 对照跳过: {run['reason']}")
    print("\n✅ Demo 跑通：synthesis -> feature -> train -> eval -> pitch 全链路 0 手工干预。")
    print("结果已写入 benchmark.json")
    return report


if __name__ == "__main__":
    demo()
