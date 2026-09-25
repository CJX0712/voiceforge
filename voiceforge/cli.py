"""CLI 入口：argparse 包装 VoicePipeline。"""
import argparse
import json
import sys

from .core.config import Config
from .pipeline.pipeline import VoicePipeline


def main(argv=None):
    p = argparse.ArgumentParser(prog="voiceforge", description="VoiceForge 语音/音频 ML 系统")
    p.add_argument("--backend", choices=["numpy", "librosa"], default=None,
                   help="特征后端；默认 numpy（离线）")
    p.add_argument("--seed", type=int, default=None, help="随机种子")
    p.add_argument("--n-classes", type=int, default=None)
    p.add_argument("--n-per-class", type=int, default=None)
    p.add_argument("--out", default="benchmark.json", help="基准结果输出路径")
    p.add_argument("--pitch-n", type=int, default=20, help="基频样本数")
    p.add_argument("--no-compare", action="store_true", help="不做 librosa 对照")
    args = p.parse_args(argv)

    cfg = Config.from_env()
    if args.backend:
        cfg.feature_backend = args.backend
    if args.seed is not None:
        cfg.random_seed = args.seed
    if args.n_classes is not None:
        cfg.n_classes = args.n_classes
    if args.n_per_class is not None:
        cfg.n_per_class = args.n_per_class

    pipe = VoicePipeline(cfg)
    report = pipe.full_benchmark(out_path=args.out, compare_librosa=not args.no_compare)

    # 打印摘要表
    print("\n=== VoiceForge 基准 ===")
    for run in report["runs"]:
        if run.get("skipped"):
            print(f"  [{run['tag']}] 跳过: {run['reason']}")
            continue
        m = run["metrics"]
        print(f"  [{run['tag']}] backend={run['backend']:8s} model={run['model']:12s} "
              f"acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} n={run['n_samples']}")
    pk = report["pitch"]
    print(f"  [pitch] rmse={pk['pitch_rmse']:.3f}Hz mae={pk['pitch_mae']:.3f}Hz n={pk['n']}")
    print(f"\n结果已写入: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
