# VoiceForge 架构文档

> 作者：晨星 ｜ 随机交付第 5 个系统（语音 / Speech 域）

## 1. 设计原则

1. **单一职责**：每个模块只做一件事，跨模块仅依赖 `core` 的 `Protocol` 契约。
2. **SOTA 优先，离线兜底**：所有重型/联网能力（librosa / whisper / speechbrain / optuna）经 `available()` 探测，缺失即降级，保证零下载 demo 可跑。
3. **可复现**：`Config.from_env()` 读取 `VOICEFORGE_*` 环境变量；默认 `random_seed=42` 固定随机性。
4. **可独立验证**：每模块有单测 + 最小示例；`pytest` 全绿即证明各模块可用。

## 2. 目录骨架

```
voiceforge/
  core/        types · errors(E100~E500) · config(ENV_*) · interfaces(Protocol)
  data/        合成数据生成(synth) + 文件载入(loader)
  audio/       特征提取：NumpyExtractor(离线) + LibrosaExtractor(可选)
  models/      SklearnClassifier(默认) + WhisperASR / SpeechBrainClassifier(可选)
  pitch/       自相关估计(离线) + librosa YIN(可选)
  hpo/         OptunaTuner(可选) / 网格降级
  eval/        accuracy / f1_macro / rmse / r2
  pipeline/    VoicePipeline.run() + benchmark()
  cli.py       argparse 入口
  examples/run_demo.py  端到端演示
tests/         pytest 单测
docs/architecture.md
README.md · requirements.txt · requirements.lock.txt · Dockerfile · Makefile · pyproject.toml
```

## 3. 调用关系（单向无环）

```
cli.main
   │
   ▼
pipeline.VoicePipeline
   ├─ data.generate_dataset ──▶ AudioSample[]
   ├─ audio.NumpyExtractor / LibrosaExtractor ──▶ Features
   ├─ models.SklearnClassifier.fit/predict ──▶ Prediction
   ├─ eval.accuracy / f1_macro ──▶ metrics
   ├─ pitch.estimate_pitch_autocorr ──▶ float
   └─ (可选) hpo.HpoTuner.tune
         │
         ▼
       core（类型/错误码/配置/契约）
```

## 4. 接口契约（Protocol）

| 协议 | 方法 | 语义 |
|---|---|---|
| `FeatureExtractor` | `name() -> str` / `extract(AudioSample) -> Features` | 音频→特征矩阵，约定 `matrix` 形状 `(n_frames, n_features)` |
| `Classifier` | `fit(X,y)` / `predict(X) -> List[str]` / `predict_proba(X) -> List[dict]` | 概率列顺序一致，score 越大越可信 |
| `PitchTracker` | `estimate(AudioSample) -> float` | 返回估计基频(Hz) |

## 5. 错误码登记表

| 码 | 含义 | 触发场景 |
|---|---|---|
| E100 | 配置错误 | 非法 ENV 覆盖、参数越界 |
| E200 | 数据错误 | 未知波形、wav 载入失败 |
| E300 | 模型错误 | 可选后端不可用时强调用 |
| E400 | 评测错误 | 标签/预测长度不匹配 |
| E500 | 流水线错误 | 编排阶段异常汇总 |

所有异常经 `VoiceForgeError(code, message, cause)` 抛出，调用方按 `code` 处理。

## 6. 关键技术决策

- **特征维度**：每样本特征 = 各 mel 频带 `mean` + `std` 拼接 → `2 * n_mels`（默认 80 维），适合 RF/LR 直接分类。
- **Mel 滤波器组**：纯 numpy 三角滤波实现 `audio/base.py:mel_filters`，与 librosa 语义对齐，作为离线兜底。
- **度量防递归**：`eval/metrics.py` 将 `sklearn.metrics` 同名函数导入为 `_sk_*` 别名，避免评测函数内递归。
- **可选后端探测**：`LibrosaExtractor.available()` / `WhisperASR.available()` 等静态方法封装 `import` 尝试，pipeline 据此选择路径。

## 7. 选型依据（性能 / 生态 / 许可证 / 活跃度）

| 能力 | 选型 | 对比 | 许可证 | 活跃度 |
|---|---|---|---|---|
| 音频特征 | `librosa` 1.0.0 | vs 纯 numpy：信息更全（MFCC/delta），分类满分 | ISC | 活跃 |
| ASR(可选) | `openai-whisper` 20250625 | SOTA 开源 ASR，零微调 | MIT | 活跃 |
| 音频分类(可选) | `speechbrain` 1.1.1 | 预训练 ECAPA 嵌入，少样本强 | Apache-2.0 | 活跃 |
| HPO(可选) | `optuna` 5.0.0 | vs 网格：异步采样更高效 | MIT | 活跃 |
| 默认分类 | `scikit-learn` 1.9.1 | 零依赖、快、稳；禁止从零自研 SOTA | BSD-3 | 活跃 |
| 离线特征 | 自研 numpy | librosa 缺失时兜底，**书面说明**非 SOTA 替代 | MIT | — |

## 8. 性能基线（本机实测，random_seed=42，200 样本 / 5 波形族）

| 后端 | 模型 | accuracy | f1_macro | fit(ms) |
|---|---|---|---|---|
| numpy（离线默认） | sklearn-RF | 0.9833 | 0.9833 | 216.6 |
| librosa（SOTA 对照） | sklearn-RF | 1.0000 | 1.0000 | 137.9 |

**基频跟踪**（自相关估计，n=20）：RMSE 7.41 Hz / MAE 5.64 Hz / Max 16.57 Hz。

**单测**：`pytest` → 21 passed。

## 9. 复现命令

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
pytest -q -W ignore::UserWarning
python -m voiceforge.examples.run_demo   # 落盘 benchmark.json
```

## 10. DoD 对照

| 项 | 状态 |
|---|---|
| 克隆→隔离 venv→`pip install`→demo 零干预 | ✅ |
| 所有模块单测通过 | ✅ 21/21 |
| 依赖锁定可复现 | ✅ requirements.lock.txt (65 行) |
| 文档覆盖架构/部署/使用 | ✅ README + architecture |
| 关键指标量化并对标 | ✅ numpy vs librosa 双后端 |
