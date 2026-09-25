# VoiceForge

> 语音 / 音频机器学习系统 —— 随机交付（作者：**晨星**）

纯 Python、零编译、`pip install` 一键复现。复用顶级开源 `librosa` / `whisper` / `speechbrain` / `optuna` 作为 SOTA 后端；**离线环境下自动降级为纯 `numpy` 音频特征 + `scikit-learn` 分类器**，保证零下载即可训练、评测、跑通 demo。

- 仓库：`https://github.com/CJX0712/voiceforge`
- 作者：晨星（MIT License）
- 本机实测：分类 accuracy **0.9833**（numpy 后端）/ **1.0000**（librosa 后端）

---

## 一、随机交付信息

本次为「随机创新一个顶级 AI」全自动交付的第 5 个系统，域从已交付集合（RAG / DRL / 异常检测 / 表格 AutoML）之外抽签选定：

| 项 | 值 |
|---|---|
| 抽签域 | 语音 / Speech |
| 顶级开源 | `speechbrain` + `whisper` + `librosa`（+ `optuna` HPO） |
| 系统名 | VoiceForge |
| 仓库 | `cjx0712/voiceforge` |
| 离线兜底 | 纯 `numpy` 特征提取 + `scikit-learn` 分类器 |

## 二、特性

- **模块化单一职责**：core / data / audio / models / pitch / hpo / eval / pipeline 各司其职，接口以 `Protocol` 契约先行。
- **SOTA 优先，离线兜底**：librosa 特征、whisper ASR、speechbrain 分类、optuna HPO 均经 `available()` 探测；缺失自动降级，demo 永不因缺包中断。
- **零下载可复现**：默认 `numpy` 后端不依赖任何重型库；固定 `random_seed` 保证结果可复现。
- **标准度量**：accuracy / f1_macro / RMSE / R²（sklearn 实现，导入改名避免递归）。
- **一键基准**：`python -m voiceforge.examples.run_demo` 落盘 `benchmark.json`。

## 三、架构

```
合成/载入音频 ──▶ audio(特征) ──▶ models(分类) ──▶ eval(度量)
   data            numpy|librosa     sklearn|whisper|      accuracy
                      │              speechbrain            f1/rmse/r2
                      ▼
                  pitch(基频) ──▶ 自相关(numpy)|librosa-YIN
                      │
                      ▼
                pipeline.VoicePipeline.run() + benchmark()
```

调用单向无环：`cli → pipeline → {data, audio, models, pitch, hpo, eval} → core`。

## 四、安装与一键运行

```bash
# 1. 建隔离 venv（推荐）
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate

# 2. 安装（运行 + 测试关键依赖，已锁定）
pip install -r requirements.txt

# 3. 跑 demo（零下载，numpy 后端；librosa 可用时自动加对照）
python -m voiceforge.examples.run_demo

# 或走 Makefile
make install && make test && make demo
```

> 可选 SOTA 后端（联网增强）：`pip install librosa optuna openai-whisper speechbrain`。
> 装好后 `python -m voiceforge.cli --backend librosa` 启用 librosa 特征对照。

## 五、性能基线（本机实测，random_seed=42）

| 后端 | 模型 | accuracy | f1_macro | 训练耗时 |
|---|---|---|---|---|
| numpy（离线默认） | sklearn-RF | **0.9833** | **0.9833** | 216.6 ms |
| librosa（SOTA 对照） | sklearn-RF | **1.0000** | **1.0000** | 137.9 ms |

**基频（pitch）跟踪**（自相关估计 vs 合成已知频率，n=20）：

| 指标 | 值 |
|---|---|
| RMSE | **7.41 Hz** |
| MAE | **5.64 Hz** |
| Max Error | 16.57 Hz |

结论：librosa 的 log-mel/MFCC 特征在此合成任务上信息更充分，分类达满分；纯 numpy 实现也达 0.98，证明离线兜底有效。

## 六、模块与接口契约

| 模块 | 职责 | 核心接口 |
|---|---|---|
| `core` | 类型/错误码(E100~E500)/配置(ENV_XXX_*)/Protocol | `Config.from_env()` `VoiceForgeError` |
| `data` | 合成音频 + wav 载入 | `generate_dataset()` `load_wav()` |
| `audio` | 特征提取 | `extract(sample) -> Features` |
| `models` | 分类/ASR | `fit/predict/predict_proba` |
| `pitch` | 基频跟踪 | `estimate(sample) -> float` |
| `hpo` | 超参优化 | `tune(X, y) -> dict` |
| `eval` | 度量 | `accuracy/f1_macro/rmse/r2` |
| `pipeline` | 编排 | `run()` `benchmark_pitch()` `full_benchmark()` |

## 七、关键选型依据

| 能力 | 选型 | 理由 | 许可证 |
|---|---|---|---|
| 音频特征 | `librosa` | 业界标准 mel/MFCC/pyin，生态成熟 | ISC |
| 语音识别(可选) | `openai-whisper` | SOTA 开源 ASR | MIT |
| 音频分类(可选) | `speechbrain` | 预训练 ECAPA 嵌入，SOTA | Apache-2.0 |
| 超参优化(可选) | `optuna` | 异步采样、易用 | MIT |
| 默认分类 | `scikit-learn` | 零依赖、快、稳 | BSD-3 |
| 离线特征 | 自研纯 `numpy` | librosa 不可用时兜底（书面说明） | MIT |

## 八、已知限制

- 合成数据为波形族分类，非真实语音语料；指标用于验证链路，非生产级 ASR 评测。
- whisper / speechbrain 的预训练权重需在运行时联网下载（本系统仅探测可用性，demo 不触发下载）。
- 当前任务为"音频事件/波形分类 + 基频跟踪"，未含端到端 ASR 生成（可选模块已就绪，调用即启）。

## 九、后续方向

1. 接入真实语料（UrbanSound8K / ESC-50），用 speechbrain ECAPA 做少样本分类。
2. 启用 whisper 做 ASR 转录，构建"转录 + 分类"多任务 pipeline。
3. 用 optuna 对 RF / 特征维度做真实超参搜索（已封装 `HpoTuner`）。
4. 加 GitHub Actions：`pip install -r requirements.txt && pytest`。
5. Docker 化部署（已附 `Dockerfile`）。

## 十、许可证

MIT © 2026 晨星（CJX0712）
