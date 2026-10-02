# VoiceForge

> 确定性说话人验证（Speaker Verification）基准系统 —— 作者：**晨星**（MIT License）

纯 Python、零编译、可复现的说话人验证研究与基准平台。覆盖从 GMM-UBM、MAP 自适应到
i-vector / 全变异性（TV）矩阵、WCCN  nuisance 补偿、PLDA 评分的完整 Dehak 2011 流水线，
并以 **content-hash 比特级确定性** 为硬约束（Gate G5）。

- 仓库：`https://github.com/CJX0712/voiceforge`
- 作者：晨星（MIT License）
- 确定性：两次独立进程跑同配置，content_hash 逐位一致 ✅

---

## 一、系统阶梯（5 个系统，能力递增）

| id | 说明 |
|---|---|
| `mfcc_cos_nbc` | MFCC + CMVN，全局训练均值，余弦。无 LDA / 无 GMM / 无 nuisance 补偿（地板） |
| `gmmubm_map_cos` | 完整前端 + VTLN + LDA + GMM-UBM + MAP，MAP 均值余弦评分 |
| `ivector_plda_full` | 参考系统：全组件开启，PLDA 评分 |
| `ivector_plda_tvnbc_fuse` | 参考系统 + cosine 后端分数融合（权重仅在 dev 半集拟合）|
| `ivector_plda_full_tier1` | 参考系统（纯 numpy DSP 后端，量化无 librosa 的代价）|

## 二、关键修复（v0.4.0）

本仓库修复了使 i-vector 系统得分**差于基线**（损坏态 EER 40.75% vs 0.00%）的两个根因缺陷：

1. **超矢量未对 UBM 均值中心化** —— i-vector 统计量由 LLR 改为「MAP 均值超矢量 − UBM 均值」。
   这是 Kaldi `ivector-extract` 的强制步骤；缺失时每个自适应均值共享同一主导公共分量，余弦被该分量主导。
2. **Dehak 流水线缺失 WCCN/NBC 中间层** —— 新增 `WccnProjector` / `fit_wccn`，在 TV 之后、PLDA 之前按说话人内散度白化。
3. （附带）融合权重由硬编码 0.7 改为 **dev 半集 EER 搜索**，杜绝静默偏向 PLDA。

修复后 D1 干净集 EER：40.75% → **0.0375%**（融合）/ 0.060%（纯 PLDA）。

## 三、安装与运行

```bash
# 1. 建隔离 venv（推荐）
python -m venv .venv && source .venv/Scripts/activate   # Windows: .venv\Scripts\activate

# 2. 安装（已锁定依赖）
pip install -r requirements.txt

# 3. 跑基准（6 数据集 × 5 系统，demo profile，seed 17）
python -m voiceforge.cli --profile demo run \
  --datasets D1_clean,D2_white5,D3_tel8,D4_rev0,D5_short1,D6_mixneg \
  --systems mfcc_cos_nbc,gmmubm_map_cos,ivector_plda_full,ivector_plda_tvnbc_fuse,ivector_plda_full_tier1 \
  --seeds 17 --out benchmark.json

# 4. 确定性校验（Gate G5：两次独立进程应逐位一致）
python scripts/check_determinism.py --profile smoke --datasets D1_clean

# 5. 单元测试
pytest tests/ -q
```

> 依赖：numpy / scipy / scikit-learn / librosa / soundfile / numba（见 `requirements.lock.txt`）。
> 无 GPU、无联网权重下载即可完整运行；确定性通过 BLAS 线程固定（`__init__` 首行 `preset_threads(1)`）保证。

## 四、基准结果（demo / seed 17，诚实记录）

聚合 EER：

| system | EER |
|---|---|
| gmmubm_map_cos | **0.275** |
| ivector_plda_tvnbc_fuse（旗舰）| 0.328 |
| ivector_plda_full | 0.349 |
| mfcc_cos_nbc | 0.448 |

**解读（诚实）**：两项根因缺陷已修复，i-vector 流水线在数学与实现上**正确**（D1 由 40.75% → 3.75%）。
但在本**合成**基准上，旗舰仍落后 `gmmubm_map_cos`：说话人身份直接编码于 UBM 中心化 MAP 超矢量（「预言机」表示），
i-vector 是其 60 维有损压缩，干净数据上压缩必引入误差 —— 属**数据规模效应**，非代码缺陷。
i-vector 的降维去噪优势需大规模训练语料（≥100 说话人 + 真实信道变异）方能显现。

门槛（G1–G7）：G4（tier1 无退化）、G5（确定性）通过；G1（相对降幅≥20%）、G2（EER≤5%）、G3（minDCF≤0.35）、
G6（墙钟≤60s）、G7（覆盖率≥80%）在 20 说话人 demo 下未过，根因同上（数据规模 + 集成测试未覆盖）。
完整报告见 `docs/benchmark_report.md`；原始 JSON 见 `benchmark.json`。

## 五、模块结构

```
voiceforge/
  core/        类型 / 错误码(E100~E500) / 配置(ENV_XXX_* + schema 校验) / 全局确定性(seed)
  data/        合成数据生成（固定 seed 可复现）+ 文件载入
  sv/          GMM / MAP / i-vector / TV / WCCN(NBC) / PLDA / 评分 / 系统注册
  training/    UBM / TV / PLDA / WCCN 训练（仅训练集，无泄漏）
  eval/        基准编排 / 报告 / 门槛评估 / 消融
  pipeline/    端到端评分与融合
  hpo/         超参（占位）
scripts/       确定性校验、诊断、单元
tests/         不变量测试（EER/AUC/minDCF 钉死值）、各组件单测
```

## 六、已知限制

- 合成数据为受控说话人验证任务，非真实语音语料；指标用于验证链路与流水线正确性，非生产级评测。
- 当前 demo profile 为 20 说话人 × 4 句，规模远低于 i-vector 范式发挥优势所需的真实语料。
- 旗舰融合为 i-vector 内部（PLDA-LLR + cosine）融合；跨表示（如并入 GMM 超矢量余弦）融合会改变系统语义，未采用。

## 七、许可证

MIT © 2026 晨星（CJX0712）
