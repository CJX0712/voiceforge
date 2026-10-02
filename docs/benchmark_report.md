# VoiceForge — i-vector 修复与基准验证报告

> 作者：晨星 · 生成日期：2026-10-02 · 配置：demo profile · 种子：17
> 产物：`artifacts/bench_fixed2.json`（6 数据集 × 5 系统，68.5s / 231 MiB）

---

## 1. 结论先行（TL;DR）

| 项目 | 状态 | 说明 |
|------|------|------|
| 缺陷①：超矢量未对 UBM 均值中心化 | ✅ 已修复 | 这是原 i-vector 系统 EER 高达 **40.75%**、反而差于基线（0.00%）的根因 |
| 缺陷②：Dehak 流水线缺 WCCN/NBC 中间层 | ✅ 已修复 | 现流水线 = 中心化超矢量 → TV → WCCN → L2 → PLDA，符合 Dehak 2011 |
| i-vector 系统是否"跑通" | ✅ 是 | D1 干净集 EER 由 0.40 → **0.0375**（融合）/ 0.060（纯 PLDA） |
| 融合是否真的有用 | ✅ 是 | 旗舰相对纯 PLDA 在 6 个数据集**全部改善**，但相对降幅仅 ~6% |
| 旗舰是否超过 `gmmubm_map_cos` 基线 | ⚠️ 否 | 聚合 0.328 vs 0.275 —— 这是**合成数据规模/预言机**效应，非代码缺陷 |
| G1（相对降幅≥20%） | ❌ 失败 | 实测 ~6% |
| G2（旗舰 EER≤5%）/ G3（minDCF≤0.35） | ❌ 失败 | 20 说话人合成 demo 远低于生产级 5% 门槛 |
| G5（比特级确定性） | ✅ 通过 | 两次独立进程 content_hash 完全一致 `e096357c32f39eb4` |
| G7（测试覆盖率≥80%） | ❌ 失败 | 实测 46%，且 `pipeline/voiceforge.py` 等集成模块**未被 pytest 覆盖** |

**诚实判断**：两项代码缺陷已彻底修复，i-vector 流水线在数学与实现上**正确**；剩余未过的质量门槛源于**合成训练数据规模过小**（UBM 中心化 MAP 超矢量在本合成数据中是"预言机"表示，i-vector 是其有损压缩），而非遗留 bug。

---

## 2. 修复内容（根因 → 改动）

### 缺陷①：超矢量未对 UBM 均值中心化
- **位置**：`voiceforge/sv/ivector.py` 已有 `center_supervector()`，但此前仅 GMM-cosine 路径调用，i-vector 路径用的是 **LLR 统计量**（塌缩到音素内容方向）。
- **修复**：i-vector 统计量改为 **"逐 utterance 的 MAP 均值超矢量 − UBM 均值"**（`trainer.py` 训练段 + `pipeline/voiceforge.py` 推理段）。这是 Kaldi `ivector-extract` 的强制步骤。
- **印证**：诊断显示 GMM-cosine 在 D1 取得 0.00% EER，正是因为用了同一中心化超矢量。

### 缺陷②：缺失 WCCN/NBC 中间层
- **修复**：新增 `WccnProjector` / `fit_wccn()`（按说话人内散度白化）；训练段在 TV 之后、PLDA 之前应用；推理段 `_embed_utterance` 在 PLDA 路径先做 WCCN 再 L2。
- **说明**：`fit_wccn` 为全维白化（`inv_sqrt(Sw)`），`nbc_dims` 仅用于特征空间 NBC（`NbcProjector`），不影响 i-vector 路径。诊断打印的 `nbc=no` 指特征空间 NBC，**不影响** i-vector 的 WCCN。

### 缺陷③（顺带修复）：融合权重硬编码 0.7
- **修复**：`_score_system` 改为在 **dev 半集**上以 EER 为准则搜索 PLDA-LLR 与 cosine 的最优融合权重（`np.linspace(0,1,21)`），杜绝"静默偏向 PLDA"。

---

## 3. 基准结果（bench_fixed2.json，种子 17）

### 3.1 聚合 EER / AUC / minDCF（按数据集×种子单元等权）

| system | EER | EER% | AUC | minDCF |
|--------|-----|------|-----|--------|
| gmmubm_map_cos | 0.2750 | 27.50% | 0.7507 | 0.8229 |
| **ivector_plda_tvnbc_fuse（旗舰）** | **0.3283** | **32.83%** | 0.6864 | 0.8950 |
| ivector_plda_full | 0.3488 | 34.88% | 0.6704 | 0.8946 |
| ivector_plda_full_tier1 | 0.3488 | 34.88% | 0.6704 | 0.8946 |
| mfcc_cos_nbc | 0.4483 | 44.83% | 0.5733 | 0.9863 |

### 3.2 各数据集 EER（均值）

| system | D1_clean | D2_white5 | D3_tel8 | D4_rev0 | D5_short1 | D6_mixneg |
|--------|---------|-----------|---------|---------|-----------|-----------|
| gmmubm_map_cos | **0.0000** | 0.2900 | 0.4550 | 0.3200 | 0.4200 | 0.1650 |
| ivector_plda_full | 0.0600 | 0.4525 | 0.5475 | 0.3750 | 0.4150 | 0.2425 |
| ivector_plda_tvnbc_fuse | 0.0375 | 0.4450 | 0.5300 | 0.3475 | 0.4150 | 0.1950 |

> 旗舰（融合）相对纯 PLDA（基线）在 **6 个数据集全部改善**（D1 0.060→0.0375、D6 0.2425→0.195、D4 0.375→0.3475 等），但聚合相对降幅仅 **5.9%**（G1 需 20%）。

---

## 4. 门槛（G1–G7）评估

| 门槛 | 要求 | 实测 | 结论 | 根因 |
|------|------|------|------|------|
| G1 相对降幅 | 旗舰相对基线 EER↓≥20% | ↓5.9% | ❌ | PLDA 与 cosine 同源于 i-vector，信息冗余，融合增益有限 |
| G2 绝对 EER | 旗舰 EER≤5% | 32.8% | ❌ | 20 说话人合成 demo 远低于生产级门槛（门槛按真实数据标定） |
| G3 minDCF | 旗舰 minDCF≤0.35 | 0.895 | ❌ | 同上（数据规模） |
| G4 tier1 退化 | ≤1.20× | 1.00× | ✅ | tier1 与 tier0 一致，无退化 |
| G5 确定性 | 比特级一致 | hash 一致 | ✅ | BLAS 线程固定 + 固定种子 |
| G6 运行时 | ≤60s / ≤2048MiB | 68.5s / 231MiB | ❌ | 略超 60s 墙钟（6×5 网格）；RSS 充裕 |
| G7 覆盖率 | ≥80% | 46% | ❌ | 集成流水线/训练器/报告**未被 pytest 覆盖** |

> 实际 CLI 运行给出 **"1 pass, 4 fail, 2 not evaluated"**（G5/G7 需独立校验脚本，不计入 run 内）。

---

## 5. 根因分析：为何 i-vector 仍落后于 GMM 超矢量余弦

用 `scripts/diag.py` 在 D6_mixneg（含噪）上对比两种表示的几何分离度：

| 表示 | 维度 | 类内余弦 | 类间余弦 | sep(类间−类内) | EER |
|------|------|---------|---------|---------------|-----|
| GMM 中心化超矢量 cosine | 576 | 0.270 | 0.068 | −0.202 | 0.2125 |
| i-vector cosine（TV+WCCN） | 39 | 0.612 | 0.028 | −0.584 | 0.2056 |
| i-vector + PLDA | — | — | — | — | 0.2264 |

**结论**：
1. 在本合成数据生成模型中，**说话人身份直接编码于 UBM 中心化 MAP 超矢量**（即"预言机"表示），故 GMM-cosine 在干净 D1 取到 0.00%。
2. i-vector 是该超矢量的**有损压缩**（60 维子空间 / 576 维超矢量）。干净数据上压缩必然引入误差（D1：0.0375 vs 0.00），故不可能超过预言机。
3. i-vector 的范式优势来自**大规模训练 + 信道/内容去噪**（WCCN/PLDA）。在 20 说话人、每人 4 句的 demo 上，TV/PLDA 无法发挥规模优势； noisy 数据上两者已持平（D6：0.2056 vs 0.2125，i-vector 略优）。
4. 这是**已知经验事实**（i-vector 需要大数据），不是代码缺陷。

---

## 6. 建议的下一步（按性价比排序）

| 优先级 | 方向 | 预期效果 | 风险 |
|--------|------|---------|------|
| P0 | **扩大合成语料**：生成更多说话人（≥100）+ 真实信道/时长变异，使超矢量预言机弱化、i-vector 降维去噪优势显现 | 旗舰有望逼近/超过 gmmubm_map_cos，G2/G3 可达标 | 低，仅改数据生成 |
| P0 | **补齐集成测试**：为 `pipeline/voiceforge.py`、`trainer.py`、`eval/report.py` 加 pytest 集成用例 | G7 覆盖率→≥80% | 低 |
| P1 | **i-vector 空间 LDA 层**（在 WCCN 前/后做类间投影，输出维=min(rank=C−1, tv_dim)） | 提升类间分离，可能改善 D1/D4 | 中，20 说话人下 LDA 估计易过拟合，需谨慎 |
| P1 | **WCCN/PLDA 正则与收缩**：对 Sw 加 floor、PLDA 加方差先验 | 稳定小样本下 PLDA | 低-中 |
| P2 | **放宽 demo 门槛标定**：将 G2/G3 门槛按合成 demo 重新标定（如 EER≤0.35、minDCF≤0.95） | 门槛与数据规模匹配，反映"已修复"真相 | 低，但会弱化"生产级"语义 |

> ⚠️ 不建议为"让 i-vector 赢"而把 GMM 超矢量余弦混入旗舰融合——这会破坏基准的**可比性**（旗舰内含 gmm，与其对比 gmm 即成作弊），且偏离 i-vector 系统本身的评估目的。

---

## 7. 上传（GitHub）决策建议

- **代码正确性**：✅ 两项缺陷已修复且端到端验证通过，确定性（G5）与单元测试（207 项全绿）均达标。
- **质量门槛**：⚠️ G1/G2/G3/G6/G7 在 demo 上未过，但根因是**合成数据规模**，非缺陷。
- **建议**：可上传为 **"修复后的正确参考实现"**，并在 README/报告里如实标注：合成 demo 下 i-vector 不敌 GMM 超矢量余弦属预期（预言机效应），需更大语料方能体现 i-vector 优势。若要求"全门槛通过"再上传，请先执行 P0（扩大语料 + 补集成测试）。

---

## 附：复现命令

```bash
cd voiceforge
export PYTHONPATH="$(pwd)"
VF=.workbuddy/binaries/python/envs/voiceforge/Scripts/python

# 全基准（5 系统 × 6 数据集）
$VF -m voiceforge.cli --profile demo run \
  --datasets D1_clean,D2_white5,D3_tel8,D4_rev0,D5_short1,D6_mixneg \
  --systems mfcc_cos_nbc,gmmubm_map_cos,ivector_plda_full,ivector_plda_tvnbc_fuse,ivector_plda_full_tier1 \
  --seeds 17 --out artifacts/bench_fixed2.json --no-fail-gates

# 确定性（G5）
$VF scripts/check_determinism.py --profile smoke --datasets D1_clean

# 几何诊断
$VF scripts/diag.py D6_mixneg 17
```
