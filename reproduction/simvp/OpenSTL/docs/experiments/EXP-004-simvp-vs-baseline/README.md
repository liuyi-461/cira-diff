# EXP-004 — 统一 Baseline 协议与已复用的外部基线

## Identity
- 实验 ID：`EXP-004`
- 状态：`BASELINE COPIED / EVAL READY`（我们的模型尚未用该协议出数）
- 日期：2026-10-06
- 关联：`EXP-002`（小样本流水线）、`EXP-003`（单样本过拟合）、`DEC-003`（不改上游）
- 后续：**`EXP-005`（SimVP 多 seed rollout 方差，与 UNet 对照）**——
  本文件所有 rollout 结论均为**单 seed**，其稳健性由 EXP-005 判定
- 基线来源（**只读引用，未修改对方任何文件**）：
  - `/home/group1/26fall_aiclass/ly/cira-diff/` → Vanilla UNet（EXP-EVAL-001 / EXP-EVAL-003）
  - `/home/group1/26fall_aiclass/cb/.../simvp/full_train/` → SimVP 全量训练（报告 + test_metrics.json）

## 目的
把"别人的结果"固化成**本项目可引用的 baseline**，并定义一套**统一评估协议**，
使我们的 SimVP 与 UNet / SimVP(cb) 能在**同一把尺子**上比较。

---

## 统一评估协议（Baseline Protocol v1）

后续任何"我们的模型 vs baseline"的对比都必须满足以下 6 条，否则数字不可比：

| # | 项 | 规定 |
|---|---|---|
| 1 | 评估集 | `edm_GOES_ch13_test_dataset.zarr`（**1024 条，Chase 2025 官方 test split**）。不得用 train zarr 的随机子集冒充 test |
| 2 | 任务口径 | 2 帧输入 → 1 帧输出；`output[:, 0]` = **t+10min**（已由 cb 验证 `|output[:,j] − input[:,1]|` 随 j 严格单调递增） |
| 3 | 数值空间 | zarr 内已是归一化空间（mean=0, std=1）。物理量换算：`Tb(K) = zarr × STD_K + MEAN_K`；`MEAN_K=279.0699458792467`、`STD_K=19.32967519050003`（源自 `satcast/simple_code.md`，cb 与本项目已核对一致） |
| 4 | 指标口径 | **逐像素**：`MSE = 全部元素均方误差`、`RMSE = √MSE`、`MAE = 全部元素绝对误差均值`。**禁止**使用 OpenSTL `openstl/core/metrics.py` 的默认实现——它对 `(C,H,W)` 用 `.sum()`，会把 MSE/MAE 放大 65536 倍、RMSE 放大 256 倍（见下方"口径坑"） |
| 5 | 评估权重 | 必须用 **best.ckpt**。OpenSTL 的 `trainer.test()` 跑在**最终权重**上，不是 best |
| 6 | 对照基线 | 至少报 **persistence**（把输入最近帧 t 直接当作 t+10min）；它是弱基线，仅作下限参考 |

**指标清单**（对齐 ly 的 UNet 评估脚本）：MSE / MAE / RMSE / SSIM / PSNR
- SSIM：11×11 窗口、`F.avg_pool2d`、`C1=0.01²`、`C2=0.03²`
- PSNR：`10·log10(1/(MSE+1e-12))`（沿用 ly 定义；注意该式隐含 [0,1] 输入假设，
  而 zarr 是标准化空间，故 PSNR 仅作**横向同口径对比**用，不宜单独解释）

---

## 基线 A — Vanilla UNet（来源：ly，只读引用）

- 数据：同一份 `edm_GOES_ch13_test_dataset.zarr`（1024 条）
- 场景 1 单步 teacher-forced（1024 条）；场景 2 **18 步自回归 rollout**（128 条子集，逐 LT 统计）

### 单步（1024 样本）

| Checkpoint | MSE | MAE | SSIM | PSNR |
|---|---|---|---|---|
| seed=0 (EXP-014) | 0.008050 ± 0.00885 | 0.04604 ± 0.0286 | 0.8929 ± 0.0534 | 24.04 ± 5.82 |
| seed=42 (EXP-010) | 0.008092 ± 0.00899 | 0.04634 ± 0.0290 | 0.8913 ± 0.0536 | 23.96 ± 5.82 |
| seed=123 (EXP-015) | 0.008157 ± 0.00920 | 0.04627 ± 0.0293 | 0.8929 ± 0.0537 | 23.92 ± 5.87 |
| Official OpenSource | 0.008272 ± 0.00932 | 0.04517 ± 0.0299 | 0.8939 ± 0.0541 | 24.04 ± 6.05 |

> 换算到亮温：`RMSE(K) = √MSE × STD_K(19.33)` → UNet 单步 RMSE ≈ **1.74 K**

### 18 步 rollout（128 样本，逐 LT 的 MSE）

| LT | seed0 | seed42 | seed123 | Official |
|---|---|---|---|---|
| 1 | 0.006878 | 0.006881 | 0.006968 | 0.007019 |
| 2 | 0.01893 | 0.01888 | 0.01904 | 0.01917 |
| 3 | 0.03421 | 0.03438 | 0.03416 | 0.03439 |
| 4 | 0.05159 | 0.05257 | 0.05209 | 0.05181 |
| 5 | 0.07092 | 0.07348 | 0.07222 | 0.07119 |
| 6 | 0.09209 | 0.09661 | 0.09403 | 0.09231 |
| 7 | 0.11445 | 0.12193 | 0.11743 | 0.11499 |
| 8 | 0.13837 | 0.15013 | 0.14324 | 0.13911 |
| 9 | 0.16376 | 0.18187 | 0.17191 | 0.16491 |
| 10 | 0.19072 | 0.21620 | 0.20323 | 0.19137 |
| 11 | 0.21937 | 0.25311 | 0.23705 | 0.21918 |
| 12 | 0.24961 | 0.29247 | 0.27337 | 0.24819 |
| 13 | 0.28173 | 0.33451 | 0.31243 | 0.27941 |
| 14 | 0.31622 | 0.37919 | 0.35437 | 0.31418 |
| 15 | 0.35342 | 0.42538 | 0.39951 | 0.35209 |
| 16 | 0.39332 | 0.47346 | 0.44802 | 0.39374 |
| 17 | 0.43578 | 0.52340 | 0.49997 | 0.44001 |
| **18** | **0.4192** | **0.5743** | **0.4717** | **0.4909** |

LT=18 汇总（MSE / SSIM）：seed0 `0.419 / 0.241`、seed123 `0.472 / 0.194`、
Official `0.491 / 0.238`、seed42 `0.574 / 0.103`；**3-seed 均值 0.488 ± 0.066（CV 13.4%）**。

> ⚠️ 引用规则（EXP-EVAL-003 结论）：UNet rollout **对随机种子极敏感**
> （单步 CV<1.3%，LT=18 CV 13.4%）。必须报 **3-seed 均值 ± std** 或 Official，**禁止单 seed**。

---

## 基线 B — SimVP 2→1 全量训练（来源：cb，只读引用）

- 100 epoch 全量训练（35595 条），best.ckpt @ epoch 96，test 1024 条单步
- 超参：`hid_S=64, hid_T=256, N_S=2, N_T=4, drop_path=0.1, batch=8, lr=2e-3, onecycle`

| 指标 | SimVP(cb) | persistence | 改善 |
|---|---|---|---|
| RMSE (K) | **1.5745** | 4.5492 | ↓65.4% |
| MAE (K) | 0.7910 | 2.2022 | ↓64.1% |
| MSE (K²) | 2.4791 | 20.6956 | ↓88.0% |

> 换算到归一化空间：`MSE_norm = 2.4791 / 19.33² = 0.00663`

---

## 三方横向对照（单步，同一 test 集，换算到亮温 K）

| 模型 | MSE (归一化) | RMSE (K) | 备注 |
|---|---|---|---|
| SimVP（cb 全量训练） | 0.00663 | **1.57** | 100 epoch 全量 |
| Vanilla UNet（ly, seed42） | 0.00809 | **1.74** | 单 seed，rollout 表现最差 |
| Vanilla UNet（ly, Official） | 0.00827 | **1.76** | 官方 ckpt |
| persistence | — | **4.55** | 弱基线（下限） |

**当前唯一可立即对外说的结论**：同为 2→1 单步、同一 test 集、同一归一化空间下，
SimVP 的单步 RMSE（1.57 K）优于 Vanilla UNet（1.74–1.76 K）约 **10%**（MSE 约优 18%）。

> 口径警告：cb 的 SimVP 是**确定性单步**，UNet 的 18 步是**自回归 rollout**，
> 两者**不能直接并列**。上表只比 **单步 LT=1**。要比较 3 小时预报能力，
> 必须先给我们的 SimVP 实现 rollout（本项目的 `eval_simvp_baseline.py` 已支持 `--rollout-steps 18`）。

---

## 独立验证记录（对 cb / ly 的结论做过取舍，不盲信）

凡是写进本文件的外部结论，都标注了验证状态。**未验证的不得作为结论引用。**

| # | 结论 | 来源 | 我的验证 | 状态 |
|---|---|---|---|---|
| 1 | `MEAN_K=279.0699458792467`、`STD_K=19.32967519050003` | cb | 与数据集自带 `satcast/simple_code.md`（第 72/103/158 行）逐位比对一致 | ✅ 采信（有独立一手来源） |
| 2 | zarr 已是归一化空间（mean=0, std=1） | cb | `simple_code.md` 第 16 行明确写 "stored in normalized space"；实测 train input `mean≈-0.007/std≈0.97`、test `mean≈0.13/std≈0.81` | ✅ 采信（test 偏离 0/1 是因为共用**训练集**统计量，符合预期） |
| 3 | **`output[:,0]` = t+10min**（单步任务的 target） | cb | 实测：已知 10 分钟步长的 `\|X[:,1]−X[:,0]\| = 0.13027`，而 `\|Y[:,0]−X[:,1]\| = 0.13064`，**比值 1.003**；且 `\|Y[:,j]−Y[:,j-1]\|≈0.130` 对全部 j 成立、`\|Y[:,j]−X[:,1]\|` 随 j 严格单调递增 | ✅ 采信，且证据强于 cb 原文（原文只验了单调性，我额外验了步长一致） |
| 4 | 18 帧输出 = 等间隔 10 分钟 → 覆盖到 **3 小时** | 推论 | 由上一条 `\|Y[:,j]−Y[:,j-1]\|` 恒定 ≈0.130 直接推出 | ✅ 采信 |
| 5 | OpenSTL `metrics.py` 用 `.sum()` 放大 65536/256 倍 | cb | 直接读我们自己 `openstl/core/metrics.py` 第 32–45 行确认 `np.mean(...,axis=(0,1)).sum()` | ✅ 采信（在本仓库复现确认） |
| 6 | cb 的 SimVP 结果（RMSE 1.5745 K 等） | cb 报告 | 交叉核对 cb 的**实际产物** `outputs/satcast_simvp_full/test_metrics.json`：`mse_norm=0.006635`、`rmse_K=1.5745278`、`persistence rmse_K=4.5492461`、`epoch=96`、4.7053M 参数，与报告数字**完全一致** | ✅ 采信（报告↔产物一致） |
| 7 | cb 的超参 / "每样本激活 2.53 GB" | cb | **未验证**（属 cb 自己的工程选择，与我们的结论无关） | ⚠️ 仅作参考，不引用为事实 |
| 8 | cb 称 `trainer.test()` 跑在最终权重而非 best.ckpt | cb | **未逐行验证** | ⚠️ 但我们**无论如何都显式加载 best.ckpt**，该结论不影响我们的正确性 |
| 9 | ly 文档写 "input [0.139,0.695], mean≈0.575" | ly | 与盘上 zarr 实测（mean≈0.13, std≈0.81, range≈[-4.8,2.5]）**矛盾**，且与 `simple_code.md` 冲突 | ❌ **判定为过时/错误描述，不采信**；以 `simple_code.md` 为准 |
| 10 | ly 的 UNet 单步 MSE 0.00809 可直接与 cb 的 SimVP 0.006635 比较 | 我的推论 | 双方都在同一 test zarr（1024 条）、同一归一化空间、同为逐像素口径（`F.mse_loss` ≡ 全元素均值） | ✅ 采信，但**仅限 MSE/MAE/RMSE**；SSIM/PSNR 因隐含 [0,1] 动态范围假设，只作同口径横向对比，不解释绝对值 |

### 🔒 最强验证：persistence 全量独立复算（与 cb 逐位比对）

不依赖任何模型，只用 `persistence = 最近输入帧` 在全量 **1024 条** test 集上独立复算，
与 cb 产物 `test_metrics.json` 的同名指标比对：

| 指标 | 我方独立复算 | cb 产物 | 相对偏差 |
|---|---|---|---|
| `mse_norm` | 0.05538979405537248 | 0.05538979434246005 | **0.0000 %** |
| `mae_norm` | 0.11392806656658649 | 0.11392806698495406 | **0.0000 %** |
| `rmse_norm` | 0.23535036446832300 | 0.23535036507823830 | **0.0000 %** |
| `rmse_K` | 4.5492461 | 4.549246112927948 | **0.0000 %** |
| `mae_K` | 2.2021925 | 2.202192529900692 | **0.0000 %** |

（差异仅出现在第 9 位小数，属 float32 累加舍入，非口径差异。）

**这一条同时证明了 5 件事**：test zarr 读取正确、`output[:,0]` 确为 t+10min、
逐像素口径与 cb 一致、`STD_K` 换算一致、persistence 定义一致。
→ 我们的 baseline 对齐链路**已闭环验证**；cb 的这批数字**可以放心引用**。

### 由此确认的横向结论（单步 LT=1）
SimVP(cb) `mse_norm=0.006635` → RMSE **1.57 K**；UNet(ly) `mse=0.008092` → RMSE **1.74 K**；
persistence → **4.55 K**。同集合、同空间、同口径下 **SimVP 单步优于 UNet 约 10%（RMSE）/18%（MSE）**。

> 反例提醒：这个结论**不能**外推到 3 小时。cb 只做了单步，ly 的 UNet 做了 18 步；
> 两者预报时长不同，**不可并列**。要比 3 小时必须先给我们的 SimVP 跑 rollout。

---

<!-- RESULTS:BEGIN -->
## Results — 本项目 SimVP（**自动生成**于 2026-10-07 13:07）

> 本段由 `tools/finalize_baseline_report.py` 从评估 JSON 自动生成，**请勿手工编辑**（改数请改脚本里的基线常量或重跑评估）。

**checkpoint**：`work_dirs/satcast_simvp_full/best_ep32.ckpt`（best epoch = **30**，1024 样本评估，4.7053M 参数）

### 场景 1：单步 teacher-forced（test 全量 1024 条，权威数字）

| 模型 | MSE(归一化) | RMSE(K) | MAE(K) | SSIM |
|---|---|---|---|---|
| **本项目 SimVP** | 0.007336 | 1.6556 | 0.8698 | 0.8980 |
| SimVP（cb, 100 ep） | 0.006635 | 1.5745 | 0.7910 | — |
| Vanilla UNet（ly, seed42） | 0.008092 | 1.7388 | 0.8960 | 0.8913 |
| Vanilla UNet（ly, Official） | 0.008272 | 1.7580 | — | 0.8939 |
| persistence | 0.055390 | 4.5492 | 2.2022 | — |

- MSE 相对 persistence 降低 **86.76%**
- persistence **4.5492 K** 与 cb 独立复算（4.5492 K）偏差 **0.0010%** → 口径闭环
- **单步结论**：优于 UNet（MSE 好 **9.3%**）
  与 cb 的 100-epoch 版相比 RMSE **高 5.2%**

### 场景 2：18 步自回归 rollout（128 条子集，对齐 ly 协议）

| LT | SimVP MSE | RMSE(K) | persistence(K) | UNet seed0 MSE | UNet Official MSE |
|---|---|---|---|---|---|
| 1 | 0.006353 | 1.5407 | 4.0131 | 0.006878 | 0.007019 |
| 2 | 0.018278 | 2.6133 | 5.5770 | — | — |
| 3 | 0.035468 | 3.6404 | 6.6409 | 0.034210 | 0.034390 |
| 4 | 0.058102 | 4.6593 | 7.4632 | — | — |
| 5 | 0.086736 | 5.6928 | 8.1429 | — | — |
| 6 | 0.121676 | 6.7426 | 8.7163 | 0.092090 | 0.092310 |
| 7 | 0.162864 | 7.8008 | 9.2191 | — | — |
| 8 | 0.209554 | 8.8486 | 9.6709 | — | — |
| 9 | 0.260940 | 9.8740 | 10.0931 | 0.163760 | 0.164910 |
| 10 ← 起劣于 persistence | 0.315227 | 10.8527 | 10.4656 | — | — |
| 11 | 0.371976 | 11.7891 | 10.8200 | — | — |
| 12 | 0.430138 | 12.6774 | 11.1679 | 0.249610 | 0.248190 |
| 13 | 0.489128 | 13.5187 | 11.5065 | — | — |
| 14 | 0.548782 | 14.3194 | 11.8263 | — | — |
| 15 | 0.608257 | 15.0754 | 12.1217 | 0.353420 | 0.352090 |
| 16 | 0.666728 | 15.7833 | 12.4099 | — | — |
| 17 | 0.725106 | 16.4598 | 12.7000 | — | — |
| 18 | 0.782793 | 17.1020 | 12.9810 | 0.419200 | 0.490900 |

### 关键发现

1. **单步强、长时弱**：LT=1 SimVP MSE 0.006353 vs UNet(seed0) 0.006878（略胜）；但 LT=18 SimVP 0.782793 vs UNet(seed0) 0.419200 → 差 **86.7%**，比 UNet 3-seed 均值（0.488）差 **60.4%**。
2. **约 100 分钟后"预测不如不动"**：从 **LT=10** 起 SimVP 的 RMSE 反超 persistence，LT=18 达 17.10 K vs 12.98 K。
3. **可能原因（待验证，勿当结论）**：容量差（SimVP gSTA 4.7M vs UNet2DModel 47.6M，约 10 倍）；无任何 rollout-aware 训练；仅单 seed，未测 SimVP 自身方差（对照：ly 测得 UNet 在 LT=18 的 CV=13.4%）。

> ⚠️ 引用限制：以上为**单 seed** 结果；与 UNet 做因果性对比前建议补 SimVP 多 seed（≥3）。
<!-- RESULTS:END -->

---

## 口径坑记录（必须记住）

1. **OpenSTL 指标放大 bug**：`openstl/core/metrics.py` 的 `MSE/MAE/RMSE` 用
   `np.mean(..., axis=(0,1)).sum()`，对空间维求和 → MSE/MAE ×65536、RMSE ×256。
   本项目 EXP-002 的 `metrics.npy = [14507.12, 7322.60]` 正是被放大的值
   （换算回逐像素约 MSE 0.221 / MAE 0.112）。**该数不可引用。**
   为遵守 `DEC-003`（不改上游），我们**不修改** `metrics.py`，而是在
   `tools/eval_simvp_baseline.py` 里自己按逐像素口径算。
2. **`ly` 文档的数据范围描述与盘上 zarr 不符**：ly 文档写 "input [0.139, 0.695], mean≈0.575"，
   但实测 test zarr 为 `mean≈0.13, std≈0.81, range≈[-4.8, 2.5]`，且
   `satcast/simple_code.md` 明确写 "stored in normalized space (mean=0, std=1)"。
   以 `simple_code.md` 为准，ly 文档该行为过时描述。
3. **老 checkpoint 的归一化不一致**：EXP-002 的 smoke 模型用**子集自算**
   `mean=0.093006 / std=0.865920` 训练，而 baseline 空间是物理归一化。
   评估老 ckpt 必须先用同一组常数还原（见评估脚本 `--norm-mean/--norm-std`）。
4. **⚠️ RMSE 有两种口径，差 15%（本项目踩过并已修）**
   * `rmse_global` = `sqrt(全局 MSE)` —— **cb 的口径**
   * `rmse_persample_mean` = `mean(逐样本 RMSE)` —— **ly 的口径**
   由 Jensen 不等式 `E[√X] ≤ √(E[X])`，后者**系统性偏小**。
   实测同一份 persistence（全量 1024 条）：
   | 口径 | 值 | 与 cb 偏差 |
   |---|---|---|
   | global（cb） | **4.5492461 K** | 0.0000 % |
   | per-sample（ly） | 3.8772185 K | −14.8 % |
   * MSE / MAE **无此歧义**（样本像素数相同，`mean(逐样本)==全局`），可安全跨源比较。
   * **SSIM / PSNR 按 ly 的逐样本口径**（ly 报这两个）；cb 未报。
   * 本项目 `eval_simvp_baseline.py` **两个都输出**，主值（`rmse_K`）用 global 以对齐 cb。

---

## 我们如何对齐（实现）

- 统一数据适配器：`openstl/datasets/dataloader_satcast.py`
  （train/val/test 各读一份 zarr；暴露 `MEAN_K/STD_K`；惰性加载；严格 shape 校验）
- 统一评估脚本：`tools/eval_simvp_baseline.py`
  （加载 best.ckpt；单步 + 可选 18 步 rollout；逐像素 MSE/MAE/RMSE/SSIM/PSNR；
   同时输出归一化空间与亮温 K；含 persistence 基线）
- 提交脚本：`test_dl/eval_baseline.slurm`

运行示例：
```bash
cd /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
# 评估 EXP-002 的 smoke ckpt（老归一化常数）
python tools/eval_simvp_baseline.py \
    --ckpt work_dirs/simvp_goes13_smoke/checkpoints/best.ckpt \
    --config configs/goes13/simvp/SimVP_gSTA.py \
    --norm-mean 0.093006 --norm-std 0.865920
# 未来按 baseline 协议训练的模型（无需额外归一化）
python tools/eval_simvp_baseline.py --ckpt <...>/best.ckpt --rollout-steps 18
```

## Deviation notes
- **D5** 不修改上游 `openstl/core/metrics.py`（遵守 `DEC-003`），改为在评估脚本内
  按 baseline 的逐像素口径自行计算，与 ly 的 `F.mse_loss` 口径一致。
- **D6** 新增 `dataloader_satcast.py` 与旧 `dataloader_goes13.py` **并存**：
  旧脚本（EXP-002/003）保持不动以免破坏既有产物；新的 baseline 对齐流程用新适配器。
