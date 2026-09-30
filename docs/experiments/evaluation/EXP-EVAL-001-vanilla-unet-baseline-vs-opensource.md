# EXP-EVAL-001 — Vanilla UNet Baseline vs Paper OpenSource: Test-Set Evaluation

## Identity
- 实验 ID：EXP-EVAL-001
- 状态：**Completed**
- 日期：2026-09-27
- 负责人：liuyi
- 关联 RQ：RQ-001 / RQ-003 / RQ-005
- 关联 HYP：HYP-005 / HYP-006
- 依赖：EXP-010

## Research question 与 hypothesis

- Research Question（原始，已过时）：本地训练的 Vanilla UNet（EXP-010，47 epoch early stopping）与论文开源 Vanilla UNet（原假设 210 epoch full training）在 held-out test set 上的差距有多大？差距来自哪里？
- Research Question（审计修正后）：两模型训练 epoch 数几乎相同（开源 48，我们 47，均被 early stopping 停），但 rollout 表现差 14% MSE。**在训练条件本质相同的前提下，rollout gap 的真实来源是什么？**
- Hypothesis（原始，已过时）：(1) 单步 teacher-forced 指标应与开源模型在 ±5% 以内（因为单步回归在 30–40 epoch 后已收敛）；(2) 长步 autoregressive rollout 可能存在差距（训练 epoch 不足导致动力学稳定性缺失）。
- Hypothesis（审计修正后）：(1) 已 confirmed — 单步 ±3%；(2) 已 confirmed — rollout 差距存在，但**原因非 epoch 数差** → 需要独立调查 checkpoint 时机 / diffusers 版本 / early stopping 监控指标
- Scientific motivation：CIRA-Diff 论文中所有扩散模型增益都是相对于 Vanilla UNet baseline 报告的。如果 reproduction 的 baseline 与开源 baseline 在 rollout 上差异显著，后续对比 EDM/LDM/CorrDiff 时必须使用"同一 reproduction 训练流程"而非直接引用论文数字。

## Dataset

- Test set：`/data1/satcast/edm_GOES_ch13_test_dataset.zarr`（1024 samples）
- input_images shape：(1024, 2, 256, 256), dtype=float16
- output_images shape：(1024, 18, 256, 256), dtype=float16
- Time period：Aug 2024 – Feb 2025（见 splits.md）
- Sensor/channel/unit：GOES-16 ABI C13（亮温）
- 数值范围：input [0.139, 0.695], output [0.152, 0.708], mean ≈ 0.575
- 归一化：无显式 mean/std

## Model comparison

| 属性 | EXP-010 Best (A) | Paper OpenSource (B) |
|------|-----------------|---------------------|
| Checkpoint | `outputs/vanilla_unet_full/best_internal_unet/` | `/data1/satcast/unet_vanilla/unet_vanilla/` |
| Architecture | UNet2DModel 47.6M params | UNet2DModel 47.6M params |
| diffusers_version (config) | 0.40.0 | 0.31.0 |
| 训练上限 num_epochs | 210 | 210（main 分支原始脚本） |
| **实际训练终止 epoch** | **47 (best at 37, early stopping)** | **48（TensorBoard events step 0..48）** |
| early stopping | 有 (patience=10, moving window=5) | **有（main 分支脚本 patience=10）** |
| Rollout 训练 | 无 (纯单步 MSE) | **无（main 分支脚本 grep 未找到任何 rollout/multi-step/scheduled_sampling 实现）** |
| Batch size (effective) | 24 × 2 = 48 | 45 × 1 = 45 |
| LR schedule 配置 | warmup_steps=500, num_training_steps=len(dl)*210 | **同左（main 分支相同配置）** |
| **Checkpoint 保存时机** | **显式保存 internal val best at epoch 37** | **原始脚本每 epoch 覆盖 save_pretrained，不存 best；开源 tar 里的 checkpoint 是 epoch 48 还是 best val loss epoch → 未知** |

> 两侧 config.json 已人工 diff，唯一差异是 diffusers_version（0.31.0 vs 0.40.0）。
>
> **【重要修正：2026-09-27 审计】** 之前版本错误地假设开源模型"训练了完整 210 epoch"。直接证据来自 `/data1/satcast/unet_vanilla.tar` 内的 TensorBoard events（`train_example/events.out.tfevents.*`）：
> - `epoch_loss`: 49 个事件，step range `[0, 48]` → **epoch 0..48**
> - `val_loss`: 49 个事件，step range `[0, 48]` → **epoch 0..48**
> - 因此开源模型也在 epoch 48 被 early stopping 停掉，和我们 EXP-010（epoch 47）几乎完全相同。
> - **Rollout loss / scheduled sampling / rollout-aware checkpoint 全量 grep 均未在 main 分支原始脚本中找到**。论文明确描述单步 MSE + held-out val early stopping。

## Evaluation protocol

- Metrics（纯 PyTorch 实现，无 torchmetrics/skimage/kornia 依赖）：
  - MSE = F.mse_loss(pred, target)
  - MAE = F.l1_loss(pred, target)
  - SSIM = 11x11 window SSIM (F.avg_pool2d 实现), C1=0.01^2, C2=0.03^2
  - PSNR = 10 * log10(1 / (MSE + 1e-12)), 输入范围 [0,1]
- Scenario 1 — Single-step teacher-forced: (X[t-1], X[t]) -> X[t+1], 全量 1024 test samples
- Scenario 2 — 18-step autoregressive rollout: 递归 18 步, 128 样本子集, 每 lead time 独立统计
- 每 lead time 逐样本算指标 -> mean / std / median
- Deterministic UNet, 无 ensemble/seed

## Runtime 与 artifacts

- Command:
  ```bash
  CUDA_VISIBLE_DEVICES=1 python -u scripts/Chase_2025/eval_unet_comparison.py \
    --ckpt-a outputs/vanilla_unet_full/best_internal_unet \
    --ckpt-b /data1/satcast/unet_vanilla/unet_vanilla \
    --max-rollout-samples 128
  ```
- Environment: conda `cira-diff-ly`, GPU RTX 4090 #1（GPU 0 被占用）
- Git commit: `cf67d6e`
- 新增 script: `scripts/Chase_2025/eval_unet_comparison.py`（未 git commit）
- 总耗时: single-step ~4 min, rollout ~8 min
- Output dir: `outputs/eval_unet_comparison/`
- Artifacts:
  - `comparison_summary.json` — 两模型单步 mean/std/median
  - `single_step_metrics_{a,b}.npz` — 1024 样本逐样本指标
  - `rollout_metrics_{a,b}.json` — 18 LT x 128 样本逐样本指标
  - `rollout_metrics_curves.png` — 4 指标 vs lead time 曲线图
  - `visual_comparison_single_step.png` — 4 样本并排可视化

## Results

### Scenario 1 — Single-step Teacher-Forced (1024 test samples)

| Metric | EXP-010 Best | Paper OpenSource | Δ (B - A) |
|--------|-------------|-----------------|-----------|
| MSE | 0.008092 +- 0.00899 | 0.008272 +- 0.00932 | +0.00018 (A 优 2.2%) |
| MAE | 0.04634 +- 0.0290 | 0.04517 +- 0.0299 | -0.00118 (B 优 2.6%) |
| SSIM | 0.8913 +- 0.0536 | 0.8939 +- 0.0541 | +0.0027 (B 优 0.3%) |
| PSNR | 23.96 +- 5.82 | 24.04 +- 6.05 | +0.07 (基本打平) |

### Scenario 2 — 18-step Autoregressive Rollout (128 样本子集)

| LT | MSE A | MSE B | dMSE | SSIM A | SSIM B | dSSIM |
|----|-------|-------|------|--------|--------|-------|
| 1 | 0.00688 | 0.00702 | +0.00014 | 0.9023 | 0.9048 | +0.0025 |
| 2 | 0.01888 | 0.01917 | +0.00029 | 0.7895 | 0.7974 | +0.0078 |
| 3 | 0.03438 | 0.03439 | +0.00001 | 0.6865 | 0.7017 | +0.0152 |
| 4 | 0.05257 | 0.05181 | -0.00076 | 0.5972 | 0.6204 | +0.0232 |
| 5 | 0.07348 | 0.07119 | -0.00229 | 0.5221 | 0.5538 | +0.0317 |
| 6 | 0.09661 | 0.09231 | -0.00430 | 0.4587 | 0.5000 | +0.0413 |
| 7 | 0.12193 | 0.11499 | -0.00694 | 0.4040 | 0.4557 | +0.0517 |
| 8 | 0.15013 | 0.13911 | -0.01102 | 0.3581 | 0.4193 | +0.0612 |
| 9 | 0.18187 | 0.16491 | -0.01697 | 0.3183 | 0.3883 | +0.0700 |
| 10 | 0.21620 | 0.19137 | -0.02483 | 0.2841 | 0.3622 | +0.0780 |
| 11 | 0.25311 | 0.21918 | -0.03393 | 0.2546 | 0.3398 | +0.0852 |
| 12 | 0.29247 | 0.24819 | -0.04428 | 0.2281 | 0.3207 | +0.0926 |
| 13 | 0.33451 | 0.27941 | -0.05510 | 0.2044 | 0.3036 | +0.0992 |
| 14 | 0.37919 | 0.31418 | -0.06501 | 0.1823 | 0.2881 | +0.1059 |
| 15 | 0.42538 | 0.35209 | -0.07330 | 0.1615 | 0.2740 | +0.1125 |
| 16 | 0.47346 | 0.39374 | -0.07972 | 0.1412 | 0.2614 | +0.1202 |
| 17 | 0.52340 | 0.44001 | -0.08339 | 0.1219 | 0.2491 | +0.1272 |
| 18 | 0.5743 | 0.4909 | -0.0834 | 0.1034 | 0.2375 | +0.1341 |

> 注：MSE 从 LT=4 起 B 优于 A；SSIM 从 LT=1 起 B 始终优于 A，差距随 lead time 单调递增。LT=18 是最关键对比点。

### 统计显著性

128 样本 LT=18 MSE paired t-test 量级：mean diff = 0.083, A std = 0.019, B std = 0.014, 差距高度显著 (p << 0.01)。

## Interpretation

**【重要修正：2026-09-27 审计】** 原始 Interpretation 的第 2–5 点全部基于"开源模型训练了完整 210 epoch，我们只训了 47 epoch"的错误前提。审计确认两模型实际训练 epoch 数几乎相同（开源 48，我们 47），因此：

1. **Hypothesis-001 Supported（单步 ±5%）**：MSE/MAE/SSIM/PSNR 差异均在 ±3% 以内，确认 30–40 epoch 后单步 2->1 回归已充分收敛。

2. **Hypothesis-002 Supported（rollout 差距显著）**：LT=18 MSE 差 14%，SSIM 差 130%；差距随 lead time 单调递增。**但原因不是 epoch 数差**——两模型训练了几乎相同的 epoch 数（47 vs 48）。差异必须来自其他因素。

3. **Rollout 差距的真实候选来源（按证据强度排序）**：

   | 候选因素 | 证据强度 | 说明 |
   |---------|---------|------|
   | **Checkpoint 保存时机** | ⭐⭐⭐⭐⭐ | 我们显式保存了 internal val best at epoch 37；原始脚本每 epoch 覆盖 `save_pretrained`（不存 best），开源 tar 里的 checkpoint 是 epoch 48（接近平台期末尾）还是 best val loss epoch → 这是当前最强候选 |
   | **diffusers 版本差异** | ⭐⭐⭐⭐ | 开源 config 声明 `_diffusers_version=0.31.0`，我们用 0.40.0。UNet2DModel 的 ResNet block / time embedding / GroupNorm 内部实现是否在两个版本间有变更？→ 需要查 diffusers release notes |
   | **LR schedule 是否走完** | ⭐⭐⭐ | 原始脚本 `num_training_steps = len(train_dl) * num_epochs(210)`，但实际只训了 ~11K steps（48 epoch）。warmup_steps=500，210 epoch 预算下的总 LR decay 曲线在 step ~11K 时可能只走完了前 23% → 两模型都没走完完整 schedule，但 LR 走势完全一样 |
   | **Early stopping 监控指标** | ⭐⭐⭐ | main 分支原始脚本 early stopping 用的是 held-out validation zarr 还是 training set 内 80/20 split？如果是 held-out val，我们的 internal val（training set 内 80/20 split）可能 early stop 时机不同 |
   | **Dataset split seed** | ⭐⭐ | 论文说 random 80/20 split，我们用 seed=42。如果 split 种子不同，训练/验证样本不同 → 影响 early stopping 触发时机和 best epoch |
   | **Batch size 差异** | ⭐ | 原始 45 vs 我们 48，差距极小，不太可能解释 14% rollout MSE |
   | **Rollout loss / scheduled sampling** | ❌ | **已排除**。main 分支脚本全量 grep 未找到任何相关实现。论文明确描述单步 MSE training + held-out val early stopping |

4. **为什么 SSIM 差距比 MSE 大**：这条观察本身仍然成立，但不能再归因于"开源模型训练更久，学到更一致的空间频率响应"。更可能的解释是两模型 checkpoint 对应的平台期阶段不同（我们是 best at epoch 37，开源可能是 last at epoch 48），模型在平台期不同点的误差空间分布特征有差异。

5. **MSE 方向在 LT=1-3 反转（A 优）但 SSIM 始终同方向（B 优）**：这条观察仍然成立。在 teacher-forced 条件下（短 LT），两模型像素级 MSE 接近；但 B 的结构一致性更好。随着 rollout 推进，A 的动力学不稳定性导致 MSE 累积。这个动力学不稳定性的差异可能源自上面第 3 点中的任何因素（checkpoint 时机、diffusers 版本实现差异等）。

### 原始 Interpretation 错误溯源

原始版本的所有 epoch 数相关推断都来自：
- main 分支脚本第 62 行 `num_epochs = 210`（**上限，不是实际值**）
- 我们在 `VANILLA_UNET_HANDOVER.md` 里从上限推导了"210 epoch 总耗时 ≈ 70h"
- 然后把"开源模型训了完整 210"当成了事实 → 这是一个典型的把**脚本参数**当成**实际行为**的错误

**修正后结论**：两模型训练方法、epoch 数、early stopping 策略本质相同。Rollout 差距是一个在同等训练条件下仍存在的可复现差异，需要进一步调查 checkpoint 选择时机和 diffusers 版本实现差异。

## Conclusion

| Hypothesis | 判定 | 依据 |
|------------|------|------|
| HYPOTHESIS-001（单步 ±5%） | **Supported** | 所有指标差异 < 3% |
| HYPOTHESIS-002（rollout 差距存在） | **Supported** | LT=18 MSE 差 14%，SSIM 差 130%；单调递增 |
| HYP-005（Vanilla UNet 作为 det ref） | **Supported** | 单步指标对齐 |
| HYP-006（diffusion 增益边界） | **Partially Supported** | rollout 端存在显著 gap，且 gap 原因不是 epoch 数差 → 需要深入调查 checkpoint 时机和 diffusers 版本差异 |

### Reproduction protocol 启示

- 本地 reproduction 的 baseline 在单步水平上可信，足以判断扩散模型是否在单步上有增益
- Rollout 差距**不是 epoch 数差导致**（两模型都是 47-48 epoch 被 early stopping 停）。这意味着 gap 来自其他因素（checkpoint 保存时机、diffusers 版本实现差异、early stopping 监控指标差异等）
- **原始建议"跑满 210 epoch 不用 early stopping"已作废**——因为原始脚本的 early stopping 逻辑和我们一样，两模型也都被 early stop 停了，跑满 210 epoch 是违反协议的
- **修正后的建议**：
  1. 先搞清楚 checkpoint 时机和 diffusers 版本差异（快速实验：用 diffusers 0.31.0 重训 1-2 epoch 对比；或用 main 分支脚本原版 early stopping + 原版 checkpoint 保存逻辑再训一遍）
  2. 后续 EDM/LDM/CorrDiff reproduction 训练脚本**必须保留 early stopping 逻辑**，checkpoint 保存策略必须统一
  3. 可以考虑新增一个 EXP（如 EXP-012）专门调查 rollout gap 来源——控制 checkpoint 时机 + diffusers 版本，纯单因素实验

## Limitations

- Rollout 评估只有 128 样本子集（完整 1024 样本 ~60 min/模型，本次只跑了 128）
- **开源 checkpoint 保存时机不确定**：原始脚本每 epoch 覆盖 save_pretrained，不存 best。开源 tar 里的 checkpoint 是 epoch 48 last epoch 还是 best val loss epoch → 这是 rollout gap 的首要嫌疑，但当前无法确认
- diffusers 版本差异（0.31.0 vs 0.40.0）是否影响 UNet2DModel 内部实现 → 未量化
- 原始脚本 early stopping 监控指标是 held-out val zarr 还是 training set 内 80/20 split → 未确认（影响 best checkpoint 定义）
- 未做 per-region / per-event case analysis（EXP-007 计划内容）
- 未做 cold-cloud vs warm-cloud 分桶分析（论文 protocol 要求）
- 未做 CSI/POD/FAR/ETS 阈值指标（EXP-006/EXP-007 计划内容）
- eval script 未加入 `scripts/evaluation/` 标准化目录结构

## Reproducibility

- Eval script：`scripts/Chase_2025/eval_unet_comparison.py`（新建，未 git commit）
- 环境：`cira-diff-ly`，RTX 4090 #1
- 运行命令：见 Runtime 章节
- 数据：`/data1/satcast/edm_GOES_ch13_test_dataset.zarr`
- 模型权重：`outputs/vanilla_unet_full/best_internal_unet` + `/data1/satcast/unet_vanilla/unet_vanilla`
- 指标实现：纯 PyTorch F.avg_pool2d + F.mse_loss + F.l1_loss，无 torchmetrics/skimage 依赖
- 所有中间结果以 .npz / .json / .png 固化，可复算
- **审计发现可复算**：开源 TensorBoard events 直接证据（`/data1/satcast/unet_vanilla/unet_vanilla/logs/train_example/events.out.tfevents.1733156481.ed3e02f3f84a.324384.0`），可通过 `tensorboard.backend.event_processing` 读取

## 关联文献 / 决策 / 下一实验

- 关联文献：NVIDIA CIRA-Diff 2023；Chase et al. 2025；开源 Vanilla UNet（diffusers hub id 未知，本窗口通过 tar 包 `/data1/satcast/unet_vanilla.tar` 解压获取）
- 关联 EXP：EXP-010（训练权重）、EXP-011（EDM full training 并行运行）
- **关联 DEC 修正**：原始 DEC"后续 reproduction 跑满 210 epoch 不用 early stopping"已作废。修正为：
  - 必须保留 early stopping（原始脚本就有）
  - 必须统一 checkpoint 保存策略（显式保存 best val loss epoch）
  - 下一步需要单因素实验调查 rollout gap 来源
- 下一实验：
  - EXP-012（新）：调查 rollout gap 来源——控制 checkpoint 时机 + diffusers 版本，纯单因素实验
  - EXP-EVAL-002：EDM reproduction vs 开源 EDM（需 EXP-011 训练完成）
  - EXP-EVAL-003：CorrDiff reproduction vs 开源 CorrDiff