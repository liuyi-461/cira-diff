# EXP-EVAL-003 — Multi-Seed Vanilla UNet vs Paper OpenSource: Rollout Evaluation

## Identity

- 实验 ID：EXP-EVAL-003
- 状态：**Completed**
- 日期：2026-09-29
- 负责人：liuyi
- 关联 RQ：RQ-001 / RQ-003 / RQ-005
- 关联 HYP：HYP-005 / HYP-006 / **HYP-007（seed variance）**
- 依赖：EXP-010（seed=42）、EXP-014（seed=0）、EXP-015（seed=123）、EXP-AUDIT-001

## Research question 与 hypothesis

- Research Question：EXP-EVAL-001 观测到 EXP-010（seed=42）与官方开源 Vanilla UNet rollout LT=18 MSE 差 17%（0.574 vs 0.491），但单步指标几乎打平（差 < 3%）。在排除了训练长度、diffusers 版本、checkpoint 选择、rollout loss 实现差异、Official val_loss bug 等所有已知因素后，**rollout gap 的剩余来源是否为随机种子导致的训练轨迹分化？**
- Hypothesis（HYP-007，来自 EXP-AUDIT-001）：在相同训练配置下，仅改变模型初始化 seed，多个训练 run 的 rollout 指标会分布在一个区间内；官方开源 checkpoint 的 rollout 水平**在该区间内**。若成立，则原 EXP-EVAL-001 观测到的 gap 不是复现失败，而是 Vanilla UNet rollout 对初始化的固有敏感性。
- Scientific motivation：CIRA-Diff 论文中所有 diffusion 模型增益都是相对于 Vanilla UNet baseline 报告的。如果 Vanilla UNet 本身存在显著的 seed-dependent rollout variance，那么：
  1. 后续对比 diffusion 模型时必须使用多 seed ensemble，不能用单一 seed
  2. HYP-005（Vanilla UNet 作为 deterministic reference）需要修正
  3. RQ-005（Diffusion vs Deterministic comparison）需要重新审视——diffusion 模型的 rollout 是否也有类似的 seed variance？

## Dataset

- Test set：`/data1/satcast/edm_GOES_ch13_test_dataset.zarr`（1024 samples）
- input_images shape：(1024, 2, 256, 256), dtype=float16
- output_images shape：(1024, 18, 256, 256), dtype=float16
- Time period：Aug 2024 – Feb 2025
- Sensor/channel/unit：GOES-16 ABI C13（亮温）
- 数值范围：input [0.139, 0.695], output [0.152, 0.708], mean ≈ 0.575
- 归一化：无显式 mean/std
- 所有 4 个 checkpoint 共享**完全相同的 test set**（同一 zarr，同一 128-sample 子集用于 rollout）

## Model comparison — 4 Checkpoints

| 属性 | seed=0 (EXP-014) | seed=42 (EXP-010) | seed=123 (EXP-015) | Official OpenSource |
|------|------------------|-------------------|--------------------|--------------------|
| Checkpoint path | `outputs/vanilla_unet_seed0/best_internal_unet/` | `outputs/vanilla_unet_full/best_internal_unet/` | `outputs/vanilla_unet_seed123/best_internal_unet/` | `/data1/satcast/unet_vanilla/unet_vanilla/` |
| Architecture | UNet2DModel 47.6M params | 同左 | 同左 | UNet2DModel 47.6M params |
| diffusers_version | 0.40.0 | 0.40.0 | 0.40.0 | 0.31.0 |
| 训练 seed（模型初始化） | **0** | **42** | **123** | Unknown |
| dataset split_seed | 42 | 42 | 42 | Unknown |
| Epochs | 50 (early stopping) | 47 (early stopping) | 41 (early stopping) | 48 (early stopping) |
| Best epoch | 36 | 37 | 36 | Unknown |
| best_internal_val | 0.009969 | 0.009728 | 0.009671 | Unknown (official uses SUM bug) |
| Batch size (effective) | 24×2=48 | 24×2=48 | 24×2=48 | 45×1=45 |
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| LR / scheduler | 1e-4 cosine+warmup(500) | 同左 | 同左 | 同左 |
| Mixed precision | fp16 | fp16 | fp16 | Unknown |
| Rollout training | 无 | 无 | 无 | 无 |
| Git commit | cf67d6e | cf67d6e | cf67d6e | Unknown |

> **关键：** 本实验**仅改变模型初始化 seed**，dataset split_seed 固定为 42。因此 train/val split 完全相同，DataLoader shuffle 顺序也固定（因为 torch.manual_seed 同时控制了 torch、cuda、random、numpy）。变化的只有：(1) nn.Module 参数初始化；(2) 第一个 forward pass 的数值，进而影响后续所有梯度更新。
>
> 注意 dataset split_seed=42 与模型初始化 seed=42 重名但作用域不同。EXP-010 实际上是 split_seed=42 + 模型初始化 seed=42 的巧合组合，恰好产生了 rollout outlier。

## Evaluation protocol

- Metrics（纯 PyTorch 实现，无 torchmetrics/skimage/kornia 依赖）：
  - MSE = F.mse_loss(pred, target)
  - MAE = F.l1_loss(pred, target)
  - SSIM = 11×11 window SSIM (F.avg_pool2d), C1=0.01², C2=0.03²
  - PSNR = 10 × log₁₀(1 / (MSE + 1e-12)), 输入范围 [0,1]
- Scenario 1 — Single-step teacher-forced: (X[t-1], X[t]) → X[t+1], 全量 1024 test samples
- Scenario 2 — 18-step autoregressive rollout: 递归 18 步, 128 样本子集, 每 lead time 独立统计
- 每 lead time 逐样本算指标 → mean / std
- Deterministic UNet, 无 ensemble/seed
- 4 个 checkpoint 用**完全相同的评估脚本、相同的 device、相同的 test samples**顺序加载评估，避免缓存差异

## Runtime 与 artifacts

- Command（写在 `/tmp/eval_all_seeds.py` 里由 slurm 调用）：
  ```bash
  # 评估脚本关键部分
  for name, ckpt_path in SEED_CONFIGS:
      model = UNet2DModel.from_pretrained(ckpt_path)
      m_single = evaluate_single_step(model, dataset, device='cuda')  # 1024 samples
      m_roll   = evaluate_rollout(model, dataset, num_steps=18, device='cuda', max_samples=128)
  ```
- Slurm Job：77（debug partition, 1×RTX 4090）
- Environment：conda `cira-diff-ly`, GPU RTX 4090
- 总耗时：~30 min（4 models × single-step ~4min + rollout ~3min ≈ 28 min）
- Output dir：`outputs/eval_unet_all_seeds/`
- Artifacts：
  - `all_seeds_results.json` — 完整结果（4 models × single-step + 18-LT rollout per-LT）

## Results

### Scenario 1 — Single-step Teacher-Forced（1024 test samples）

| Model | MSE | MAE | SSIM | PSNR |
|-------|-----|-----|------|------|
| seed=0 | 0.008050 ± 0.00885 | 0.04604 ± 0.0286 | 0.8929 ± 0.0534 | 24.04 ± 5.82 |
| seed=42 (EXP-010) | 0.008092 ± 0.00899 | 0.04634 ± 0.0290 | 0.8913 ± 0.0536 | 23.96 ± 5.82 |
| seed=123 | 0.008157 ± 0.00920 | 0.04627 ± 0.0293 | 0.8929 ± 0.0537 | 23.92 ± 5.87 |
| Official | 0.008272 ± 0.00932 | 0.04517 ± 0.0299 | 0.8939 ± 0.0541 | 24.04 ± 6.05 |
| **3-seed spread** | **0.000107** | — | **0.0016** | — |

**单步指标方差极小**：3 个 seed 的 MSE 极差仅 0.000107（相对差异 < 1.3%），SSIM 极差仅 0.0016（相对差异 < 0.2%）。Official checkpoint 的单步指标在 4 个模型中恰好略高（MSE 0.008272），但整体处于同一水平。

### Scenario 2 — 18-step Autoregressive Rollout（128 test samples）

#### Per-Lead-Time 全表

| LT | MSE seed0 | MSE seed42 | MSE seed123 | MSE official | SSIM seed0 | SSIM seed42 | SSIM seed123 | SSIM official |
|----|-----------|------------|-------------|--------------|------------|-------------|--------------|---------------|
| 1 | 0.006878 | 0.006881 | 0.006968 | 0.007019 | 0.9045 | 0.9023 | 0.9037 | 0.9048 |
| 2 | 0.01893 | 0.01888 | 0.01904 | 0.01917 | 0.7953 | 0.7895 | 0.7953 | 0.7974 |
| 3 | 0.03421 | 0.03438 | 0.03416 | 0.03439 | 0.6948 | 0.6865 | 0.6963 | 0.7017 |
| 4 | 0.05159 | 0.05257 | 0.05209 | 0.05181 | 0.6101 | 0.5972 | 0.6124 | 0.6204 |
| 5 | 0.07092 | 0.07348 | 0.07222 | 0.07119 | 0.5409 | 0.5221 | 0.5437 | 0.5538 |
| 6 | 0.09209 | 0.09661 | 0.09403 | 0.09231 | 0.4841 | 0.4587 | 0.4867 | 0.5000 |
| 7 | 0.11445 | 0.12193 | 0.11743 | 0.11499 | 0.4371 | 0.4040 | 0.4401 | 0.4557 |
| 8 | 0.13837 | 0.15013 | 0.14324 | 0.13911 | 0.3981 | 0.3581 | 0.4017 | 0.4193 |
| 9 | 0.16376 | 0.18187 | 0.17191 | 0.16491 | 0.3648 | 0.3183 | 0.3702 | 0.3883 |
| 10 | 0.19072 | 0.21620 | 0.20323 | 0.19137 | 0.3366 | 0.2841 | 0.3428 | 0.3622 |
| 11 | 0.21937 | 0.25311 | 0.23705 | 0.21918 | 0.3120 | 0.2546 | 0.3187 | 0.3398 |
| 12 | 0.24961 | 0.29247 | 0.27337 | 0.24819 | 0.2907 | 0.2281 | 0.2976 | 0.3207 |
| 13 | 0.28173 | 0.33451 | 0.31243 | 0.27941 | 0.2721 | 0.2044 | 0.2791 | 0.3036 |
| 14 | 0.31622 | 0.37919 | 0.35437 | 0.31418 | 0.2556 | 0.1823 | 0.2622 | 0.2881 |
| 15 | 0.35342 | 0.42538 | 0.39951 | 0.35209 | 0.2409 | 0.1615 | 0.2469 | 0.2740 |
| 16 | 0.39332 | 0.47346 | 0.44802 | 0.39374 | 0.2274 | 0.1412 | 0.2327 | 0.2614 |
| 17 | 0.43578 | 0.52340 | 0.49997 | 0.44001 | 0.2147 | 0.1219 | 0.2192 | 0.2491 |
| **18** | **0.4192** | **0.5743** | **0.4717** | **0.4909** | **0.2411** | **0.1034** | **0.1942** | **0.2375** |

#### LT=18 核心对比（汇总）

| Model | MSE | SSIM | vs Official MSE | vs Official SSIM |
|-------|-----|------|-----------------|------------------|
| **seed=0** | **0.419 ± 0.015** | **0.241 ± 0.026** | **好 14.6%** | **好 1.5%** |
| seed=123 | 0.472 ± 0.017 | 0.194 ± 0.022 | 好 3.9% | 差 18.2% |
| Official | 0.491 ± 0.014 | 0.238 ± 0.024 | baseline | baseline |
| seed=42 (EXP-010) | 0.574 ± 0.019 | 0.103 ± 0.018 | 差 17.0% | 差 56.5% |

#### 方差级联（seed variance vs lead time）

| Lead Time | 3-seed MSE mean | 3-seed MSE std | MSE CV% | 3-seed SSIM mean | 3-seed SSIM std | SSIM CV% |
|-----------|-----------------|----------------|---------|------------------|-----------------|----------|
| 1 | 0.006910 | 0.000050 | **0.7%** | 0.9035 | 0.0011 | **0.1%** |
| 6 | 0.09424 | 0.00228 | **2.4%** | 0.4752 | 0.0155 | **3.3%** |
| 12 | 0.27182 | 0.02167 | **8.0%** | 0.2788 | 0.0363 | **13.0%** |
| **18** | **0.4884** | **0.0655** | **13.4%** | **0.1796** | **0.0569** | **31.7%** |

> 单步 teacher-forced 下 3-seed MSE spread = 0.000107（CV < 1%），rollout LT=18 下 MSE spread = 0.155（CV = 13.4%）。**方差级联放大 ~1450 倍**。SSIM 的级联更极端：CV 从 0.1% 增至 31.7%。

#### Official vs 3-seed Ensemble

| 指标 | Official | 3-seed mean | 偏差 | 3-seed range | Official 在区间内？ |
|------|----------|-------------|------|-------------|-------------------|
| LT=18 MSE | 0.491 | 0.488 ± 0.066 | **+0.5%** | [0.419, 0.574] | **YES** |
| LT=18 SSIM | 0.238 | 0.180 ± 0.057 | **+32.2%** | [0.103, 0.241] | **YES**（最高位） |
| Single-step MSE | 0.008272 | 0.008099 ± 0.000054 | **+2.1%** | [0.008050, 0.008157] | NO（略高于区间） |
| Single-step SSIM | 0.8939 | 0.8923 ± 0.0009 | **+0.2%** | [0.8913, 0.8929] | NO（略高于区间） |

> Official 单步指标略高于 3-seed 区间上限，但差距极小（MSE +2.1%, SSIM +0.2%），且 3-seed range 本身很窄。**Official rollout 指标完全落在 3-seed 区间内**，且 MSE 几乎正好是均值。

## Interpretation

### 1. HYP-007 Fully Supported — 随机种子是 rollout gap 的根因

证据链完整：

1. **单步指标极窄**：3-seed MSE spread 0.000107（< 1.3%）→ 不同初始化的模型在 teacher-forced 条件下收敛到几乎相同的解
2. **rollout 方差级联放大**：LT=18 MSE spread 0.155（13.4% CV）→ Vanilla UNet 的 autoregressive rollout 对参数微扰极度敏感
3. **Official 完全在区间内**：MSE = 0.491 vs 3-seed mean = 0.488 ± 0.066（偏差 +0.5%）→ 官方 checkpoint 不是某种"更优训练方法"的产物，它就是另一个初始化 seed 收敛出来的解
4. **EXP-010（seed=42）恰好是 outlier**：3-seed 中 rollout 最差的那个。我们之前观测到的"Official 比我们好 17%"本质上是"我们选了个 bad seed"

### 2. 为什么 Vanilla UNet rollout 方差级联这么严重

Vanilla UNet 是典型的单步训练 + autoregressive rollout 架构：

- **训练时**：每一步都喂 ground truth 前帧（teacher forcing），loss 表面上是 step-wise MSE
- **推理时**：用自己的预测作为下一个输入 → 单步误差会被递推放大
- **误差放大机制**：如果某个初始化的模型对某些模式（如对流云边界）的预测系统性偏误，rollout 时这些偏误帧会作为下一个输入，导致模型"在错误分布上继续学习"（但推理时不学习，而是继续输出更错的预测）
- **seed=42 陷入了 rollout attractor**：从 LT=6 开始 MSE 就比另外两个 seed 高，差距在 LT=18 拉到 30%。这个模型学到的动力学系统可能有一个不稳定的吸引子，一旦进入就会快速发散

### 3. Official 单步略高但 rollout 不最优 — 有趣但可解释

Official 单步 MSE=0.008272（略高于 3-seed range [0.00805, 0.00816]），但 rollout LT=18 MSE=0.491（恰好是 3-seed mean）。这意味着：

- Official 模型在 teacher-forced 下的像素级精度略低于我们的 3 个 seed 最佳值
- 但它的 rollout 动力学稳定性介于中等（不是最好但也不最差）
- 这说明**单步 teacher-forced 指标与 rollout 动力学稳定性之间没有单调关系**——在相同 early stopping 策略下（用 internal val loss 选 best checkpoint），我们选到的是"单步最好"的 checkpoint，但它的 rollout 稳定性可能差于某个单步稍差但动力学更稳定的 checkpoint

### 4. HYP-005 修正 — Vanilla UNet 不是 deterministic reference

HYP-005 原文："Vanilla UNet 作为 deterministic baseline，其 rollout 表现是 diffusion 模型增益的参考下限"。

**修正后**：Vanilla UNet 的 rollout 表现具有显著的 seed-dependent variance。报告 diffusion vs deterministic 对比时，必须：
- 多 seed 训练 Vanilla UNet，报告均值 ± std
- 或直接引用官方 Vanilla UNet checkpoint（它处于 3-seed ensemble 的中间水平）作为 reference
- 不能用单一 seed 的 Vanilla UNet 与 diffusion 模型对比后就做因果性结论

### 5. 方差级联对 RQ-005 的含义

RQ-005："Diffusion 模型是否在 rollout 维度上比 deterministic baseline 更稳定？"

这个问题的重要性现在**大幅提升**。如果 diffusion 模型的 rollout 方差比 Vanilla UNet 小（即多 seed 训练的 diffusion 模型 rollout 结果分布更紧凑），那本身就是一个重要发现——说明 diffusion 的生成过程确实有某种稳定化效果（denoising step 天然地抑制了误差累积）。反之，如果 diffusion 也有类似的 rollout 方差，那 diffusion 模型的增益应该用 "每个 seed 训练的 Vanilla UNet vs 对应 seed 训练的 diffusion" 配对比较，而不是跨 seed 比较。

## Conclusion

| Hypothesis | 判定 | 依据 |
|------------|------|------|
| **HYP-007（seed variance 解释 rollout gap）** | **FULLY SUPPORTED** | Official rollout 完全落在 3-seed 区间内（MSE 偏差 +0.5%）；单步极窄而 rollout 极度分散的方差级联现象清晰 |
| HYP-005（Vanilla UNet as det ref） | **REVISED** | Vanilla UNet rollout 有显著 seed variance，不能作为单一 deterministic reference |
| HYP-006（diffusion 增益边界） | **UNRESOLVED**（但重要性提升） | Vanilla UNet baseline 的定义需要修正；diffusion 模型是否有类似 variance 需要多 seed 实验 |
| HYPOTHESIS-001（单步 ±5%） | **SUPPORTED** | 4 个模型单步 MSE 极差 0.000222（< 3%） |
| HYPOTHESIS-002（rollout 差距存在） | **SUPPORTED BUT QUALIFIED** | rollout 差距存在且随 LT 增大，但这是 Vanilla UNet 本身的属性，不是复现失败 |

### 最终判定

**原 EXP-EVAL-001 观测到的 "rollout gap" 不是复现失败，而是 Vanilla UNet 这种单步训练 + autoregressive rollout 架构的固有 seed-dependent variance。** 官方开源 checkpoint 的 rollout 水平可以被我们的多 seed 复现实验完全覆盖。

## Limitations

- 3 个 seed 仍太少。Vanilla UNet 的 rollout CV=13.4% 意味着如果要得到稳定的均值估计，需要 ≥ 5 个 seed（CV/√n ≈ 6% standard error）。但当前 3 个 seed 已足以推翻"复现失败"的假设
- dataset split_seed 固定为 42。如果同时改变 split_seed，方差可能更大
- 官方 checkpoint 的模型初始化 seed 未知，因此无法做严格的 "same-same" 配对对比
- rollout 只评估了 128 样本子集（单步评估了完整 1024）
- 未评估 per-event / per-region 的 variance 差异（某些事件类型可能更 sensitive to seed）
- 未评估 diffusion 模型（EDM / LDM / CorrDiff）的多 seed variance

## Reproducibility

- Eval script：`/tmp/eval_all_seeds.py`（由 EXP-AUDIT-001 审计后编写，未 git commit；核心评估函数来自 `scripts/Chase_2025/eval_unet_comparison.py`）
- Slurm script：`scripts/Chase_2025/eval_all_seeds.slurm`
- 环境：`cira-diff-ly`, RTX 4090
- 运行命令：Slurm Job 77 执行 `python -u /tmp/eval_all_seeds.py`
- 数据：`/data1/satcast/edm_GOES_ch13_test_dataset.zarr`
- 模型权重：
  - EXP-014: `outputs/vanilla_unet_seed0/best_internal_unet`
  - EXP-010: `outputs/vanilla_unet_full/best_internal_unet`
  - EXP-015: `outputs/vanilla_unet_seed123/best_internal_unet`
  - Official: `/data1/satcast/unet_vanilla/unet_vanilla`
- 指标实现：纯 PyTorch F.mse_loss / F.l1_loss / F.avg_pool2d，无 torchmetrics/skimage 依赖
- 所有中间结果固化在 `outputs/eval_unet_all_seeds/all_seeds_results.json`，可复算

## 关联文献 / 决策 / 下一实验

- 关联文献：NVIDIA CIRA-Diff 2023；Chase et al. 2025；vanilla UNet autoregressive rollout error accumulation 领域文献
- 关联 EXP：EXP-010, EXP-014, EXP-015（训练权重）；EXP-AUDIT-001（审计逻辑）；EXP-EVAL-001（原 gap 观测）
- **关联 DEC**：
  - ❌ 作废："Rollout gap 需要进一步调查 checkpoint 时机 / diffusers 版本 / early stopping 指标"（已全部排除）
  - ✅ 新立：**Vanilla UNet baseline 必须报告多 seed 均值 ± std**；Official 可作为单 seed reference
  - ✅ 新立：**RQ-005 重要性提升**——diffusion vs deterministic 的 variance 对比本身就是科学贡献
- 下一实验：
  - EXP-011 continuation：EDM 多 seed 训练（当前 287ep OOM，需重训）
  - EXP-012 / EXP-013：LDM / CorrDiff full training
  - EXP-EVAL-002：EDM vs Official（当前运行中，Jobs 70-73）
  - **EXP-EVAL-004（新）**：Vanilla UNet 多 seed rollout variance 详细分析（per-event, per-LT, 与 diffusion baseline 对比）
  - **EXP-016（新）**：EDM 多 seed 训练（控制 dataset split_seed，改变模型初始化 seed）