# EXP-015 — Vanilla UNet Seed 123 Full Training

## Identity

- 实验 ID：EXP-015
- 状态：**Completed**
- 日期：2026-09-28 — 2026-09-29（Slurm Job 60，完成时间 05:03 AM CST）
- 负责人：liuyi
- 关联 RQ：RQ-001 / RQ-005 / EXP-AUDIT-001-H7
- 关联 HYP：HYP-005 / H7（随机因素是否导致 rollout gap）
- 依赖：EXP-AUDIT-001（审计触发）；与 EXP-010 / EXP-014 / Official 对比见 EXP-EVAL-003

## Research question 与 hypothesis

- Research Question：更换模型初始化 seed 是否会改变 Vanilla UNet 的收敛轨迹和最终 rollout 表现？EXP-010 与官方开源 checkpoint 的 rollout gap 是否源于随机初始化差异？
- Hypothesis（H7，EXP-AUDIT-001 提出）：不同模型初始化 seed 的训练会收敛到不同的局部最优，rollout 指标存在显著方差；官方 checkpoint 的 rollout 水平落在多 seed 方差范围内 → gap 可由随机因素解释。
- Scientific motivation：同 EXP-014，与 EXP-014（seed=0）、EXP-010（seed=42）共同组成 3-seed ensemble，用于量化训练方差并对比官方 checkpoint。EXP-015 是 3-seed 中 rollout 表现介于中间的模型。

## Dataset

与 EXP-014 / EXP-010 **完全相同**，包括 split_seed=42。

- 训练集：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`（35595 原始样本）
- 独立验证集：`/data1/satcast/edm_GOES_ch13_validation_dataset.zarr`（1024 样本，不变）
- 内部 80/20 split from split_seed=42 → 28476 train / 7119 internal val
- Sample construction：(X[t-1], X[t]) (2ch input) → X[t+1] (1ch target)
- Normalization：mean=0.0, std=1.0
- Time period / Region / Sensor：GOES-16 ABI C13（亮温）

> **关键实验控制**：本实验仅改变 **模型初始化 seed=123**，dataset split_seed 固定为 42。train/val partition 和 DataLoader shuffle 顺序**完全不变**，变化的只有 nn.Module 参数初始化。

## Model 与 temporal semantics

与 EXP-014 / EXP-010 / Official 完全相同。UNet2DModel ~47.6M params，输入 2 帧 → 输出 1 帧，训练单步 MSE，无 scheduled sampling / rollout loss。

## Training configuration

与 EXP-014 / EXP-010 **完全相同**，唯一差异：seed=123。

| 参数 | 值 |
|------|-----|
| train_batch_size | 24 |
| gradient_accumulation | 2（effective = 48） |
| num_epochs | 210（上限） |
| **实际训练 epoch** | **41（early stopping 触发）** |
| learning_rate | 1e-4 |
| lr_warmup_steps | 500 |
| lr_scheduler | cosine_schedule_with_warmup |
| optimizer | AdamW |
| loss | MSELoss (reduction='none').mean() |
| mixed_precision | fp16 |
| early_stopping_patience | 10 |
| early_stopping_min_delta | 1e-06 |
| early_stopping_window_size | 5 |
| early_stopping_criterion | internal_val_moving_average |
| num_workers | 4 |
| pin_memory | True |
| **seed** | **123** |
| **split_seed** | **42（固定）** |

## Runtime 与 artifacts

- Command：
  ```bash
  SEED=123 \
  OUTPUT_DIR=/home/group1/26fall_aiclass/ly/cira-diff/outputs/vanilla_unet_seed123 \
  python -u scripts/Chase_2025/train_vanilla_unet_Chase2025.py
  ```
- Environment：conda env `cira-diff-ly`（Python 3.11.16, torch 2.6.0+cu124, diffusers 0.40.0, accelerate 1.10.1）
- Git commit：`cf67d6e`；branch：`feature/vanilla-unet-baseline-fix`
- Slurm Job：**60**（partition=debug, 1×RTX 4090 GPU 1, 4 CPU, 32G, walltime=0）
- 运行时长：约 12 小时（epoch 0..41，~18 min/epoch）
- Slurm script：`scripts/Chase_2025/train_vanilla_unet_seed123.slurm`
- Log：`outputs/slurm_logs/vanilla_unet_seed123_60.{out,err}`
- TensorBoard：`outputs/vanilla_unet_seed123/logs/`
- Output dir：`outputs/vanilla_unet_seed123/`
- Artifacts：
  - `training_state.json` — 训练状态与 loss history（41 epochs, 42 entries）
  - `run_metadata.json` — 运行配置快照
  - `best_internal_unet/` — best checkpoint at epoch 36
  - optimizer / random_states — 完整状态
  - `config.json` — 模型配置（diffusers 0.40.0）

## Results

### Training trajectory（关键 epoch）

| Epoch | train_loss | internal_val | independent_val | global_step | 事件 |
|-------|------------|--------------|-----------------|-------------|------|
| 0 | 0.022572 | 0.022572 | 0.022091 | 1187 | first best |
| 10 | 0.011541 | 0.011477 | — | 11870 | — |
| 20 | 0.011066 | 0.010349 | — | 23740 | — |
| **28** | — | 0.009773 | — | 34423 | best checkpoint saved |
| 30 | 0.007852 | 0.009822 | — | 36797 | — |
| **31** | — | 0.009767 | — | 37984 | best checkpoint saved |
| **36** | 0.007002 | **0.009671** | 0.009727 | 43919 | **best checkpoint saved** |
| 41 | 0.006489 | 0.009795 | 0.009874 | 49854 | **early stopping**（no improvement for 10 epochs） |

### 训练终止状态

- 实际训练：**41 epochs, 49854 global steps**
- early stopping 触发：**epoch 41**，连续 10 epochs 无 improvement
- best_internal_val：**0.009671** at **epoch 36**（3 个 seed 中**最低**）
- best_moving_val_loss：0.009832

### 与同批次 3-seed ensemble 对比

| Seed | 训练 Epochs | Best Epoch | best_internal_val | Rollout LT=18 MSE | Rollout LT=18 SSIM |
|------|------------|------------|-------------------|-------------------|--------------------|
| 0 (EXP-014) | 50 | 36 | 0.009969 | **0.419** | **0.241** |
| 42 (EXP-010) | 47 | 37 | 0.009728 | **0.574** | **0.103** |
| **123 (EXP-015)** | **41** | **36** | **0.009671** | **0.472** | **0.194** |
| Official | 48 | Unknown | Unknown | 0.491 | 0.238 |

### EXP-015 关键发现

**EXP-015（seed=123）的单步 internal_val 是 3 个 seed 中最低的**（0.009671），但 rollout LT=18 表现处于中间（MSE=0.472，介于 EXP-014 的 0.419 和 EXP-010 的 0.574 之间）。这说明：

1. **单步 internal_val 最优 ≠ rollout 最优**。这是一个极重要的观察——early stopping 用单步 loss 选 best checkpoint，但单步 loss 不能代表 rollout 动力学稳定性。
2. 3 个 seed 的单步指标排序（seed123 < seed42 < seed0）与 rollout 指标排序（seed0 < seed123 < seed42）**完全不一致**。

## Interpretation

1. EXP-015 与 EXP-014 / EXP-010 唯一差异是模型初始化 seed，但 early stopping 触发 epoch 差了 9 epochs（41 vs 50），说明初始化对训练早期收敛速度也有影响。
2. 单步 internal_val 排序和 rollout 排序不一致的发现意味着：**Vanilla UNet 训练时没有任何机制选择"rollout 最优"的 checkpoint**——early stopping 选的是单步 teacher-forced 最优的，但这个 checkpoint 的动力学稳定性可能很差。
3. 这解释了为什么 seed=42 恰好是 rollout 最差的：它的单步 internal_val 不是最差的（排第二），但它收敛到的模型恰好有一个不稳定的 rollout attractor。

## Conclusion

| Hypothesis | 判定 | 依据 |
|------------|------|------|
| H7（seed variance 解释 rollout gap） | **Contributes to FULL SUPPORT** | EXP-015 单步最优但 rollout 中间，展示了单步-rollout 脱耦 |
| HYP-005（Vanilla UNet as det ref） | **WEAKENED** | 单步最优 ≠ rollout 最优，说明 early stopping 指标与下游目标函数不一致 |

## Limitations

- 仅 1 个额外 seed；完整 3-seed 组由 EXP-010 + EXP-014 + EXP-015 组成
- dataset split_seed 固定为 42
- Official checkpoint 的真实 seed 未知
- 单步 internal_val 与 rollout 表现脱耦的发现还需要更多 seed 验证

## Reproducibility

- 训练脚本：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`
- Slurm script：`scripts/Chase_2025/train_vanilla_unet_seed123.slurm`
- 运行命令：见 Runtime 章节
- 环境：`cira-diff-ly`, RTX 4090 #1
- 数据路径：Dataset 章节
- Git commit：`cf67d6e`
- Best checkpoint：`outputs/vanilla_unet_seed123/best_internal_unet/diffusion_pytorch_model.safetensors`

## 关联文献 / 决策 / 下一实验

- 关联 EXP：EXP-010（seed=42）、EXP-014（seed=0）、EXP-AUDIT-001（审计逻辑）、EXP-EVAL-003（完整对比评估）
- 关联 DEC：Vanilla UNet baseline 必须报告多 seed；early stopping 指标应考虑 rollout 代理（如 val_rollout_loss）
- 下一实验：EXP-EVAL-003（已完成）、EXP-016+（diffusion baseline 多 seed）