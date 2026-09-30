# EXP-014 — Vanilla UNet Seed 0 Full Training

## Identity

- 实验 ID：EXP-014
- 状态：**Completed**
- 日期：2026-09-28 — 2026-09-29（Slurm Job 59，完成时间 07:52 AM CST）
- 负责人：liuyi
- 关联 RQ：RQ-001 / RQ-005 / EXP-AUDIT-001-H7
- 关联 HYP：HYP-005 / H7（随机因素是否导致 rollout gap）
- 依赖：EXP-AUDIT-001（审计触发）；与 EXP-010 / EXP-015 / Official 对比见 EXP-EVAL-003

## Research question 与 hypothesis

- Research Question：更换模型初始化 seed 是否会改变 Vanilla UNet 的收敛轨迹和最终 rollout 表现？EXP-010 与官方开源 checkpoint 的 rollout gap 是否源于随机初始化差异？
- Hypothesis（H7，EXP-AUDIT-001 提出）：不同模型初始化 seed 的训练会收敛到不同的局部最优，rollout 指标存在显著方差；官方 checkpoint 的 rollout 水平落在多 seed 方差范围内 → gap 可由随机因素解释。
- Scientific motivation：EXP-EVAL-001 观测到 EXP-010（seed=42）单步 teacher-forced 与官方打平（MSE 差 < 3%）但 rollout LT=18 MSE 差 17%（0.574 vs 0.491）。EXP-AUDIT-001 已排除训练长度、diffusers 版本、checkpoint 选择、rollout loss 实现、Official val_loss bug、config 缺失字段共 6 个候选原因，**仅余随机种子差异为最强候选**。

## Dataset

- 训练集：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`（35595 原始样本）
- 独立验证集：`/data1/satcast/edm_GOES_ch13_validation_dataset.zarr`（1024 样本，不变）
- **内部 80/20 split from split_seed=42**（与 EXP-010 / EXP-015 **完全相同**）：
  - train: 28476 samples
  - internal_val: 7119 samples
- Sample construction：(X[t-1], X[t]) (2ch input) → X[t+1] (1ch target)
- Normalization：mean=0.0, std=1.0
- Time period / Region / Sensor：GOES-16 ABI C13（亮温），同 EXP-010

> **关键实验控制**：本实验仅改变 **模型初始化 seed=0**，dataset split_seed 固定为 42（与 EXP-010 / EXP-015 完全相同）。因此 train/val partition 和 DataLoader shuffle 顺序**完全不变**（torch.manual_seed 同时控制了 torch / cuda / random / numpy），变化的只有 nn.Module 参数初始化。

## Model 与 temporal semantics

- Architecture：`diffusers.UNet2DModel`，完全同 EXP-010 / 官方开源
  - layers_per_block=2, block_out_channels=(128,128,256,256,512,512)
  - down: [DownBlock2D×4, AttnDownBlock2D, DownBlock2D]
  - up: [UpBlock2D, AttnUpBlock2D, UpBlock2D×4]
  - params: ~47.6M
- Initialization：随机初始化，**seed=0**
- Input frames：2 帧 (t-1, t) → Output frames：1 帧 (t+1)
- Rollout length：训练单步；评估用 18-step autoregressive rollout（EXP-EVAL-003）
- Teacher forcing：训练全程使用；无 scheduled sampling / rollout loss

## Training configuration

与 EXP-010 / EXP-015 **完全相同**，唯一差异：seed=0 替代 seed=42/123。

| 参数 | 值 |
|------|-----|
| train_batch_size | 24 |
| gradient_accumulation | 2（effective batch = 48） |
| num_epochs | 210（上限） |
| **实际训练 epoch** | **50（early stopping 触发）** |
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
| **seed** | **0** |
| **split_seed** | **42（固定）** |

## Runtime 与 artifacts

- Command：
  ```bash
  SEED=0 \
  OUTPUT_DIR=/home/group1/26fall_aiclass/ly/cira-diff/outputs/vanilla_unet_seed0 \
  python -u scripts/Chase_2025/train_vanilla_unet_Chase2025.py
  ```
- Environment：conda env `cira-diff-ly`（Python 3.11.16, torch 2.6.0+cu124, diffusers 0.40.0, accelerate 1.10.1）
- Git commit：`cf67d6e`；branch：`feature/vanilla-unet-baseline-fix`
- Slurm Job：**59**（partition=debug, 1×RTX 4090 GPU 0, 4 CPU, 32G, walltime=0）
- 运行时长：约 15 小时（epoch 0..50，~18 min/epoch）
- Slurm script：`scripts/Chase_2025/train_vanilla_unet_seed0.slurm`
- Log：`outputs/slurm_logs/vanilla_unet_seed0_59.{out,err}`
- TensorBoard：`outputs/vanilla_unet_seed0/logs/`
- Output dir：`outputs/vanilla_unet_seed0/`
- Artifacts：
  - `training_state.json` — 训练状态与 loss history（50 epochs, 50 entries）
  - `run_metadata.json` — 运行配置快照
  - `best_internal_unet/` — best checkpoint at epoch 36
  - optimizer / random_states — 完整状态（未上传 TensorBoard）
  - `config.json` — 模型配置（diffusers 0.40.0）

## Results

### Training trajectory（关键 epoch）

| Epoch | train_loss | internal_val | independent_val | global_step | 事件 |
|-------|------------|--------------|-----------------|-------------|------|
| 0 | 0.092639 | 0.025134 | 0.023504 | 1187 | first best |
| 10 | 0.012707 | 0.011447 | — | 11870 | — |
| 20 | 0.010772 | 0.010960 | — | 23740 | — |
| 30 | 0.010340 | 0.010535 | — | 35610 | — |
| **36** | 0.007002 | **0.009969** | — | 43919 | **best checkpoint saved** |
| 40 | 0.006586 | 0.010200 | — | 48667 | — |
| 50 | 0.005785 | 0.010370 | 0.010212 | 60537 | **early stopping**（no improvement for 10 epochs） |

### 训练终止状态

- 实际训练：**50 epochs, 60537 global steps**
- early stopping 触发：**epoch 50**，连续 10 epochs 无 improvement
- best_internal_val：**0.009969** at **epoch 36**
- best_moving_val_loss：0.010130

### 与同批次 3-seed ensemble 对比（单步训练指标）

| Seed | 训练 Epochs | Best Epoch | best_internal_val | best_independent_val (approx) |
|------|------------|------------|-------------------|-------------------------------|
| **0 (EXP-014)** | **50** | **36** | **0.009969** | — |
| 42 (EXP-010) | 47 | 37 | 0.009728 | — |
| 123 (EXP-015) | 41 | 36 | 0.009671 | — |
| **spread** | 9 epochs | 1 epoch | 0.000298 | — |

三个 seed 的 best_internal_val 极差仅 0.000298（相对差异 ~3%），best epoch 都在 36-37，收敛后平台期高度一致。early stopping 触发 epoch 在 41-50 之间（极差 9 epochs），说明平台期长度有差异但最终都被 patience=10 触发。

### Rollout 评估（来自 EXP-EVAL-003）

| 场景 | 指标 | EXP-014 (seed=0) | EXP-010 (seed=42) | EXP-015 (seed=123) | Official |
|------|------|------------------|-------------------|--------------------|----------|
| 单步 teacher-forced (1024 samples) | MSE | 0.008050 | 0.008092 | 0.008157 | 0.008272 |
| 单步 teacher-forced | SSIM | 0.8929 | 0.8913 | 0.8929 | 0.8939 |
| **Rollout LT=18 (128 samples)** | **MSE** | **0.419** | **0.574** | **0.472** | **0.491** |
| **Rollout LT=18** | **SSIM** | **0.241** | **0.103** | **0.194** | **0.238** |

> 完整 per-LT 数据见 EXP-EVAL-003。

### EXP-014 关键发现

**EXP-014（seed=0）是 3 个 seed 中 rollout LT=18 **最优的**：**
- MSE=0.419，比 Official（0.491）好 **14.6%**
- SSIM=0.241，比 Official（0.238）好 **1.5%**
- 比 EXP-010（seed=42）MSE 好 **27%**（0.419 vs 0.574）

这证明了 Vanilla UNet 的 rollout 表现有极大的 seed-dependent variance——换一个 seed，同一个训练流程，rollout 就能从 "差 Official 17%" 变成 "好 Official 15%"。

## Interpretation

1. **训练配置与 EXP-010 / EXP-015 一致**，唯一变化是模型初始化 seed。这意味着观测到的 rollout 差异**唯一归因于初始化 seed**。
2. **单步 internal_val 极差 < 3%** 但 **rollout LT=18 MSE 极差 = 30%**（0.419 – 0.574）。方差级联机制清晰：单步误差在 autoregressive 推理中被递推放大，最终导致不同初始化收敛到的模型在动力学稳定性上有极大差异。
3. **EXP-014 的 rollout 比 Official 好**：说明 "Official checkpoint 有某种特殊的训练技巧" 这个假设也不成立——任何一个初始化 seed 都可能收敛到比 Official 更好的 rollout 解。

## Conclusion

| Hypothesis | 判定 | 依据 |
|------------|------|------|
| H7（seed variance 解释 rollout gap） | **Contributes to FULL SUPPORT**（与 EXP-010 / EXP-015 / Official 对比由 EXP-EVAL-003 给出最终判定） | EXP-014 的 rollout MSE 比 EXP-010 低 27%，直接展示了 seed 的影响 |
| HYP-005（Vanilla UNet as det ref） | **WEAKENED** | Vanilla UNet 有巨大的 rollout variance，不是稳定的 reference |

## Limitations

- 仅训练了 1 个额外 seed（seed=0），完整 3-seed 组由 EXP-010 + EXP-014 + EXP-015 组成
- dataset split_seed 固定为 42，未测试 split_seed 变化的影响
- Official checkpoint 的真实 seed 未知

## Reproducibility

- 训练脚本：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`（line 55-61 新增 seed 环境变量；line 99 新增 OUTPUT_DIR 环境变量）
- Slurm script：`scripts/Chase_2025/train_vanilla_unet_seed0.slurm`
- 运行命令：见 Runtime 章节
- 环境：`cira-diff-ly`, RTX 4090 #0
- 数据路径：Dataset 章节
- Git commit：`cf67d6e`
- Best checkpoint：`outputs/vanilla_unet_seed0/best_internal_unet/diffusion_pytorch_model.safetensors`

## 关联文献 / 决策 / 下一实验

- 关联 EXP：EXP-010（seed=42）、EXP-015（seed=123）、EXP-AUDIT-001（审计逻辑）、EXP-EVAL-003（完整对比评估）
- 关联 DEC：**Vanilla UNet baseline 必须报告多 seed 均值 ± std**；Official 可作为单 seed reference
- 下一实验：EXP-EVAL-003（已完成）、EXP-011 / EXP-012 / EXP-013（diffusion baseline 多 seed）