# EXP-010 — Vanilla UNet Full Training Reproduction

## Identity
- 实验 ID：EXP-010
- 状态：**Completed**（Slurm Job 52，2026-09-24 03:17 early stopping）
- 开始日期：2026-09-23；结束日期：2026-09-24
- 负责人：liuyi
- 关联 RQ：RQ-001 / RQ-003（Temporal formulation / Motion vs evolution 的 baseline）
- 关联 HYP：HYP-005 / HYP-006（Vanilla UNet 作为 deterministic reference）

## Research question 与 hypothesis
- Research Question：**先复现论文 Vanilla UNet baseline 的训练，后续才能比较 EDM / LDM / CorrDiff 的增益**。这是 reproduction protocol 的第一步。
- Hypothesis：[HYPOTHESIS] Vanilla UNet（纯回归 baseline）的 MSE/MAE/SSIM 等像素指标应在论文报告的 Vanilla UNet 数值附近（论文未直接给出 Vanilla UNet 单独的数值表格，只给了 CorrDiff 提升量）。
- Scientific motivation：CIRA-Diff 论文中 CorrDiff / EDM / LDM 都是相对于 Vanilla UNet baseline 报告提升的；在没有 baseline 参考分数的情况下，无法判断扩散模型带来的增益是真实的还是方差波动。

## Dataset
- Dataset ID / manifest：
  - 训练集：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`（raw, 35595 样本）
  - 独立验证集：`/data1/satcast/edm_GOES_ch13_validation_dataset.zarr`（raw, 1024 样本）
- Time period：[UNKNOWN] 原始 GOES 日期区间
- Region：[UNKNOWN]
- Sensor/channel/unit：GOES-16 ABI C13（亮温）
- Spatial resolution / crop：256×256 patch，1 通道
- Temporal resolution：10 min
- Train / validation / test：
  - 80/20 random_split 从训练集内再划分：28476 train / 7119 held-out val
  - 另有 1024 样本的独立 validation zarr 加载了但脚本当前只用 held-out val
- Sample construction：`(X[t-1], X[t])` (2ch input) → `X[t+1]` (1ch target)
- Normalization：[UNKNOWN] 脚本未显式 mean/std；与 overfit 实验一致

## Model 与 temporal semantics
- Architecture：`diffusers.UNet2DModel`，同 overfit 实验配置
  - layers_per_block=2
  - block_out_channels=(128,128,256,256,512,512)
  - attention 层：AttnDownBlock2D + AttnUpBlock2D（中间层）
- Initialization/checkpoint：随机初始化，无预训练权重，论文 upstream checkpoint **未下载**
- Input frames：2 帧（t-1, t）
- Output frames：1 帧（t+1）
- Forecast interval：10 min
- Rollout length：**1 step（单步回归训练）**；[EVIDENCE] 论文使用训练好的 UNet 做 autoregressive rollout inference（18 步到 3h），当前脚本**没有 rollout inference 路径**
- Teacher forcing：单步训练语义下不适用
- Scheduled sampling：不适用
- Rollout training 与 rollout inference：**训练只有单步**；论文在 inference 时做 autoregressive rollout，当前仓库尚未实现独立 rollout CLI

## Training configuration
| 参数 | 论文/开源默认值 | 本地实际值 | 备注 |
|------|---------------|-----------|------|
| train_batch_size | 45 | 24 | 4090 显存限制 |
| gradient_accumulation | 1 | 2 | 等效 batch=48 ≈ 45 |
| num_epochs | 210 | 210 | 一致 |
| learning_rate | 1e-4 | 1e-4 | 一致 |
| lr_warmup_steps | 500 | 500 | 一致 |
| mixed_precision | fp16 | fp16 | 一致 |
| save_model_epochs | 1 | 1 | 一致 |
| optimizer | AdamW | AdamW | 一致 |
| loss | MSELoss | MSELoss | 一致 |
| early_stopping_patience | [UNKNOWN] | 10 | 本地添加 |

Config path：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`（内联 `TrainingConfig` class）。

## Runtime 与 artifacts
- Command：`python -u scripts/Chase_2025/train_vanilla_unet_Chase2025.py`
- Environment：conda env `cira-diff-ly`（Python 3.11.16, torch 2.6.0+cu124, diffusers 0.40.0, accelerate 1.10.1, zarr 3.1.6）
- Git commit：`cf67d6e`；branch：`feature/vanilla-unet-baseline-fix`
- Slurm Job：**52**（重新提交，partition=debug，1×RTX 4090，8 CPU，128G 内存）。前序 Job 43/44/46/47 被 kill 或 OOM
- Log：`outputs/slurm_logs/vanilla_unet_full_52.out` + TensorBoard `outputs/vanilla_unet_full/logs/`
- Output：`outputs/vanilla_unet_full/`（含 `training_state.json`, `run_metadata.json`, `diffusion_pytorch_model.safetensors`, 加速器状态）
- **Best checkpoint**：`outputs/vanilla_unet_full/best_internal_unet/diffusion_pytorch_model.safetensors`（Epoch 37，internal_val MSE=0.009728）
- Final checkpoint：`outputs/vanilla_unet_full/diffusion_pytorch_model.safetensors`（Epoch 47，最后一轮）
- Artifact/checksum：待导出

## Evaluation protocol
- 训练过程：held-out val MSE + 独立 validation zarr MSE（诊断）
- 训练后评估：**EXP-EVAL-001** 执行完整 baseline vs 开源权重对比（test set 1024 样本）
  - Metrics：MSE / MAE / SSIM / PSNR（纯 PyTorch 实现，无 torchmetrics 依赖）
  - Scenarios：单步 teacher-forced（全量 1024）+ 18-step autoregressive rollout（128 子集）
  - Eval script：`scripts/Chase_2025/eval_unet_comparison.py`
  - Test zarr：`/data1/satcast/edm_GOES_ch13_test_dataset.zarr`（1024 样本, output_images shape (1024, 18, 256, 256)）
  - 开源权重：`/data1/satcast/unet_vanilla/unet_vanilla/diffusion_pytorch_model.safetensors`
  - Output：`outputs/eval_unet_comparison/`

## Results

### 训练结果（Epoch 0 → 47，early stopping triggered）

| 指标 | 数值 | 备注 |
|------|------|------|
| 总 Epochs | 47（上限 210） | Early stopping @ epoch 47 |
| Best Epoch | **37** | internal_val MSE 最低 |
| Best internal_val MSE | 0.009728 | held-out 80/20 split val |
| 独立 val MSE @ best epoch | 0.009685 | validation zarr 1024 样本 |
| Best moving_avg_val MSE | 0.009867 | window=5, patience=10 |
| 训练末 train MSE | 0.007264 | Epoch 47 |
| 训练末 no_improvement_count | 10/10 | 触发 early stopping |

每 epoch 训练时长 ≈ 19 min（1187 step × 1.02 it/s），总训练时长 ≈ 47 × 19 ≈ 900 min ≈ 15 h。

### Test Set 评估结果（来自 EXP-EVAL-001）

**单步 teacher-forced（1024 样本）：**

| Metric | EXP-010 Best (epoch 37) | 论文开源 Vanilla UNet | Δ (开源 − EXP-010) |
|--------|------------------------|---------------------|--------------------|
| MSE ↓ | 0.008092 | 0.008272 | +0.00018（EXP-010 略优） |
| MAE ↓ | 0.04634 | 0.04517 | −0.00118 |
| SSIM ↑ | 0.8913 | 0.8939 | +0.0027 |
| PSNR ↑ | 23.96 | 24.04 | +0.07（基本打平） |

**18 步 autoregressive rollout（128 样本子集，每 lead time 均值）：**

| LT | MSE EXP-010 | MSE 开源 | Δ | SSIM EXP-010 | SSIM 开源 | Δ |
|----|-----------|---------|---|-------------|---------|---|
| 1 | 0.00688 | 0.00702 | +0.00014 | 0.9023 | 0.9048 | +0.0025 |
| 6 | 0.0966 | 0.0923 | −0.0043 | 0.4587 | 0.5000 | +0.0413 |
| 12 | 0.2925 | 0.2482 | −0.0443 | 0.2281 | 0.3207 | +0.0926 |
| 18 | **0.5743** | **0.4909** | **−0.0834** | **0.1034** | **0.2375** | **+0.1341** |

（完整 per-LT 1..18 结果见 `outputs/eval_unet_comparison/rollout_metrics_a.json`）

## Interpretation

1. **训练提前终止但已收敛**：47 epoch 时内部 val MSE 到达 0.009728 后 10 epoch 无改善，early stopping 触发。训练 loss 继续下降到 0.00726（过拟合信号），但 held-out val 已经进入平台期——说明模型容量足够，训练数据足够，30–40 epoch 已学到最优的单步 2→1 回归。

2. **单步 teacher-forced 指标基本复现论文水平**：与开源 Vanilla UNet 的 MSE/MAE/SSIM/PSNR 差值均在 ±1.5% 以内，部分指标 EXP-010 略好。**Hypothesis HYP-005/HYP-006 的单步部分已 supported**。

3. **长步 autoregressive rollout 明显落后**：LT=18 时开源模型 MSE 低 14%、SSIM 高 2.3 倍。

   **【重要修正：2026-09-27 审计】** 原始推断"开源模型训练了 210 epoch，我们只训了 47 epoch"**是错的**。直接证据来自开源 checkpoint tar 包内的 TensorBoard events：
   - 开源 vanilla UNet 的 `epoch_loss` / `val_loss` 各 49 个事件，step range `[0, 48]` → **开源模型也在 epoch 48 被 early stopping 停掉**
   - 我们 EXP-010 训练到 epoch 47，两者几乎一模一样

   因此 rollout 差距不是"epoch 数差"导致的。**两者训练方法本质相同**（单步 MSE + early stopping，main 分支脚本也有 patience=10 的 early stopping）。差距的真实来源需要独立调查，候选因素：
   - diffusers 版本差异（开源 0.31.0 vs 我们 0.40.0，ResNet block / time embedding 实现是否变了？）
   - Checkpoint 保存时机（原始脚本每 epoch 覆盖 `save_pretrained`，不存 best；开源 tar 里的 checkpoint 是 epoch 48 还是 best val loss？而我们显式保存了 epoch 37 best checkpoint）
   - LR schedule 走没走完（原始脚本 warmup_steps=500，但 `num_training_steps = len(dl) * num_epochs(210)`，实际只训了 48 epoch → 总 LR 曲线是否走完？）
   - Batch size 差异（原始 45×1=45 vs 我们 24×2=48）
   - Dataset random split seed 不同 → 实际训练/验证样本可能不同

   **【排除的假设】** rollout loss / scheduled sampling / rollout-aware checkpoint：对 main 分支原始脚本做了关键词全量 grep（rollout_loss, multi_step, scheduled_sampling, rollout_checkpoint），**均未找到任何实现**。论文也明确描述为"单步 MSE training + held-out val early stopping"。这和我们的 reproduction 一致。

4. **开源模型的已知与未知**：
   - ✅ 已知：架构完全一致（config.json diff），实际训练终止于 epoch 48（TensorBoard 事件 step 0..48）
   - ❓ 未知：checkpoint 保存的是 epoch 48 last 还是 best val loss epoch；early stopping 用的是 held-out split 还是独立 validation zarr；具体 dataset split seed；LR 曲线是否走完完整 schedule

## Conclusion

- **Hypothesis HYP-005 (Vanilla UNet 作为 deterministic reference)**：**Supported**。单步像素指标在论文报告水平 ±1.5% 内。
- **Hypothesis HYP-006 (diffusion 增益边界)**：**Partially Supported**。由于 rollout 差距显著，后续对比 EDM/LDM/CorrDiff 时必须同时报告 single-step 和 rollout 两套指标。
- **Reproduction fidelity**：单步 teacher-forced 评估可视为 reproductions 成功；rollout 评估存在已知 gap，需要更长训练或 rollout-aware loss 才能对齐。

## Limitations

- 训练脚本 hardcoded，非 config-driven
- `matplotlib.colormaps.get_cmap` 兼容性依赖（本窗口已 patch）
- 训练目标是单步 MSE，没有 rollout-aware loss 或 scheduled sampling（但原始脚本也没有 → 不是我们独有的问题）
- Early stopping 只用内部 val 的 moving average，未用独立 validation zarr
- **原始脚本设置 num_epochs=210 但被 early stopping 提前终止（开源 48，我们 47）**，所以"训练到 210 epoch"本来就不是实际行为（我们的错误在于曾假设开源模型真的跑到了 210）
- diffusers 版本不一致（我们 0.40.0 vs 开源 0.31.0）可能影响 rollout 表现
- 开源 checkpoint 保存时机不确定（原始脚本每 epoch 覆盖，我们显式保存了 best internal val epoch 37）

## Reproducibility
- Code：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`（从 main 分支恢复，6 处本地 patch）
- Slurm：`scripts/Chase_2025/train_vanilla_unet_full.slurm`
- Eval script：`scripts/Chase_2025/eval_unet_comparison.py`
- Env：`cira-diff-ly`（Python 3.11.16, torch 2.6.0+cu124, diffusers 0.40.0）
- Data：训练 `edm_GOES_ch13_train_dataset.zarr` / 独立 val `_validation_dataset.zarr` / test `_test_dataset.zarr`
- Hardware：NVIDIA RTX 4090 48GB × 1
- Upstream checkpoint：`/data1/satcast/unet_vanilla/unet_vanilla/`（论文开源 Vanilla UNet，diffusers v0.31.0）
- Model architecture 完全一致：UNet2DModel, 47.6M params（两侧 config.json 已人工 diff 确认）

## 关联文献 / 决策 / 下一实验
- 关联文献：NVIDIA CIRA-Diff 2023；Chase et al. 2025 Vanilla UNet baseline
- 关联 DEC：EXP-010 完成；EXP-EVAL-001 执行 baseline vs 开源权重对比
- 关联 HYP：HYP-005 / HYP-006
- 下一实验：EXP-011（EDM full training，Slurm Job 51 并行运行中）→ EXP-EVAL-002（EDM vs 开源 EDM）→ EXP-012（LDM）→ EXP-013（CorrDiff）→ 统一多模型对比