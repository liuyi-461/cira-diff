# EXP-013 — CorrDiff (Correction Diffusion) Full Training Reproduction

## Identity

- 实验 ID：EXP-013
- 状态：**Running**（Slurm Job 94，epoch 13/1000，started 2026-09-30 ~12:03）
- 日期：2026-09-30
- 负责人：liuyi
- 关联 RQ：RQ-005（diffusion vs deterministic baseline）
- 关联 HYP：CorrDiff 的残差学习收敛动力学是否与 EDM（EXP-011）不同？残差空间 val loss 的收敛速度 vs EDM 的直接 denoising val loss
- 依赖：无（CorrDiff 原始训练脚本 + EDM 训练脚本的 robust resume 机制移植）

## Research question 与 hypothesis

- **Research Question**：复现 NVIDIA StormCast 论文中 CorrDiff baseline 的完整 1000 epoch 训练。CorrDiff 是在 Vanilla UNet forecast 基础上学习 residual correction（forecast error = truth - UNet forecast），本质是一个条件扩散模型。与 EDM（直接 denoise 真实亮温）的核心差异在于：CorrDiff 的输出空间是 residual（mean≈0, std≈0.08）而非 clean image（mean=0, std=1），训练信号的量级小一个数量级。
- **Hypothesis**：
  1. CorrDiff 的残差学习收敛速度可能快于 EDM 的直接 denoising（残差空间的 signal-to-noise ratio 更高？）
  2. 但残差空间的 val loss 收敛 plateau 可能更高（因为 residual 本质是 forecast error，存在 irreducible noise）
  3. CorrDiff 训练完成后，单步 forecast correction MSE 应低于原始 UNet forecast MSE
- **Scientific motivation**：EXP-EVAL-001 揭示 Vanilla UNet reproduction 与开源 baseline 存在显著 rollout gap（14% MSE），且 gap 源于架构本质差异而非 epoch 数。EXP-EVAL-002 揭示 EDM reproduction gap=54% optimizer step 不足，可修复。CorrDiff 作为 forecast correction 范式，其 reproduction gap pattern 是 forecast correction 机制能否泛化的关键证据。

## Dataset

- **Dataset ID / manifest**：
  - Train：`/data1/satcast/edm_GOES_ch13_train_dataset_CorrDiff.zarr`
  - Independent Val：`/data1/satcast/edm_GOES_ch13_val_dataset_CorrDiff.zarr`（从 train zarr 切 5% holdout）
- **Time period**：Unknown（继承自 EDM train zarr 的时间范围）
- **Region**：Unknown
- **Sensor/channel/unit**：GOES-16 ABI C13（亮温）
- **Spatial resolution / crop**：256×256，fixed crop
- **Temporal resolution**：Unknown
- **Train / validation / test split**：
  - Train：35,595 samples（95% of original train zarr）
  - Independent Val：1,779 samples（5% holdout，从原始 train zarr 切分）
  - Test：CorrDiff 无专用 test zarr，评估时使用 EDM val zarr（需在 eval 脚本中构建 CorrDiff condition）
- **Sample construction**：每个 sample = (3ch UNet forecasts, 1ch residual target)，residual = truth - UNet_forecast
- **Normalization**（残差专用，已在数据预构建时应用）：
  - mean = -0.0009，std = 0.0807（**与 EDM 的 mean=0, std=1 完全不同**）

## Model 与 temporal semantics

- **Architecture**：`EDMPrecond(UNet2DModel)` — 与 EDM 相同的 Karras 2022 EDM preconditioning wrapper
- **Initialization/checkpoint**：从头训练（`accelerator_state_exists=False`, `resume=False`）
- **UNet 配置**：
  - in_channels = **4**（1 noisy residual + 3 condition channels）
  - out_channels = 1
  - layers_per_block = 2
  - block_out_channels = (128, 128, 256, 256, 512, 512)
  - down_block_types = ("DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
  - up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")
- **Condition channel semantics**（来自原始脚本 line 314 确认）：
  - ch0, ch1, ch2：3 个不同时间步的 UNet forecast
  - **残差重建公式**：`full_forecast = CorrDiff_residual_output + condition_ch2`
  - 训练数据中 3 个 channel 高度相关（pairwise corr > 0.998）
- **Forecast interval**：Unknown
- **Rollout length**：训练阶段无 rollout（仅 single-step denoising loss）
- **Teacher forcing**：训练阶段使用 UNet forecast 作为 condition（非 autoregressive）
- **Scheduled sampling**：None
- **Rollout training vs rollout inference**：CorrDiff 训练无 rollout；评估时需要 autoregressive rollout（每步 residual + 当前 forecast = next forecast → 作为下一步 condition）

## Training configuration

| 项 | 论文/原始 Chase2025 | EXP-013 实际 | 偏差 |
|----|---------------------|--------------|------|
| **Loss** | EDM weighted L2（Karras 2022） | 同左 | 无 |
| **Optimizer** | AdamW | AdamW | 无 |
| **Learning rate** | 1e-4 | 1e-4 | 无 |
| **LR schedule** | cosine_schedule_with_warmup | cosine_schedule_with_warmup | 无 |
| **LR warmup steps** | 500 | 500 | 无 |
| **train_batch_size** | 45 | **22** | ↓ 49% |
| **gradient_accumulation** | 2 | **4** | ↑ 100% |
| **effective_batch** | 90 | **88** | ↓ 2.2% |
| **Epoch/step budget** | 1000 epochs | 1000 epochs | 无 |
| **Precision** | fp16 | fp16 | 无 |
| **GPU** | GH200 (?) | **RTX 4090 48GB × 1** | 硬件不同 |
| **Seed** | 0 | 0 | 无 |
| **num_workers** | 8 | **4** | ↓ 50%（保守设置） |
| **EDM P_mean** | -1.2 | -1.2 | 无 |
| **EDM P_std** | 1.2 | 1.2 | 无 |
| **EDM sigma_data** | 0.5 | 0.5 | 无 |
| **数据加载** | **全量 CPU** (`ds[:]`) | **懒加载** (`__getitem__`) | 防 OOM |
| **Resume 机制** | 手写 `restart` flag + `load_checkpoint` | `accelerator.load_state` + `training_state.json` | robust resume |
| **Val loss** | 无（仅 early stopping on train loss） | **有**（独立 val zarr） | 改进（但目前 NaN bug） |
| **Early stopping** | patience=100 on train loss | 已移除 | 有 val loss 后不需要 |
| **Slurm Job ID** | N/A | **94** | — |
| **Git commit** | N/A | **cf67d6e** | — |
| **Git branch** | N/A | feature/vanilla-unet-baseline-fix | — |

- **Epoch steps**：35,595 / 22 ≈ 1,618 steps/epoch
- **速度**：~1.40 it/s → ~19.3 min/epoch → 1000 epochs ≈ **13.4 days**
- **总 global_step**：1,618 × 1000 = 1,618,000 optimizer steps

## Runtime 与 artifacts

- **Command**：`python -u scripts/Chase_2025/train_edm_CorrDiff_Chase2025.py`
- **Slurm 脚本**：`scripts/Chase_2025/train_corrdiff_full.slurm`
- **Slurm 资源**：partition=debug, gpu:1, cpu=4, mem=32G, exclude=ai01,ai02
- **Environment**：conda `cira-diff-ly`, diffusers 0.31.0, PyTorch
- **Git commit**：`cf67d6e`
- **Git branch**：`feature/vanilla-unet-baseline-fix`
- **Log**：
  - Slurm stdout：`outputs/slurm_logs/corrdiff_full_94.out`
  - Slurm stderr：`outputs/slurm_logs/corrdiff_full_94.err`
  - TensorBoard：`outputs/corrdiff_full/logs/corrdiff_train/`
- **Output**（`/home/group1/26fall_aiclass/ly/cira-diff/outputs/corrdiff_full/`）：
  - `run_metadata.json` — 完整训练元数据（git commit, dataset sizes, architecture 配置等）
  - `training_state.json` — 最新 epoch 状态
  - `config.json` — diffusers 模型配置
  - `diffusion_pytorch_model.safetensors` — 最新模型权重
  - `model.safetensors` — 最新模型权重（另一份）
  - `optimizer.bin` — optimizer state
  - `scaler.pt` — fp16 gradient scaler
  - `scheduler.bin` — LR scheduler state
  - `random_states_0.pkl` — 随机数状态
- **Checkpoint 格式**：diffusers format + accelerator save_state（两者同时保存）
- **Best checkpoint**：`outputs/corrdiff_full/best_corrdiff/`（目前尚未产生，val loss NaN）
- **Artifacts/checksum**：
  - `diffusion_pytorch_model.safetensors`: 454,736,492 bytes (~434 MB)
  - `optimizer.bin`: 909,754,520 bytes (~868 MB)

## Code modifications（从原始 Chase2025 到 EXP-013）

修改 diff 总计 659 行，核心改动 5 处：

1. **硬编码路径替换**
   - `output_dir`: `/mnt/data1/rchas1/edm_10_CorrDiff_TEST/` → `/home/group1/26fall_aiclass/ly/cira-diff/outputs/corrdiff_full/`
   - `dataset_path`: `/mnt/data1/rchas1/diffusion_10_4_2inputs_v3_gh200_CorrDiff.zarr` → `/data1/satcast/edm_GOES_ch13_train_dataset_CorrDiff.zarr`
   - 新增 `val_heldout_path`: `/data1/satcast/edm_GOES_ch13_val_dataset_CorrDiff.zarr`

2. **全量 CPU 加载 → 懒加载**（修复 OOM 风险）
   - 删除原始 `ZarrDataset` 中的 `self.input_images = torch.tensor(self.data['input_images'][:], ...)`
   - 改用 `__getitem__` 按需从 zarr 读取
   - 同时将 `float16` 改为 `float32`（懒加载 + float16 可能导致 precision loss）

3. **手写 checkpoint → accelerator.load_state robust resume**
   - 删除原始 `restart` flag + 手写 `load_checkpoint` / `save_checkpoint`
   - 移植 EDM 训练脚本的 `training_state.json` + `accelerator.save_state` / `accelerator.load_state` 机制
   - 自动检测 `optimizer.bin` + `training_state.json` 是否存在决定 resume

4. **新增 independent val loss + val-based best checkpoint**
   - 新增 `ZarrDatasetVal` 类 + `val_heldout_dataloader`
   - 训练循环中每个 epoch 后计算 val loss
   - val loss 改善时保存 `best_corrdiff/` 子目录

5. **Batch size OOM 修复**
   - `train_batch_size`: 45 → 22
   - `gradient_accumulation_steps`: 2 → 4
   - effective batch: 90 → 88（仅差 2.2%）

**训练循环参数传递方式也重新组织**：原始 `train_loop(config, model, optimizer, dataset, lr_scheduler)` → 改为 `train_loop(config, model, optimizer, train_dataloader, lr_scheduler, val_heldout_dataloader=..., run_metadata=...)`

## Runtime metrics（实时，epoch 13）

| 指标 | 值 | 说明 |
|------|----|------|
| epoch 速度 | ~19.3 min/epoch | |
| iteration 速度 | ~1.40 it/s | |
| elapsed wall time | 4h38min for 13 epochs | |
| 预估剩余 | ~13 days | |
| train_loss（epoch 均值） | 1.56 (ep0) → 0.81 (ep12) | 下降中 |
| train_loss（实时） | 0.69 @ step 22,217 | |
| independent_val_loss | **NaN**（始终） | **需要调查** |
| global_step | 22,217 @ epoch 13 | |
| lr | 0.0001（warmup 已结束） | |

### TB scalar 历史

| Tag | entries | last value |
|-----|---------|------------|
| `loss` | 5,000 | 0.6897 @ step 22,217 |
| `lr` | 5,000 | 0.0001 @ step 22,217 |
| `epoch_loss` | 13 | 0.8105 @ epoch 12 |
| `independent_val_loss` | 13 | **nan** @ epoch 12 |
| `epoch` | 26 | 12 @ epoch 12 |

**independent_val_loss NaN 根因分析**（待修复）：val loss 计算循环可能有 shape mismatch（`ZarrDatasetVal.__getitem__` 中 `output_images[idx, 0]` 切片方式 vs `unsqueeze(0)`）。训练不受影响，但 best checkpoint 选择失效。

## Evaluation protocol

评估尚未进行，计划使用 `scripts/Chase_2025/eval_corrdiff_comparison.py`：

- **Metrics**：MSE, MAE, SSIM, PSNR（残差空间 + 重建 forecast 空间双报告）
- **Lead times**：CorrDiff 无 rollout（单步 teacher-forced）；若做 rollout 需特殊处理
- **Ensemble members**：10（Euler sampler 10 次取平均）
- **Sampling params**：EDM 默认 (num_steps=36, sigma_min=0.002, sigma_max=140, rho=4, S_churn=7.2)
- **Comparison targets**：
  - UNet baseline（无 correction）
  - EXP-013 reproduced CorrDiff
  - Official CorrDiff（`/data1/satcast/edm_corrdiff/edm_corrdiff/checkpoint.pth`）

## Results

训练尚未完成。中间结果：

- Train loss 从 ~1.56 (ep0) 下降到 ~0.81 (ep12)，下降速率正常
- **Val loss NaN bug**：best checkpoint 选择暂不可用

## Interpretation

**训练前 13 epoch 的 loss 收敛速度**：从 1.56 → 0.81 仅用 13 epoch，对应 EDM 的前 13 epoch train loss 应该在类似范围（EDM 的 train_loss 从 ~1.2 降到 ~0.1 用了 287 epoch）。CorrDiff 残差空间 loss 收敛速度是否比 EDM 的直接 denoising 更快，需要等到更多 epoch 才能判断。

## Conclusion

Hypothesis 尚未 tested（训练未完成）。

## Limitations

1. **Val loss NaN bug**（已知问题）：训练循环中的 independent val loss 计算有 shape bug，导致始终 NaN。影响：
   - best checkpoint 选择失效
   - 无法用 val loss 判断过拟合
   - 修复方案：检查 `ZarrDatasetVal.__getitem__` 返回 shape vs loss 计算循环中的 target shape
2. **硬件差异**：原始脚本在 GH200 上训练，EXP-013 在 RTX 4090 48GB 上。batch size 从 45 降到 22 是主要调整（effective 88 vs 90，差异仅 2.2%，影响很小）
3. **num_workers 不同**：原始 8 vs EXP-013 4，可能影响 I/O 效率但不影响训练结果
4. **数据加载方式改变**：从全量 CPU 加载改为懒加载，zarr 随机读取可能略慢但更稳
5. **无 early stopping**：原始脚本有 patience=100 on train loss 的 early stopping，EXP-013 移除了它（改为 val loss，但 val loss NaN）
6. **GPU 显存利用**：RTX 4090 48GB 训练 CorrDiff batch=22 时占用约 46GB，接近上限

## Reproducibility

- Training script：`scripts/Chase_2025/train_edm_CorrDiff_Chase2025.py`（修改了 659 行）
- Slurm script：`scripts/Chase_2025/train_corrdiff_full.slurm`
- run_metadata.json：`outputs/corrdiff_full/run_metadata.json`（完整元数据）
- Git commit：`cf67d6e`
- Evaluation script：`scripts/Chase_2025/eval_corrdiff_comparison.py`（新建，未跑）

## 关联文献 / 决策 / 下一实验

- **关联文献**：NVIDIA StormCast 论文（CorrDiff 原始方法）、Karras et al. 2022（EDM preconditioning）
- **决策记录**：EXP-013 启动决策见 registry.md
- **下一实验**：
  1. 修复 independent_val_loss NaN bug（训练过程中）
  2. 训练完成后运行 EXP-EVAL-004（CorrDiff baseline vs opensource 评估）
  3. CorrDiff vs EDM vs Vanilla UNet 的完整 gap pattern 对比分析
## Bug fix: independent_val_loss NaN → sigma=0.002 workaround

**根因**（line 338）：原始 val loss 计算用 `sigma = torch.zeros(...)`。EDMPrecond.forward 中 `c_noise = sigma.log() / 4 = -inf`，UNet timestep embedding 收到 -inf → NaN。

**修复**：`sigma = 0.002`（EDM sampler 的 sigma_min）。

**副作用**：sigma=0.002 远低于 EDM 训练 sigma 范围（P_mean=-1.2, P_std=1.2 → sigma ∈ [0.09, 1.0]）。此 sigma 下 c_out ≈ 0，pred ≈ c_skip * clean ≈ clean，所以 val loss 极小（2.42e-08）但**不代表模型质量高**。best checkpoint 选择在此 sigma 下无区分度。

**正确做法（TODO）**：val loss 应该：
- 每个 batch 随机采样 sigma ~ N(P_mean, P_std)（跟训练一样）
- 计算 EDM weighted L2 loss（跟训练 loss_fn 一样）
- 或者用完整 EDM sampler 从纯噪声开始 denoise → 跟 clean 算 MSE

当前 workaround 至少让 val loss 不再 NaN、TB 有日志，但 best checkpoint 暂不具备科学意义。
