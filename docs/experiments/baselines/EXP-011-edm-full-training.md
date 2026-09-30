# EXP-011 — EDM (Karras 2022) Full Training Reproduction

## Identity

- 实验 ID：EXP-011
- 状态：Running（Slurm Job 51）
- 开始日期：2026-09-23
- 负责人：liuyi
- 关联 RQ：RQ-005 / RQ-006 / RQ-007（diffusion 与 deterministic baseline 的 skill/uncertainty 对比）
- 关联 HYP：HYP-006 / HYP-007

## Research question 与 hypothesis

- Research Question：**复现 CIRA-Diff 论文中 EDM denoiser baseline 的完整训练**。Vanilla UNet（EXP-010）是 deterministic regression baseline；EDM 是 conditional diffusion denoiser。两者在相同数据/协议下对比，才能隔离出 diffusion 机制带来的增益。
- Hypothesis：[HYPOTHESIS] EDM 作为 diffusion 模型，其 single-step denoised MSE 应低于或接近 Vanilla UNet；ensemble 预测应改善 uncertainty calibration（HYP-006）；skill 改善可能集中在冷云/对流区域（HYP-007）。
- Scientific motivation：CIRA-Diff 论文将 CorrDiff 相对于 EDM 和 Vanilla UNet 的提升作为主要结论。没有 EDM baseline 的本地复现，无法判断 CorrDiff 的增益是来自 corrector 架构还是来自 diffusion 基础本身。

## Dataset

- Dataset ID / manifest：
  - 训练集：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`（完整 35595 样本，不切 80/20）
  - 独立验证集：`/data1/satcast/edm_GOES_ch13_validation_dataset.zarr`（1024 样本，仅用于监控，不参与 gradient）
- Time period：[UNKNOWN] 原始 GOES 日期区间
- Region：[UNKNOWN]
- Sensor/channel/unit：GOES-16 ABI Channel 13, Brightness Temperature (K)
- Spatial resolution / crop：256×256 patches from full disk
- Temporal resolution：10 min between consecutive frames
- Train / validation / test：35595 / 1024 / [UNKNOWN test zarr not yet located]
- Sample construction：input = [clean_t, cond_t-1, cond_t0] stacked → 3 channels; output = clean_t (1 channel)
- Normalization：mean=0, std=1（按 train zarr 统计）

## Model 与 temporal semantics

- Architecture：`EDMPrecond(UNet2DModel)` — Karras EDM preconditioning wrapper over a 6-block UNet2DModel
- Initialization/checkpoint：From scratch（论文未报告用何种初始化）
- Input frames：2 historical frames (t-10 min, t0) concatenated as conditions + 1 noisy target → total 3 channels
- Output frames：1 denoised target frame (t+10 min), 1 channel
- Forecast interval：10 min per autoregressive step
- Rollout length：3 h = 18 steps（论文 protocol）
- Teacher forcing：Not applicable to diffusion training
- Scheduled sampling：Not applicable
- Rollout training vs inference：Training is one-step noisy denoising; inference is iterative EDM sampler (default 18 steps) over the full 18-frame horizon

### EDM preconditioning details (from `scripts/Chase_2025/train_edm_Chase2025.py`)

```
c_skip = sigma_data² / (sigma² + sigma_data²)
c_out = sigma · sigma_data / √(sigma² + sigma_data²)
c_in  = 1 / √(sigma_data² + sigma²)
c_noise = log(sigma) / 4
```

- `generation_channels = 1`（只生成 target）
- `condition_channels = 2`（两帧历史）
- `in_channels = 3`, `out_channels = 1`

## Training configuration

- Loss：EDM loss = mean squared error between `c_skip·clean + c_out·F(c_in·x, c_noise)` and `clean`（log-normal noise sampling via `P_mean=-1.2, P_std=1.2`）
- Optimizer：AdamW, lr=1e-4
- Learning rate：cosine schedule with warmup (500 warmup steps)
- Batch size：train_batch_size=24, gradient_accumulation_steps=4 → **effective batch=96**
  - 论文指定 batch_size=45, grad_accum=2 → effective=90。本实验 96 vs 90 差 6.7%，因 45 在 48GB GPU 上 OOM 被迫调整
- Epoch budget：**fixed 1000 epochs, no early stopping**（严格按论文 protocol：完整 35595×1000 sample count ≈ 35.6 million）
- Precision：fp16 mixed precision via `accelerate`
- GPU：NVIDIA RTX 4090 48GB × 1（Slurm `--gres=gpu:1`，独占 1 张）
- Seed：0
- Config path：`scripts/Chase_2025/train_edm_Chase2025.py`（`TrainingConfig` dataclass at module level）

### ⚠️ Protocol deviations from paper

| 项 | 论文 | 本实验 | 原因 |
|----|------|--------|------|
| train_batch_size | 45 | 24 | 45 在 48GB GPU 上 OOM |
| gradient_accumulation | 2 | 4 | 补偿 batch 减小 |
| effective_batch | 90 | 96 | 96 vs 90（+6.7%），差异在合理范围 |

其余（1000 epochs、完整 training zarr、无 early stopping、lr=1e-4、warmup=500、P_mean/P_std/sigma_data、EDM preconditioning）全部与论文一致。

## Runtime 与 artifacts

- Command：
  ```bash
  sbatch scripts/Chase_2025/train_edm_full.slurm
  # 实际运行：python -u scripts/Chase_2025/train_edm_Chase2025.py
  ```
- Environment：`/home/group1/miniconda3/envs/cira-diff-ly`（Python 3.11, PyTorch 2.x, diffusers, accelerate, zarr, tensorboard）
- Git commit：`cf67d6e`（branch: `feature/vanilla-unet-baseline-fix`）
  - ⚠️ 工作区有未提交修改（见 `git diff`）：EDM slice bug 修复、batch size 调整、best checkpoint 追踪、run_metadata、lazy dataset loading
- Log：`outputs/slurm_logs/edm_full_51.out`（stdout）+ `.err`（tqdm）
- Output：`outputs/edm_full/`
- Checkpoint（每个 epoch 保存）：
  - `training_state.json` — epoch, global_step, train_loss, independent_val_loss, best_independent_val_loss, best_independent_epoch
  - `optimizer.bin`, `scheduler.bin`, `scaler.pt`, `random_states_*.pkl` — Accelerate state for resume
  - `diffusers/model.safetensors` — last epoch model weights (via `model.save_pretrained`)
  - `best_edm/model.safetensors` — ✅ 独立 val loss 最优 epoch 的模型副本
- Artifact/checksum：Not recorded yet（训练完成后补）

### Slurm script (`scripts/Chase_2025/train_edm_full.slurm`)

```bash
#SBATCH --job-name=edm_full
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
export OMP_NUM_THREADS=4
```

### Checkpoint/resume 行为

- Resume 触发条件：`optimizer.bin` + `training_state.json` 同时存在
- Resume 时 `accelerator.load_state(output_dir)` 恢复 optimizer/scheduler/scaler/RNG，从 `epoch+1` 继续
- `best_edm/` 子目录独立存放 best epoch 权重（不被 resume 依赖）
- NaN 保护：`training_state.json` 中 independent_val_loss 为 NaN 时跳过 best 比较

## Evaluation protocol

训练期间（每个 epoch 结束）：
- val_heldout_dataloader 在 sigma=0（完全 denoised）条件下跑独立 val zarr，报告 `independent_val_loss`（MSE）
- 这是 EDM "零噪声预测"的近似评估，不等价于完整 sampling pipeline

完整评估（训练完成后执行）：
- Metrics：MSE, MAE, SSIM, 可能的 structural/object metrics
- Lead times：10, 20, ..., 180 min（3 h）autoregressive rollout
- Ensemble：多 seed EDM sampler（如 5-10 seeds），评估 spread-skill / CRPS / reliability
- Thresholds/masks：冷云 / 深对流候选区域分层（待定义）
- 与 EXP-010 Vanilla UNet 在同一独立 val / test zarr 上对比

## Results

训练 **刚启动**（epoch 0 进行中，还未产生 checkpoint）。

运行时观测（Slurm Job 51, test-ai GPU 0, 2026-09-23）：
- GPU 显存：~48.3 GB / 48 GB
- GPU 利用率：100%
- 训练速度：~1.1 it/s
- 每 epoch：1484 batches → ~22.5 min
- 总预计：1000 epochs × 22.5 min ≈ **375 hours ≈ 15.6 days**

## Interpretation

待训练产生结果后填写。

## Conclusion

待训练完成后填写。

## Limitations

- Effective batch=96 vs paper 90（+6.7%），可能影响 convergence speed / final loss level。这是 OOM 约束下的折中，属于可接受的 canonical baseline 偏差。
- 训练速度 ~1.1 it/s 意味着 15+ 天才能完成 1000 epochs，与论文报告的训练时长一致（35.6M samples 量级）。
- Slurm 每个 job 独占 1 张 GPU，test-ai 有 8× 4090 共 7 张空闲，可并行跑 ablation（但 canonical baseline 必须先完成）。
- 工作区修改未提交到 Git，artifact traceability 暂时依赖 `run_metadata.json` 中的 git_commit（记录了 base commit `cf67d6e`，但未记录 diff）。

## Reproducibility

- 完整脚本：`scripts/Chase_2025/train_edm_Chase2025.py`（629 行）
- Slurm 提交：`scripts/Chase_2025/train_edm_full.slurm`
- 数据路径：硬编码在 `TrainingConfig.dataset_path` / `val_heldout_path`
- 环境：`cira-diff-ly` conda env

## 关联文献 / 决策 / 下一实验

- 关联论文：Karras et al. (2022) EDM, CIRA-Diff (Chase et al.)
- 关联决策：`docs/project/decisions.md` 中关于 canonical baseline 的规定
- 前置实验：EXP-001（数据审计）、EXP-009（单样本 overfit gate）
- 同期实验：EXP-010（Vanilla UNet，Slurm Job 52，GPU 1）
- 下一实验（EXP-012）：CorrDiff full training（待 EDM 和 Vanilla UNet 都完成后）