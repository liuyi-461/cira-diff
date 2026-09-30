# Vanilla UNet 全量训练交接文档

> 日期：2026-09-23
> 本窗口职责：验证环境 → smoke test → 单样本 overfit gate check → Vanilla UNet 全量训练 → 发现问题 → 中间 checkpoint 验证 → 交接新窗口

---

## 一、环境与硬件

| 项目 | 值 |
|------|-----|
| conda env | `cira-diff-ly` |
| Python | 3.11.16 |
| PyTorch | 2.6.0+cu124 |
| diffusers | 0.37.2 |
| accelerate | 1.10.1 |
| zarr | 2.24.4 |
| 硬件 | NVIDIA RTX 4090 48GB × 1 |
| GPU 实际显存占用 | fp16 + bs=24 → **47.4 GB**（稳定，无 OOM） |

---

## 二、数据集位置

| zarr | 路径 | shape | 说明 |
|------|------|-------|------|
| train | `/data1/satcast/edm_GOES_ch13_train_dataset.zarr` | input=(35595, 2, 256, 256), output=(35595, 1, 256, 256) | 单步训练数据 |
| validation | `/data1/satcast/edm_GOES_ch13_validation_dataset.zarr` | input=(1024, 2, 256, 256), output=(1024, **18**, 256, 256) | output 多 17 帧 rollout ground truth |
| train latent | `/data1/satcast/edm_GOES_ch13_latent_train_dataset.zarr` | input=(N, 2, 4, 64, 64), output=(N, 1, 4, 64, 64) | 给 LDM 用 |

**验证集 output_images[:, 0]** 是单步 target，**output_images[:, 1:]** 是后续 17 帧 ground truth（rollout 用）。

---

## 三、当前脚本修改清单（相对于 main 分支）

文件：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`，共 6 处改动：

### 改动 1：数据路径（第 48-51 行）
```python
# 原始：zarr_store 是相对路径
zarr_store = '/data1/satcast/edm_GOES_ch13_train_dataset.zarr'     # 改
dataset = ZarrDataset(zarr_store)
val_zarr_store = '/data1/satcast/edm_GOES_ch13_validation_dataset.zarr'  # 新增
val_dataset_heldout = ZarrDataset(val_zarr_store)                        # 新增
```

### 改动 2：batch_size 和梯度累积（第 59-87 行）
```python
_BS = 24                                      # 原始 45 → 24（4090 显存限制）
# ...
train_batch_size = 24                         # 同上
gradient_accumulation_steps = 2                # 原始 1 → 2，等效 batch=48 ≈ 论文 45
```

**显存测试历史**：bs=45 OOM → bs=32 OOM → bs=24 + grad_accum=2 稳定 47.4GB

### 改动 3：output_dir（第 92 行）
```python
output_dir = ".../outputs/vanilla_unet_full/"  # 原始 "vanilla_unet"
```

### 改动 4：matplotlib 兼容
```python
matplotlib.colormaps.get_cmap('magma')         # 原始 matplotlib.cm.get_cmap（新版 deprecated）
```

### 改动 5：Config 中 train_batch_size 字段保留但 DataLoader 硬编码 _BS=24
（因为 config 类定义在 DataLoader 之后，引用顺序导致 NameError）

### 改动 6：加载了独立 val zarr 但 train_loop 没用到（**BUG**）
- 第 50-51 行创建了 `val_dataset_heldout`
- 第 69-71 行创建了 `val_heldout_dataloader`
- 但 `train_loop(config, model, optimizer, train_dataloader, lr_scheduler, val_dataloader)` 只接收 `val_dataloader`（来自 train_dataset 内 80/20 的 7119 样本）
- **`val_heldout_dataloader` 完全没用上**

---

## 四、关键发现：验证集有两个，脚本只用到一个

| 名称 | 来源 | 数量 | 脚本是否用 | 用途 |
|------|------|------|-----------|------|
| held-out val | train_dataset 内 `random_split(0.8, 0.2)` | 7119 | ✅ 用了 | early stopping + val_loss 打印 |
| 独立 val zarr | `validation_dataset.zarr` 单独文件 | 1024 | ❌ **没用到** | 论文说的真正 held-out validation |

**问题**：80/20 split 的 val 和训练数据来自同一时间区间，存在时间相关性，early stopping 可能"提前停"。

---

## 五、中间 checkpoint 验证结果

Job 46 跑了 ~16 epoch 后手动停止，checkpoint 在独立 val zarr 上的表现：

### 位置
```
outputs/vanilla_unet_full/diffusion_pytorch_model.safetensors  (434 MB, 第 16 epoch)
outputs/vanilla_unet_full/config.json
outputs/eval_quick/validation_visualization.png  ← 4 张样本对比图
outputs/eval_quick/metrics.npz                   ← 完整 MSE/MAE 数组
```

### 指标（1024 样本，单步 teacher-forced）
| 指标 | Mean | Std | Median |
|------|------|-----|--------|
| MSE | **0.01028** | 0.01080 | 0.00664 |
| MAE | **0.05623** | 0.03030 | 0.04902 |

分布：Best MSE=0.00012（idx 147），Worst MSE=0.088（idx 191），725× 差距。

### 评估脚本
```bash
source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly
python -u scripts/Chase_2025/eval_vanilla_unet_quick.py
# 脚本自动加载 outputs/vanilla_unet_full checkpoint + validation zarr
```

---

## 六、训练速度与预计时长

| 指标 | 值 |
|------|-----|
| Train step 速度 | ~1.1 it/s |
| Val step 速度 | ~3.4 it/s |
| Train steps / epoch | 1187（28476 / 24） |
| Val steps / epoch | 297（7119 / 24） |
| **每 epoch 耗时** | **≈ 20 min** |
| **210 epoch 总耗时** | **≈ 70 小时（~3 天）** |
| Early stopping | patience=10（用的是 80/20 split val，不是独立 val zarr） |

---

## 七、Slurm 脚本

文件：`scripts/Chase_2025/train_vanilla_unet_full.slurm`（**当前版本申报过多资源**）

```bash
#!/bin/bash
#SBATCH --job-name=vanilla_unet_full
#SBATCH --output=outputs/slurm_logs/vanilla_unet_full_%j.out
#SBATCH --error=outputs/slurm_logs/vanilla_unet_full_%j.err
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/group1/26fall_aiclass/ly/cira-diff
python -u scripts/Chase_2025/train_vanilla_unet_Chase2025.py
```

### 资源浪费审计

| 资源 | 申报 | 实际 | 浪费率 |
|------|------|------|--------|
| CPU | 8 核 | ~1 核（Python 主进程 + DataLoader 短暂活跃） | 87.5% |
| 内存 | 128 GB | **17.6 GB**（RES） | 86% |

### 建议改小
```bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
```
（保留 DataLoader 4 worker 余量 + Zarr chunk + TensorBoard 缓冲）

---

## 八、已知问题（新窗口需修复）

### BUG-1：独立 val zarr 没用上（GAP-012）
- **现象**：`val_heldout_dataloader` 被创建但 train_loop 从不调用
- **修复**：给 `train_loop` 加一个可选参数 `val_heldout_dataloader`，每个 epoch 额外跑一次独立 val zarr 的 eval，打印两个 val_loss。early stopping 应该用这个。
- **改动量**：~15 行

### BUG-2：没有 checkpoint resume
- **现象**：只保存 `save_pretrained`（模型权重 + config.json），不保存 optimizer/scheduler/RNG state
- **后果**：训练中断必须从头开始
- **修复**：保存时同时调用 `accelerator.save_state(output_dir)`；加载前检查 `accelerator/` 子目录是否存在，存在则 `accelerator.load_state()` + 恢复 epoch/global_step
- **改动量**：~10 行

### BUG-3：Resource request 过大（8 CPU / 128 GB）
- **现象**：实测只用 17.6 GB 内存
- **修复**：改成 4 CPU / 32 GB

### BUG-4：没有 normalization / validation split 与论文对齐
- 论文说："training set split 80/20 train/held-out"
- 当前：脚本做了 80/20 split（对），但独立 val zarr 也加载了（论文的 held-out validation 应该就是这个 1024 样本）
- 需确认：原始论文的 Vanilla UNet early stopping 用的是哪个？

---

## 九、训练命令（新窗口直接用）

### 选项 A：直接用当前脚本重跑（已知 BUG 都在）
```bash
sbatch scripts/Chase_2025/train_vanilla_unet_full.slurm
```
预估 70h，early stopping 用的是 80/20 split val，可能提前停。

### 选项 B：修完 BUG-1 + BUG-2 + BUG-3 再跑（推荐）
改完脚本和 slurm 后提交：
```bash
# 改 1：train_loop 加独立 val zarr eval
# 改 2：accelerator.save_state / load_state
# 改 3：slurm --cpus-per-task=4 --mem=32G
sbatch scripts/Chase_2025/train_vanilla_unet_full.slurm
```

---

## 十、smoke test 和 overfit 结果（已完成，无需重跑）

| 测试 | 命令 | 结果 |
|------|------|------|
| 50 样本 smoke | `python -u scripts/Chase_2025/train_vanilla_unet_Chase2025.py` (改 max_samples=50, epochs=2) | ✅ 通过 |
| 单样本 overfit | `python -u scripts/Chase_2025/run_overfit_comparison.py` | ✅ 四方法都通过；Vanilla UNet MSE=3.95e-05（最低） |

---

## 十一、参考模板（新窗口提交 baseline 可复用）

Slurm script 模板（其他 EDM/LDM/CorrDiff 训练脚本只需改第 4 行的 python 命令）：

```bash
#!/bin/bash
#SBATCH --job-name=<model_name>
#SBATCH --output=outputs/slurm_logs/<model_name>_%j.out
#SBATCH --error=outputs/slurm_logs/<model_name>_%j.err
#SBATCH --partition=debug         # debug 无时间限制
#SBATCH --gpus-per-node=1         # 4090 48GB
#SBATCH --cpus-per-task=4         # DataLoader 4 worker
#SBATCH --mem=32G                 # 实测用 17.6GB

source /home/group1/miniconda3/etc/profile.d/conda.sh
conda activate cira-diff-ly
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/group1/26fall_aiclass/ly/cira-diff
python -u <训练脚本路径>
```

---

## 十二、需要更新的脚本（快速审计）

| 脚本 | 需要的改动 | 备注 |
|------|-----------|------|
| **train_vanilla_unet_Chase2025.py** | ✅ 已改（6 处，含 BUG-1~3） | 本文档重点 |
| **train_edm_Chase2025.py** | 同样 6 处 patch（数据路径、bs、matplotlib、独立 val、save_state、slurm 资源） | 新窗口做 |
| **train_ldm_Chase2025.py** | 同上 + 用 latent zarr | 新窗口做 |
| **train_corrdiff_Chase2025.py** | 同上 + 依赖 vanilla_unet checkpoint 做 prior | 需 vanilla UNet 训完 |

---

## 十三、文档更新索引（本窗口已写入）

| 文件 | 内容 |
|------|------|
| `docs/experiments/registry.md` | 注册 EXP-009（overfit）、EXP-010（full training） |
| `docs/experiments/gate_checks/EXP-009-single-sample-overfit.md` | 四方法单样本 gate check 正式实验卡 |
| `docs/experiments/baselines/EXP-010-vanilla-unet-full-training.md` | Vanilla UNet 全量训练正式实验卡（状态 Running → 中间 checkpoint 验证） |
| `docs/project/current_state.md` | 补充环境就绪、EXP-009/010 状态 |
| `docs/project/knowledge_gaps.md` | GAP-012 独立 val zarr 没用、GAP-013~015 latent 量纲、rollout、VAE 对齐 |
| `docs/task_reports/VANILLA_UNET_HANDOVER.md` | 本文档 |

---

## 十四、artifact 位置汇总

```
outputs/vanilla_unet_full/diffusion_pytorch_model.safetensors  ← 当前 Epoch 16 checkpoint
outputs/vanilla_unet_full/config.json                           ← UNet 结构
outputs/vanilla_unet_full/logs/                                 ← TensorBoard
outputs/slurm_logs/vanilla_unet_full_46.out                     ← Job 46 stdout
outputs/slurm_logs/vanilla_unet_full_46.err                      ← Job 46 stderr（空）
outputs/eval_quick/validation_visualization.png                 ← 独立 val zarr 可视化
outputs/eval_quick/metrics.npz                                  ← MSE/MAE 数组
scripts/Chase_2025/train_vanilla_unet_full.slurm                ← Slurm 脚本
scripts/Chase_2025/eval_vanilla_unet_quick.py                   ← 评估脚本（新窗口可复用）
```

---

## 十五、新窗口立即要做的事（优先级排序）

1. **修 BUG-1**：让 train_loop 同时评估 held-out val（80/20 split）和独立 val zarr（1024 样本），early stopping 用后者
2. **修 BUG-2**：加 `accelerator.save_state` / `load_state` 实现断点续训
3. **修 BUG-3**：slurm 脚本资源改成 4 CPU / 32 GB
4. **重跑 Vanilla UNet**（带上以上修复）
5. **给 EDM / LDM 训练脚本做同样 patch**
6. **提交 EDM / LDM baseline 训练任务**
7. Vanilla UNet 跑完后 → **实现 rollout evaluate.py**（当前只有单步 teacher-forced，论文核心的 18-step autoregressive rollout 还没做）
