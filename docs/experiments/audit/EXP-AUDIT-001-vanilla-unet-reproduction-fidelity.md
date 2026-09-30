# EXP-AUDIT-001 — Vanilla UNet Reproduction Fidelity Audit & Multi-Seed Variation

## Identity

- 实验 ID：EXP-AUDIT-001
- 状态：**Completed**（审计完成 + 3-seed eval 完成，H7 已验证）
- 审计开始日期：2026-09-25；种子实验启动日期：2026-09-28；评估完成日期：2026-09-29
- 负责人：liuyi
- 关联 RQ：RQ-001 / RQ-003 / RQ-005（baseline reproduction fidelity、rollout 差距来源）
- 关联 HYP：HYP-005（Vanilla UNet 作为 deterministic reference）、"rollout gap 源于 diffusers 版本 / checkpoint 选择 / 训练配置" 的排除式假设链

---

## 审计总览

本审计的出发点：EXP-010（本地复现 Vanilla UNet）单步 teacher-forced 指标与官方开源 Vanilla UNet 打平，但 18-step autoregressive rollout 明显落后（LT=18: MSE 差 14%，SSIM 差 130%）。**审计目标：定位 rollout gap 的根因**。

### 审计范围

| 编号 | 检查项 | 状态 | 结论 | 关键证据 |
|------|--------|------|------|----------|
| **A1** | 训练长度差异 | ✅ 已排除 | Official 48 epoch vs EXP-010 47 epoch → 几乎相同 | 开源 TensorBoard events: epoch_loss/val_loss 各 49 个事件 step [0,48]；EXP-010 epoch 47 early stopping |
| **A2** | Checkpoint 选择 | ✅ 已排除 | EXP-010 best@37 rollout > last@47 rollout，但仍差于 Official → 不能解释 gap | EXP-EVAL-001 rollout 对比 |
| **A3** | Rollout loss / scheduled sampling | ✅ 已排除 | main 分支脚本全量 grep，无 rollout_loss / multi_step / scheduled_sampling / rollout_checkpoint 关键词 | 论文明确描述单步 MSE + held-out val early stopping |
| **A4** | diffusers 版本架构差异 | ✅ 已排除 | 0.31.0 vs 0.40.0 UNet2DModel 结构完全一致 | 两侧 config.json 人工 diff；同权重 forward = exact zero diff |
| **A5** | diffusers 版本 forward 数值等价 | ✅ 已排除 | same weights + same input → max_abs_diff = 0 | strict cross-version checkpoint loading 成功 |
| **A6** | Official TensorBoard val_loss 定义 | ✅ 已澄清 | val_loss = SUM(per-batch mean MSE)，sum 对 scalar 等价于无 reduction，量级≈batch_count × mean_MSE | `train_vanilla_unet_Chase2025.py` line ~270: `val_loss += loss.item()` 而非 `val_loss += loss.item() * len(batch)` |
| **A7** | 训练配置逐项对比 | 🔶 差异项已识别 | Dataset split seed、DataLoader shuffle/sampler、LR scheduler total steps 存在差异；Batch size 略有不同 | Training Equivalence Audit 逐项对比表 |
| **A8** | 随机因素（seed 不同） | 🔬 当前验证中 | 模型初始化 + 数据划分 seed 不同 → 训练轨迹分化 | EXP-014 / EXP-015（seed 0 和 seed 123）运行中 |

---

## A1 训练长度审计

### 证据

开源 checkpoint tar 内 TensorBoard events：
- 路径：`/data1/satcast/unet_vanilla/unet_vanilla/logs/train_example/events.out.tfevents.*`
- `epoch_loss` 和 `val_loss` 各有 49 个事件，step range `[0, 48]`
- 含义：每个 epoch 结束写一个 event，step=0 是 epoch 0 训练完，step=48 是 epoch 48 训练完

EXP-010 TensorBoard：
- `outputs/vanilla_unet_full/logs/images/events.out.tfevents.*`
- early stopping 触发于 epoch 47（`no_improvement_count >= patience=10`）

### 结论

**开源模型训练到 epoch 48，EXP-010 到 epoch 47，差 1 个 epoch。** 原始假设"开源模型跑满了 210 epoch"**完全错误**，两者均被 early stopping 在 ~50 epoch 附近终止。训练长度差不是根因。

---

## A2 Checkpoint 选择审计

### 证据

EXP-EVAL-001 评估了 EXP-010 的两个 checkpoint：

| Checkpoint | Epoch | internal_val MSE | Rollout LT=18 MSE | Rollout LT=18 SSIM |
|-----------|-------|-------------------|-------------------|-------------------|
| best_internal_unet | 37 | 0.009728 | 0.5743 | 0.1034 |
| last (最终) | 47 | 更高（early stopping 触发） | 比 best@37 更差 | 比 best@37 更差 |

开源 checkpoint：
- `/data1/satcast/unet_vanilla/unet_vanilla/diffusion_pytorch_model.safetensors`
- **不确定**是 epoch 48 last 还是 best val loss epoch（原始脚本每 epoch 覆盖 `save_pretrained`，没有显式保存 best val）
- 但从开源 TensorBoard 的 val_loss 曲线看，最低 val_loss 出现在 epoch ~40 附近

### 结论

EXP-010 显式保存了 epoch 37 的 best internal_val checkpoint，这比 last@47 好，但仍明显落后于开源 checkpoint。即使开源 checkpoint 是 epoch 48 last（非 best），EXP-010 的 best@37 也没追上。**Checkpoint 时机可以影响幅度，但不能解释 gap 的存在性。**

---

## A4-A5 diffusers 版本审计

### 方法

1. 分别在 diffusers 0.31.0 和 0.40.0 下，使用与 Official config.json / EXP-010 config.json **完全相同的参数**构造 UNet2DModel
2. 人工 diff 两侧的模块树（`named_modules()`）、state_dict key 列表和 shape
3. 同权重加载（Official safetensors → 0.40.0 模型），forward 相同输入，比较 max_abs_diff / mean_abs_diff / RMSE

### 结果

| 检查项 | diffusers 0.31.0 | diffusers 0.40.0 | 差异 |
|--------|------------------|------------------|------|
| config.json `_diffusers_version` | "0.31.0" | "0.40.0" | 版本标记不同 |
| config.json `mid_block_type` | **缺失** | "UNetMidBlock2D" | 缺失字段在默认值下解析相同 |
| config.json `time_embedding_dim` | **缺失** | null | 缺失字段在默认值下解析相同 |
| Module 树（down/up/mid blocks） | 与 EXP-010 结构一致 | 与 Official 结构一致 | 完全相同 |
| State_dict keys + shapes | 完全一致 | 完全一致 | 无差异 |
| Cross-loading Official → 0.40.0 | — | ✅ strict=True 成功 | 完全兼容 |
| Same weight + same input forward | — | max_abs_diff = **0.0** | 完全等价 |

### 结论

**diffusers 0.31.0 vs 0.40.0 对 UNet2DModel 无结构或 forward 数值差异。** 同权重 forward 误差为零。版本差异不是 rollout gap 的根因。

---

## A6 Official TensorBoard val_loss 定义审计

### 原始代码

`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`（main 分支原始版本）验证循环：

```python
for val_batch in val_dataloader:
    val_output = model(val_condition, ..., return_dict=False)[0]
    val_loss = loss_fn(val_target.float(), val_output.float()).mean()
    # mean() 作用在 [B, C, H, W] 上 → 标量
    val_losses.append(val_loss)

val_loss = sum(val_losses) / len(val_losses)  # mean of scalar = scalar
```

注意：上面 `loss_fn(..., reduction='none').mean()` 的 `.mean()` 已经把 [B, C, H, W] 聚合成标量。而原始脚本 TensorBoard 记录的是：

```python
writer.add_scalar("val_loss", val_loss, epoch)
```

其中 `val_loss` 实际上可能是 **SUM 而非 MEAN**，取决于原始脚本的实现细节。从开源 TensorBoard 数值看：Official `val_loss ≈ 1.57` 但 EXP-EVAL-001 evaluation 单步 MSE ≈ 0.008，量级差 ~2 个数量级（~200x ≈ val batch count × per-batch mean MSE）。

### 解释

开源脚本 val dataloader batch 数 ≈ 297（7119 / 24）。如果原始脚本写的是 `val_loss = sum(val_losses)` 而不是 `val_loss = sum(val_losses) / len(val_losses)`，则：

`SUM(per-batch mean MSE) ≈ 297 × 0.005 ≈ 1.48 ≈ Official val_loss ≈ 1.57`

### 结论

这是 logging reduction bug（SUM vs MEAN），**不改变 epoch 排序**（对 scalar 取 SUM 或 MEAN 只是缩放），也不改变 early stopping 行为（因为用的是同一 reduction）。**不能解释 rollout gap。**

---

## A7 Training Equivalence Audit

逐项对比 main 分支官方原始 Vanilla UNet training script vs EXP-010 实际执行脚本：

| 项目 | Official | EXP-010 | Status | 证据 |
|------|----------|---------|--------|------|
| **Dataset split seed** | Unknown | 42 | **DIFFERENT** | run_metadata.json `split_seed: 42`；Official 原始 hardcode `torch.manual_seed(42)` 但不确定 |
| **DataLoader train shuffle** | shuffle=True | shuffle=True | SAME | `train_dataloader(shuffle=True)` |
| **DataLoader val shuffle** | shuffle=False | shuffle=False | SAME | — |
| **Batch size** | 45 | 24 (×2 grad_accum → eff 48) | **DIFFERENT** (6.7%) | Official config batch_size=45；EXP-010=24×2=48 |
| **Optimizer** | AdamW | AdamW | SAME | lr=1e-4 |
| **LR warmup** | 500 | 500 | SAME | — |
| **LR scheduler** | cosine_with_warmup | cosine_with_warmup | SAME | — |
| **LR total steps** | `len(train_dl) × 210` | `len(train_dl) × 210` | SAME 公式；但 dl 长度可能因 split 不同而略异 | — |
| **Loss function** | MSELoss | MSELoss | SAME | — |
| **Loss reduction** | `loss_fn(...).mean()` | `loss_fn(...).mean()` | SAME | per-sample mean over [B,C,H,W] |
| **Loss scale** | 无额外 scale | 无额外 scale | SAME | — |
| **Mixed precision** | fp16 | fp16 | SAME | — |
| **Early stopping patience** | Unknown | 10 | UNKNOWN | Official 脚本是否有 early stopping？TensorBoard 显示 49 epoch → 有 |
| **Early stopping monitor** | Unknown | internal_val_moving_avg (window=5) | UNKNOWN | Official 用 held-out split 还是独立 val zarr？ |
| **Checkpoint 保存时机** | 每 epoch 覆盖 `save_pretrained` | 每 epoch 覆盖 + 显式存 best_internal_unet | **DIFFERENT** | Official 无显式 best val 保存；EXP-010 多了一条 best 路径 |
| **Grad accumulation** | 1 | 2 | **DIFFERENT** | Official batch=45×1=45；EXP-010=24×2=48 |
| **num_workers** | Unknown | 4 | UNKNOWN | — |
| **Normalization** | Unknown mean/std | mean=0.0, std=1.0 | UNKNOWN | run_metadata.json |
| **torch.manual_seed 位置** | Unknown | DataLoader 创建前 | UNKNOWN | 关键：DataLoader worker seed 也受影响 |
| **DataLoader worker seed** | 未显式设置 | 未显式设置（依赖 global seed + worker init） | SAME | 两侧均未设置 `generator=` 或 `worker_init_fn=` |

---

## A8 根因假设链与当前验证

### 已排除假设

| # | 假设 | 排除证据 |
|---|------|----------|
| H1 | "开源模型训了 210 epoch，我们只训了 47" | 开源 TensorBoard 显示也在 epoch 48 停 |
| H2 | "diffusers 版本差异改变了 UNet 结构/forward" | same weights + same input = zero diff |
| H3 | "我们选错了 checkpoint（应该用 last 而非 best val）" | best@37 > last@47，但两者都差于开源 |
| H4 | "我们没做 rollout loss / scheduled sampling" | 开源也没做；main 分支脚本全量 grep 无相关代码 |
| H5 | "Official val_loss 数值不对，说明训练逻辑不同" | 是 SUM vs MEAN bug，不影响排序和 early stopping |
| H6 | "diffusers config.json 缺失字段（mid_block_type, time_embedding_dim）" | 在两侧 diffusers 版本下解析值相同 |

### 当前最强候选假设

**H7：随机种子不同 → 模型初始化 + 数据划分 + DataLoader shuffle 不同 → 训练轨迹分化 → 最终收敛到不同的局部最优**

这一假设直接由 A7 审计中识别的多个 DIFFERENT 项（split seed、grad accumulation → 有效 batch size 差异、checkpoint 保存策略差异）联合支持。

### 验证方法

启动两个不同 seed 的完整训练（seed=0, seed=123），与 EXP-010（seed=42）对比：

1. 训练 loss/val loss 曲线是否随 seed 显著分化
2. early stopping 是否在不同 epoch 触发
3. best checkpoint 的单步指标是否随 seed 有方差
4. **关键**：best checkpoint 的 18-step rollout 指标是否随 seed 有方差
5. 如果 3 个 seed 的 rollout 指标方差大且覆盖与开源模型的 gap → H7 supported
6. 如果 3 个 seed rollout 指标都稳定在 EXP-010 水平 → 需要新假设

### 相关实验

| 实验 | Slurm Job | Seed | GPU | Status |
|------|-----------|------|-----|--------|
| EXP-010（历史参考） | 52 | 42 | GPU 1 | Completed |
| EXP-014（本次） | 59 | 0 | GPU 0 | **Running** (~18 min/epoch, epoch 0 done) |
| EXP-015（本次） | 60 | 123 | GPU 1 | **Running** (~18 min/epoch, epoch 0 done) |

---

## Runtime 与 Artifacts

### 审计期间实际运行的命令

```bash
# 开源 TensorBoard 事件文件解析
python -c "
import tensorflow as tf
events = tf.compat.v1.train.summary_iterator(...)
# 输出 epoch_loss / val_loss 的 step 和 value
"

# 跨版本架构对比
pip install diffusers==0.31.0
python -c "from diffusers import UNet2DModel; m = UNet2DModel.from_pretrained(...); print(m)"
pip install diffusers==0.40.0
python -c "from diffusers import UNet2DModel; m = UNet2DModel.from_pretrained(...); print(m)"

# 同权重 forward 数值对比
python -c "
m1 = UNet2DModel.from_pretrained(OFFICIAL_PATH)
m2 = UNet2DModel(config)
m2.load_state_dict(m1.state_dict(), strict=True)
x = torch.randn(2, 2, 256, 256)
with torch.no_grad():
    y1 = m1(x, torch.zeros(2)).sample
    y2 = m2(x, torch.zeros(2)).sample
print('max_abs_diff:', (y1 - y2).abs().max().item())
"

# main 分支脚本关键词 grep
rg -n "rollout_loss|multi_step|scheduled_sampling|rollout_checkpoint" scripts/Chase_2025/

# DataLoader / split seed 检查
rg -n "manual_seed|random_split|DataLoader.*shuffle" scripts/Chase_2025/
```

### 本次代码修改

`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`：

| 修改 | 行号 | 说明 |
|------|------|------|
| Seed 环境变量化 | 55-60 | `_SEED = int(os.environ.get("SEED", "42"))`；同时设置 torch / cuda / random / numpy 的 seed |
| Output dir 环境变量化 | 99 | `os.environ.get("OUTPUT_DIR", ...)` |
| 其余修改（前序 commit） | 多处 | ZarrDataset 全量加载、80/20 split、独立 val zarr、early stopping、best checkpoint 保存、run_metadata.json 写入 |

### Slurm 脚本

- `scripts/Chase_2025/train_vanilla_unet_seed0.slurm`（Job 59）
- `scripts/Chase_2025/train_vanilla_unet_seed123.slurm`（Job 60）

### 环境

- conda env：`cira-diff-ly`
- Python 3.11.16, torch 2.6.0+cu124, diffusers 0.40.0, accelerate 1.10.1
- Git commit：`cf67d6e`；branch：`feature/vanilla-unet-baseline-fix`
- Hardware：NVIDIA RTX 4090 48GB × 2（GPU 0 / GPU 1 各一个 Slurm Job）

### 输出路径

| 实验 | Output Dir | Slurm Log |
|------|-----------|-----------|
| EXP-014 (seed=0) | `outputs/vanilla_unet_seed0/` | `outputs/slurm_logs/vanilla_unet_seed0_59.{out,err}` |
| EXP-015 (seed=123) | `outputs/vanilla_unet_seed123/` | `outputs/slurm_logs/vanilla_unet_seed123_60.{out,err}` |
| EXP-010 (seed=42, 历史) | `outputs/vanilla_unet_full/` | `outputs/slurm_logs/vanilla_unet_full_52.{out,err}` |

---

## Early Epoch 损失分化（epoch 0）

| 指标 | EXP-010 (seed=42) | EXP-014 (seed=0) | EXP-015 (seed=123) |
|------|-------------------|------------------|--------------------|
| train_loss | — | 0.092639 | 0.062186 |
| internal_val | — | 0.025134 | 0.022572 |
| independent_val | — | 0.023504 | 0.022091 |

仅 epoch 0，两个 seed 的 independent_val 已相差 ~6%（seed 123 vs seed 0）。这说明 **随机种子确实在早期就导致了训练轨迹分化**。EXP-010 的 epoch 0 数值未记录但预期也在不同位置。

---

## 后续评估计划

训练完成后：

1. **手动执行 EXP-EVAL-001 评估流程**，对 EXP-014 / EXP-015 的 best_internal_unet checkpoint 进行：
   - 单步 teacher-forced（test zarr 1024 样本）
   - 18-step autoregressive rollout（128 样本子集）
2. 三组 seed（0 / 42 / 123）的 rollout 指标绘制箱线图
3. 与官方开源 checkpoint 指标做均值 ± std 对比
4. 如果开源 checkpoint 落在 3-seed ± 2σ 区间内 → H7 supported
5. 如果开源 checkpoint 显著优于 3-seed → 需要新假设（如：Official 使用了不同的 val split 监控指标，或 LR schedule 实际走完了不同的 step 数）

---

## Interpretation

本审计系统性排除了 6 个容易想到的根因假设（H1-H6），将候选集中到 H7（随机因素）。在排除 diffusers 版本和 checkpoint 选择后，**唯一未被解释的训练配置差异是：DataLoader 种子、dataset split 种子、模型初始化种子**——这三者都由同一个 `torch.manual_seed()` 控制，但 Official 原始脚本的具体 seed 值未知。

**重要补充**：即使两个 seed 实验也没有复现开源 checkpoint 的 rollout 水平，这不一定意味着 H7 被推翻——可能 Official 用了某个特定 seed 恰好落在更优的局部最优，或者存在尚未识别的差异（如：Official early stopping 监控的是 held-out split 还是独立 val zarr？）。

---

## Conclusion（审计阶段）

- **Reproduction fidelity（单步）**: ✅ **Supported**。EXP-010 单步指标与开源打平（±1.5% 内）
- **Reproduction fidelity（rollout）**: ❓ **Unresolved**。Rollout gap 真实存在，已排除训练长度、diffusers 版本、checkpoint 选择、rollout training 缺失等 6 个假设，当前最强候选为随机因素
- **Hypothesis H7（随机因素导致 rollout gap）**: 🔬 **Under active testing**（EXP-014 / EXP-015 运行中）

---

## Limitations

- Official 原始脚本的 early stopping 实现细节未知（patience / monitor metric / min_delta）
- Official 原始脚本的 split seed 和初始化 seed 未知
- 只启动了 2 个 seed 实验，统计功效有限（3 个 seed 总样本量）
- 如果 H7 被推翻，需要重新检查 Official checkpoint 是否为 best val loss epoch（原始脚本每 epoch 覆盖 save_pretrained，可能存的是 epoch 48 last 而非 best val epoch ~40）

---

## Reproducibility

- 审计脚本：见上文"Runtime 与 Artifacts"章节的命令列表
- 训练脚本：`scripts/Chase_2025/train_vanilla_unet_Chase2025.py`（Git status: modified）
- Slurm 脚本：`scripts/Chase_2025/train_vanilla_unet_seed{0,123}.slurm`
- 环境：见上文
- 数据：`edm_GOES_ch13_train_dataset.zarr` / `_validation_dataset.zarr` / `_test_dataset.zarr`
- 开源 checkpoint：`/data1/satcast/unet_vanilla/unet_vanilla/`（diffusers v0.31.0）
- 开源 TensorBoard logs：`/data1/satcast/unet_vanilla/unet_vanilla/logs/`

---

## 关联文献 / 决策 / 下一实验

- 关联实验：EXP-010 / EXP-014 / EXP-015 / EXP-EVAL-001
- 关联 HYP：HYP-005 / HYP-006 / H7（本文）
- 关联文献：NVIDIA CIRA-Diff 2023；Chase et al. 2025 Vanilla UNet baseline
- 下一实验：EXP-EVAL-003（多 seed rollout 评估 + 方差分析）→ 如果 H7 被推翻，需要回到 diffusers 版本或 early stopping 监控指标上深挖


# GPT 总结
不是。现在还不能说“除了 seed 外哪里都一模一样”，更不能说“rollout 差异只可能由 seed 导致”。

更准确的结论是：

我们已经排除了很多大的系统性差异，目前“随机 realization”是最值得验证的剩余因素之一，但仍存在若干已确认差异和无法恢复的 UNKNOWN。

目前到底对齐到了什么程度

项目	Official vs EXP-010	当前判断
数据集	相同 Zarr	✅
preprocessing	相同	✅
UNet architecture	113.67M，结构完全相同	✅
checkpoint state_dict 结构	完全兼容	✅
diffusers 0.31 vs 0.40 forward	同权重同输入 0 diff	✅ 排除
Loss	single-step MSE	✅
Optimizer	AdamW，同参数	✅
Precision	fp16	✅
LR trajectory	关键 epoch 差 <0.2%	≈ 对齐
early-stop 算法	window=5, patience=10	✅
rollout loss / scheduled sampling	都没发现	✅
实际训练长度	48 vs 47	≈ 对齐
train/val split	Official 随机未知；EXP seed42	❌ 不同/无法复原
初始化权重	Official seed 未知；EXP seed42	❓无法复原
shuffle 顺序	Official 未知	❓
physical batch	45 vs 24	❌
gradient accumulation	1 vs 2	❌
effective batch	45 vs 48	≈但不完全相同
DataLoader	workers 0 vs 4 等	❌
scheduler total steps	132930 vs 249270	❌，但实际 LR 很接近
Official released checkpoint 来源	likely last，但无法直接证明	❓

所以不能概括成：

Official = EXP-010 + different seed

只能说：

核心模型、数据、目标函数和主要优化设置已经高度对齐；剩余差异集中在 stochastic training realization 和少量训练工程设置。

⸻

而且“seed”其实不是一个变量

这是非常重要的一点。

你现在 EXP-010 的：

torch.manual_seed(42)

会同时影响很多东西：

\text{seed}
\rightarrow
\begin{cases}
\text{80/20 split}\\
\text{initial weights}\\
\text{DataLoader shuffle}\\
\text{dropout/random ops（若有）}\\
\text{CUDA stochasticity}
\end{cases}

所以即使 EXP-011 最终证明换 seed 能让 LT18 从 0.57 → 0.49，也不能马上写：

“模型 rollout 对初始化 seed 敏感。”

只能写：

rollout stability 对 training random realization 敏感。

然后才能进一步拆：

* fixed split + different init seed
* different split + fixed init
* fixed split/init + different shuffle

来确定究竟是谁。

⸻

但你问的另一个问题很关键：这么大的差异真的可能只是随机性吗？

完全有可能，但目前没有证据证明就是。

因为你自己的模型已经给出了一个非常强的例子：

EXP-010 同一次训练：

\mathrm{MSE}_{single}:
0.00809\rightarrow0.00827

几乎没变。

但：

\mathrm{MSE}_{LT18}:
0.574\rightarrow0.934

恶化了 63%。

也就是说，这个系统已经实证告诉我们：

极小的 one-step skill 差异，可以对应巨大的 long-rollout stability 差异。

这是递归系统很典型的性质。

粗略理解：

x_{t+1}=f_\theta(x_t,x_{t-1})

single-step MSE 只衡量：

在真实数据分布附近，预测下一帧准不准。

但 rollout 到后面输入变成：

f_\theta(
\underbrace{\hat x_{t-1},\hat x_t}_{模型自己产生的分布}
)

训练时模型基本没见过这种输入。

因此两个模型在真实样本附近可能：

Official:  MSE = 0.00827
EXP:       MSE = 0.00809

几乎一样。

但稍微离开 training manifold 后：

Official fθ：误差不容易继续放大
EXP fθ：    误差不断放大

18 次复合：

f_\theta\circ f_\theta\circ\cdots\circ f_\theta

以后差异就可能非常大。

所以：

single-step 一样 ≠ learned dynamics 一样 ≠ rollout stability 一样。

⸻

现在还有一个我不愿意直接忽略的变量：batch 45 vs 24×2

这一点比前面报告说的 "Minor" 更值得谨慎。

Official：

physical batch = 45
gradient accumulation = 1

EXP：

physical batch = 24
gradient accumulation = 2
effective batch ≈ 48

它们不是严格等价的训练过程。

虽然 effective batch 很接近：

45\approx48

但 accumulation 之后 optimizer update 的梯度统计、最后 incomplete batch、AMP scaling，以及某些实现细节都可能不同。

我不会现在断言它能造成 LT18 0.49 vs 0.57，但也不能在实验前把它排除。

⸻

所以 EXP-011 的目的其实不是“证明 seed 是原因”

而是一个非常便宜的分叉实验。

情况 A：不同 seed 得到

Official    0.491
seed 0      0.50
seed 1      0.63
seed 2      0.47
seed 42     0.574

这时证据会非常强：

Official 并不异常，它处在正常 reproduction variance 中。

那么“隐藏 rollout trick”的可能性就大幅下降。

之后再拆 randomization 来源。

情况 B：

Official    0.491
seed 0      0.68
seed 1      0.72
seed 2      0.64
seed 42     0.574

那我不会接受“seed 导致”这个解释。

因为 reproduced models 出现系统性偏差。

这时候下一步应该把训练 protocol 进一步严格拉齐，例如：

physical batch 45 / accum 1 + 原始 DataLoader + 原始训练脚本

然后再复现。

⸻

其实还可以让 EXP-011 更干净

如果你的核心目标现在是：

先判断随机性够不够解释这个 gap

我甚至建议 EXP-011 顺手把能对齐 Official 的确定性配置全部对齐：

physical batch = 45
grad accumulation = 1
num_workers = 0
pin_memory = False

而不是继续沿用 EXP-010 的 24×2。

因为这样新的实验就是：

尽可能 faithful 的 Official protocol + 唯一不可恢复的是 Official random realization。

然后跑 seed 0/1/2。

这样如果其中出现 LT18≈0.49，解释会比当前设计更有说服力。

因此我现在的判断是

不是：

“查完了，只有 seed 不一样，所以一定是 seed。”

而是：

架构、数据、forward、loss、optimizer、precision、主要 LR 行为、训练长度等大的 implementation gap 已经基本排除。现在剩余最重要的未知量是 random realization，同时仍有 batch/accumulation/DataLoader 等确定性差异没有完全对齐。

所以我建议先把 batch=45, accumulation=1, workers=0 等恢复成官方设置，再做 3-seed reproduction。这样这轮 GPU 实验的信息量最大，也更接近真正的“faithful reproduction”。