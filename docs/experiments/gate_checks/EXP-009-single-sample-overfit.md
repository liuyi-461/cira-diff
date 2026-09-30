# EXP-009 — 四种方法单样本 Overfit Gate Check

## Identity
- 实验 ID：EXP-009
- 状态：Completed（gate check，非正式科学实验）
- 日期：2026-09-22
- 负责人：liuyi
- 关联 RQ：RQ-006（scale dependence 的前提）
- 关联 HYP：HYP-005（pixel error 不等于 object skill — 本文不直接检验）

## Research question 与 hypothesis
- Research Question：**本窗口不检验科学 RQ**；本实验是工程 gate check，确认四种 baseline 代码路径（Vanilla UNet / EDM / LDM / CorrDiff）在单样本上能完整跑通前向、反向和推理。
- Hypothesis：作为 gate check 的隐式假设是"如果模型连单样本都拟合不了，它在大规模数据上不可能学好"；这是深度学习社区的标准 sanity check（overfit one batch test）。
- Scientific motivation：在运行完整训练之前，先用最小代价确认代码路径没有 forward/backward bug、loss 计算 bug、模型容量不足等问题。

## Dataset
- Dataset ID / manifest：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`（raw）+ `/data1/satcast/edm_GOES_ch13_train_dataset_latent.zarr`（latent）
- Time period：[EVIDENCE] GOES-16 ABI Channel 13 10 min 间隔；[UNKNOWN] 具体日期区间未记录在 Zarr metadata 中
- Region：[UNKNOWN]
- Sensor/channel/unit：GOES-16 ABI C13（亮温，单位 [UNKNOWN]）
- Spatial resolution / crop：256×256 patch（raw），64×64 patch（latent）
- Temporal resolution：10 min
- Train / validation / test：全部只用 sample index **0**（单样本，无 split）
- Sample construction：
  - Vanilla UNet / EDM：`input_images[0:2]` (2ch) → `output_images[0]` (1ch)
  - LDM：latent `input_images[0:8]` (8ch) → `output_images[0:4]` (4ch)
  - CorrDiff：`cat([raw input 2ch, vanilla_unet_pred 1ch])` → residual = `target - vanilla_unet_pred`
- Normalization：[FACT] 脚本未显式做 mean/std 归一化；[EVIDENCE] 原始 CIRA-Diff 论文是否做了归一化未记录在本窗口

## Model 与 temporal semantics
- Architecture：全部四种方法使用 `diffusers.UNet2DModel`，相同 backbone（layers_per_block=2, block_out_channels=(128,128,256,256,512,512), 一个 AttnDownBlock / 一个 AttnUpBlock）
- Initialization/checkpoint：全部随机初始化，无预训练权重
- Input frames：
  - Vanilla UNet：2 帧历史（t-1, t）
  - EDM：同上，拼接加噪后的 target
  - LDM：latent 空间 8 通道（2 帧历史 encode 后）
  - CorrDiff：3 通道（2 帧历史 + Vanilla UNet baseline forecast）
- Output frames：全部 1 帧（单步预测）
- Forecast interval：10 min
- Rollout length：1（单步预测；本实验不检验 autoregressive rollout）
- Teacher forcing：不适用（单样本 overfit）
- Scheduled sampling：不适用
- Rollout training 与 rollout inference：本实验只检验单步 overfit；CorrDiff 在 inference 阶段的 residual rollout 由 EDM sampler 完成（Heun 二阶 20 步）

## Training configuration
| 方法 | Loss | Optimizer | LR | Batch | Epoch | Precision | Seed | GPU |
|------|------|-----------|----|-------|-------|-----------|------|-----|
| Vanilla UNet | MSELoss | AdamW | 1e-4 | 1 | 1000 | fp32 | 42 | RTX 4090 48GB |
| EDM | EDMLoss (weighted MSE) | AdamW | 1e-4 | 1 | 1000 | fp32 | 42 | RTX 4090 48GB |
| LDM | EDMLoss (weighted MSE) | AdamW | 1e-4 | 1 | 1000 | fp32 | 42 | RTX 4090 48GB |
| CorrDiff | EDMLoss (weighted MSE) | AdamW | 1e-4 | 1 | 1000 | fp32 | 42 | RTX 4090 48GB |

EDM / LDM / CorrDiff 共享相同的 `EDMPrecond` 包装：P_mean=-1.2, P_std=1.2, sigma_data=0.5。
Config path：独立脚本 `/home/group1/26fall_aiclass/ly/cira-diff/scripts/Chase_2025/run_overfit_comparison.py`。

## Runtime 与 artifacts
- Command：`python scripts/Chase_2025/run_overfit_comparison.py`
- Environment：conda env `cira-diff-ly`，Python 3.11.16，torch 2.6.0+cu124
- Git commit：`cf67d6e`（HEAD）
- Log：stdout（tqdm progress bar）
- Output：`outputs/overfit_comparison/`
- Checkpoint：
  - Vanilla UNet：`outputs/vanilla_unet_overfit/diffusion_pytorch_model.safetensors`（434 MB）
  - EDM / LDM / CorrDiff：未保存 checkpoint（overfit 实验不归档模型权重）
- Artifact/checksum：
  - `outputs/overfit_comparison/overfit_comparison.png`（4 方法预测 + Error 可视化）
  - `outputs/overfit_comparison/loss_curves.png`（四种方法 loss 曲线）

## Evaluation protocol
- Metrics：MSE, MAE（逐像素）
- Lead times：10 min（单步）
- Thresholds/masks：无
- Ensemble members/seeds：1 seed
- Aggregation：N/A

## Results
[RESULT] 同一 sample index=0，四种方法各训练 1000 epoch 后在同一 target 上的 MSE/MAE：

| 方法 | MSE | MAE | 备注 |
|------|-----|-----|------|
| Vanilla UNet | **3.95e-05** | **5.00e-03** | 直接回归，无扩散 |
| CorrDiff | 3.49e-04 | 1.28e-02 | UNet prior + residual（次优） |
| EDM | 3.50e-03 | 4.46e-02 | 20-step Heun sampling |
| LDM | 5.26e-03 | 5.53e-02 | latent 空间 64×64 4ch |

[RESULT] CorrDiff residual 本身分布：mean=-0.003, std=0.0056, max=0.0495（远小于 target 尺度 ~0.5）。

## Interpretation
1. [CONCLUSION] **四种方法均通过 gate check**：所有 loss 均显著下降至远小于初始水平，代码路径（forward / backward / loss / inference sampler）无 bug。
2. [RESULT] Vanilla UNet 在单样本 overfit 上 MSE 最低（~4e-5），这是预期结果——没有扩散噪声的确定性回归在 overfit 场景下比扩散模型更容易达到零误差。**这不是关于泛化能力的结论**。
3. [CONCLUSION] CorrDiff 的 residual 分布本身很小（std=0.0056），扩散模型只需要学习在这个小量上做去噪。这直接验证了 CorrDiff 论文的核心设计（"先跑 baseline，再让扩散模型纠正残余"）在工程上是可实现的。
4. [ISSUE] LDM 的 latent 输出直接与 pixel-space target 比较存在量纲不匹配；如需公平比较，应加载 VAE decode 回来再算 MSE。本实验未做此 decode 步骤。

## Conclusion
本实验 **通过了工程 gate check**，四种 baseline 代码路径均可正常运行。Vanilla UNet、EDM、LDM、CorrDiff 各脚本现在可以正式投入完整训练。

**本实验结果不能被解读为关于泛化能力或 forecast skill 的结论**——它是在单样本上的 overfit，验证的是"模型能拟合"而非"模型能预测"。

## Limitations
- 单样本，无验证集，无泛化指标
- CorrDiff 的 vanilla_unet_prior 是训练好的 overfit 模型（不是独立训练的 baseline）
- LDM 的 latent MSE 未经过 VAE decode 到像素空间
- 所有方法仅 seed=42，无重复性验证

## Reproducibility
- Code：`scripts/Chase_2025/run_overfit_comparison.py`
- Env：`cira-diff-ly`（conda，Python 3.11.16，torch 2.6.0+cu124）
- Data：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`（raw）+ `_latent.zarr`
- Hardware：NVIDIA RTX 4090 48GB
- Smoke rerun time：约 2-3 分钟/方法，总计 10 分钟

## 关联文献 / 决策 / 下一实验
- 关联文献：CIRA-Diff（NVIDIA 2023），Karras EDM（2022）
- 关联 DEC：DEC-001（数据审计完成，正式训练就绪）
- 下一实验：**EXP-010**（Vanilla UNet full training reproduction），之后依次是 EDM / LDM / CorrDiff full training
- 关联 HYP：HYP-006（扩散不必然优于 deterministic baseline 的初步 overfit 证据，但不等价于泛化比较）