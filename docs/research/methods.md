# 方法设计

## 1. 官方 baseline 任务

[EVIDENCE] 目标是单通道 GOES-16 ABI Channel 13、10 min 时间间隔的下一帧亮温预报：

```text
condition: [X(t-10 min), X(t)]
target:    X(t+10 min)
rollout:   repeat 18 steps -> t+3 h
```

训练 ordinary sample 是 `2 -> 1`；validation/test sample 另外保存 `18` 帧真实 future truth。训练样本是预处理后的 256×256 patch；论文和远端说明按训练集统计量做标准化。

## 2. Diffusion formulation

给定 clean target `y`、condition `c`，从 log-normal 分布采样噪声等级 `sigma`，构造：

```text
noisy_y = y + sigma * epsilon
model_input = concat(noisy_y, c, channel=1)
```

`EDMPrecond` 使用 `sigma_data` 计算 `c_skip`、`c_out`、`c_in`、`c_noise`，U-Net 输出经预条件组合成 denoised image。`EDMLoss` 返回逐像素加权平方误差，训练循环再取 mean。

## 3. Diff 与 CorrDiff

### Diff

[FACT] 直接把下一帧亮温作为 diffusion generation target；condition 是两帧历史图像。

### CorrDiff

[FACT] 当前配置注释和训练脚本表明，condition 由两帧历史图像加一帧普通 U-Net 预测组成，diffusion target 是 residual。最终结果需要把 residual 加回普通 U-Net 预测。

[UNKNOWN] 当前仓库没有提供已验证的、从数据生成普通 U-Net 预测并与 residual Zarr 对齐的完整 pipeline。

## 4. Sampling

`edm_sampler` 从高噪声 latent 开始，默认 18 个 noise steps，使用 Euler 更新和二阶校正；`S_churn`/`S_noise` 可引入额外随机性。不同随机 seed 可以生成 ensemble member。

## 5. Rollout 约定

[EVIDENCE] 论文定义为：模型先使用 t−10、t 预测 t+10；随后将预测结果递归反馈，得到任意长度的 rollout，本文选择 18 步即 3 h。

[FACT] 当前公开数据的 3 h truth 不是从 ordinary train 的三帧 sample 推出来的，而是直接存放在 validation/test 的 `output_images`，shape 为 `(1024,18,256,256)`。forecast notebook 将模型预测序列的前两帧 condition 与 18 帧 truth 分开保存，并在 `forecast_time` 维对齐。

[UNKNOWN] `/data1/satcast/` 不含 per-sample timestamp 或生成 Zarr 的完整索引脚本，因此无法仅凭该目录确认每个 patch 的原始 GOES 文件名和跨 sample 连续性。

[BLOCKED] 本地 smoke test 仍受 `torch` 和 `zarr` 缺失影响；已完成远端 Zarr 元数据与少量样本读取。

## 6. 任务区分

[DECISION] Lab 使用“satellite cloud forecasting”作为目标任务。Generic video prediction 可以提供方法和时间建模思路，但不能建立 brightness temperature 的物理含义。Precipitation nowcasting 的 target 和 threshold semantics 不同。NWP/AI weather forecasting 可能预测大气状态或使用额外变量。只有 input、target、sampling、horizon 和 verification contracts 匹配时，比较才有效。

## 7. 实验必须控制的变量

每个 temporal-formulation comparison 必须记录：

- sensor/channel 以及物理单位或 normalization；
- temporal interval 和 input/output frame count；
- spatial crop/grid 和 split policy；
- 分别记录 teacher forcing、scheduled sampling、rollout training 和 free-running inference；
- parameter count、training budget 和 sampler settings；
- lead-time metrics 以及 structural/scale/event diagnostics；
- checkpoint selection rule，以及结果是 single-sample、ensemble mean 还是 ensemble distribution。
