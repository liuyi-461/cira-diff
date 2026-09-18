# 科学问题与研究假设

## 已解决的数据组织问题：三帧不是所有 split 的最大序列

[CONCLUSION] 远端 Zarr 证据表明，ordinary train 的 sample 是 2 帧输入 + 1 帧 target 的三帧训练窗口；validation/test 的 sample 是 2 帧输入 + 18 帧 future truth。后者用于验证 18-step、3 h autoregressive rollout。

[UNKNOWN] 远端派生 Zarr 没有 per-sample timestamp，因此暂不能从 `/data1/satcast/` 单独恢复跨 sample 的原始 GOES 连续序列；这不影响确认 eval sample 内部保存了 18 帧 truth。

## Q1：短历史单步 rollout 与多帧到多帧，哪种 temporal formulation 更适合云图预报？

### 问题描述

比较：

```text
范式 A： (X[t-1], X[t]) -> X_hat[t+1] -> autoregressive rollout
范式 B： (X[t-n+1], ..., X[t]) -> (X_hat[t+1], ..., X_hat[t+m])
```

### 当前认识

[EVIDENCE] CIRA-Diff 属于范式 A：两帧、10 min 间隔、单步预测后 autoregressive rollout。

[HYPOTHESIS] 两帧可能已包含较强的一阶平流信息，但更长历史可能对云生成、增强、消散和趋势识别更有帮助。

### 假设

[HYPOTHESIS] 范式 A 具有低显存、任意 horizon 和易于 ensemble 的优势，但可能面临 rollout error accumulation；范式 B 可能改善时间上下文和短期 temporal consistency，但训练成本与输出长度耦合。

### 验证方法

- 在同一数据切分、分辨率、时间间隔、参数预算下比较 history=2/4/8/12/24。
- 固定 1 h、3 h horizon，分别报告单步和 rollout lead-time 曲线。
- 同时报告像素、结构、多尺度、冷云事件和时间一致性指标。
- 明确训练是否 teacher forcing、scheduled sampling 或 rollout training。

## Q2：扩散随机性是否提供具有气象意义的不确定性？

### 问题描述

扩散模型可以用多个随机种子生成 ensemble，但“图像多样性”不等于“概率预报可靠”。

### 当前认识

[EVIDENCE] 官方论文报告 ensemble spread-skill 关系，但同时指出不同误差区间可能出现 under-dispersion 或 over-dispersion。

### 假设

[HYPOTHESIS] 不确定性质量可能随天气型、lead time 和冷云区域变化，单一全域 spread 统计不足以代表强对流预报价值。

### 验证方法

- 预先固定 ensemble seeds、成员数和采样超参。
- 报告 CRPS/Brier 或等价概率指标、reliability、spread-skill。
- 按普通云、冷云、对流候选区域和天气型分层分析。

## Q3：pixel skill 是否与云系结构 skill 一致？

### 当前认识

[HYPOTHESIS] MSE/RMSE 较好的预测可能仍然出现云团边缘变软、频谱能量损失或位移错误。

### 验证方法

- 以 RMSE/MAE/Bias 为基础。
- 增加 SSIM、对象面积/质心/重叠、频谱或尺度相关误差。
- 用亮温阈值定义冷云候选，并报告 CSI/POD/FAR；阈值需基于文献和数据分布确定，不能先验写死。
