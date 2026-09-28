# 评价指标与协议

## 官方 baseline 已确认指标

[EVIDENCE] 论文使用 mean error（ME，作为 bias/drift）、MAE、RMSE；在 1,000 个 validation/test patch 上按像素聚合，并比较 persistence、普通 MSE U-Net、Diff、CorrDiff、LDM 及 ensemble 变体。

[EVIDENCE] 论文还讨论频谱/高分辨率特征、冷亮温区域、case study 和 ensemble spread-skill；这些不能等同于当前仓库已有评估代码。

## 本项目分层评价框架

| 科学问题 | 指标候选 | 解释 |
|---|---|---|
| 整体亮温场是否准确 | ME/Bias、MAE、RMSE | 整体误差和随 lead time 的 drift |
| 云系结构是否保持 | SSIM、对象面积/质心/重叠 | 区分位移、边缘和形态变化 |
| 小尺度是否过快衰减 | power spectrum、scale-dependent error | 检查 smoothing 与频谱能量损失 |
| 冷云/强对流是否命中 | BT threshold、CSI/POD/FAR/ETS | 防止大面积普通云或晴空掩盖高影响区域 |
| 运动还是发展预测正确 | displacement、area growth、BT tendency | 区分 advection 与 initiation/intensification/decay |
| 概率预报是否可靠 | CRPS、Brier、reliability、spread-skill | 评价 ensemble，而不是只看 ensemble mean |

## 首轮协议

1. 固定 data split、时间间隔、patch size、lead times 和随机种子。
2. 至少比较 persistence、普通 U-Net、Diff、CorrDiff。
3. 单步报告 t+10 min；rollout 报告每个 10 min lead 到 3 h。
4. ensemble 报告成员数、seed、采样超参，并与单成员和 ensemble mean 分开。
5. 先复现论文的 ME/MAE/RMSE，再扩展结构、多尺度和事件指标。
6. 每个指标记录 mask、阈值、聚合顺序和单位；不能只保存最终数字。

## 状态

[UNKNOWN] 本仓库当前没有经确认可运行的 evaluation CLI；EXP-001 需先建立它。
