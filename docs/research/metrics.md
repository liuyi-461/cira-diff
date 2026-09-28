# 科研指标

本文是 research-level index；可执行定义和实现状态位于 `docs/evaluation/`。

| 维度 | 候选指标 | 科学问题 | 当前状态 |
| --- | --- | --- | --- |
| Pixel | ME/Bias、MAE、MSE、RMSE、PSNR | BT field 数值是否准确？ | 论文 evidence；本地 evaluator 未实现 |
| Structure | SSIM、LPIPS（需 applicability audit） | 空间模式和边缘是否保持？ | Planned |
| Threshold | CSI、POD、FAR、ETS、Brier | 冷云/事件区域是否命中？ | Planned；threshold 来源未知 |
| Spectral | energy spectrum、scale-dependent error | 哪些空间尺度失去可预测性？ | Planned |
| Object | area、centroid/displacement、overlap、growth/decay、lifetime、organization | 云对象是否正确演变？ | Planned；object definition 未知 |
| Temporal | lead-time curves、tendency error、temporal consistency | 时间演变是否连贯？ | Planned |
| Probabilistic | CRPS、reliability、spread-skill、rank/histogram diagnostics | ensemble 是否校准且有用？ | Planned |
| Cases | deep convection、cold cloud、tropical cyclone | aggregate skill 是否掩盖高影响失败？ | Planned；labels 不可用 |

## 指标纪律

每个报告值必须说明单位、normalization/inverse transform、mask、threshold、lead-time aggregation、sample weighting、ensemble reduction，以及它是 pixel、object 还是 case statistic。“论文中实现过”不等于“本仓库已实现”。
