# Evaluation 系统

Evaluation 是核心科研子系统。当前仓库已有 metric 计划和论文中报告的指标，但没有已验证的项目级 evaluator。

## 状态含义

- **Implemented：** 已在本仓库实现，并在有记录的 sample 上验证。
- **Planned：** 只有设计意图。
- **Under Evaluation：** 正在检查实现或适用性，不能作为稳定结果。
- **Not Implemented：** 没有已验证实现。

## 当前矩阵

| 家族 | 状态 | Canonical 文档 |
| --- | --- | --- |
| Pixel | 本地 Not Implemented；论文有 evidence | `pixel_metrics.md` |
| Structural | Planned；需要 applicability audit | `structural_metrics.md` |
| Threshold | Planned；threshold source 未解决 | `threshold_metrics.md` |
| Spectral | Planned | `spectral_metrics.md` |
| Object | Planned；object definition 未解决 | `object_metrics.md` |
| Temporal | Planned | `temporal_metrics.md` |
| Probabilistic | Planned | `probabilistic_metrics.md` |
| Protocol | Planned contract | `evaluation_protocol.md` |

不能因为论文列出了指标，就隐藏本地没有实现这一事实。
