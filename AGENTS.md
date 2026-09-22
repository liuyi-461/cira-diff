# Satellite Forecasting Lab 协作规范

## 项目范围

本仓库是卫星云图预报科研实验室。CIRA-Diff 是当前复现对象和 baseline，不是整个项目的定义。

## 证据纪律

使用 `docs/project/knowledge.md` 定义的证据标签：`[FACT]`、`[EVIDENCE]`、`[HYPOTHESIS]`、`[RESULT]`、`[CONCLUSION]`、`[DECISION]`、`[ISSUE]`、`[PLAN]`、`[UNKNOWN]`、`[CONFLICT]`。

不得把论文结论、计划实验、成功导入或视觉上合理的预报提升为科研结论。必须区分 rollout training 与 autoregressive rollout inference。

## 受保护实现

- 在迁移方案经过明确评审前，将 `cira_diff/` 与 `scripts/Chase_2025/` 视为受保护的 working implementation。
- 不得仅为了匹配新目录结构而移动或重写 CIRA-Diff 代码。
- 新的稳定代码只有在接口和行为得到验证后，才能进入 `src/satforecast/`。
- 模型专用的 reproduction patch 与 adapter 放在 `reproduction/<model>/` 下。

## 科研可追溯性

每个实验都必须关联 research question、hypothesis、data contract、代码/config、命令、artifact、result、interpretation 和 decision。未知字段必须保留为 `Unknown`、`Not Recorded` 或 `TODO`。

## 分支与数据安全

- 未经用户明确指示，不得合并分支。
- 未经明确请求，不得提交本地数据、checkpoint、TensorBoard 日志或生成媒体。
- 将历史文档作为 provenance 保留；只有在原文仍可访问时，才能用 pointer 替代正文。
