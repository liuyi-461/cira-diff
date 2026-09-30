# 气象领域知识规范（Skill）

> 可复用能力，不绑定单个项目。

## 适用场景
卫星云图 / 天气预测中的时空预测任务。

## 关键知识
- 气象变量通道：温度、湿度、风场（u/v）、云量等，注意单位与量级差异。
- weather 任务在 OpenSTL 中开启 `spatial_norm=True`（见 `Base_method`）。
- 评估时常用纬度加权（latitude weighting）以处理极地偏差。

## 复用建议
- 数据归一化务必基于训练集统计量（mean / std），并随模型保存以便推理对齐。
- 指标报告区分 per-channel，便于与 CIRA-Diff 对比。
