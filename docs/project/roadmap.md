# Roadmap

## Phase 1 — 官方 baseline audit 与最小复现

目标：确认数据、训练、采样、单步预测和 3 h rollout 的真实代码路径。

任务：

- 修复远程数据访问并记录 Zarr 元数据。
- 建立本地最小 train/validation/test 子集。
- 完成 Dataset/DataLoader smoke test。
- 整理可运行 config 与训练命令。
- 复现 persistence、U-Net、Diff、CorrDiff 的单步和 rollout 评估。

## Phase 2 — 评估协议与误差诊断

目标：从单纯 pixel metrics 扩展到结构、多尺度和冷云事件诊断。

任务：

- 复核 ME/MAE/RMSE 的实现和 lead-time 聚合方式。
- 增加结构、频谱和阈值事件指标，但先与文献和数据分布核对。
- 区分 motion error、cloud development error 与 rollout drift。

## Phase 3 — temporal formulation 对比

目标：在同一数据与预算下比较短历史单步 rollout 和多帧到多帧预测。

任务：

- 设计 2/4/8/12/24 帧 history ablation。
- 固定 forecast horizon、采样分辨率和数据切分。
- 分析 long-lead、temporal consistency、small-scale structure 与训练成本。

## Phase 4 — 研究改进

候选方向：

- 多尺度/频谱约束。
- motion-conditioned 或光流辅助。
- 面向冷云/强对流的事件评价与条件建模。
- ensemble calibration 与 regime-aware uncertainty。

以上方向均为候选计划，不代表已作出模型选择。
