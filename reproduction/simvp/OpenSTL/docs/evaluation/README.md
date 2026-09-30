# 验证与评价指标

本目录定义 SimVP reproduction 的验证协议与评价指标，确保与 CIRA-Diff 可比。

## 指标
- MAE（Mean Absolute Error）
- MSE（Mean Squared Error）
- 由 `openstl/core/metric.py` 的 `metric()` 计算，支持 `spatial_norm`（weather 任务开启）
  与 `channel_names`。

## 验证协议（关键）
- 多步预测协议必须与 CIRA-Diff 一致：direct multi-step vs autoregressive（见 DEC-002）。
- `aft_seq_length` 两边对齐；SimVP 在 `aft>pre` 时走 `SimVP.forward` 的递归自回归分支。

## 输出产物
- `on_test_epoch_end` 将 `inputs / preds / trues / metrics` 保存为 `.npy` 至
  `save_dir/saved/`，作为单实验证据（见 `docs/experiments/`）。

## 待办
- 明确 weather 任务的 `spatial_norm` 与 `metric_threshold` 设置。
- 制定与 CIRA-Diff 的并排对比表模板。
