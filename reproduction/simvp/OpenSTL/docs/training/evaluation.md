# 评价指标（Training）

## 指标口径
- MAE / MSE，由 `openstl/core/metric.py` 计算。
- weather 任务开启 `spatial_norm`；支持 `channel_names` 与 `metric_threshold`。

## 与 CIRA-Diff 对比
- 必须统一多步预测协议（DEC-002）：direct multi-step vs autoregressive。
- 报告 FLOPs / 吞吐（由 `display_method_info` 的 `FlopCountAnalysis` 产出）。

## 产物
- `on_test_epoch_end` 保存 `metrics.npy / inputs.npy / trues.npy / preds.npy`
  至 `save_dir/saved/`。
