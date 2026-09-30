# 科学问题与研究假设

## 问题描述
在卫星云图预测任务中，扩散类生成模型（CIRA-Diff，自回归）与确定性端到端模型
（SimVP，直接多帧）的预测质量与成本差异如何？确定性低成本基线能否作为公平对照？

## 当前认识
- SimVP 直接输出全部未来帧，训练用 MSELoss，确定性强、推理成本低。
- CIRA-Diff 为自回归扩散，单步需多步去噪，成本更高但可能捕捉更复杂的分布。

## 假设
在统一多步预测协议下（见 DEC-002），SimVP 在 MAE / MSE 上可提供具有竞争力的
确定性基线，且推理吞吐显著优于 CIRA-Diff，适合作为 low-cost comparison。

## 验证方法
- 在目标数据集上复现 SimVP 基线指标（MAE / MSE，见 `docs/evaluation/`）。
- 与 CIRA-Diff 在同一 `aft_seq_length` 下对比，记录 FLOPs / 吞吐。
- 证据包写入 `docs/experiments/`。
