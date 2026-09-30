# 部署相关规范（Skill）

## 推理部署要点
- 导出 `best.ckpt`（`BestCheckpointCallback` 产出）后做推理。
- 确定性模型（SimVP）适合批量离线推理，吞吐优势明显。

## 复用建议
- 推理脚本可参考 `tools/test.py` 与 `BaseExperiment.test()`。
- 部署产物与指标证据包一并归档至 `docs/experiments/`。
