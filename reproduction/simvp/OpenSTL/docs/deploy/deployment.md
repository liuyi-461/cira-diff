# 部署流程（Deploy）

## 步骤
1. 环境就绪（`docs/dev/environment.md`）。
2. 训练得到 `best.ckpt`（`docs/training/training_config.md`）。
3. 推理：`tools/test.py` 或 `BaseExperiment.test()` 加载 checkpoint。
4. 产出预测与指标，归档至 `docs/experiments/`。

## 注意
- 推理时使用与训练一致的 mean / std 归一化。
- 确定性模型适合批量离线推理。
