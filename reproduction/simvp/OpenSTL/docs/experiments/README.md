# 单个实验证据包

本目录存放每个已运行实验的完整证据包（evidence package），与 `docs/research/experiments.md`
中的 `EXP-XXX` 记录一一对应。

## 证据包内容约定
- `EXP-XXX/` 子目录，包含：
  - `config.yaml` / 使用的 config 路径
  - 训练 / 测试日志
  - 指标结果（`metrics.npy` 等，由 `on_test_epoch_end` 产出）
  - 可视化图（`docs/figs/` 或本目录内）
  - 复现命令（command）与 revision

## 与 evaluation 的关系
指标计算与协议见 `docs/evaluation/`；实验条件记录见 `docs/research/experiments.md`。

## 当前状态
- 暂无实验证据包（reproduction 仍处 candidate only）。
