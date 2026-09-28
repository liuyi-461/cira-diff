# 训练配置

## 当前代码可见配置

来自 `config_CorrDiff.py` 与 `scripts/Chase_2025/`：

| 参数 | 当前值/行为 | 状态 |
|---|---|---|
| image size | 256 | `[FACT]` |
| train batch size | 45 | `[FACT]`，硬件相关 |
| epochs | 1000（旧脚本） | `[FACT]`，不是本项目已验证设置 |
| gradient accumulation | 2 | `[FACT]` |
| learning rate | 1e-4 | `[FACT]` |
| mixed precision | fp16 | `[FACT]` |
| P_mean / P_std | -1.2 / 1.2 | `[FACT]` |
| sigma_data | 0.5 | `[FACT]` |
| EDM sampler steps | 18 | `[FACT]` 默认值 |
| DataLoader workers | 8 | `[FACT]` 旧训练脚本 |

## 复现注意事项

- `dataset_path`、`output_dir`、GPU id 是绝对路径/机器相关设置，不能直接复用。
- `config_CorrDiff.py` 定义 `llr_warmup_steps`，而训练脚本引用 `lr_warmup_steps`；运行前必须修正或显式记录该差异。
- 训练脚本写死了若干 mean/std（例如 `-0.0009`、`0.0807`），必须和数据审计结果比对。
- 训练循环会保存 checkpoint，但当前代码没有独立验证集 DataLoader 与完整评估协议。
- smoke test 要先把 epochs、batch size、workers、采样步数和输出目录设为小值，并保存完整 config。

## 推荐配置追踪字段

每次训练至少记录：Git commit、config 路径、远程/本地数据路径、数据 hash、normalization、seed、GPU、batch size、累积步数、学习率、noise 参数、sampler 参数、checkpoint 路径和运行命令。
