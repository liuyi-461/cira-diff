# 实验记录

实验是 evidence package，不是普通 prose plan。每条记录必须分开写 `Results`、`Interpretation`、`Conclusion` 和 `Decision`。

## 实验卡

使用 `templates/experiment_card.md`，文件命名为 `EXP-XXX-short-name.md`。

## 实验家族

- `cira_diff/`：当前 baseline 和 reproduction gates；
- `temporal_history/`：history length 与 temporal formulation；
- `baselines/`：deterministic/probabilistic comparison；
- `evaluation/`：metric 和 protocol validation；
- `ablations/`：受控 component tests。

run-local config、日志、输出和大型 artifact 放在顶层 `experiments/`，canonical narrative card 只保留小型且可审阅内容。
