# 受保护迁移计划

状态：active plan；尚未迁移 CIRA-Diff 核心源码  
日期：2026-09-22

## 原则

[DECISION] 在 imports、upstream compatibility、checkpoint loading、single-step inference 和 rollout alignment 得到验证前，保留 `cira_diff/` 作为 working implementation。Lab 架构围绕它增加，而不是强行改造它。

## 当前路径到目标角色

| 当前路径 | 目标角色 | 当前行动 | Promotion gate |
| --- | --- | --- | --- |
| `cira_diff/` | 受保护的 CIRA-Diff implementation | 保留原位 | verified API 和 regression tests |
| `scripts/Chase_2025/` | 旧/reference scripts | 保留并记录 | reproducible config + artifact |
| `config_CorrDiff.py` | 旧 config | 保留，记录 mismatch | validated replacement config |
| `docs/training/` | 历史 training knowledge | 保留，链接到 `docs/data/` 和 `docs/evaluation/` | reviewed canonical replacement |
| `src/satforecast/data/` | stable data contracts/adapters | 仅 scaffold | contract tests 和 provenance |
| `src/satforecast/models/` | promoted model interfaces | 仅 scaffold | reproduction/reference behavior fixed |
| `src/satforecast/evaluation/` | shared evaluator | 仅 scaffold | metric tests 和 artifact contract |
| `configs/` | stable configs | 仅 scaffold | config 与 executable path 一致 |
| `reproduction/` | upstream provenance、adapter 和复现记录 | 添加 records，不下载 | status gate 通过 |

## 迁移阶段

1. **Audit**：已完成，见 `repository_audit.md`。
2. **Contract**：定义 sample、forecast、rollout 和 evaluation artifact schema，不修改模型。
3. **Reference smoke test**：验证依赖，读取一个 Zarr sample，构造模型并运行一次受控 prediction。
4. **Rollout extraction**：提取 notebook 行为为测试脚本，同时保持原 forecast semantics。
5. **Evaluation adapter**：将预测的 18-step 序列与 truth 对齐比较，使用明确 protocol。
6. **Promotion**：只有通过上述 gate，才考虑将通用小工具 promotion 到 `src/satforecast/`。

## 本次迁移明确不做

- 大规模移动 `cira_diff/`；
- 重写 EDM、CorrDiff 或训练逻辑；
- 下载新的 baseline 或 checkpoint；
- 声称 reproduction 成功；
- 在不保留 provenance 的情况下替换历史文档。

## 未解决的迁移风险

- 硬编码路径和机器相关 GPU selection；
- `llr_warmup_steps` 与 `lr_warmup_steps` 命名不一致；
- 旧脚本 mean/std 与数据审计统计量可能不一致；
- 缺少独立 rollout/evaluation entry point；
- derived Zarr 的 timestamp/source-index lineage 不完整。
