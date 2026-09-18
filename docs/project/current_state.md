# 当前版本

`baseline-audit-2026-09-18`（数据审计与最小 subset 同步标记，非模型版本）

## 已完成

- 阅读并核对 `docs/README.md`、仓库 README、源码和 `scripts/Chase_2025/` 入口。
- 建立 CIRA-Diff baseline 的项目、研究、训练和开发文档初稿。
- [EVIDENCE] 核对官方论文：GOES-16 ABI Channel 13、10 min、256×256 patch、两帧条件、单步预测后 18 步 rollout 到 3 h。
- 检查 `.gitignore`，加入本地数据 payload 排除规则。
- [EVIDENCE] 远端 `/data1/satcast/` 已验证可访问，host 为 `test-ai`。
- [EVIDENCE] train/validation/test Zarr 的 shape、dtype、chunks、规模和 sample 组织已完成审计。
- 已通过 rsync 同步 ordinary train、CorrDiff train、validation、test 的最小本地 subset，共约 145 MB。

## 当前实现

- 上游仓库代码已存在，HEAD 为 `e59d4ba`。
- `cira_diff/dataset.py` 提供 ZarrDataset。
- `cira_diff/edm.py` 提供 EDMPrecond、EDMLoss、edm_sampler。
- 完整的旧训练脚本在 `scripts/Chase_2025/`。
- `cira_diff/train_Diff.py` 和 `train_CorrDiff.py` 的 main 中关键加载/建模/训练代码目前被注释，不能直接视为可运行入口。

## 当前问题

- [BLOCKED] 本地 Python 环境缺少 `torch` 和 `zarr`，尚未运行本地 Dataset/DataLoader smoke test。
- [UNKNOWN] 每个 patch 的原始 GOES 文件时间戳不在 Zarr metadata 中，跨 sample 的原始序列连续性无法仅从 `/data1/satcast/` 还原。
- [ISSUE] 没有独立、经过测试的 rollout/evaluation CLI。
- [ISSUE] 当前训练脚本存在硬编码路径、硬编码 mean/std、`lr_warmup_steps` 命名不一致等复现风险。
- [UNKNOWN] 官方 diffusion 训练是否做过额外 rollout fine-tuning；当前代码没有该路径，论文只明确描述了单步条件与 autoregressive inference。

## 下一步计划

1. 在已有环境或隔离环境中补齐最小 `torch`/`zarr` 依赖，运行本地 Dataset/DataLoader smoke test。
2. 实现独立 single-step、rollout、metrics 评估脚本，并用本地 validation/test truth 检查 18-step 序列对齐。
3. 将官方训练入口收敛到可配置命令，先做极小 smoke run，再做正式 baseline reproduction。
