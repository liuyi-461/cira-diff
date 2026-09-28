# 实验记录

## EXP-001

日期：2026-09-17

实验目的：官方 CIRA-Diff baseline reproduction；确认数据读取、单步预测、采样和 3 h autoregressive rollout。

数据：

- Remote Full/Open Dataset: `26fall-AIclass:/data1/satcast/`（当前 SSH 会话无法访问，待修复）。
- Local Development Subset: `data-cira-diff/`（尚未下载）。

方法：Diff、CorrDiff、U-Net、persistence 的官方实现路径；首轮只做 Dataset/DataLoader smoke test 和极小训练，不声称已复现论文结果。

参数：待数据审计后填写；必须记录 config 文件、Git commit、GPU、seed、batch size、learning rate、noise/sampling 参数和 normalization。

结果：完成远端只读数据审计和本地 subset 同步；未运行模型训练。远端 ordinary train 为 35,595 个单步样本，validation/test 各为 1,024 个 18-frame truth 样本；本地保留 train 45、CorrDiff train 45、validation 32、test 32 个样本。

结论：`[RESULT]` 数据组织已确认：三帧窗口只属于 ordinary train 单步样本，3 h truth 直接来自 validation/test 的 18-frame `output_images`。模型仍未训练，不能填写论文指标或“复现成功”。

下一步：补齐本地 `torch`/`zarr` 依赖后运行 Dataset/DataLoader smoke test，建立可运行命令和评估输出。

复现教程：待补充，目标文档为 `docs/training/dataset.md`、`docs/training/training_config.md` 和 `docs/training/evaluation.md`。

## Canonical experiment layer

上面的历史 EXP-001 记录保留。canonical registry 和详细实验卡位于 [`docs/experiments/registry.md`](../experiments/registry.md) 与 [`docs/experiments/cira_diff/EXP-001-cira-diff-data-audit.md`](../experiments/cira_diff/EXP-001-cira-diff-data-audit.md)。`docs/training/` 文档继续作为 training-specific historical context，不构成第二套 experiment source of truth。
