# DEC-001 项目第一阶段采用官方 CIRA-Diff 为 baseline

## 日期

2026-09-17

## 背景

项目刚开始，需要先建立可核验的基线，避免在未理解数据和 rollout 机制前直接修改模型。

## 决策内容

[DECISION] 第一阶段只复现官方 CIRA-Diff 任务和实现；暂不修改核心模型。

## 备选方案

- 直接转向 ConvLSTM/PredRNN/SimVP 等多帧到多帧模型。
- 先设计新的物理约束或多尺度 loss。

## 原因

官方论文、代码和公开数据具有直接的可追溯关系，适合作为后续 temporal formulation、结构保持和概率预报实验的共同参照。

## 影响

后续改进必须先能在相同数据、时间间隔、空间 patch 和 evaluation protocol 下复现 baseline，否则无法解释收益来源。

# DEC-002 以证据等级管理科研知识

## 日期

2026-09-17

## 背景

历史 AI 调研中有很多有价值的线索，但论文结论、源码行为和本项目实验不能混用。

## 决策内容

[DECISION] 文档统一使用 `[FACT]`、`[EVIDENCE]`、`[HYPOTHESIS]`、`[RESULT]`、`[CONCLUSION]`、`[DECISION]`、`[ISSUE]`、`[PLAN]`、`[UNKNOWN]` 标签。

## 影响

新成员或 AI 必须能从文档中区分“论文声称什么”“当前代码做什么”“本项目实际验证了什么”。

# DEC-003 本地只保留开发子集

## 日期

2026-09-17

## 背景

完整数据在服务器，本地数据目录目前没有真实数据。

## 决策内容

[DECISION] 本地 `data-cira-diff/` 只保存用于 inspection、Dataset/DataLoader smoke test 的最小子集；远程原始数据只读，不把本地子集当作完整数据集规模。

## 影响

所有实验记录必须同时写明 `Remote Full/Open Dataset` 与 `Local Development Subset`，并通过 `.gitignore` 避免数据 payload 被提交。
