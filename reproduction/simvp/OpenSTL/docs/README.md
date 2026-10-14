# OpenSTL / SimVP 文档体系

本目录遵循 Satellite Forecasting Lab 的统一文档规范。规范原文保留在仓库根的
`README copy.md`（来自 CIRA-Diff knowledge-base commit），作为 provenance，**不得静默删除或修改**。

## 范围与 Canonical 位置

| 范围 | Canonical 位置 |
| --- | --- |
| 项目身份、架构、状态和决策 | `docs/project/` |
| 科学问题、假设、文献和方法 | `docs/research/` |
| 数据合同、谱系、预处理和切分 | `docs/data/` |
| 验证与评价指标 | `docs/evaluation/` |
| 单个实验的证据包 | `docs/experiments/` |
| Reproduction 工作流和状态语义 | `docs/reproduction/` |
| 可复用技术能力 | `docs/skills/` |
| AI 工具使用说明 | `docs/ai/` |
| 开发文档 | `docs/dev/` |
| 模型训练相关 | `docs/training/` |
| 部署文档 | `docs/deploy/` |
| 运维文档 | `docs/ops/` |
| 文档图片 | `docs/figs/` |

> 注意：上游 OpenSTL 自带文档位于 `docs/en/`，属于第三方 provenance。本规范新增的
> 结构不应覆盖、移动或删除 `docs/en/` 下的任何内容。

## 代码入口（自研脚本）

| 内容 | 位置 |
| --- | --- |
| **训练 / 评估 / 对比 / 报告脚本**（含完整 CLI、实测结论、9 条踩坑记录） | `tools/README.md` |
| Slurm 提交脚本（环境自检 + GPU 独占 + 依赖链） | `test_dl/*.slurm` |
| satcast 数据适配器（上游唯一新增文件） | `openstl/datasets/dataloader_satcast.py` |
| 训练超参（唯一来源，改动必须登记） | `configs/satcast/SimVP.py` |
| 主线结论材料（精度 vs 真实性） | `docs/research/forecast-realism-report.md` |
| 图表集 | `docs/figs/README.md` |

> 上游 `openstl/` 其余部分**未做任何修改**（见 `docs/project/decisions.md` DEC-003）。

## 文档书写原则（摘要）

- 面向未来阅读，记录结论而非过程。
- 重要修改须可追溯：日期、修改人、原因、影响范围。
- 代码/架构/决策/状态变化须同步更新对应文档（见 `README copy.md` 第 8 节）。

## AI 协作推荐阅读顺序

1. `docs/project/context.md`
2. `docs/project/current_state.md`
3. `docs/project/decisions.md`
4. `docs/project/architecture.md`
5. `tools/README.md`（自研代码入口：CLI、用法、踩坑记录）
6. `docs/research/`
