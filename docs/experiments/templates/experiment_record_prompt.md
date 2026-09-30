# 实验记录自动生成标准提示词

> 用途：在新的 Codex 窗口或其他代码智能体中，完成代码修改、实验运行、数据审计、推理或评估后，依据本项目统一标准补齐实验记录及相关知识库文件。
>
> 语言要求：生成或修改的 Markdown 文档必须使用中文；代码、命令、路径、配置键名和原始日志内容按实际情况保留。

## 推荐调用方式

将下面的提示词复制到新窗口中使用。`[本次任务补充信息]` 部分可按实际情况填写；如果没有补充信息，保留为“未提供”，不要让智能体自行猜测。

```text
你现在负责为当前卫星云图预报科研项目补齐本次工作的实验记录和知识库更新。

请先完整阅读以下项目标准文件，再开始判断和修改：

1. AGENTS.md
2. docs/project/knowledge.md
3. docs/project/current_state.md
4. docs/project/decisions.md
5. docs/project/architecture.md
6. docs/project/knowledge_gaps.md
7. docs/research/questions.md
8. docs/research/hypotheses.md
9. docs/experiments/templates/experiment_card.md
10. docs/experiments/registry.md
11. docs/evaluation/evaluation_protocol.md

如果本次工作涉及某个复现模型，还要阅读：

- reproduction/<模型名>/README.md
- reproduction/<模型名>/STATUS.md
- reproduction/<模型名>/UPSTREAM.md

本次任务补充信息：

- 工作类型：<代码修改 / 数据审计 / 环境检查 / 冒烟测试 / 训练 / 推理 / 自回归 rollout / 评估 / 复现实验 / 其他>
- 实验目的或对应 RQ/HYP：<未提供>
- 使用的数据、切分和版本：<未提供>
- 使用的命令或入口：<未提供>
- 输出目录、日志、checkpoint、图表或指标文件：<未提供>
- 用户特别要求：<未提供>

请按以下流程执行：

一、先审计事实

1. 查看 git status、git diff、最近提交、实际修改过的代码和配置。
2. 检查本次是否真的运行了命令，以及是否存在可验证的日志、指标、图像、checkpoint 或其他产物。
3. 区分“代码已经修改”“命令已启动”“训练完成”“推理完成”“评估完成”“科学实验完成”，不能把其中一种自动等同于另一种。
4. 检查数据路径、数据切分、随机种子、依赖环境、checkpoint、输出目录和指标是否可追溯。
5. 若证据不足，明确写为 [UNKNOWN]、Not Recorded 或 TODO，不要补写猜测值。

二、判断是否建立实验卡片

1. 如果只有代码修改、文档修改、环境安装或静态检查，没有形成实验结果：
   - 不要伪造完成的 EXP 编号或科学结论；
   - 更新必要的任务报告、current_state.md、knowledge_gaps.md 或 reproduction 状态；
   - 可以记录为“准备工作”“环境检查”或“未完成实验”。

2. 如果已经有可验证的实验过程和产物：
   - 在 docs/experiments/<对应实验族>/ 下，依据
     docs/experiments/templates/experiment_card.md
     创建下一个未占用编号的 EXP-XXX-<short-name>.md；
   - 填写问题、假设、数据契约、代码/配置、命令、环境、产物、结果、解释、局限、决策和复现信息；
   - 更新 docs/experiments/registry.md；
   - 只有在证据确实支持时，才更新 current_state.md、knowledge.md、conclusions.md 或 research/questions.md；
   - 如果产生了新的未知项、冲突或阻塞，更新 knowledge_gaps.md。

3. 如果是已有复现模型：
   - 更新 reproduction/<模型名>/STATUS.md；
   - 严格区分 Surveyed、Code Available、Downloaded、Environment Working、Inference Reproduced、Training Reproduced、Evaluation Reproduced；
   - 不要因为代码可下载、能够 import 或训练命令能够启动，就宣称推理、训练或评估已经复现。

三、证据和表述规则

- 使用项目规定的证据标签：[FACT]、[EVIDENCE]、[HYPOTHESIS]、[RESULT]、[CONCLUSION]、[DECISION]、[ISSUE]、[PLAN]、[UNKNOWN]、[CONFLICT]。
- 论文中的结果只能作为背景或待验证主张，不能直接写成当前项目结果。
- “视觉上看起来合理”不能替代指标或统计检验。
- 训练阶段的 rollout 与推理阶段的自回归 rollout 必须分开记录。
- 未记录的时间、硬件、随机种子、checkpoint、阈值、指标、数据版本和 DOI 必须保持 Unknown 或 Not Recorded。
- 保留历史内容和来源，不要删除已有记录来制造一致性；若发现矛盾，显式记录 [CONFLICT]。
- 不修改受保护的 cira_diff/ 和 scripts/Chase_2025/ 实现，除非任务明确授权并经过审查。
- 不提交数据、checkpoint、TensorBoard 日志和生成媒体，除非任务明确要求。

四、完成后必须返回

1. 本次实际修改的文件列表；
2. 新建或更新的实验编号；
3. 每个结论对应的证据来源；
4. 实际结果与解释/推断的区分；
5. 仍然未知、冲突或无法验证的内容；
6. 执行过的验证命令及其结果；
7. 是否需要用户补充数据、日志、checkpoint 或路径；
8. 推荐的下一步实验。

不要只返回“已完成”。如果没有足够证据生成完整实验记录，请生成一份明确标注未完成项的记录，并说明缺少什么。
```

## 使用后的最小补充信息

为了让自动记录更准确，建议每次实验结束后至少补充以下信息：

```text
本次工作名称：
对应 RQ/HYP：
工作类型：
实际运行命令：
数据路径和切分：
代码或配置提交：
日志/指标/checkpoint/图像路径：
是否完成训练、推理和评估：
希望智能体重点更新的文件：
```

如果某一项没有记录，直接填写“未记录”，不要为了让模板完整而估计。

## 标准文件职责

| 文件 | 职责 |
|---|---|
| `AGENTS.md` | 仓库级安全边界、证据等级和受保护实现规则 |
| `docs/experiments/templates/experiment_card.md` | 单个实验的字段和记录结构 |
| `docs/experiments/registry.md` | 实验编号、状态和索引 |
| `docs/evaluation/evaluation_protocol.md` | 评估口径、指标和报告要求 |
| `docs/project/current_state.md` | 当前已经确认的项目状态 |
| `docs/project/knowledge.md` | 事实、证据、假设、结果和未知项的表达标准 |
| `docs/project/knowledge_gaps.md` | 未解决问题、冲突、阻塞和证据缺口 |
| `docs/research/questions.md` | 研究问题及其与实验的对应关系 |
| `docs/research/hypotheses.md` | 可检验假设及其验证状态 |
| `reproduction/<模型名>/STATUS.md` | 外部模型复现阶段和证据状态 |

## 维护规则

本文件是“调用流程模板”，不是实验事实库。实验事实仍必须写入实验卡片、registry、项目状态或复现状态文件中。若项目标准发生变化，应先更新对应标准文件，再同步修改本提示词，避免在提示词中复制一套与仓库不一致的规则。

