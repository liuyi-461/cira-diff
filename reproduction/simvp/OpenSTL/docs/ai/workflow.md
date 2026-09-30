# AI 协作流程

任何 AI 参与项目时：

## 阅读顺序
1. `docs/project/context.md`
2. `docs/project/current_state.md`
3. `docs/project/decisions.md`
4. `docs/project/architecture.md`
5. `docs/research/`

## 修改代码前必须确认
1. 当前设计约束（`docs/project/ai_prompt.md`）。
2. 已存在技术决策（`docs/project/decisions.md`）。
3. 是否会影响已有实验结果（`docs/experiments/`）。

## 状态维护
- reproduction 状态迁移须记录 command / revision / artifact / deviation
  （`docs/reproduction/`）。
- 文档随代码同步更新（架构→architecture.md，决策→decisions.md，状态→current_state.md，
  经验→knowledge.md）。
