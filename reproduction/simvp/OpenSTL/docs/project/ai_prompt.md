# AI 协作规范

## AI 角色
- 协助理解 OpenSTL / SimVP 代码与文档。
- 在 reproduction 工作流中补全文档、编写 adapter、修复测试、运行训练 / 评测。
- 维护本 `docs/` 体系与 `README copy.md` 规范的对应关系。

## 修改原则
- 优先最小修改，不改变 OpenSTL 已有接口。
- 不移动 / 删除 `docs/en/`（上游 provenance）与仓库根 `README copy.md`。
- 修改前说明影响范围（参照 `docs/project/decisions.md`）。

## 项目特殊约束
- 科学约束：SimVP 直接多帧输出与 CIRA-Diff 自回归对比须公平（DEC-002）。
- 工程限制：reproduction 未达 `Inference Reproduced` 前不得声称 locally reproduced。
- 数据限制：数据 adapter 与评测 adapter 未编写前不做 dataset adaptation。
