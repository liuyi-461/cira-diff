# Reproduction 工作流与状态语义

本目录是 SimVP reproduction 的 provenance 与状态追踪中枢，遵循 cira-diff 的
reproduction 规范（见仓库根 `README copy.md` 及 `reproduction/README.md`）。

## 状态词汇（线性迁移）
`Surveyed` → `Code Available` → `Downloaded` → `Environment Working` →
`Inference Reproduced` → `Training Reproduced` → `Evaluation Reproduced`。

每次状态迁移都必须记录：**command、revision、artifact、deviation note**。
仅有论文 / 仓库链接不构成 reproduction result。

## 当前条目状态
- 角色：low-cost deterministic multi-frame comparison。
- 当前状态：`candidate only`；除 `Surveyed` 外所有 gate 为 `Not Started`。
- 待审计：Paper、Official repository 与 OpenSTL 的关系、Upstream commit。
- 待填写：Dataset、Original task、Environment、Weights、Data adapter、Evaluation adapter。

## 公平性约束
- `Known issues`：direct multi-step output 必须与 autoregressive CIRA-Diff 公平比较（DEC-002）。

## 登记维护
- 在达到 verified status 并附 command / artifact 前，不得将本条目描述为 locally reproduced。
- 状态变更同步更新 `docs/project/current_state.md` 与 `reproduction/README.md`。
