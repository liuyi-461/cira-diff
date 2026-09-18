# AI Development Workflow

## 1. Purpose

本文件定义 CIRA-Diff 项目中 **人类研发人员、Claude、Codex 及其他 AI Agent 的标准协作流程**。

目标是保证：

* AI 在修改代码前理解当前项目状态；
* 不同 Agent 使用一致的项目上下文；
* 需求、代码、决策和文档保持同步；
* 历史对话不会被误认为当前代码事实；
* 每一次修改都能够追踪“为什么改、改了什么、现在是什么状态”。

---

## 2. Source of Truth

项目中的信息优先级如下：

```text
Current Repository
        ↓
current_state.md
        ↓
decisions.md
        ↓
context.md / architecture.md
        ↓
task-specific docs
        ↓
historical prompts / AI conversations
```

当信息冲突时：

> **当前代码和用户最新明确要求具有最高优先级。**

历史 Claude / ChatGPT 对话、ZIP 交付记录和旧 Prompt 只能作为参考，不自动视为当前实现。

---

## 3. Standard Task Workflow

每一个研发任务按照以下流程执行：

```text
User Requirement
      ↓
Context Review
      ↓
Current Code Inspection
      ↓
Task Classification
      ↓
Design / Implementation
      ↓
Validation
      ↓
Repository Update
      ↓
Documentation Sync
```

---

## 4. Step 1 — Requirement Definition

用户首先明确当前任务。

任务应尽量说明：

* 要解决什么问题；
* 当前表现是什么；
* 希望最终达到什么效果；
* 哪些已有行为不能改变；
* 是否允许修改接口或架构。

如果需求已经明确，AI 不应重复询问已经知道的信息。

对于复杂任务，先输出：

```text
Current understanding
Files/modules likely involved
Proposed modification scope
Potential risks
```

再开始修改。

---

## 5. Step 2 — Read Project Context

开始任务前，AI 应至少阅读：

```text
docs/project/context.md
docs/project/current_state.md
docs/project/decisions.md
```

涉及系统结构时继续阅读：

```text
docs/project/architecture.md
```

涉及未来计划时阅读：

```text
docs/project/roadmap.md
```

涉及术语时阅读：

```text
docs/project/glossary.md
```

涉及项目知识时阅读：

```text
docs/project/knowledge.md
```

---

## 6. Step 3 — Inspect Actual Repository

AI 不得仅根据文档或历史对话判断代码状态。

必须实际检查：

```text
project tree
target files
git status
existing interfaces
existing tests
configuration
```

尤其需要确认：

* 文件是否真的存在；
* 历史对话中声称生成的代码是否已经合入；
* 当前 Web 是否仍使用 mock；
* API 是否真实存在；
* 配置是否与文档一致。

原则：

> **Repo State > Chat History**

---

## 7. Step 4 — Task Classification

根据任务类型选择最合适的工作方式。

### 7.1 Product / Requirement / Architecture

优先使用 Claude 或 ChatGPT 进行：

* 需求梳理；
* 产品交互分析；
* 页面信息结构；
* 架构讨论；
* 技术方案比较；
* 文档设计。

输出应以：

```text
Requirement
Architecture
Interface
Decision candidate
```

为主，不直接假设代码已经修改。

---

### 7.2 Precise Code Modification

优先使用 Codex：

* 修改现有代码；
* 接真实数据；
* 重构低质量函数；
* 增加 API；
* 修 bug；
* 增加测试；
* 前端局部调整；
* 后续算法实现。

要求：

> 优先 patch 当前代码，而不是重新生成整个项目。

---

### 7.3 Research / Algorithm Exploration

优先使用 ChatGPT / Claude 进行：

* 文献调研；
* baseline 设计；
* 算法比较；
* 评价指标设计；
* 气象机理分析。

确定方案后再交给 Codex 实现。

---

## 8. Step 5 — Design Before Modification

涉及以下内容时，修改代码前先明确设计：

* API Contract；
* 数据结构；
* 数据库 Schema；
* 新模块边界；
* 新的数据流；
* 可能影响多个页面的状态管理；
* 模型输入输出。

对于小型 UI 修改、明确 bug 修复等，可以直接修改。

---

## 9. Step 6 — Minimal-change Principle

代码修改遵循：

```text
Understand existing behavior
        ↓
Find smallest modification point
        ↓
Add adapter/service if possible
        ↓
Avoid unrelated refactoring
```

特别禁止：

* 为了“代码更优雅”整体重构 Web；
* 修改用户没有要求的功能；
* 为未来需求提前引入复杂基础设施；
* 将实验参数写死为架构约束。

---

## 10. Real Data Integration Workflow

当前项目最高优先级任务遵循：

```text
Raw FTP Data
      ↓
Inspect Real Sample
      ↓
Reader / Parser
      ↓
Validation
      ↓
Unified Data Model
      ↓
Backend Service
      ↓
FastAPI
      ↓
Frontend API Client
      ↓
Existing Web Component
```

不要反过来先改前端结构。

---

## 11. Mock → Real Migration

当前 Web 已有 mock 数据。

迁移方式：

```text
Existing Mock Contract
        ↓
Define Stable API Contract
        ↓
Backend adapts real data
        ↓
Frontend switches provider
```

优先保持：

```text
Component props
UI behavior
interaction logic
```

不变。

允许：

```text
EC        → real
METAR     → real
Prediction → mock/simple
Training  → mock
```

即 Hybrid 模式。

---

## 12. Validation Workflow

每次代码修改后至少执行与任务相关的验证。

### Backend

包括：

* parser unit test；
* service test；
* API TestClient；
* 异常输入；
* 缺测情况；
* 真实样例。

### Frontend

包括：

* TypeScript compile；
* `npm run build`；
* 页面真实数据展示；
* loading/error/empty 状态；
* 交互回归。

### Data

需要验证：

* 时间；
* 单位；
* 经纬度；
* missing value；
* special value；
* forecast hour；
* station mapping。

### Algorithm

后续至少验证：

* baseline；
* train/validation split；
* metrics；
* forecast lead time；
* station-level performance。

---

## 13. Completion Report

AI 完成任务后必须说明：

### Changed

```text
Modified files
Added files
Deleted files
```

### Result

说明：

* 实现了什么；
* 哪些需求已经完成；
* 当前真实运行链路是什么。

### Validation

说明：

```text
What was tested
How it was tested
Test result
```

### Remaining

说明：

* 尚未完成的问题；
* 当前限制；
* 下一步建议。

不得只说：

> “已经完成。”

---

## 14. Documentation Sync

修改代码后根据变化类型同步项目文档。

### Update `current_state.md`

当：

* 新功能已经真正进入仓库；
* 某模块从 mock 变 real；
* API 已经落地；
* 项目阶段发生变化。

### Update `decisions.md`

当：

* 一个候选方案正式确定；
* 接口结构稳定；
* 数据库/框架/模型等发生重要选择。

### Update `architecture.md`

当：

* 模块边界发生变化；
* 数据流变化；
* 服务关系变化。

### Update `roadmap.md`

当：

* 阶段完成；
* 优先级改变；
* 新阶段进入执行。

### Update `knowledge.md`

当：

* 获得新的稳定业务知识；
* 明确真实数据格式；
* 明确变量语义；
* 发现可复用的实现规律。

---

## 15. Decision Lifecycle

一个技术想法按以下状态演进：

```text
Idea
 ↓
Open Issue
 ↓
Experiment / Discussion
 ↓
Decision
 ↓
Implementation
 ↓
Current State
```

对应文档：

```text
Open Issue      → context.md / roadmap.md
Candidate       → research/dev notes
Decision        → decisions.md
Implementation  → code
Current Fact    → current_state.md
```

这样避免“讨论过 = 已决定”。

---

## 16. AI Conversation Handling

历史 AI 对话可以作为：

* 需求来源；
* 设计思路；
* 候选实现；
* 决策背景。

但不得作为：

* 当前仓库状态证明；
* 测试通过证明；
* 已部署证明。

当历史 AI 声称：

```text
Created 10 files
Tests passed
Generated ZIP
```

必须检查当前仓库后才能写入 `current_state.md`。

---

## 17. Current Project Workflow

当前推荐实际执行链路：

```text
需求讨论
  ↓
Claude / ChatGPT
  ↓
形成明确任务与接口设计
  ↓
Codex 打开服务器仓库
  ↓
读取 project docs
  ↓
检查真实代码
  ↓
最小修改
  ↓
运行测试
  ↓
用户验证 Web / 数据结果
  ↓
更新 current_state / decisions
```

---

## 18. Current Development Order

当前 CIRA-Diff 项目严格按以下顺序推进：

```text
① Real EC Data
② Real METAR Data
③ Backend API
④ Real Data → Web
⑤ Web Business Refinement
⑥ Simple Prediction
⑦ Formal AI Algorithm
⑧ Training & Evaluation
⑨ Weather Event Detection
⑩ TAF Generation
```

除非用户明确调整优先级，否则 AI 不应跳过前置阶段。

---

## 19. Definition of Done

一个任务只有同时满足以下条件，才视为完成：

```text
Code exists in current repository
        +
Behavior matches requirement
        +
Relevant tests pass
        +
No unintended regression
        +
Documentation updated when necessary
```

“AI 曾经生成过代码”不属于完成条件。
