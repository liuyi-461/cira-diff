# Codex 协作指南

## 角色

Codex 在本项目中的主要职责是：

- 在真实仓库上完成精确代码修改；
- 真实数据接入；
- API 实现；
- 测试；
- 重构已有低质量实现；
- 根据明确 UI 需求修改 TypeScript Web；
- 后续正式算法工程实现。

相比重新生成项目，优先在现有代码上做 patch。

---

## 强制启动流程

开始任何任务前：

```text
1. Read docs/project/context.md
2. Read docs/project/current_state.md
3. Read docs/project/decisions.md
4. Read docs/project/architecture.md
5. Inspect actual files
6. Check git diff/status
```

如果历史文档写“已实现”，但仓库不存在对应代码：

> 以仓库为准，并报告差异。

---

## 当前最高优先级

不要先写正式 AI 模型。

当前：

```text
P0 Real Data Integration
P1 Web Business Refinement
P2 Simple Prediction
P3 Formal Algorithm
```

---

## 代码修改原则

### Minimal Patch

用户未要求的地方不改。

尤其：

- 不重写现有 React；
- 不随意改文件命名；
- 不改变已有交互；
- 不替换技术栈；
- 不重构整个工程只为“更优雅”。

### Backend Boundary

目标：

```text
Raw File
→ Reader/Parser
→ Service
→ Schema
→ Router
```

Router 只做 HTTP 适配。

复杂解析不能写在 Router。

### API Stability

Frontend 不应该知道：

- 文件路径；
- NPZ key；
- SQLite table；
- GRIB message；
- model checkpoint。

---

## Recommended First Coding Sequence

### Step 1 — Inspect Real Data

读取真实：

- EC 文件样例；
- METAR JSON 样例；
- `cfg/stations.yaml`
- `cfg/variables.yaml`
- `extractor/`
- 当前 Web mock contract。

### Step 2 — Implement Backend Minimum Loop

优先：

```text
FastAPI main
METAR parser + SQLite + API
EC field API
```

### Step 3 — Frontend API Layer

新增统一：

```text
web/src/api/
```

然后逐个替换 mock。

不要一次把所有页面一起改。

### Step 4 — Tests

至少：

- Parser unit tests
- DB ingestion tests
- EC extraction tests
- API TestClient tests
- TypeScript build

---

## Real/Mock/Hybrid Rule

在切换期允许：

```text
EC        real
METAR     real
Prediction simple/mock
Training  mock
```

不要为了“全真实”提前实现正式模型。

---

## Configuration Rule

不得硬编码：

```text
ZSSS
64×64
/data/product/...
forecast hour
special value
variable list
```

从配置读取。

---

## Algorithm Rule

正式算法开始前：

- 不继承早期 UNet/multi-head 方案；
- 先做 baseline；
- 用真实数据定义 task；
- 评估不同 forecast hour；
- 明确 classification vs regression；
- 处理 circular wind direction；
- 明确 categorical weather/cloud variables。

---

## Completion Report

每次修改后输出：

```text
Changed files
Why
How to run
How tested
Test result
Remaining issues
Docs that should be updated
```

如果修改了项目事实，更新 `current_state.md`。

如果形成新技术决策，更新 `decisions.md`。
