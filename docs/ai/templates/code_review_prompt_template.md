# Code Review Prompt Template

> 用途：让 AI 在修改代码之前先建立项目上下文、理解 Data Flow、识别风险并给出可验证的修改建议。
> 建议路径：`docs/ai/templates/code_review_prompt_template.md`

---

你是一名负责科研软件审查、算法实现核验和代码质量检查的开发助手。

请对当前项目进行 Code Review。

本次任务的目标不是简单找语法错误，而是检查：

- 实现是否符合科研目标；
- Data Flow 是否正确；
- 输入输出是否与任务定义一致；
- 数据预处理是否正确；
- 训练 / 推理 / 评估逻辑是否一致；
- 是否存在数据泄漏；
- 是否存在物理量、单位、归一化或反归一化错误；
- Evaluation 是否公平；
- 是否存在影响科研结论的实现风险；
- 修改是否会破坏已有功能。

---

## 1. Project Context

### Project Goal
【填写】

### Research Question
【填写】

### AI Task
【填写】

### Input
【填写】

### Target
【填写】

### Dataset
【填写】

### Baseline
【填写】

### Current Model
【填写】

### Evaluation
【填写】

### Current Problem
【填写本次为什么要Review】

---

## 2. Required Context Files

在开始代码审查前，请优先阅读：

- `README.md`
- `docs/current_state.md`
- `docs/decisions.md`
- `docs/roadmap.md`
- `docs/workflow.md`
- `docs/ai/codex.md`

如果文件不存在，请明确指出，不要猜测其中内容。

---

## 3. Review Order

请严格按照以下顺序进行。

### Step 1: Understand the Project

先总结：

- 项目目标；
- Research Question；
- Input / Target；
- Dataset；
- 当前模型；
- 当前阶段；
- 当前已知问题。

不要立刻修改代码。

---

### Step 2: Build the Data Flow

梳理完整 Data Flow：

**Raw Data → Preprocessing → Dataset → DataLoader → Model Input → Model → Output → Postprocessing → Evaluation → Saved Result**

对每一步给出：

- 文件路径；
- 关键函数 / 类；
- 输入；
- 输出；
- Shape；
- dtype；
- 单位；
- 数值范围；
- 是否归一化；
- 是否存在插值、裁剪、重采样或坐标变换。

---

### Step 3: Review Critical Scientific Logic

重点检查以下问题。

#### Data
- Train / Validation / Test 是否严格分开；
- 是否存在时间泄漏；
- 是否存在空间泄漏；
- 缺测值如何处理；
- QC 是否合理；
- 数据配准是否正确；
- 时间匹配是否正确。

#### Input / Target
- Input 是否包含推理时无法获得的信息；
- Target 是否与 Research Question 一致；
- Input / Target 时间窗口是否正确；
- Shape 和 Channel 顺序是否一致。

#### Preprocessing
- Normalize 是否正确；
- Train / Inference 是否使用一致的 Normalize；
- 是否错误重复归一化；
- 是否丢失物理单位；
- Resize / Interpolation 是否合理；
- Mask 是否正确传播。

#### Model
- 模型输入输出是否与配置一致；
- Channel 数量是否正确；
- Temporal Dimension 是否正确；
- Checkpoint 是否正确加载；
- Eval / Train Mode 是否正确切换。

#### Loss
- Loss 公式是否与论文 / 设计一致；
- Loss 权重是否正确；
- Reduction 是否正确；
- Mask 是否参与 Loss；
- 各 Loss 的量纲 / 数值尺度是否合理；
- 是否存在某项 Loss 主导训练的问题。

#### Evaluation
- Metric 实现是否正确；
- HR / LR / SR 是否使用相同有效区域；
- Data Range 是否一致；
- Threshold 是否一致；
- Baseline 是否公平；
- 是否发生二次插值；
- 是否存在只保留有利样本的问题。

#### Output
- 反归一化是否正确；
- 输出物理单位是否正确；
- 保存格式是否损失精度；
- Visualization 是否可能误导；
- Colormap / Range 是否一致。

---

## 4. Review Severity

发现问题后按以下等级分类：

### Critical
可能导致科研结论错误，例如：
- 数据泄漏
- Target 错位
- 归一化错误
- 指标计算错误
- Baseline 不公平

### High
明显影响实验结果，但不一定完全推翻结论。

### Medium
影响稳定性、可维护性或复现性。

### Low
代码风格、命名、重复逻辑等问题。

---

## 5. Output Format

请输出：

### A. Project Understanding

简要总结项目目标和 Data Flow。

### B. Critical Findings

| Severity | File | Location | Problem | Scientific Impact | Suggested Fix |
|---|---|---|---|---|---|

### C. Data Flow

用结构化方式说明：

`input file → preprocessing → tensor → model → output → evaluation`

### D. Scientific Risks

单独列出可能影响科研结论的问题。

### E. Engineering Risks

列出：
- 重复代码
- 异常处理
- 配置硬编码
- 路径硬编码
- 性能问题
- GPU / CPU 不一致
- 随机种子
- 日志
- 可复现性

### F. Recommended Fix Order

按照：

1. 必须先修
2. 建议修
3. 可选优化

排列。

---

## 6. Modification Rule

除非我明确要求你修改代码，否则本阶段：

**只 Review，不修改。**

如果后续需要修改代码：

1. 先给出 Plan；
2. 明确要改哪些文件；
3. 说明每个修改对应哪个问题；
4. 给出验证方法；
5. 修改后运行测试；
6. 对比修改前后结果；
7. 更新相关 Docs。

---

## 7. Verification Requirements

对于每一个重要问题：

- 给出具体文件路径；
- 给出函数 / 类 / 关键代码位置；
- 不要只说“可能有问题”；
- 如果无法确认，请写“需要进一步验证”；
- 不要凭经验猜测论文实现；
- 如果项目声称复现某篇论文，请核对实现与论文公式、Loss、数据处理和 Evaluation 是否一致。
