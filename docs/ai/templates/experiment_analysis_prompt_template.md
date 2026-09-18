# Experiment Analysis Prompt Template

> 用途：在一次或多次实验完成后，辅助完成结果分析、误差诊断、科学解释和下一实验设计。
> 建议路径：`docs/ai/templates/experiment_analysis_prompt_template.md`

---

你是一名熟悉【研究领域】、【AI任务】和【实验设计】的科研助手。

请帮助我分析当前实验结果。

本次任务的目标不是简单判断“哪个模型指标最高”，而是回答：

- 实验是否真正回答了 Research Question；
- 当前结果能够支持什么结论；
- 哪些结果只是 Result，哪些能够构成 Evidence；
- 模型在哪些场景下表现更好或更差；
- 是否存在指标与气象意义不一致的问题；
- 失败模式是什么；
- 下一步应该验证什么 Hypothesis。

---

## 1. Research Context

### Research Question
【填写】

### Hypothesis
【填写】

### AI Task
【填写】

### Input
【填写】

### Target
【填写】

### Dataset
【填写】

### Train / Validation / Test Split
【填写】

### Baseline
【填写】

### Compared Models
【填写】

### Metrics
【填写】

### Scientific / Business Goal
【填写】

---

## 2. Experiment Description

### Experiment ID
【填写】

### Experiment Purpose
【填写】

### Changed Variable
【填写，例如：
- Model
- Loss
- Input Channel
- Resolution
- Data Augmentation
- Forecast Lead Time
】

### Controlled Variables
【填写】

### Training Configuration
【填写】

### Evaluation Protocol
【填写】

### Current Results
【粘贴表格、日志或关键结果】

---

## 3. First Check: Is the Experiment Fair?

请首先检查：

- Dataset 是否一致；
- Train / Validation / Test 是否一致；
- Random Seed 是否一致；
- Epoch / Iteration 是否可比；
- Input 是否一致；
- Preprocessing 是否一致；
- Resolution 是否一致；
- Evaluation Mask 是否一致；
- Threshold 是否一致；
- Baseline 是否公平；
- 是否存在不同模型使用不同后处理；
- 是否存在挑选最佳个例的问题。

如果实验不可公平比较，请优先指出。

---

## 4. Analyze Results at Multiple Levels

不要只分析平均指标。

### Level 1: Overall Metrics

分析：
- Mean
- Median
- Standard Deviation
- Best / Worst
- Relative Improvement

说明：
- 提升是否稳定；
- 是否只是很小的数值波动；
- 是否可能来自随机性。

---

### Level 2: Lead Time

如果是预测任务，请分析不同 Forecast Lead Time：

- 短时
- 中时
- 长时

回答：

- 哪个模型衰减更快；
- 性能差距何时开始扩大；
- 是否存在某个提前量后优势消失；
- 是否与可预报性有关。

---

### Level 3: Intensity / Threshold

针对不同强度分析，例如：

- 弱事件
- 中等事件
- 强事件
- 极端事件

检查：

- 模型是否只改善弱事件；
- 是否牺牲极端事件；
- FAR / POD / CSI 是否存在 trade-off；
- 平均误差下降是否来自大量弱样本。

---

### Level 4: Spatial Structure

分析：

- 位置误差；
- 结构误差；
- 边界；
- 梯度；
- 极值；
- 面积；
- 形态；
- 位移；
- 生消。

不要只依据像素级误差判断空间结构。

---

### Level 5: Case Study

对典型个例进行分类：

- Best Case
- Worst Case
- Typical Case
- Extreme Case
- Failure Case

每一类分析：

- 输入条件；
- 模型输出；
- Baseline 输出；
- 真值；
- 误差位置；
- 可能原因。

---

## 5. Result vs Evidence

请严格区分：

### Result
实验直接得到的数字、图像、曲线、统计量。

### Evidence
能够直接支持或否定 Research Question / Hypothesis 的结果。

请建立：

| Research Claim | Required Evidence | Current Result | Supported? | Confidence |
|---|---|---|---|---|

不要因为某一个 Metric 提升，就直接认为 Hypothesis 成立。

---

## 6. Metric Consistency

检查不同 Metrics 是否给出一致结论。

例如：

- RMSE 改善但 CSI 下降；
- SSIM 提升但极值被平滑；
- POD 提升但 FAR 同时大幅增加；
- CRPS 提升但 Reliability 变差。

如果指标冲突，请分析原因，而不是简单平均。

---

## 7. Scientific Interpretation

请从领域机理解释结果。

避免仅写：

“模型学习能力更强。”

优先回答：

- 模型为什么可能改善该现象；
- 哪种结构被更好保留；
- 哪些物理过程仍然无法表达；
- 数据分辨率是否限制模型；
- Lead Time 是否超过可预报性范围；
- 标签或观测误差是否限制上限；
- 是否存在样本不平衡。

---

## 8. Failure Analysis

请总结主要失败模式。

建议分类：

- Data Failure
- Label Failure
- Model Failure
- Loss Failure
- Optimization Failure
- Generalization Failure
- Extreme Event Failure
- Physical Consistency Failure
- Evaluation Failure

每类说明：

1. 现象；
2. 可能原因；
3. 如何验证；
4. 推荐下一实验。

---

## 9. Ablation Logic

如果存在多个模块，请判断是否能够证明每个模块有效。

要求：

- 一次只改变一个核心变量；
- 不把多个改动同时归因于一个模块；
- 检查模块是否只在特定场景有效；
- 检查性能提升是否值得增加的计算成本。

输出：

| Component | Hypothesis | Evidence | Improvement | Cost | Keep? |
|---|---|---|---|---|---|

---

## 10. Next Experiment

基于当前 Evidence，而不是凭直觉给出下一实验。

每个建议写成：

### Hypothesis
【下一步要验证什么】

### Why
【为什么当前结果指向这个问题】

### Experiment
【如何设计】

### Controlled Variables
【哪些不变】

### Expected Evidence
【什么结果说明 Hypothesis 成立】

### Failure Criterion
【什么结果说明该方向应停止或调整】

### Priority
High / Medium / Low

---

## 11. Final Output

最终请按以下结构输出：

1. Experiment Summary
2. Fairness Check
3. Main Results
4. Evidence Supporting Hypothesis
5. Evidence Against Hypothesis
6. Metric Conflicts
7. Case Analysis
8. Failure Modes
9. Scientific Interpretation
10. Reproducibility Issues
11. Recommended Next Experiments
12. Current Conclusion

最后用一句话明确：

> 基于当前 Evidence，我们现在能够确定什么，仍然不能确定什么。

---

## 12. Documentation Update

最后告诉我本次实验结束后应该更新哪些项目文档：

- `docs/current_state.md`
- `docs/decisions.md`
- `docs/roadmap.md`
- `docs/knowledge.md`
- `docs/workflow.md`

并分别说明应该记录什么。

不要直接编造实验结论；如果数据不足，请明确指出“当前证据不足”。
