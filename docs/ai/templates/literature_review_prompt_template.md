# Literature Review Prompt Template

> 用途：针对一个已经相对明确的研究主题，生成结构化、可核验、面向科研决策的文献综述任务。
> 建议路径：`docs/ai/templates/literature_review_prompt_template.md`

---

你是一名熟悉【研究领域】、【具体研究方向】和【相关AI/气象方法】的科研助手。

我正在开展关于 **【研究主题】** 的文献综述。

当前目标不是简单罗列论文，而是建立一个能够支持后续科研决策的 Literature Review，重点回答：

- 这个研究问题是如何被定义的；
- 该方向经历了怎样的方法演进；
- 代表性论文解决了什么问题；
- 不同方法之间真正的差异是什么；
- 当前有哪些已经解决的问题；
- 哪些问题仍然没有解决；
- 哪些方法适合作为 Baseline；
- 哪些方向值得继续研究。

请基于公开、可核验的同行评审论文、会议论文、官方项目主页和官方开源仓库进行综述。

---

## 1. Research Context

### 研究领域
【填写】

### 研究主题
【填写】

### 核心科学问题
【填写1~3句话】

### 业务/应用目标
【填写】

### Input
【填写】

### Target / Output
【填写】

### 时间尺度
【填写】

### 空间尺度
【填写】

### 数据条件
【填写】

### 当前已知方法
【填写；未知可写“未知”】

### 当前最关心的问题
【填写，例如：
- 哪些论文真正推动了该方向发展？
- 当前主流方法的核心差异是什么？
- 哪些模型适合作为 Baseline？
- 现有研究在哪些气象问题上仍然不足？
】

---

## 2. Review Scope

请首先明确本综述的任务边界，并回答：

- 哪些工作属于本研究主题；
- 哪些工作看似相关但任务不同；
- 哪些任务不能直接比较；
- 是否存在不同 Input / Target / Lead Time / Resolution / Evaluation Protocol；
- 不同研究是否使用不同标签定义或事件标准。

如果任务定义不同，请分组综述，禁止直接比较指标。

---

## 3. Technical Evolution

请按照“方法为什么出现、解决了什么问题”的逻辑梳理技术演进，而不是按年份简单堆论文。

建议结构：

1. 传统方法 / 物理方法 / 统计方法
2. 早期机器学习
3. 早期深度学习
4. 主流网络架构
5. 时空建模方法
6. 生成式 / 概率方法
7. 多源融合方法
8. Foundation Model / Pretrained Model
9. 最新研究方向

每一类方法请说明：

- 核心思想；
- 主要解决什么问题；
- 相比前一阶段的改进；
- 主要优点；
- 主要局限；
- 适用任务；
- 对数据量的要求；
- 对算力的要求；
- 是否支持概率预测或多解；
- 是否具有业务部署价值。

---

## 4. Representative Papers

每个主要方向至少选择若干篇真正具有代表性的论文。

优先包括：

- Landmark Paper
- Frequently Used Baseline
- Strong Baseline
- Recent Representative Work
- Potential SOTA
- Operational / Applied Work

对每篇论文至少整理：

| Field | Content |
|---|---|
| Title | |
| Authors | |
| Year | |
| Journal / Conference | |
| DOI / Official Link | |
| Task | |
| Input | |
| Target | |
| Dataset | |
| Spatial Resolution | |
| Temporal Resolution | |
| Forecast Lead Time | |
| Model | |
| Key Innovation | |
| Baseline | |
| Metrics | |
| Main Result | |
| Limitation | |
| Code | |
| Weight | |
| Training Details | |
| Loss | |

不要只写论文摘要，要说明其在技术发展链条中的作用。

---

## 5. Method Comparison

请建立方法对比表：

| Method | Core Idea | Task | Strength | Weakness | Deterministic / Probabilistic | Compute Cost | Reproducibility |
|---|---|---|---|---|---|---|---|

重点比较：

- 建模对象是否相同；
- 是否考虑空间结构；
- 是否考虑时间演变；
- 是否能够处理生消过程；
- 是否表达不确定性；
- 是否容易出现模糊平均；
- 是否适合极端事件；
- 是否适合长提前量；
- 是否容易复现。

---

## 6. Dataset and Evaluation

请分别整理主要数据集和评价协议。

### Dataset Table

| Dataset | Data Type | Region | Resolution | Time Range | Availability | Representative Papers |
|---|---|---|---|---|---|---|

### Evaluation

请区分：

- Pixel-level Metrics
- Structure Metrics
- Event-based Metrics
- Threshold-based Metrics
- Probabilistic Metrics
- Physical / Domain-specific Metrics

对每种指标说明：

- 衡量什么；
- 优点；
- 局限；
- 是否可能与真实科学价值不一致；
- 是否可能对极端事件产生误导。

---

## 7. Scientific Challenges

不要只从“模型结构”总结问题。

请从以下角度分析：

- 数据质量
- 标签可靠性
- 观测误差
- 时空尺度
- 可预报性
- 极端样本稀缺
- 多解性
- 泛化能力
- 物理一致性
- 模型稳定性
- 计算效率
- 实时业务部署

每个问题请回答：

1. 为什么它重要；
2. 当前研究如何处理；
3. 仍然存在什么不足；
4. 哪些论文对这个问题有直接讨论。

---

## 8. Evidence Chain

对于任何“某方法更好”“某方法是SOTA”“某方向更有前景”的判断，请给出完整证据链：

- 比较任务是否相同；
- Dataset 是否相同；
- Split 是否一致；
- Evaluation Protocol 是否一致；
- Baseline 是否公平；
- 是否有统计显著性或多个案例支持；
- 是否只在单一 Benchmark 上领先；
- 是否存在计算成本显著增加的问题。

不要把“指标更高”直接等价为“科学问题解决得更好”。

---

## 9. Reproducibility

对重要方法检查：

- Paper
- Official Code
- Pretrained Weight
- Dataset
- Training Script
- Configuration
- Hyperparameters
- Loss
- Data Preprocessing
- Evaluation Script

请明确区分：

**论文公开 ≠ 代码公开 ≠ 权重公开 ≠ 训练细节公开 ≠ 可完整复现。**

---

## 10. Final Output

最终综述请包含：

1. Research Scope
2. Task Definition
3. Technical Evolution
4. Representative Papers
5. Method Comparison
6. Dataset Comparison
7. Evaluation Protocol
8. Scientific Challenges
9. Reproducibility Analysis
10. Strong Baselines
11. Potential SOTA
12. Open Research Questions
13. Recommended Next Research Step
14. Search Keywords

最后给出一个简短判断：

> 如果现在开始一个新的【研究主题】项目，最值得复现的 Baseline 是什么，最值得进一步研究的方向是什么，为什么？

---

## 11. Verification Requirements

必须遵守：

- 不编造论文；
- 不编造 DOI；
- 不仅凭标题判断论文内容；
- 优先使用论文原文、出版社页面、作者主页和官方仓库；
- Official Repository 与 Third-party Implementation 必须区分；
- 无法确认的信息写“未确认”；
- 最新 / SOTA 必须核验时间和 Benchmark；
- 对关键结论尽量进行交叉验证；
- 不得将不同任务、不同数据集、不同评价协议的数字直接比较。
