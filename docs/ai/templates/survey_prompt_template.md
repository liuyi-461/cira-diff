你是一名熟悉【研究领域】、【具体任务方向】和【相关AI方法】的科研助手。

我正在开展一个关于【研究主题】的初步文献调研。

当前阶段的目标不是立即选择某一个模型，而是先建立这个方向的：

* 任务定义
* 技术发展路线
* 代表性方法
* 代表性论文
* 数据集
* Evaluation Protocol
* 当前主要科学问题
* 可复现性情况
* Baseline 与潜在 SOTA 候选

为后续开展正式研究和模型选择提供依据。

请先不要直接回答 Survey 内容，而是根据我下面提供的研究背景，帮我生成一份完整、结构化、可直接用于 ChatGPT / Claude 等大模型开展 Survey 的 Prompt。

---

# 1. 我的研究背景

## 研究领域

【例如：气象人工智能 / 遥感 / 计算机视觉 / 医学影像】

## 具体研究问题

【例如：基于天气雷达的短临降水预报】

## 我真正希望解决的问题

【用1~3句话描述真实科学问题或业务问题，而不是模型名称】

例如：

我希望利用过去一段时间的天气雷达观测预测未来0–2小时的降水演变，重点关注强对流区域的生消、移动以及极端降水结构能否被准确预测。

## Input

【模型推理时能够获得的数据】

例如：

* 过去10帧雷达组合反射率
* 时间分辨率6 min
* 空间分辨率1 km

## Target / Output

【最终希望预测、反演或识别的对象】

例如：

* 未来20帧雷达回波
* 或未来2小时降水概率

## 时间尺度

【例如：0–2 h / 0–6 h / 日尺度】

## 空间尺度

【例如：1 km / 单体尺度 / 区域尺度】

## 数据条件

【目前已有的数据、计划使用的数据或公开数据】

## 当前已知方法

【如果知道就填写；不知道可以写“未知”】

例如：
ConvLSTM、PredRNN、GAN、Diffusion

## 当前最关心的问题

【例如：

* 应该选什么 Baseline？
* 当前生成式方法发展到了什么程度？
* 哪些方法真正开源可复现？
* 当前方法在强降水方面有什么问题？
  】

---

# 2. 请为我生成的 Survey Prompt 必须覆盖以下内容

## A. Task Definition

要求 Survey 首先区分该领域中可能存在的不同任务定义。

至少分析：

* Input
* Target
* AI Task
* 时间分辨率
* 空间分辨率
* Forecast Lead Time 或预测范围
* Deterministic / Probabilistic
* Evaluation Protocol

如果不同论文研究的任务不同，不允许直接比较性能。

---

## B. Technical Taxonomy

要求建立技术谱系，而不是简单罗列论文。

按照：

【传统方法】
→【早期深度学习】
→【主流深度学习】
→【生成式方法】
→【最新研究方向】

梳理方法发展。

每一类方法说明：

* 核心思想
* 主要解决什么问题
* 相比前一类方法改进在哪里
* 优点
* 局限
* 适用场景
* 计算成本
* 是否表达不确定性

---

## C. Representative Papers

每个主要方向至少给出若干篇代表论文。

每篇论文至少整理：

* Title
* Authors
* Year
* Journal / Conference
* DOI 或正式论文链接
* Task
* Input
* Target
* Dataset
* Model
* Innovation
* Baseline
* Metrics
* Main Results

要求区分：

* Landmark Paper
* Strong Baseline
* Recent Representative Work
* Potential SOTA

---

## D. Dataset Survey

对主要公开 Dataset 整理：

* 数据来源
* 数据类型
* 空间覆盖
* 时间覆盖
* 时间分辨率
* 空间分辨率
* 样本规模
* Train / Validation / Test 划分
* 下载方式
* License
* 哪些论文使用过

---

## E. Evaluation

不要只列指标名称。

要求分析：

* 指标定义
* 适合评价什么
* 不适合评价什么
* 是否存在指标偏差
* 极端事件中是否可能产生误导

要求明确区分：

* Pixel-level Metric
* Structure Metric
* Event-based Metric
* Probabilistic Metric
* Physical / Domain-specific Evaluation

---

## F. Scientific Challenges

不要只总结“模型存在的问题”。

请从：

* 数据
* 物理过程
* 可预报性
* 极端事件
* 泛化
* 不确定性
* 观测误差
* 计算效率
* 业务应用

等角度总结当前主要科学问题。

每个问题至少回答：

1. 为什么它是问题；
2. 当前已有解决方法；
3. 目前还没有解决好的部分。

---

## G. Reproducibility

对每个重要模型检查：

* Paper 是否公开
* Code 是否公开
* Official Repository 是否存在
* Pretrained Weight 是否公开
* Dataset 是否公开
* Training Details 是否完整
* Loss 是否公开
* Hyperparameters 是否公开
* 能否完整复现

请特别强调：

**论文公开 ≠ 代码公开 ≠ 权重公开 ≠ 可完整复现。**

---

## H. Baseline 与 SOTA

不要简单回答：

“当前最好的模型是什么？”

而是分别判断：

* Classical Baseline
* Deep Learning Baseline
* Strong Baseline
* Generative Baseline
* Potential SOTA
* Operational Baseline

对于每个候选方法，从以下角度评价：

* Task Match
* Dataset Match
* Benchmark Performance
* Reproducibility
* Compute Cost
* Engineering Complexity
* Scientific Value

---

# 3. 文献核验要求

生成的 Survey Prompt 必须明确要求模型：

* 优先检索同行评审论文；
* 优先使用出版社页面、作者主页、官方仓库；
* 不得编造论文；
* 不得编造 DOI；
* 不得仅依据标题推断论文内容；
* 第三方 GitHub 与 Official Repository 必须区分；
* 无法确认的信息必须写“未确认”；
* Recent / SOTA 必须核验论文发表时间和 Benchmark；
* 尽量交叉验证论文、代码和项目主页。

---

# 4. 最终输出结构

生成的 Survey Prompt 应要求最终 Survey 至少包含：

1. Research Scope
2. Task Taxonomy
3. Technical Evolution
4. Representative Models
5. Representative Papers
6. Dataset Comparison
7. Evaluation Protocol
8. Reproducibility Comparison
9. Scientific Challenges
10. Baseline Recommendation
11. Potential Research Directions
12. Further Search Keywords

同时至少生成以下表格：

### Model Table

| Method | Task | Input | Output | Deterministic / Probabilistic | Dataset | Code | Weight | Loss |

### Paper Table

| Paper | Year | Task | Dataset | Model | Baseline | Metrics | DOI |

### Dataset Table

| Dataset | Data Type | Resolution | Coverage | Availability |

### Reproducibility Table

| Model | Code | Weight | Training Details | Loss | Dataset | Reproducible |

---

# 5. 输出要求

请首先分析我提供的研究背景中是否存在：

* 问题定义不清；
* Input / Target 不明确；
* Task 混淆；
* 时间尺度或空间尺度缺失；
* Survey 范围过宽；
* 已经预设某一种模型导致技术路线偏置。

如果存在这些问题，请先指出。

然后生成一份：

**完整、可直接复制给 ChatGPT / Claude 使用的 Survey Prompt。**

不要直接替我完成 Survey。

最终生成的 Prompt 应具有足够约束，使模型不会简单输出“论文列表”，而是完成一个结构化、可核验、面向科研决策的初步 Survey。
