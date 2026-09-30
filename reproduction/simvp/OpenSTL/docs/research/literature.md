# 文献调研

| 字段 | 说明 |
| --- | --- |
| 论文名称 | SimVP: Simpler yet Better Video Prediction |
| 作者 | Gao, Zhangyang; Tan, Cheng; Wu, Lirong; Li, Stan Z |
| 年份 | 2022 |
| 来源 | CVPR |
| DOI | — |
| 核心贡献 | 纯 CNN 的编码-翻译-解码视频预测，性能优于当时 RNN 类方法 |
| 项目关联 | 本 reproduction 的主对象 |

| 字段 | 说明 |
| --- | --- |
| 论文名称 | SimVP: Towards Simple yet Powerful Spatiotemporal Predictive Learning |
| 作者 | Tan, Cheng; Gao, Zhangyang; Li, Siyuan; Li, Stan Z |
| 年份 | 2022 |
| 来源 | arXiv:2211.12509 |
| DOI | — |
| 核心贡献 | 引入 MetaFormer 骨干（gSTA 等）作为隐空间翻译 |
| 项目关联 | 默认 `model_type='gSTA'` 来源 |

| 字段 | 说明 |
| --- | --- |
| 论文名称 | Temporal Attention Unit: Towards Efficient Spatiotemporal Predictive Learning |
| 作者 | Tan, Cheng; et al. |
| 年份 | 2023 |
| 来源 | CVPR |
| DOI | — |
| 核心贡献 | TAU 单元（在 gSTA 上 + SE 通道注意力） |
| 项目关联 | `TAUSubBlock` 实现，作为时序建模对比 |

| 字段 | 说明 |
| --- | --- |
| 论文名称 | OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning |
| 作者 | Tan, Cheng; Li, Siyuan; et al. |
| 年份 | 2023 |
| 来源 | NeurIPS D&B Track |
| DOI | arXiv:2306.11249 |
| 核心贡献 | 统一 STL 基准与三层抽象框架 |
| 项目关联 | 本目录所依赖的代码框架 |

> 注：reproduction 级 `README.md` 中 Paper / Official repository / Upstream commit 仍为 `TODO`，
> 需在 `docs/reproduction/` 完成 provenance 审计后回填。
