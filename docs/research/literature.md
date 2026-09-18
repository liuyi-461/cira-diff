# 文献调研

> 更新时间：2026-09-17。历史 AI 对话中的论文和结论仅作检索线索；下表只把当前已由论文、官方页面或官方仓库确认的内容写成 `[EVIDENCE]`，其余明确标注状态。

## 研究范围

优先关注直接预测未来静止卫星图像/红外亮温的 satellite-only forecasting。雷达降水 nowcasting、NWP 状态预测和 satellite-conditioned weather forecasting 作为相邻任务，不能直接比较。

## 已确认核心文献

| 论文名称 | 作者 | 年份/来源 | DOI/链接 | 核心贡献 | 项目关联 |
|---|---|---|---|---|---|
| How to Use Score-Based Diffusion in Earth System Science: A Satellite Nowcasting Example | Randy J. Chase et al. | 2026, *Artificial Intelligence for the Earth Systems* | [10.1175/AIES-D-25-0046.1](https://doi.org/10.1175/AIES-D-25-0046.1)；[arXiv](https://arxiv.org/abs/2505.10432) | `[EVIDENCE]` 比较 Diff、CorrDiff、LDM，预测 GOES-16 ABI Channel 13 亮温，10 min 单步并 rollout 到 3 h；报告 CorrDiff、ensemble 与 U-Net/persistence 的比较。 | 官方 baseline 与主要方法依据 |
| CIRA-Diffusion repository | Randy Chase / CIRA | 代码仓库 | [GitHub](https://github.com/dopplerchase/cira-diff) | `[FACT]` 提供 ZarrDataset、EDM preconditioning、loss、sampler 及 Chase_2025 脚本。 | 实现核验来源 |
| Elucidating the Design Space of Diffusion-Based Generative Models | T. Karras et al. | 2022, NeurIPS | [arXiv](https://arxiv.org/abs/2206.00364) | EDM 预条件、噪声设计与采样基础。 | `cira_diff/edm.py` 的上游理论来源 |

## 历史调研中的待核验条目

| 条目 | 当前状态 | 下一步 |
|---|---|---|
| DaYu | `[UNKNOWN]` 当前未在本仓库证实论文版本、代码和精确输入输出 | 用正式论文/作者主页/官方代码交叉核验 |
| FY-4A DDMS | `[UNKNOWN]` 历史对话提供了 2 h→4 h、扩散残差等线索，但本轮未把它作为 CIRA-Diff 事实 | 单独建立文献核验记录后再比较 |
| Himawari 2024 PredRNN++ 工作 | `[UNKNOWN]` 历史对话给出 24→24 线索，本轮未完成一手来源核验 | 确认论文、数据、时间间隔和是否直接预测 IR BT |

## 当前知识空白

- 尚未找到在同一 satellite-only 数据、相同 horizon 和统一预算下系统比较 2/4/8/12/24 帧 history 的证据。
- 尚未确认 CIRA-Diff 官方实现是否做过 rollout training 或 checkpoint selection 的额外细节；当前代码可见的是单步 loss 和采样。
- 文献中的“预测能生成/消散对流”不能直接等价于本项目已经具备可校准的强对流预报能力。
