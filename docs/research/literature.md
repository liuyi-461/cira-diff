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

## 标准化 literature/reproduction records

上面的既有表格保留。下面的记录明确 verification status；当前仓库没有核验 primary source 的字段保留为 `TODO`，不从模型名称推断 DOI。

### LIT-001 — CIRA-Diff / satellite score-based diffusion

- **Title:** *How to Use Score-Based Diffusion in Earth System Science: A Satellite Nowcasting Example*
- **作者：** Randy J. Chase et al.（来自既有项目文档记录）
- **年份/期刊：** 2026，*Artificial Intelligence for the Earth Systems*（已记录）
- **DOI：** `10.1175/AIES-D-25-0046.1`（已记录并链接）
- **Repository：** `https://github.com/dopplerchase/cira-diff`（已记录）
- **数据集/任务：** GOES-16 ABI Channel 13 IR brightness-temperature forecasting
- **输入/输出/horizon：** 两帧历史图像 → 下一帧；10 分钟间隔；rollout 到 3 小时（已有 evidence）
- **模型/指标：** Diff、CorrDiff、LDM、U-Net/persistence comparison；ME/MAE/RMSE 和 ensemble/cold-cloud 讨论（paper evidence）
- **项目关系：** 当前受保护 baseline 和 research starting point
- **Verification status：** 论文/repository 链接已记录；本地 reproduction 未验证

### LIT-002 — EDM

- **Title:** *Elucidating the Design Space of Diffusion-Based Generative Models*
- **作者/年份/期刊：** T. Karras et al.，2022，NeurIPS（已记录）
- **DOI：** `TODO`——本次代码审计不需要，不得编造
- **Repository：** 本项目的官方 upstream pin 为 `TODO`
- **相关方法：** EDM preconditioning、noise design 和 sampling
- **项目关系：** `cira_diff/edm.py` 的理论/代码来源
- **Verification status：** 既有文档保留了 arXiv 链接；未声称完成本地理论复现

### LIT-003–LIT-005 — Candidate satellite/video baselines

当前 docs 提到 DaYu、FY-4A DDMS 和 Himawari PredRNN++ study，作为 research leads。它们的 exact title、authors、year、venue、DOI、repository、dataset、input/output 和 metrics 仍是 `TODO` / `[UNKNOWN]`。它们仅作为 surveyed scaffold 登记在 `reproduction/`，primary sources 核验前不得引用为 verified literature。
