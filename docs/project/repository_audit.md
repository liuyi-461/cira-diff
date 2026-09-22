# 仓库审计

审计日期：2026-09-22  
审计分支：`main`，commit `12c4bf9`  
对比分支：已检查 `feature/ly`，未合并  
参考仓库：`/Users/liuyi/Projects/OpenNowcastLab`

本文记录当前仓库和既有项目知识库中的状态，不为了形成完整故事而填补缺失证据。

## 1. 当前代码实际实现了什么

[FACT] 受保护实现包含：

| 能力 | 位置 | 审计状态 |
| --- | --- | --- |
| Zarr 数据集读取 | `cira_diff/dataset.py` | 源码已实现；本地运行 smoke test 未验证 |
| EDM 预条件与 loss | `cira_diff/edm.py` | 源码已实现 |
| EDM sampler 与 seeded generator | `cira_diff/edm.py` | 源码已实现 |
| checkpoint/config helper 与 model factory | `cira_diff/util.py` | 源码已实现 |
| 普通 EDM 训练循环 | `scripts/Chase_2025/train_edm_Chase2025.py` | 旧脚本存在；复现性未建立 |
| CorrDiff-style 训练循环 | `scripts/Chase_2025/train_edm_CorrDiff_Chase2025.py` | 旧脚本存在；复现性未建立 |
| Latent diffusion 训练路径 | `scripts/Chase_2025/train_edm_LDM_Chase2025.py` | 旧脚本存在；复现性未建立 |
| Vanilla U-Net 路径 | `scripts/Chase_2025/train_vanilla_unet_Chase2025.py` | 旧脚本存在；复现性未建立 |
| 预报 notebook | `scripts/Chase_2025/Run_Forecasts_Chase2025.ipynb` | 历史路径；没有独立 CLI |
| 独立生成 API | `cira_diff/generate.py` | 只有占位说明 |
| 独立 evaluation CLI | 当前没有 canonical 文件 | 未实现 |

[FACT] `cira_diff/train_Diff.py` 和 `cira_diff/train_CorrDiff.py` 的 `main` 中，dataset/model/training setup 仍被注释。不能因为文件存在就认为入口可运行。

## 2. 原始代码与项目新增内容

[EVIDENCE] Git 历史显示，既有 CIRA-Diff 实现主要由 Randy Chase 编写，早于 documentation commit。

[FACT] commit `15c8390`（`docs: add CIRA-Diff project knowledge base`）新增了原始 `docs/` knowledge base 并修改 `.gitignore`，没有修改 `cira_diff/` 或 Chase_2025 源码树。

[FACT] 当前 `main` 是 `12c4bf9`（`chore: ignore local data and generated artifacts`）。

[FACT] `feature/ly` 与 `main` 的差异只有 commit `05a038d` 对 `README.md` 增加的两个空行；未合并。

[UNKNOWN] 当前仓库没有锁定原始 CIRA-Diff upstream commit。现有本地 Git 历史是当前可用的 provenance。

## 3. 已保留的既有知识

原有 knowledge base 记录了：

- CIRA-Diff 作为 GOES-16 ABI Channel 13 satellite-only baseline；
- 两帧输入、一个 10 分钟 target，以及 18 步/3 小时 rollout 语义；
- ordinary train 与 validation/test 的 Zarr shape 差异；
- CorrDiff residual-conditioning 解释；
- 远端 Zarr metadata 和此前记录的本地 development subset；
- 硬编码路径、warmup 字段命名不一致等代码/数据风险；
- 关于 temporal formulation、uncertainty、pixel/structure skill 的初始研究问题；
- `EXP-001` 数据组织审计记录，但没有声称模型复现；
- 已确认 CIRA-Diff 与 EDM，以及明确标为 unresolved 的 DaYu/FY-4A/Himawari 文献线索。

旧文档仍保留。新 canonical 文档通过链接组织它们，不静默替换。

## 4. 已发现的研究问题和假设

[FACT] 既有 `docs/research/hypotheses.md` 已包含 Q1–Q3。它们在 canonical research layer 中整理并扩展为 RQ-001–RQ-008 与 HYP-001–HYP-007。所有候选因果性表述仍为 `[HYPOTHESIS]` / `UNTESTED`。

## 5. 已完成工作与计划工作

### 已完成或已有记录

- [RESULT] 已完成仓库、文档和源码审计，并写入既有 knowledge base。
- [RESULT] 已记录 CIRA-Diff Zarr metadata 审计：ordinary train input `(35595, 2, 256, 256)`、output `(35595, 1, 256, 256)`；validation/test input `(1024, 2, 256, 256)`、output `(1024, 18, 256, 256)`。
- [RESULT] 既有文档记录了代表性 sample 检查和未发现 NaN。
- [RESULT] 本次完成了 Git 分支/日志检查和 OpenNowcastLab 架构比较。

### 尚未完成或仅为计划

- [UNKNOWN] 本地 Dataset/DataLoader smoke test；既有文档记录缺少 `torch` 和 `zarr`。
- [UNKNOWN] 已验证的模型训练运行。
- [UNKNOWN] 已验证的 checkpoint load 和 forecast generation。
- [UNKNOWN] 独立的 18-step rollout CLI。
- [UNKNOWN] 本地复现论文指标。
- [UNKNOWN] 受控 baseline comparison 或 history-length ablation。

## 6. 数据状态

[EVIDENCE] 既有文档记录远端数据路径 `26fall-AIclass:/data1/satcast/`、GOES-16 ABI Channel 13、10 分钟、256×256 patch，以及 train statistics `mean=279.0699458792467 K`、`std=19.32967519050003 K`。

[FACT] 当前 Git working tree 没有 tracked data store。远端审计和本地 subset 是此前审计留下的文档证据，不是本轮重新读取的数据。

[UNKNOWN] 当前记录中没有 per-sample timestamp、source filename、原始 sequence index 和完整 Zarr generation script。

任务提示中的 Himawari 信息保留为 `docs/data/himawari.md` 中的未验证研究上下文，未提升为数据事实。

[CONFLICT] 论文记录说 validation/test 约为 1,000 个 patch，而已审计 derived Zarr 为 1,024 个 sample。声称 exact reproduction 前必须解决该 revision/sampling 差异。

## 7. CIRA-Diff reproduction 状态

[FACT] 本地存在源码。[UNKNOWN] environment、inference、training 和 evaluation reproduction 仍未验证，见 `reproduction/cira_diff/STATUS.md`。

源码和论文记录的是单步训练后自回归推理。本次代码审计没有证据表明做过 rollout training。该区别已在 `docs/research/cira_diff.md` 中明确记录。

## 8. Baseline 与 evaluation 状态

[PLAN] 现有 CIRA-Diff 家族比较计划包括 persistence、vanilla U-Net、CIRA-Diff Diff、CorrDiff 和 LDM。

[PLAN] PredRNN++、ConvLSTM、SimVP、TAU 和 Earthformer 是未来 reproduction candidates；仓库中没有它们的本地代码、weights 或运行证据。

[ISSUE] 当前没有已验证的项目级 metric implementation。候选 pixel、structural、threshold、spectral、object、temporal、probabilistic 和 case-based metrics 已在 `docs/evaluation/` 中标明状态。

## 9. 尚未验证的重要讨论

- 两帧是否足以同时描述 motion 和 intrinsic evolution；
- 更长历史是否主要改善 long lead time；
- autoregressive drift 来自单步误差、distribution shift，还是二者共同作用；
- diffusion 改善的是 calibrated uncertainty、perceptual realism、meteorological skill，还是其中一部分；
- pixel metrics 是否与 cloud-object 和 scale-dependent evolution 一致；
- satellite forecasting 结论是否能跨 GOES 与 Himawari 迁移。

## 10. 可追溯性结论

[CONCLUSION] 仓库现在已经具备可导航的 knowledge/reproduction/experiment 架构，但科研证据链目前停留在已记录的数据与代码证据，尚未到达已验证的 forecast results，也不能支持哪种 temporal paradigm 最优的结论。
