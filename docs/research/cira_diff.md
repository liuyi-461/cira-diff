# CIRA-Diff 科研记录

本文根据当前仓库和既有 project docs 重建 CIRA-Diff baseline，并将论文证据与本地 reproduction 状态分开记录。

## 任务定义

| 字段 | 记录值 | 状态 |
| --- | --- | --- |
| Satellite | GOES-16 | `[EVIDENCE]`，来自既有 docs/paper record |
| Channel | ABI Channel 13，10.3 μm IR | `[EVIDENCE]` |
| Target | brightness temperature image | `[EVIDENCE]` |
| Temporal interval | 10 min | `[EVIDENCE]` |
| Input | 两帧历史图像，`t-10 min`、`t` | `[EVIDENCE]` |
| Training target | 下一帧，`t+10 min` | `[EVIDENCE]` |
| Evaluation target | 18 个 future frames，至 3 h | `[EVIDENCE]`，来自项目数据审计 |
| Spatial sample | 256×256 patch | `[EVIDENCE]` |
| Forecast paradigm | single-step conditional model + autoregressive rollout | `[EVIDENCE]` |

## Training 与 rollout inference

[FACT] 可见训练循环计算 one-step target loss。ordinary dataset 每个 sample 提供一个 target image。记录中的 evaluation store 提供 18 个 future truth frames。

[EVIDENCE] 官方任务记录为先预测下一帧，再递归反馈预测结果生成 18-step forecast。

[UNKNOWN] 当前项目没有证据证明论文模型使用了 rollout training、scheduled sampling 或 multi-step loss。不能把 rollout inference 写成 rollout training。

## Diffusion formulation

[FACT] `cira_diff/edm.py` 实现 EDM-style preconditioning、`EDMLoss` 中的 log-normal noise-level sampling，以及默认 18 个 noise steps 的 EDM sampler。18 个 sampler steps 是 denoising steps，不是 18 个 forecast lead times。

[FACT] ordinary path 将 noisy generation channels 与 condition channels 拼接后输入 U-Net。

[FACT] 记录中的 CorrDiff path 使用两帧 history 加一帧 first-guess/U-Net condition，学习 residual target；最终重建需要将 residual 加回 first guess。

## Dataset 与 normalization

[EVIDENCE] 既有文档记录 ordinary train input `(35595,2,256,256)`、output `(35595,1,256,256)`，validation/test input `(1024,2,256,256)`、truth output `(1024,18,256,256)`。记录的 train normalization 为 `mean=279.0699458792467 K`、`std=19.32967519050003 K`。

[UNKNOWN] 当前 checkout 没有远端 Zarr payload 或 per-sample timestamp，因此不能从当前树独立恢复 source-sequence continuity。

## Code 与 release provenance

- 官方论文记录：Chase et al., *How to Use Score-Based Diffusion in Earth System Science: A Satellite Nowcasting Example*，DOI `10.1175/AIES-D-25-0046.1`，arXiv `2505.10432`。
- 既有 docs 记录的官方 repository：`https://github.com/dopplerchase/cira-diff`。
- 本地 repository origin：`https://github.com/liuyi-461/cira-diff`。
- exact upstream commit/checkpoint/data checksum：`UNKNOWN`，当前项目未锁定。

## Reproduction 状态

见 `reproduction/cira_diff/STATUS.md`。当前为 source available、data audit documented、environment 和 forecast execution unverified。
