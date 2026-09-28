# CIRA-Diff Reproduction

## 角色

当前受保护的 reproduction/baseline，也是 satellite-only forecasting 的研究起点。

## 必须记录的字段

| 字段 | 当前记录 |
| --- | --- |
| Paper | Chase et al., *How to Use Score-Based Diffusion in Earth System Science: A Satellite Nowcasting Example* |
| DOI / paper link | `10.1175/AIES-D-25-0046.1`；`https://arxiv.org/abs/2505.10432` |
| Official repository | `https://github.com/dopplerchase/cira-diff` |
| Local repository | `https://github.com/liuyi-461/cira-diff`；当前分支 `main` |
| Upstream commit | `UNKNOWN / not pinned` |
| Dataset | GOES-16 ABI Channel 13 derived Zarr；见 `docs/data/cira.md` |
| Original task | 两帧历史图像 → 下一帧 10-minute IR BT；18-step autoregressive rollout 到 3 h |
| Environment | `UNKNOWN`；既有 docs 记录本地 smoke test 缺少 `torch`/`zarr` |
| Weights | `UNKNOWN`；没有 checkpoint 被跟踪 |
| Reproduction status | source available；data audit documented；inference/training/evaluation pending |
| Local modification | 本次只有文档和 Lab scaffold；没有移动核心源码 |
| Data adapter | `cira_diff/dataset.py:ZarrDataset`；没有 promoted adapter |
| Evaluation adapter | 旧 forecast notebook path；没有 verified standalone evaluator |
| Known issues | 硬编码路径、normalization mismatch risk、warmup field mismatch、placeholder generation module、缺少 timestamp |

## 必须保留的语义

可见实现支持 single-step loss 和 EDM sampling。autoregressive rollout 是 inference/evaluation 行为；rollout training 尚未建立证据。见 `docs/research/cira_diff.md`。

## 目录策略

在 exact external revision 被获取前，暂不创建 `upstream/` checkout。`adapters/`、`patches/`、`tests/` 和 `reference/` 仅用于未来可审计 reproduction artifact。
