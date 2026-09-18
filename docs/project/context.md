# CIRA-Diff 云图预报项目

## 项目简介

[FACT] 本项目以 CIRA-Diff 开源仓库为 baseline，目标是理解、复现并在后续研究中扩展基于静止卫星红外亮温图像的云图预报方法。

[FACT] 当前优先级是官方 baseline 的数据接口、单步训练、扩散采样和自回归 rollout；本阶段不修改 CIRA-Diff 核心模型。

[EVIDENCE] 上游项目将任务描述为 GOES 红外亮温预测，目标时效为 0–3 h；论文与代码见 [论文](https://doi.org/10.1175/AIES-D-25-0046.1)、[arXiv 版本](https://arxiv.org/abs/2505.10432) 和 [官方仓库](https://github.com/dopplerchase/cira-diff)。

## 技术背景

卫星红外图像的像素主要表示 brightness temperature（亮温），不是 RGB 视频纹理。云场的时间演变同时包含平流、形变、生成、增强和消散，因此需要同时关注像素误差、空间结构、多尺度信息和高影响云系。

本项目暂时聚焦 satellite-only observation-space forecasting，不把雷达降水预报、NWP 状态预报或卫星条件气象场预报混为同一任务。

## 输入数据

[EVIDENCE] 官方论文使用 GOES-16 ABI Channel 13（10.3 μm），全圆盘数据时间分辨率为 10 min，近天底空间分辨率约 2 km；为适配显存，构造成 256×256 patch。

[FACT] 当前仓库的数据类读取一个 Zarr store 的 `input_images` 和 `output_images` 数组，并将样本返回为 `(output_images[idx], input_images[idx])`，见 `cira_diff/dataset.py` 与 `scripts/Chase_2025/train_edm_Chase2025.py`。

[EVIDENCE] 论文的数据选择为：2023 年训练集超过 30,000 个 patch；2024 年 1–8 月验证集 1,000 个 patch；2024 年 8 月至 2025 年 2 月测试集 1,000 个 patch。所有集合使用训练集统计量做零均值、单位方差标准化。

[EVIDENCE] `26fall-AIclass:/data1/satcast/` 已完成只读审计。普通 train 为 35,595 个单步样本；validation/test 各为 1,024 个、带 18 帧 truth 的评估样本；数组为 `float16` 256×256 patch。Zarr metadata 没有 per-sample timestamp，详见 `docs/training/dataset.md`。

## 输出结果

[FACT] 单步输出是下一时刻的单通道 256×256 预测图；论文任务使用 t−10 min、t 两帧预测 t+10 min。

[FACT] 3 h forecast 由 18 次 10 min 单步预测组成。扩散模型可以从不同随机种子生成 ensemble members。

[EVIDENCE] 3 h rollout 的真实 target 来自 validation/test `output_images` 的 18 帧，而不是从 train 的三帧 sample 推导出来。

[UNKNOWN] 当前仓库没有可直接运行的独立 rollout/inference 脚本；`cira_diff/generate.py` 只有占位说明，完整历史脚本位于 `scripts/Chase_2025/` 的 notebook/训练脚本体系中，需后续整理并验证。

## 技术栈

- Python >= 3.8
- PyTorch
- Hugging Face Diffusers `UNet2DModel`
- Accelerate
- Zarr
- TensorBoard
- NVIDIA EDM 风格 preconditioning 与 sampler

## 项目约束

- 不修改上游核心模型，除非后续任务明确授权。
- 远程原始数据只读；本地仅保存小型 development / inspection / smoke-test subset。
- 论文结果属于 `[EVIDENCE]`，不能写成本项目 `[RESULT]`。
- 历史 AI 对话只能作为调查线索；源码、论文和实际命令优先。
- 任一未从源码、数据或可复现实验确认的结论必须标为 `[UNKNOWN]` 或 `[HYPOTHESIS]`。
