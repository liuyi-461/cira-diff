# 数据集说明

## 数据来源

- Remote Full/Open Dataset: `26fall-AIclass:/data1/satcast/`。
- Local Development Subset: `data-cira-diff/`。

[FACT] 2026-09-18 SSH 已验证成功：host=`test-ai`，user=`group1`，远端路径 `/data1/satcast/` 存在且可读。

## 论文与远端数据规格

[EVIDENCE] 官方论文和远端 README 使用 GOES-16 ABI Channel 13（10.3 μm）、10 min full-disk 数据，近天底约 2 km，原始 full-disk 约 5424×5424；训练/推理使用随机 256×256 patch。训练集来自 2023 年且超过 30,000 patch，验证与测试各 1,000 patch，标准化统计量来自训练集。

远端 `du -sh` 结果和 Zarr 元数据如下：

| Store | Remote size | `input_images` | `output_images` | dtype |
|---|---:|---|---|---|
| `edm_GOES_ch13_train_dataset.zarr` | 9.5G | `(35595, 2, 256, 256)` | `(35595, 1, 256, 256)` | `float16` |
| `edm_GOES_ch13_train_dataset_CorrDiff.zarr` | 14G | `(35595, 3, 256, 256)` | `(35595, 1, 256, 256)` | `float16` |
| `edm_GOES_ch13_validation_dataset.zarr` | 1.8G | `(1024, 2, 256, 256)` | `(1024, 18, 256, 256)` | `float16` |
| `edm_GOES_ch13_test_dataset.zarr` | 1.8G | `(1024, 2, 256, 256)` | `(1024, 18, 256, 256)` | `float16` |
| `edm_GOES_ch13_train_dataset_latent.zarr` | 2.8G | `(35595, 8, 64, 64)` | `(35595, 4, 64, 64)` | `float16` |
| `edm_GOES_ch13_validation_dataset_latent.zarr` | 545M | `(1024, 8, 64, 64)` | `(1024, 72, 64, 64)` | `float16` |
| `edm_GOES_ch13_test_dataset_latent.zarr` | 543M | `(1024, 8, 64, 64)` | `(1024, 72, 64, 64)` | `float16` |

所有普通/latent store 都是 Zarr v2、Blosc/LZ4 压缩；远端没有 `.zattrs` 时间戳元数据。顶层另有 `README.md`、`simple_code.md`、exploration notebook，以及 train/validation/test 三个 tar archive。

归一化常数由远端 `simple_code.md` 给出：`mean=279.0699458792467 K`、`std=19.32967519050003 K`，即 `normalized=(Tb-mean)/std`。

## 三帧与 18-step truth 的组织

[CONCLUSION] CIRA-Diff 公开派生数据不是“所有 split 只有三帧”：

- train 普通 store 的单个 sample 是 2 帧输入 + 1 帧 target，构成三帧单步训练窗口。
- validation/test 普通 store 的单个 sample 是 2 帧输入 + 18 帧 target truth；这是一段用于评估 18-step rollout 的真实未来序列。
- CorrDiff train 的 3 个 input channels 是两帧历史加一帧 first-guess/U-Net 条件，target 为 residual；这是由远端 shape 和源码注释共同支持的实现事实。

[EVIDENCE] 官方 forecast notebook 将 validation/test 的 `output_images` 读成 18 帧 truth，初始化两帧 condition，循环 `time_steps_forward=np.arange(0,18)`，每次把新预测写回窗口，并将预测序列与 18 帧 truth 对齐保存。

[UNKNOWN] `/data1/satcast/` 的 Zarr 只保存派生 patch arrays；chunk 文件名是数字索引，且没有 per-sample timestamp。因而可以确认 eval sample 内有 18 个按数据构造顺序排列的 future frames，但不能仅凭该目录恢复每个 patch 对应的原始 GOES 文件名或跨 sample 的连续原始序列。

远端 README 说明原始数据来源为 NOAA GOES Open Data；`/data1/satcast/` 本身不是原始 full-disk 时间序列归档。

## 已完成的远程审计

- 已验证 SSH、hostname、用户和 `/data1/satcast/`。
- 已读取目录结构、Zarr shape/chunks/dtype/compressor、README 和 simple code。
- 已读取每个主要 split 的 sample 0；input/output 无 NaN。
- validation/test sample 0 的 18 帧 target 均有逐帧数值变化，支持其为序列 truth；时间间隔依据论文/README 为 10 min。
- 已核对 train/validation/test 的样本数量和 chunk 组织。

## 本地最小子集

已通过 `rsync` 同步每个代表性 store 的首个完整数据 chunk，并在本地只调整 `.zarray` 的第一维 shape 以匹配已同步 chunk；未修改远端数据。

| Local store | samples | target frames | purpose |
|---|---:|---:|---|
| `edm_GOES_ch13_train_dataset.zarr` | 45 | 1 | ordinary training Dataset/DataLoader |
| `edm_GOES_ch13_train_dataset_CorrDiff.zarr` | 45 | 1 | CorrDiff channel/target smoke inspection |
| `edm_GOES_ch13_validation_dataset.zarr` | 32 | 18 | rollout truth sequence inspection |
| `edm_GOES_ch13_test_dataset.zarr` | 32 | 18 | rollout truth sequence inspection |

本地 subset 共 55 个文件、约 145 MB；validation/test 的 18 个 output chunks 均已保留。时间范围没有 timestamp metadata，能支持的时间语义是每个 eval sample 的 18×10 min future truth（3 h）。

不得把上述本地子集规模写成完整数据集规模。

## 代码接口

`ZarrDataset`：

```python
dataset = ZarrDataset(zarr_store)
target, condition = dataset[0]
```

当前实现会把两个数组整体载入 CPU，并转换为 `torch.float16`；完整 train store 不适合直接用该缓存 Dataset 做普通本地 smoke test。远端 `simple_code.md` 同时提供 lazy-loading 版本。

## Smoke test 状态

[BLOCKED] 当前本地 Python 环境同时缺少 `torch` 和 `zarr`，因此尚未运行本地 `cira_diff.dataset.ZarrDataset` / `DataLoader`。已完成元数据、chunk 完整性和本地目录结构检查；不安装或大规模修改环境。

## 仍待确认事项

- `[UNKNOWN]` 每个 patch 对应的原始 GOES 文件时间戳和跨 sample 连续性。
- `[UNKNOWN]` 生成 train/validation/test Zarr 的完整上游索引脚本未随当前目录提供。
