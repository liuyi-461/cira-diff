# CIRA-Diff 数据合同

状态：`DOCUMENTED FROM PRIOR AUDIT`；独立 runtime verification pending。

## Source 与物理含义

- **Source：** GOES-16 ABI Channel 13，10.3 μm IR brightness temperature——`[EVIDENCE]`，来自既有项目 docs 和 paper record。
- **Temporal interval：** 10 minutes——`[EVIDENCE]`。
- **Spatial sample：** 256×256 patch——`[EVIDENCE]`。
- **Unit：** standardization 前为 brightness temperature，单位 K——`[EVIDENCE]`。
- **Region/full-disk semantics：** 既有 docs 记录了 full-disk/near-nadir 背景；exact patch geolocation 为 `[UNKNOWN]`。

## 已记录 stores

| Store | Input shape | Target shape | dtype | 角色 |
| --- | --- | --- | --- | --- |
| ordinary train | `(35595, 2, 256, 256)` | `(35595, 1, 256, 256)` | `float16` | one-step training |
| CorrDiff train | `(35595, 3, 256, 256)` | `(35595, 1, 256, 256)` | `float16` | 两帧 history + first guess；residual target |
| validation | `(1024, 2, 256, 256)` | `(1024, 18, 256, 256)` | `float16` | rollout truth |
| test | `(1024, 2, 256, 256)` | `(1024, 18, 256, 256)` | `float16` | rollout truth |
| latent train | `(35595, 8, 64, 64)` | `(35595, 4, 64, 64)` | `float16` | latent path |
| latent validation/test | `(1024, 8, 64, 64)` | `(1024, 72, 64, 64)` | `float16` | latent rollout truth |

## Normalization

[EVIDENCE] 既有 docs 记录：

```text
mean = 279.0699458792467 K
std  = 19.32967519050003 K
normalized = (Tb - mean) / std
```

[ISSUE] 旧训练脚本还包含 normalized-space hard-coded constants。正式运行前必须依据 exact source artifact 解决该不一致。

## 时间语义

[RESULT] 记录中的 ordinary sample 为 `history=2 → target=1`；validation/test 为 `history=2 → future=18`。后者是 18-step rollout 对齐的依据。

[UNKNOWN] Zarr metadata 没有 per-sample timestamp 或 source filename。仅靠 derived store 不能恢复 sample 之间的原始序列连续性。
