# 数据谱系

## CIRA-Diff 已记录的 lineage

```text
NOAA GOES Open Data［既有 docs 记录的 source］
  → ABI Channel 13 IR brightness temperature
  → calibration / patch construction［部分步骤不可用］
  → 10-minute temporal samples
  → 256×256 crop/patch
  → train mean/std normalization
  → Zarr input_images/output_images
  → ZarrDataset
  → CIRA-Diff single-step forecast
  → autoregressive 18-step forecast
  → pixel/structure/event evaluation［evaluation CLI 尚未验证］
```

## Traceability 要求

每个未来 forecast artifact 应能追溯：

`conclusion → experiment ID → config → data manifest → code revision → checkpoint → forecast artifact → metric artifact`。

## 已知断点

- 记录中的 Zarr metadata 没有 per-sample timestamp/source index；
- 当前 tree 没有 exact generation script；
- checkpoint 和 forecast output provenance 未记录；
- evaluation 层尚未形成已验证 CLI。
