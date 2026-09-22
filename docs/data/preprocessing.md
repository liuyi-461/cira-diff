# 预处理合同

| 阶段 | 预期操作 | 状态 |
| --- | --- | --- |
| Raw satellite data | 获取 calibrated ABI/IR observations | CIRA-Diff source family 为 `[EVIDENCE]`；raw files 不在当前 tree |
| Calibration/conversion | 保留 brightness-temperature 语义 | exact upstream implementation 为 `[UNKNOWN]` |
| Spatial crop/patch | 生成 256×256 samples | CIRA-Diff derived data 为 `[EVIDENCE]` |
| Temporal sampling | 10-minute history/target spacing | CIRA-Diff task record 为 `[EVIDENCE]` |
| Normalization | 使用 train mean/std | 数据值为 `[EVIDENCE]`；脚本一致性未解决 |
| Sample construction | `2→1` train，`2→18` eval | 既有 metadata audit 的 `[RESULT]` |
| Missing/quality control | 拒绝或 mask invalid observations | `[UNKNOWN]` |
| Geolocation metadata | 保留 projection/crop coordinates | 当前 Zarr record 为 `[UNKNOWN]` |

在 input/output shape、dtype、range、physical unit 和 provenance 被记录前，预处理步骤不视为可复用代码。
