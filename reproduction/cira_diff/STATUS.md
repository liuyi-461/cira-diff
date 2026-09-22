# CIRA-Diff Reproduction 状态

| Gate | 状态 | 证据/下一步 |
| --- | --- | --- |
| Surveyed | Complete | 既有 docs、source 和 paper record |
| Code available | Complete | `cira_diff/`、`scripts/Chase_2025/` |
| Downloaded separate upstream checkout | Not recorded | 声称前先锁定 upstream revision |
| Environment working | Blocked / unknown | 既有 docs 记录缺少 `torch` 和 `zarr` |
| Model construction | 当前 audit 未运行 | 运行 L1 smoke test |
| Checkpoint load | Not recorded | 定位并计算 checkpoint checksum |
| Single-step inference | Not reproduced | 提取并验证 generation path |
| 18-step rollout | Not reproduced | 与 18-frame truth 对齐 |
| Training reproduced | Not reproduced | 先解决 config/path，再运行 tiny test |
| Evaluation reproduced | Not implemented | rollout gate 后建立 common evaluator |

总体状态：`SOURCE AVAILABLE / DOCUMENTED DATA AUDIT / RUNTIME PENDING`。
