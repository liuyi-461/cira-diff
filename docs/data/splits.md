# Split Policy

## CIRA-Diff 记录中的 split

| Split | 记录中的 period/size | 角色 | 状态 |
| --- | --- | --- | --- |
| Train | 2023；超过 30,000 patches；derived store 为 35,595 samples | one-step training | 既有 docs 的 `[EVIDENCE]` |
| Validation | paper record 为 Jan–Aug 2024；derived store 为 1,024 samples | 18-step rollout validation | `[EVIDENCE]`；exact index lineage `[UNKNOWN]` |
| Test | paper record 为 Aug 2024–Feb 2025；derived store 为 1,024 samples | 18-step rollout test | `[EVIDENCE]`；exact index lineage `[UNKNOWN]` |

[CONFLICT] paper-level record 写 validation/test 为 1,000 patches，而 audited derived Zarr 为 1,024 samples。声称 reproduction fidelity 前必须解决 revision/sampling discrepancy。

## 未来工作 split 规则

- 有 source timestamp 时，先按时间或事件切分，再做 random patch sampling。
- temporal-history 和 model comparison 使用相同 split manifest。
- 不得用 test truth 做 checkpoint selection 或 normalization estimation。
- 报告 sample count、frame count、region、event overlap 和 missing-data filtering。
- 保留 source identifier manifest；当前 derived Zarr metadata 不足以满足这一要求。
