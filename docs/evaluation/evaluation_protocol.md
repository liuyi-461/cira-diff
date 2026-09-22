# Evaluation Protocol

状态：`PLANNED / REQUIRED BEFORE MODEL COMPARISON`。

## 最小 protocol

1. 冻结 dataset ID、split manifest、spatial crop、temporal interval 和 normalization。
2. 冻结 checkpoint-selection rule 以及 code/config revision。
3. 定义 forecast lead times，包括第一个 10-minute step，以及适用时到 3 h 的每个 10-minute rollout lead。
4. 先评价 persistence 和其他 deterministic baseline，再解释 learned method。
5. 先报告 ME/MAE/RMSE，再加入 structure、scale、object 和 event metrics。
6. 明确 threshold、mask、units、aggregation 和 missing-data policy。
7. 对 ensemble 同时报告 deterministic summaries 和 probabilistic diagnostics。
8. 保存 predictions、truth references、metric tables、plots、commands、environment 和 checksums。

## Validation ladder

```text
L0 contract → L1 reference inference → L2 one-sample smoke test
→ L3 tiny controlled set → L4 first real experiment → L5 robust comparison
```

如果 L0/L1 语义未解决，L4/L5 结论无效。
