# EXP-001 — CIRA-Diff 仓库与数据审计

## Identity

- 实验 ID：`EXP-001`（保留既有 research record 的编号）
- 状态：`COMPLETED AUDIT; MODEL RUN NOT EXECUTED`
- 日期：既有记录为 2026-09-17，本次重建为 2026-09-22
- 关联 RQ：RQ-001、RQ-004
- 关联 HYP：没有已测试假设

## Research question 与 hypothesis

- Research question：当前 CIRA-Diff sample semantics 和 reproduction boundary 是什么？
- Hypothesis：ordinary train sample 与 rollout evaluation sample 的 target length 不同。
- Scientific motivation：避免将三帧训练窗口误认为 18-step truth sequence，或误认为 rollout training。

## Dataset

- Remote dataset：`26fall-AIclass:/data1/satcast/`——既有 docs 记录，当前 Git tree 不包含。
- Local development subset：既有 docs 记录 45 train、45 CorrDiff train、32 validation、32 test；当前 checkout 不含 tracked payload。
- Sensor/channel：GOES-16 ABI Channel 13 IR brightness temperature。
- Spatial resolution/crop：记录为 256×256 patch；exact patch geolocation unknown。
- Temporal resolution：记录为 10 minutes。
- Split：规模见 `docs/data/cira.md`。
- Sample construction：ordinary `2→1`；evaluation `2→18`；CorrDiff input 含 first guess 的 3 channels。
- Normalization：train mean/std 已记录；脚本一致性未解决。

## Model

- Architecture：本实验未运行。
- Input/output/rollout：根据 source/paper 记录；没有 forecast artifact。
- Teacher forcing/scheduled sampling/rollout training：`Not Recorded`；不得从 autoregressive inference 推断。

## Training configuration

所有模型训练字段：`Not Applicable — no model training executed`。

## Runtime 与 artifacts

- Command：此前 audit command 未完整保存在本卡。
- Environment：既有 docs 记录本地 smoke test 缺少 `torch` 和 `zarr`。
- Git commit：本次重建为 `12c4bf9`；本次没有迁移源码。
- Log/output/checkpoint：`Not Recorded`。

## Evaluation protocol

- 已评价：Zarr shape、dtype/chunks/compressor，以及代表性 sample/no-NaN inspection，均来自既有 audit 记录。
- Forecast metrics：无。
- Lead times/thresholds：不适用。

## Results

[RESULT] 既有 audit 记录 ordinary train 为 2 个 input frame 加 1 个 target frame，validation/test 为 2 个 input frame 加 18 个 future truth frame；同时记录了 `float16`、256×256 sample 和 inspected sample 无 NaN。

[RESULT] 本 audit 没有运行模型、load checkpoint、生成 forecast 或计算 forecast metrics。

## Interpretation

仓库知识现在可以区分 sample organization 与 forecast behavior，但尚不能证明模型复现论文或 CIRA-Diff temporal design 科学上最优。

## Conclusion

[CONCLUSION] 数据组织 hypothesis 得到已有 metadata audit 支持。所有 model-skill 和 temporal-formulation 问题仍 unresolved。

## Limitations

- 本记录是此前 audit 的重建，没有在本轮重新运行；
- derived Zarr metadata 没有 timestamp/source index；
- 没有 executable rollout/evaluation CLI；
- 没有 model artifact。

## Reproducibility

见 `docs/data/cira.md`、`docs/project/repository_audit.md` 和 `reproduction/cira_diff/STATUS.md`。在 environment、checkpoint 和 command 锁定前，本记录不足以重现 forecast。

## 关联

- Literature：`docs/research/cira_diff.md`
- Hypotheses：`docs/research/hypotheses.md`
- 下一实验：EXP-002 reference smoke/rollout gate
