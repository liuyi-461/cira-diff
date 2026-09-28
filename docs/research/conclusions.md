# 科研结论

## 当前结论

### CON-001：项目范围大于 CIRA-Diff

[CONCLUSION] 研究目标是卫星云图预报。CIRA-Diff 是当前 baseline 和 reproduction 起点，不是关于卫星预报正确 temporal architecture 的结论。

**证据：** repository audit、既有 project context 和 temporal-formulation 问题。

### CON-002：记录中的 CIRA-Diff 数据合同区分 training truth 与 rollout truth

[CONCLUSION] ordinary train sample 记录为 2 个输入帧加 1 个 target，validation/test sample 记录为 2 个输入帧加 18 个 future truth 帧。这是数据组织结论，不是 model-skill 结论。

**证据：** `docs/training/dataset.md` 和 `docs/data/cira.md` 中记录的此前 Zarr metadata audit。

### CON-003：当前不支持任何科学模型结论

[CONCLUSION] 当前仓库没有已验证的本地 forecast run、统一 evaluation artifact 或受控比较，因此不能支持 model superiority、temporal formulation、diffusion value 或 event skill 的结论。

**证据：** `docs/project/repository_audit.md`、`reproduction/cira_diff/STATUS.md` 和 `docs/experiments/registry.md`。

## 明确不是当前结论

- “两帧已经足够。”
- “autoregressive forecasting 是卫星预报的正确范式。”
- “diffusion 提升了 meteorological forecast skill。”
- “视觉上真实的预报就是物理或气象上准确的预报。”
- “计划中的 baseline 已经 reproduced。”
