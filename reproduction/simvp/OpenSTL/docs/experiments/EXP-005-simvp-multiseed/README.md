# EXP-005 — SimVP 多 seed rollout 方差（与 UNet 对照）

## Identity
- 实验 ID：`EXP-005`
- 状态：`AUTO-GENERATED`（由 `tools/aggregate_multiseed.py` 生成）
- 日期：2026-10-07
- 关联：`EXP-004`（统一 baseline 协议）、ly 的 `EXP-EVAL-003`（UNet 多 seed）

## 目的
ly 已证明 Vanilla UNet 的 rollout 对随机种子极敏感（LT=18 CV = 13.4%），单 seed 之间比 rollout 无意义。本实验训练 SimVP 的多个 seed，拿到 SimVP 自身的 rollout 方差，再与 UNet 做**带方差**的对比，判断 EXP-004 中"SimVP 在 LT=18 比 UNet 差 60%"是否为稳健结论。

## 设置
- 与 EXP-004 完全相同的配置（`configs/satcast/SimVP.py`）、相同数据（三份 zarr）
- **仅改变模型初始化 seed**（数据 split 固定，不存在 split_seed 变化）
- seed 集合：0 / 42 / 123（与 ly 保持一致，便于对照）
- 每个 seed 评估：单步（test 1024 条）+ 18 步 rollout（128 条子集）

<!-- MULTISEED:BEGIN -->
## Results — SimVP 多 seed（自动生成于 2026-10-07 20:27）

> ⚠️ 当前只有 1 个 seed 的结果（42），无法计算方差。至少 2 个才有 std，建议 ≥3。
<!-- MULTISEED:END -->
