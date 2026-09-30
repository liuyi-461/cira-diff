# 知识缺口

| ID | 缺口 | 影响 | 当前证据 | 下一步安全行动 |
| --- | --- | --- | --- | --- |
| GAP-001 | ~~本地 `torch`/`zarr` smoke test 未运行~~ | **已解决** | [FACT] EXP-009（overfit gate check）和 EXP-010（full training）均验证了 Dataset/DataLoader 正常运行 | 无需进一步行动 |
| GAP-002 | 未锁定 exact upstream source revision | 无法区分原始行为和本地漂移 | 当前 main 为 `3ffa797`，无 upstream commit hash | reproduction 前记录 upstream URL、revision 和 local diff |
| GAP-003 | 缺少独立 rollout/evaluation CLI | 预报语义仍埋在旧 notebook/script 中；当前训练脚本只输出单步 MSE，无论文要求的 18-step rollout | EXP-010 正在训练，但尚无 rollout inference | 从 `cira_diff/generate.py` placeholder 或旧 notebook 提取 rollout 路径，实现 alignment test |
| GAP-004 | Zarr 没有 per-sample timestamp/source index | 限制物理解释和序列重建 | 既有 metadata 审计 | 获取 generation manifest 或原始 index 文件 |
| GAP-005 | normalization 存在不一致风险 | 会改变 metric 和模型输入语义 | [FACT] 本地训练脚本未显式做 mean/std；overfit 和 full training 均一致跳过 | 检查原始 CIRA-Diff 脚本是否有隐藏归一化，或直接用原始值 |
| GAP-006 | ~~没有本地 forecast result artifact~~ | **已部分解决**：EXP-009 有单样本 overfit 结果；EXP-010 正在训练，但尚无泛化评估结果 | EXP-009 单样本 MSE=3.95e-05；EXP-010 train loss ~0.01（running） | 等 EXP-010 完成后做 held-out val + rollout evaluation |
| GAP-007 | threshold policy 未定义 | event metrics 无法公平比较 | 目前只有候选指标 | 定义 threshold 来源和 mask |
| GAP-008 | Himawari data contract 不完整 | 跨传感器比较可能无效 | 只有任务提示级上下文 | 验证 raw source、calibration、grid、timestamp 和 split |
| GAP-009 | baseline upstream provenance 不完整 | 计划模型不能称为 reproduced | 当前只有 scaffold + 本地修改的训练脚本 | 审计论文官方仓库是否提供 Vanilla UNet / EDM / LDM / CorrDiff checkpoint；论文数字是否给出 Vanilla UNet 单独指标 |
| GAP-010 | 没有 case/regime labels | 无法分层支持对流和热带气旋结论 | 当前数据记录中没有 labels | 定义 labels，或明确限制结论 |
| GAP-011 | 论文写 1,000 个 validation/test patch，derived Zarr 审计写 1,024 | reproduction sample count 可能不同 | `docs/data/splits.md` 标记 `[CONFLICT]` | 识别数据 revision/sampling rule |
| GAP-012 | 训练脚本加载独立 val zarr 但实际未使用 | 浪费 ~145MB 内存 | [FACT] EXP-010 脚本中独立 val zarr 被 `print` 后未进入 DataLoader | 清理未使用的独立 val zarr 加载；或修改脚本让 held-out val split 和独立 val 同时评估 |
| GAP-013 | LDM latent MSE 未经过 VAE decode 到像素空间 | EXP-009 中 LDM 与其他方法的 MSE 比较量纲不一致 | [ISSUE] latent 空间 64×64×4ch vs pixel 256×256×1ch | 实现 VAE decode + 统一 pixel-space evaluation |
| GAP-014 | EDM / LDM / CorrDiff full training 尚未开始 | 只有 Vanilla UNet baseline 在跑 | [RUNNING] EXP-010（Vanilla UNet）；EXP-011–013 Planned | 依次提交，复用 bs=24 + grad_accum=2 配置 |
| GAP-015 | 没有从 checkpoint 到 forecast skill 的完整评估 pipeline | 即使训练完成也无法产出论文要求的完整评估报告 | 当前只有单步 MSE | 实现 `evaluate.py` 加载 checkpoint → autoregressive rollout → 18-step metrics → cold-cloud analysis → ensemble calibration |