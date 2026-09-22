# 知识缺口

| ID | 缺口 | 影响 | 当前证据 | 下一步安全行动 |
| --- | --- | --- | --- | --- |
| GAP-001 | 本地 `torch`/`zarr` smoke test 未运行 | 无法验证 Dataset/DataLoader 行为 | 既有文档标记为 blocked | 使用隔离环境和已记录的小 subset |
| GAP-002 | 未锁定 exact upstream source revision | 无法区分原始行为和本地漂移 | 有 Git history，但无 upstream commit | reproduction 前记录 upstream URL、revision 和 local diff |
| GAP-003 | 缺少独立 rollout CLI | 预报语义仍埋在旧 notebook/script 中 | `cira_diff/generate.py` 是 placeholder | 保持语义提取并增加 alignment test |
| GAP-004 | Zarr 没有 per-sample timestamp/source index | 限制物理解释和序列重建 | 既有 metadata 审计 | 获取 generation manifest 或原始 index 文件 |
| GAP-005 | normalization 存在不一致风险 | 会改变 metric 和模型输入语义 | 数据统计量与脚本常数均有记录 | 正式训练前依据 source artifact 解决 |
| GAP-006 | 没有本地 forecast result artifact | 当前不存在项目级科学结果 | 只有数据审计结果 | 先运行 L0/L1 reference checks |
| GAP-007 | threshold policy 未定义 | event metrics 无法公平比较 | 目前只有候选指标 | 定义 threshold 来源和 mask |
| GAP-008 | Himawari data contract 不完整 | 跨传感器比较可能无效 | 只有任务提示级上下文 | 验证 raw source、calibration、grid、timestamp 和 split |
| GAP-009 | baseline upstream provenance 不完整 | 计划模型不能称为 reproduced | 当前只有 scaffold | 先审计论文、官方仓库和 checkpoint，不下载 |
| GAP-010 | 没有 case/regime labels | 无法分层支持对流和热带气旋结论 | 当前数据记录中没有 labels | 定义 labels，或明确限制结论 |
| GAP-011 | 论文写 1,000 个 validation/test patch，derived Zarr 审计写 1,024 | reproduction sample count 可能不同 | `docs/data/splits.md` 标记 `[CONFLICT]` | 识别数据 revision/sampling rule |
