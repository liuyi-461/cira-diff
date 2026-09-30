# 卫星数据处理规范（Skill）

## 数据形态
- 静止 / 极轨卫星影像常为 `(time, channel, lat, lon)` 网格化场。
- 映射到 OpenSTL：`(B, T, C, H, W)`，H/W 为空间网格。

## 处理要点
- 缺失值 / 云遮挡处理（reproduction 待定）。
- 时间采样间隔决定 `pre_seq_length` 与 `aft_seq_length` 的物理含义。
- 投影一致性：对比实验须保持同一网格 / 投影。

## 复用建议
- 通过 data adapter 将卫星数据接入 `openstl.datasets.BaseDataModule`。
