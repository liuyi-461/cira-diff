# 数据集说明（Training）

## 内置数据集
OpenSTL 支持：mmnist、mfmnist、kth、kitticaltech、bair、human、kinetics、taxibj、
weather、sevir 等（见 `configs/`）。

## 本项目目标数据集
- 待定（reproduction Dataset = TODO）。建议与 CIRA-Diff 所用卫星/天气数据对齐。

## 接入方式
- 已有数据集：通过 `get_dataset(dataname, config)` 自动加载。
- 外部数据集：编写 data adapter 映射为 `BaseDataModule`
  （输入 `(B, T, C, H, W)`），见 `docs/data/`。
