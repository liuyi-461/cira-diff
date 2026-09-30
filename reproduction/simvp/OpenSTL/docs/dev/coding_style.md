# 编码规范（Dev）

## 风格
- 遵循上游 OpenSTL 的命名与模块划分（method / model / module 三层）。
- 新功能优先以 config 驱动，不在核心代码硬编码。

## 扩展方式
- 新增模型：在 `openstl/models/` 加模型类，在 `openstl/methods/` 加 method，
  并在 `method_maps` 注册。
- 新增数据集：在 `openstl/datasets/` 实现并在 `get_dataset` 注册。

## 约束
- 不修改上游核心接口；reproduction 适配通过 adapter 完成。
