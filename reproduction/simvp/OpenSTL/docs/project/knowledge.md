# 长期经验与知识沉淀

## 已验证规律
- OpenSTL 训练入口 `tools/train.py` 通过 `configs/<dataname>/<method>.py` 加载超参；
  `--overwrite` 控制是否用 config 覆盖 CLI 默认值。
- `BaseExperiment.display_method_info` 对 simvp/tau/mmvp/wast 使用
  `input_dummy = ones(1, pre_seq_length, C, H, W)` 计算 FLOPs。

## 常见问题 / 避坑
- 不要直接修改上游 `openstl/` 核心代码来适配本项目；优先通过 config 与 adapter 扩展。
- `tests/test_models/test_simvp.py` 已过期：`SimVP_Model(arch=...)` 与
  `SimVP_Model(num_layers=3, num_hidden=1)` 均不符合当前构造签名
  `(in_shape, hid_S, hid_T, N_S, N_T, model_type, ...)`，运行会触发非预期的 AssertionError。
  修复时应同步更新该测试而非修改模型签名。

## 技术经验
- SimVP 的 `MidMetaNet` 把时间维拼到通道维（`B, T*C, H, W`）后做空间翻译，
  因此输入分辨率需满足下采样比例（`N_S/2` 次 2× 下采样）。

## 数据集接入经验（2026-09-30，新增）
- `edm_GOES_ch13_train_dataset.zarr` **没有** xarray 的 `_ARRAY_DIMENSIONS` 元数据，
  `xr.open_zarr()` 会抛 `KeyError`；必须用原生 `zarr.open/open_group` 读取。
- 接入新数据集优先用 `BaseExperiment(dataloaders=...)` 注入，不要用 `-d <name>`
  的注册路径去改上游文件（见 DEC-003）。
- `BaseExperiment._init_trainer` 硬编码 `accelerator='gpu'`，无 GPU 机器会直接失败；
  需要 CPU 时子类化并重写 `_init_trainer`，而不是改上游。
- `create_loader` 默认 `persistent_workers=True`，当 `num_workers=0` 时 PyTorch 会报错
  （`ValueError`，不会被其内部的 `except TypeError` 捕获）；务必显式传
  `persistent_workers=(num_workers > 0)`。
- 自定义 Dataset 必须暴露 `mean` / `std` / `data_name` 三个属性，否则
  `BaseDataModule` 构造即失败；`metric()` 会用它们反归一化后再算指标。

## 单样本过拟合 / 可视化经验（2026-09-30，新增）
- 单样本 loader 必须 `drop_last=False` 且 `shuffle=False`：
  `load_data()` 里 train loader 用 `drop_last=True`，只有 1 个样本会被直接丢弃、变空 loader。
- 归一化统计量**不能**从单个样本算（`std` 会退化），应从参考样本池估算
  （见 `--ref-samples`，偏差 D4）。
- 不走 `BaseExperiment` 加载配置时，务必显式 `update_config(...)` 读入 config 文件，
  否则 `hid_S/hid_T/N_T` 等只会落到 `SimVP_Model` 的内部默认值，实验不可复现。
- 可视化前必须先**反归一化**（`pred * std + mean`），否则量纲不对，与既有
  `test_dl/test_single_sample_Chase2025.py` 的 `vmin=-4, vmax=2` 不可比。
- 判读顺序：先看单样本 loss 曲线是否数量级下降，再看并列图 predicted 是否复现 truth；
  loss 不降 → 查 lr/归一化；输出无结构 → 查通道 / 时间轴 / skip 连接。
