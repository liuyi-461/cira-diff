# SimVP 源码使用说明（卫星云图临近预报）

## 一、已下载内容

| 项目 | 说明 |
| --- | --- |
| 来源仓库 | https://github.com/chengtan9907/OpenSTL |
| 本地位置 | `project_none/OpenSTL/` |
| 版本 | `1.0.0`（`openstl/version.py`） |
| 默认分支 | `OpenSTL-Lightning` |
| Commit | `eecf8a3078f0a178dbc7b28723da20f94ce36985`（2025-10-22） |
| 授权 | Apache License 2.0 |
| 体积 | 约 9.6 MB，551 个文件 |

> 完整打包版另存为 `OpenSTL_SimVP.zip`（已剔除 `.git` 目录）。

---

## 二、重要说明：SimVP 不是一个独立可运行的包

SimVP 是 **OpenSTL 框架内置的一个方法**，不是单独发布的 PyPI 包。它的模型代码依赖框架的数据加载、训练循环、指标计算等模块：

```
SimVP_Model (模型)  →  依赖 openstl.modules（模块层）
                    →  由 openstl.methods.simvp 包装（训练/验证/预测流程）
                    →  由 openstl.api.BaseExperiment 驱动（实验管理）
                    →  依赖 openstl.datasets / openstl.core（数据与训练核心）
```

**因此，单独把 `simvp_model.py` 复制出去是无法运行的。** 所以这里保留了完整的 OpenSTL 仓库，这样拿到手就能直接训练。下面的章节帮你快速定位 SimVP 相关的部分。

---

## 三、SimVP 核心文件位置

| 路径 | 行数 | 作用 |
| --- | --- | --- |
| `openstl/models/simvp_model.py` | 247 | **模型本体**：`SimVP_Model`、`Encoder`、`Decoder`、`MidIncepNet`、`MetaBlock`、`MidMetaNet` |
| `openstl/modules/simvp_modules.py` | 586 | **算子模块**：`ConvSC`、`gInception_ST`、`GASubBlock`、各 backbone 子模块 |
| `openstl/methods/simvp.py` | 48 | **训练方法封装**：train / validation / predict 流程 |
| `tests/test_models/test_simvp.py` | — | 单元测试 |
| `configs/*/SimVP.py` | — | 基础配置（按数据集分目录） |
| `configs/*/simvp/SimVP_*.py` | — | 各 backbone 变体配置 |

### 可用的 backbone 变体

`configs/<数据集>/simvp/` 下每个数据集都提供了同一套变体：

`SimVP_gSTA`（默认）、`SimVP_IncepU`、`SimVP_ViT`、`SimVP_Swin`、`SimVP_ConvMixer`、`SimVP_ConvNeXt`、`SimVP_HorNet`、`SimVP_MLPMixer`、`SimVP_MogaNet`、`SimVP_Poolformer`、`SimVP_Uniformer`、`SimVP_VAN`，以及大模型 `SimVP-L` / `SimVP_gSTA-L`。

模块定义见 `openstl/modules/simvp_modules.py`（`GASubBlock`、`ConvMixerSubBlock`、`ConvNeXtSubBlock`、`HorNetSubBlock`、`MLPMixerSubBlock`、`MogaSubBlock`、`PoolFormerSubBlock`、`SwinSubBlock`、`VANSubBlock`、`ViTSubBlock`）。

---

## 四、与「卫星云图临近预报」直接相关的配置

OpenSTL 内置了多个气象类数据集，**这两个和你的任务最贴近**：

### 1. `configs/sevir/` —— 强烈推荐

SEVIR 是临近预报领域的标准 benchmark（含 GOES-16 卫星红外云图 + NEXRAD 雷达 VIL 数据）。

```python
# configs/sevir/SimVP.py
method = 'SimVP'
model_type = 'IncepU'
hid_S = 64
hid_T = 256
N_T = 4
N_S = 2
lr = 5e-3
batch_size = 8
drop_path = 0.1
sched = 'onecycle'
metric_threshold = 74      # ← 临近预报标准评测阈值（VIL >= 74）
```

注意 `metric_threshold = 74`：SEVIR 临近预报惯例是对 VIL ≥ 74 的强回波区域单独评测，这正是 nowcasting 的标准做法。

### 2. `configs/weather/tcc_5_625/` —— 总云量

`tcc` = Total Cloud Cover（总云量），与卫星云图任务语义最接近。

### 其他气象配置

`configs/weather/` 下还有：`t2m_5_625`（2米气温）、`t2m_1_40625`、`r_5_625`（相对湿度）、`uv10_5_625`（10米风场）、`mv_4_s6_5_625`。

---

## 五、环境安装

仓库自带 conda 环境文件（`environment.yml`），这是最省事的方式：

```bash
cd /Users/yeechan/Desktop/project_none/OpenSTL
conda env create -f environment.yml
conda activate OpenSTL
python setup.py develop
```

**依赖要求**（来自 `requirements/runtime.txt`）：

- `torch`、`lightning==2.2.1`、`timm`、`einops`
- `numpy`、`xarray`、`netcdf4`、`dask`（气象数据处理）
- `opencv-python`、`scikit-image`、`matplotlib`、`decord`
- 关键约束：**`python<=3.10.8`**

### ⚠️ 当前机器环境提示

```
Python 3.9.6   （版本本身兼容）
torch          未安装
平台           macOS (darwin)
```

Python 版本没问题，但**当前没有安装 PyTorch**，而且这是 macOS 机器。SimVP 训练卫星云图数据（尤其 SEVIR 这类高分辨率时序数据）显存占用较高，实际训练基本需要 NVIDIA CUDA GPU——macOS 只能走 MPS/CPU，速度会慢很多。

建议：**在这台 Mac 上做代码阅读、数据处理和调试；实际训练放到 Linux + CUDA 的机器或服务器上。** 如果你确实要在 Mac 上跑，请装支持 MPS 的 PyTorch，并把 `--device` 显式指定。

---

## 六、运行训练

### 基本命令格式

```bash
python tools/train.py -d <数据集> -m <方法> -c <配置文件> --ex_name <实验名>
```

参数说明（来自 `openstl/utils/parser.py`）：

- `-d / --dataname`：数据集名，默认 `mmnist`
- `-m / --method`：方法名，默认 `SimVP`
- `-c / --config_file`：配置文件路径。**若不指定**，会自动去找 `./configs/{dataname}/{method}.py`
- `--data_root`：数据根目录，默认 `./data`
- `--ex_name / -ex`：实验名，决定输出目录
- `--res_dir`：结果目录，默认 `work_dirs`
- `--device`：默认 `cuda`
- `-e / --epoch`、`-b / --batch_size`、`--lr`：可覆盖配置

### 示例

```bash
# 最简写法：自动匹配 configs/sevir/SimVP.py
python tools/train.py -d sevir -m SimVP --ex_name sevir_simvp

# 指定变体配置 + 显式超参覆盖
python tools/train.py -d sevir -m SimVP \
    -c configs/sevir/SimVP.py \
    -b 8 --lr 5e-3 -e 200 \
    --data_root ./data \
    --ex_name sevir_simvp_incepu

# 官方 README 中 Moving MNIST 的完整示例
python tools/train.py -d mmnist --lr 1e-3 \
    -c configs/mmnist/simvp/SimVP_gSTA.py \
    --ex_name mmnist_simvp_gsta
```

训练脚本会自动在训练结束后执行测试（见 `tools/train.py` 末尾的 `exp.test()`）。单独测试用 `python tools/test.py`。

---

## 七、用自己的卫星云图数据训练

这是你最终要走的路，仓库已提供官方教程：

- **教程 Notebook**：`examples/tutorial.ipynb` —— 讲解如何在自定义数据上训练、评估、可视化
- **自定义配置模板**：`configs/custom/example_model.py`
- **数据接口**：`openstl/datasets/base_data.py`（自定义 Dataset 基类）
- **现成参考**：`openstl/datasets/dataloader_sevir.py`、`dataloader_weather.py`（照着写自己的 loader 最快）
- **数据转换工具**：`openstl/datasets/pipelines/transforms.py`
- **数据预处理脚本**：`tools/prepare_data/generate_sevir.py`（可参考其切片/归一化逻辑）

典型做法是：把你的卫星云图序列整理成 `(T, C, H, W)` 的时序样本，继承 `base_data.py` 里的 Dataset，然后在 `configs/custom/` 下写自己的配置。

---

## 八、随仓库附带的说明文档

| 文档 | 内容 |
| --- | --- |
| `README.md` | 项目总览、Model Zoo、数据集、可视化 |
| `docs/en/install.md` | 详细安装说明 |
| `docs/en/get_started.md` | 快速上手 |
| `docs/en/changelog.md` | 版本变更记录 |
| `docs/en/model_zoos/weather_benchmarks.md` | **气象数据集上的精度对标表** |
| `docs/en/model_zoos/video_benchmarks.md` | 视频数据集精度对标表 |
| `docs/en/visualization/weather_visualization.md` | 气象预测结果可视化 |
| `examples/tutorial.ipynb` | 自定义数据完整教程 |
| `examples/` | 训练/验证/测试的示例动图 |

> 注意：官方文档只有英文版（`docs/en/`），仓库中不含中文文档。

---

## 九、关于分支版本的提醒

本仓库默认分支是 **`OpenSTL-Lightning`**（v1.0.0），已迁移到 **PyTorch Lightning** 架构。

你在网上看到的早期 SimVP 教程/博客，很多是基于旧的 `master` 分支（非 Lightning 版本），代码组织和 API 会有差异。如果按老教程操作对不上，请以本仓库实际代码为准。

---

## 十、建议的下一步

1. 先装环境（conda 方式），在 `sevir` 或 `mmnist` 上跑通一次训练，确认框架可用
2. 读 `examples/tutorial.ipynb`，理解自定义数据的接入方式
3. 参考 `dataloader_sevir.py` 写你自己的卫星云图 Dataset
4. 从 `configs/sevir/SimVP.py` 起步调参（先跑 `IncepU` 或默认 `gSTA`）

如需我帮你写自定义数据集的 Dataset / DataLoader，或把某份卫星云图数据整理成 OpenSTL 格式，随时说。
