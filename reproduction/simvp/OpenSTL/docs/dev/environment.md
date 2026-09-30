# 环境配置（Dev）

## 已验证可用的环境（2026-09-30）

状态：**`Environment Working` 已达成**。

conda env `openstl`（由 `test` 克隆后调整得到，避免动共享环境）：

| 包 | 版本 | 备注 |
| --- | --- | --- |
| Python | 3.11.7 | 上游要求 <=3.10.8，实测 3.11 可跑通 |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 | 8 张 GPU 可见 |
| lightning (pytorch-lightning) | 2.6.6 | |
| **timm** | **0.9.12** | 必须 < 1.0，见 DBG-005 |
| fvcore | 0.1.5.post20221221 | `openstl/api/exp.py` 顶层依赖 |
| opencv-python-headless | 5.0.0.93 | 提供 `cv2` |
| imageio / imageio-ffmpeg | 2.38.0 / 0.6.0 | `openstl/utils/visualization.py` |
| PyWavelets / scikit-image | latest | 提供 `pywt` / `skimage` |
| numpy / zarr / matplotlib | 2.4.6 / 2.18.7 / 3.11.1 | |
| einops / iopath / torchmetrics | 0.8.2 / 0.1.10 / 1.9.0 | |

不需要 `xarray`：zarr 没有 xarray 元数据，adapter 用原生 zarr API 读取。

## 复现上述环境的命令

```shell
conda create -y -n openstl --clone test      # 继承 torch 2.6 + cuda 12.4（免下载）
conda activate openstl
pip install "timm==0.9.12" opencv-python-headless imageio imageio-ffmpeg \
            PyWavelets scikit-image

cd /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
export PYTHONPATH=$PWD:$PYTHONPATH           # openstl 未安装到 site-packages
python -c "from openstl.api import BaseExperiment; print('ok')"
```

## 上游标准流程（作为对照，未采用）

```shell
conda env create -f environment.yml          # 要求 Python<=3.10.8、xarray==0.19.0
conda activate OpenSTL
python setup.py develop
```

未采用的原因：`environment.yml` 会装一套独立的旧 torch/xarray，与本机 CUDA 12.4 +
torch 2.6 冲突；克隆 + pip 补丁更省时，也不会破坏共享的 `test` 环境。

## 注意事项

- **`timm` 必须钉在 0.9.12**；任何把它升到 >= 1.0 的操作都会让 `import openstl` 直接失败。
- `openstl` 是一次性 import 全量 method/dataset 的，所以即便只用 SimVP，WaST、KittiCaltech
  的依赖也必须装齐，详见 `debugging.md` 的 `DBG-005`。
- 直接在 `test` 环境里运行会失败（`timm 1.0.30` 缺 Nadam），slurm 脚本里的
  `CONDA_ENV` 已改为 `openstl`。
