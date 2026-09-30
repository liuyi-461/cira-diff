# SimVP 单样本测试（GOES-16 ABI ch13，2 帧输入 -> 1 帧输出）

目的：**先把 SimVP 整条链路跑通**。用 /data1/satcast 的 GOES 通道 13 数据，
在单条样本上做「过拟合 / 记忆」测试，通过 slurm 提交到 GPU 节点训练，
并和 persistence 基线对比。

> 这是单样本过拟合测试：只说明链路能跑通、模型能记住这一条样本，
> **不代表任何泛化能力**。

---

## 一、目录结构

```
single_sample_test/
├── OpenSTL/                        # 复制自 ../../OpenSTL（原版未改动），本目录内已做下述修改
├── eval_single_sample.py           # 训练后评估：加载 best.ckpt + persistence 基线对比 + 出图
├── run_single_sample_simvp.slurm   # slurm 提交脚本（训练 + 评估）
├── logs/
│   ├── output/                     # slurm stdout（--output）
│   └── err/                        # slurm stderr（--error）
├── outputs/                        # 训练产物：checkpoints / 指标 JSON / 对比图 / lightning_logs
└── README.md
```

## 二、数据契约

来源：`/data1/satcast/edm_GOES_ch13_train_dataset.zarr`

| 数组 | 形状 | dtype | 含义 |
| --- | --- | --- | --- |
| `input_images` | (35595, **2**, 256, 256) | float16 | 条件帧 `[t-10min, t]` |
| `output_images` | (35595, **1**, 256, 256) | float16 | 目标帧 `[t+10min]` |

即 **2 帧输入 -> 1 帧输出**。zarr 里的 "2" 是**时间维**不是通道维，
所以送进 SimVP 的张量是 `(B, T, C, H, W) = (1, 2, 1, 256, 256)`（C=1 灰度）。

数据已经是归一化空间，反归一化常数（见 `/data1/satcast/simple_code.md`）：

```
normalized = (Tb - 279.0699458792467) / 19.32967519050003
```

> 注意：`validation/test` 两个 zarr 的 `output_images` 是 **18 帧**（多步 rollout 用），
> 与本任务的 1 帧目标形状不匹配。所以本测试对 train/val/test **统一使用训练集 zarr**，
> 靠 `single_sample_idx` 只取其中一条样本 —— 这是有意的，单样本测试里 val/test
> 本来就该是同一条样本（与 `cb/train/train_unet_single_sample.py` 的做法一致）。

## 三、相对 OpenSTL 原版的修改

所有改动都在本目录的 `OpenSTL/` 副本内，`../../OpenSTL/` 保持原样。

| # | 文件 | 改动 | 原因 |
| --- | --- | --- | --- |
| 1 | `openstl/datasets/dataloader_satcast.py` | **新增** | 2 帧输入/1 帧输出的 zarr dataloader |
| 2 | `openstl/datasets/dataloader.py` | 注册 `satcast` 分支；透传 `zarr_name` / `single_sample_idx` | 让框架能调到新 loader |
| 3 | `openstl/datasets/dataset_constant.py` | 新增 `dataset_parameters['satcast']` | 声明 `in_shape=[2,1,256,256]`、`pre/aft_seq_length=2/1`、`metrics` |
| 4 | `openstl/datasets/__init__.py` | 导出 `SatcastZarrDataset` | 顺带修了原文件里 `'SEVIRDataset' 'load_data'` 缺逗号的字符串拼接 bug |
| 5 | `openstl/utils/parser.py` | `--dataname` 的 choices 加上 `satcast` | 否则 argparse 直接拒绝 |
| 6 | `openstl/core/optim_scheduler.py` | timm 的 `Nadam` / `RAdam` 加 `try/except` 回落到 `torch.optim` | 见下 |
| 7 | `configs/satcast/SimVP.py` | **新增** | 实验配置 |

### 关键设计说明

**「2 帧进、1 帧出」没有改模型代码。** OpenSTL 的 `openstl/methods/simvp.py`
本身支持 `aft_seq_length < pre_seq_length`：

```python
elif aft_seq_length < pre_seq_length:
    pred_y = self.model(batch_x)
    pred_y = pred_y[:, :aft_seq_length]     # 模型出 2 帧，取前 1 帧
```

所以模型按 `T=2` 编码、解码出 2 帧，只有第 0 帧被监督。走官方已有代码路径，
不引入自定义的前向逻辑。

**单样本是怎么实现的。** `single_sample_idx` 不为 `None` 时，三个 split 都用
`Subset(dataset, [idx])` 只取那一条样本，并把 batch_size 强制成 1。
这里用了一个 `Subset` 子类 `_SubsetWithAttrs`：`Subset` 不转发底层 dataset 的属性，
而 OpenSTL 的 `BaseDataModule` 要读 `test_loader.dataset.{mean,std,data_name}`。

**指标单位。** `dataset.mean/std` 存的是**物理常数**（279.07 K / 19.33 K），
OpenSTL 的 `metric()` 会用它把预测和真值反归一化回亮温(K)，所以日志里的数字是 K 空间。
但注意 `openstl/core/metrics.py` 的 `MSE/MAE` 在 `axis=(0,1)` 求均值后做的是
**`.sum()` 而不是 `.mean()`** —— 即对 `(C,H,W)` 求和。所以框架打印的
`mse:4147.9, mae:11895.1` 是**像素和**，不是逐像素均值。
逐像素均值的口径见 `eval_single_sample.py` 的输出（那个是干净的）。

## 四、环境（conda env: `cira-diff-cb`）

原环境缺以下依赖，已补装（`pip install`）：

```
lightning==2.2.1  timm  einops  fvcore  setuptools<81
opencv-python-headless  scikit-image  h5py  pandas  PyWavelets
```

说明：

- `setuptools<81`：lightning 2.2.1 依赖 `pkg_resources`，新版 setuptools 已移除。
- `opencv-python-headless` / `scikit-image` / `h5py` / `pandas` / `PyWavelets`：
  `openstl.datasets` / `openstl.core.metrics` 在 import 阶段就会用到它们。
- **没有降级 timm**：OpenSTL 需要的 `timm.optim.nadam.Nadam` 和 `timm.optim.radam.RAdam`
  在 timm >= 1.0 已被移除。因为 `cira-diff-cb` 是多人共用的 conda 环境，
  降级 timm 可能影响别人的流程，所以改成在**代码副本里**打 `try/except` 补丁
  回落到 `torch.optim.NAdam` / `torch.optim.RAdam`。本任务 `opt='adam'`，
  这两个类不会被实例化，只是 import 时不能炸。

## 五、运行

```bash
cd /home/group1/26fall_aiclass/cb/cira-diff/reproduction/simvp/single_sample_test
mkdir -p logs/output logs/err        # SLURM 不会自动创建父目录
sbatch run_single_sample_simvp.slurm

squeue -u $USER
tail -f logs/output/simvp_single_sample_<jobid>.out
```

脚本做的两件事：

1. `python tools/train.py -d satcast -m SimVP -c configs/satcast/SimVP.py ...`
   —— OpenSTL 原生训练入口，2000 epoch（单样本 = 2000 个梯度步）
2. `python eval_single_sample.py --ckpt .../best.ckpt`
   —— 加载 best.ckpt 推理，与 persistence 基线对比并出图

关键超参（`configs/satcast/SimVP.py` + `dataset_parameters['satcast']`）：

```python
in_shape = [2, 1, 256, 256]   pre_seq_length = 2   aft_seq_length = 1
model_type = 'gSTA'   hid_S = 64   hid_T = 256   N_T = 4   N_S = 2
lr = 1e-3   sched = 'onecycle'   batch_size = 1   epoch = 2000
single_sample_idx = 0         # 训练集里的第 0 条
```

## 六、结果

见 `outputs/satcast_simvp_single_sample/`：

- `checkpoints/best.ckpt`、`last.ckpt`
- `single_sample_metrics.json` —— 指标（归一化空间 + 亮温 K）
- `single_sample_result.png` —— 输入 / 真值 / 预测 / 基线 / 误差 对比图

运行记录与数字见本文件末尾的「运行日志」小节。

---

## 八、运行日志

### 2026-09-27 — 单样本链路跑通（SLURM job 55）

| 字段 | 值 |
| --- | --- |
| 命令 | `sbatch run_single_sample_simvp.slurm` |
| SLURM job | `55`（partition=`debug`, gres=`gpu:1`, node=`test-ai`, RTX 4090） |
| conda env | `cira-diff-cb`（Python 3.11.16 / torch 2.6.0+cu124 / lightning 2.2.1） |
| 样本 | `edm_GOES_ch13_train_dataset.zarr` 第 0 条 |
| 模型 | `SimVP_Model` gSTA, 4.705 M 参数, 单步前向 84.16 GFLOPs |
| 训练 | 2000 epoch / 2000 梯度步（单样本 = 每 epoch 1 步），loss 从 ~0.0281 -> 1.7e-4 |
| exit code | `0`（训练 + 评估两段都成功） |
| 日志 | `logs/output/simvp_single_sample_55.out`, `logs/err/simvp_single_sample_55.err` |

**指标**

| 指标 | SimVP 预测 | persistence 基线 |
| --- | ---: | ---: |
| MSE（归一化空间） | 0.000169 | 0.153797 |
| MAE（归一化空间） | 0.009390 | 0.250590 |
| RMSE（亮温 K） | **0.2516** | 7.5805 |

MSE 相对 persistence 基线降低 **99.89%**。

预测范围 221.2 ~ 298.2 K，真值范围 221.2 ~ 298.0 K（没有出现越界的异常输出）。

**结论**：链路跑通。模型把这一条样本记了下来（RMSE 0.25 K，肉眼几乎与真值重合，
`|SimVP - GT|` 图基本是黑的），而 persistence 基线在云移动区域有明显位移误差。
**这只是单样本过拟合，不等于泛化能力。**

### 备注：job 54

正式跑通前先提了一个 job 54（当时 slurm 脚本里还没有评估那一段）。
训练部分行为与 job 55 完全一致（相同 seed -> 相同曲线），
评估结果也与 job 55 逐位一致。日志保留在 `logs/` 下作为过程记录。
job 54 的框架 `trainer.test()` 输出为 `mse:4147.9, mae:11895.1` —— 注意这是
**像素和**不是均值（见第三节说明），逐像素 RMSE 约 0.25 K，与上表吻合。


## 七、注意事项 / 已知限制

- **单样本 != 泛化能力**。这里只验证链路。
- `metrics` 只用了 `['mse', 'mae']`：`lpips` 未安装、`skimage` 虽已安装但
  为减少依赖面没启用 `ssim`/`psnr`。要做正式评测再打开。
- dataloader 是**惰性**的（每次 `__getitem__` 从 zarr 读一条）。
  训练集 35595 条全量缓存约 13 GiB，对单样本测试是浪费，
  而且在 slurm `--mem` 限制下容易 OOM。要跑全量训练需另行考虑缓存策略。
- `validation/test` zarr（2 -> 18 帧）**没有接入**，本测试刻意避开。
