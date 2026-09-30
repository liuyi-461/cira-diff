# EXP-003 — SimVP 单样本过拟合 + 预报可视化（GOES-16 ABI ch13）

## Identity
- 实验 ID：`EXP-003`
- 状态：`SCRIPTED & STATICALLY CHECKED; RUN NOT EXECUTED`
- 日期：2026-09-30
- 关联：`EXP-002`（多样本流水线验证）、`DEC-003`（不改上游的接入方式）
- Summary entry：`docs/research/experiments.md#exp-003`

## 目的
EXP-002 验证的是**流水线**能否跑通；本实验验证的是**模型与结构本身**是否正确。
做法：取向单个样本训练至过拟合。结构接线正确的 SimVP 必然能把单样本 loss 压到极低
并复现目标帧；通道错位、时间轴错置、skip 连接断裂、loss 算错张量等问题会立刻暴露在
并列图上，而这些在聚合指标里看不出来。

## 复现命令（Command）

```shell
cd /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
python tools/train_simvp_goes13_single_sample.py \
    --zarr /home/group1/26fall_aiclass/yr/data1/data1/satcast/edm_GOES_ch13_train_dataset.zarr \
    --index 0 --epochs 100 --lr 1e-3 \
    --ex_name simvp_goes13_single \
    --res_dir /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/work_dirs \
    --output-dir /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/outputs/single_sample_goes13
```

### slurm 提交（推荐，已配好 GPU 独占 + openstl 环境）
```shell
cd /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
sbatch test_dl/test_dl_single_sample.slurm
# 覆盖示例：INDEX=5 EPOCHS=50 REF_SAMPLES=64 sbatch test_dl/test_dl_single_sample.slurm
```
脚本内置绝对路径、`#SBATCH --gres=gpu:1`（独占一张卡，规避 GPU 争抢）、openstl 环境、
PYTHONPATH 与依赖预检，与 `test_dl/test_dl.slurm`（EXP-002）同构。产物落在
`work_dirs/single_sample_goes13_output/`，checkpoints 落在 `work_dirs/simvp_goes13_single_sample/`。

可调：`--ref-samples 32`（归一化统计量参考池）、`--vmin`/`--vmax`（色标，默认 -4/2）、
`--config configs/goes13/simvp/SimVP_gSTA.py`、`--cpu`、`--display-method-info`。

> 拼写注意：`--ex_name` / `--res_dir` 是**下划线**（OpenSTL 原生选项），
> 连字符写法会被 argparse 拒绝，见 `DBG-003`。
> 路径建议一律写**绝对路径**：默认值都是相对路径，落点会随作业工作目录漂移，见 `DBG-004`。
> 默认 `--lr 1e-3` 与 `--model-type gSTA` 现由 `parser.set_defaults(...)` 生效。

## 环境（Environment）
- 已验证：numpy 2.4.6 / torch 2.6.0+cu124 / zarr 3.1.6 / matplotlib 3.11.2（CPU 侧静态验证）
- 运行环境：`openstl` conda 环境（克隆 `test` + `timm==0.9.12`，另补 cv2/imageio/pywt/skimage）
  已提供 `timm` / `lightning` / `fvcore`，依赖约束已解除；但本实验**尚未在 GPU 上正式执行**
  （仅 CPU 静态验证过行为）。直接用 `test` 环境会在 import openstl 阶段因 timm>=1.0 报错，
  见 `docs/dev/debugging.md` 的 `DBG-005`。

## Artifacts（预期产物，`--output-dir` 下）
| 文件 | 内容 |
| --- | --- |
| `sample_<i>_comparison.png` | condition t-2 / condition t-1 / truth / predicted / error 五联图 |
| `sample_<i>_error.png` | 误差图（coolwarm，自适应量程） |
| `sample_<i>_loss_curve.png` | 逐 epoch 训练 loss 曲线 |
| `sample_<i>_condition.npy` / `_truth.npy` / `_pred.npy` / `_error.npy` | 已**反归一化**的原始数组 |
| `sample_<i>_summary.json` | 配置 + loss 首尾值 + 指标（便于登记证据） |

**此外框架自身还会写**（在 `work_dirs/<ex_name>/` 下，与 `--output-dir` 无关）：
- `checkpoints/best.ckpt`、`checkpoints/last.ckpt`
- `model_param.json`（全部超参）、`train_<时间戳>.log`

注意：本脚本**不调用 `exp.test()`**（预测是手动 forward 做的），因此**不会**
生成 `saved/*.npy`——那是 EXP-002 独有的产物。若与 EXP-002 用同一个 `--ex_name`，
两者会互相覆盖 `best.ckpt` / `last.ckpt` / `model_param.json`，必须用不同的名字。

## 已完成的静态校验
- ✅ `python -m py_compile` 通过；关键符号（load_config / create_loader / GOES13Dataset 等）确认引用正确
- ⚠️ 未执行：依赖 `timm` / `lightning`

## Deviation notes
- **D1** `SingleSampleExperiment` 重写 `_init_trainer` 以支持无 GPU 机器（上游硬编码 `accelerator='gpu'`），
  并关闭进度条便于日志归档。
- **D4** 归一化统计量取自 `--ref-samples`（默认 32）参考池，**不**取单样本自身
  ——单个样本的 std 会退化，导致可视化与指标失真。
- 配色（`Spectral_r` / `vmin=-4, vmax=2`）固定在 EXP-003 脚本里，保证两种视图可直接对比；
  该约定源自已删除的 `test_dl/test_single_sample_Chase2025.py`，来源见
  `docs/experiments/cira_diff/EXP-001-cira-diff-data-audit.md`。

## 判读准则（Results 判据）
执行后按下表判读，结论填入 Results：

| 现象 | 判读 |
| --- | --- |
| loss 数量级下降 + predicted 与 truth 高度相似 | 模型/结构接线**正确** ✅ |
| loss 不降或震荡不收敛 | 检查 lr / 数据归一化 / 是否 loss 算错张量 |
| 输出无结构（近似常数或模糊） | 极可能是通道/时间轴错位或 skip 连接问题 |
| 报错（shape mismatch） | `in_shape` 与 `pre/aft_seq_length` 不一致 |

## Results / Conclusion
- Results：`NOT EXECUTED`，尚无 loss 曲线与指标数值。
- Conclusion：脚本与判据已就绪，阻塞项仍为环境依赖。

## Next
1. 安装 `timm` / `lightning` / `fvcore` 后执行上方命令 → 更新本文件为 `EXECUTED` 并贴入 loss 首尾值与 MAE/RMSE。
2. 用 `--index` 换 2–3 个不同样本复核，排除单样本偶然性。
3. 通过后再跑 EXP-002 多样本流水线，最后做 full-split 训练。
