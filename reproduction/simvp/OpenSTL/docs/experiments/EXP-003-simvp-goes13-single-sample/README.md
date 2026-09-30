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
    --output-dir ./single_sample_goes13_output
```

可调：`--ref-samples 32`（归一化统计量参考池）、`--vmin/-vmax`（色标，默认 -4/2）、
`--config configs/goes13/simvp/SimVP_gSTA.py`、`--cpu`、`--display-method-info`。

## 环境（Environment）
- 已验证：numpy 2.4.6 / torch 2.6.0+cu124 / zarr 3.1.6 / matplotlib 3.11.2
- **缺失**：`timm` / `lightning` / `fvcore` → 本实验尚未执行

## Artifacts（预期产物，`--output-dir` 下）
| 文件 | 内容 |
| --- | --- |
| `sample_<i>_comparison.png` | condition t-2 / condition t-1 / truth / predicted / error 五联图 |
| `sample_<i>_error.png` | 误差图（coolwarm，自适应量程） |
| `sample_<i>_loss_curve.png` | 逐 epoch 训练 loss 曲线 |
| `sample_<i>_condition.npy` / `_truth.npy` / `_pred.npy` / `_error.npy` | 已**反归一化**的原始数组 |
| `sample_<i>_summary.json` | 配置 + loss 首尾值 + 指标（便于登记证据） |

## 已完成的静态校验
- ✅ `python -m py_compile` 通过；关键符号（load_config / create_loader / GOES13Dataset 等）确认引用正确
- ⚠️ 未执行：依赖 `timm` / `lightning`

## Deviation notes
- **D1** `SingleSampleExperiment` 重写 `_init_trainer` 以支持无 GPU 机器（上游硬编码 `accelerator='gpu'`），
  并关闭进度条便于日志归档。
- **D4** 归一化统计量取自 `--ref-samples`（默认 32）参考池，**不**取单样本自身
  ——单个样本的 std 会退化，导致可视化与指标失真。
- 配色沿用 `test_dl/test_single_sample_Chase2025.py` 的 `Spectral_r` / `vmin=-4, vmax=2`，
  保证两种视图可直接对比。

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
