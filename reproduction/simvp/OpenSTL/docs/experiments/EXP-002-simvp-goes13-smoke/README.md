# EXP-002 — SimVP 小样本训练流程验证（GOES-16 ABI ch13）

## Identity
- 实验 ID：`EXP-002`
- 状态：`SCRIPTED & STATICALLY CHECKED; RUN NOT EXECUTED`
- 日期：2026-09-30
- 关联 DEC：DEC-002（多步协议公平性）、DEC-003（自定义 dataloaders 而非改上游）
- Summary entry：`docs/research/experiments.md#exp-002`

## 复现命令（Command）

```shell
cd /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
pip install timm lightning fvcore opencv-python      # 环境依赖
python tools/train_simvp_goes13_smoke.py \
    --zarr /home/group1/26fall_aiclass/yr/data1/data1/satcast/edm_GOES_ch13_train_dataset.zarr \
    --n-train 128 --n-val 32 --n-test 32 \
    --epochs 3 --batch-size 8 --lr 1e-3 \
    --ex-name simvp_goes13_smoke
```

可选：`--config configs/goes13/simvp/SimVP_gSTA.py`（默认即该文件）、
`--no-preload`（省内存，改为逐条读 zarr）、`--cpu`（强制 CPU）、
`--display-method-info`（打印 FLOPs，需 fvcore）。

## 环境（Environment / Revision）
- 仓库：`reproduction/simvp/OpenSTL`（上游 `github.com/chengtan9907/OpenSTL`；
  上游 commit 尚未 pin，见 `docs/reproduction/provenance_audit.md`）
- 已验证：numpy 2.4.6 / torch 2.6.0+cu124 / zarr 3.1.6 / xarray 2026.7.0
- **缺失**：`timm`、`lightning`、`fvcore` → 因此本实验未执行训练循环

## Artifacts（预期产物）
运行后落在 `work_dirs/simvp_goes13_smoke/`：
- `checkpoints/best.ckpt`、`checkpoints/last.ckpt`
- `saved/inputs.npy`、`saved/preds.npy`、`saved/trues.npy`、`saved/metrics.npy`
- `args.txt`（由 SetupCallback 写入，含全部超参）

## 已完成的静态校验
- ✅ 三个新增文件通过 `python -m py_compile`
- ✅ 数据契约在真实 zarr 上抽样验证（见下）

## 数据契约验证结果（真实数据抽样）
| 项 | 观测值 |
| --- | --- |
| `input_images` 形状 | (2, 1, 256, 256) 经 adapter 加通道维后 |
| `output_images` 形状 | (1, 1, 256, 256) |
| dtype | float16 → adapter 转 float32 |
| 抽样 mean / std | 0.518297 / 0.440660 |
| 值域 min / max | -3.742 / 1.924（与既有脚本 `vmin=-4, vmax=2` 一致） |
| 有限性 | 全部 finite，无 NaN |

## Deviation notes（相对上游的偏差）
- **D1** `SmokeExperiment` 子类重写 `_init_trainer`，使无 GPU 机器可跑（上游硬编码 `accelerator='gpu'`）。
- **D2** zarr 无 `_ARRAY_DIMENSIONS` 元数据，改用原生 zarr API 读取（`xr.open_zarr` 会 KeyError）。
- **D3** 归一化统计量取自抽样出的 train 子集而非全量 train split —— 仅够验证流水线，**不可用于指标对比**；全量复现时须重算并随权重保存。

## Results / Conclusion
- Results：`NOT EXECUTED`，无 MAE / MSE / RMSE 数值。
- Conclusion：流水线代码实现与数据契约已就绪；阻塞项为环境依赖（timm / lightning / fvcore）。

## Next
1. 安装依赖后执行上方命令，确认 train / val / test 各跑通一轮 → 更新本文件为 `EXECUTED`。
2. 扩大样本 + 恢复参考超参（`hid_S=64, hid_T=512, N_T=8`）产出可比对指标。
3. 与 CIRA-Diff 在同一多步协议下对比（DEC-002）。
