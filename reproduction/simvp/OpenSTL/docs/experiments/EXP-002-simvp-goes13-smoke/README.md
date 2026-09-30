# EXP-002 — SimVP 小样本训练流程验证（GOES-16 ABI ch13）

## Identity
- 实验 ID：`EXP-002`
- 状态：`SCRIPTED & STATICALLY CHECKED; RUN NOT EXECUTED`
- 日期：2026-09-30
- 关联 DEC：DEC-002（多步协议公平性）、DEC-003（自定义 dataloaders 而非改上游）
- Summary entry：`docs/research/experiments.md#exp-002`

## 复现命令（Command）

交互式（本机 / 登录节点）：

```shell
cd /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL
pip install timm lightning fvcore opencv-python      # 环境依赖
python tools/train_simvp_goes13_smoke.py \
    --zarr /home/group1/26fall_aiclass/yr/data1/data1/satcast/edm_GOES_ch13_train_dataset.zarr \
    --config configs/goes13/simvp/SimVP_gSTA.py \
    --res_dir /home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/work_dirs \
    --ex_name simvp_goes13_smoke \
    --n-train 128 --n-val 32 --n-test 32 \
    --epochs 3 --batch-size 8 --lr 1e-3
```

集群（slurm）：

```shell
sbatch test_dl/test_dl.slurm      # 绝对路径已内置，任意目录提交均可
```

> 拼写注意：OpenSTL 注册的是 **`--ex_name`**（别名 `-ex`）与 **`--res_dir`**，均为**下划线**。
> 写成连字符 `--ex-name` 会被 argparse 判为无法识别的参数直接退出（见 `DBG-003`）。

可选：`--no-preload`（省内存，改为逐条读 zarr）、`--cpu`（强制 CPU）、
`--display-method-info`（打印 FLOPs，需 fvcore）、`--num-workers 2`（须 ≤ cpus-per-task-1）。

未显式给出的 `--epochs / --batch-size / --model-type / --lr / --num-workers` 现由
`make_cli_parser()` 中的 `parser.set_defaults(...)` 保证生效（3 / 8 / gSTA / 1e-3 / 2）；
修复前这些默认值会被上游同名 dest 静默覆盖，详见 `docs/dev/debugging.md` 的 `DBG-003`。

## 环境（Environment / Revision）
- 仓库：`reproduction/simvp/OpenSTL`（上游 `github.com/chengtan9907/OpenSTL`；
  上游 commit 尚未 pin，见 `docs/reproduction/provenance_audit.md`）
- 运行环境：conda env `openstl`（克隆 `test` + `timm==0.9.12`，另补 cv2/imageio/pywt/skimage）；
  实测 numpy 2.4.6 / torch 2.6.0+cu124 / timm 0.9.12 / lightning 2.6.6 / fvcore 0.1.5 /
  zarr 2.18.7 / opencv 5.0.0 / imageio 2.38 / PyWavelets / scikit-image。配方见
  `docs/dev/environment.md`。

## Artifacts（预期产物）
运行后落在（绝对路径）
`/home/group1/26fall_aiclass/yr/cira-diff/reproduction/simvp/OpenSTL/work_dirs/simvp_goes13_smoke/`：

- `checkpoints/best.ckpt`、`checkpoints/last.ckpt`
  （另有模板名 `checkpoints/best-epoch=XX-val_loss=Y.ckpt`）
- `saved/inputs.npy`、`saved/preds.npy`、`saved/trues.npy`、`saved/metrics.npy`
  —— 只有本实验会产生，因为只有它调用了 `exp.test()`
- `model_param.json`（由 SetupCallback 写入，含全部超参）与 `train_<时间戳>.log`

> 日志会分流：`print_log` 同时走 stdout 与 logging，`logging` 在 fit 开始时被
> `SetupCallback` 重定向到 `train_<时间戳>.log`；slurm 的 `.out` 里只有 `print` 的部分。

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
- **状态：`EXECUTED`**（2026-09-30，slurm job 102，节点 `test-ai`，单卡 RTX 4090）。
- 运行命令：默认 slurm 参数（EPOCHS=3 / N_TRAIN=128 / N_VAL=32 / N_TEST=32 / BS=8 / lr=1e-3）。
- 训练曲线：val_loss 0.365 → 0.295（3 epoch，`onecycle` 调度）；测试 mse 7322.57 /
  mae 14507.10 / rmse 85.57（反归一化后，量与原始 brightness 量纲一致）。
- **⚠️ 这些指标不是复现结论**：① 仅 3 epoch、192 样本；② 归一化统计量取自该 train 子集
  （mean=0.093 / std=0.866），与全量 split 不同；③ 模型几乎未收敛。仅供流水线端到端验证。
- 产物已落盘：`work_dirs/simvp_goes13_smoke/{checkpoints,saved,model_param.json,train_*.log}`，
  与「Artifacts」小节预测完全一致。
- Conclusion：流水线 + GPU 路径均已验证；下一步扩大样本并按参考超参（hid_S=64 / hid_T=512 /
  N_T=8）训练，再与 CIRA-Diff 公平对比（DEC-002）。

## Next
1. 安装依赖后执行上方命令，确认 train / val / test 各跑通一轮 → 更新本文件为 `EXECUTED`。
2. 扩大样本 + 恢复参考超参（`hid_S=64, hid_T=512, N_T=8`）产出可比对指标。
3. 与 CIRA-Diff 在同一多步协议下对比（DEC-002）。
