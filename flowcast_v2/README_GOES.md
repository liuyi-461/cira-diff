# FlowCast GOES / SatCast 使用说明

本目录使用 GOES-16 Ch13 Zarr（`input_images: N,T_in,H,W`、`output_images: N,T_out,H,W`），默认任务为 2 帧预测 1 帧。模型主体保持原有 CuboidTransformerUNet；GOES 训练入口位于 `experiments/goes/`。

## 迁移须知

**必须重新训练 VAE、重新编码 latent，再训练 CFM。** 旧模型使用 `[0,1]` 输入，而旧编码甚至没有使用相同的归一化；旧 latent 不能直接用于这套流程。旧结果 `cfm_val_loss=0.5392` 存在数据泄漏，不能当作泛化成绩。

- 训练/验证必须提供独立原始 Zarr；测试集另行提供，禁止拿训练集尾部当测试集。程序拒绝相同/同源路径，但不同目录中内容重复、时间窗口重叠仍需数据提供方确认。
- 原数据标准化参数采用报告提供的 `mean=279.0699, std=19.3297`；请根据实际数据生成说明核对。通过 `Tb=x*std+mean` 恢复 Kelvin，仅在训练数据上统计 Kelvin 最小/最大值，线性映射到 `[-1,1]`。不做 clip，因此验证/测试超出训练范围的值会保留。
- VAE 使用全部输入和目标帧，一个序列展开为多个独立帧。`max_samples` 指原始序列数，默认全量。
- latent 使用 `latent_dist.mode()`，按批写 Zarr，记录来源、归一化参数、VAE SHA256 和完成标志。中途失败的数据不会进入 CFM。
- 训练 snapshot 放在 checkpoint 所属 run 目录；推理使用该 snapshot，拒绝缺失/不匹配的 VAE 配置。

## 环境

建议在 Linux GPU 服务器的新 Python 3.11 环境运行，安装匹配服务器驱动的 PyTorch CUDA wheel，然后：

```bash
pip install -r requirements.txt
pip install -e . --no-deps
pip install -r requirements-dev.txt
```

依赖版本已在 requirements 中固定；完整 GPU 依赖组合仍需要服务器实测。不要修改其他项目已有环境。VAE 的 LPIPS 可能首次下载预训练权重。

## 一键训练 / 参数扫描

在 `flowcast/` 下运行，修改 `sweep_config.yaml` 的网格参数：

```bash
python -m tools.run_sweep \
  --train_file /path/to/train.zarr \
  --val_file /path/to/validation.zarr \
  --test_file /path/to/test.zarr \
  --output_dir runs
```

`--test_file` 可省略；未提供时不生成测试 latent，也不声称有测试成绩。这里的路径是占位符，需要替换为服务器真实数据路径。

`common.max_samples/val_max_samples/test_max_samples: null` 表示各自全量。小规模自检可设为正整数。`num_gpus/gpu_ids: null` 优先沿用 Slurm 或外部 `CUDA_VISIBLE_DEVICES`；无分配时默认 1 GPU。可通过 `CUDA_VISIBLE_DEVICES=0,1` 指定可见卡。torchrun 使用独立 rendezvous 自动选择端口。

每次运行创建独立目录，不修改源码或配置模板：

```text
runs/sweep_<唯一ID>/
  results.csv                       # 各阶段返回码，失败原因，最佳 CFM 验证 loss
  run_000/
    config_snapshot.yaml            # sweep 参数、源路径、归一化
    vae.yaml / cfm.yaml              # 本轮生成配置
    vae.log / encode_*_rc.log / cfm.log
    vae/{config_snapshot.yaml,losses.csv,models/early_stopping_model.pt}
    train.zarr / val.zarr / test.zarr
    cfm/{config_snapshot.yaml,losses.csv,models/early_stopping_model.pt}
```

任何训练/编码阶段失败都会阻止本轮后续阶段，记录错误，再继续其他组合。整个 sweep 有失败时返回非零。输出目录必须是新目录；不隐式覆盖旧模型或 latent。

## 单独运行

先生成训练归一化配置：

```bash
python -m scripts.fit_normalization --train_file /path/to/train.zarr \
  --out runs/manual/vae.yaml
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module experiments.goes.autoencoder.dist_train_autoencoder_kl \
  --config runs/manual/vae.yaml --train_file /path/to/train.zarr \
  --val_file /path/to/validation.zarr --output_dir runs/manual/vae
python -m scripts.encode_latent --zarr_path /path/to/train.zarr \
  --ckpt_path runs/manual/vae/models/early_stopping_model.pt \
  --config runs/manual/vae/config_snapshot.yaml --out_path runs/manual/train.zarr
```

对验证/测试集分别重复编码，始终使用同一个 VAE 和归一化配置。

单独训练 CFM 时，复制 `configs/goes/flowcast.yaml` 到运行目录，将 `normalization` 从 VAE snapshot 复制进去，并把 `autoencoder_params.autoencoder_checkpoint` 指向该 VAE checkpoint。然后调用 `experiments.goes.runner.flowcast.dist_train_flowcast`，传入不同的训练/验证 latent、`--config` 和新的 `--output_dir`。

两个训练入口支持 `--max_samples`、`--val_max_samples`。Slurm 脚本要求环境变量 `TRAIN_FILE`、`VAL_FILE`、`CONFIG`、`OUTPUT_DIR`，提交前激活训练环境，并从 flowcast 目录执行 `sbatch slurm/vae_train.slurm` 或 `sbatch slurm/cfm_train.slurm`。也可设置 `WORKDIR`。

旧的 SEVIR partial evaluation 仍使用雷达指标和数据结构，GOES 训练入口明确拒绝启用它；使用下面的 GOES 离线评测。

## 评测、推理和 VAE 质量

用本轮真实测试源和对应 test latent：

```bash
python -m scripts.evaluate \
  --data /path/to/test.zarr --split test \
  --latent_path runs/sweep_ID/run_000/test.zarr \
  --vae_checkpoint runs/sweep_ID/run_000/vae/models/early_stopping_model.pt \
  --vae_config runs/sweep_ID/run_000/vae/config_snapshot.yaml \
  --cfm_checkpoint runs/sweep_ID/run_000/cfm/models/early_stopping_model.pt \
  --cfm_config runs/sweep_ID/run_000/cfm/config_snapshot.yaml \
  --samples 8 --steps 20 --batch_size 1 --threshold 235 --fss_window 9 \
  --figure --output_dir runs/sweep_ID/run_000/evaluation
```

- `--split test` 拒绝训练和验证源；`--split val` 仅拒绝训练源，用于调参验证。不同拷贝之间的时间泄漏不能靠文件路径检查发现。
- 直接与原始目标亮温比较，不把 VAE 重建当作真值。输出 `metrics.json`，包含参数、样本数、MSE(K²)、MAE(K)、RMSE(K)、PSNR、SSIM、CRPS(K)、CSI、FSS。
- 同时输出 VAE 重建和 persistence（最后观测帧）基线。当前任务没有额外低分辨率输入，单独做 bicubic 不构成有意义的预测基线，因此没有虚构该基线。
- CSI/FSS 的事件定义为 **Tb <= threshold**，默认 235 K 是可调整示例，不能直接等同降水阈值。CSI/FSS 对集合均值计算；SSIM 使用 sigma=1.5 的高斯窗口；FSS 使用指定奇数窗口和反射边界。没有事件时 CSI/FSS 为 null；MSE 为零时 PSNR 为 null（对应无穷大）。
- `--figure` 保存第一条样本的真实目标、VAE 重建、预测集合均值、10%/90% 分位图，共用 Kelvin 色标。
- 将入口换成 `scripts.infer_one_sample`，同样提供上述模型/数据参数，使用 `--index` 选样本，自动评测一条并画图。
- 只评测 VAE：使用 `scripts.check_vae`，仅传 `--data --vae_checkpoint --vae_config --split --output_dir`，可加 `--max_samples`。

## 检查与代码结构

```bash
python -m pytest tests -q
python -m scripts.check_dataset --kind raw --data /path/to/train.zarr --config runs/manual/vae.yaml
python -m scripts.check_dataset --kind latent --data runs/manual/train.zarr
```

`datasets/goes/`：数据适配、可逆归一化、指标和模型加载。
`configs/goes/`：配置模板。
`experiments/goes/`：训练入口。
`scripts/`：编码、检查、预测、评测。
`tools/`：参数扫描。
`tests/`：合成数据回归测试。
可用 `pip install -e . --no-deps` 注册包，使 `python -m` 入口不依赖当前工作目录（配置和数据仍应传绝对路径）。
根目录旧脚本与原训练路径保留兼容入口，避免旧启动命令突然失效；旧参数仍需要按本文补齐。

当前工作目录原本没有 Git 仓库，未擅自创建仓库或提交。修改前源码备份保存在上级 `flowcast_backup_20260929.zip`；产物目录已写入 `.gitignore`。本地检查使用合成数据，不能替代服务器上的真实多卡训练和独立测试集验证。

## 小规模端到端测试

在 flowcast 文件夹运行（需真实数据和 CUDA 环境）：

```bash
python run_small_test.py --train_file /path/to/train.zarr --val_file /path/to/validation.zarr
```

默认单卡、16 条训练序列、4 条验证序列、batch size 1、VAE 2 轮、CFM 1 轮，随后对最多 2 条验证样本执行 2 次采样、4 步 Euler 预测并画图。保持正式模型结构，用于检查完整流程，不用于判断模型质量。首次运行 LPIPS 仍可能需要下载权重。

通过 `--train_samples`、`--val_samples` 调整样本数，`--gpu 0` 可显式选单卡。VAE 至少 2 轮，以便原训练流程在激活判别器后保存 checkpoint。加 `--dry_run` 只生成配置并展示命令，不读取数据或训练。

结果位于 `runs/small_tests/<唯一运行目录>/`，包括 `pipeline.log`、`evaluation.log`、`summary.json`，以及下级 sweep 目录里的分阶段日志、checkpoint、`metrics.json` 和 `comparison.png`。阶段失败返回非零并停止后续步骤，不覆盖已有运行。
