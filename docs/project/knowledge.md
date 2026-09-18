# 长期知识沉淀

## 已验证规律

- [FACT] `ZarrDataset` 返回顺序是 `(target, condition)`，不是常见的 `(condition, target)`；训练脚本按 `batch[0]` 取 target、`batch[1]` 取 condition。
- [FACT] EDM wrapper 只对生成通道应用 `c_in/c_skip/c_out`，条件通道直接拼接进入 U-Net。
- [FACT] 当前仓库的 CorrDiff 配置为 1 个 noisy generation channel + 3 个 condition channels（两帧历史 + 一张普通 U-Net 预测），因此 `in_channels=4`。
- [FACT] `edm_sampler` 默认 18 个采样步；采样步数不是 forecast lead time，不能把两者混为一谈。
- [FACT] 远端 ordinary train store 的 shape 是 input `(35595,2,256,256)`、output `(35595,1,256,256)`；validation/test 的 output 是 `(1024,18,256,256)`。
- [FACT] validation/test 的 18 帧 truth 由 Zarr `output_images` 保存，forecast notebook 以 `np.arange(0,18)` 逐步生成预测并与它对齐。
- [CONCLUSION] “三帧”是 ordinary train 单步样本的窗口定义，不是公开数据或所有 split 的最大连续长度。
- [FACT] `/data1/satcast/` 只包含派生 Zarr patch、说明文件、notebook 和 tar archives；当前 Zarr metadata 没有 per-sample timestamps。

## 复现经验与风险

- [ISSUE] `cira_diff/train_Diff.py` 与 `train_CorrDiff.py` 的 main 中训练准备代码被注释；优先审计并运行 `scripts/Chase_2025/`，不要根据文件名假设新封装已可用。
- [ISSUE] 配置和脚本存在绝对路径、GPU 编号、mean/std 等硬编码；正式实验前必须把它们写入可追溯 config。
- [ISSUE] 当前 `config_CorrDiff.py` 字段名为 `llr_warmup_steps`，训练脚本使用的是 `lr_warmup_steps`；这是潜在运行错误，属于后续工程修复范围。
- [BLOCKED] 本地已同步 subset，但本机缺少 `torch`/`zarr`，尚未实际调用 `ZarrDataset` 和 `DataLoader`。

## 证据纪律

- 论文报告的 CorrDiff 优势、ensemble calibration 和 3 h 结果是 `[EVIDENCE]`，不是本项目 `[RESULT]`。
- “两帧足够”不是当前项目结论；最多是需要通过 history-length ablation 验证的 `[HYPOTHESIS]`。
