# 当前版本
`vanilla-unet-full-training-running-2026-09-23`（Vanilla UNet full training in progress；环境与 smoke 已通过）
## 已完成
- 阅读并核对 `docs/README.md`、仓库 README、源码和 `scripts/Chase_2025/` 入口。
- 建立 CIRA-Diff baseline 的项目、研究、训练和开发文档初稿。
- [EVIDENCE] 核对官方论文：GOES-16 ABI Channel 13、10 min、256×256 patch、两帧条件、单步预测后 18 步 rollout 到 3 h。
- 检查 `.gitignore`，加入本地数据 payload 排除规则。
- [EVIDENCE] 远端 `/data1/satcast/` 已验证可访问，host 为 `test-ai`。
- [EVIDENCE] train/validation/test Zarr 的 shape、dtype、chunks、规模和 sample 组织已完成审计。
- 已通过 rsync 同步 ordinary train、CorrDiff train、validation、test 的最小本地 subset，共约 145 MB。
- [FACT] conda env `cira-diff-ly` 已配置完整：Python 3.11.16、torch 2.6.0+cu124、diffusers 0.37.2、accelerate 1.10.1、zarr 2.24.4。
- [FACT] **EXP-009（Gate Check）已通过**：四种 baseline 代码路径（Vanilla UNet / EDM / LDM / CorrDiff）在单样本（index=0, 1000 epoch）上均成功收敛；Vanilla UNet MSE 最低（3.95e-05），CorrDiff 残差分布合理（std=0.0056）。详见 `docs/experiments/gate_checks/EXP-009-single-sample-overfit.md`。
- [FACT] `train_vanilla_unet_Chase2025.py` 已从 main 分支恢复并做 6 处本地 patch（数据路径、batch_size、grad_accum、matplotlib 兼容性、80/20 random_split + 独立 val zarr、num_workers）。
- [FACT] Vanilla UNet full training 已提交 Slurm Job 46 并在 Epoch 11 稳定运行。
## 当前实现
- 上游仓库代码已存在，当前分支为 `feature/ly`，HEAD `cf67d6e`。
- `cira_diff/dataset.py` 提供 ZarrDataset。
- `cira_diff/edm.py` 提供 EDMPrecond、EDMLoss、edm_sampler。
- 完整的旧训练脚本在 `scripts/Chase_2025/`。
- `cira_diff/train_Diff.py` 和 `train_CorrDiff.py` 的 main 中关键加载/建模/训练代码目前被注释，不能直接视为可运行入口。
[CONFLICT] 上面的历史记录写的是 `e59d4ba`；当前分支审计记录 `main` 为 `3ffa797`（HEAD after rebase）。保留旧记录作为 provenance，以 `repository_audit.md` 为 source of truth。
## 当前问题
- [ISSUE] 没有独立、经过测试的 rollout/evaluation CLI。当前训练脚本只输出单步 MSE，无论文要求的 18-step autoregressive rollout 评估。
- [ISSUE] 训练脚本存在硬编码路径（非 config-driven），不利于参数管理。
- [UNKNOWN] 官方 diffusion 训练是否做过额外 rollout fine-tuning；当前代码没有该路径，论文只明确描述了单步条件与 autoregressive inference。
- [UNKNOWN] 训练脚本加载了独立 validation zarr 但实际只用 held-out val split（独立 val 未被使用）。
## 下一步计划
1. **等待 EXP-010 完成**：Vanilla UNet full training（Slurm Job 46，预估 ~60h），之后评估 checkpoint 并与论文 Vanilla UNet 参考对比。
2. **实现 rollout/evaluation CLI**：基于 checkpoint 跑 18-step autoregressive inference，计算 MSE/MAE/SSIM/像素指标/阈值指标。
3. **依次提交 EDM / LDM / CorrDiff full training**：复用同一 batch=24 + grad_accum=2 的显存配置。
4. 正式实现 history-length 和 model comparison 受控实验（EXP-003 → EXP-005）。
## Canonical Lab 状态 — 2026-09-23（本窗口审计）
### Research State（科研状态）
- [FACT] 项目已有 RQ-001–RQ-008（`docs/research/questions.md`）和 HYP-001–HYP-007（`docs/research/hypotheses.md`）。
- [RESULT] EXP-009 有单样本 overfit 结果（四种方法均收敛），但**不是**泛化能力结论。
- [RUNNING] EXP-010 Vanilla UNet full training 正在进行，尚无完整科学结果。
- [UNKNOWN] 尚未记录任何受控 history-length comparison、model comparison、概率校准分析或案例研究。
### Engineering State（工程状态）
- [FACT] conda env `cira-diff-ly` 已就绪，`torch.cuda.is_available()` 验证通过。
- [FACT] `diffusers.UNet2DModel` forward/backward/inference 在 RTX 4090 48GB 上 fp16 + bs=24 + grad_accum=2 稳定运行（显存 ~47GB）。
- [FACT] `setuptools_distribution` 与新版 Python 的 deprecation 警告已 patch（降级到 setuptools<81 解决）。
- [ISSUE] 独立 rollout/evaluation CLI 未实现；当前没有从 checkpoint 到 18-step forecast skill 的完整评估路径。
- [PLAN] 后续 EDM / LDM / CorrDiff 的训练脚本同样需要 6 处本地 patch（路径、bs、matplotlib、num_workers）。
详细审计和 evidence ledger 维护在 `docs/project/repository_audit.md`。